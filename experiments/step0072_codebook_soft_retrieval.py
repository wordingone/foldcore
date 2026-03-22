#!/usr/bin/env python3
"""
Step 72 — Soft retrieval in classify().

Baseline: spawn=0.95, full_grad update, 1-NN classify = 33.5% AA.

Part A: Softmax-weighted voting (Modern Hopfield retrieval)
  Sweep tau = {0.01, 0.05, 0.1, 0.2, 0.5}
  output = softmax(sims / tau) weighted vote per class

Part B: k-NN comparison
  k = {1, 5, 20, 50, 100}
  Two variants: uniform vote, distance-weighted vote
  Separates "using more prototypes" from "soft weighting"

All on Split-CIFAR-100, spawn=0.95, ResNet-18 features, full_grad update, RTX 4090.
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, 'B:/M/avir/research/fluxcore')

CACHE_PATH    = 'C:/Users/Admin/cifar100_resnet18_features.npz'
N_TASKS       = 20
CLASSES_TASK  = 5
D_EMBED       = 512
SPAWN_THRESH  = 0.95
MERGE_THRESH  = 0.95
LR            = 0.015
SOFTMAX_TEMP  = 0.1   # training softmax temp (full_grad mode)


class FullGradCodebook:
    """Full-gradient update codebook (Step 71 best). CUDA-accelerated."""

    def __init__(self, d, spawn_thresh=SPAWN_THRESH, device=None):
        self.d            = d
        self.spawn_thresh = spawn_thresh
        self.device       = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.vectors      = torch.empty((0, d), dtype=torch.float32, device=self.device)
        self.labels       = []
        self.n_spawned    = 0

    def step(self, r, label=None):
        if len(self.vectors) == 0:
            self._spawn(r, label)
            return
        sims    = self.vectors @ r
        winner  = int(sims.argmax())
        max_sim = float(sims[winner])
        if max_sim < self.spawn_thresh:
            self._spawn(r, label)
            return
        # Full gradient update (Step 71 best)
        s = sims / SOFTMAX_TEMP
        s = s - s.max()
        probs = s.exp() / (s.exp().sum() + 1e-15)
        y = torch.tensor([1.0 if self.labels[i] == label else 0.0
                          for i in range(len(self.labels))],
                         dtype=torch.float32, device=self.device)
        errors = probs - y
        active = errors.abs() > 1e-4
        if active.any():
            V_act   = self.vectors[active]
            e_act   = errors[active].unsqueeze(1)
            cos_act = sims[active].unsqueeze(1)
            perp    = r.unsqueeze(0) - V_act * cos_act
            self.vectors[active] = self.vectors[active] - LR * e_act * perp
            norms = self.vectors[active].norm(dim=1, keepdim=True).clamp(min=1e-15)
            self.vectors[active] = self.vectors[active] / norms

    def _spawn(self, r_norm, label):
        if len(self.vectors) > 0:
            abs_sims = (self.vectors @ r_norm).abs()
            best_i   = int(abs_sims.argmax())
            if float(abs_sims[best_i]) > MERGE_THRESH:
                fused = self.vectors[best_i] + r_norm
                self.vectors[best_i] = fused / (fused.norm() + 1e-15)
                return
        self.vectors = torch.cat([self.vectors, r_norm.unsqueeze(0)], dim=0)
        self.labels.append(label)
        self.n_spawned += 1

    # ── Readout variants ──────────────────────────────────────────────────────

    def classify_hard(self, X):
        """1-NN baseline."""
        sims    = F.normalize(X, dim=1) @ self.vectors.T   # (n, N)
        winners = sims.argmax(dim=1).tolist()
        return [self.labels[w] for w in winners]

    def classify_soft(self, X, tau=0.1):
        """Softmax-weighted voting over all prototypes (Modern Hopfield retrieval)."""
        sims    = F.normalize(X, dim=1) @ self.vectors.T   # (n, N)
        weights = F.softmax(sims / tau, dim=1)              # (n, N)
        # Build per-class score matrix: (n, n_classes)
        unique_labels = sorted(set(self.labels))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        n_classes     = len(unique_labels)
        scores        = torch.zeros(len(X), n_classes, device=self.device)
        for j, lbl in enumerate(self.labels):
            scores[:, label_to_idx[lbl]] += weights[:, j]
        pred_idx = scores.argmax(dim=1).tolist()
        return [unique_labels[i] for i in pred_idx]

    def classify_knn(self, X, k=5, weighted=False):
        """k-NN: uniform vote or distance-weighted vote."""
        k       = min(k, len(self.vectors))
        sims    = F.normalize(X, dim=1) @ self.vectors.T    # (n, N)
        top_k   = sims.topk(k, dim=1)                       # values, indices
        unique_labels = sorted(set(self.labels))
        label_to_idx  = {l: i for i, l in enumerate(unique_labels)}
        n_classes     = len(unique_labels)
        scores        = torch.zeros(len(X), n_classes, device=self.device)
        for rank in range(k):
            idx  = top_k.indices[:, rank]    # (n,)
            w    = top_k.values[:, rank] if weighted else torch.ones(len(X), device=self.device)
            for sample_i in range(len(X)):
                lbl = self.labels[idx[sample_i].item()]
                scores[sample_i, label_to_idx[lbl]] += w[sample_i]
        pred_idx = scores.argmax(dim=1).tolist()
        return [unique_labels[i] for i in pred_idx]


def load_data():
    data = np.load(CACHE_PATH)
    X_tr, y_tr = data['X_train'], data['y_train']
    X_te, y_te = data['X_test'],  data['y_test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        splits.append((
            torch.tensor(X_tr[np.isin(y_tr, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_tr[np.isin(y_tr, range(c0, c1))]),
            torch.tensor(X_te[np.isin(y_te, range(c0, c1))], dtype=torch.float32, device=device),
            torch.tensor(y_te[np.isin(y_te, range(c0, c1))]),
        ))
    return splits


def train_codebook(splits):
    """Train one codebook (full_grad, spawn=0.95). Returns trained cb."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cb = FullGradCodebook(D_EMBED, device=device)
    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        X_n = F.normalize(X_t, dim=1)
        for i in range(len(X_n)):
            cb.step(X_n[i], label=int(y_t[i]))
    return cb


def eval_with_readout(cb, splits, classify_fn):
    """Evaluate a trained codebook with a given classify function."""
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    # We evaluate as if trained sequentially: acc_matrix[et][task_id]
    # Since codebook is already fully trained, we evaluate the final codebook
    # against each test set independently (post-training evaluation).
    for et in range(N_TASKS):
        X_te, y_te = splits[et][2], splits[et][3]
        preds = classify_fn(X_te)
        acc_matrix[et][N_TASKS - 1] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)
    final  = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    return sum(final) / N_TASKS


def run_sequential_eval(splits, classify_fn_factory):
    """
    Full sequential evaluation (trains task-by-task, evaluates all past tasks).
    More expensive but gives forgetting metric.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cb = FullGradCodebook(D_EMBED, device=device)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]
    t0 = time.time()
    for task_id in range(N_TASKS):
        X_t, y_t = splits[task_id][0], splits[task_id][1]
        X_n = F.normalize(X_t, dim=1)
        for i in range(len(X_n)):
            cb.step(X_n[i], label=int(y_t[i]))
        classify_fn = classify_fn_factory(cb)
        for et in range(task_id + 1):
            X_te, y_te = splits[et][2], splits[et][3]
            preds = classify_fn(X_te)
            acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)
    elapsed = time.time() - t0
    final = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    aa    = sum(final) / N_TASKS
    fgt   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS - 1]) for i in range(N_TASKS - 1)]
    return {'aa': aa, 'forgetting': sum(fgt) / len(fgt), 'cb': len(cb.vectors), 'elapsed': elapsed}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.6)
    print('=' * 70)
    print('  Step 72 -- Soft Retrieval in classify()')
    print('  Split-CIFAR-100, spawn=0.95, full_grad update, ResNet-18 features')
    print(f'  Device: {device} ({torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"})')
    print('  Baseline: 33.5% AA (Step 71 full_grad, 1-NN readout)')
    print('=' * 70)

    print('\nLoading features...')
    splits = load_data()

    results = []

    # ── Part A: Softmax voting (soft retrieval) ───────────────────────────────
    print('\n--- Part A: Softmax-Weighted Voting (Modern Hopfield Retrieval) ---')
    print(f'  {"Config":<40} | {"AA":>7} | {"Forgetting":>10} | {"CB":>7} | {"Time":>6}')
    print('  ' + '-' * 72)

    taus = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    for tau in taus:
        r = run_sequential_eval(splits, lambda cb, t=tau: lambda X: cb.classify_soft(X, tau=t))
        tag = f'tau={tau:.2f} (soft-{"hard" if tau <= 0.01 else "soft"})' if tau == 0.01 else f'tau={tau:.2f}'
        print(f'  {tag:<40} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>9.1f}pp | {r["cb"]:>7} | {r["elapsed"]:>5.1f}s')
        results.append(('soft', tau, r))

    # ── Part B: k-NN (uniform and distance-weighted) ──────────────────────────
    print('\n--- Part B: k-NN Readout ---')
    print(f'  {"Config":<40} | {"AA":>7} | {"Forgetting":>10} | {"CB":>7} | {"Time":>6}')
    print('  ' + '-' * 72)

    # 1-NN baseline first
    r = run_sequential_eval(splits, lambda cb: cb.classify_hard)
    print(f'  {"k=1 (1-NN baseline)":<40} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>9.1f}pp | {r["cb"]:>7} | {r["elapsed"]:>5.1f}s')
    results.append(('knn_uniform', 1, r))

    for k in [3, 5, 10]:
        # Uniform vote
        r = run_sequential_eval(splits, lambda cb, kk=k: lambda X: cb.classify_knn(X, k=kk, weighted=False))
        print(f'  {f"k={k} uniform":<40} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>9.1f}pp | {r["cb"]:>7} | {r["elapsed"]:>5.1f}s')
        results.append(('knn_uniform', k, r))

        # Distance-weighted
        r = run_sequential_eval(splits, lambda cb, kk=k: lambda X: cb.classify_knn(X, k=kk, weighted=True))
        print(f'  {f"k={k} distance-weighted":<40} | {r["aa"]*100:>6.1f}% | {r["forgetting"]*100:>9.1f}pp | {r["cb"]:>7} | {r["elapsed"]:>5.1f}s')
        results.append(('knn_weighted', k, r))

    # ── Summary ───────────────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('  STEP 72 SUMMARY')
    print('=' * 70)
    print(f'  Baseline: 33.5% AA (Step 71 full_grad, 1-NN)')
    print(f'  Published: Fine-tune ~6% | EWC ~33% | SI ~36% | DER++ ~51%\n')

    best = max(results, key=lambda x: x[2]['aa'])
    print(f'  Best config: type={best[0]}, param={best[1]}')
    print(f'    AA={best[2]["aa"]*100:.1f}%  F={best[2]["forgetting"]*100:.1f}pp')
    delta = (best[2]['aa'] - 0.335) * 100
    print(f'    Delta over baseline: {delta:+.1f}pp')

    if best[2]['aa'] > 0.40:
        print('\n  VERDICT: Soft retrieval significantly improves AA (>40%).')
        print('           The fold IS attention — empirically validated.')
    elif best[2]['aa'] > 0.335 + 0.03:
        print('\n  VERDICT: Soft retrieval improves AA (>3pp). Meaningful gain.')
    else:
        print('\n  VERDICT: Soft retrieval does not significantly improve AA.')
        print('           Codebook energy landscape too flat for soft retrieval benefit.')
        print('           Readout limitation is coverage/geometry, not aggregation method.')
    print('=' * 70)

    # ── P-MNIST regression check ──────────────────────────────────────────────
    # Run best tau on Permuted-MNIST to verify no forgetting regression.
    # Baseline (Step 65): AA=56.7%, Forgetting=0.0pp with hard 1-NN.
    best_tau = best[1] if best[0] == 'soft' else 0.01

    print('\n--- P-MNIST Regression Check ---')
    print(f'  Using best tau={best_tau} (from CIFAR-100 sweep)')
    print(f'  Baseline (Step 65): AA=56.7%, Forgetting=0.0pp (hard 1-NN, CPU, NumpyCodebook)\n')

    import torchvision, torchvision.transforms as transforms
    tf = transforms.Compose([transforms.ToTensor()])
    mnist_tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True, transform=tf)
    mnist_te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True, transform=tf)
    X_mn_tr = mnist_tr.data.float().reshape(len(mnist_tr), -1) / 255.0   # (60000, 784)
    y_mn_tr = mnist_tr.targets
    X_mn_te = mnist_te.data.float().reshape(len(mnist_te), -1) / 255.0   # (10000, 784)
    y_mn_te = mnist_te.targets

    # Random projection 784 → 384 (same as Step 65 convention, fixed seed)
    rng_proj = torch.Generator()
    rng_proj.manual_seed(42)
    proj = torch.randn(784, 384, generator=rng_proj, dtype=torch.float32).to(device)
    proj = F.normalize(proj, dim=0)  # column-normalize

    MN_TASKS       = 10
    MN_TRAIN_TASK  = 6000

    # Build task splits (fixed permutations, seed per task)
    mn_splits = []
    for t in range(MN_TASKS):
        perm = torch.randperm(784, generator=torch.Generator().manual_seed(t * 7919))
        Xtr_p = X_mn_tr[:, perm] @ proj
        Xte_p = X_mn_te[:, perm] @ proj
        # Stratified 6000 sample
        rng_s = torch.Generator(); rng_s.manual_seed(t)
        idx = []
        for c in range(10):
            ci = (y_mn_tr == c).nonzero(as_tuple=True)[0]
            ci = ci[torch.randperm(len(ci), generator=rng_s)[:MN_TRAIN_TASK // 10]]
            idx.append(ci)
        idx = torch.cat(idx)
        mn_splits.append((Xtr_p[idx].to(device), y_mn_tr[idx], Xte_p.to(device), y_mn_te))

    def run_pmn_eval(classify_fn_factory):
        cb = FullGradCodebook(384, spawn_thresh=SPAWN_THRESH, device=device)
        acc_matrix = [[None] * MN_TASKS for _ in range(MN_TASKS)]
        t0 = time.time()
        for task_id in range(MN_TASKS):
            X_t, y_t = mn_splits[task_id][0], mn_splits[task_id][1]
            X_n = F.normalize(X_t, dim=1)
            for i in range(len(X_n)):
                cb.step(X_n[i], label=int(y_t[i]))
            classify_fn = classify_fn_factory(cb)
            for et in range(task_id + 1):
                X_te, y_te = mn_splits[et][2], mn_splits[et][3]
                preds = classify_fn(F.normalize(X_te, dim=1))
                acc_matrix[et][task_id] = sum(p == int(g) for p, g in zip(preds, y_te.tolist())) / len(y_te)
        elapsed = time.time() - t0
        final = [acc_matrix[i][MN_TASKS - 1] for i in range(MN_TASKS)]
        aa    = sum(final) / MN_TASKS
        fgt   = [max(0., acc_matrix[i][i] - acc_matrix[i][MN_TASKS - 1]) for i in range(MN_TASKS - 1)]
        return {'aa': aa, 'forgetting': sum(fgt) / len(fgt), 'cb': len(cb.vectors), 'elapsed': elapsed}

    # Hard 1-NN baseline
    r_hard = run_pmn_eval(lambda cb: cb.classify_hard)
    print(f'  {"1-NN (hard)":<30} | AA={r_hard["aa"]*100:.1f}%  F={r_hard["forgetting"]*100:.2f}pp  CB={r_hard["cb"]}  t={r_hard["elapsed"]:.0f}s')

    # Best tau soft
    r_soft = run_pmn_eval(lambda cb, t=best_tau: lambda X: cb.classify_soft(X, tau=t))
    print(f'  {f"soft tau={best_tau}":<30} | AA={r_soft["aa"]*100:.1f}%  F={r_soft["forgetting"]*100:.2f}pp  CB={r_soft["cb"]}  t={r_soft["elapsed"]:.0f}s')

    delta_aa  = (r_soft['aa'] - r_hard['aa']) * 100
    delta_fgt = (r_soft['forgetting'] - r_hard['forgetting']) * 100
    print(f'\n  Delta (soft vs hard): AA {delta_aa:+.1f}pp  |  Forgetting {delta_fgt:+.2f}pp')
    if abs(delta_fgt) < 0.5:
        print('  REGRESSION CHECK: PASS — soft retrieval does not increase forgetting on P-MNIST.')
    else:
        print('  REGRESSION CHECK: FAIL — soft retrieval changes forgetting on P-MNIST.')
    print('=' * 70)


if __name__ == '__main__':
    main()
