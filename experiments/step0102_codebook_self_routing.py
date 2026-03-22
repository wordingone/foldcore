#!/usr/bin/env python3
"""
Step 102 -- Self-Routing Codebook. Targeting Separation 4: System/State.
Spec.

Each codebook entry carries a gate scalar (learned participation weight).
Gate increases when winner label matches input, decreases when not.
Readout: gate-weighted class scores.

lr=0 (no vector updates). Sweep gate_lr={0.001, 0.01, 0.1}.
P-MNIST full 10-task, sp=0.7.

Compare on SAME codebook:
- classify_gated: gate-weighted readout
- classify_topk(k=5): Step 99 bar
- classify_1nn: raw 1-NN

Kill criterion:
- gated <= topk(5) on same codebook -> DISPROVED
- gate std < 0.01 -> DISPROVED (gates didn't learn)
"""

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS       = 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

SPAWN_THRESH  = 0.7
K_TOPK        = 5
GATE_LR_VALS  = [0.001, 0.01, 0.1]
# lr=0: no vector updates, only gate learning


# ─── SelfRoutingFold ──────────────────────────────────────────────────────────

class SelfRoutingFold:
    def __init__(self, d, gate_lr=0.01, spawn_thresh=0.7):
        self.V          = torch.empty(0, d, device=DEVICE)
        self.labels     = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.gates      = torch.empty(0, device=DEVICE)
        self.gate_lr    = gate_lr
        self.spawn_thresh = spawn_thresh
        self.d          = d
        self.n_spawned  = 0

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.gates  = torch.cat([self.gates,
                                     torch.tensor([1.0], device=DEVICE)])
            self.n_spawned += 1
            return
        sims   = self.V @ r
        winner = sims.argmax().item()
        # Gate update: increase if correct, decrease if wrong
        if self.labels[winner].item() == label:
            self.gates[winner] = min(2.0, self.gates[winner].item() + self.gate_lr)
        else:
            self.gates[winner] = max(0.1, self.gates[winner].item() - self.gate_lr)
        # lr=0: no vector update

    def eval_batch(self, R):
        """
        Batch eval all three classifiers.
        R: (n, d) GPU tensor.
        Returns: gated_preds, topk_preds, nn_preds (all CPU long tensors)
        """
        R     = F.normalize(R, dim=1)
        sims  = R @ self.V.T                  # (n, N)
        n     = len(R)
        n_cls = int(self.labels.max().item()) + 1

        # ── 1-NN ──────────────────────────────────────────────────────────────
        nn_preds = self.labels[sims.argmax(dim=1)].cpu()

        # ── Top-K (k=5) ───────────────────────────────────────────────────────
        topk_scores = torch.zeros(n, n_cls, device=DEVICE)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0:
                continue
            cs   = sims[:, mask]
            k_eff = min(K_TOPK, cs.shape[1])
            topk_scores[:, c] = cs.topk(k_eff, dim=1).values.sum(dim=1)
        topk_preds = topk_scores.argmax(dim=1).cpu()

        # ── Gated ─────────────────────────────────────────────────────────────
        # weighted[i, j] = sims[i,j] * gates[j]
        weighted = sims * self.gates.unsqueeze(0)   # (n, N)
        gate_scores = torch.zeros(n, n_cls, device=DEVICE)
        labels_exp  = self.labels.unsqueeze(0).expand(n, -1)  # (n, N)
        gate_scores.scatter_add_(1, labels_exp, weighted)
        gated_preds = gate_scores.argmax(dim=1).cpu()

        return gated_preds, topk_preds, nn_preds

    def gate_stats(self):
        g = self.gates.cpu()
        return {
            'mean': g.mean().item(),
            'std':  g.std().item(),
            'min':  g.min().item(),
            'max':  g.max().item(),
            'n_high': (g > 1.5).sum().item(),
            'n_low':  (g < 0.3).sum().item(),
        }


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=D_OUT, seed=12345):
    rng = np.random.RandomState(seed)
    P   = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed(X_flat_np, perm, P):
    X_t = torch.from_numpy(X_flat_np[:, perm]).to(DEVICE)
    return F.normalize(X_t @ P.T, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_aa(mat):
    vals = [mat[t][N_TASKS - 1] for t in range(N_TASKS)
            if mat[t][N_TASKS - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat):
    vals = []
    for t in range(N_TASKS - 1):
        if mat[t][t] is not None and mat[t][N_TASKS - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][N_TASKS - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Run one config ───────────────────────────────────────────────────────────

def run_config(gate_lr, X_tr, y_tr, test_embeds, perms, P, y_te_t):
    print(f"\n{'='*65}", flush=True)
    print(f"CONFIG: gate_lr={gate_lr}, lr=0, sp={SPAWN_THRESH}", flush=True)
    print(f"{'='*65}", flush=True)

    model = SelfRoutingFold(D_OUT, gate_lr=gate_lr, spawn_thresh=SPAWN_THRESH)

    # acc[classifier][eval_task][train_task]
    acc_gated = [[None] * N_TASKS for _ in range(N_TASKS)]
    acc_topk  = [[None] * N_TASKS for _ in range(N_TASKS)]
    acc_nn    = [[None] * N_TASKS for _ in range(N_TASKS)]

    for task_id in range(N_TASKS):
        print(f"\n--- Task {task_id} ---", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P)
        labels_list = y_sub.tolist()

        cb_before = model.V.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], labels_list[i])
        cb_after = model.V.shape[0]
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after-cb_before})", flush=True)
        print(f"  Train: {time.time()-t_task:.1f}s", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            gp, tp, np_ = model.eval_batch(test_embeds[eval_task])
            acc_gated[eval_task][task_id] = (gp  == y_te_t).float().mean().item()
            acc_topk [eval_task][task_id] = (tp  == y_te_t).float().mean().item()
            acc_nn   [eval_task][task_id] = (np_ == y_te_t).float().mean().item()
        print(f"  Eval ({task_id+1} tasks): {time.time()-t_eval:.1f}s", flush=True)

        g = acc_gated[task_id][task_id]
        t = acc_topk [task_id][task_id]
        n = acc_nn   [task_id][task_id]
        print(f"  Peek: gated={g*100:.1f}% topk={t*100:.1f}% 1nn={n*100:.1f}%",
              flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    aa_g  = compute_aa(acc_gated)
    aa_t  = compute_aa(acc_topk)
    aa_n  = compute_aa(acc_nn)
    fgt_g = compute_fgt(acc_gated)
    fgt_t = compute_fgt(acc_topk)
    fgt_n = compute_fgt(acc_nn)

    stats = model.gate_stats()

    print(f"\n  RESULT gate_lr={gate_lr}:", flush=True)
    print(f"  {'':12} {'AA':>7} {'Forgetting':>11}", flush=True)
    print(f"  {'gated':12} {aa_g*100:>6.1f}% {fgt_g*100:>10.1f}pp", flush=True)
    print(f"  {'topk(k=5)':12} {aa_t*100:>6.1f}% {fgt_t*100:>10.1f}pp", flush=True)
    print(f"  {'1-NN':12} {aa_n*100:>6.1f}% {fgt_n*100:>10.1f}pp", flush=True)
    print(f"  gated vs topk: {(aa_g-aa_t)*100:+.1f}pp", flush=True)
    print(f"  Gate stats: mean={stats['mean']:.3f} std={stats['std']:.3f} "
          f"min={stats['min']:.3f} max={stats['max']:.3f}", flush=True)
    print(f"  Gates > 1.5: {stats['n_high']}  Gates < 0.3: {stats['n_low']}  "
          f"CB size: {model.V.shape[0]}", flush=True)

    verdict = "DISPROVED" if aa_g <= aa_t else "PASSES"
    gate_verdict = "DISPROVED (uniform)" if stats['std'] < 0.01 else "STRUCTURED"
    print(f"  Kill: {verdict}  Gates: {gate_verdict}", flush=True)

    return {
        'gate_lr': gate_lr,
        'aa_gated': aa_g, 'fgt_gated': fgt_g,
        'aa_topk': aa_t,  'fgt_topk': fgt_t,
        'aa_nn': aa_n,    'fgt_nn': fgt_n,
        'gate_stats': stats,
        'cb_size': model.V.shape[0],
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 102 -- Self-Routing Codebook, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"sp={SPAWN_THRESH}, lr=0 (gates only), gate_lr sweep: {GATE_LR_VALS}", flush=True)
    print(f"Classifiers: gated, topk(k={K_TOPK}), 1-NN", flush=True)
    print(f"Kill: gated <= topk(5) OR gate std < 0.01", flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P      = make_projection()
    y_te_t = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    print("Pre-embedding test sets...", flush=True)
    test_embeds = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    print(f"  Done: {N_TASKS} tasks x {N_TEST_TASK} samples", flush=True)

    results = []
    for gate_lr in GATE_LR_VALS:
        r = run_config(gate_lr, X_tr, y_tr, test_embeds, perms, P, y_te_t)
        results.append(r)

    elapsed = time.time() - t0

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}", flush=True)
    print("STEP 102 FINAL SUMMARY -- Self-Routing Codebook", flush=True)
    print(f"Step 99 bar: topk(5)=91.8%, 1-NN=86.8%, forgetting=0.0pp", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n{'gate_lr':>9} | {'gated AA':>9} {'fgt':>6} | {'topk AA':>8} {'fgt':>6} | "
          f"{'vs topk':>8} | {'gate std':>9} | {'verdict':>12}", flush=True)
    print("-" * 90, flush=True)

    any_passes = False
    for r in results:
        delta   = r['aa_gated'] - r['aa_topk']
        verdict = "DISPROVED" if r['aa_gated'] <= r['aa_topk'] else "PASSES"
        if r['gate_stats']['std'] < 0.01:
            verdict = "DISPROVED(unif)"
        if verdict == "PASSES":
            any_passes = True
        print(f"{r['gate_lr']:>9} | {r['aa_gated']*100:>8.1f}% "
              f"{r['fgt_gated']*100:>5.1f}pp | "
              f"{r['aa_topk']*100:>7.1f}% {r['fgt_topk']*100:>5.1f}pp | "
              f"{delta*100:>+7.1f}pp | "
              f"{r['gate_stats']['std']:>9.4f} | "
              f"{verdict:>12}", flush=True)

    print(flush=True)
    if any_passes:
        best = max(results, key=lambda r: r['aa_gated'])
        print(f"VERDICT: PASSES -- best gate_lr={best['gate_lr']}: "
              f"gated {best['aa_gated']*100:.1f}% vs topk {best['aa_topk']*100:.1f}% "
              f"({(best['aa_gated']-best['aa_topk'])*100:+.1f}pp)", flush=True)
        print(f"  Gate stats: mean={best['gate_stats']['mean']:.3f} "
              f"std={best['gate_stats']['std']:.3f}", flush=True)
        if best['fgt_gated'] > 0.05:
            print(f"  FORGETTING: FAIL ({best['fgt_gated']*100:.1f}pp > 5pp)", flush=True)
        else:
            print(f"  FORGETTING: PASS ({best['fgt_gated']*100:.1f}pp <= 5pp)", flush=True)
    else:
        print("VERDICT: DISPROVED -- gated readout does not beat top-k on any config",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
