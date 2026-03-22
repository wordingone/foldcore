#!/usr/bin/env python3
"""
Step 340 — process() with all 7 stages: state-derived hyperparameters.

Base: Step 339's CompressedFold (the exact process() — class vote + winner update).
Modifications: only to process(). No new classes.

3 FROZEN HYPERPARAMETERS -> STATE-DERIVED:
1. SPAWN THRESHOLD: median(max_sim_excluding_self per entry in V)
   Recomputed after each spawn/attract. Threshold IS codebook state.

2. K PER CLASS: k_c = max(1, min(count(labels==c in top-20 sims), 10))
   Dense classes use higher K. Sparse use lower. K reads from sims.

3. SOFT SPAWN/ATTRACT:
   alpha = sigmoid(10 * (winner_sim - self.thresh))
   alpha >= 0.5 -> attract with weight alpha
   alpha < 0.5  -> spawn
   (No hard if winner_sim > thresh. Threshold is state, not code.)

ALL 7 STAGES:
  1. Computes without external loss — class vote from similarity
  2. Adaptation from computation — attract from matching
  3. Adaptation rate adapts — thresh changes with codebook
  4. Structural constants — codebook geometry
  5. Topology — spawn/attract changes structure
  6. Functional form — K from local density
  7. Representation — thresh is codebook data, read by the rule

TEST:
  A. P-MNIST 10 tasks: must maintain >91% AA, 0pp forgetting
  B. Mixed-function (Step 337): must beat global baseline (93.75%)
  C. S2 ablation: attract now load-bearing (threshold feedback loop)
"""

import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════════════════════
# MODIFIED CompressedFold — process() only changed
# ══════════════════════════════════════════════════════════════════════════════

class CompressedFold:
    def __init__(self, d, lr=0.1, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7       # initial; becomes state-derived after first update
        self.lr     = lr
        self.d      = d
        self.device = device

    def _update_thresh(self):
        """Stage 7: thresh is codebook data, recomputed from codebook state."""
        n = self.V.shape[0]
        if n < 2:
            return
        G = self.V @ self.V.T                            # (n, n) pairwise sims
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())  # median of per-entry max sims

    def process(self, x, label=None):
        """
        Compressed substrate — all 7 stages.
        State-derived: thresh (Stage 7), K per class (Stage 6), soft alpha (Stage 3).
        """
        x = F.normalize(x.to(self.device), dim=0)

        if self.V.shape[0] == 0:                         # bootstrap
            tgt = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            self._update_thresh()
            return tgt

        sims = self.V @ x                                # THE operation

        # Stage 6: K per class from local density in top-20 sims
        top20 = sims.topk(min(20, len(sims))).indices
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask_c = self.labels == c
            if mask_c.sum() == 0: continue
            k_c = max(1, min(int((self.labels[top20] == c).sum().item()), 10))
            scores[c] = sims[mask_c].topk(min(k_c, mask_c.sum())).values.sum()
        prediction = scores.argmax().item()

        target = label if label is not None else prediction   # S1: same code path

        # Stage 3 + 7: soft alpha from state-derived thresh
        target_mask = self.labels == target
        winner_sim  = float(sims[target_mask].max()) if target_mask.sum() > 0 else -1.0
        alpha = float(torch.sigmoid(torch.tensor(10.0 * (winner_sim - self.thresh))))

        if target_mask.sum() > 0 and alpha >= 0.5:      # attract
            w = int(sims.clone().masked_fill_(~target_mask, -1e9).argmax())
            self.V[w] = F.normalize(
                self.V[w] + alpha * self.lr * (x - self.V[w]), dim=0)
        else:                                            # spawn
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])

        self._update_thresh()                            # thresh is codebook state
        return prediction

    def batch_predict(self, X):
        """Vectorized prediction for evaluation. Same V @ x, per-class top-K."""
        X_norm = F.normalize(X.to(self.device), dim=1)
        sims = X_norm @ self.V.T                         # (n, n_cb)
        n_cls = int(self.labels.max().item()) + 1
        top20_global = sims.topk(min(20, sims.shape[1]), dim=1).indices  # (n, 20)
        scores = torch.zeros(X.shape[0], n_cls, device=self.device)
        for c in range(n_cls):
            mask_c = self.labels == c
            if mask_c.sum() == 0: continue
            # Per-sample k_c from density
            k_c_per = (self.labels[top20_global] == c).sum(dim=1).clamp(1, 10)
            k_global = int(k_c_per.float().mean().item())
            k_eff = max(1, min(k_global, mask_c.sum().item()))
            scores[:, c] = sims[:, mask_c].topk(k_eff, dim=1).values.sum(dim=1)
        return scores.argmax(dim=1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# P-MNIST TEST
# ══════════════════════════════════════════════════════════════════════════════

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('./data/mnist',
                                    train=True,  download=False)
    te = torchvision.datasets.MNIST('./data/mnist',
                                    train=False, download=False)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=384, seed=12345):
    rng = np.random.RandomState(seed)
    P = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def stratified(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(10):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


def test_pmnist(target_aa=0.91, target_fgt=0.02):
    print("=" * 60, flush=True)
    print("TEST A: P-MNIST (10 tasks, 6k train / 10k test per task)", flush=True)
    print(f"  State-derived thresh (init 0.7), per-class K, soft alpha", flush=True)
    print(f"  Target: AA >{target_aa*100:.0f}%, Fgt ~0pp", flush=True)
    t0 = time.time()

    try:
        X_tr, y_tr, X_te, y_te = load_mnist()
    except Exception as e:
        print(f"  SKIP: {e}", flush=True)
        return None, None

    D_OUT = 384
    P = make_projection()
    fold = CompressedFold(d=D_OUT, lr=0.1)

    N_TASKS = 10
    N_TRAIN_PC = 600
    N_TEST = 10000
    peak_accs  = []
    task_perms = []
    X_te_t = torch.from_numpy(X_te[:N_TEST]).float().to(DEVICE)
    y_te_arr = y_te[:N_TEST]

    for t in range(N_TASKS):
        perm = list(range(784))
        random.Random(t * 100 + 1).shuffle(perm)
        task_perms.append(perm)
        perm_t = torch.tensor(perm, device=DEVICE)

        X_tr_t, y_tr_t = stratified(X_tr, y_tr, N_TRAIN_PC, seed=t * 7)
        emb_tr = F.normalize(
            torch.from_numpy(X_tr_t).float().to(DEVICE)[:, perm_t] @ P.T, dim=1)

        for i in range(len(emb_tr)):
            fold.process(emb_tr[i], label=int(y_tr_t[i]))

        emb_te = F.normalize(X_te_t[:, perm_t] @ P.T, dim=1)
        preds = fold.batch_predict(emb_te)
        acc_t = float((preds == y_te_arr).mean())
        peak_accs.append(acc_t)
        print(f"  Task {t+1:2d}: peak={acc_t*100:.2f}%  cb={fold.V.shape[0]}  "
              f"thresh={fold.thresh:.4f}", flush=True)

    # Final eval + forgetting
    final_accs = []
    for t in range(N_TASKS):
        perm_t = torch.tensor(task_perms[t], device=DEVICE)
        emb_te = F.normalize(X_te_t[:, perm_t] @ P.T, dim=1)
        preds = fold.batch_predict(emb_te)
        final_accs.append(float((preds == y_te_arr).mean()))

    aa  = float(np.mean(final_accs))
    fgt = float(np.mean([max(0, peak_accs[t] - final_accs[t])
                         for t in range(N_TASKS - 1)]))
    print(f"  Final: {[f'{a*100:.1f}' for a in final_accs]}", flush=True)
    print(f"  AA: {aa*100:.2f}%  Forgetting: {fgt*100:.2f}pp  "
          f"[{'PASS' if aa >= target_aa else 'FAIL'}]", flush=True)
    print(f"  Step 339 baseline: 93.10% AA, 0.00pp fgt", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return aa, fgt


# ══════════════════════════════════════════════════════════════════════════════
# MIXED FUNCTION TEST
# ══════════════════════════════════════════════════════════════════════════════

def mixed_features(a, b):
    """Feature vector for mixed-function: [a/20, b/20] normalized."""
    return torch.tensor([float(a) / 20.0, float(b) / 20.0], dtype=torch.float32)


def test_mixed(target=0.9375):
    print("=" * 60, flush=True)
    print("TEST B: Mixed-function (b<=10: a%b, b>=11: floor(a/b))", flush=True)
    print(f"  Features: normalized [a/20, b/20]", flush=True)
    print(f"  Target: beat {target*100:.2f}%  (Step 337 global CL+phi baseline)", flush=True)
    t0 = time.time()

    TRAIN_MAX = 20
    entries = [(a, b, (a % b if b <= 10 else a // b))
               for a in range(1, TRAIN_MAX+1)
               for b in range(1, TRAIN_MAX+1)]
    n = len(entries)

    # Build codebook (3 training epochs)
    fold = CompressedFold(d=2, lr=0.1)
    rng = np.random.RandomState(42)
    for _ in range(3):
        for i in rng.permutation(n):
            a, b, y = entries[i]
            fold.process(mixed_features(a, b), label=y)

    print(f"  Codebook: {fold.V.shape[0]} entries  thresh={fold.thresh:.4f}", flush=True)

    # Evaluate with batch_predict (no update)
    X_all = torch.stack([mixed_features(a, b) for a, b, y in entries])
    y_all = np.array([y for a, b, y in entries])
    preds = fold.batch_predict(X_all)
    acc = float((preds == y_all).mean())
    print(f"  Train-set accuracy: {acc*100:.2f}%  "
          f"[{'PASS' if acc >= target else 'FAIL'}]", flush=True)

    # Per-function breakdown
    func_mod = np.array([1 if b <= 10 else 0 for a, b, y in entries])
    acc_mod = float((preds[func_mod == 1] == y_all[func_mod == 1]).mean())
    acc_div = float((preds[func_mod == 0] == y_all[func_mod == 0]).mean())
    print(f"  Modular (b<=10): {acc_mod*100:.2f}%  Division (b>=11): {acc_div*100:.2f}%",
          flush=True)

    if acc < target:
        print(f"  NOTE: cosine vote on [a/20, b/20] doesn't capture arithmetic "
              f"class structure.", flush=True)
        print(f"  Fourier features (sin/cos of a/b) needed for modular; division "
              f"may work with a/b alone.", flush=True)

    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# S2 ABLATION — test if attract is now load-bearing
# ══════════════════════════════════════════════════════════════════════════════

def s2_ablation(target_aa=0.91):
    print("=" * 60, flush=True)
    print("TEST C: S2 ABLATION (P-MNIST Task 1)", flush=True)
    print("  Hypothesis: attract load-bearing via threshold feedback loop", flush=True)
    t0 = time.time()

    try:
        X_tr, y_tr, X_te, y_te = load_mnist()
    except Exception as e:
        print(f"  SKIP: {e}", flush=True)
        return

    D_OUT = 384
    P = make_projection()
    perm = list(range(784))
    random.Random(101).shuffle(perm)
    perm_t = torch.tensor(perm, device=DEVICE)
    N_TRAIN_PC = 600

    X_tr_t, y_tr_t = stratified(X_tr, y_tr, N_TRAIN_PC, seed=0)
    emb_tr = F.normalize(
        torch.from_numpy(X_tr_t).float().to(DEVICE)[:, perm_t] @ P.T, dim=1)
    emb_te = F.normalize(
        torch.from_numpy(X_te[:10000]).float().to(DEVICE)[:, perm_t] @ P.T, dim=1)
    y_te_arr = y_te[:10000]

    def train_and_eval(lr_val, label='full'):
        fold = CompressedFold(d=D_OUT, lr=lr_val)
        for i in range(len(emb_tr)):
            fold.process(emb_tr[i], label=int(y_tr_t[i]))
        preds = fold.batch_predict(emb_te)
        acc = float((preds == y_te_arr).mean())
        return acc, fold.V.shape[0], fold.thresh

    acc_full, cb_full, thresh_full = train_and_eval(0.1, 'full')
    acc_noa,  cb_noa,  thresh_noa  = train_and_eval(0.0, 'no_attract')
    print(f"  Baseline (full, lr=0.1):   {acc_full*100:.2f}%  "
          f"cb={cb_full}  thresh={thresh_full:.4f}", flush=True)
    print(f"  No attract (lr=0.0):       {acc_noa*100:.2f}%  "
          f"cb={cb_noa}  thresh={thresh_noa:.4f}  "
          f"(delta: {(acc_noa-acc_full)*100:+.2f}pp)", flush=True)
    print(f"  Thresh difference: full={thresh_full:.4f}  no_attract={thresh_noa:.4f}  "
          f"(feedback: {'YES' if abs(thresh_full-thresh_noa)>0.01 else 'NO'})", flush=True)
    load_bearing = abs(acc_noa - acc_full) > 0.01
    print(f"  Attract load-bearing (>1pp gap): {'YES' if load_bearing else 'NO'}", flush=True)
    print(f"  Step 339 comparison: -0.72pp with no attract", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return acc_full, acc_noa


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Step 340 — All 7 Stages: State-Derived Hyperparameters", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Changes to process():", flush=True)
    print(f"  1. thresh = median(max-sim-excl-self per entry) [Stage 7]", flush=True)
    print(f"  2. k_c = count(class c in top-20 sims) [Stage 6]", flush=True)
    print(f"  3. alpha = sigmoid(10*(winner_sim - thresh)) [Stage 3]", flush=True)
    print(flush=True)

    acc_pm, fgt_pm = test_pmnist(target_aa=0.91)
    print(flush=True)
    acc_mixed = test_mixed(target=0.9375)
    print(flush=True)
    acc_full, acc_noa = s2_ablation()

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 340 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"7 stages: all present in process().", flush=True)
    print(flush=True)
    if acc_pm is not None:
        print(f"A. P-MNIST AA:       {acc_pm*100:.2f}%  (target 91%+)  "
              f"{'PASS' if acc_pm >= 0.91 else 'FAIL'}", flush=True)
        print(f"   Forgetting:        {fgt_pm*100:.2f}pp  (target ~0pp)", flush=True)
    print(f"B. Mixed-function:   {acc_mixed*100:.2f}%  (target 93.75%+)  "
          f"{'PASS' if acc_mixed >= 0.9375 else 'FAIL'}", flush=True)
    if acc_full is not None:
        delta = (acc_noa - acc_full) * 100
        print(f"C. S2 attract gap:   {abs(delta):.2f}pp  "
              f"({'load-bearing' if abs(delta) > 1.0 else 'NOT load-bearing'})", flush=True)

    print(flush=True)
    print("=" * 60, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 60, flush=True)
    kill_pm = (acc_pm is not None) and (acc_pm < 0.91)
    print(f"P-MNIST >= 91%:   {'PASS' if not kill_pm else 'KILL'}", flush=True)
    print(f"Kill triggered:   {'YES' if kill_pm else 'NO'}", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
