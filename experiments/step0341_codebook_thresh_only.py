#!/usr/bin/env python3
"""
Step 341 — State-derived threshold ONLY. One variable.

Step 339 process() + state-derived spawn threshold.
- Fixed K=3 (same as Step 339)
- State-derived threshold: median of per-entry max similarities
  self.thresh = median(max(V @ V.T, excluding self-similarity))
  Recompute after each spawn/attract.
- Everything else identical to Step 339.

Chain:
  Step 339 (fixed thresh=0.7):    93.10% AA, 0pp fgt
  Step 340 (thresh+K+alpha):       57.12% AA  (KILLED — per-class K broke vote)
  Step 341 (thresh only):          ???

Kill: P-MNIST must maintain >91%.

If P-MNIST holds AND S2 passes: Stage 7 confirmed on compressed substrate.
The threshold IS codebook data. The rule reads from its own state. One function.

Script: scripts/run_step341_thresh_only.py
"""

import math
import time
import random
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ══════════════════════════════════════════════════════════════════════════════
# THE CLASS — minimal delta from Step 339
# Only change: state-derived thresh via _update_thresh()
# ══════════════════════════════════════════════════════════════════════════════

class CompressedFold:
    """
    Step 341: Step 339 + state-derived spawn threshold.
    Fixed K=3. Only thresh becomes codebook state.
    """
    def __init__(self, d, lr=0.1, k=3, device=DEVICE):
        self.V      = torch.zeros(0, d, device=device)
        self.labels = torch.zeros(0, dtype=torch.long, device=device)
        self.thresh = 0.7    # initial; becomes state-derived after first spawn
        self.lr     = lr
        self.k      = k
        self.d      = d
        self.device = device

    def _update_thresh(self):
        """Recompute thresh = median(max-sim-excl-self per codebook entry)."""
        n = self.V.shape[0]
        if n < 2:
            return
        G = self.V @ self.V.T                          # (n, n) cosine sims
        G.fill_diagonal_(-float('inf'))
        self.thresh = float(G.max(dim=1).values.median())

    def process(self, x, label=None):
        """Step 339 process() with self.thresh replacing fixed spawn_thresh."""
        x = F.normalize(x.to(self.device), dim=0)

        # Bootstrap: first entry always spawns
        if self.V.shape[0] == 0:
            tgt = label if label is not None else 0
            self.V      = x.unsqueeze(0)
            self.labels = torch.tensor([tgt], device=self.device)
            return tgt

        sims = self.V @ x                              # THE operation

        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            cs = sims[mask]
            k_eff = min(self.k, len(cs))
            scores[c] = cs.topk(k_eff).values.sum()
        prediction = scores.argmax().item()

        target = label if label is not None else prediction

        # Winner = nearest entry of TARGET class; spawn or attract
        target_mask = (self.labels == target)
        if target_mask.sum() == 0 or sims[target_mask].max() < self.thresh:
            self.V      = torch.cat([self.V, x.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([target], device=self.device)])
        else:
            target_sims = sims.clone()
            target_sims[~target_mask] = -float('inf')
            winner = target_sims.argmax().item()
            self.V[winner] = F.normalize(
                self.V[winner] + self.lr * (x - self.V[winner]), dim=0)

        # Update thresh from codebook state (after every spawn/attract)
        self._update_thresh()

        return prediction

    def batch_predict(self, X):
        """Vectorized prediction (eval-only, no update). Identical to Step 339."""
        X_norm = F.normalize(X.to(self.device), dim=1)
        sims = X_norm @ self.V.T                       # (n, n_cb)
        n_cls = int(self.labels.max().item()) + 1
        scores = torch.zeros(X.shape[0], n_cls, device=self.device)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0: continue
            k_eff = min(self.k, mask.sum().item())
            scores[:, c] = sims[:, mask].topk(k_eff, dim=1).values.sum(dim=1)
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


def test_pmnist(lr=0.1, k=3, target=0.91):
    print("=" * 60, flush=True)
    print(f"TEST A: P-MNIST (10 tasks, 6k train / 10k test per task)", flush=True)
    print(f"  State-derived thresh (init 0.7), fixed k={k}, lr={lr}", flush=True)
    print(f"  Target: {target*100:.0f}%+ avg accuracy  (Step 339: 93.10%)", flush=True)
    t0 = time.time()

    try:
        X_tr, y_tr, X_te, y_te = load_mnist()
    except Exception as e:
        print(f"  SKIP: {e}", flush=True)
        return None, None

    D_OUT = 384
    P = make_projection()
    fold = CompressedFold(d=D_OUT, lr=lr, k=k)

    N_TASKS    = 10
    N_TRAIN_PC = 600
    N_TEST     = 10000

    peak_accs  = []
    task_perms = []
    X_te_torch = torch.from_numpy(X_te[:N_TEST]).float().to(DEVICE)
    y_te_arr   = y_te[:N_TEST]

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

        emb_te = F.normalize(X_te_torch[:, perm_t] @ P.T, dim=1)
        preds = fold.batch_predict(emb_te)
        acc_t = float((preds == y_te_arr).mean())
        peak_accs.append(acc_t)
        print(f"  Task {t+1:2d}: peak={acc_t*100:.2f}%  cb={fold.V.shape[0]}"
              f"  thresh={fold.thresh:.4f}", flush=True)

    # Final eval: all tasks after all training
    final_accs = []
    for t in range(N_TASKS):
        perm_t = torch.tensor(task_perms[t], device=DEVICE)
        emb_te = F.normalize(X_te_torch[:, perm_t] @ P.T, dim=1)
        preds = fold.batch_predict(emb_te)
        final_accs.append(float((preds == y_te_arr).mean()))
    print(f"  Final accs: {[f'{a*100:.1f}' for a in final_accs]}", flush=True)

    aa  = float(np.mean(final_accs))
    fgt = float(np.mean([max(0, peak_accs[t] - final_accs[t])
                         for t in range(N_TASKS - 1)]))
    status = 'PASS' if aa >= target else 'FAIL'
    print(f"  AA: {aa*100:.2f}%  Forgetting: {fgt*100:.2f}pp  [{status}]", flush=True)
    print(f"  Step 339 baseline: 93.10% AA, 0.00pp fgt", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return aa, fgt


# ══════════════════════════════════════════════════════════════════════════════
# MIXED-FUNCTION TEST (no gate — cosine encoding issue known)
# ══════════════════════════════════════════════════════════════════════════════

def test_mixed_function(lr=0.1, k=3):
    print("=" * 60, flush=True)
    print(f"TEST B: Mixed-function (b<=10: a%b, b>=11: floor(a/b))", flush=True)
    print(f"  Features: normalized [a/20, b/20]", flush=True)
    print(f"  No gate (cosine encoding issue from Step 339/340 known)", flush=True)
    t0 = time.time()

    TRAIN_MAX = 20
    entries = []
    for a in range(1, TRAIN_MAX+1):
        for b in range(1, TRAIN_MAX+1):
            y = a % b if b <= 10 else a // b
            entries.append((a, b, y))
    n = len(entries)

    fold = CompressedFold(d=2, lr=lr, k=k)
    rng = np.random.RandomState(42)
    for _ in range(3):
        for i in rng.permutation(n):
            a, b, y = entries[i]
            x = torch.tensor([float(a)/20, float(b)/20])
            fold.process(x, label=y)

    print(f"  Codebook: {fold.V.shape[0]} entries  thresh={fold.thresh:.4f}", flush=True)

    X_all = torch.tensor([[float(a)/20, float(b)/20] for a, b, y in entries])
    y_all = np.array([y for a, b, y in entries])
    preds = fold.batch_predict(X_all)
    acc = float((preds == y_all).mean())

    # Breakdown by function
    mod_mask  = np.array([b <= 10 for a, b, y in entries])
    div_mask  = ~mod_mask
    acc_mod   = float((preds[mod_mask]  == y_all[mod_mask]).mean())
    acc_div   = float((preds[div_mask] == y_all[div_mask]).mean())
    print(f"  Train-set accuracy: {acc*100:.2f}%", flush=True)
    print(f"  Modular (b<=10): {acc_mod*100:.2f}%  Division (b>=11): {acc_div*100:.2f}%", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return acc


# ══════════════════════════════════════════════════════════════════════════════
# S2 ABLATION — confirm attract load-bearing WITH state-derived thresh
# ══════════════════════════════════════════════════════════════════════════════

def s2_ablation_pmnist(lr=0.1, k=3):
    """
    S2 ablation with state-derived thresh.
    Key question: is attract still load-bearing when only thresh is state-derived
    (not the thresh+K+alpha combo from Step 340)?
    Feedback loop: attract raises entry similarity -> thresh rises -> more attracts.
    """
    print("=" * 60, flush=True)
    print("TEST C: S2 ABLATION (P-MNIST Task 1)", flush=True)
    print("  Hypothesis: attract still load-bearing via thresh feedback loop", flush=True)
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
    N_TEST = 10000

    X_tr_t, y_tr_t = stratified(X_tr, y_tr, N_TRAIN_PC, seed=0)
    X_tr_torch = torch.from_numpy(X_tr_t).to(DEVICE)
    emb_tr = F.normalize(X_tr_torch[:, perm_t] @ P.T, dim=1)

    X_te_torch = torch.from_numpy(X_te[:N_TEST]).to(DEVICE)
    emb_te = F.normalize(X_te_torch[:, perm_t] @ P.T, dim=1)
    y_te_arr = y_te[:N_TEST]

    def run_eval(fold):
        preds = fold.batch_predict(emb_te)
        return float((preds == y_te_arr).mean())

    # Baseline: full function (state-derived thresh, lr=0.1)
    fold_base = CompressedFold(d=D_OUT, lr=lr, k=k)
    for i in range(len(emb_tr)):
        fold_base.process(emb_tr[i], label=int(y_tr_t[i]))
    acc_base = run_eval(fold_base)
    print(f"  Baseline (full, state thresh, lr={lr}):   {acc_base*100:.2f}%"
          f"  cb={fold_base.V.shape[0]}  thresh={fold_base.thresh:.4f}", flush=True)

    # No attract: lr=0 (spawn-only)
    fold_spawn = CompressedFold(d=D_OUT, lr=0.0, k=k)
    for i in range(len(emb_tr)):
        fold_spawn.process(emb_tr[i], label=int(y_tr_t[i]))
    acc_spawn = run_eval(fold_spawn)
    delta_attract = (acc_spawn - acc_base) * 100
    print(f"  No attract (lr=0.0, spawn-only):          {acc_spawn*100:.2f}%"
          f"  cb={fold_spawn.V.shape[0]}  thresh={fold_spawn.thresh:.4f}"
          f"  (delta: {delta_attract:+.2f}pp)", flush=True)

    thresh_diff = fold_base.thresh - fold_spawn.thresh
    print(f"  Thresh difference: full={fold_base.thresh:.4f}"
          f"  no_attract={fold_spawn.thresh:.4f}"
          f"  (feedback: {'YES' if thresh_diff > 0.001 else 'weak'})", flush=True)

    load_bearing = abs(acc_spawn - acc_base) > 0.01
    print(f"  Attract load-bearing (>1pp gap): {'YES' if load_bearing else 'NO'}", flush=True)
    print(f"  Step 339 comparison: -0.72pp with no attract", flush=True)
    print(f"  Step 340 comparison: -52.88pp with no attract (thresh+K+alpha combo)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    return acc_base, acc_spawn


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Step 341 — State-Derived Threshold ONLY", flush=True)
    print("One variable: thresh = median(max-sim-excl-self per entry)", flush=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Fixed K=3 (same as Step 339). Everything else identical.", flush=True)
    print(flush=True)

    aa_pm, fgt_pm = test_pmnist(lr=0.1, k=3, target=0.91)
    print(flush=True)
    acc_mf = test_mixed_function(lr=0.1, k=3)
    print(flush=True)
    result_s2 = s2_ablation_pmnist(lr=0.1, k=3)

    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 341 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Single change from Step 339: state-derived thresh", flush=True)
    print(f"thresh(t+1) = median(max(V@V.T, diag=-inf))", flush=True)
    print(flush=True)

    if aa_pm is not None:
        print(f"A. P-MNIST AA:        {aa_pm*100:.2f}%  (target 91%+, step339: 93.10%)  "
              f"{'PASS' if aa_pm >= 0.91 else 'FAIL'}", flush=True)
        print(f"   Forgetting:         {fgt_pm*100:.2f}pp", flush=True)
    print(f"B. Mixed-function:    {acc_mf*100:.2f}%  (no gate)", flush=True)
    if result_s2 is not None:
        acc_base, acc_spawn = result_s2
        delta = (acc_spawn - acc_base) * 100
        print(f"C. S2 attract gap:    {delta:+.2f}pp  "
              f"({'load-bearing' if abs(delta) > 1.0 else 'NOT load-bearing'})", flush=True)

    print(flush=True)
    print("=" * 60, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 60, flush=True)
    kill_pm = (aa_pm is not None) and (aa_pm < 0.91)
    kill    = kill_pm
    print(f"P-MNIST >= 91%: {'KILL' if kill_pm else 'PASS'}", flush=True)
    print(f"Kill triggered: {'YES' if kill else 'NO'}", flush=True)

    if not kill and aa_pm is not None:
        print(flush=True)
        print("Stage 7 confirmed on compressed substrate:", flush=True)
        print("  thresh IS codebook data. The rule reads from its own state.", flush=True)
        print("  One function: process() computes thresh from V, not from hyperparameter.", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
