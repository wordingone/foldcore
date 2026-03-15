#!/usr/bin/env python3
"""
Step 104 -- Accumulated Outer Product substrate. P-MNIST full 10-task.
Leo mail 1249.

Simplest possible substrate: one d×C matrix.
No codebook. No spawning. No dynamics.

Variants:
  1. classify       -- count-normalized (per-class mean)
  2. classify_raw   -- unnormalized sum
  3. EMA            -- running mean, alpha sweep {0.001, 0.01, 0.1}

Kill criterion: AA <= 30% -> DISPROVED.
Baselines: fold 56.7%, 1-NN 86.8%, top-k(5) 91.8%.
If AA > 56.7%: codebook was unnecessary complexity.
If AA > 86.8%: even 1-NN was unnecessary.
"""

import math
import random
import sys
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

EMA_ALPHAS = [0.001, 0.01, 0.1]


# ─── Substrates ───────────────────────────────────────────────────────────────

class AccumulatedOP:
    def __init__(self, d, n_classes=10):
        self.M      = torch.zeros(d, n_classes, device=DEVICE)
        self.counts = torch.zeros(n_classes, device=DEVICE)

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        self.M[:, label] += r
        self.counts[label] += 1

    def classify(self, R):
        """Count-normalized (per-class mean). R: (n, d)."""
        R = F.normalize(R, dim=1)
        M_norm = self.M / (self.counts.unsqueeze(0) + 1e-10)  # (d, C)
        scores = R @ M_norm  # (n, C)
        return scores.argmax(dim=1).cpu()

    def classify_raw(self, R):
        """Unnormalized sum. R: (n, d)."""
        R = F.normalize(R, dim=1)
        scores = R @ self.M  # (n, C)
        return scores.argmax(dim=1).cpu()


class EMASubstrate:
    def __init__(self, d, n_classes=10, alpha=0.01):
        self.M     = torch.zeros(d, n_classes, device=DEVICE)
        self.alpha = alpha

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        self.M[:, label] = (1 - self.alpha) * self.M[:, label] + self.alpha * r

    def classify(self, R):
        R = F.normalize(R, dim=1)
        scores = R @ self.M  # (n, C)
        return scores.argmax(dim=1).cpu()


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        os.environ.get('MNIST_DATA', './data'), train=True,  download=True)
    te = torchvision.datasets.MNIST(
        os.environ.get('MNIST_DATA', './data'), train=False, download=True)
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

def compute_aa(mat, n_tasks):
    vals = [mat[t][n_tasks - 1] for t in range(n_tasks)
            if mat[t][n_tasks - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat, n_tasks):
    vals = []
    for t in range(n_tasks - 1):
        if mat[t][t] is not None and mat[t][n_tasks - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n_tasks - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Run one substrate variant ─────────────────────────────────────────────

def run_variant(name, substrate, tasks_train, tasks_test, perms, P):
    """
    Train on all tasks sequentially, eval all seen tasks after each.
    Returns (aa, fgt).
    """
    n = N_TASKS
    acc_mat = [[None] * n for _ in range(n)]

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed(X_tr, perms[task_id], P)
        for i in range(len(y_tr)):
            substrate.step(R_tr[i], int(y_tr[i]))
        train_s = time.time() - t_train

        # Eval all seen tasks
        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = tasks_test[eval_task]
            R_te = embed(X_te, perms[eval_task], P)
            if hasattr(substrate, 'classify'):
                preds = substrate.classify(R_te)
            else:
                preds = substrate.classify(R_te)
            y_te_t = torch.from_numpy(y_te)
            acc = (preds == y_te_t).float().mean().item()
            acc_mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_now  = sum(acc_mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s aa={aa_now*100:.1f}%",
              flush=True)

    aa  = compute_aa(acc_mat, n)
    fgt = compute_fgt(acc_mat, n)
    return aa, fgt


def run_aop_both(substrate, name_norm, name_raw, tasks_train, tasks_test, perms, P):
    """
    Run AccumulatedOP and eval both classify and classify_raw from same trained state.
    """
    n = N_TASKS
    mat_norm = [[None] * n for _ in range(n)]
    mat_raw  = [[None] * n for _ in range(n)]

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed(X_tr, perms[task_id], P)
        for i in range(len(y_tr)):
            substrate.step(R_tr[i], int(y_tr[i]))
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = tasks_test[eval_task]
            R_te = embed(X_te, perms[eval_task], P)
            y_te_t = torch.from_numpy(y_te)

            preds_n = substrate.classify(R_te)
            preds_r = substrate.classify_raw(R_te)

            mat_norm[eval_task][task_id] = (preds_n == y_te_t).float().mean().item()
            mat_raw[eval_task][task_id]  = (preds_r == y_te_t).float().mean().item()
        eval_s = time.time() - t_eval

        aa_n = sum(mat_norm[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        aa_r = sum(mat_raw[t][task_id]  for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"norm={aa_n*100:.1f}% raw={aa_r*100:.1f}%", flush=True)

    aa_n  = compute_aa(mat_norm, n)
    fgt_n = compute_fgt(mat_norm, n)
    aa_r  = compute_aa(mat_raw, n)
    fgt_r = compute_fgt(mat_raw, n)
    return aa_n, fgt_n, aa_r, fgt_r


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 104 -- Accumulated Outer Product, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Baselines: fold=56.7%, 1-NN=86.8%, top-k(5)=91.8%", flush=True)
    print(f"Kill: AA <= 30%", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    P    = make_projection()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train = []
    tasks_test  = []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))
        tasks_test.append((X_te_all, y_te_all))

    results = {}

    # --- Variant 1: AccumulatedOP (normalized + raw) ---
    print("\n" + "="*60, flush=True)
    print("AccumulatedOP (count-normalized + raw)", flush=True)
    print("="*60, flush=True)
    aop = AccumulatedOP(D_OUT, n_classes=N_CLASSES)
    aa_n, fgt_n, aa_r, fgt_r = run_aop_both(
        aop, "norm", "raw", tasks_train, tasks_test, perms, P)
    results['aop_norm'] = (aa_n, fgt_n)
    results['aop_raw']  = (aa_r, fgt_r)

    # --- Variant 2: EMA sweep ---
    for alpha in EMA_ALPHAS:
        print("\n" + "="*60, flush=True)
        print(f"EMA substrate alpha={alpha}", flush=True)
        print("="*60, flush=True)
        ema = EMASubstrate(D_OUT, n_classes=N_CLASSES, alpha=alpha)
        aa, fgt = run_variant(f"ema_{alpha}", ema, tasks_train, tasks_test, perms, P)
        results[f'ema_{alpha}'] = (aa, fgt)

    # --- Summary ---
    elapsed = time.time() - t0
    print("\n" + "="*72, flush=True)
    print("STEP 104 FINAL SUMMARY -- Accumulated Outer Product", flush=True)
    print(f"Baselines: fold=56.7%, 1-NN=86.8%, top-k(5)=91.8%", flush=True)
    print(f"Kill: AA <= 30%", flush=True)
    print("="*72, flush=True)
    print(f"  {'variant':<20} {'AA':>8} {'fgt':>8}  verdict", flush=True)
    print("-"*60, flush=True)

    for name, (aa, fgt) in results.items():
        if aa <= 0.30:
            verdict = "DISPROVED"
        elif aa > 0.868:
            verdict = "PASSES (> 1-NN!)"
        elif aa > 0.567:
            verdict = "PASSES (> fold)"
        else:
            verdict = "WEAK (< fold)"
        print(f"  {name:<20} {aa*100:>7.1f}% {fgt*100:>7.1f}pp  {verdict}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
