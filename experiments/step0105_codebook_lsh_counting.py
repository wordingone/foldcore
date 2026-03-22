#!/usr/bin/env python3
"""
Step 105 -- LSH Counting substrate. P-MNIST full 10-task.
Spec.

No codebook. No spawning. Fixed structure: R (n_hashes x d) + counts (n_hashes x C).
step()    : relu(R @ normalize(r)) -> accumulate into counts[:,label]
classify(): relu(R @ normalize(r)) * counts -> sum over hashes -> argmax

Sweep n_hashes: {256, 1024, 4096, 16384}.

Kill: AA <= 30% (no better than centroid).
Baselines: centroid=30.0%, fold=56.7%, 1-NN=86.8%, top-k(5)=91.8%.
"""

import math
import random
import sys
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

N_HASHES_VALS = [256, 1024, 4096, 16384]


# ─── LSH Substrate ────────────────────────────────────────────────────────────

class LSHSubstrate:
    def __init__(self, d, n_hashes=1024, n_classes=10):
        torch.manual_seed(0)  # fixed R across all instances
        self.R      = F.normalize(torch.randn(n_hashes, d, device=DEVICE), dim=1)
        self.counts = torch.zeros(n_hashes, n_classes, device=DEVICE)
        self.n_hashes = n_hashes

    def step_batch(self, R_embed, labels):
        """
        Batched training. R_embed: (n, d) normalized. labels: (n,) int tensor.
        """
        # (n, n_hashes)
        sims    = F.normalize(R_embed, dim=1) @ self.R.T
        weights = F.relu(sims)  # (n, n_hashes)
        # scatter_add into counts: for each sample, add weights to counts[:,label]
        for i in range(len(labels)):
            self.counts[:, labels[i]] += weights[i]

    def classify_batch(self, R_embed):
        """
        Batched classify. R_embed: (n, d). Returns (n,) CPU long tensor.
        """
        R_n     = F.normalize(R_embed, dim=1)          # (n, d)
        sims    = R_n @ self.R.T                        # (n, n_hashes)
        weights = F.relu(sims)                          # (n, n_hashes)
        scores  = weights @ self.counts                 # (n, n_classes)
        return scores.argmax(dim=1).cpu()


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


# ─── Run one config ───────────────────────────────────────────────────────────

def run_config(n_hashes, tasks_train, tasks_test, perms, P):
    n   = N_TASKS
    sub = LSHSubstrate(D_OUT, n_hashes=n_hashes, n_classes=N_CLASSES)
    acc_mat = [[None] * n for _ in range(n)]

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed(X_tr, perms[task_id], P)
        y_t  = torch.from_numpy(y_tr).long()
        sub.step_batch(R_tr, y_t)
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = tasks_test[eval_task]
            R_te   = embed(X_te, perms[eval_task], P)
            preds  = sub.classify_batch(R_te)
            y_te_t = torch.from_numpy(y_te)
            acc    = (preds == y_te_t).float().mean().item()
            acc_mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_now = sum(acc_mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s aa={aa_now*100:.1f}%",
              flush=True)

    aa  = compute_aa(acc_mat, n)
    fgt = compute_fgt(acc_mat, n)
    return aa, fgt


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 105 -- LSH Counting, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Baselines: centroid=30.0%, fold=56.7%, 1-NN=86.8%, top-k(5)=91.8%", flush=True)
    print(f"Kill: AA <= 30%", flush=True)
    print(f"n_hashes sweep: {N_HASHES_VALS}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    P     = make_projection()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train = []
    tasks_test  = []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))
        tasks_test.append((X_te_all, y_te_all))

    results = {}
    for n_hashes in N_HASHES_VALS:
        print(f"\n{'='*60}", flush=True)
        print(f"n_hashes={n_hashes}", flush=True)
        print(f"{'='*60}", flush=True)
        aa, fgt = run_config(n_hashes, tasks_train, tasks_test, perms, P)
        results[n_hashes] = (aa, fgt)

    elapsed = time.time() - t0
    print(f"\n{'='*72}", flush=True)
    print(f"STEP 105 FINAL SUMMARY -- LSH Counting", flush=True)
    print(f"Baselines: centroid=30.0%, fold=56.7%, 1-NN=86.8%, top-k(5)=91.8%", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'n_hashes':<12} {'AA':>8} {'fgt':>8}  verdict", flush=True)
    print(f"  {'-'*55}", flush=True)
    for n_hashes, (aa, fgt) in results.items():
        if aa <= 0.30:
            verdict = "DISPROVED (≤ centroid)"
        elif aa > 0.868:
            verdict = "PASSES (> 1-NN!)"
        elif aa > 0.918:
            verdict = "PASSES (> top-k!)"
        elif aa > 0.567:
            verdict = "PASSES (> fold)"
        else:
            verdict = "WEAK (< fold)"
        print(f"  {n_hashes:<12} {aa*100:>7.1f}% {fgt*100:>7.1f}pp  {verdict}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
