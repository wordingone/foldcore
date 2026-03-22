#!/usr/bin/env python3
"""
Step 108 -- Ablation: which parts of process() are load-bearing?
Spec.

Full system: 91.9% AA (Step 106/99). Three ablations on P-MNIST 10-task:

A. No competitive learning (lr=0): spawn only, no winner update.
B. No spawning (fixed codebook): pre-seed N={100,500,1000} from task 0, never add.
C. No top-k (1-NN readout): argmax(sims) instead of class top-k sum.

Prior predictions: A≈full (Step 101 confirmed lr=0 = lr=0.001 on P-MNIST),
C drops ~5pp (Step 99: k=1 → 86.8%), spawning most load-bearing.
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

K             = 5
SPAWN_THRESH  = 0.7
LR            = 0.001
SEED_SIZES    = [100, 500, 1000]


# ─── Unified process variants ─────────────────────────────────────────────────

def process_full(V, labels, r, label, k=K, spawn_thresh=SPAWN_THRESH, lr=LR):
    """Full system: competitive update + spawn + top-k."""
    r = F.normalize(r, dim=0)
    if V.shape[0] == 0 or (V @ r).max().item() < spawn_thresh:
        V      = torch.cat([V, r.unsqueeze(0)])
        labels = torch.cat([labels, torch.tensor([label], device=DEVICE)])
        return V, labels
    sims   = V @ r
    winner = int(sims.argmax().item())
    V[winner] = F.normalize(V[winner] + lr * (r - V[winner]), dim=0)
    return V, labels


def process_A(V, labels, r, label, spawn_thresh=SPAWN_THRESH):
    """Ablation A: no competitive learning (lr=0). Spawn only."""
    r = F.normalize(r, dim=0)
    if V.shape[0] == 0 or (V @ r).max().item() < spawn_thresh:
        V      = torch.cat([V, r.unsqueeze(0)])
        labels = torch.cat([labels, torch.tensor([label], device=DEVICE)])
    return V, labels


def process_B(V, labels, r, label, lr=LR):
    """Ablation B: no spawning. Update winner only, codebook size frozen."""
    r = F.normalize(r, dim=0)
    if V.shape[0] == 0:
        return V, labels  # empty codebook — shouldn't happen after seeding
    sims   = V @ r
    winner = int(sims.argmax().item())
    V[winner] = F.normalize(V[winner] + lr * (r - V[winner]), dim=0)
    return V, labels


def eval_batch_topk(V, labels, R, k=K):
    """Standard top-k batched eval (condition 1, frozen)."""
    R    = F.normalize(R, dim=1)
    sims = R @ V.T                            # (n, N)
    n    = R.shape[0]
    n_cls = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def eval_batch_1nn(V, labels, R):
    """Ablation C: 1-NN readout."""
    R    = F.normalize(R, dim=1)
    sims = R @ V.T                            # (n, N)
    return labels[sims.argmax(dim=1)].cpu()


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
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


def compute_aa(mat, n_tasks):
    vals = [mat[t][n_tasks - 1] for t in range(n_tasks) if mat[t][n_tasks - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat, n_tasks):
    vals = []
    for t in range(n_tasks - 1):
        if mat[t][t] is not None and mat[t][n_tasks - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n_tasks - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Generic run ─────────────────────────────────────────────────────────────

def run_experiment(name, train_fn, eval_fn, tasks_train, tasks_test, perms, P,
                   V_init=None, L_init=None):
    n    = N_TASKS
    mat  = [[None] * n for _ in range(n)]
    V      = V_init.clone() if V_init is not None else torch.empty(0, D_OUT, device=DEVICE)
    labels = L_init.clone() if L_init is not None else torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed(X_tr, perms[task_id], P)
        for i in range(len(y_tr)):
            V, labels = train_fn(V, labels, R_tr[i], int(y_tr[i]))
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = tasks_test[eval_task]
            R_te = embed(X_te, perms[eval_task], P)
            preds = eval_fn(V, labels, R_te)
            acc   = (preds == torch.from_numpy(y_te)).float().mean().item()
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_now = sum(mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={V.shape[0]} aa={aa_now*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 108 -- Ablation, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, DEVICE={DEVICE}", flush=True)
    print(f"Full system baseline: 91.9% AA, 0.0pp fgt", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    P     = make_projection()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train, tasks_test = [], []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))
        tasks_test.append((X_te_all, y_te_all))

    results = {}

    # --- Ablation A: lr=0 (spawn only, no competitive update) ---
    print(f"\n{'='*60}", flush=True)
    print(f"Ablation A: No competitive learning (lr=0, spawn only)", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_experiment(
        "A", process_A, eval_batch_topk, tasks_train, tasks_test, perms, P)
    results['A_no_lr'] = (aa, fgt)

    # --- Ablation C: 1-NN readout ---
    print(f"\n{'='*60}", flush=True)
    print(f"Ablation C: No top-k (1-NN readout)", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_experiment(
        "C", process_full, eval_batch_1nn, tasks_train, tasks_test, perms, P)
    results['C_1nn'] = (aa, fgt)

    # --- Ablation B: no spawning, fixed seeded codebook ---
    # Build seed from task 0 embeddings (stratified, first seed_n per class)
    X_t0, y_t0 = tasks_train[0]
    R_t0 = embed(X_t0, perms[0], P)

    for seed_n in SEED_SIZES:
        print(f"\n{'='*60}", flush=True)
        print(f"Ablation B: No spawning, seed_n={seed_n} (from task 0)", flush=True)
        print(f"{'='*60}", flush=True)
        # Build initial codebook: seed_n/N_CLASSES per class from task 0
        per_cls = seed_n // N_CLASSES
        seed_idx = []
        for c in range(N_CLASSES):
            mask = (y_t0 == c)
            idxs = np.where(mask)[0][:per_cls]
            seed_idx.extend(idxs.tolist())
        V_seed = R_t0[seed_idx].clone()
        L_seed = torch.tensor([y_t0[i] for i in seed_idx], dtype=torch.long, device=DEVICE)

        aa, fgt = run_experiment(
            f"B_{seed_n}", process_B, eval_batch_topk,
            tasks_train, tasks_test, perms, P,
            V_init=V_seed, L_init=L_seed)
        results[f'B_seed{seed_n}'] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 108 FINAL SUMMARY -- Ablation", flush=True)
    print(f"Full system: AA=91.9%, fgt=0.0pp", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  {'variant':<20} {'AA':>8} {'fgt':>8}  vs full", flush=True)
    print(f"  {'-'*55}", flush=True)
    for name, (aa, fgt) in results.items():
        delta = aa - 0.919
        print(f"  {name:<20} {aa*100:>7.1f}% {fgt*100:>7.1f}pp  {delta*100:+.1f}pp", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
