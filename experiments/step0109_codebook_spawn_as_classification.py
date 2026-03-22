#!/usr/bin/env python3
"""
Step 109 -- Spawn-as-Classification. P-MNIST 10-task.
Spec.

Always-spawn (no threshold) vs threshold-spawn (sp=0.7).
Kill: always-spawn AA < threshold-spawn AA - 2pp.
If ~equal: spawn threshold was just compression, not discriminative.

Codebook grows to 60k (all training samples). Batched eval over 60k OK on GPU.
"""

import math
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

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


def eval_topk_batch(V, labels, R_te):
    R_te    = F.normalize(R_te, dim=1)
    sims    = R_te @ V.T                          # (n, N)
    n       = R_te.shape[0]
    n_cls   = int(labels.max().item()) + 1
    scores  = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cs   = sims[:, mask]
        keff = min(K, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(seed=12345):
    rng = np.random.RandomState(seed)
    P   = rng.randn(D_OUT, 784).astype(np.float32) / math.sqrt(784)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed(X, perm, P):
    X_t = torch.from_numpy(X[:, perm]).to(DEVICE)
    return F.normalize(X_t @ P.T, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


def compute_aa(mat, n):
    vals = [mat[t][n - 1] for t in range(n) if mat[t][n - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat, n):
    vals = []
    for t in range(n - 1):
        if mat[t][t] is not None and mat[t][n - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n - 1]))
    return sum(vals) / len(vals) if vals else 0.0


def run(name, use_threshold, tasks_train, tasks_test, perms, P):
    n      = N_TASKS
    mat    = [[None] * n for _ in range(n)]
    V      = torch.empty(0, D_OUT, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        t0 = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed(X_tr, perms[task_id], P)
        y_t  = torch.from_numpy(y_tr).long().to(DEVICE)

        if use_threshold:
            # Threshold spawn: only add if max_sim < 0.7
            for i in range(len(y_tr)):
                r = F.normalize(R_tr[i], dim=0)
                if V.shape[0] == 0 or (V @ r).max().item() < SPAWN_THRESH:
                    V      = torch.cat([V, r.unsqueeze(0)])
                    labels = torch.cat([labels, y_t[i:i+1]])
        else:
            # Always spawn: append all training samples
            R_norm = F.normalize(R_tr, dim=1)
            V      = torch.cat([V, R_norm])
            labels = torch.cat([labels, y_t])

        train_s = time.time() - t0

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            X_te, y_te = tasks_test[eval_task]
            R_te  = embed(X_te, perms[eval_task], P)
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == torch.from_numpy(y_te)).float().mean().item()
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_now = sum(mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={V.shape[0]} aa={aa_now*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n), V.shape[0]


def main():
    t0 = time.time()
    print(f"Step 109 -- Spawn-as-Classification, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline (threshold sp=0.7): ~91.9% AA", flush=True)

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

    print(f"\n{'='*60}", flush=True)
    print(f"Threshold spawn (sp={SPAWN_THRESH})", flush=True)
    print(f"{'='*60}", flush=True)
    aa_th, fgt_th, cb_th = run("threshold", True, tasks_train, tasks_test, perms, P)

    print(f"\n{'='*60}", flush=True)
    print(f"Always spawn (no threshold)", flush=True)
    print(f"{'='*60}", flush=True)
    aa_al, fgt_al, cb_al = run("always", False, tasks_train, tasks_test, perms, P)

    elapsed = time.time() - t0
    delta   = aa_al - aa_th

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 109 FINAL SUMMARY -- Spawn-as-Classification", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"  Threshold spawn: AA={aa_th*100:.1f}%  fgt={fgt_th*100:.1f}pp  cb={cb_th}", flush=True)
    print(f"  Always spawn:    AA={aa_al*100:.1f}%  fgt={fgt_al*100:.1f}pp  cb={cb_al}", flush=True)
    print(f"  Delta: {delta*100:+.1f}pp", flush=True)

    if delta < -0.02:
        verdict = f"DISPROVED -- always-spawn degrades by {abs(delta)*100:.1f}pp > 2pp"
    elif abs(delta) <= 0.02:
        verdict = "PASSES -- threshold was just compression, not discriminative"
    else:
        verdict = f"PASSES -- always-spawn HELPS (+{delta*100:.1f}pp)"

    print(f"\n  VERDICT: {verdict}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
