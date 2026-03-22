#!/usr/bin/env python3
"""
Step 110 -- Raw pixels, no feature extractor. P-MNIST 10-task.
[researcher] mail 1264. The honest test.

Same always-spawn + top-k system. Input: raw 784-dim normalized pixels.
No random projection, no ResNet, no feature extractor.

Report: k=1 and k=5 AA and forgetting. Compare to Step 109 (95.0% with projection).
"""

import random, sys, time, math
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
K_VALS        = [1, 5]


def eval_topk_batch(V, labels, R_te, k):
    sims   = R_te @ V.T                           # (n, N)
    n      = R_te.shape[0]
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed_raw(X_np, perm):
    """Permute pixels, normalize. No projection."""
    X_t = torch.from_numpy(X_np[:, perm]).to(DEVICE)
    return F.normalize(X_t, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals) / len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n - 1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n-1]))
    return sum(vals) / len(vals) if vals else 0.0


def main():
    t0 = time.time()
    print(f"Step 110 -- Raw pixels (D=784, no projection), P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, always-spawn, k={K_VALS}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline (Step 109 with proj): k=5→95.0%", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train, tasks_test = [], []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))
        tasks_test.append((X_te_all, y_te_all))

    # Pre-embed test sets
    test_embeds = [embed_raw(tasks_test[t][0], perms[t]) for t in range(N_TASKS)]
    y_te_t = torch.from_numpy(y_te_all)

    n     = N_TASKS
    mats  = {k: [[None]*n for _ in range(n)] for k in K_VALS}
    V      = torch.empty(0, 784, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr   = embed_raw(X_tr, perms[task_id])
        y_t    = torch.from_numpy(y_tr).long().to(DEVICE)
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_t])
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            R_te = test_embeds[eval_task]
            for k in K_VALS:
                preds = eval_topk_batch(V, labels, R_te, k)
                acc   = (preds == y_te_t).float().mean().item()
                mats[k][eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_k5 = sum(mats[5][t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        print(f"  T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={V.shape[0]} k5={aa_k5*100:.1f}%", flush=True)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 110 FINAL -- Raw pixels, always-spawn", flush=True)
    print(f"Compare: Step 109 (random proj D=384): k=1→86.2%, k=5→95.0%", flush=True)
    print(f"{'='*72}", flush=True)
    for k in K_VALS:
        aa  = compute_aa(mats[k], n)
        fgt = compute_fgt(mats[k], n)
        print(f"  k={k}: AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp", flush=True)

    print(f"\nFrozen frame audit:", flush=True)
    print(f"  - Feature extractor: NONE (raw pixels)", flush=True)
    print(f"  - Normalization: L2 (cosine = dot of L2-normalized)", flush=True)
    print(f"  - Distance metric: cosine", flush=True)
    print(f"  - k: {K_VALS}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
