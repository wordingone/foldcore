#!/usr/bin/env python3
"""
Step 117 -- Adaptive k P-MNIST raw pixels. Spec.
Fixed k={1,5,10,20} + density-adaptive k. Always-spawn.
Kill: adaptive k <= best fixed k.
"""

import random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N_TRAIN_TASK  = 6000; N_CLASSES = 10; TRAIN_PER_CLS = 600
SEED = 42; DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
FIXED_K_VALS  = [1, 5, 10, 20]
DENSITY_THRESH = 0.5
K_MIN, K_MAX   = 1, 20


def eval_topk_batch(V, labels, R_te, k):
    sims   = R_te @ V.T
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(R_te.shape[0], n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0: continue
        cs = sims[:, mask]; keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def eval_adaptive_batch(V, labels, R_te):
    """Adaptive k per test sample based on local density."""
    sims   = R_te @ V.T                       # (n, N)
    n      = R_te.shape[0]
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    k_vals = []
    for i in range(n):
        density = int((sims[i] > DENSITY_THRESH).sum().item())
        k = max(K_MIN, min(K_MAX, max(1, int(density * 0.1))))
        k_vals.append(k)
        for c in range(n_cls):
            mask = (labels == c)
            if mask.sum() == 0: continue
            cs = sims[i, mask]; keff = min(k, cs.shape[0])
            scores[i, c] = cs.topk(keff).values.sum()
    return scores.argmax(dim=1).cpu(), k_vals


def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float32)/255.0
    Xte = te.data.numpy().reshape(-1, 784).astype(np.float32)/255.0
    return Xtr, tr.targets.numpy(), Xte, te.targets.numpy()

def make_perm(seed):
    p = list(range(784)); random.Random(seed).shuffle(p); return p

def embed(X, perm):
    return F.normalize(torch.from_numpy(X[:, perm]).to(DEVICE), dim=1)

def stratified(X, y, n, seed):
    rng = np.random.RandomState(seed); idx = []
    for c in range(N_CLASSES):
        idx.extend(rng.choice(np.where(y==c)[0], n, replace=False).tolist())
    rng.shuffle(idx); return X[idx], y[idx]

def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals)/len(vals) if vals else 0.0
def compute_fgt(mat, n):
    vals = [max(0., mat[t][t]-mat[t][n-1]) for t in range(n-1)
            if mat[t][t] is not None and mat[t][n-1] is not None]
    return sum(vals)/len(vals) if vals else 0.0


def run(k_val, tasks_train, perms, te_embeds, y_te_t, adaptive=False):
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, 784, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    k_dist = {}

    for tid in range(n):
        X_tr, y_tr = tasks_train[tid]
        R_tr = embed(X_tr, perms[tid])
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, torch.from_numpy(y_tr).long().to(DEVICE)])

        for et in range(tid+1):
            R_te = te_embeds[et]
            if adaptive:
                preds, kvs = eval_adaptive_batch(V, labels, R_te)
                for kv in kvs: k_dist[kv] = k_dist.get(kv, 0) + 1
            else:
                preds = eval_topk_batch(V, labels, R_te, k_val)
            acc = (preds == y_te_t).float().mean().item()
            mat[et][tid] = acc

        aa = sum(mat[t][tid] for t in range(tid+1))/(tid+1)
        print(f"    T{tid}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n), k_dist


def main():
    t0 = time.time()
    print(f"Step 117 -- Adaptive k, P-MNIST raw pixels", flush=True)
    print(f"N_TASKS={N_TASKS}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline: k=5 → 95.4% (Step 110)", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr, ytr, Xte, yte = load_mnist()
    perms = [make_perm(t*100) for t in range(N_TASKS)]
    tasks_train = [stratified(Xtr, ytr, TRAIN_PER_CLS, t*7) for t in range(N_TASKS)]
    te_embeds   = [embed(Xte, perms[t]) for t in range(N_TASKS)]
    y_te_t      = torch.from_numpy(yte)

    results = {}
    for k in FIXED_K_VALS:
        print(f"\n{'='*50}\nFixed k={k}\n{'='*50}", flush=True)
        aa, fgt, _ = run(k, tasks_train, perms, te_embeds, y_te_t)
        results[f'k{k}'] = (aa, fgt)

    print(f"\n{'='*50}\nAdaptive k (density threshold={DENSITY_THRESH})\n{'='*50}", flush=True)
    aa, fgt, kdist = run(None, tasks_train, perms, te_embeds, y_te_t, adaptive=True)
    results['adaptive'] = (aa, fgt)

    elapsed = time.time() - t0
    best_fixed = max(aa for k,(aa,_) in results.items() if k.startswith('k'))

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 117 FINAL -- Adaptive k, P-MNIST", flush=True)
    print(f"{'='*72}", flush=True)
    for name, (aa, fgt) in results.items():
        tag = ''
        if name == 'adaptive':
            delta = aa - best_fixed
            tag = f'  delta_vs_best_fixed={delta*100:+.1f}pp'
            tag += ' [PASSES]' if aa > best_fixed else ' [DISPROVED]'
        print(f"  {name:<12} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp{tag}", flush=True)
    if kdist:
        total = sum(kdist.values())
        kd_sorted = sorted(kdist.items())
        print(f"\n  k distribution (adaptive, {total} decisions):", flush=True)
        for kv, cnt in kd_sorted:
            print(f"    k={kv}: {cnt/total*100:.1f}%", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
