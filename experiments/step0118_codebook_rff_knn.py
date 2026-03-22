#!/usr/bin/env python3
"""
Step 118 -- Random Fourier Features + k-NN, CIFAR-100 raw pixels.
Spec.

RFF transform: z = cos(W @ r + b), W ~ N(0, 1/sigma^2), b ~ U(0, 2pi)
Approximates Gaussian RBF kernel. Fixed (no learning, no forgetting).

Conditions:
1. Raw pixel k-NN (d=3072, baseline ~32.6%)
2. Random linear proj k-NN (d=128, baseline ~31.6%)
3. RFF k-NN sweep (d_feat, sigma)

Kill: RFF <= raw pixel at ALL configs -> DISPROVED.
Proves: RFF > raw pixel k-NN.

sys.argv[1] = N_TASKS (default 2 for Tier 1)
"""

import sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS      = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CLASSES_TASK = 10
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
K            = 5
D_IN         = 3072

D_FEAT_VALS  = [512, 1024, 2048, 4096]
SIGMA_VALS   = [0.1, 1.0, 10.0]


def eval_topk_batch(V, labels, R_te, k=K):
    sims   = R_te @ V.T
    n      = R_te.shape[0]
    n_cls  = int(labels.max().item()) + 1
    scores = torch.zeros(n, n_cls, device=DEVICE)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0: continue
        cs   = sims[:, mask]
        keff = min(k, cs.shape[1])
        scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def rff_transform(X_norm, W_rff, b_rff):
    """X_norm: (N, d_in) -> (N, d_feat) RFF features, L2-normalized."""
    Z = torch.cos(X_norm @ W_rff.T + b_rff.unsqueeze(0))  # (N, d_feat)
    return F.normalize(Z, dim=1)


def run_knn(splits, transform_fn=None, d_feat=None):
    """Run continual k-NN. transform_fn: (X_norm) -> features, or None for raw."""
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    d   = d_feat if d_feat is not None else D_IN
    V      = torch.empty(0, d, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        t_train = time.time()
        X_norm = F.normalize(X_tr, dim=1)
        R_tr = transform_fn(X_norm) if transform_fn else X_norm
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_tr.to(DEVICE)])
        train_s = time.time() - t_train

        t_eval = time.time()
        for et in range(task_id + 1):
            _, _, X_te, y_te = splits[et]
            X_te_norm = F.normalize(X_te, dim=1)
            R_te  = transform_fn(X_te_norm) if transform_fn else X_te_norm
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te).float().mean().item()
            mat[et][task_id] = acc
        eval_s = time.time() - t_eval

        aa = sum(mat[t][task_id] for t in range(task_id+1))/(task_id+1)
        print(f"    T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


def load_cifar100_raw():
    import torchvision
    tr = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=True,  download=True)
    te = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X_tr = torch.from_numpy(np.array(tr.data, dtype=np.float32).reshape(-1, 3072)).to(DEVICE) / 255.0
    y_tr = torch.tensor(tr.targets, dtype=torch.long)
    X_te = torch.from_numpy(np.array(te.data, dtype=np.float32).reshape(-1, 3072)).to(DEVICE) / 255.0
    y_te = torch.tensor(te.targets, dtype=torch.long)
    return X_tr, y_tr, X_te, y_te

def make_splits(X_tr, y_tr, X_te, y_te):
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t*CLASSES_TASK, (t+1)*CLASSES_TASK
        mtr = torch.isin(y_tr, torch.arange(c0, c1))
        mte = torch.isin(y_te, torch.arange(c0, c1))
        splits.append((X_tr[mtr], y_tr[mtr], X_te[mte], y_te[mte]))
    return splits

def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals)/len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n-1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t]-mat[t][n-1]))
    return sum(vals)/len(vals) if vals else 0.0


def main():
    t0 = time.time()
    print(f"Step 118 -- Random Fourier Features k-NN, CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baselines: raw=32.6%, random_128=31.6% (Steps 113/114)", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    print("Loading CIFAR-100...", flush=True)
    X_tr, y_tr, X_te, y_te = load_cifar100_raw()
    splits = make_splits(X_tr, y_tr, X_te, y_te)

    results = {}

    print(f"\n{'='*60}\nCondition 1: Raw pixel k-NN (d={D_IN})\n{'='*60}", flush=True)
    aa, fgt = run_knn(splits)
    results['raw'] = (aa, fgt)
    raw_aa = aa

    print(f"\n{'='*60}\nCondition 2: Random linear proj (d=128)\n{'='*60}", flush=True)
    rng = np.random.RandomState(SEED)
    P = torch.from_numpy(
        rng.randn(128, D_IN).astype(np.float32) / np.float32(np.sqrt(D_IN))
    ).to(DEVICE)
    def rand_proj(X_norm): return F.normalize(X_norm @ P.T, dim=1)
    aa, fgt = run_knn(splits, transform_fn=rand_proj, d_feat=128)
    results['random_128'] = (aa, fgt)

    # Sweep RFF
    for d_feat in D_FEAT_VALS:
        for sigma in SIGMA_VALS:
            name = f"rff_d{d_feat}_s{sigma}"
            print(f"\n{'='*60}\nRFF: d_feat={d_feat} sigma={sigma}\n{'='*60}", flush=True)
            torch.manual_seed(SEED)
            W_rff = torch.randn(d_feat, D_IN, device=DEVICE) / sigma
            b_rff = torch.rand(d_feat, device=DEVICE) * 2 * 3.14159265
            def make_transform(W, b):
                def fn(X_norm): return rff_transform(X_norm, W, b)
                return fn
            aa, fgt = run_knn(splits, transform_fn=make_transform(W_rff, b_rff), d_feat=d_feat)
            results[name] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 118 FINAL -- RFF k-NN, CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    best_rff = max(
        ((k, aa) for k, (aa, _) in results.items() if k.startswith('rff_')),
        key=lambda x: x[1], default=(None, 0)
    )
    for name, (aa, fgt) in results.items():
        tag = ''
        if name.startswith('rff_'):
            delta = aa - raw_aa
            if aa > raw_aa:
                tag = f'  delta={delta*100:+.1f}pp [PASSES]'
            else:
                tag = f'  delta={delta*100:+.1f}pp [DISPROVED]'
        print(f"  {name:<25} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp{tag}", flush=True)

    if best_rff[0]:
        delta_best = best_rff[1] - raw_aa
        overall = 'PROVES' if best_rff[1] > raw_aa else 'DISPROVED'
        print(f"\n  OVERALL: {overall}", flush=True)
        print(f"  Best RFF: {best_rff[0]} -> {best_rff[1]*100:.1f}% (delta={delta_best*100:+.1f}pp vs raw)", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
