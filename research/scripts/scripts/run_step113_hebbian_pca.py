#!/usr/bin/env python3
"""
Step 113 -- Hebbian PCA + k-NN on CIFAR-100 raw pixels.
[researcher] mail 1270.

Four conditions:
1. Raw pixel k-NN (d=3072) — floor
2. Random projection k-NN (d_proj=128) — control
3. Hebbian PCA k-NN (d_proj sweep: 32,64,128,256) — candidate
4. Frozen ResNet-18 k-NN — ceiling

Kill: Hebbian PCA <= random projection at same d_proj.
Proves if: Hebbian PCA > random AND approaches ResNet.

sys.argv[1] = N_TASKS (default 2 for Tier 1)
"""

import random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CLASSES_TASK  = 10
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
K             = 5
D_IN          = 3072            # 32×32×3 raw pixels
D_PROJ_VALS   = [32, 64, 128, 256]
RESNET_CACHE  = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'


# ─── Batched top-k eval (shared) ──────────────────────────────────────────────

def eval_topk_batch(V, labels, R_te, k=K):
    """V: (N, d) normed; R_te: (n, d) normed; labels: (N,) long."""
    sims   = R_te @ V.T                       # (n, N)
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


# ─── Condition 1: Raw pixel k-NN ──────────────────────────────────────────────

def run_raw(splits, y_te_all):
    """Store all raw L2-normalized pixels, batched eval."""
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, D_IN, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        R_tr = F.normalize(X_tr, dim=1)
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_tr.to(DEVICE)])

        for eval_task in range(task_id + 1):
            _, _, X_te, y_te = splits[eval_task]
            R_te  = F.normalize(X_te, dim=1)
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te).float().mean().item()
            mat[eval_task][task_id] = acc

        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Condition 2: Random projection k-NN ─────────────────────────────────────

def run_random_proj(splits, d_proj, y_te_all):
    rng = np.random.RandomState(SEED)
    P   = torch.from_numpy(
        rng.randn(d_proj, D_IN).astype(np.float32) / np.float32(np.sqrt(D_IN))
    ).to(DEVICE)

    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, d_proj, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        R_tr = F.normalize(F.normalize(X_tr, dim=1) @ P.T, dim=1)
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_tr.to(DEVICE)])

        for eval_task in range(task_id + 1):
            _, _, X_te, y_te = splits[eval_task]
            R_te  = F.normalize(F.normalize(X_te, dim=1) @ P.T, dim=1)
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te).float().mean().item()
            mat[eval_task][task_id] = acc

        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Condition 3: Hebbian PCA k-NN ───────────────────────────────────────────

def run_hebbian(splits, d_proj, hebbian_lr, y_te_all):
    torch.manual_seed(SEED)
    W = torch.randn(d_proj, D_IN, device=DEVICE) * 0.01
    W = F.normalize(W, dim=1)

    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, d_proj, device=DEVICE)   # projected codebook
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        t_train = time.time()

        # Per-sample Hebbian update + project + store
        R_tr_norm = F.normalize(X_tr, dim=1)   # (N, D_IN)
        for i in range(R_tr_norm.shape[0]):
            r = R_tr_norm[i]
            # Oja's rule (vectorized, no .item() sync)
            y = W @ r                                     # (d_proj,)
            outer_yr = torch.outer(y, r)                  # (d_proj, D_IN)
            outer_yy_W = torch.outer(y, y) @ W            # (d_proj, D_IN)
            W = W + hebbian_lr * (outer_yr - outer_yy_W)
            W = F.normalize(W, dim=1)
            # Project and store
            r_proj = F.normalize(W @ r, dim=0)
            V      = torch.cat([V, r_proj.unsqueeze(0)])
            labels = torch.cat([labels, y_tr[i:i+1].to(DEVICE)])

        train_s = time.time() - t_train

        # Reproject all stored vectors with final W for this task
        t_rep = time.time()
        all_raw = torch.cat([splits[t][0] for t in range(task_id+1)], dim=0)
        all_raw_norm = F.normalize(all_raw, dim=1)
        V = F.normalize(all_raw_norm @ W.T, dim=1)
        rep_s = time.time() - t_rep

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            _, _, X_te, y_te = splits[eval_task]
            R_te  = F.normalize(F.normalize(X_te, dim=1) @ W.T, dim=1)
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te).float().mean().item()
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: train={train_s:.1f}s rep={rep_s:.1f}s "
              f"eval={eval_s:.1f}s cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Condition 4: ResNet-18 k-NN (ceiling) ───────────────────────────────────

def run_resnet(resnet_splits, y_te_all_cls):
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, 512, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = resnet_splits[task_id]
        R_tr = F.normalize(X_tr, dim=1)
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_tr.to(DEVICE)])

        for eval_task in range(task_id + 1):
            _, _, X_te, y_te = resnet_splits[eval_task]
            R_te  = F.normalize(X_te, dim=1)
            preds = eval_topk_batch(V, labels, R_te)
            acc   = (preds == y_te).float().mean().item()
            mat[eval_task][task_id] = acc

        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_cifar100_raw():
    import torchvision
    tr = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=True,  download=True)
    te = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X_tr = torch.from_numpy(np.array(tr.data, dtype=np.float32).reshape(-1, 3072)).to(DEVICE) / 255.0
    y_tr = torch.tensor(tr.targets, dtype=torch.long)
    X_te = torch.from_numpy(np.array(te.data, dtype=np.float32).reshape(-1, 3072)).to(DEVICE) / 255.0
    y_te = torch.tensor(te.targets, dtype=torch.long)
    return X_tr, y_tr, X_te, y_te


def make_task_splits_raw(X_tr, y_tr, X_te, y_te):
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        mask_tr = torch.isin(y_tr, torch.arange(c0, c1))
        mask_te = torch.isin(y_te, torch.arange(c0, c1))
        splits.append((X_tr[mask_tr], y_tr[mask_tr], X_te[mask_te], y_te[mask_te]))
    return splits


def load_resnet_splits():
    data   = np.load(RESNET_CACHE)
    X_tr   = torch.tensor(data['X_train'], dtype=torch.float32, device=DEVICE)
    y_tr   = torch.tensor(data['y_train'], dtype=torch.long)
    X_te   = torch.tensor(data['X_test'],  dtype=torch.float32, device=DEVICE)
    y_te   = torch.tensor(data['y_test'],  dtype=torch.long)
    splits = []
    for t in range(N_TASKS):
        c0, c1   = t * CLASSES_TASK, (t + 1) * CLASSES_TASK
        mask_tr  = torch.isin(y_tr, torch.arange(c0, c1))
        mask_te  = torch.isin(y_te, torch.arange(c0, c1))
        splits.append((X_tr[mask_tr], y_tr[mask_tr], X_te[mask_te], y_te[mask_te]))
    return splits


def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals) / len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n - 1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n-1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 113 -- Hebbian PCA + k-NN, CIFAR-100 raw pixels", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    print("Loading CIFAR-100 raw pixels...", flush=True)
    X_tr, y_tr, X_te, y_te = load_cifar100_raw()
    splits = make_task_splits_raw(X_tr, y_tr, X_te, y_te)
    for t in range(N_TASKS):
        print(f"  Task {t}: tr={splits[t][0].shape[0]}, te={splits[t][2].shape[0]}", flush=True)

    print("Loading ResNet-18 features...", flush=True)
    resnet_splits = load_resnet_splits()

    results = {}

    # Condition 1: Raw pixel k-NN
    print(f"\n{'='*60}", flush=True)
    print(f"Condition 1: Raw pixel k-NN (d={D_IN})", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_raw(splits, y_te)
    results['raw_knn'] = (aa, fgt, D_IN)

    # Condition 4: ResNet-18 ceiling
    print(f"\n{'='*60}", flush=True)
    print(f"Condition 4: ResNet-18 k-NN (d=512, ceiling)", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_resnet(resnet_splits, y_te)
    results['resnet_ceiling'] = (aa, fgt, 512)

    # Condition 2: Random projection (d_proj=128 as control)
    print(f"\n{'='*60}", flush=True)
    print(f"Condition 2: Random projection k-NN (d_proj=128)", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_random_proj(splits, 128, y_te)
    results['random_128'] = (aa, fgt, 128)

    # Condition 3: Hebbian PCA sweep
    HEBBIAN_LR = 0.001
    for d_proj in D_PROJ_VALS:
        print(f"\n{'='*60}", flush=True)
        print(f"Condition 3: Hebbian PCA k-NN (d_proj={d_proj}, lr={HEBBIAN_LR})", flush=True)
        print(f"{'='*60}", flush=True)
        aa, fgt = run_hebbian(splits, d_proj, HEBBIAN_LR, y_te)
        results[f'hebbian_{d_proj}'] = (aa, fgt, d_proj)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 113 FINAL -- Hebbian PCA + k-NN, CIFAR-100", flush=True)
    print(f"Kill: Hebbian <= random_128 at same d_proj", flush=True)
    print(f"{'='*72}", flush=True)
    rand_aa = results['random_128'][0]
    for name, (aa, fgt, d) in results.items():
        tag = ''
        if name.startswith('hebbian_'):
            delta_vs_rand = aa - rand_aa
            tag = f'  vs_rand={delta_vs_rand*100:+.1f}pp'
            if aa > rand_aa:
                tag += ' [PASSES]'
            else:
                tag += ' [DISPROVED]'
        print(f"  {name:<20} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp  d={d}{tag}", flush=True)

    best_hebb = max((aa for k, (aa, _, _) in results.items() if k.startswith('hebbian_')), default=0)
    overall = "PASSES" if best_hebb > rand_aa else "DISPROVED"
    resnet_aa = results['resnet_ceiling'][0]
    print(f"\n  OVERALL: {overall}", flush=True)
    if best_hebb > rand_aa:
        gap_closed = (best_hebb - rand_aa) / max(resnet_aa - rand_aa, 0.001)
        print(f"  Gap closed vs ResNet: {gap_closed*100:.0f}%", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
