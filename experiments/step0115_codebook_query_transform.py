#!/usr/bin/env python3
"""
Step 115 -- Query Transformation KNN, CIFAR-100 raw pixels.
Spec.

Store all samples raw (never modified). Learn per-class query biases.
Zero forgetting structural: biases are per-class, independent indices.

Kill: query-transform <= raw k-NN (~32.6%).

sys.argv[1] = N_TASKS (default 2)
"""

import sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS      = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CLASSES_TASK = 10
N_CLASSES    = 100
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
K            = 5
D_IN         = 3072
QUERY_LR_VALS = [0.001, 0.01, 0.1]


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


def eval_query_transform_batch(V, labels, R_te, biases, n_cls, k=K):
    """Batched query-transform eval. For each test sample, use class-shifted queries."""
    n      = R_te.shape[0]
    scores = torch.zeros(n, n_cls, device=DEVICE)
    B_cls  = biases[:n_cls]                         # (n_cls, d)
    # For each class c, query = normalize(R_te + B[c]).
    # Full approach: iterate over classes (n_cls=100, manageable)
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() == 0: continue
        Q_c = F.normalize(R_te + B_cls[c].unsqueeze(0), dim=1)  # (n, d)
        sims_c = (V[mask] @ Q_c.T).T                            # (n, m_c)
        keff = min(k, sims_c.shape[1])
        scores[:, c] = sims_c.topk(keff, dim=1).values.sum(dim=1)
    return scores.argmax(dim=1).cpu()


def run_baseline_raw(splits):
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, D_IN, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        R = F.normalize(X_tr, dim=1)
        V = torch.cat([V, R]); labels = torch.cat([labels, y_tr.to(DEVICE)])
        for et in range(task_id+1):
            _, _, Xte, yte = splits[et]
            acc = (eval_topk_batch(V, labels, F.normalize(Xte, dim=1)) == yte).float().mean().item()
            mat[et][task_id] = acc
        aa = sum(mat[t][task_id] for t in range(task_id+1))/(task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)
    return compute_aa(mat, n), compute_fgt(mat, n)


def run_query_transform(splits, query_lr):
    n      = N_TASKS
    mat    = [[None]*n for _ in range(n)]
    V      = torch.empty(0, D_IN, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    biases = torch.zeros(N_CLASSES, D_IN, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        t_train = time.time()
        R_tr = F.normalize(X_tr, dim=1)
        # Train: update biases per class, store raw
        for i in range(len(y_tr)):
            label = int(y_tr[i])
            r = R_tr[i]
            V      = torch.cat([V, r.unsqueeze(0)])
            labels = torch.cat([labels, torch.tensor([label], device=DEVICE)])
            # Update bias: shift toward class center
            c_mask = (labels == label)
            if c_mask.sum() > 1:
                class_center = V[c_mask].mean(0)
                biases[label] = biases[label] + query_lr * (class_center - r)
        train_s = time.time() - t_train

        n_cls = int(labels.max().item()) + 1
        t_eval = time.time()
        for et in range(task_id + 1):
            _, _, Xte, yte = splits[et]
            R_te = F.normalize(Xte, dim=1)
            preds = eval_query_transform_batch(V, labels, R_te, biases, n_cls)
            acc   = (preds == yte).float().mean().item()
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
    print(f"Step 115 -- Query Transform k-NN, CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline: raw k-NN ~32.6% (Step 113/114)", flush=True)
    print(f"Kill: query-transform <= raw k-NN", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    print("Loading CIFAR-100...", flush=True)
    X_tr, y_tr, X_te, y_te = load_cifar100_raw()
    splits = make_splits(X_tr, y_tr, X_te, y_te)

    results = {}

    print(f"\n{'='*60}", flush=True)
    print("Baseline: Raw k-NN", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_baseline_raw(splits)
    results['raw'] = (aa, fgt)
    raw_aa = aa

    for qlr in QUERY_LR_VALS:
        print(f"\n{'='*60}", flush=True)
        print(f"Query Transform lr={qlr}", flush=True)
        print(f"{'='*60}", flush=True)
        aa, fgt = run_query_transform(splits, qlr)
        results[f'qt_lr{qlr}'] = (aa, fgt)

    elapsed = time.time() - t0
    print(f"\n{'='*72}", flush=True)
    print(f"STEP 115 FINAL -- Query Transform k-NN, CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    for name, (aa, fgt) in results.items():
        tag = ''
        if name.startswith('qt_'):
            delta = aa - raw_aa
            verdict = 'PASSES' if aa > raw_aa else 'DISPROVED'
            tag = f'  delta={delta*100:+.1f}pp [{verdict}]'
        print(f"  {name:<20} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp{tag}", flush=True)
    best_qt = max(((k,aa) for k,(aa,_) in results.items() if k.startswith('qt_')), key=lambda x:x[1], default=(None,0))
    overall = 'PASSES' if best_qt[1] > raw_aa else 'DISPROVED'
    print(f"\n  OVERALL: {overall} (best: {best_qt[0]} → {best_qt[1]*100:.1f}%)", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
