#!/usr/bin/env python3
"""
Step 120 -- S1 eval on P-MNIST + multi-pass CIFAR-100 ResNet.
[researcher] mail 1286.

Test A: P-MNIST raw pixels — standard vs S1 eval (does it boost 95.4%?)
Test B/C: CIFAR-100 ResNet — multi-pass S1 eval (1,2,3,5 passes), does it saturate?
"""

import random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = 10
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
K             = 5
# P-MNIST
N_CLASSES     = 10
TRAIN_PER_CLS = 600
# CIFAR-100
CLASSES_TASK  = 10
RESNET_CACHE  = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'


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


def compute_aa(mat, n):
    vals = [mat[t][n-1] for t in range(n) if mat[t][n-1] is not None]
    return sum(vals)/len(vals) if vals else 0.0

def compute_fgt(mat, n):
    vals = []
    for t in range(n-1):
        if mat[t][t] is not None and mat[t][n-1] is not None:
            vals.append(max(0.0, mat[t][t]-mat[t][n-1]))
    return sum(vals)/len(vals) if vals else 0.0


# ── P-MNIST ────────────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    Xtr = tr.data.numpy().reshape(-1, 784).astype(np.float32)/255.0
    Xte = te.data.numpy().reshape(-1, 784).astype(np.float32)/255.0
    return Xtr, tr.targets.numpy(), Xte, te.targets.numpy()

def make_perm(seed):
    p = list(range(784)); random.Random(seed).shuffle(p); return p

def embed_mnist(X, perm):
    return F.normalize(torch.from_numpy(X[:, perm]).to(DEVICE), dim=1)

def stratified(X, y, n, seed):
    rng = np.random.RandomState(seed); idx = []
    for c in range(N_CLASSES):
        idx.extend(rng.choice(np.where(y==c)[0], n, replace=False).tolist())
    rng.shuffle(idx); return X[idx], y[idx]


def run_pmnist(s1_eval=False):
    torch.manual_seed(SEED); np.random.seed(SEED)
    Xtr, ytr, Xte, yte = load_mnist()
    perms = [make_perm(t*100) for t in range(N_TASKS)]
    tasks_train = [stratified(Xtr, ytr, TRAIN_PER_CLS, t*7) for t in range(N_TASKS)]
    te_embeds   = [embed_mnist(Xte, perms[t]) for t in range(N_TASKS)]
    y_te_t      = torch.from_numpy(yte)

    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, 784, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    eval_spawns = 0

    for tid in range(n):
        X_tr, y_tr = tasks_train[tid]
        R_tr = embed_mnist(X_tr, perms[tid])
        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, torch.from_numpy(y_tr).long().to(DEVICE)])

        for et in range(tid+1):
            R_te = te_embeds[et]
            preds = eval_topk_batch(V, labels, R_te)
            acc = (preds == y_te_t).float().mean().item()
            mat[et][tid] = acc
            if s1_eval:
                V      = torch.cat([V, R_te])
                labels = torch.cat([labels, preds.to(DEVICE)])
                eval_spawns += R_te.shape[0]

        aa = sum(mat[t][tid] for t in range(tid+1))/(tid+1)
        print(f"    T{tid}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n), int(V.shape[0]), eval_spawns


# ── CIFAR-100 ResNet multi-pass ────────────────────────────────────────────

def load_resnet():
    data = np.load(RESNET_CACHE)
    X_tr = torch.from_numpy(data['X_train'].astype(np.float32)).to(DEVICE)
    y_tr = torch.from_numpy(data['y_train'].astype(np.int64))
    X_te = torch.from_numpy(data['X_test'].astype(np.float32)).to(DEVICE)
    y_te = torch.from_numpy(data['y_test'].astype(np.int64))
    return X_tr, y_tr, X_te, y_te

def make_cifar_splits(X_tr, y_tr, X_te, y_te):
    splits = []
    for t in range(N_TASKS):
        c0, c1 = t*CLASSES_TASK, (t+1)*CLASSES_TASK
        mtr = torch.isin(y_tr, torch.arange(c0, c1))
        mte = torch.isin(y_te, torch.arange(c0, c1))
        splits.append((X_tr[mtr], y_tr[mtr], X_te[mte], y_te[mte]))
    return splits

def eval_pass(V, labels, splits, n_tasks, k=K):
    """One eval pass: evaluate all tasks, optionally spawn. Returns acc_per_task, new V, labels."""
    accs = []
    for et in range(n_tasks):
        _, _, X_te, y_te = splits[et]
        R_te = F.normalize(X_te, dim=1)
        preds = eval_topk_batch(V, labels, R_te, k)
        acc = (preds == y_te).float().mean().item()
        accs.append(acc)
        # Spawn with predicted labels
        V      = torch.cat([V, R_te])
        labels = torch.cat([labels, preds.to(DEVICE)])
    return accs, V, labels

def run_cifar_multipass(n_passes=5):
    torch.manual_seed(SEED); np.random.seed(SEED)
    X_tr, y_tr, X_te, y_te = load_resnet()
    splits = make_cifar_splits(X_tr, y_tr, X_te, y_te)

    # Train: always-spawn all training data
    V      = torch.empty(0, X_tr.shape[1], device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)
    for t in range(N_TASKS):
        X_t, y_t, _, _ = splits[t]
        R_t = F.normalize(X_t, dim=1)
        V      = torch.cat([V, R_t])
        labels = torch.cat([labels, y_t.to(DEVICE)])
    print(f"  Training done: CB={V.shape[0]}", flush=True)

    # Multi-pass eval
    results = {}
    for p in range(1, n_passes+1):
        accs, V, labels = eval_pass(V, labels, splits, N_TASKS)
        aa = sum(accs)/len(accs)
        results[p] = (aa, int(V.shape[0]))
        print(f"  Pass {p}: AA={aa*100:.1f}%  CB={V.shape[0]}", flush=True)
    return results


def main():
    t0 = time.time()
    print(f"Step 120 -- S1 multi-pass, P-MNIST + CIFAR-100 ResNet", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)

    # ── Test A: P-MNIST ──────────────────────────────────────────────────
    print(f"\n{'='*60}\nTest A1: P-MNIST standard eval\n{'='*60}", flush=True)
    aa1, fgt1, cb1, _ = run_pmnist(s1_eval=False)

    print(f"\n{'='*60}\nTest A2: P-MNIST S1 eval\n{'='*60}", flush=True)
    aa2, fgt2, cb2, esp = run_pmnist(s1_eval=True)

    # ── Test B/C: CIFAR-100 multi-pass ───────────────────────────────────
    print(f"\n{'='*60}\nTest B/C: CIFAR-100 ResNet multi-pass S1 eval (5 passes)\n{'='*60}", flush=True)
    mp_results = run_cifar_multipass(n_passes=5)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 120 FINAL", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"\nTest A — P-MNIST (reference: 95.4% AA, 0.0pp fgt):", flush=True)
    print(f"  standard:  AA={aa1*100:.1f}%  fgt={fgt1*100:.1f}pp  CB={cb1}", flush=True)
    delta = aa2 - aa1
    print(f"  S1 eval:   AA={aa2*100:.1f}%  fgt={fgt2*100:.1f}pp  CB={cb2}  eval_spawns={esp}  delta={delta*100:+.2f}pp", flush=True)

    print(f"\nTest B/C — CIFAR-100 ResNet multi-pass (ref: pass1=39.7% from Step 119):", flush=True)
    prev_aa = None
    for p, (aa, cb) in mp_results.items():
        if prev_aa is not None:
            delta_pass = aa - prev_aa
            print(f"  Pass {p}: AA={aa*100:.1f}%  CB={cb}  delta_from_prev={delta_pass*100:+.2f}pp", flush=True)
        else:
            print(f"  Pass {p}: AA={aa*100:.1f}%  CB={cb}", flush=True)
        prev_aa = aa
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
