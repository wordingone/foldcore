#!/usr/bin/env python3
"""
Step 116 -- Augmentation-as-Learning KNN, CIFAR-100 raw pixels.
Spec.

Store augmented orbit instead of single raw sample.
n_augs={2,4,8} → 3x/5x/9x storage. Batched train, batched eval.

Kill: augmented <= raw at all n_augs.
Proves: augmented > raw + 1pp.

sys.argv[1] = N_TASKS (default 2)
"""

import sys, time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

N_TASKS      = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CLASSES_TASK = 10
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
K            = 5
D_IN         = 3072
N_AUGS_VALS  = [2, 4, 8]


def make_augmentor():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4),
    ])


def augment_batch(X_flat, n_augs, aug):
    """X_flat: (N, 3072) in HWC order → augment, return (N*(1+n_augs), 3072)."""
    N = X_flat.shape[0]
    # CIFAR data: stored as HWC, reshape to (N, 32, 32, 3), permute to (N, 3, 32, 32)
    imgs = X_flat.view(N, 32, 32, 3).permute(0, 3, 1, 2)  # (N, C, H, W)
    all_versions = [X_flat]                                  # original stays flat
    for _ in range(n_augs):
        aug_list = []
        for i in range(N):
            a = aug(imgs[i])                                 # (C, H, W)
            aug_list.append(a.flatten())
        aug_batch = torch.stack(aug_list)                    # (N, 3072)
        all_versions.append(aug_batch)
    return torch.cat(all_versions, dim=0)                    # (N*(1+n_augs), 3072)


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


def run_knn(splits, n_augs=0):
    """n_augs=0: raw k-NN. n_augs>0: augmented storage."""
    aug = make_augmentor() if n_augs > 0 else None
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    V      = torch.empty(0, D_IN, device=DEVICE)
    labels = torch.empty(0, dtype=torch.long, device=DEVICE)

    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        t_train = time.time()

        if n_augs > 0:
            X_aug = augment_batch(X_tr, n_augs, aug)        # (N*(1+n_augs), 3072)
            R_tr  = F.normalize(X_aug, dim=1)
            # Labels: repeat each label (1+n_augs) times
            N = X_tr.shape[0]
            y_rep = y_tr.repeat_interleave(1)               # original
            for _ in range(n_augs):
                y_rep = torch.cat([y_rep, y_tr])
        else:
            R_tr  = F.normalize(X_tr, dim=1)
            y_rep = y_tr

        V      = torch.cat([V, R_tr])
        labels = torch.cat([labels, y_rep.to(DEVICE)])
        train_s = time.time() - t_train

        t_eval = time.time()
        for et in range(task_id + 1):
            _, _, X_te, y_te = splits[et]
            R_te  = F.normalize(X_te, dim=1)
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
    print(f"Step 116 -- Augmented k-NN, CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baseline: raw k-NN ~32.6%. Kill: aug <= raw. Proves: aug > raw+1pp", flush=True)

    torch.manual_seed(SEED); np.random.seed(SEED)
    print("Loading CIFAR-100...", flush=True)
    X_tr, y_tr, X_te, y_te = load_cifar100_raw()
    splits = make_splits(X_tr, y_tr, X_te, y_te)

    results = {}

    print(f"\n{'='*60}\nBaseline: Raw k-NN\n{'='*60}", flush=True)
    aa, fgt = run_knn(splits, n_augs=0)
    results['raw'] = (aa, fgt)
    raw_aa = aa

    for n_augs in N_AUGS_VALS:
        print(f"\n{'='*60}\nAugmented k-NN (n_augs={n_augs}, {1+n_augs}x storage)\n{'='*60}", flush=True)
        aa, fgt = run_knn(splits, n_augs=n_augs)
        results[f'aug_{n_augs}'] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 116 FINAL -- Augmented k-NN, CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    for name, (aa, fgt) in results.items():
        tag = ''
        if name.startswith('aug_'):
            delta = aa - raw_aa
            n_a   = int(name.split('_')[1])
            if aa > raw_aa + 0.01:
                tag = f'  delta={delta*100:+.1f}pp [PROVES]'
            elif aa > raw_aa:
                tag = f'  delta={delta*100:+.1f}pp [PASSES]'
            else:
                tag = f'  delta={delta*100:+.1f}pp [DISPROVED]'
        print(f"  {name:<12} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp{tag}", flush=True)

    best = max(((k,aa) for k,(aa,_) in results.items() if k.startswith('aug_')), key=lambda x:x[1], default=(None,0))
    overall = 'PROVES' if best[1] > raw_aa+0.01 else ('PASSES' if best[1] > raw_aa else 'DISPROVED')
    print(f"\n  OVERALL: {overall} (best: {best[0]} → {best[1]*100:.1f}%)", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
