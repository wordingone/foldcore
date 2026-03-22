#!/usr/bin/env python3
"""
Step 114 -- Online Sparse Coding + k-NN, CIFAR-100 raw pixels.
Spec.

Hypothesis: sparse activation → different tasks update different atoms →
structural zero-forgetting. Learned dictionary beats random projection.

Conditions:
1. Raw pixel k-NN (d=3072, baseline ~32.6%)
2. Random proj k-NN (d=128, baseline ~31.6%)
3. Sparse coding k-NN sweep (n_atoms, sparsity, lr)

Kill: sparse coding <= random projection at best config.
Proves: sparse coding > raw pixel k-NN.

sys.argv[1] = N_TASKS (default 2 for Tier 1)
"""

import sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS      = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CLASSES_TASK = 10
SEED         = 42
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
K            = 5
D_IN         = 3072
RESNET_CACHE = '/mnt/c/Users/Admin/cifar100_resnet18_features.npz'

# Sweep: Tier 1 uses reduced set, Tier 3 uses full
N_ATOMS_VALS  = [128, 256, 512]
SPARSITY_VALS = [5, 10, 20]
LR_VALS       = [0.001, 0.01, 0.1]


# ─── Batched top-k eval ───────────────────────────────────────────────────────

def eval_topk_batch(V, labels, R_te, k=K):
    sims   = R_te @ V.T
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


# ─── Sparse coding ────────────────────────────────────────────────────────────

class SparseCodingKNN:
    def __init__(self, d_in, n_atoms=256, sparsity=10, lr=0.01, k=K):
        torch.manual_seed(SEED)
        self.D        = F.normalize(torch.randn(n_atoms, d_in, device=DEVICE), dim=1)
        self.V        = torch.empty(0, n_atoms, device=DEVICE)
        self.labels   = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.sparsity = sparsity
        self.lr       = lr
        self.k        = k
        self.n_atoms  = n_atoms

    def encode_batch(self, R):
        """Batch sparse encode: (N, d_in) → (N, n_atoms) sparse codes."""
        sims = R @ self.D.T                             # (N, n_atoms)
        n = R.shape[0]
        codes = torch.zeros(n, self.n_atoms, device=DEVICE)
        topk = sims.topk(self.sparsity, dim=1)
        codes.scatter_(1, topk.indices, topk.values)
        return codes

    def update_dict_batch(self, R, codes):
        """Fully vectorized dict update — no Python atom loop."""
        # active_mask: (N, n_atoms) bool — which atoms are active per sample
        active_mask = (codes != 0).float()              # (N, n_atoms)
        # all_dots: (n_atoms, N) — dot of each atom with each sample
        all_dots = self.D @ R.T                         # (n_atoms, N)
        active_T = active_mask.T                        # (n_atoms, N)
        count    = active_T.sum(dim=1, keepdim=True).clamp(min=1)  # (n_atoms, 1)
        # sum_R[i] = mean of R over samples activating atom i
        sum_R    = active_T @ R                         # (n_atoms, d_in)
        # sum_dots[i] = mean dot(D[i], R) over active samples
        sum_dots = (active_T * all_dots).sum(dim=1, keepdim=True)  # (n_atoms, 1)
        delta    = (sum_R - self.D * sum_dots) / count  # (n_atoms, d_in)
        self.D   = F.normalize(self.D + self.lr * delta, dim=1)

    def train_task(self, X_tr, y_tr):
        """Train on one task. Batch process for efficiency."""
        BATCH = 256
        n = X_tr.shape[0]
        R_norm = F.normalize(X_tr, dim=1)
        codes_list = []
        for i in range(0, n, BATCH):
            R_b = R_norm[i:i+BATCH]
            codes_b = self.encode_batch(R_b)
            self.update_dict_batch(R_b, codes_b)
            # Re-encode after update (codes may be stale but acceptable for storage)
            codes_b_final = self.encode_batch(R_b)
            codes_list.append(codes_b_final)
        codes_all = torch.cat(codes_list, dim=0)
        self.V = torch.cat([self.V, codes_all])
        self.labels = torch.cat([self.labels, y_tr.to(DEVICE)])

    def eval_task(self, X_te, y_te):
        R_norm = F.normalize(X_te, dim=1)
        codes_te = self.encode_batch(R_norm)
        # Normalize codes for cosine similarity
        V_norm = F.normalize(self.V, dim=1)
        c_norm = F.normalize(codes_te, dim=1)
        preds = eval_topk_batch(V_norm, self.labels, c_norm)
        return (preds == y_te).float().mean().item()


# ─── Condition 1: Raw pixel k-NN ──────────────────────────────────────────────

def run_raw(splits):
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
            R_te = F.normalize(X_te, dim=1)
            acc  = (eval_topk_batch(V, labels, R_te) == y_te).float().mean().item()
            mat[eval_task][task_id] = acc
        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)
    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Condition 2: Random projection ──────────────────────────────────────────

def run_random_proj(splits, d_proj=128):
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
            R_te = F.normalize(F.normalize(X_te, dim=1) @ P.T, dim=1)
            acc  = (eval_topk_batch(V, labels, R_te) == y_te).float().mean().item()
            mat[eval_task][task_id] = acc
        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: cb={V.shape[0]} aa={aa*100:.1f}%", flush=True)
    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Condition 3: Sparse coding ──────────────────────────────────────────────

def run_sparse(splits, n_atoms, sparsity, lr):
    sc  = SparseCodingKNN(D_IN, n_atoms=n_atoms, sparsity=sparsity, lr=lr)
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    for task_id in range(n):
        X_tr, y_tr, _, _ = splits[task_id]
        t_train = time.time()
        sc.train_task(X_tr, y_tr)
        train_s = time.time() - t_train
        t_eval = time.time()
        for eval_task in range(task_id + 1):
            _, _, X_te, y_te = splits[eval_task]
            acc = sc.eval_task(X_te, y_te)
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval
        aa = sum(mat[t][task_id] for t in range(task_id+1)) / (task_id+1)
        print(f"    T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={sc.V.shape[0]} aa={aa*100:.1f}%", flush=True)
    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Data ─────────────────────────────────────────────────────────────────────

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
    print(f"Step 114 -- Online Sparse Coding + k-NN, CIFAR-100", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}", flush=True)
    print(f"Baselines: raw=32.6%, random_128=31.6% (Step 113, 2-task)", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading CIFAR-100...", flush=True)
    X_tr, y_tr, X_te, y_te = load_cifar100_raw()
    splits = make_splits(X_tr, y_tr, X_te, y_te)

    results = {}

    print(f"\n{'='*60}", flush=True)
    print(f"Condition 1: Raw pixel k-NN (d={D_IN})", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_raw(splits)
    results['raw'] = (aa, fgt)
    raw_aa = aa

    print(f"\n{'='*60}", flush=True)
    print(f"Condition 2: Random proj k-NN (d=128)", flush=True)
    print(f"{'='*60}", flush=True)
    aa, fgt = run_random_proj(splits, 128)
    results['random_128'] = (aa, fgt)
    rand_aa = aa

    # Sweep sparse coding
    for n_atoms in N_ATOMS_VALS:
        for sparsity in SPARSITY_VALS:
            for lr in LR_VALS:
                name = f"sparse_a{n_atoms}_s{sparsity}_lr{lr}"
                print(f"\n{'='*60}", flush=True)
                print(f"Sparse: n_atoms={n_atoms} sparsity={sparsity} lr={lr}", flush=True)
                print(f"{'='*60}", flush=True)
                aa, fgt = run_sparse(splits, n_atoms, sparsity, lr)
                results[name] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 114 FINAL -- Online Sparse Coding + k-NN, CIFAR-100", flush=True)
    print(f"{'='*72}", flush=True)
    best_sparse = max(
        ((k, aa) for k, (aa, _) in results.items() if k.startswith('sparse_')),
        key=lambda x: x[1], default=(None, 0)
    )
    for name, (aa, fgt) in results.items():
        tag = ''
        if name.startswith('sparse_'):
            vs_rand = aa - rand_aa
            vs_raw  = aa - raw_aa
            tag = f'  vs_rand={vs_rand*100:+.1f}pp  vs_raw={vs_raw*100:+.1f}pp'
            if aa > rand_aa:
                tag += ' [PASSES]'
            else:
                tag += ' [DISPROVED]'
        print(f"  {name:<35} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp{tag}", flush=True)

    overall = 'PASSES' if best_sparse[1] > rand_aa else 'DISPROVED'
    proves  = 'PROVES' if best_sparse[1] > raw_aa else 'NOT PROVED'
    print(f"\n  OVERALL: {overall} (kill: sparse > random_128)", flush=True)
    print(f"  PROVES:  {proves} (best sparse > raw pixels)", flush=True)
    if best_sparse[0]:
        print(f"  Best: {best_sparse[0]} → {best_sparse[1]*100:.1f}%", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
