#!/usr/bin/env python3
"""
Step 112 -- Retrieval-Driven Projection. P-MNIST 10-task, raw pixels.
Spec.

Learn a d×d projection W from retrieval patterns. Same-class neighbors:
suppress difference dimensions (W -= outer(diff,diff)/norm²).
Different-class neighbors: amplify difference dimensions (W += ...).

Baseline: k-NN raw pixels 95.4% (Step 110/111).
Kill: AA <= 95.4%.

CAUTION: _reproject_all() is O(N×d²). Running per-step makes training
O(N²×d²). Implementation: reproject lazily (once per task, not per sample)
to keep Tier-2 feasible. Also run per-step variant on 2-task Tier 1.

sys.argv[1] = N_TASKS (default 2 for Tier 1)
sys.argv[2] = reproject_mode: 'lazy' (per-task) or 'online' (per-step, 2-task only)
"""

import random, sys, time
import numpy as np
import torch
import torch.nn.functional as F

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 2
REPROJECT     = sys.argv[2] if len(sys.argv) > 2 else 'lazy'
N_TRAIN_TASK  = 6000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
K             = 5
PROJ_LR_VALS  = [0.0001, 0.001, 0.01]


# ─── Retrieval Projection class ───────────────────────────────────────────────

class RetrievalProjection:
    def __init__(self, d, k=K, proj_lr=0.001, reproject_mode='lazy'):
        self.W      = torch.eye(d, device=DEVICE)
        self.V_raw  = torch.empty(0, d, device=DEVICE)
        self.V_proj = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.k      = k
        self.proj_lr = proj_lr
        self.d      = d
        self.dirty  = False
        self.mode   = reproject_mode

    def _project_batch(self, X):
        """Project batch (N, d) with current W, L2-normalize."""
        return F.normalize(X @ self.W.T, dim=1)

    def _reproject_all(self):
        if self.dirty and self.V_raw.shape[0] > 0:
            self.V_proj = self._project_batch(self.V_raw)
            self.dirty  = False

    def store(self, r_raw, label):
        """Add a new vector to codebook (raw + projected)."""
        r_proj = F.normalize(self.W @ r_raw, dim=0)
        self.V_raw  = torch.cat([self.V_raw,  r_raw.unsqueeze(0)])
        self.V_proj = torch.cat([self.V_proj, r_proj.unsqueeze(0)])
        self.labels = torch.cat([self.labels, torch.tensor([label], device=DEVICE)])

    def predict_batch(self, R_raw):
        """Batched top-k prediction on raw inputs (projects first)."""
        if self.dirty:
            self._reproject_all()
        R_proj = self._project_batch(R_raw)
        sims   = R_proj @ self.V_proj.T       # (n, N)
        n      = R_proj.shape[0]
        n_cls  = int(self.labels.max().item()) + 1
        scores = torch.zeros(n, n_cls, device=DEVICE)
        for c in range(n_cls):
            mask = (self.labels == c)
            if mask.sum() == 0:
                continue
            cs   = sims[:, mask]
            keff = min(self.k, cs.shape[1])
            scores[:, c] = cs.topk(keff, dim=1).values.sum(dim=1)
        return scores.argmax(dim=1).cpu()

    def update_W(self, r_raw, label):
        """Update W using retrieval pattern around r_raw."""
        N = self.V_proj.shape[0]
        if N <= self.k * 2:
            return

        r_proj  = F.normalize(self.W @ r_raw, dim=0)
        sims    = self.V_proj @ r_proj
        top_idx = sims.topk(min(self.k * 2, N)).indices

        dW = torch.zeros_like(self.W)
        for idx in top_idx:
            diff = r_raw - self.V_raw[idx]
            norm2 = diff.norm() ** 2 + 1e-10
            rank1 = torch.outer(diff, diff) / norm2
            if self.labels[idx].item() == label:
                dW -= rank1       # suppress same-class differences
            else:
                dW += rank1       # amplify cross-class differences

        self.W = F.normalize(self.W + self.proj_lr * dW, dim=1)
        self.dirty = True

        if self.mode == 'online':
            self._reproject_all()

    def reproject(self):
        """Explicit lazy reproject (call after task training ends)."""
        self._reproject_all()


# ─── Data loading ─────────────────────────────────────────────────────────────

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
    return torch.from_numpy(X_np[:, perm]).to(DEVICE)   # NOT normalized — W handles it


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


# ─── Run ──────────────────────────────────────────────────────────────────────

def run_proj_lr(proj_lr, tasks_train, tasks_test, perms, X_te_all, y_te_all):
    n   = N_TASKS
    mat = [[None]*n for _ in range(n)]
    rp  = RetrievalProjection(d=784, k=K, proj_lr=proj_lr, reproject_mode=REPROJECT)

    for task_id in range(n):
        t_train = time.time()
        X_tr, y_tr = tasks_train[task_id]
        R_tr = embed_raw(X_tr, perms[task_id])   # (6000, 784), unnormalized

        for i in range(len(y_tr)):
            r_raw = F.normalize(R_tr[i], dim=0)
            label = int(y_tr[i])
            # Update W from retrieval pattern (before storing, consistent with Avir spec)
            if proj_lr > 0.0:
                rp.update_W(r_raw, label)
            rp.store(r_raw, label)

        # Lazy reproject after task ends
        if REPROJECT == 'lazy':
            rp.reproject()
        train_s = time.time() - t_train

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            R_te = embed_raw(X_te_all, perms[eval_task])
            R_te_norm = F.normalize(R_te, dim=1)
            preds = rp.predict_batch(R_te_norm)
            y_te_t = torch.from_numpy(y_te_all)
            acc   = (preds == y_te_t).float().mean().item()
            mat[eval_task][task_id] = acc
        eval_s = time.time() - t_eval

        aa_now = sum(mat[t][task_id] for t in range(task_id + 1)) / (task_id + 1)
        cb = rp.V_raw.shape[0]
        print(f"    T{task_id}: train={train_s:.1f}s eval={eval_s:.1f}s "
              f"cb={cb} aa={aa_now*100:.1f}%", flush=True)

    return compute_aa(mat, n), compute_fgt(mat, n)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 112 -- Retrieval-Driven Projection, P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, k={K}, DEVICE={DEVICE}, reproject={REPROJECT}", flush=True)
    print(f"Baseline (Step 110/111 k-NN): 95.4% AA", flush=True)
    print(f"Kill: AA <= 95.4%", flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X_tr_all, y_tr_all, X_te_all, y_te_all = load_mnist()
    perms = [make_permutation(seed=t * 100) for t in range(N_TASKS)]

    tasks_train = []
    for t in range(N_TASKS):
        Xtr, ytr = stratified_sample(X_tr_all, y_tr_all, TRAIN_PER_CLS, seed=t * 7)
        tasks_train.append((Xtr, ytr))

    # Run lr=0.0 first (k-NN control)
    all_lr_vals = [0.0] + PROJ_LR_VALS
    results = {}
    for proj_lr in all_lr_vals:
        label = f"proj_lr={proj_lr}"
        print(f"\n{'='*60}", flush=True)
        print(f"Run: {label}", flush=True)
        print(f"{'='*60}", flush=True)
        aa, fgt = run_proj_lr(proj_lr, tasks_train, [None]*N_TASKS, perms, X_te_all, y_te_all)
        results[label] = (aa, fgt)

    elapsed = time.time() - t0

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 112 FINAL -- Retrieval-Driven Projection", flush=True)
    print(f"Baseline (Step 110/111 k-NN): AA=95.4%", flush=True)
    print(f"{'='*72}", flush=True)
    baseline = 0.954
    for name, (aa, fgt) in results.items():
        delta = aa - baseline
        verdict = "PASSES" if aa > baseline else "DISPROVED"
        print(f"  {name:<22} AA={aa*100:.1f}%  fgt={fgt*100:.1f}pp  "
              f"delta={delta*100:+.1f}pp  [{verdict}]", flush=True)

    best_aa = max(aa for aa, _ in results.values())
    best_name = max(results, key=lambda k: results[k][0])
    overall = "PASSES" if best_aa > baseline else "DISPROVED"
    print(f"\n  OVERALL: {overall} (best: {best_name} → {best_aa*100:.1f}%)", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
