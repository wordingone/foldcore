#!/usr/bin/env python3
"""
Step 101 -- Spawn-Only Codebook (lr=0) on P-MNIST.
Spec. Verify P-MNIST performance doesn't degrade with lr=0.

Same as Step 99 (lr=0.001) but no updates. Quick comparison.
sp=0.7, k_vote={1,3,5,10}.

Compare to Step 99: k=3 91.8% AA, 0.0pp forgetting.
"""

import sys
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# ─── Config ───────────────────────────────────────────────────────────────────

N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

SPAWN_THRESH = 0.7
K_VOTE_VALS  = [1, 3, 5, 10]


# ─── SpawnOnlyFold ────────────────────────────────────────────────────────────

class SpawnOnlyFold:
    """TopKFold with lr=0. No updates, only spawning."""

    def __init__(self, d, spawn_thresh=0.7):
        self.V          = torch.empty(0, d, device=DEVICE)
        self.labels     = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.spawn_thresh = spawn_thresh
        self.d          = d
        self.n_spawned  = 0

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])
            self.n_spawned += 1
        # No update.

    def eval_batch(self, R, k_vals):
        R    = F.normalize(R, dim=1)
        sims = R @ self.V.T                   # (n, N)
        n    = len(R)
        n_cls = int(self.labels.max().item()) + 1
        results = {}
        for k in k_vals:
            scores = torch.zeros(n, n_cls, device=DEVICE)
            for c in range(n_cls):
                mask = (self.labels == c)
                if mask.sum() == 0:
                    continue
                class_sims = sims[:, mask]
                k_eff      = min(k, class_sims.shape[1])
                scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
            results[k] = scores.argmax(dim=1).cpu()
        return results


# ─── MNIST + embedding ────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST(
        'C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=D_OUT, seed=12345):
    rng = np.random.RandomState(seed)
    P   = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
    return torch.from_numpy(P).to(DEVICE)


def make_permutation(seed):
    perm = list(range(784))
    random.Random(seed).shuffle(perm)
    return perm


def embed(X_flat_np, perm, P):
    X_t = torch.from_numpy(X_flat_np[:, perm]).to(DEVICE)
    return F.normalize(X_t @ P.T, dim=1)


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_aa(mat, n_tasks):
    vals = [mat[t][n_tasks - 1] for t in range(n_tasks)
            if mat[t][n_tasks - 1] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def compute_fgt(mat, n_tasks):
    vals = []
    for t in range(n_tasks - 1):
        if mat[t][t] is not None and mat[t][n_tasks - 1] is not None:
            vals.append(max(0.0, mat[t][t] - mat[t][n_tasks - 1]))
    return sum(vals) / len(vals) if vals else 0.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Step 101 -- Spawn-Only (lr=0), P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"Config: lr=0 (spawn-only), spawn_thresh={SPAWN_THRESH}", flush=True)
    print(f"k_vote sweep: {K_VOTE_VALS}", flush=True)
    print(f"Compare: Step 99 (lr=0.001) k=3 91.8% AA, 0.0pp forgetting", flush=True)
    print(flush=True)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P      = make_projection()
    y_te_t = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    print("Pre-embedding test sets...", flush=True)
    test_embeds = [embed(X_te, perms[t], P) for t in range(N_TASKS)]
    print(f"  Done: {N_TASKS} tasks x {N_TEST_TASK} samples", flush=True)
    print(flush=True)

    model = SpawnOnlyFold(D_OUT, spawn_thresh=SPAWN_THRESH)

    acc = {k: [[None] * N_TASKS for _ in range(N_TASKS)]
           for k in K_VOTE_VALS}

    for task_id in range(N_TASKS):
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb       = embed(X_sub, perms[task_id], P)
        labels_list = y_sub.tolist()

        cb_before = model.V.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], labels_list[i])
        cb_after = model.V.shape[0]
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after - cb_before})", flush=True)
        print(f"  Train: {time.time() - t_task:.1f}s", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            preds = model.eval_batch(test_embeds[eval_task], K_VOTE_VALS)
            for k in K_VOTE_VALS:
                acc[k][eval_task][task_id] = \
                    (preds[k] == y_te_t).float().mean().item()

        print(f"  Eval ({task_id+1} tasks): {time.time()-t_eval:.1f}s", flush=True)
        k1  = acc[1][task_id][task_id]
        k3  = acc[3][task_id][task_id]
        k10 = acc[10][task_id][task_id]
        if k1 is not None:
            print(f"  Peek: k=1={k1*100:.1f}% k=3={k3*100:.1f}% "
                  f"k=10={k10*100:.1f}%", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # ─── Summary ──────────────────────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("STEP 101 SUMMARY -- Spawn-Only P-MNIST", flush=True)
    print(f"N_TASKS={N_TASKS}, lr=0, sp={SPAWN_THRESH}, DEVICE={DEVICE}", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'k':>4} | {'AA':>7} {'Forgetting':>11} | {'vs k=1':>8}", flush=True)
    print("-" * 40, flush=True)

    aa_vals  = {k: compute_aa(acc[k], N_TASKS)  for k in K_VOTE_VALS}
    fgt_vals = {k: compute_fgt(acc[k], N_TASKS) for k in K_VOTE_VALS}
    aa_1nn   = aa_vals[1]

    for k in K_VOTE_VALS:
        delta  = aa_vals[k] - aa_1nn
        marker = " <-- 1-NN baseline" if k == 1 else ""
        print(f"{k:>4} | {aa_vals[k]*100:>6.1f}% {fgt_vals[k]*100:>10.1f}pp | "
              f"{delta*100:>+7.1f}pp{marker}", flush=True)

    print(flush=True)
    print(f"Final codebook size: {model.V.shape[0]}", flush=True)
    print(f"Total spawned: {model.n_spawned}", flush=True)
    print(flush=True)
    print(f"Step 99 comparison (lr=0.001): k=3 91.8% AA, 0.0pp forgetting", flush=True)
    print(f"Baseline (fold Step 65): 56.7% AA, 0.0pp forgetting", flush=True)
    print(f"1-NN (lr=0): {aa_1nn*100:.1f}% AA", flush=True)
    print(f"vs fold baseline (56.7%): {(aa_1nn - 0.567)*100:+.1f}pp", flush=True)

    best_k       = max((k for k in K_VOTE_VALS if k > 1), key=lambda k: aa_vals[k])
    best_topk_aa = aa_vals[best_k]
    print(flush=True)
    print("VERDICT:", flush=True)
    print(f"  Best top-k (k={best_k}): {best_topk_aa*100:.1f}%", flush=True)
    print(f"  Forgetting (k={best_k}): {fgt_vals[best_k]*100:.1f}pp", flush=True)
    delta_vs_99 = best_topk_aa - 0.918
    print(f"  vs Step 99 best (91.8%): {delta_vs_99*100:+.1f}pp", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
