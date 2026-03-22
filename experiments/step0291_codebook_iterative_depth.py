#!/usr/bin/env python3
"""
Step 291 -- Iterative Depth: does re-entering the fold improve accuracy?

The wall: fold runs once. This tests whether K passes through the codebook
improve accuracy by iterative refinement.

Pass 1: input -> top-k lookup -> weighted avg of K nearest vectors -> intermediate
Pass 2: intermediate -> top-k lookup -> class vote
...up to K_DEPTH passes.

Test on P-MNIST (Lipschitz, should benefit from denoising) and report
accuracy at each depth level.

Kill criterion: if depth=5 accuracy <= depth=1 accuracy, iterative fold
produces no useful intermediates. The fold cannot usefully re-enter itself.

Baseline: Step 99 k=3 91.8% AA, Step 101 spawn-only k=3 ~91% AA.
"""

import sys
import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

# --- Config ---
N_TASKS       = int(sys.argv[1]) if len(sys.argv) > 1 else 10
D_OUT         = 384
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'

SPAWN_THRESH  = 0.7
K_VOTE        = 3
K_REFINE      = 5       # neighbors used to build intermediate vector
DEPTH_VALS    = [1, 2, 3, 5]  # number of fold passes


class IterativeDepthFold:
    """Codebook with iterative re-entry. Same as SpawnOnlyFold but eval
    supports multiple depth passes."""

    def __init__(self, d, spawn_thresh=0.7):
        self.V      = torch.empty(0, d, device=DEVICE)
        self.labels = torch.empty(0, dtype=torch.long, device=DEVICE)
        self.spawn_thresh = spawn_thresh
        self.d      = d

    def step(self, r, label):
        r = F.normalize(r, dim=0)
        if self.V.shape[0] == 0 or (self.V @ r).max().item() < self.spawn_thresh:
            self.V      = torch.cat([self.V, r.unsqueeze(0)])
            self.labels = torch.cat([self.labels,
                                     torch.tensor([label], device=DEVICE)])

    def _refine(self, R, k_refine):
        """One refinement pass: replace each input with weighted avg of
        k_refine nearest codebook vectors (weighted by cosine sim)."""
        sims = R @ self.V.T                        # (n, N_cb)
        k_eff = min(k_refine, self.V.shape[0])
        top_sims, top_idx = sims.topk(k_eff, dim=1)  # (n, k_eff)
        # Softmax weights over top-k similarities
        weights = F.softmax(top_sims * 10.0, dim=1)  # temperature-scaled
        # Weighted combination of codebook vectors
        # top_idx: (n, k_eff), weights: (n, k_eff)
        gathered = self.V[top_idx]                    # (n, k_eff, d)
        refined = (weights.unsqueeze(-1) * gathered).sum(dim=1)  # (n, d)
        return F.normalize(refined, dim=1)

    def eval_batch(self, R, k_vote, depths, k_refine):
        """Evaluate at multiple depth levels. depth=1 is standard (no refinement).
        depth=2+ applies _refine before final classification."""
        R = F.normalize(R, dim=1)
        n = len(R)
        n_cls = int(self.labels.max().item()) + 1
        results = {}

        for depth in depths:
            R_cur = R
            # Apply (depth-1) refinement passes before final classification
            for _ in range(depth - 1):
                R_cur = self._refine(R_cur, k_refine)

            # Final classification: top-k class vote
            sims = R_cur @ self.V.T
            scores = torch.zeros(n, n_cls, device=DEVICE)
            for c in range(n_cls):
                mask = (self.labels == c)
                if mask.sum() == 0:
                    continue
                class_sims = sims[:, mask]
                k_eff = min(k_vote, class_sims.shape[1])
                scores[:, c] = class_sims.topk(k_eff, dim=1).values.sum(dim=1)
            results[depth] = scores.argmax(dim=1).cpu()

        return results


# --- MNIST + embedding (same as Step 101) ---

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST(
        './data/mnist', train=True, download=True)
    te = torchvision.datasets.MNIST(
        './data/mnist', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()


def make_projection(d_in=784, d_out=D_OUT, seed=12345):
    rng = np.random.RandomState(seed)
    P = rng.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
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


def main():
    t0 = time.time()
    print(f"Step 291 -- Iterative Depth Test", flush=True)
    print(f"N_TASKS={N_TASKS}, D_OUT={D_OUT}, DEVICE={DEVICE}", flush=True)
    print(f"k_vote={K_VOTE}, k_refine={K_REFINE}, depths={DEPTH_VALS}", flush=True)
    print(f"Kill criterion: depth=5 acc <= depth=1 acc => fold cannot re-enter", flush=True)
    print(flush=True)

    X_tr, y_tr, X_te, y_te = load_mnist()
    P = make_projection()
    y_te_t = torch.from_numpy(y_te).long()

    perms = [list(range(784))]
    for t in range(1, N_TASKS):
        perms.append(make_permutation(seed=t * 1000))

    test_embeds = [embed(X_te, perms[t], P) for t in range(N_TASKS)]

    model = IterativeDepthFold(D_OUT, spawn_thresh=SPAWN_THRESH)

    acc = {d: [[None] * N_TASKS for _ in range(N_TASKS)] for d in DEPTH_VALS}

    for task_id in range(N_TASKS):
        print(f"=== Task {task_id} ===", flush=True)
        t_task = time.time()

        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS,
                                         seed=task_id * 1337)
        X_emb = embed(X_sub, perms[task_id], P)

        cb_before = model.V.shape[0]
        for i in range(len(X_emb)):
            model.step(X_emb[i], y_sub[i])
        cb_after = model.V.shape[0]
        print(f"  cb: {cb_before}->{cb_after} (+{cb_after - cb_before})", flush=True)

        t_eval = time.time()
        for eval_task in range(task_id + 1):
            preds = model.eval_batch(test_embeds[eval_task], K_VOTE,
                                     DEPTH_VALS, K_REFINE)
            for d in DEPTH_VALS:
                acc[d][eval_task][task_id] = \
                    (preds[d] == y_te_t).float().mean().item()

        print(f"  Eval: {time.time()-t_eval:.1f}s", flush=True)
        for d in DEPTH_VALS:
            a = acc[d][task_id][task_id]
            if a is not None:
                print(f"    depth={d}: {a*100:.1f}%", flush=True)
        print(flush=True)

    elapsed = time.time() - t0

    # --- Summary ---
    print("=" * 70, flush=True)
    print("STEP 291 SUMMARY -- Iterative Depth on P-MNIST", flush=True)
    print(f"k_vote={K_VOTE}, k_refine={K_REFINE}, sp={SPAWN_THRESH}", flush=True)
    print("=" * 70, flush=True)

    aa_vals = {d: compute_aa(acc[d], N_TASKS) for d in DEPTH_VALS}
    fgt_vals = {d: compute_fgt(acc[d], N_TASKS) for d in DEPTH_VALS}
    aa_d1 = aa_vals[1]

    print(f"\n{'depth':>6} | {'AA':>7} {'Forgetting':>11} | {'vs depth=1':>10}", flush=True)
    print("-" * 45, flush=True)
    for d in DEPTH_VALS:
        delta = aa_vals[d] - aa_d1
        marker = " <-- baseline" if d == 1 else ""
        print(f"{d:>6} | {aa_vals[d]*100:>6.1f}% {fgt_vals[d]*100:>10.1f}pp | "
              f"{delta*100:>+9.1f}pp{marker}", flush=True)

    print(f"\nCodebook size: {model.V.shape[0]}", flush=True)
    print(f"Step 99 baseline: k=3 91.8% AA", flush=True)

    # Verdict
    print(flush=True)
    if aa_vals[5] > aa_d1 + 0.001:
        print("VERDICT: DEPTH HELPS. Iterative fold improves accuracy.", flush=True)
        print("The fold can usefully re-enter itself.", flush=True)
    elif aa_vals[5] >= aa_d1 - 0.001:
        print("VERDICT: DEPTH NEUTRAL. No improvement but no degradation.", flush=True)
        print("Refinement intermediates are not harmful but not useful.", flush=True)
    else:
        print("VERDICT: DEPTH HURTS. Iterative fold degrades accuracy.", flush=True)
        print("The fold's intermediates are noise, not computation.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
