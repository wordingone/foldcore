#!/usr/bin/env python3
"""
Step 336 — CL with embedded per-entry weights on a%b. Stage 7 attempt.

Stage 7: the update rule becomes modifiable data. Per-entry distance weights
stored IN the codebook, updated BY competitive learning dynamics.

Chain:
  Step 296 (uniform, same-b):         86.8%
  Step 308 (global learned weights):   91.2%
  Step 333 (CL filter, uniform):       92.0%
  Step 336 (CL filter + per-entry):    ???

Kill: must beat Step 333 (92.0%).

Algorithm:
1. Build CL codebook from a%b with spawn_thresh=4.0 (~20 entries)
2. Each CL entry c gets per-entry phi weights w_c (K-dim, init uniform)
3. Training:
   For each (a,b,y):
     a. Winner c = nearest CL entry (L2 in [a,b] space)
     b. Group g = all data points assigned to c
     c. Compute phi for (a,b) within group g (LOO, no same-b filter)
     d. Predict: find nearest j in group g by weighted phi distance (using w_c)
     e. If wrong: upweight k-positions where phi_query differs most from phi_wrong
     f. Update w_c (as in auto_loop.py's learn_weights)
4. Evaluate: LOO accuracy using per-entry weights

Stage 7: the per-entry weights are part of the codebook (same data structure).
They are updated by the same CL dynamics (winner gets weight update).
No external evaluator — update signal comes from competitive match result.
"""

import numpy as np
import time

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX    = 20
K            = 5
SENTINEL     = TRAIN_MAX * 3
SPAWN_THRESH = 4.0  # from Step 333 best
N_EPOCHS     = 10
LR_FEAT      = 0.1
LR_W         = 0.1
SEED         = 42

# ─── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


# ─── Phi computation within CL group (no same-b filter) ───────────────────────

def compute_phi_group(query_a, A_group, Y_group, exclude_idx_in_group, K, max_class):
    """
    Compute phi for query_a restricted to group members.
    exclude_idx_in_group: index within the group to exclude (LOO).
    """
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    for c in range(max_class):
        class_mask = Y_group == c
        if exclude_idx_in_group is not None and exclude_idx_in_group < len(class_mask):
            if Y_group[exclude_idx_in_group] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx_in_group] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_group[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


# ─── Competitive learning ──────────────────────────────────────────────────────

def build_cl_codebook(X, spawn_thresh, lr=0.1, n_epochs=3, seed=42):
    """
    Build CL codebook in [a,b] space. Returns codebook, assignments.
    Same as Step 333.
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            if dists[winner] > spawn_thresh:
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T  # (n, n_cb)
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments


# ─── LOO evaluation (uniform weights, CL filter) ──────────────────────────────

def loo_cl_uniform(A, B, Y, assignments, K, max_class):
    """Baseline: CL filter + uniform weights (Step 333 result)."""
    n = len(A)
    w_uniform = np.ones(K, dtype=np.float64)

    # Precompute phi for all entries within their groups
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idx = np.where(group_mask)[0]
        pos_in_group = np.where(group_idx == i)[0]
        exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
        all_phi[i] = compute_phi_group(
            A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    correct = 0
    w_expanded = np.tile(w_uniform, max_class)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue  # Can't predict
        phi_q = all_phi[i]
        phi_group = all_phi[group_idxs]
        diffs = phi_group - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        # Exclude self
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        if Y[best_j] == Y[i]:
            correct += 1

    return correct / n


# ─── Per-entry weight learning ─────────────────────────────────────────────────

def train_per_entry_weights(A, B, Y, assignments, K, max_class,
                             n_epochs=N_EPOCHS, lr_w=LR_W, seed=SEED):
    """
    Learn per-CL-entry phi weights via competitive learning.
    Returns: cb_weights (n_cb, K) — per-entry phi weights.
    """
    rng = np.random.RandomState(seed)
    n = len(A)
    n_cb = assignments.max() + 1

    # Precompute phi for all entries within their groups (LOO)
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idx = np.where(group_mask)[0]
        pos_in_group = np.where(group_idx == i)[0]
        exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
        all_phi[i] = compute_phi_group(
            A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    # Initialize per-entry weights: uniform
    cb_weights = np.ones((n_cb, K), dtype=np.float64)

    order = np.arange(n)
    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            g = int(assignments[i])
            group_mask = assignments == g
            group_idxs = np.where(group_mask)[0]
            if len(group_idxs) <= 1:
                continue

            phi_q = all_phi[i]
            phi_group = all_phi[group_idxs]
            w = cb_weights[g]
            w_expanded = np.tile(w, max_class)

            diffs = phi_group - phi_q
            dists = (diffs ** 2 * w_expanded).sum(axis=1)

            # Exclude self
            self_pos = np.where(group_idxs == i)[0]
            if len(self_pos) > 0:
                dists[self_pos[0]] = float('inf')

            best_local = np.argmin(dists)
            best_j = group_idxs[best_local]

            if Y[best_j] != Y[i]:
                # Wrong: upweight k-positions where phi_q differs from phi_wrong
                diff_sq = (phi_q - all_phi[best_j]) ** 2
                per_k_signal = np.zeros(K)
                for k in range(K):
                    indices = [c * K + k for c in range(max_class)]
                    per_k_signal[k] = diff_sq[indices].mean()
                cb_weights[g] += lr_w * per_k_signal
                cb_weights[g] = np.maximum(cb_weights[g], 0.01)
                # Normalize per entry
                cb_weights[g] = cb_weights[g] / cb_weights[g].sum() * K

    return cb_weights


# ─── LOO evaluation with per-entry weights ────────────────────────────────────

def loo_cl_per_entry(A, B, Y, assignments, K, max_class, cb_weights):
    """LOO with CL filter + per-entry phi weights."""
    n = len(A)

    # Precompute phi for all entries within their groups (LOO)
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idx = np.where(group_mask)[0]
        pos_in_group = np.where(group_idx == i)[0]
        exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
        all_phi[i] = compute_phi_group(
            A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    correct = 0
    for i in range(n):
        g = int(assignments[i])
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue

        phi_q = all_phi[i]
        phi_group = all_phi[group_idxs]
        w = cb_weights[g]
        w_expanded = np.tile(w, max_class)

        diffs = phi_group - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        if Y[best_j] == Y[i]:
            correct += 1

    return correct / n


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(SEED)

    print("Step 336 — CL with embedded per-entry weights on a%b", flush=True)
    print(f"K={K}, spawn_thresh={SPAWN_THRESH}, n_epochs={N_EPOCHS}", flush=True)
    print("Stage 7: per-entry weights stored in codebook, updated by CL dynamics", flush=True)
    print("Kill: must beat Step 333 (92.0%, CL filter + uniform weights)", flush=True)
    print(flush=True)

    A, B, Y = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    X = np.column_stack([A, B]).astype(np.float32)
    print(f"Dataset: {n} entries, max_class={max_class}", flush=True)
    print(flush=True)

    # Build CL codebook (same as Step 333)
    print(f"Building CL codebook (spawn_thresh={SPAWN_THRESH})...", flush=True)
    codebook, assignments = build_cl_codebook(X, SPAWN_THRESH)
    n_cb = len(codebook)
    group_sizes = [(assignments == c).sum() for c in range(n_cb)]
    print(f"  Codebook size: {n_cb}", flush=True)
    print(f"  Group sizes: min={min(group_sizes)} max={max(group_sizes)} "
          f"mean={np.mean(group_sizes):.1f}", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Baseline: CL filter + uniform weights (Step 333)
    print("Computing baseline (CL filter + uniform weights)...", flush=True)
    acc_uniform = loo_cl_uniform(A, B, Y, assignments, K, max_class)
    print(f"  Step 333 baseline LOO: {acc_uniform*100:.2f}%  (expected: ~92.0%)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Learn per-entry weights
    print(f"Training per-entry weights ({N_EPOCHS} epochs, lr={LR_W})...", flush=True)
    cb_weights = train_per_entry_weights(A, B, Y, assignments, K, max_class)
    print(f"  Per-entry weights:", flush=True)
    for c in range(n_cb):
        w = cb_weights[c]
        size = group_sizes[c] if c < len(group_sizes) else 0
        print(f"    entry {c:2d} (n={size:3d}): {np.round(w, 3).tolist()}", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Evaluate with per-entry weights
    print("Evaluating with per-entry weights...", flush=True)
    acc_per_entry = loo_cl_per_entry(A, B, Y, assignments, K, max_class, cb_weights)
    delta = acc_per_entry - acc_uniform
    print(f"  Per-entry LOO: {acc_per_entry*100:.2f}%  (delta: {delta*100:+.2f}pp)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)

    # Average weight profile across entries
    print(flush=True)
    print("Average weight profile across CL entries:", flush=True)
    w_avg = cb_weights.mean(axis=0)
    w_avg = w_avg / w_avg.sum() * K
    print(f"  Avg: {np.round(w_avg, 3).tolist()}", flush=True)
    # Weight diversity
    norms = np.linalg.norm(cb_weights, axis=1, keepdims=True)
    W_norm = cb_weights / np.maximum(norms, 1e-8)
    cosine_sim = W_norm @ W_norm.T
    idxs = np.triu_indices(n_cb, k=1)
    avg_cosine = float(cosine_sim[idxs].mean())
    print(f"  Avg pairwise cosine sim: {avg_cosine:.4f}  (1=identical, lower=diverse)", flush=True)

    elapsed = time.time() - t0

    # Summary
    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 336 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"A. Uniform, same-b (Step 296):          86.8%", flush=True)
    print(f"B. Global learned (Step 308):            91.2%", flush=True)
    print(f"C. CL filter, uniform (Step 333):        {acc_uniform*100:.2f}%", flush=True)
    print(f"D. CL filter, per-entry (Step 336):      {acc_per_entry*100:.2f}%", flush=True)
    print(f"   Delta D vs C: {delta*100:+.2f}pp", flush=True)

    kill    = acc_per_entry <= acc_uniform
    success = acc_per_entry > acc_uniform

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Step 333 baseline: {acc_uniform*100:.2f}%", flush=True)
    print(f"Step 336 result:   {acc_per_entry*100:.2f}%", flush=True)
    print(f"Delta: {delta*100:+.2f}pp", flush=True)
    print(f"Kill (per-entry <= uniform): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (per-entry > uniform): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — per-entry weights don't improve over CL filter alone", flush=True)
    else:
        print(f"\nSUCCESS — Stage 7 works: per-entry weights add {delta*100:.2f}pp",
              flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
