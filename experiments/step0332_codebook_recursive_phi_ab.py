#!/usr/bin/env python3
"""
Step 332 — Recursive phi on a%b.

Step 328 killed recursive phi on ARC: identical patches -> identical phi.
But a%b has UNIQUE entries -> unique phi_1 -> phi_2 can encode group structure.

Algorithm:
1. Build a%b (400 entries, a,b in 1..20)
2. phi_1: per-class sorted top-K=5 distances (LOO, same-b filtering) [as auto_loop]
3. phi_2: per-class sorted top-K=5 distances IN PHI_1 SPACE (LOO, no same-b filter)
4. Classify:
   C. phi_2 alone: 1-NN in phi_2 space
   D. phi_1 + phi_2 concat (200-dim): 1-NN

Compare:
  A. Raw 1-NN on [a,b]:   ~5%
  B. phi_1 (Step 296):    86.8%
  C. phi_2 alone:         ???
  D. phi_1 + phi_2:       ???

Kill: phi_2 or phi_1+phi_2 must beat phi_1 (86.8%).

Key difference from Step 328 (ARC): a%b entries are unique, phi_1 vectors
are unique -> phi_2 encodes b-grouping structure (same-b entries cluster
in phi_1 space). If phi_2 captures this, it should improve or match phi_1.
"""

import numpy as np
import time
from scipy.spatial.distance import cdist

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
K         = 5
SENTINEL  = TRAIN_MAX * 3

# ─── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


# ─── Phi_1 (from auto_loop.py, with same-b filtering) ─────────────────────────

def compute_phi1_entry(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    """phi_1 with same-b filtering (as in auto_loop.py)."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b_mask = (B == query_b)
    for c in range(max_class):
        class_mask = (Y == c) & same_b_mask
        if exclude_idx is not None and class_mask[exclude_idx]:
            if Y[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def compute_all_phi1(A, B, Y, K):
    """Compute LOO phi_1 for all entries."""
    n = len(A)
    max_class = int(Y.max()) + 1
    dim = max_class * K
    all_phi1 = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        all_phi1[i] = compute_phi1_entry(A[i], B[i], A, B, Y, i, K, max_class)
    return all_phi1, max_class


# ─── Phi_2 (in phi_1 space, no same-b filtering) ──────────────────────────────

def compute_phi2(all_phi1, Y, K, max_class):
    """
    phi_2[i, c*K:c*K+K] = top-K sorted distances from phi_1[i] to class-c
    phi_1 vectors, LOO (exclude self), NO same-b filtering.
    """
    n = len(all_phi1)
    dim = max_class * K
    phi2 = np.zeros((n, dim), dtype=np.float32)

    # Pairwise distances in phi_1 space
    print("  Computing pairwise phi_1 distances...", flush=True)
    dists_phi1 = cdist(all_phi1, all_phi1, metric='sqeuclidean')  # (n, n)

    for i in range(n):
        for c in range(max_class):
            # Class-c entries, excluding self
            class_idxs = [j for j in range(n) if j != i and Y[j] == c]
            if len(class_idxs) == 0:
                phi2[i, c*K:(c+1)*K] = float(SENTINEL)
                continue
            class_dists = dists_phi1[i, class_idxs]
            k_eff = min(K, len(class_dists))
            sorted_dists = np.sort(class_dists)[:k_eff]
            phi2[i, c*K:c*K + k_eff] = sorted_dists
            # Fill remaining slots with max
            if k_eff < K:
                phi2[i, c*K + k_eff:c*K + K] = sorted_dists[-1] * 2 if k_eff > 0 else SENTINEL

    return phi2


# ─── LOO evaluation ────────────────────────────────────────────────────────────

def loo_1nn(features, Y):
    """LOO 1-NN accuracy in given feature space."""
    dists = cdist(features, features, metric='sqeuclidean')
    np.fill_diagonal(dists, float('inf'))
    nn = np.argmin(dists, axis=1)
    return (Y[nn] == Y).mean()


def raw_1nn_ab(A, B, Y):
    """Baseline: 1-NN on raw [a, b] features (LOO)."""
    X = np.column_stack([A, B]).astype(np.float32)
    return loo_1nn(X, Y)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(42)

    print("Step 332 — Recursive phi on a%b", flush=True)
    print(f"K={K}, TRAIN_MAX={TRAIN_MAX}", flush=True)
    print("phi_1: same-b filtered (auto_loop.py), phi_2: phi_1 space, no same-b", flush=True)
    print("Kill: phi_2 or phi_1+phi_2 must beat phi_1 (86.8%)", flush=True)
    print(flush=True)

    A, B, Y = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    print(f"Dataset: {n} entries, max_class={max_class}", flush=True)
    print(flush=True)

    # ─── Baseline A: raw 1-NN on [a,b] ───────────────────────────────────────
    acc_raw = raw_1nn_ab(A, B, Y)
    print(f"A. Raw 1-NN [a,b]:  {acc_raw*100:.1f}%  (expected: ~5%)", flush=True)
    print(f"   Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ─── Phi_1 ───────────────────────────────────────────────────────────────
    print("Computing phi_1 (LOO, same-b filtered)...", flush=True)
    all_phi1, max_class = compute_all_phi1(A, B, Y, K)
    acc_phi1 = loo_1nn(all_phi1, Y)
    print(f"B. phi_1 1-NN:      {acc_phi1*100:.1f}%  (expected: 86.8%)", flush=True)
    print(f"   phi_1 shape: {all_phi1.shape}", flush=True)
    print(f"   Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ─── Phi_2 ───────────────────────────────────────────────────────────────
    print("Computing phi_2 (LOO, no same-b filtering, in phi_1 space)...", flush=True)
    all_phi2 = compute_phi2(all_phi1, Y, K, max_class)
    print(f"   phi_2 shape: {all_phi2.shape}", flush=True)

    acc_phi2 = loo_1nn(all_phi2, Y)
    print(f"C. phi_2 1-NN:      {acc_phi2*100:.1f}%", flush=True)
    print(f"   Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ─── phi_1 + phi_2 concat ────────────────────────────────────────────────
    print("Concatenating phi_1 + phi_2 (200-dim)...", flush=True)
    phi_cat = np.concatenate([all_phi1, all_phi2], axis=1)
    acc_cat = loo_1nn(phi_cat, Y)
    print(f"D. phi_1+phi_2 1-NN: {acc_cat*100:.1f}%", flush=True)
    print(f"   Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # ─── Analysis ────────────────────────────────────────────────────────────
    print("Analysis — phi_2 structure:", flush=True)

    # Do same-b entries cluster in phi_2 space?
    # R² of b from 1-NN in phi_2 space
    dists_phi2 = cdist(all_phi2, all_phi2, metric='sqeuclidean')
    np.fill_diagonal(dists_phi2, float('inf'))
    nn_phi2 = np.argmin(dists_phi2, axis=1)
    b_match_phi2 = (B[nn_phi2] == B).mean()
    print(f"  1-NN in phi_2 space: {b_match_phi2*100:.1f}% of pairs share same b-value", flush=True)

    dists_phi1 = cdist(all_phi1, all_phi1, metric='sqeuclidean')
    np.fill_diagonal(dists_phi1, float('inf'))
    nn_phi1 = np.argmin(dists_phi1, axis=1)
    b_match_phi1 = (B[nn_phi1] == B).mean()
    print(f"  1-NN in phi_1 space: {b_match_phi1*100:.1f}% of pairs share same b-value", flush=True)

    # phi_2 variance vs phi_1 variance
    print(f"  phi_1 variance: {all_phi1.var():.4f}", flush=True)
    print(f"  phi_2 variance: {all_phi2.var():.4f}", flush=True)

    # Correlation between phi_1 and phi_2 distances
    phi1_dists_flat = dists_phi1[np.triu_indices(n, k=1)]
    phi2_dists_flat = dists_phi2[np.triu_indices(n, k=1)]
    np.fill_diagonal(dists_phi2, 0)  # reset
    np.fill_diagonal(dists_phi1, 0)
    # Quick Pearson on flat arrays (sampled)
    sample_idx = np.random.choice(len(phi1_dists_flat), min(10000, len(phi1_dists_flat)), replace=False)
    corr = np.corrcoef(phi1_dists_flat[sample_idx], phi2_dists_flat[sample_idx])[0, 1]
    print(f"  Corr(phi_1 dists, phi_2 dists): {corr:.4f}", flush=True)

    # ─── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 332 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"A. Raw 1-NN [a,b]:    {acc_raw*100:.2f}%  (baseline: ~5%)", flush=True)
    print(f"B. phi_1 1-NN:        {acc_phi1*100:.2f}%  (baseline: 86.8%)", flush=True)
    print(f"C. phi_2 alone:       {acc_phi2*100:.2f}%", flush=True)
    print(f"D. phi_1 + phi_2:     {acc_cat*100:.2f}%", flush=True)
    print(flush=True)

    delta_phi2 = acc_phi2 - acc_phi1
    delta_cat  = acc_cat  - acc_phi1
    print(f"phi_2 vs phi_1:       {delta_phi2*100:+.2f}pp", flush=True)
    print(f"phi_1+phi_2 vs phi_1: {delta_cat*100:+.2f}pp", flush=True)

    best_acc = max(acc_phi2, acc_cat)
    best_label = 'phi_2' if acc_phi2 >= acc_cat else 'phi_1+phi_2'
    kill = best_acc <= acc_phi1
    success = best_acc > acc_phi1

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"phi_1 baseline: {acc_phi1*100:.2f}%", flush=True)
    print(f"Best (={best_label}): {best_acc*100:.2f}%", flush=True)
    print(f"Kill (best <= phi_1): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (best > phi_1): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — recursive phi adds no information on a%b", flush=True)
        if acc_phi2 > acc_raw:
            print("(phi_2 does capture structure, but not better than phi_1)", flush=True)
    else:
        print(f"\nSUCCESS — recursive phi improves: {best_label} {best_acc*100:.2f}% "
              f"(+{(best_acc-acc_phi1)*100:.2f}pp)", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
