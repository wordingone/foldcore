#!/usr/bin/env python3
"""
Step 296 -- Per-class distribution matching on a%b.

Leo mail 1341. Hypothesis: per-class sorted distance vectors (φ) are Lipschitz
even when the raw function isn't. Same-class examples should have identical φ
(verified by hand: φ(7,3) == φ(10,3), dist²=0). Different-class examples have
dist²=18 in distribution space.

Architecture:
- Codebook: all 400 training examples (a,b) ∈ 1..20
- Distance metric: |a-a'| for same-b pairs only; ∞ otherwise
- φ(x) = concat over classes of sorted(top-K distances from x to same-b same-class vectors)
- Predict: argmin_y ||φ(x) - φ(y)|| in distribution space

Compare to: 1-NN (same distance metric), top-K sum readout

Kill criterion: distribution matching LOO ≤ top-K sum LOO
Success: distribution matching LOO > top-K sum LOO
"""

import time
import numpy as np
from itertools import product

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
K_SWEEP   = [1, 2, 3, 5]
SENTINEL  = TRAIN_MAX * 3   # distance for missing class members
SPECIFIC_B = [3, 4, 5]     # verify per-b first

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_dataset():
    """Returns A, B, Y arrays."""
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)

# ─── φ computation ────────────────────────────────────────────────────────────

def compute_phi(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    """
    Compute distribution feature vector φ for query (query_a, query_b).
    Uses same-b codebook vectors only. Excludes exclude_idx from codebook.
    """
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b_mask = (B == query_b)

    for c in range(max_class):
        class_mask = (Y == c) & same_b_mask
        if exclude_idx is not None and exclude_idx < len(A) and class_mask[exclude_idx]:
            # Don't exclude from wrong class (self-exclusion only when own class)
            if Y[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False

        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            phi[c * K:(c + 1) * K] = SENTINEL
            continue

        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
        # remaining slots stay at SENTINEL

    return phi

# ─── Readout methods ──────────────────────────────────────────────────────────

def predict_1nn(query_a, query_b, A, B, Y, exclude_idx):
    """1-NN using integer distance, same-b only."""
    same_b = (B == query_b)
    if exclude_idx is not None:
        same_b = same_b.copy(); same_b[exclude_idx] = False
    idxs = np.where(same_b)[0]
    if len(idxs) == 0:
        return -1
    dists = np.abs(A[idxs] - query_a)
    return int(Y[idxs[np.argmin(dists)]])


def predict_topk_sum(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    """Top-K sum readout: argmin over classes of sum of K smallest distances."""
    same_b = (B == query_b)
    scores = np.full(max_class, float('inf'))
    for c in range(max_class):
        mask = (Y == c) & same_b
        if exclude_idx is not None and exclude_idx < len(mask) and mask[exclude_idx]:
            if Y[exclude_idx] == c:
                mask = mask.copy(); mask[exclude_idx] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a)
        dists.sort()
        k_eff = min(K, len(dists))
        scores[c] = dists[:k_eff].sum()
    if np.all(np.isinf(scores)):
        return -1
    return int(np.argmin(scores))


def predict_dist_match(phi_query, phi_all, Y, exclude_idx):
    """NN in distribution space. Returns predicted label."""
    diffs = phi_all - phi_query
    dists = (diffs * diffs).sum(axis=1)
    if exclude_idx is not None:
        dists[exclude_idx] = float('inf')
    return int(Y[np.argmin(dists)])

# ─── LOO evaluation ──────────────────────────────────────────────────────────

def evaluate_loo(A, B, Y, K, filter_b=None, verbose=False):
    """
    LOO evaluation for all three methods.
    filter_b: if set, only evaluate examples with B[i]==filter_b.
    """
    max_class = int(Y.max()) + 1
    n = len(A)
    eval_idxs = np.where(B == filter_b)[0] if filter_b is not None else np.arange(n)

    # Precompute φ for all examples (self-excluded)
    phi_all = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        phi_all[i] = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i, K=K,
                                 max_class=max_class)

    # Metrics
    correct_1nn = 0
    correct_topk = 0
    correct_dist = 0
    n_eval = len(eval_idxs)

    for i in eval_idxs:
        true_y = Y[i]

        # 1-NN
        pred_1nn = predict_1nn(A[i], B[i], A, B, Y, exclude_idx=i)
        if pred_1nn == true_y: correct_1nn += 1

        # Top-K sum
        pred_topk = predict_topk_sum(A[i], B[i], A, B, Y, exclude_idx=i,
                                     K=K, max_class=max_class)
        if pred_topk == true_y: correct_topk += 1

        # Distribution matching
        phi_q = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i, K=K,
                            max_class=max_class)
        pred_dist = predict_dist_match(phi_q, phi_all, Y, exclude_idx=i)
        if pred_dist == true_y: correct_dist += 1

    acc_1nn  = correct_1nn  / n_eval
    acc_topk = correct_topk / n_eval
    acc_dist = correct_dist / n_eval

    # Margin: same-class vs diff-class distances in φ space
    if filter_b is not None:
        b_idxs = eval_idxs
        same_dists, diff_dists = [], []
        for i in b_idxs:
            for j in b_idxs:
                if i >= j:
                    continue
                d2 = float(((phi_all[i] - phi_all[j])**2).sum())
                if Y[i] == Y[j]:
                    same_dists.append(d2)
                else:
                    diff_dists.append(d2)
        margin_same = float(np.mean(same_dists)) if same_dists else float('nan')
        margin_diff = float(np.mean(diff_dists)) if diff_dists else float('nan')
    else:
        margin_same = margin_diff = float('nan')

    return acc_1nn, acc_topk, acc_dist, margin_same, margin_diff


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 296 -- Per-class Distribution Matching on a%b", flush=True)
    print(f"Dataset: a,b in 1..{TRAIN_MAX}, 400 pairs", flush=True)
    print(f"Distance: |a-a'| for same-b pairs only", flush=True)
    print(f"K sweep: {K_SWEEP}", flush=True)
    print(f"Step 286 baseline (plain thermometer 1-NN): 41.8%", flush=True)
    print(flush=True)

    A, B, Y = build_dataset()

    # Hand-check: verify φ(7,3) == φ(10,3), dist²(φ(7,3), φ(8,3)) == 18
    print("=== Hand verification (K=3, b=3 examples) ===", flush=True)
    K_check = 3
    max_class_check = int(Y.max()) + 1
    phi_7_3 = compute_phi(7, 3, A, B, Y, exclude_idx=None, K=K_check,
                          max_class=max_class_check)
    phi_10_3 = compute_phi(10, 3, A, B, Y, exclude_idx=None, K=K_check,
                           max_class=max_class_check)
    phi_8_3  = compute_phi(8, 3, A, B, Y, exclude_idx=None, K=K_check,
                           max_class=max_class_check)
    idx_73  = int(np.where((A==7)  & (B==3))[0][0])
    idx_103 = int(np.where((A==10) & (B==3))[0][0])
    idx_83  = int(np.where((A==8)  & (B==3))[0][0])
    # Classes
    print(f"  (7,3) class={Y[idx_73]}, (10,3) class={Y[idx_103]}, (8,3) class={Y[idx_83]}",
          flush=True)
    d2_same = float(((phi_7_3 - phi_10_3)**2).sum())
    d2_diff = float(((phi_7_3 - phi_8_3 )**2).sum())
    print(f"  φ(7,3):  {phi_7_3[:9].astype(int).tolist()}...", flush=True)
    print(f"  φ(10,3): {phi_10_3[:9].astype(int).tolist()}...", flush=True)
    print(f"  φ(8,3):  {phi_8_3[:9].astype(int).tolist()}...", flush=True)
    print(f"  dist²(φ(7,3), φ(10,3)) [same-class]: {d2_same:.1f} (expect 0)", flush=True)
    print(f"  dist²(φ(7,3), φ(8,3))  [diff-class]:  {d2_diff:.1f} (expect 18)", flush=True)
    print(flush=True)

    # Per-b evaluation (b=3,4,5)
    print("=== Per-b LOO evaluation (K=3) ===", flush=True)
    K = 3
    print(f"{'b':>4} | {'1-NN':>7} {'Top-K sum':>10} {'Dist match':>11} | "
          f"{'same-class dist²':>17} {'diff-class dist²':>17}", flush=True)
    print("-" * 80, flush=True)
    for b_val in SPECIFIC_B:
        acc1, acct, accd, ms, md = evaluate_loo(A, B, Y, K=K, filter_b=b_val)
        print(f"  {b_val:>2} | {acc1*100:>6.1f}% {acct*100:>9.1f}% {accd*100:>10.1f}% | "
              f"{ms:>17.1f} {md:>17.1f}", flush=True)
    print(flush=True)

    # Combined LOO across all b values, K sweep
    print("=== Combined LOO (all b values) ===", flush=True)
    print(f"{'K':>3} | {'1-NN':>7} {'Top-K sum':>10} {'Dist match':>11} | "
          f"{'Dist vs TopK':>13}", flush=True)
    print("-" * 55, flush=True)
    best_dist = 0.0
    best_k    = 1
    for K in K_SWEEP:
        acc1, acct, accd, _, _ = evaluate_loo(A, B, Y, K=K)
        delta = accd - acct
        print(f"  {K:>1} | {acc1*100:>6.1f}% {acct*100:>9.1f}% {accd*100:>10.1f}% | "
              f"{delta*100:>+12.1f}pp", flush=True)
        if accd > best_dist:
            best_dist = accd
            best_k = K

    elapsed = time.time() - t0

    # Summary
    acc1_best, acct_best, accd_best, _, _ = evaluate_loo(A, B, Y, K=best_k)
    print(flush=True)
    print("=" * 60, flush=True)
    print("STEP 296 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"Best K: {best_k}", flush=True)
    print(f"1-NN LOO:              {acc1_best*100:.1f}%", flush=True)
    print(f"Top-K sum LOO:         {acct_best*100:.1f}%", flush=True)
    print(f"Distribution matching: {accd_best*100:.1f}%", flush=True)
    print(f"Step 286 baseline:     41.8% (thermometer 1-NN)", flush=True)
    print(flush=True)
    print("KILL CRITERION (Leo mail 1341):", flush=True)
    if accd_best <= acct_best:
        print(f"  KILLED — dist match ({accd_best*100:.1f}%) <= top-K sum ({acct_best*100:.1f}%)",
              flush=True)
        print(f"  Sorted distribution structure adds no value over sum.", flush=True)
    else:
        delta = accd_best - acct_best
        delta_base = accd_best - 0.418
        print(f"  PASSES — dist match ({accd_best*100:.1f}%) > top-K sum ({acct_best*100:.1f}%) "
              f"by {delta*100:+.1f}pp", flush=True)
        print(f"  vs Step 286 baseline: {delta_base*100:+.1f}pp", flush=True)
        if accd_best > 0.418:
            print(f"  Beats Step 286 baseline.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)

if __name__ == '__main__':
    main()
