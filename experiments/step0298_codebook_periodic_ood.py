#!/usr/bin/env python3
"""
Step 298 -- Periodic OOD via congruence class lookup.

Step 297 showed OOD failure: 18% on a in 21..50 (random chance).
Root cause: OOD points have only one-sided training neighbors → φ vectors
are asymmetric → discriminative structure collapses.

Hypothesis: a%b is periodic with period b. If the model detects this from
training data, it can map OOD point a_ood → training point a_in ≡ a_ood (mod b).
Then use the training φ for prediction.

Two readout strategies:
  A. Exact congruence: find a_in = ((a_ood - 1) % b) + 1 in 1..b, use its φ
  B. KNN with periodicity: for OOD a, embed as a % b (its congruence class),
     find k nearest training examples with same congruence class and same b

The experiment asks: does knowing the period suffice to generalize OOD?

Kill criterion: periodic OOD < 50% (barely better than random near threshold)
Success: periodic OOD within 10pp of in-distribution (86.8%)
"""

import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
OOD_MIN   = 21
OOD_MAX   = 50
K         = 5
SENTINEL  = (OOD_MAX + TRAIN_MAX) * 2

IN_DIST_ACC = 0.868   # Step 296 reference

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_train():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


def build_ood():
    A, B, Y = [], [], []
    for a in range(OOD_MIN, OOD_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)

# ─── φ computation (same as Step 296) ────────────────────────────────────────

def compute_phi(query_a, query_b, A_tr, B_tr, Y_tr, exclude_idx, K, max_class):
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b = (B_tr == query_b)
    for c in range(max_class):
        mask = (Y_tr == c) & same_b
        if exclude_idx is not None and exclude_idx < len(mask) and mask[exclude_idx]:
            if Y_tr[exclude_idx] == c:
                mask = mask.copy(); mask[exclude_idx] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            phi[c * K:(c + 1) * K] = SENTINEL
            continue
        dists = np.abs(A_tr[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 298 -- Periodic OOD via congruence class lookup", flush=True)
    print(f"Train: a,b in 1..{TRAIN_MAX} (400 vectors)", flush=True)
    print(f"OOD:   a in {OOD_MIN}..{OOD_MAX}, b in 1..{TRAIN_MAX} (600 test pairs)", flush=True)
    print(f"In-distribution reference (Step 296): {IN_DIST_ACC*100:.1f}%", flush=True)
    print(flush=True)

    A_tr, B_tr, Y_tr = build_train()
    A_ood, B_ood, Y_ood = build_ood()
    max_class = max(int(Y_tr.max()), int(Y_ood.max())) + 1
    n_tr = len(A_tr)

    # Precompute training phi (self-excluded, same as Step 296)
    print("Precomputing training phi vectors...", flush=True)
    phi_train = np.zeros((n_tr, max_class * K), dtype=np.float32)
    for i in range(n_tr):
        phi_train[i] = compute_phi(A_tr[i], B_tr[i], A_tr, B_tr, Y_tr,
                                   exclude_idx=i, K=K, max_class=max_class)

    print(flush=True)

    # ─── Strategy A: Exact congruence mapping ─────────────────────────────────
    # For OOD a_ood: find the unique a_in in 1..b with a_in ≡ a_ood (mod b)
    # That is: a_in = ((a_ood - 1) % b) + 1 (maps to 1..b range)
    # Use the phi vector of (a_in, b) directly.
    print("=== Strategy A: Exact congruence mapping ===", flush=True)
    print("Map a_ood -> a_in = ((a_ood-1) % b) + 1, use phi(a_in, b)", flush=True)
    print(flush=True)

    n_ood = len(A_ood)
    correct_a = 0
    per_b_a = {b: {'correct': 0, 'total': 0} for b in range(1, TRAIN_MAX + 1)}

    for i in range(n_ood):
        a_ood, b, true_y = int(A_ood[i]), int(B_ood[i]), int(Y_ood[i])
        # Map to canonical representative in 1..b
        # a%b: same for a_ood and a_ood - b, so pick within training range
        a_in = ((a_ood - 1) % b) + 1   # gives 1..b
        # Find phi of (a_in, b) in training codebook
        tr_idx = np.where((A_tr == a_in) & (B_tr == b))[0]
        if len(tr_idx) == 0:
            continue
        phi_q = phi_train[tr_idx[0]]
        # NN in phi space over all training vectors
        diffs = phi_train - phi_q
        dists2 = (diffs * diffs).sum(axis=1)
        dists2[tr_idx[0]] = float('inf')  # exclude the anchor itself
        pred = int(Y_tr[np.argmin(dists2)])
        if pred == true_y:
            correct_a += 1
            per_b_a[b]['correct'] += 1
        per_b_a[b]['total'] += 1

    acc_a = correct_a / n_ood
    print(f"Strategy A OOD accuracy: {acc_a*100:.1f}%", flush=True)
    print(f"OOD drop vs in-distribution: {(acc_a - IN_DIST_ACC)*100:+.1f}pp", flush=True)
    print(flush=True)

    print("Per-b accuracy (Strategy A):", flush=True)
    per_b_a_accs = []
    for b_val in sorted(per_b_a.keys()):
        n = per_b_a[b_val]['total']
        if n == 0:
            continue
        acc = per_b_a[b_val]['correct'] / n
        per_b_a_accs.append(acc)
        print(f"  b={b_val:>2}: {acc*100:.1f}% ({per_b_a[b_val]['correct']}/{n})",
              flush=True)
    print(flush=True)

    # ─── Strategy B: Congruence-shifted phi ───────────────────────────────────
    # More general: compute phi with distances from a_ood to training a-values,
    # but shift distances as if a_ood is "located" at its congruence class.
    # For each class c and b, compute distances as |a_ood - (a_tr - k*b)| for
    # the best shift k. Equivalent to: dist = (a_ood - a_tr) % b (mod-b distance).
    print("=== Strategy B: Mod-b distance (circular) ===", flush=True)
    print("dist(a_ood, a_tr) = min over k of |a_ood - a_tr + k*b|", flush=True)
    print(flush=True)

    correct_b = 0
    per_b_b = {b: {'correct': 0, 'total': 0} for b in range(1, TRAIN_MAX + 1)}

    for i in range(n_ood):
        a_ood, b, true_y = int(A_ood[i]), int(B_ood[i]), int(Y_ood[i])
        # Compute phi with circular (mod-b) distances
        phi_q = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
        same_b_mask = (B_tr == b)
        for c in range(max_class):
            mask = (Y_tr == c) & same_b_mask
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            # Circular distance: min over all shifts k*b
            raw_diffs = a_ood - A_tr[idxs]
            # mod-b circular distance: min(raw % b, b - raw % b) ... actually
            # we want the minimum |a_ood - a_tr + k*b| over all integer k
            circular_dists = np.abs(raw_diffs % b).astype(np.float32)
            # Also consider the complement
            circular_dists = np.minimum(circular_dists, b - circular_dists)
            circular_dists = circular_dists.astype(np.float32)
            circular_dists.sort()
            k_eff = min(K, len(circular_dists))
            phi_q[c * K: c * K + k_eff] = circular_dists[:k_eff]

        # Compute training phi with the same circular distances
        # We need phi_train_circular for comparison
        # But we precomputed phi_train with raw distances.
        # For training points (in-dist), raw = circular when min dist < b/2.
        # Use raw phi_train for NN (approximate), note this is a confound.
        # For now use it as-is — the phi_q is circular, phi_train is raw.
        # This tests whether the circular query matches raw training patterns.
        diffs = phi_train - phi_q
        dists2 = (diffs * diffs).sum(axis=1)
        pred = int(Y_tr[np.argmin(dists2)])
        if pred == true_y:
            correct_b += 1
            per_b_b[b]['correct'] += 1
        per_b_b[b]['total'] += 1

    acc_b = correct_b / n_ood
    print(f"Strategy B OOD accuracy: {acc_b*100:.1f}%", flush=True)
    print(f"OOD drop vs in-distribution: {(acc_b - IN_DIST_ACC)*100:+.1f}pp", flush=True)
    print(flush=True)

    print("Per-b accuracy (Strategy B):", flush=True)
    per_b_b_accs = []
    for b_val in sorted(per_b_b.keys()):
        n = per_b_b[b_val]['total']
        if n == 0:
            continue
        acc = per_b_b[b_val]['correct'] / n
        per_b_b_accs.append(acc)
        print(f"  b={b_val:>2}: {acc*100:.1f}% ({per_b_b[b_val]['correct']}/{n})",
              flush=True)
    print(flush=True)

    # ─── Periodicity verification ─────────────────────────────────────────────
    print("=== Periodicity check (train data) ===", flush=True)
    print("Verify: phi(a,b) == phi(a+b,b) for in-distribution pairs.", flush=True)
    # For b=3, a in 1..17: check phi(a,3) vs phi(a+3,3)
    b_check = 3
    K_check = 5
    max_class_check = max_class
    same_dists = []
    for a in range(1, TRAIN_MAX - b_check + 1):
        i1 = np.where((A_tr == a) & (B_tr == b_check))[0][0]
        i2 = np.where((A_tr == a + b_check) & (B_tr == b_check))[0][0]
        d2 = float(((phi_train[i1] - phi_train[i2])**2).sum())
        same_dists.append((a, a + b_check, d2))

    exact_periodic = sum(1 for _, _, d in same_dists if d < 1e-3)
    print(f"b={b_check}: {exact_periodic}/{len(same_dists)} pairs have phi(a,b)==phi(a+b,b) [dist²<1e-3]",
          flush=True)
    if len(same_dists) <= 10:
        for a, a2, d in same_dists:
            print(f"  phi({a},{b_check}) vs phi({a2},{b_check}): dist²={d:.1f}", flush=True)
    else:
        print("  Sample (first 5):", flush=True)
        for a, a2, d in same_dists[:5]:
            print(f"  phi({a},{b_check}) vs phi({a2},{b_check}): dist²={d:.1f}", flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 298 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"In-distribution (Step 296, K=5):     {IN_DIST_ACC*100:.1f}%", flush=True)
    print(f"Step 297 OOD (raw distances):          18.0%  (random chance)", flush=True)
    print(f"Strategy A (exact congruence map):     {acc_a*100:.1f}%", flush=True)
    print(f"Strategy B (circular/mod-b distance):  {acc_b*100:.1f}%", flush=True)
    print(flush=True)

    best_acc = max(acc_a, acc_b)
    kill_threshold = 0.50

    print("KILL CRITERION:", flush=True)
    if best_acc < kill_threshold:
        print(f"  KILLED — best OOD ({best_acc*100:.1f}%) < {kill_threshold*100:.0f}%",
              flush=True)
        print(f"  Periodicity alone does not fix OOD for distribution matching.",
              flush=True)
    else:
        gap = best_acc - IN_DIST_ACC
        print(f"  PASSES — best OOD ({best_acc*100:.1f}%) >= {kill_threshold*100:.0f}%",
              flush=True)
        print(f"  Gap from in-distribution: {gap*100:+.1f}pp", flush=True)
        if best_acc >= IN_DIST_ACC - 0.10:
            print(f"  STRONG PASS — within 10pp of in-distribution!", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
