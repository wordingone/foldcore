#!/usr/bin/env python3
"""
Step 302 -- Does phi generalize beyond a%b?

Two tests from RESEARCH_STATE.md active hypothesis:
  A. Larger range: phi on (a,b) in 1..50, LOO in-dist (same mechanism as Step 296)
  B. Different non-Lipschitz function: floor(a/b) on (a,b) in 1..20

floor(a/b) has contiguous class blocks (class c = a in {c*b..(c+1)*b-1}), not
arithmetic progressions. Different structural property than a%b.

Questions:
  1. Does phi accuracy hold for larger a,b ranges?
  2. Does phi work for floor(a/b) — a different non-Lipschitz function?
  3. If phi fails for floor(a/b), is it because of structure mismatch or readout?

Kill criterion for Test A: phi accuracy drops > 20pp when range doubles (1..20 -> 1..40).
Kill criterion for Test B: phi accuracy <= 1-NN for floor(a/b).
"""

import time
import numpy as np

# ─── Test A: Scaling to larger ranges ─────────────────────────────────────────

def compute_phi(query_a, query_b, A, B, Y, exclude_idx, K, max_class, sentinel):
    phi = np.full(max_class * K, float(sentinel), dtype=np.float32)
    same_b = (B == query_b)
    for c in range(max_class):
        mask = (Y == c) & same_b
        if exclude_idx is not None and exclude_idx < len(mask) and mask[exclude_idx]:
            if Y[exclude_idx] == c:
                mask = mask.copy(); mask[exclude_idx] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def evaluate_phi_loo(A, B, Y, K, max_class, sentinel, label="", max_n=None):
    """
    LOO evaluation: distribution matching + 1-NN.
    max_n: if set, use at most max_n vectors for codebook (speed limit).
    """
    n = len(A)
    if max_n is not None and n > max_n:
        # Sample randomly for speed
        idxs = np.random.choice(n, max_n, replace=False)
        A, B, Y = A[idxs], B[idxs], Y[idxs]
        n = max_n

    # Precompute phi (LOO)
    phi_all = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        phi_all[i] = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i,
                                 K=K, max_class=max_class, sentinel=sentinel)

    correct_phi = 0
    correct_1nn = 0

    for i in range(n):
        true_y = Y[i]

        # Distribution matching
        phi_q = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i,
                            K=K, max_class=max_class, sentinel=sentinel)
        diffs = phi_all - phi_q
        dists2 = (diffs * diffs).sum(axis=1)
        dists2[i] = float('inf')
        if Y[np.argmin(dists2)] == true_y:
            correct_phi += 1

        # 1-NN
        same_b = (B == B[i])
        same_b[i] = False
        same_b_idxs = np.where(same_b)[0]
        if len(same_b_idxs) > 0:
            dists_1nn = np.abs(A[same_b_idxs] - A[i])
            if Y[same_b_idxs[np.argmin(dists_1nn)]] == true_y:
                correct_1nn += 1

    return correct_phi / n, correct_1nn / n


def test_scaling():
    """Test A: phi accuracy for different training ranges."""
    print("=" * 65, flush=True)
    print("TEST A: Scaling — phi accuracy vs training range", flush=True)
    print("=" * 65, flush=True)
    print(f"Function: a%b", flush=True)
    print(f"K=5 (best from Step 296)", flush=True)
    print(flush=True)

    K = 5
    ranges = [10, 20, 30, 50]
    print(f"{'Range':>12} | {'N':>6} | {'1-NN':>7} | {'Phi':>7} | {'vs Step296':>11}",
          flush=True)
    print("-" * 55, flush=True)

    ref_acc = 0.868  # Step 296 reference

    for train_max in ranges:
        A, B, Y = [], [], []
        for a in range(1, train_max + 1):
            for b in range(1, train_max + 1):
                A.append(a); B.append(b); Y.append(a % b)
        A, B, Y = np.array(A), np.array(B), np.array(Y)
        max_class = int(Y.max()) + 1
        sentinel = train_max * 3

        t0 = time.time()
        acc_phi, acc_1nn = evaluate_phi_loo(A, B, Y, K, max_class, sentinel)
        elapsed = time.time() - t0

        delta = (acc_phi - ref_acc) * 100
        n = len(A)
        print(f"  1..{train_max:>2} x 1..{train_max:>2} | {n:>6} | {acc_1nn*100:>6.1f}% | "
              f"{acc_phi*100:>6.1f}% | {delta:>+10.1f}pp  [{elapsed:.1f}s]",
              flush=True)

    print(flush=True)


def test_floor_div():
    """Test B: phi on floor(a/b)."""
    print("=" * 65, flush=True)
    print("TEST B: Generalization — phi on floor(a/b)", flush=True)
    print("=" * 65, flush=True)
    print(f"Dataset: a,b in 1..20 (400 pairs)", flush=True)
    print(f"Function: floor(a/b) — contiguous class blocks, not arithmetic progressions", flush=True)
    print(flush=True)

    TRAIN_MAX = 20
    K = 5

    A, B, Y_mod, Y_floor = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b)
            Y_mod.append(a % b)
            Y_floor.append(a // b)

    A = np.array(A); B = np.array(B)
    Y_mod = np.array(Y_mod); Y_floor = np.array(Y_floor)

    max_class_mod = int(Y_mod.max()) + 1
    max_class_floor = int(Y_floor.max()) + 1
    sentinel_mod = TRAIN_MAX * 3
    sentinel_floor = TRAIN_MAX * 3

    print(f"a%b: {max_class_mod} classes, floor(a/b): {max_class_floor} classes",
          flush=True)
    print(flush=True)

    # Per-b analysis
    print(f"{'b':>3} | {'a%b phi':>8} | {'a%b 1nn':>8} | {'f(a/b) phi':>11} | {'f(a/b) 1nn':>11}",
          flush=True)
    print("-" * 55, flush=True)

    phi_mod_accs = []
    phi_floor_accs = []

    for b_val in [3, 5, 7, 10, 15, 20]:
        b_mask = (B == b_val)
        A_b, B_b = A[b_mask], B[b_mask]
        Y_mod_b = Y_mod[b_mask]
        Y_floor_b = Y_floor[b_mask]
        mc_mod = max_class_mod
        mc_fl = max_class_floor

        acc_phi_mod, acc_1nn_mod = evaluate_phi_loo(A_b, B_b, Y_mod_b, K, mc_mod, sentinel_mod)
        acc_phi_floor, acc_1nn_floor = evaluate_phi_loo(A_b, B_b, Y_floor_b, K, mc_fl, sentinel_floor)

        phi_mod_accs.append(acc_phi_mod)
        phi_floor_accs.append(acc_phi_floor)

        print(f"  {b_val:>1} | {acc_phi_mod*100:>7.1f}% | {acc_1nn_mod*100:>7.1f}% | "
              f"{acc_phi_floor*100:>10.1f}% | {acc_1nn_floor*100:>10.1f}%",
              flush=True)

    print(flush=True)

    # Full dataset comparison
    print("Full dataset (all b values):", flush=True)
    t0 = time.time()
    acc_phi_mod_full, acc_1nn_mod_full = evaluate_phi_loo(A, B, Y_mod, K, max_class_mod,
                                                          sentinel_mod)
    t1 = time.time()
    acc_phi_floor_full, acc_1nn_floor_full = evaluate_phi_loo(A, B, Y_floor, K, max_class_floor,
                                                              sentinel_floor)
    t2 = time.time()

    print(f"  a%b:      phi={acc_phi_mod_full*100:.1f}%, 1-NN={acc_1nn_mod_full*100:.1f}%  [{t1-t0:.1f}s]",
          flush=True)
    print(f"  floor(a/b): phi={acc_phi_floor_full*100:.1f}%, 1-NN={acc_1nn_floor_full*100:.1f}%  [{t2-t1:.1f}s]",
          flush=True)
    print(flush=True)

    # Structural comparison
    print("Structural comparison:", flush=True)
    print("  a%b: same-class vectors form arithmetic progressions (step=b)", flush=True)
    print("  floor(a/b): same-class vectors form contiguous blocks (width=b)", flush=True)
    print("  Both are non-Lipschitz at class boundaries, but different structure.", flush=True)
    print(flush=True)

    # Kill criterion
    phi_better_mod = acc_phi_mod_full > acc_1nn_mod_full
    phi_better_floor = acc_phi_floor_full > acc_1nn_floor_full
    return acc_phi_mod_full, acc_phi_floor_full, phi_better_floor


def main():
    t0 = time.time()
    print("Step 302 -- Generalization Test for Distribution Matching", flush=True)
    print("RESEARCH_STATE: does phi generalize to larger ranges + other functions?",
          flush=True)
    print(flush=True)

    test_scaling()
    acc_mod, acc_floor, floor_passes = test_floor_div()

    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 302 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Test A (scaling): see per-range breakdown above", flush=True)
    print(f"Test B (floor(a/b)): phi={acc_floor*100:.1f}% vs a%b phi={acc_mod*100:.1f}%",
          flush=True)
    print(flush=True)

    print("KILL CRITERIA:", flush=True)
    print("Test B:", flush=True)
    if floor_passes:
        print(f"  PASSES -- phi ({acc_floor*100:.1f}%) > 1-NN for floor(a/b)", flush=True)
        print(f"  Distribution matching generalizes beyond a%b.", flush=True)
    else:
        print(f"  KILLED -- phi ({acc_floor*100:.1f}%) <= 1-NN for floor(a/b)", flush=True)
        print(f"  The mechanism is specific to a%b structure.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
