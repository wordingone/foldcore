#!/usr/bin/env python3
"""
Step 312 -- Two-level matching with self-discovered b-partition. Spec.

Level 1: 1-NN on full codebook (LOO) → winner v1 → extract b1 = v1's b-value.
Level 2: restrict codebook to b==b1. Apply three readouts:
  A. 1-NN within partition
  B. Top-K within partition (K=3,5)
  C. Phi within partition (per-class sorted top-K, Step 296 protocol)

Kill: all partitioned readouts <= unpartitioned equivalents.
Success: any partitioned readout > unpartitioned by 10pp.

Unpartitioned baselines from prior steps:
  1-NN: 26.0% (Step 311)
  phi: 86.8% (Step 296)
"""

import time
import numpy as np

TRAIN_MAX = 20
K_PHI = 5
MAX_CLASS = 20
PHI_DIM = MAX_CLASS * K_PHI   # 100
SENTINEL = float(TRAIN_MAX * 3)
K_VALS = [3, 5]


def build_data():
    X_list, A_list, B_list, Y_list = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            X_list.append([a / TRAIN_MAX, b / TRAIN_MAX])
            A_list.append(a)
            B_list.append(b)
            Y_list.append(a % b)
    return (np.array(X_list, dtype=np.float32),
            np.array(A_list, dtype=np.int32),
            np.array(B_list, dtype=np.int32),
            np.array(Y_list, dtype=np.int32))


def phi_in_partition(a_q, A_part, Y_part, excl=-1):
    """
    Phi for query a_q within a b-partition.
    A_part, Y_part: integer a-values and labels for entries in partition.
    excl: index within partition to exclude (LOO). -1 = no exclusion.
    Returns: PHI_DIM float32 vector.
    """
    phi = np.full(PHI_DIM, SENTINEL, dtype=np.float32)
    for c in range(MAX_CLASS):
        mask = (Y_part == c)
        if excl >= 0 and mask[excl]:
            mask = mask.copy()
            mask[excl] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_part[idxs] - a_q).astype(np.float32)
        dists.sort()
        k_eff = min(K_PHI, len(idxs))
        phi[c * K_PHI: c * K_PHI + k_eff] = dists[:k_eff]
    return phi


def run_312_loo(X, A, B, Y):
    n = len(Y)
    n_correct_b1 = 0  # level-1 b-match rate

    correct_A = 0
    correct_B = {k: 0 for k in K_VALS}
    correct_C = 0

    for i in range(n):
        # Level 1: full codebook NN (LOO exclude i)
        dists_full = np.linalg.norm(X - X[i], axis=1)
        dists_full[i] = float('inf')
        v1_idx = int(np.argmin(dists_full))
        b1 = int(B[v1_idx])

        if b1 == B[i]:
            n_correct_b1 += 1

        # b-partition: all entries with B==b1, excluding i
        part_mask = (B == b1)
        part_idxs = np.where(part_mask)[0]
        part_idxs = part_idxs[part_idxs != i]

        if len(part_idxs) == 0:
            continue

        A_part = A[part_idxs]
        Y_part = Y[part_idxs]
        X_part = X[part_idxs]

        # Distances from query to partition entries
        d_part = np.linalg.norm(X_part - X[i], axis=1)

        # Readout A: 1-NN within partition
        nn_a = int(np.argmin(d_part))
        if Y_part[nn_a] == Y[i]:
            correct_A += 1

        # Readout B: top-K within partition
        for k in K_VALS:
            k_eff = min(k, len(part_idxs))
            top_k = np.argsort(d_part)[:k_eff]
            counts = np.bincount(Y_part[top_k], minlength=MAX_CLASS)
            pred = int(np.argmax(counts))
            if pred == Y[i]:
                correct_B[k] += 1

        # Readout C: phi within partition
        # Compute phi_q (for query i, excl=-1 since i not in partition)
        a_q = int(A[i])
        phi_q = phi_in_partition(a_q, A_part, Y_part, excl=-1)

        # Compute phi for each entry in partition (LOO: exclude self within partition)
        best_dist = float('inf')
        best_label = -1
        for j_local in range(len(part_idxs)):
            phi_j = phi_in_partition(int(A_part[j_local]), A_part, Y_part, excl=j_local)
            diff = phi_q - phi_j
            d = float((diff * diff).sum())
            if d < best_dist:
                best_dist = d
                best_label = int(Y_part[j_local])

        if best_label == Y[i]:
            correct_C += 1

    b_match_rate = n_correct_b1 / n
    return {
        'b_match_rate': b_match_rate,
        'A': correct_A / n,
        'B': {k: correct_B[k] / n for k in K_VALS},
        'C': correct_C / n,
    }


def main():
    t0 = time.time()
    print("Step 312 -- Two-level matching with self-discovered b-partition", flush=True)
    print(f"Unpartitioned baselines: 1-NN=26.0%, phi=86.8%", flush=True)
    print(flush=True)

    X, A, B, Y = build_data()

    print("Running LOO...", flush=True)
    res = run_312_loo(X, A, B, Y)

    print(f"  Level-1 b-match rate: {res['b_match_rate']*100:.1f}%", flush=True)
    print(flush=True)

    print("=== Readout results ===", flush=True)
    print(f"  A. 1-NN within partition:       {res['A']*100:.1f}%  "
          f"(baseline: 26.0%  delta: {(res['A']-0.260)*100:+.1f}pp)", flush=True)
    for k in K_VALS:
        base = 0.260  # approx; no direct prior for K-NN unpartitioned
        print(f"  B. Top-{k} within partition:      {res['B'][k]*100:.1f}%  "
              f"(delta vs 1-NN: {(res['B'][k]-res['A'])*100:+.1f}pp)", flush=True)
    print(f"  C. Phi within partition:        {res['C']*100:.1f}%  "
          f"(baseline: 86.8%  delta: {(res['C']-0.868)*100:+.1f}pp)", flush=True)
    print(flush=True)

    # Kill/success
    unpart_1nn = 0.260
    unpart_phi = 0.868
    partitioned_vals = [res['A'], res['C']] + list(res['B'].values())
    unpart_best = max(unpart_1nn, unpart_phi)

    kill = all(v <= max(unpart_1nn, unpart_phi) for v in partitioned_vals)
    success = any(v > unpart_1nn + 0.10 for v in [res['A']] + list(res['B'].values())) or \
              res['C'] > unpart_phi + 0.10

    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 312 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Level-1 b-match rate:   {res['b_match_rate']*100:.1f}%", flush=True)
    print(f"A. 1-NN partitioned:    {res['A']*100:.1f}%  (unpart: 26.0%)", flush=True)
    for k in K_VALS:
        print(f"B. Top-{k} partitioned:  {res['B'][k]*100:.1f}%", flush=True)
    print(f"C. Phi partitioned:     {res['C']*100:.1f}%  (unpart: 86.8%)", flush=True)
    print(flush=True)
    print(f"Kill (all partitioned <= unpartitioned): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (any partitioned > unpartitioned + 10pp): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Self-discovered partition improves accuracy.", flush=True)
    elif kill:
        print("KILLED -- Partition adds nothing.", flush=True)
    else:
        print("PARTIAL -- Some improvement but below success threshold.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
