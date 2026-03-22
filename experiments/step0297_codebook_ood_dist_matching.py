#!/usr/bin/env python3
"""
Step 297 -- OOD Generalization Test for Per-class Distribution Matching.

Spec. Hypothesis: per-class distribution matching generalizes OOD
because the distribution shift is uniform across classes.

Training codebook: a,b in 1..20 (400 vectors, same as Step 296)
OOD test set: a in 21..50, b in 1..20 (600 pairs, OOD a, in-distribution b)

Prediction: compute phi(x) for OOD x using training codebook, find nearest
training phi in distribution space.

Kill criterion: OOD accuracy < in-distribution accuracy (86.8%) - 20pp = 66.8%
Success: OOD within 20pp of in-distribution.
"""

import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
OOD_MIN   = 21
OOD_MAX   = 50
K         = 5            # best K from Step 296
SENTINEL  = (OOD_MAX + TRAIN_MAX) * 2   # large distance for missing class

IN_DIST_ACC = 0.868   # Step 296 reference

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_train():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


def build_ood():
    """OOD: a in 21..50, b in 1..20."""
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
    print("Step 297 -- OOD Generalization, Distribution Matching", flush=True)
    print(f"Train: a,b in 1..{TRAIN_MAX} (400 vectors)", flush=True)
    print(f"OOD:   a in {OOD_MIN}..{OOD_MAX}, b in 1..{TRAIN_MAX} (600 test pairs)", flush=True)
    print(f"K={K}, same as Step 296 best", flush=True)
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

    # Evaluate OOD
    print("Evaluating OOD test set...", flush=True)
    n_ood = len(A_ood)
    correct = 0

    # Per-b accuracy
    per_b = {}
    for b_val in range(1, TRAIN_MAX + 1):
        per_b[b_val] = {'correct': 0, 'total': 0}

    for i in range(n_ood):
        a, b, true_y = int(A_ood[i]), int(B_ood[i]), int(Y_ood[i])
        phi_q = compute_phi(a, b, A_tr, B_tr, Y_tr, exclude_idx=None,
                            K=K, max_class=max_class)
        # NN in phi space (over training codebook)
        diffs = phi_train - phi_q
        dists2 = (diffs * diffs).sum(axis=1)
        pred = int(Y_tr[np.argmin(dists2)])
        if pred == true_y:
            correct += 1
            per_b[b]['correct'] += 1
        per_b[b]['total'] += 1

    ood_acc = correct / n_ood
    print(flush=True)

    # Per-b breakdown
    print("=== OOD per-b accuracy ===", flush=True)
    per_b_accs = []
    for b_val in sorted(per_b.keys()):
        n = per_b[b_val]['total']
        if n == 0:
            continue
        acc = per_b[b_val]['correct'] / n
        per_b_accs.append(acc)
        print(f"  b={b_val:>2}: {acc*100:.1f}% ({per_b[b_val]['correct']}/{n})",
              flush=True)
    print(flush=True)

    # Summary
    elapsed = time.time() - t0
    print("=" * 60, flush=True)
    print("STEP 297 SUMMARY", flush=True)
    print("=" * 60, flush=True)
    print(f"In-distribution (Step 296, K=5): {IN_DIST_ACC*100:.1f}%", flush=True)
    print(f"OOD accuracy (a in 21..50):       {ood_acc*100:.1f}%", flush=True)
    print(f"OOD drop vs in-distribution:      {(ood_acc - IN_DIST_ACC)*100:+.1f}pp",
          flush=True)
    print(f"Mean per-b OOD accuracy:          {np.mean(per_b_accs)*100:.1f}%", flush=True)
    print(flush=True)
    print("KILL CRITERION:", flush=True)
    kill_threshold = IN_DIST_ACC - 0.20
    if ood_acc < kill_threshold:
        print(f"  KILLED — OOD ({ood_acc*100:.1f}%) < {kill_threshold*100:.1f}% threshold",
              flush=True)
        print(f"  Distribution matching does not generalize OOD.", flush=True)
    else:
        print(f"  PASSES — OOD within 20pp of in-distribution.", flush=True)
        if ood_acc >= IN_DIST_ACC:
            print(f"  STRONG PASS — OOD >= in-distribution!", flush=True)
    print(f"\nElapsed: {elapsed:.2f}s", flush=True)

if __name__ == '__main__':
    main()
