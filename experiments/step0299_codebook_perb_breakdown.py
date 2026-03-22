#!/usr/bin/env python3
"""
Step 299 -- Full per-b accuracy breakdown for distribution matching (Step 296).

Step 296 reported overall 86.8% and checked b=3,4,5 only.
This step runs the full per-b sweep to reveal the sample coverage effect.

Hypothesis: accuracy decreases with b because larger b means fewer training
examples per class (a in 1..20 covers ~20/b periods, so ~1 example per class
for b >= 10). The phi mechanism degrades when class distributions are sparse.

Measures:
  - Per-b LOO accuracy for distribution matching (K=5)
  - Per-b: n_classes, mean examples per class, mean same-class dist^2
  - Correlation: accuracy vs examples_per_class

Kill criterion: none (diagnostic step — no kill needed)
"""

import time
import numpy as np

TRAIN_MAX = 20
K         = 5
SENTINEL  = TRAIN_MAX * 3

def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


def compute_phi(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
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


def main():
    t0 = time.time()
    print("Step 299 -- Full per-b breakdown for distribution matching", flush=True)
    print(f"Dataset: a,b in 1..{TRAIN_MAX}, K={K}", flush=True)
    print(f"Overall Step 296 accuracy: 86.8%", flush=True)
    print(flush=True)

    A, B, Y = build_dataset()
    max_class = int(Y.max()) + 1
    n = len(A)

    # Precompute all phi vectors (LOO)
    print("Precomputing phi vectors...", flush=True)
    phi_all = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        phi_all[i] = compute_phi(A[i], B[i], A, B, Y,
                                 exclude_idx=i, K=K, max_class=max_class)
    print("Done.", flush=True)
    print(flush=True)

    # Per-b evaluation
    print(f"{'b':>3} | {'n_cls':>5} | {'ex/cls':>6} | {'acc':>7} | {'margin_same':>12} | {'margin_diff':>12}",
          flush=True)
    print("-" * 65, flush=True)

    per_b_results = {}
    for b_val in range(1, TRAIN_MAX + 1):
        b_idxs = np.where(B == b_val)[0]
        n_b = len(b_idxs)

        # Classes present for this b
        classes_present = np.unique(Y[b_idxs])
        n_classes = len(classes_present)
        mean_ex_per_class = n_b / n_classes if n_classes > 0 else 0.0

        # LOO accuracy via distribution matching
        correct = 0
        for i in b_idxs:
            phi_q = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i,
                                K=K, max_class=max_class)
            diffs = phi_all - phi_q
            dists2 = (diffs * diffs).sum(axis=1)
            dists2[i] = float('inf')
            pred = int(Y[np.argmin(dists2)])
            if pred == Y[i]:
                correct += 1
        acc = correct / n_b

        # Margin: same-class vs diff-class distances in phi space (within same b)
        same_dists, diff_dists = [], []
        for ii in range(len(b_idxs)):
            for jj in range(ii + 1, len(b_idxs)):
                i1, i2 = b_idxs[ii], b_idxs[jj]
                d2 = float(((phi_all[i1] - phi_all[i2])**2).sum())
                if Y[i1] == Y[i2]:
                    same_dists.append(d2)
                else:
                    diff_dists.append(d2)
        ms = np.mean(same_dists) if same_dists else float('nan')
        md = np.mean(diff_dists) if diff_dists else float('nan')

        per_b_results[b_val] = {
            'n_classes': n_classes,
            'mean_ex_per_class': mean_ex_per_class,
            'acc': acc,
            'margin_same': ms,
            'margin_diff': md,
        }

        print(f"  {b_val:>1} | {n_classes:>5} | {mean_ex_per_class:>6.1f} | {acc*100:>6.1f}% | "
              f"{ms:>12.1f} | {md:>12.1f}", flush=True)

    print(flush=True)

    # Correlation: accuracy vs examples per class
    accs = [per_b_results[b]['acc'] for b in range(1, TRAIN_MAX + 1)]
    ex_per_cls = [per_b_results[b]['mean_ex_per_class'] for b in range(1, TRAIN_MAX + 1)]
    corr = np.corrcoef(ex_per_cls, accs)[0, 1]

    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 299 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Overall LOO accuracy (dist matching, K=5): {np.mean(accs)*100:.1f}%", flush=True)
    print(f"Best b:  b={np.argmax(accs)+1} at {max(accs)*100:.1f}%", flush=True)
    print(f"Worst b: b={np.argmin(accs)+1} at {min(accs)*100:.1f}%", flush=True)
    print(f"Correlation(examples_per_class, accuracy): {corr:.3f}", flush=True)
    print(flush=True)

    # Group by examples per class
    print("Accuracy by examples-per-class tier:", flush=True)
    tiers = [(1, 2), (2, 4), (4, 8), (8, 21)]
    for lo, hi in tiers:
        tier_bs = [b for b in range(1, TRAIN_MAX + 1)
                   if lo <= per_b_results[b]['mean_ex_per_class'] < hi]
        if not tier_bs:
            continue
        tier_accs = [per_b_results[b]['acc'] for b in tier_bs]
        print(f"  {lo:.0f}-{hi:.0f} ex/class (b={tier_bs}): {np.mean(tier_accs)*100:.1f}%",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
