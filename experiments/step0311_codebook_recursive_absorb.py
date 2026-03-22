#!/usr/bin/env python3
"""
Step 311 -- Recursive absorption. Spec.

Encoding: x = (a/20, b/20), 2D float.
Algorithm: peel residuals at each depth using same nearest-neighbor op.
  current = x
  for level in range(depth):
      idx = nearest(codebook, current)  [LOO: exclude self]
      matches.append(idx)
      current = current - codebook[idx]
  predict = majority_vote(labels[matches])

Kill: depth=10 LOO <= depth=1 LOO.
Success: depth>=3 LOO > depth=1 + 10pp.
Compare: depth=1 1-NN (~5%), phi 86.8%.
"""

import time
import numpy as np

TRAIN_MAX = 20
DEPTHS = [1, 2, 3, 5, 10]


def build_data():
    A_list, Y_list = [], []
    b_list = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_list.append([a / TRAIN_MAX, b / TRAIN_MAX])
            b_list.append(b)
            Y_list.append(a % b)
    X = np.array(A_list, dtype=np.float32)   # (400, 2)
    B = np.array(b_list, dtype=np.int32)      # (400,) original b values
    Y = np.array(Y_list, dtype=np.int32)      # (400,) labels
    return X, B, Y


def classify_loo(X, Y, excl, depth):
    """
    Recursive absorption for training example excl.
    Excludes excl at all levels (LOO).
    Returns: predicted label, list of matched indices, residual norms.
    """
    current = X[excl].copy()
    matches = []
    residual_norms = []

    for level in range(depth):
        dists = np.linalg.norm(X - current, axis=1)
        dists[excl] = float('inf')
        idx = int(np.argmin(dists))
        matches.append(idx)
        current = current - X[idx]
        residual_norms.append(float(np.linalg.norm(current)))

    # Majority vote
    labels = Y[np.array(matches)]
    counts = np.bincount(labels, minlength=TRAIN_MAX)
    pred = int(np.argmax(counts))
    return pred, matches, residual_norms


def loo_at_depth(X, B, Y, depth, track_structure=False):
    """
    LOO accuracy at given depth. If track_structure: also return
    b-match rate at level 1 and class-match rate at level 2+.
    """
    n = len(Y)
    correct = 0
    b_match_l1 = 0    # level-1 match has same b
    class_match_l2 = 0  # level-2 match has same class
    residual_norms_by_level = [[] for _ in range(depth)]

    for i in range(n):
        pred, matches, res_norms = classify_loo(X, Y, i, depth)
        if pred == Y[i]:
            correct += 1
        if track_structure:
            if B[matches[0]] == B[i]:
                b_match_l1 += 1
            if depth >= 2 and Y[matches[1]] == Y[i]:
                class_match_l2 += 1
            for lv, rn in enumerate(res_norms):
                residual_norms_by_level[lv].append(rn)

    acc = correct / n
    if track_structure:
        return acc, b_match_l1 / n, class_match_l2 / n if depth >= 2 else None, residual_norms_by_level
    return acc


def main():
    t0 = time.time()
    print("Step 311 -- Recursive absorption (depth sweep)", flush=True)
    print(f"Depths tested: {DEPTHS}", flush=True)
    print(f"Encoding: x = (a/20, b/20), 2D float", flush=True)
    print(f"Compare: depth=1 (~5%), phi 86.8%", flush=True)
    print(flush=True)

    X, B, Y = build_data()
    n = len(Y)

    results = {}
    for depth in DEPTHS:
        t_d = time.time()
        acc = loo_at_depth(X, B, Y, depth)
        results[depth] = acc
        print(f"  depth={depth:>2}: LOO={acc*100:.1f}%  [{time.time()-t_d:.2f}s]", flush=True)

    print(flush=True)

    # Detailed structure analysis at depth=3
    print("=== Structure analysis at depth=3 ===", flush=True)
    acc3, b_rate, c2_rate, res_norms = loo_at_depth(X, B, Y, 3, track_structure=True)
    print(f"  LOO: {acc3*100:.1f}%", flush=True)
    print(f"  Level-1 b-match rate: {b_rate*100:.1f}%  (same b as query)", flush=True)
    print(f"  Level-2 class-match rate: {c2_rate*100:.1f}%  (same class as query)", flush=True)
    for lv in range(3):
        norms = res_norms[lv]
        print(f"  Level-{lv+1} residual norm: mean={np.mean(norms):.4f}  "
              f"min={np.min(norms):.4f}  max={np.max(norms):.4f}", flush=True)
    print(flush=True)

    # Sample: show a few examples at depth=3
    print("=== Sample classifications (depth=3) ===", flush=True)
    print(f"  {'i':>3} {'a':>3} {'b':>3} {'y':>3} | {'pred':>4} | matches(class) | residuals", flush=True)
    for i in [0, 21, 42, 100, 200, 300, 399]:
        if i >= n:
            break
        a_val = int(round(X[i, 0] * TRAIN_MAX))
        b_val = int(round(X[i, 1] * TRAIN_MAX))
        pred, matches, res_norms = classify_loo(X, Y, i, 3)
        match_classes = [Y[m] for m in matches]
        res_str = " ".join(f"{r:.3f}" for r in res_norms)
        correct_mark = "OK" if pred == Y[i] else "XX"
        print(f"  {i:>3} {a_val:>3} {b_val:>3} {Y[i]:>3} | {pred:>4}{correct_mark}| {match_classes} | {res_str}", flush=True)
    print(flush=True)

    # Summary
    elapsed = time.time() - t0
    depth1_loo = results[1]
    depth3_loo = results.get(3, depth1_loo)
    depth10_loo = results[10]

    killed = depth10_loo <= depth1_loo
    success = any(results[d] > depth1_loo + 0.10 for d in DEPTHS if d >= 3)

    print("=" * 65, flush=True)
    print("STEP 311 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Depth':>6} | {'LOO':>7} | {'Delta vs depth=1':>16}", flush=True)
    print("-" * 38, flush=True)
    for d in DEPTHS:
        delta = results[d] - depth1_loo
        print(f"  {d:>4} | {results[d]*100:>6.1f}% | {delta*100:>+15.1f}pp", flush=True)
    print(flush=True)
    print(f"Kill (depth=10 <= depth=1): {'TRIGGERED' if killed else 'not triggered'}", flush=True)
    print(f"Success (depth>=3 > depth=1 + 10pp): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Residuals carry structure. Recursion discovers hierarchy.", flush=True)
    elif killed:
        print("KILLED -- Depth adds nothing. Residuals are uninformative.", flush=True)
    else:
        print("PARTIAL -- Some improvement but below success threshold.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
