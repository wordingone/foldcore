#!/usr/bin/env python3
"""
The Automated Discovery-Prescription Loop

The system that runs the loop without a human in it.

Each turn:
1. Train substrate (phi matching with current weight schedule)
2. Learn w (upweight same-class matches, downweight cross-class)
3. Analyze w (find the structure: which k-indices matter? which class patterns?)
4. Prescribe (update weight schedule from analysis)
5. Evaluate (LOO accuracy — did it improve?)
6. If improved: lock discovery, next turn. If not: saturated, stop.

The loop IS the search. Each turn shrinks the frozen frame by one discovery.
"""

import numpy as np
import time

TRAIN_MAX = 20
K = 5
SENTINEL = TRAIN_MAX * 3
MAX_TURNS = 20


def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


def compute_phi(query_a, query_b, A, B, Y, exclude_idx, K, max_class):
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b_mask = (B == query_b)
    for c in range(max_class):
        class_mask = (Y == c) & same_b_mask
        if exclude_idx is not None and exclude_idx < len(A) and class_mask[exclude_idx]:
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


# === Step 1: Train substrate (phi matching with weights) ===

def loo_with_weights(A, B, Y, weights, K):
    """LOO accuracy using weighted phi distance."""
    n = len(A)
    max_class = int(Y.max()) + 1
    dim = max_class * K

    # Precompute all phi (LOO-excluded)
    all_phi = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)

    correct = 0
    for i in range(n):
        phi_q = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)
        # Weighted distance to all
        diffs = all_phi - phi_q  # (n, dim)
        # Apply per-k weights: weights[k] applies to positions c*K+k for all c
        w_expanded = np.tile(weights, max_class)  # repeat weights for each class
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        dists[i] = float('inf')
        best_j = np.argmin(dists)
        if Y[best_j] == Y[i]:
            correct += 1
    return correct / n


# === Step 2: Learn w (one pass of weight learning) ===

def learn_weights(A, B, Y, current_weights, K, lr_w=0.1, epochs=10):
    """Learn weights by upweighting same-class matches, downweighting cross-class."""
    n = len(A)
    max_class = int(Y.max()) + 1
    dim = max_class * K

    # Precompute all phi
    all_phi = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)

    w = current_weights.copy()

    for epoch in range(epochs):
        order = np.random.permutation(n)
        for i in order:
            phi_q = all_phi[i]
            diffs = all_phi - phi_q
            w_expanded = np.tile(w, max_class)
            dists = (diffs ** 2 * w_expanded).sum(axis=1)
            dists[i] = float('inf')
            best_j = np.argmin(dists)

            if Y[best_j] == Y[i]:
                # Same class: these weights are good, slight reinforcement
                pass
            else:
                # Cross class: upweight dimensions where they DIFFER most
                diff_sq = (phi_q - all_phi[best_j]) ** 2
                # Per-k aggregation: average across classes for each k
                per_k_signal = np.zeros(K)
                for k in range(K):
                    indices = [c * K + k for c in range(max_class)]
                    per_k_signal[k] = diff_sq[indices].mean()
                # Upweight k-indices with largest cross-class difference
                w += lr_w * per_k_signal
                w = np.maximum(w, 0.01)  # keep positive

    # Normalize
    w = w / w.sum() * K
    return w


# === Step 3: Analyze w ===

def analyze_weights(w, K):
    """Analyze the learned weight vector for structure."""
    analysis = {}
    analysis['weights'] = w.copy()
    analysis['ranking'] = np.argsort(-w)  # highest first
    analysis['top_k'] = analysis['ranking'][0]
    analysis['ratio_top_to_mean'] = w[analysis['ranking'][0]] / w.mean()
    analysis['monotonic_decreasing'] = all(w[i] >= w[i+1] for i in range(K-1))

    # Derive a prescribed schedule from the learned structure
    # Fit: w_k ≈ a * exp(-b * k) — find best b
    if w[0] > 0 and w[-1] > 0:
        b_est = np.log(w[0] / w[-1]) / (K - 1)
        prescribed = np.array([w[0] * np.exp(-b_est * k) for k in range(K)])
        prescribed = prescribed / prescribed.sum() * K
    else:
        prescribed = w.copy()

    analysis['prescribed_schedule'] = prescribed
    return analysis


# === Step 4-5: Prescribe and Evaluate ===

def run_loop():
    """The automated discovery-prescription loop."""
    t0 = time.time()
    A, B, Y = build_dataset()

    print("=" * 60)
    print("THE AUTOMATED DISCOVERY-PRESCRIPTION LOOP")
    print("=" * 60)
    print(f"Dataset: a%b, (a,b) in 1..{TRAIN_MAX}")
    print(f"K={K}, max turns={MAX_TURNS}")
    print()

    # Initial physics: uniform weights
    weights = np.ones(K, dtype=np.float64)
    best_acc = loo_with_weights(A, B, Y, weights, K)
    print(f"Turn 0 (uniform): LOO = {best_acc*100:.1f}%")
    print(f"  weights: {np.round(weights, 3).tolist()}")
    print()

    history = [(0, best_acc, weights.copy(), 'uniform')]

    for turn in range(1, MAX_TURNS + 1):
        print(f"--- Turn {turn} ---")

        # Step 2: Learn w from current physics
        learned_w = learn_weights(A, B, Y, weights, K, lr_w=0.05, epochs=5)

        # Step 3: Analyze
        analysis = analyze_weights(learned_w, K)
        print(f"  Learned w: {np.round(learned_w, 3).tolist()}")
        print(f"  Top k-index: k={analysis['top_k']}")
        print(f"  Ratio top/mean: {analysis['ratio_top_to_mean']:.2f}x")
        print(f"  Monotonic decreasing: {analysis['monotonic_decreasing']}")

        # Step 4: Prescribe — use the analyzed schedule
        prescribed = analysis['prescribed_schedule']
        print(f"  Prescribed: {np.round(prescribed, 3).tolist()}")

        # Step 5: Evaluate prescribed weights
        new_acc = loo_with_weights(A, B, Y, prescribed, K)
        delta = new_acc - best_acc
        print(f"  LOO: {new_acc*100:.1f}% (delta: {delta*100:+.1f}pp)")

        # Also evaluate the raw learned weights
        learned_acc = loo_with_weights(A, B, Y, learned_w, K)
        print(f"  Learned w LOO: {learned_acc*100:.1f}%")

        # Step 6: Lock or stop
        if new_acc > best_acc + 0.001:
            print(f"  >> IMPROVED. Locking prescribed weights as new physics.")
            weights = prescribed.copy()
            best_acc = new_acc
            history.append((turn, new_acc, weights.copy(), 'prescribed'))
        elif learned_acc > best_acc + 0.001:
            print(f"  >> Prescribed didn't improve but learned did ({learned_acc*100:.1f}%).")
            print(f"  >> Locking learned weights (in-dist only, won't generalize OOD).")
            weights = learned_w.copy()
            best_acc = learned_acc
            history.append((turn, learned_acc, weights.copy(), 'learned'))
        else:
            print(f"  >> No improvement. Loop saturated at turn {turn}.")
            history.append((turn, new_acc, weights.copy(), 'saturated'))
            break
        print()

    elapsed = time.time() - t0

    # === Summary ===
    print()
    print("=" * 60)
    print("LOOP SUMMARY")
    print("=" * 60)
    print(f"Turns completed: {len(history) - 1}")
    print(f"Initial accuracy: {history[0][1]*100:.1f}% (uniform)")
    print(f"Final accuracy:   {history[-1][1]*100:.1f}% ({history[-1][3]})")
    print(f"Total improvement: {(history[-1][1] - history[0][1])*100:+.1f}pp")
    print()
    print("Turn history:")
    for turn, acc, w, source in history:
        print(f"  Turn {turn:>2}: {acc*100:.1f}% [{source}] w={np.round(w, 2).tolist()}")
    print()
    print(f"Final weights: {np.round(history[-1][2], 4).tolist()}")
    print(f"Elapsed: {elapsed:.1f}s")
    print()

    # === OOD test with final weights ===
    print("=== OOD test (a in 21..50) ===")
    A_ood, B_ood, Y_ood = [], [], []
    for a in range(21, 51):
        for b in range(1, TRAIN_MAX + 1):
            A_ood.append(a); B_ood.append(b); Y_ood.append(a % b)
    A_ood = np.array(A_ood); B_ood = np.array(B_ood); Y_ood = np.array(Y_ood)

    max_class = int(max(Y.max(), Y_ood.max())) + 1
    n_train = len(A)
    train_phi = np.zeros((n_train, max_class * K), dtype=np.float32)
    for i in range(n_train):
        train_phi[i] = compute_phi(A[i], B[i], A, B, Y, None, K, max_class)

    final_w = history[-1][2]
    w_expanded = np.tile(final_w, max_class)

    correct = 0
    n_test = len(A_ood)
    for i in range(n_test):
        phi_q = compute_phi(A_ood[i], B_ood[i], A, B, Y, None, K, max_class)
        diffs = train_phi - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        best_j = np.argmin(dists)
        if Y[best_j] == Y_ood[i]:
            correct += 1
    ood_acc = correct / n_test
    print(f"  OOD accuracy: {ood_acc*100:.1f}%")
    print(f"  (Step 297 baseline: 18.0%, Step 309 learned w: 17.3%)")
    print()
    print("Done.")


if __name__ == '__main__':
    np.random.seed(42)
    run_loop()
