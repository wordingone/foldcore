#!/usr/bin/env python3
"""
Step 313 — Loop Turn 2: Prescribe the substrate's discovery as fixed physics.

The substrate discovered k=0 importance (Step 308b). We prescribe decreasing
weights as fixed physics and test whether it generalizes OOD.

Turn 1: Human designed phi → 86.8%
Turn 2: Substrate discovered k=0 importance → prescribed as decreasing weights
"""

import numpy as np

TRAIN_MAX = 20
K = 5
SENTINEL = TRAIN_MAX * 3

def build_dataset(max_a=20):
    A, B, Y = [], [], []
    for a in range(1, max_a + 1):
        for b in range(1, max_a + 1):
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

def weighted_dist(phi_a, phi_b, weights, K, max_class):
    """Weighted squared distance between two phi vectors."""
    d = 0.0
    for c in range(max_class):
        for k in range(K):
            idx = c * K + k
            diff = phi_a[idx] - phi_b[idx]
            d += weights[k] * diff * diff
    return d

def loo_accuracy(A, B, Y, weights, K):
    n = len(A)
    max_class = int(Y.max()) + 1

    # Precompute all phi (LOO-excluded)
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        all_phi[i] = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)

    correct = 0
    for i in range(n):
        phi_q = compute_phi(A[i], B[i], A, B, Y, i, K, max_class)
        best_d = float('inf')
        best_y = -1
        for j in range(n):
            if j == i:
                continue
            d = weighted_dist(phi_q, all_phi[j], weights, K, max_class)
            if d < best_d:
                best_d = d
                best_y = Y[j]
        if best_y == Y[i]:
            correct += 1
    return correct / n

def ood_accuracy(A_train, B_train, Y_train, A_test, B_test, Y_test, weights, K):
    n_train = len(A_train)
    max_class = max(int(Y_train.max()), int(Y_test.max())) + 1

    # Precompute train phi (no exclusion)
    train_phi = np.zeros((n_train, max_class * K), dtype=np.float32)
    for i in range(n_train):
        train_phi[i] = compute_phi(A_train[i], B_train[i], A_train, B_train, Y_train,
                                    None, K, max_class)

    correct = 0
    n_test = len(A_test)
    for i in range(n_test):
        phi_q = compute_phi(A_test[i], B_test[i], A_train, B_train, Y_train,
                            None, K, max_class)
        best_d = float('inf')
        best_y = -1
        for j in range(n_train):
            d = weighted_dist(phi_q, train_phi[j], weights, K, max_class)
            if d < best_d:
                best_d = d
                best_y = Y_train[j]
        if best_y == Y_test[i]:
            correct += 1
    return correct / n_test

def main():
    print("Step 313 — Loop Turn 2: Prescribed weights from substrate discovery")
    print()

    A, B, Y = build_dataset(TRAIN_MAX)

    # Weight schedules to test
    schedules = {
        'uniform':     [1.0] * K,
        '1/(k+1)':     [1.0/(k+1) for k in range(K)],
        'exp(-k)':     [np.exp(-k) for k in range(K)],
        'exp(-2k)':    [np.exp(-2*k) for k in range(K)],
        'k0_only':     [1.0] + [0.0] * (K-1),
        'linear_decay': [1.0 - k/(K) for k in range(K)],
    }

    # === In-distribution LOO ===
    print("=== In-distribution LOO (a%b, 1..20) ===")
    print(f"{'Schedule':<15} | {'LOO':>7} | {'vs uniform':>10}")
    print("-" * 40)

    uniform_acc = None
    results = {}
    for name, w in schedules.items():
        acc = loo_accuracy(A, B, Y, w, K)
        results[name] = acc
        if name == 'uniform':
            uniform_acc = acc
        delta = acc - uniform_acc if uniform_acc else 0
        print(f"{name:<15} | {acc*100:>6.1f}% | {delta*100:>+9.1f}pp")

    # Best non-uniform schedule
    best_name = max((n for n in results if n != 'uniform'), key=lambda n: results[n])
    best_acc = results[best_name]

    print()
    print(f"Best: {best_name} ({best_acc*100:.1f}%)")
    print(f"Step 296 reference: 86.8% (uniform)")
    print(f"Step 308b reference: 91.2% (learned w)")
    print()

    # === OOD test with best schedule ===
    print("=== OOD (a in 21..50, b in 1..20) ===")
    A_ood, B_ood, Y_ood = [], [], []
    for a in range(21, 51):
        for b in range(1, TRAIN_MAX + 1):
            A_ood.append(a); B_ood.append(b); Y_ood.append(a % b)
    A_ood, B_ood, Y_ood = np.array(A_ood), np.array(B_ood), np.array(Y_ood)

    for name in ['uniform', best_name]:
        w = schedules[name]
        ood_acc = ood_accuracy(A, B, Y, A_ood, B_ood, Y_ood, w, K)
        print(f"  {name:<15}: OOD={ood_acc*100:.1f}%")

    print(f"  Step 297 reference: 18.0% (uniform OOD)")
    print(f"  Step 309 reference: 17.3% (learned w OOD)")
    print()

    # === floor(a/b) test ===
    print("=== floor(a/b) generalization ===")
    A_f, B_f, Y_f = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_f.append(a); B_f.append(b); Y_f.append(a // b)
    A_f, B_f, Y_f = np.array(A_f), np.array(B_f), np.array(Y_f)

    for name in ['uniform', best_name]:
        w = schedules[name]
        floor_acc = loo_accuracy(A_f, B_f, Y_f, w, K)
        print(f"  {name:<15}: LOO={floor_acc*100:.1f}%")

    print(f"  Step 302 reference: 86.8% (uniform)")
    print()

    # === Verdict ===
    print("=" * 50)
    if best_acc > uniform_acc + 0.001:
        print(f"LOOP TURN 2: prescribed weights ({best_name}) beat uniform")
        print(f"  In-dist: {best_acc*100:.1f}% vs {uniform_acc*100:.1f}%")
        print(f"  The substrate's discovery persists as physics.")
    else:
        print(f"LOOP TURN 2: no improvement from prescribed weights")
    print()

if __name__ == '__main__':
    main()
