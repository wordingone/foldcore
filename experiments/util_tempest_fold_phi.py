#!/usr/bin/env python3
"""
Tempest Fold + Phi Observer

Substrate: f = absorb (same as tempest_fold.py)
Observer: phi (per-class sorted top-K distances from Step 296)

The substrate builds the codebook. The observer reads the geometry.
Same codebook that gave 0% with label-dim observer. Phi should give 86.8%.
"""

import numpy as np

MAX_A = 20
N_CLASSES = MAX_A
K_VOTE = 5
SENTINEL = MAX_A * 3


def build_dataset():
    A, B, Y = [], [], []
    for a in range(1, MAX_A + 1):
        for b in range(1, MAX_A + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A), np.array(B), np.array(Y)


# === The Substrate (f = absorb) ===

class TempestFold:
    def __init__(self, alpha=0.01, spawn_radius=0.7):
        self.alpha = alpha
        self.spawn_radius = spawn_radius
        self.vectors = []   # list of raw (a, b, label) tuples absorbed
        self.state_a = []   # the a-values in codebook (for distance computation)
        self.state_b = []   # the b-values
        self.state_y = []   # the labels (absorbed from D, part of state geometry)

    def absorb(self, a, b, label):
        """f(State, D). Absorb input into state. No output."""
        if len(self.state_a) == 0:
            self.state_a.append(a)
            self.state_b.append(b)
            self.state_y.append(label)
            return

        # Find nearest same-b vector
        best_idx = -1
        best_dist = float('inf')
        for i in range(len(self.state_a)):
            if self.state_b[i] != b:
                continue
            d = abs(self.state_a[i] - a)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx == -1 or best_dist > (MAX_A * self.spawn_radius):
            # Nothing close — spawn
            self.state_a.append(a)
            self.state_b.append(b)
            self.state_y.append(label)
        else:
            # Absorb: blend position and label
            self.state_a[best_idx] = (1 - self.alpha) * self.state_a[best_idx] + self.alpha * a
            # Label blends too (part of state geometry)
            self.state_y[best_idx] = int(round(
                (1 - self.alpha) * self.state_y[best_idx] + self.alpha * label
            ))


# === The Observer (phi — per-class sorted top-K distances) ===

def compute_phi(query_a, query_b, cb_a, cb_b, cb_y, exclude_idx, K, max_class):
    """Compute per-class sorted top-K distance vector."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)

    for c in range(max_class):
        dists = []
        for i in range(len(cb_a)):
            if i == exclude_idx:
                continue
            if cb_b[i] != query_b:
                continue
            if cb_y[i] != c:
                continue
            dists.append(abs(cb_a[i] - query_a))

        dists.sort()
        k_eff = min(K, len(dists))
        for j in range(k_eff):
            phi[c * K + j] = dists[j]

    return phi


def predict_phi(query_a, query_b, cb_a, cb_b, cb_y, exclude_idx, K, max_class,
                all_phi, all_y):
    """Predict by NN in phi space."""
    phi_q = compute_phi(query_a, query_b, cb_a, cb_b, cb_y, exclude_idx, K, max_class)
    best_dist = float('inf')
    best_label = -1
    for j in range(len(all_phi)):
        if j == exclude_idx:
            continue
        d = float(np.sum((phi_q - all_phi[j]) ** 2))
        if d < best_dist:
            best_dist = d
            best_label = all_y[j]
    return best_label


def predict_1nn(query_a, query_b, cb_a, cb_b, cb_y, exclude_idx):
    """Standard 1-NN for comparison."""
    best_dist = float('inf')
    best_label = -1
    for i in range(len(cb_a)):
        if i == exclude_idx:
            continue
        if cb_b[i] != query_b:
            continue
        d = abs(cb_a[i] - query_a)
        if d < best_dist:
            best_dist = d
            best_label = cb_y[i]
    return best_label


def main():
    print("TEMPEST FOLD + PHI OBSERVER")
    print("Substrate: f = absorb")
    print("Observer: phi (per-class sorted top-K distances)")
    print()

    A, B, Y = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1

    # === Phase 1: Build substrate via absorption ===
    # Sweep spawn_radius to find the right compression level
    for sr in [0.02, 0.05, 0.1, 0.2, 0.5, 0.7]:
        sub_test = TempestFold(alpha=0.01, spawn_radius=sr)
        for i in range(n):
            sub_test.absorb(A[i], B[i], Y[i])
        print(f"  spawn_radius={sr:.2f}: CB={len(sub_test.state_a)}")
    print()

    substrate = TempestFold(alpha=0.01, spawn_radius=0.02)
    for i in range(n):
        substrate.absorb(A[i], B[i], Y[i])

    cb_a = np.array(substrate.state_a, dtype=np.float32)
    cb_b = np.array(substrate.state_b, dtype=np.float32)
    cb_y = np.array(substrate.state_y, dtype=np.int32)
    cb_size = len(cb_a)
    print(f"Codebook size after absorption: {cb_size}")
    print()

    # === Phase 2: Observer reads geometry (LOO) ===
    # Precompute all phi
    K = K_VOTE
    all_phi = np.zeros((cb_size, max_class * K), dtype=np.float32)
    for i in range(cb_size):
        all_phi[i] = compute_phi(cb_a[i], cb_b[i], cb_a, cb_b, cb_y, i, K, max_class)

    # LOO with phi observer
    correct_phi = 0
    correct_1nn = 0
    total = 0

    for i in range(cb_size):
        true_y = cb_y[i]
        query_a = cb_a[i]
        query_b = cb_b[i]

        pred_phi = predict_phi(query_a, query_b, cb_a, cb_b, cb_y, i, K, max_class,
                               all_phi, cb_y)
        pred_1nn = predict_1nn(query_a, query_b, cb_a, cb_b, cb_y, i)

        if pred_phi == true_y:
            correct_phi += 1
        if pred_1nn == true_y:
            correct_1nn += 1
        total += 1

    acc_phi = correct_phi / total
    acc_1nn = correct_1nn / total

    print(f"LOO Results (K={K}):")
    print(f"  1-NN observer:  {acc_1nn*100:.1f}%")
    print(f"  Phi observer:   {acc_phi*100:.1f}%")
    print(f"  Delta:          {(acc_phi - acc_1nn)*100:+.1f}pp")
    print(f"  Step 296 ref:   86.8%")
    print()

    # === Phase 3: Per-b breakdown ===
    print("Per-b breakdown (phi observer):")
    for b_val in [3, 4, 5, 10, 15, 20]:
        correct = 0
        count = 0
        for i in range(cb_size):
            if cb_b[i] != b_val:
                continue
            pred = predict_phi(cb_a[i], cb_b[i], cb_a, cb_b, cb_y, i, K, max_class,
                               all_phi, cb_y)
            if pred == cb_y[i]:
                correct += 1
            count += 1
        if count > 0:
            print(f"  b={b_val:>2}: {correct/count*100:>6.1f}% ({count} examples)")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
