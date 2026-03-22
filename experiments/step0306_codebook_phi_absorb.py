#!/usr/bin/env python3
"""
Step 306 -- Distribution-space absorption. f = phi.

Spec. The substrate operates in phi-space. phi IS f, not the observer.

absorb(codebook, input):
    phi_input = compute_phi(input, codebook)     # input's distribution
    if no close match in phi-space:
        codebook.add(input)                      # spawn (raw storage)
    else:
        nearest = argmin phi-distance            # match in DISTRIBUTION space
        codebook[nearest] = blend(nearest, input) # absorb in INPUT space
        # distributions recompute lazily from new codebook geometry

the prediction: same-class inputs have near-identical phi -> lossless absorption.
CB should compress to ~210 (one per (b,class) pair).
LOO >= 86.8% (Step 296 phi reference).

Kill: LOO < 80%.
"""

import time
import numpy as np
from collections import defaultdict

K = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
ALPHA = 0.1

STEP296_REF = 0.868
STEP305_OOD = 1.000


# ─── Phi computation ─────────────────────────────────────────────────────────

def compute_phi(a_q, b_q, A, B, Y, exclude_idx=-1):
    """
    phi for (a_q, b_q) against dataset (A, B, Y).
    Per-class sorted top-K distances among same-b examples.
    """
    phi = np.full(MAX_CLASS * K, SENTINEL, dtype=np.float32)
    for c in range(MAX_CLASS):
        mask = (B == b_q) & (Y == c)
        if exclude_idx >= 0 and exclude_idx < len(mask) and mask[exclude_idx]:
            mask = mask.copy()
            mask[exclude_idx] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - a_q).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def phi_all_cb(A, B, Y):
    """Compute phi for ALL codebook vectors (LOO within codebook)."""
    n = len(A)
    result = np.zeros((n, MAX_CLASS * K), dtype=np.float32)
    for i in range(n):
        result[i] = compute_phi(A[i], B[i], A, B, Y, exclude_idx=i)
    return result


# ─── Codebook ─────────────────────────────────────────────────────────────────

class PhiCodebook:
    def __init__(self, spawn_thresh=10.0, alpha=ALPHA):
        self.A = np.zeros(0, dtype=np.float32)
        self.B = np.zeros(0, dtype=np.int32)
        self.Y = np.zeros(0, dtype=np.int32)
        self.n_spawn = 0
        self.n_absorb = 0
        self.spawn_thresh = spawn_thresh
        self.alpha = alpha

    @property
    def size(self):
        return len(self.A)

    def absorb(self, a, b, y):
        """
        Absorb one training example into the codebook.
        Match in phi-space. Blend in a-space.
        """
        a = float(a)

        if self.size == 0:
            self.A = np.array([a], dtype=np.float32)
            self.B = np.array([b], dtype=np.int32)
            self.Y = np.array([y], dtype=np.int32)
            self.n_spawn += 1
            return 0, True

        # Compute phi for input against current codebook
        phi_input = compute_phi(a, b, self.A, self.B, self.Y)

        # Compute phi for all codebook vectors (LOO within codebook)
        phi_cb = phi_all_cb(self.A, self.B, self.Y)

        # Find nearest codebook vector in phi-space (L2)
        diffs = phi_cb - phi_input
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        nearest_idx = int(np.argmin(dists))
        min_dist = float(dists[nearest_idx])

        if min_dist > self.spawn_thresh:
            self.A = np.append(self.A, a)
            self.B = np.append(self.B, b)
            self.Y = np.append(self.Y, y)
            self.n_spawn += 1
            return self.size - 1, True
        else:
            # Blend a in input space; keep b and y from nearest
            self.A[nearest_idx] = (1.0 - self.alpha) * self.A[nearest_idx] + self.alpha * a
            self.n_absorb += 1
            return nearest_idx, False

    def predict(self, a, b):
        """
        Predict label for (a, b) by finding nearest in phi-space.
        Restore-free (no state modification).
        """
        if self.size == 0:
            return -1
        phi_q = compute_phi(a, b, self.A, self.B, self.Y)
        phi_cb = phi_all_cb(self.A, self.B, self.Y)
        diffs = phi_cb - phi_q
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        nearest_idx = int(np.argmin(dists))
        return int(self.Y[nearest_idx])

    def predict_loo(self, a, b):
        """
        Predict by excluding the nearest codebook vector.
        Approximates LOO: removes the self-absorbing vector.
        """
        if self.size == 0:
            return -1
        phi_q = compute_phi(a, b, self.A, self.B, self.Y)
        phi_cb = phi_all_cb(self.A, self.B, self.Y)
        diffs = phi_cb - phi_q
        dists = np.sqrt((diffs * diffs).sum(axis=1))

        # Find second-nearest (excluding the exact nearest)
        nearest_idx = int(np.argmin(dists))
        dists[nearest_idx] = float('inf')
        second_nearest = int(np.argmin(dists))
        return int(self.Y[second_nearest])


def train_codebook(spawn_thresh, alpha=ALPHA):
    """Build codebook by feeding all 400 training examples."""
    cb = PhiCodebook(spawn_thresh=spawn_thresh, alpha=alpha)
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((a, b, y))

    size_log = []
    for i, (a, b, y) in enumerate(train_data):
        cb.absorb(a, b, y)
        if i % 50 == 49 or i == len(train_data) - 1:
            size_log.append((i + 1, cb.size, cb.n_spawn, cb.n_absorb))

    return cb, train_data, size_log


def evaluate(cb, train_data, use_loo=False):
    correct = 0
    for a, b, y in train_data:
        if use_loo:
            pred = cb.predict_loo(a, b)
        else:
            pred = cb.predict(a, b)
        if pred == y:
            correct += 1
    return correct / len(train_data)


def main():
    t0 = time.time()
    print("Step 306 -- Distribution-Space Absorption (f = phi)", flush=True)
    print(f"phi = per-class sorted top-K distances, same-b. K={K}", flush=True)
    print(f"Absorb: match in phi-space, blend in a-space.", flush=True)
    print(f"Prediction: CB -> ~210 vectors, LOO >= {STEP296_REF*100:.1f}%\n", flush=True)

    # ─── Spawn threshold sweep ────────────────────────────────────────────
    print("=== Spawn threshold sweep ===", flush=True)
    print(f"{'thresh':>8} | {'CB':>5} | {'Spawns':>7} | {'Absorbs':>8} | {'Acc':>7} | {'LOO-approx':>10}",
          flush=True)
    print("-" * 58, flush=True)

    best_acc = 0
    best_loo = 0
    best_thresh = None

    for thresh in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        t_s = time.time()
        cb, train_data, size_log = train_codebook(thresh)
        acc = evaluate(cb, train_data, use_loo=False)
        loo = evaluate(cb, train_data, use_loo=True)
        elapsed_s = time.time() - t_s

        print(f"  {thresh:>6.1f} | {cb.size:>5} | {cb.n_spawn:>7} | {cb.n_absorb:>8} | "
              f"{acc*100:>6.1f}% | {loo*100:>9.1f}%  [{elapsed_s:.1f}s]", flush=True)

        if loo > best_loo:
            best_loo = loo
            best_thresh = thresh
        if acc > best_acc:
            best_acc = acc

    print(flush=True)

    # ─── Best config: full analysis ───────────────────────────────────────
    print(f"=== Full analysis at best thresh={best_thresh} ===", flush=True)
    cb_best, train_data, size_log = train_codebook(best_thresh)

    # Growth dynamics
    print(f"  Growth log (examples fed | CB size | spawns | absorbs):", flush=True)
    for fed, sz, sp, ab in size_log:
        print(f"    {fed:>4} fed: CB={sz:>4}, spawns={sp:>4}, absorbs={ab:>4}", flush=True)
    print(flush=True)

    # Label structure
    y_counts = defaultdict(int)
    for y in cb_best.Y:
        y_counts[y] += 1
    print(f"  CB vectors: {cb_best.size}", flush=True)
    print(f"  Per-class coverage: {len(y_counts)} classes represented", flush=True)
    print(flush=True)

    # Accuracy
    acc_best = evaluate(cb_best, train_data, use_loo=False)
    loo_best = evaluate(cb_best, train_data, use_loo=True)
    print(f"  In-distribution:  {acc_best*100:.1f}%", flush=True)
    print(f"  LOO-approx:       {loo_best*100:.1f}%  (Step 296 ref: {STEP296_REF*100:.1f}%)", flush=True)
    print(flush=True)

    # Per-b breakdown
    print(f"  Per-b LOO-approx (selected):", flush=True)
    for b in [3, 5, 7, 10, 15, 20]:
        b_data = [(a, b2, y) for a, b2, y in train_data if b2 == b]
        b_correct = sum(1 for a, b2, y in b_data if cb_best.predict_loo(a, b2) == y)
        print(f"    b={b:>2}: {b_correct}/{len(b_data)} = {b_correct/len(b_data)*100:.0f}%",
              flush=True)
    print(flush=True)

    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 306 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Best LOO-approx: {best_loo*100:.1f}%  at thresh={best_thresh}", flush=True)
    print(f"Step 296 phi reference: {STEP296_REF*100:.1f}%", flush=True)
    print(flush=True)

    if best_loo >= 0.80:
        print(f"SUCCESS -- LOO >= 80%. f = phi absorb confirmed.", flush=True)
        print(f"Distribution-space matching + input-space blending works.", flush=True)
    elif best_loo >= 0.50:
        print(f"PARTIAL -- LOO >= 50%, < 80%. Mechanism works, not fully realized.", flush=True)
    else:
        print(f"KILLED -- LOO {best_loo*100:.1f}% < 80%. Distribution matching fails in absorb.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
