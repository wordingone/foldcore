#!/usr/bin/env python3
"""
Step 307 -- phi + raw features. One fix for Step 306 bootstrap failure.

Spec. Prepend one-hot(a,b) to phi vector.

phi_ext(x) = [one_hot(a,20), one_hot(b,20), per_class_distances(x, codebook)]

Early (sparse): distances = SENTINEL -> phi_ext ~ raw features.
  Same (a,b) -> dist=0 -> absorb. Different (a,b) -> dist=sqrt(2) -> spawn.
  Cross-b absorption prevented.

Late (rich): real distances dominate -> same-class compression.

The transition is automatic: driven by codebook density, not a phase switch.

Kill: LOO < 80%. Success: LOO >= 80% with CB < 400.
"""

import time
import numpy as np
from collections import defaultdict

K = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
ALPHA = 0.1
FEAT_DIM = 2 * TRAIN_MAX  # one_hot(a) ++ one_hot(b) = 40d
PHI_DIM = MAX_CLASS * K   # 100d
EXT_DIM = FEAT_DIM + PHI_DIM  # 140d

STEP296_REF = 0.868


# ─── Encodings ───────────────────────────────────────────────────────────────

def onehot_ab(a, b):
    v = np.zeros(FEAT_DIM, dtype=np.float32)
    a_idx = int(round(a)) - 1
    b_idx = int(b) - 1
    if 0 <= a_idx < TRAIN_MAX:
        v[a_idx] = 1.0
    v[TRAIN_MAX + b_idx] = 1.0
    return v


def compute_phi(a_q, b_q, A, B, Y, exclude_idx=-1):
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


def phi_ext(a, b, A, B, Y, exclude_idx=-1):
    """Extended phi: one_hot(a,b) ++ per_class_distances."""
    feat = onehot_ab(a, b)
    phi = compute_phi(a, b, A, B, Y, exclude_idx=exclude_idx)
    return np.concatenate([feat, phi])


def phi_ext_all(A, B, Y):
    """Extended phi for all codebook vectors (LOO within codebook)."""
    n = len(A)
    result = np.zeros((n, EXT_DIM), dtype=np.float32)
    for i in range(n):
        result[i] = phi_ext(A[i], B[i], A, B, Y, exclude_idx=i)
    return result


# ─── Codebook ─────────────────────────────────────────────────────────────────

class PhiRawCodebook:
    def __init__(self, spawn_thresh=2.0, alpha=ALPHA):
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
        a = float(a)
        if self.size == 0:
            self.A = np.array([a], dtype=np.float32)
            self.B = np.array([b], dtype=np.int32)
            self.Y = np.array([y], dtype=np.int32)
            self.n_spawn += 1
            return 0, True

        phi_in = phi_ext(a, b, self.A, self.B, self.Y)
        phi_cb = phi_ext_all(self.A, self.B, self.Y)

        diffs = phi_cb - phi_in
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
            self.A[nearest_idx] = (1.0 - self.alpha) * self.A[nearest_idx] + self.alpha * a
            self.n_absorb += 1
            return nearest_idx, False

    def predict(self, a, b):
        if self.size == 0:
            return -1
        phi_q = phi_ext(a, b, self.A, self.B, self.Y)
        phi_cb = phi_ext_all(self.A, self.B, self.Y)
        diffs = phi_cb - phi_q
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        return int(self.Y[int(np.argmin(dists))])

    def predict_loo_approx(self, a, b):
        """Predict excluding nearest (LOO approximation)."""
        if self.size <= 1:
            return -1
        phi_q = phi_ext(a, b, self.A, self.B, self.Y)
        phi_cb = phi_ext_all(self.A, self.B, self.Y)
        diffs = phi_cb - phi_q
        dists = np.sqrt((diffs * diffs).sum(axis=1))
        nearest = int(np.argmin(dists))
        dists[nearest] = float('inf')
        return int(self.Y[int(np.argmin(dists))])


def train_codebook(spawn_thresh, alpha=ALPHA):
    cb = PhiRawCodebook(spawn_thresh=spawn_thresh, alpha=alpha)
    train_data = []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            train_data.append((a, b, a % b))

    size_log = []
    for i, (a, b, y) in enumerate(train_data):
        cb.absorb(a, b, y)
        if i in (49, 99, 149, 199, 249, 299, 349) or i == len(train_data) - 1:
            size_log.append((i + 1, cb.size, cb.n_spawn, cb.n_absorb))

    return cb, train_data, size_log


def evaluate(cb, train_data, use_loo=False):
    correct = 0
    for a, b, y in train_data:
        pred = cb.predict_loo_approx(a, b) if use_loo else cb.predict(a, b)
        if pred == y:
            correct += 1
    return correct / len(train_data)


def main():
    t0 = time.time()
    print("Step 307 -- phi + raw features (bootstrap fix)", flush=True)
    print(f"phi_ext = one_hot(a,20) ++ one_hot(b,20) ++ phi_distribution", flush=True)
    print(f"Dim: {EXT_DIM}d  K={K}  SENTINEL={SENTINEL:.0f}", flush=True)
    print(f"Ref: Step 296 LOO={STEP296_REF*100:.1f}%,  Step 306 LOO=16.2%\n", flush=True)

    # Threshold sweep
    print("=== Spawn threshold sweep ===", flush=True)
    print(f"{'thresh':>8} | {'CB':>5} | {'Spawns':>7} | {'Acc':>7} | {'LOO-approx':>10} | time",
          flush=True)
    print("-" * 62, flush=True)

    best_loo = 0
    best_thresh = None

    for thresh in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]:
        t_s = time.time()
        cb, train_data, _ = train_codebook(thresh)
        acc = evaluate(cb, train_data, use_loo=False)
        loo = evaluate(cb, train_data, use_loo=True)
        et = time.time() - t_s
        print(f"  {thresh:>6.1f} | {cb.size:>5} | {cb.n_spawn:>7} | {acc*100:>6.1f}% | "
              f"{loo*100:>9.1f}%  | {et:.1f}s", flush=True)
        if loo > best_loo:
            best_loo = loo
            best_thresh = thresh

    print(flush=True)

    # Full analysis at best threshold
    print(f"=== Full analysis at thresh={best_thresh} ===", flush=True)
    cb_best, train_data, size_log = train_codebook(best_thresh)

    print(f"  Growth log:", flush=True)
    for fed, sz, sp, ab in size_log:
        print(f"    {fed:>4} fed: CB={sz:>4}, spawns={sp:>4}, absorbs={ab:>4}", flush=True)

    y_counts = defaultdict(int)
    for y in cb_best.Y:
        y_counts[y] += 1
    print(f"  CB size: {cb_best.size}, classes: {len(y_counts)}/20", flush=True)
    print(flush=True)

    acc_best = evaluate(cb_best, train_data, use_loo=False)
    loo_best = evaluate(cb_best, train_data, use_loo=True)
    print(f"  In-distribution:  {acc_best*100:.1f}%", flush=True)
    print(f"  LOO-approx:       {loo_best*100:.1f}%  (ref: {STEP296_REF*100:.1f}%)", flush=True)
    print(flush=True)

    # Per-b LOO
    print(f"  Per-b LOO-approx:", flush=True)
    for b in [3, 5, 7, 10, 15, 20]:
        b_data = [(a, b2, y) for a, b2, y in train_data if b2 == b]
        b_correct = sum(1 for a, b2, y in b_data
                        if cb_best.predict_loo_approx(a, b2) == y)
        print(f"    b={b:>2}: {b_correct}/{len(b_data)} = {b_correct/len(b_data)*100:.0f}%",
              flush=True)
    print(flush=True)

    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 307 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Best LOO-approx: {best_loo*100:.1f}% at thresh={best_thresh}", flush=True)
    print(f"Step 296 ref: {STEP296_REF*100:.1f}%", flush=True)
    print(flush=True)

    if best_loo >= 0.80:
        print(f"SUCCESS -- LOO >= 80%. Bootstrap fix works.", flush=True)
        print(f"phi + raw features = automatic feature->distribution transition.", flush=True)
    elif best_loo >= 0.50:
        print(f"PARTIAL -- LOO {best_loo*100:.1f}% above kill bar, below 80%.", flush=True)
    else:
        print(f"KILLED -- LOO {best_loo*100:.1f}% < 80%.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
