#!/usr/bin/env python3
"""
Step 288 -- Subtraction-based GCD: is (a,b) -> (b, a-b) easier to discover?

[researcher] mail 1308. Arc question: can the substrate discover an algorithmic step
from I/O and iterate it OOD?

Subtraction GCD: (a, b) -> (b, a-b) for a > b, swap if a < b, stop if a == b.
The subtraction step is CONTINUOUS (a-b is linear in inputs), unlike a%b.

Hypothesis: if the substrate can discover a-b via k-NN, iteration should give
correct GCD (just many more steps than Euclidean -- O(max(a,b)/gcd) steps).

Tests:
  1. Step accuracy: can k-NN predict (a-b) from (a,b)?
  2. With best encoding from Step 286 (thermometer) AND raw, compare.
  3. Iterate to GCD: does the iterated predictor converge?
  4. OOD: does it work for a,b > TRAIN_MAX?

Subtraction GCD can take O(max(a,b)/gcd) steps -- for (20, 1) that's 19 steps.
For OOD with a=50, b=1: 49 steps needed. Set MAX_STEPS=200.
"""

import numpy as np
import time
import math

# -- Config -------------------------------------------------------------------

TRAIN_MAX   = 20
TEST_MAX    = 50
K           = 5
MAX_STEPS   = 200   # subtraction GCD needs more steps
OOD_SAMPLE  = 300
SEED        = 42

# -- Encodings (same as Step 286) --------------------------------------------

def encode_raw(a, b, max_val):
    return np.array([float(a), float(b)], dtype=np.float32)

def encode_thermometer(a, b, max_val):
    va = np.zeros(max_val, dtype=np.float32)
    va[:min(a, max_val)] = 1.0
    vb = np.zeros(max_val, dtype=np.float32)
    vb[:min(b, max_val)] = 1.0
    return np.concatenate([va, vb])

# -- Dataset: subtraction step (a,b) -> a-b for a >= b ----------------------

def build_sub_dataset(max_val, enc_fn, enc_max=None):
    """
    All (a,b) pairs with a >= b, a,b in 1..max_val.
    Label = a - b (the output value, range 0..max_val-1).
    We store the output as an encoded vector (the next state for iteration).
    """
    if enc_max is None:
        enc_max = max_val
    X, y, y_vec = [], [], []
    for a in range(1, max_val + 1):
        for b in range(1, a + 1):   # b <= a
            X.append(enc_fn(a, b, enc_max))
            y.append(a - b)         # output value
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)

# -- k-NN (L2) ----------------------------------------------------------------

def loo_acc_l2(X, y, k=K):
    n = len(y)
    D = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
    np.fill_diagonal(D, np.inf)
    classes = np.unique(y)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        best_c, best_s = -1, 1e18
        for c in classes:
            idx_c = np.where(y == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(D[i, idx_c])[:k_eff].mean()
            if sc < best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return (preds == y).mean()

def predict_step(V, lv, q, k=K):
    """Predict next value for single query q."""
    dists = np.sum((V - q)**2, axis=1)
    classes = np.unique(lv)
    best_c, best_s = -1, 1e18
    for c in classes:
        idx_c = np.where(lv == c)[0]
        k_eff = min(k, len(idx_c))
        sc = np.sort(dists[idx_c])[:k_eff].mean()
        if sc < best_s:
            best_s, best_c = sc, c
    return best_c

# -- GCD iteration via subtraction -------------------------------------------

def sub_gcd_iterate(V, lv, a, b, enc_fn, enc_max, max_steps=MAX_STEPS):
    """
    Subtraction GCD: (a, b) -> (b, a-b) if a > b, (a, b-a) if b > a, stop if equal.
    Uses k-NN to predict the subtracted value, then iterate.
    Returns (predicted_gcd, n_steps, trajectory).
    """
    traj = [(a, b)]
    for step in range(max_steps):
        if a == b:
            return a, step, traj
        if a == 0 or b == 0:
            return max(a, b), step, traj
        # Always orient so a >= b
        if b > a:
            a, b = b, a
        # Query: predict a - b
        q = enc_fn(a, b, enc_max)
        pred_diff = predict_step(V, lv, q, k=K)
        # Next state: (b, pred_diff)
        a, b = b, pred_diff
        traj.append((a, b))
    return a, max_steps, traj  # didn't converge

# -- Main ---------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Step 288 -- Subtraction-based GCD step discovery", flush=True)
    print(f"TRAIN_MAX={TRAIN_MAX}, TEST_MAX={TEST_MAX}, k={K}, "
          f"MAX_STEPS={MAX_STEPS}", flush=True)

    rng = np.random.RandomState(SEED)

    for enc_name, enc_fn in [('raw', encode_raw), ('thermometer', encode_thermometer)]:
        enc_max = TRAIN_MAX if enc_name == 'raw' else TEST_MAX
        print(f"\n{'='*50}", flush=True)
        print(f"Encoding: {enc_name}", flush=True)

        # Build training data
        X_tr, y_tr = build_sub_dataset(TRAIN_MAX, enc_fn,
                                        enc_max=TRAIN_MAX if enc_name == 'raw' else TEST_MAX)
        n_pairs = len(y_tr)
        n_classes = len(np.unique(y_tr))
        print(f"  Train pairs (a>=b in 1..{TRAIN_MAX}): {n_pairs}  "
              f"classes: {n_classes}", flush=True)

        # LOO accuracy
        loo = loo_acc_l2(X_tr, y_tr, k=K)
        print(f"  LOO step accuracy: {loo*100:.1f}%", flush=True)

        # OOD step accuracy (a > TRAIN_MAX)
        X_ood, y_ood = [], []
        for a in range(TRAIN_MAX + 1, TEST_MAX + 1):
            for b in range(1, a + 1):
                X_ood.append(enc_fn(a, b, enc_max))
                y_ood.append(a - b)
        X_ood = np.array(X_ood, dtype=np.float32)
        y_ood = np.array(y_ood, dtype=int)
        idx = rng.choice(len(y_ood), min(OOD_SAMPLE, len(y_ood)), replace=False)
        X_ood_s, y_ood_s = X_ood[idx], y_ood[idx]
        # OOD k-NN (train on TRAIN_MAX, test on OOD)
        V = X_tr
        n_ood = len(y_ood_s)
        preds_ood = np.array([predict_step(V, y_tr, X_ood_s[i], k=K) for i in range(n_ood)])
        acc_ood = (preds_ood == y_ood_s).mean()
        print(f"  OOD step accuracy (a>{TRAIN_MAX}): {acc_ood*100:.1f}%  "
              f"({n_ood} pairs)", flush=True)

        # GCD iteration test
        print(f"\n  GCD iteration (a,b in 1..{TEST_MAX}):", flush=True)
        test_pairs = []
        for _ in range(200):
            a = rng.randint(2, TEST_MAX + 1)
            b = rng.randint(1, a)
            test_pairs.append((a, b))

        correct = 0
        step_counts = []
        errors = []
        for a, b in test_pairs:
            true_gcd = math.gcd(a, b)
            pred_gcd, n_steps, traj = sub_gcd_iterate(
                V, y_tr, a, b, enc_fn, enc_max, MAX_STEPS
            )
            step_counts.append(n_steps)
            if pred_gcd == true_gcd:
                correct += 1
            else:
                errors.append((a, b, true_gcd, pred_gcd, n_steps))

        acc_gcd = correct / len(test_pairs)
        in_pairs = [(a, b) for a, b in test_pairs if a <= TRAIN_MAX and b <= TRAIN_MAX]
        ood_pairs = [(a, b) for a, b in test_pairs if a > TRAIN_MAX or b > TRAIN_MAX]
        in_correct = sum(1 for a, b in in_pairs
                         if math.gcd(a, b) == sub_gcd_iterate(V, y_tr, a, b, enc_fn,
                                                               enc_max, MAX_STEPS)[0])
        ood_correct = sum(1 for a, b in ood_pairs
                          if math.gcd(a, b) == sub_gcd_iterate(V, y_tr, a, b, enc_fn,
                                                                enc_max, MAX_STEPS)[0])

        print(f"  GCD acc (all):    {acc_gcd*100:.1f}%  ({correct}/{len(test_pairs)})",
              flush=True)
        print(f"  In-dist:          {in_correct}/{len(in_pairs)} = "
              f"{in_correct/max(1,len(in_pairs))*100:.1f}%", flush=True)
        print(f"  OOD (a>{TRAIN_MAX}): {ood_correct}/{len(ood_pairs)} = "
              f"{ood_correct/max(1,len(ood_pairs))*100:.1f}%", flush=True)
        print(f"  Mean steps: {np.mean(step_counts):.1f}  Max: {np.max(step_counts)}",
              flush=True)
        if errors[:2]:
            print(f"  Errors (first 2): {errors[:2]}", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 288 DONE  elapsed={elapsed:.1f}s", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == '__main__':
    main()
