#!/usr/bin/env python3
"""
Step 290 -- Can the substrate discover the Collatz step from I/O?

Spec/1314. HARD TEST: the Collatz step has TWO branches:
  n even: n/2
  n odd:  3n+1

This is fundamentally harder than GCD steps because:
  1. The function is DISCONTINUOUS at every even/odd boundary
  2. The two branches have opposite behavior (shrink vs grow)
  3. No single metric captures both branches

Tests:
  1. Raw encoding: k-NN on raw integers [n] -> Collatz(n)
  2. Thermometer: encode n as thermometer vector -> Collatz(n)
  3. One-hot: explicit class per value
  4. Binary: the parity bit is literally the first bit -- binary might work!
  5. Iteration: does iterated Collatz step converge to 1 for held-out n?

The binary encoding test is the interesting one: binary encodes parity directly
(bit 0 = 1 iff n is odd). If k-NN uses bit 0 to discriminate the two branches,
it might achieve high step accuracy.

Kill: if binary encoding doesn't reach >75% step LOO, the two-branch structure
prevents k-NN from learning any algorithmic step structure.
"""

import numpy as np
import time

# -- Config -------------------------------------------------------------------

TRAIN_MAX   = 30    # training: n in 1..TRAIN_MAX
TEST_MAX    = 100   # OOD test: n in TRAIN_MAX+1..TEST_MAX
K           = 5
MAX_ITER    = 200   # max Collatz iteration steps
OOD_SAMPLE  = 300
SEED        = 42

# -- Collatz step -------------------------------------------------------------

def collatz_step(n):
    """One Collatz step. Returns next value."""
    if n <= 0:
        return 1
    return n // 2 if n % 2 == 0 else 3 * n + 1

# -- Encodings ----------------------------------------------------------------

def encode_raw(n, max_val):
    return np.array([float(n)], dtype=np.float32)

def encode_thermometer(n, max_val):
    v = np.zeros(max_val, dtype=np.float32)
    v[:min(n, max_val)] = 1.0
    return v

def encode_onehot(n, max_val):
    v = np.zeros(max_val, dtype=np.float32)
    if 1 <= n <= max_val:
        v[n - 1] = 1.0
    return v

def encode_binary(n, max_val):
    n_bits = int(np.ceil(np.log2(max_val + 1)))
    return np.array([(n >> i) & 1 for i in range(n_bits)], dtype=np.float32)

ENCODINGS = {
    'raw':         encode_raw,
    'thermometer': encode_thermometer,
    'one-hot':     encode_onehot,
    'binary':      encode_binary,
}

# -- Dataset ------------------------------------------------------------------

def build_dataset(max_val, enc_fn, enc_max=None):
    """n in 1..max_val -> Collatz(n). Note: Collatz(n) can exceed max_val."""
    if enc_max is None: enc_max = max_val
    X, y = [], []
    for n in range(1, max_val + 1):
        nxt = collatz_step(n)
        X.append(enc_fn(n, enc_max))
        y.append(nxt)
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

# -- Collatz iteration test --------------------------------------------------

def collatz_iterate(V, lv, n_start, enc_fn, enc_max, max_steps=MAX_ITER):
    """
    Iterate predicted Collatz step until we reach 1 (or max_steps).
    Returns (reached_1, n_steps, traj).
    """
    n = n_start
    traj = [n]
    for step in range(max_steps):
        if n == 1:
            return True, step, traj
        q = enc_fn(n, enc_max)
        n_next = predict_step(V, lv, q, k=K)
        if n_next <= 0:
            return False, step, traj  # invalid prediction
        n = n_next
        traj.append(n)
        if len(set(traj[-10:])) == 1:
            return False, step, traj  # stuck in loop
    return n == 1, max_steps, traj

# -- Main ---------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Step 290 -- Collatz step discovery via k-NN", flush=True)
    print(f"TRAIN_MAX={TRAIN_MAX}, TEST_MAX={TEST_MAX}, k={K}", flush=True)

    rng = np.random.RandomState(SEED)

    results = {}

    for enc_name, enc_fn in ENCODINGS.items():
        t0 = time.time()
        enc_max = TEST_MAX if enc_name in ('thermometer', 'one-hot') else TRAIN_MAX
        print(f"\n[{enc_name}]", flush=True)

        # Training data
        X_tr, y_tr = build_dataset(TRAIN_MAX, enc_fn, enc_max)
        n_classes = len(np.unique(y_tr))
        print(f"  dim={X_tr.shape[1]}, classes={n_classes}", flush=True)

        # Collatz step output can exceed TRAIN_MAX (3n+1 for odd n).
        # For training: n in 1..30. Output: some n up to ~93. High diversity.
        print(f"  Output range: {y_tr.min()}-{y_tr.max()}", flush=True)

        # LOO accuracy
        loo = loo_acc_l2(X_tr, y_tr, k=K)
        print(f"  LOO: {loo*100:.1f}%", flush=True)

        # OOD step accuracy (n > TRAIN_MAX)
        X_ood = np.array([enc_fn(n, enc_max) for n in range(TRAIN_MAX+1, TEST_MAX+1)],
                         dtype=np.float32)
        y_ood = np.array([collatz_step(n) for n in range(TRAIN_MAX+1, TEST_MAX+1)],
                         dtype=int)
        # OOD prediction: only predict if output class seen in training
        known_classes = set(np.unique(y_tr))
        valid_idx = [i for i, c in enumerate(y_ood) if c in known_classes]
        if valid_idx:
            X_ood_v = X_ood[valid_idx]
            y_ood_v = y_ood[valid_idx]
            preds_ood = np.array([predict_step(X_tr, y_tr, X_ood_v[i]) for i in range(len(y_ood_v))])
            acc_ood = (preds_ood == y_ood_v).mean()
            print(f"  OOD step acc (valid classes): {acc_ood*100:.1f}%  "
                  f"({len(valid_idx)}/{len(y_ood)} in-vocab)", flush=True)
        else:
            acc_ood = 0.0
            print(f"  OOD: no valid classes in training output vocab", flush=True)

        # Collatz iteration test (does predicted step converge to 1?)
        test_ns = list(range(2, min(TRAIN_MAX + 1, 31)))  # test n=2..30
        reached_1 = 0
        for n in test_ns:
            ok, n_steps, traj = collatz_iterate(X_tr, y_tr, n, enc_fn, enc_max)
            if ok:
                reached_1 += 1
        acc_iter = reached_1 / len(test_ns)
        print(f"  Collatz iteration (n=2..30 reaches 1): "
              f"{reached_1}/{len(test_ns)} = {acc_iter*100:.1f}%", flush=True)

        print(f"  ({time.time()-t0:.1f}s)", flush=True)
        results[enc_name] = {
            'dim': X_tr.shape[1],
            'loo': loo * 100,
            'ood': acc_ood * 100,
            'iter': acc_iter * 100,
        }

    elapsed = time.time() - t_start

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 290 FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Encoding':<12} {'Dim':>5} {'LOO':>8} {'OOD':>8} {'Iter->1':>8}",
          flush=True)
    print(f"{'-'*46}", flush=True)
    for enc_name, r in results.items():
        print(f"{enc_name:<12} {r['dim']:>5} {r['loo']:>7.1f}% {r['ood']:>7.1f}% "
              f"{r['iter']:>7.1f}%", flush=True)

    best_loo = max(r['loo'] for r in results.values())
    best_enc = max(results, key=lambda k: results[k]['loo'])
    print(f"\n  Best LOO: {best_enc} ({best_loo:.1f}%)", flush=True)

    if best_loo >= 75.0:
        print(f"\n  PASS: {best_enc} encoding makes Collatz step discoverable!", flush=True)
    elif best_loo >= 50.0:
        print(f"\n  PARTIAL: best LOO={best_loo:.1f}%, parity partially captured",
              flush=True)
    else:
        print(f"\n  FAIL: Collatz step undiscoverable (best LOO={best_loo:.1f}%)",
              flush=True)
        print(f"  -> Two-branch structure prevents k-NN from learning Collatz step",
              flush=True)
        print(f"  -> ARC KILL CRITERION MET: emergent step discovery KILLED", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
