#!/usr/bin/env python3
"""
Step 286 -- Which encoding makes a%b k-NN discoverable?

Spec. Arc question: can the substrate discover an algorithmic step
from I/O and iterate it OOD?

This step: test 4 encodings for (a,b) -> a%b k-NN step accuracy.
  - raw:         [a, b] as floats (2D input)
  - one-hot:     a -> one-hot(max_val) ++ b -> one-hot(max_val)
  - binary:      a -> binary bits ++ b -> binary bits
  - thermometer: a -> [1]*a ++ [0]*(max-a) ++ same for b

Prediction (Lipschitz hypothesis):
  Modular arithmetic is discontinuous in raw space.
  Encodings that make it continuous -> discoverable.
  Thermometer preserves ordinal structure -> most continuous.
  One-hot makes each value orthogonal -> categorical, not ordinal.
  Binary has partial continuity (adjacent integers differ in 1+ bits).
  Raw: discontinuous ceiling.

Measure: LOO k-NN accuracy (L2) on training set + OOD accuracy (a or b > TRAIN_MAX).
"""

import numpy as np
import time
import math

# -- Config -------------------------------------------------------------------

TRAIN_MAX = 20   # training: a,b in 1..TRAIN_MAX
TEST_MAX  = 50   # OOD: a or b in TRAIN_MAX+1..TEST_MAX
K         = 5
OOD_SAMPLE = 500
SEED      = 42

# -- Encoding functions -------------------------------------------------------

def encode_raw(a, b, max_val):
    return np.array([float(a), float(b)], dtype=np.float32)

def encode_onehot(a, b, max_val):
    v = np.zeros(2 * max_val, dtype=np.float32)
    v[a - 1] = 1.0
    v[max_val + b - 1] = 1.0
    return v

def encode_binary(a, b, max_val):
    n_bits = int(np.ceil(np.log2(max_val + 1)))
    ba = np.array([(a >> i) & 1 for i in range(n_bits)], dtype=np.float32)
    bb = np.array([(b >> i) & 1 for i in range(n_bits)], dtype=np.float32)
    return np.concatenate([ba, bb])

def encode_thermometer(a, b, max_val):
    va = np.zeros(max_val, dtype=np.float32)
    va[:a] = 1.0
    vb = np.zeros(max_val, dtype=np.float32)
    vb[:b] = 1.0
    return np.concatenate([va, vb])

ENCODINGS = {
    'raw':         encode_raw,
    'binary':      encode_binary,
    'one-hot':     encode_onehot,
    'thermometer': encode_thermometer,
}

# -- Dataset builders ---------------------------------------------------------

def build_dataset(max_val, enc_fn):
    """All (a,b) pairs for a,b in 1..max_val. Label = a%b."""
    X, y = [], []
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            X.append(enc_fn(a, b, max_val))
            y.append(a % b)
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)

def build_ood_dataset(train_max, test_max, enc_fn, enc_max, rng, n=OOD_SAMPLE):
    """OOD pairs: at least one of a,b > TRAIN_MAX."""
    pairs, labels = [], []
    for a in range(train_max + 1, test_max + 1):
        for b in range(1, test_max + 1):
            pairs.append((a, b, a % b))
    if len(pairs) > n:
        idx = rng.choice(len(pairs), n, replace=False)
        pairs = [pairs[i] for i in idx]
    X = np.array([enc_fn(a, b, enc_max) for a, b, _ in pairs], dtype=np.float32)
    y = np.array([lbl for _, _, lbl in pairs], dtype=int)
    return X, y

# -- k-NN (L2) ----------------------------------------------------------------

def loo_acc_l2(X, y, k=K):
    """Vectorized LOO accuracy with L2 distance."""
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

def topk_acc_l2(V, lv, Q, lq, k=K):
    """k-NN L2 from vault V to queries Q."""
    classes = np.unique(lv)
    n_q = Q.shape[0]
    preds = np.zeros(n_q, dtype=int)
    for i in range(n_q):
        dists = np.sum((V - Q[i])**2, axis=1)
        best_c, best_s = -1, 1e18
        for c in classes:
            idx_c = np.where(lv == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(dists[idx_c])[:k_eff].mean()
            if sc < best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return (preds == lq).mean()

# -- Per-class accuracy -------------------------------------------------------

def per_class_stats(V, lv, Q, lq, k=K):
    """Return accuracy per remainder class (0 to TRAIN_MAX-1)."""
    classes = np.unique(lv)
    preds = np.zeros(len(lq), dtype=int)
    for i in range(len(lq)):
        dists = np.sum((V - Q[i])**2, axis=1)
        best_c, best_s = -1, 1e18
        for c in classes:
            idx_c = np.where(lv == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(dists[idx_c])[:k_eff].mean()
            if sc < best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    # Accuracy per true class
    stats = {}
    for c in classes:
        mask = lq == c
        if mask.sum() > 0:
            stats[c] = (preds[mask] == lq[mask]).mean()
    return stats

# -- Main ---------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Step 286 -- Encoding comparison for a%b k-NN step discovery", flush=True)
    print(f"TRAIN_MAX={TRAIN_MAX}, TEST_MAX={TEST_MAX}, k={K}", flush=True)

    rng = np.random.RandomState(SEED)

    # enc_max for OOD: use TRAIN_MAX as the encoding vocabulary
    # (OOD inputs that exceed TRAIN_MAX get clipped to TRAIN_MAX in thermometer/one-hot)
    # For raw and binary, no vocab issue.
    # For one-hot and thermometer: OOD inputs with a > TRAIN_MAX are out of encoding range.
    # We'll handle this per-encoding.

    results = {}

    for enc_name, enc_fn in ENCODINGS.items():
        t0 = time.time()
        print(f"\n[{enc_name}]", flush=True)

        # Training data
        X_tr, y_tr = build_dataset(TRAIN_MAX, enc_fn)
        d = X_tr.shape[1]
        print(f"  dim={d}", flush=True)

        # LOO accuracy
        loo = loo_acc_l2(X_tr, y_tr, k=K)
        print(f"  LOO: {loo*100:.1f}%", flush=True)

        # OOD accuracy
        # For one-hot and thermometer, OOD values > TRAIN_MAX can't be encoded in
        # the training vocabulary. We test two sub-cases:
        #   (a) a > TRAIN_MAX, b in 1..TRAIN_MAX (b is in-vocab, a is out-of-vocab for OH/TH)
        #   (b) both a,b in 1..TRAIN_MAX but a or b pushed to TEST_MAX range
        # For raw and binary: no issue, just encode with actual values.
        # For OH/TH: encode OOD a with clamped vocab (a%TRAIN_MAX or zero-padded).

        # For simplicity: OOD test only on pairs where both a,b <= TEST_MAX.
        # For encodings with fixed vocab (one-hot, thermometer), we use EXTENDED vocab = TEST_MAX.
        # Re-encode training set with TEST_MAX vocab, then OOD pairs naturally fit.
        if enc_name in ('one-hot', 'thermometer'):
            # Extended vocab encoding
            X_tr_ext, y_tr_ext = build_dataset(TRAIN_MAX, lambda a, b, _: enc_fn(a, b, TEST_MAX))
            X_ood, y_ood = build_ood_dataset(
                TRAIN_MAX, TEST_MAX,
                lambda a, b, _: enc_fn(a, b, TEST_MAX),
                TEST_MAX, rng
            )
            # Note: X_tr_ext has dim 2*TEST_MAX, X_ood also has dim 2*TEST_MAX. Match.
            acc_ood = topk_acc_l2(X_tr_ext, y_tr_ext, X_ood, y_ood, k=K)
        else:
            X_ood, y_ood = build_ood_dataset(
                TRAIN_MAX, TEST_MAX,
                enc_fn, TRAIN_MAX, rng
            )
            acc_ood = topk_acc_l2(X_tr, y_tr, X_ood, y_ood, k=K)

        print(f"  OOD: {acc_ood*100:.1f}%  ({len(y_ood)} pairs)", flush=True)
        print(f"  ({time.time()-t0:.1f}s)", flush=True)

        # Error analysis: what is most commonly predicted?
        # Check if remainder=0 is over-predicted (bias toward most common class)
        n_classes = len(np.unique(y_tr))
        class0_count = (y_tr == 0).sum()
        print(f"  class 0 (a%b=0) count in train: {class0_count}/{len(y_tr)} = "
              f"{class0_count/len(y_tr)*100:.1f}%", flush=True)

        results[enc_name] = {
            'dim': d,
            'loo': loo * 100,
            'ood': acc_ood * 100,
        }

    elapsed = time.time() - t_start

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 286 FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Encoding':<12} {'Dim':>5} {'LOO (in-dist)':>14} {'OOD (a>20)':>12}", flush=True)
    print(f"{'-'*48}", flush=True)
    for enc_name, r in results.items():
        print(f"{enc_name:<12} {r['dim']:>5} {r['loo']:>13.1f}% {r['ood']:>11.1f}%",
              flush=True)

    # Best encoding
    best_loo = max(results, key=lambda k: results[k]['loo'])
    best_ood = max(results, key=lambda k: results[k]['ood'])
    print(f"\n  Best LOO: {best_loo} ({results[best_loo]['loo']:.1f}%)", flush=True)
    print(f"  Best OOD: {best_ood} ({results[best_ood]['ood']:.1f}%)", flush=True)

    # Verdict
    loo_thresh = 75.0
    ood_thresh = 60.0
    passing = [(n, r) for n, r in results.items()
               if r['loo'] >= loo_thresh and r['ood'] >= ood_thresh]
    if passing:
        print(f"\n  PASS: {[n for n, _ in passing]} meet LOO>={loo_thresh}% "
              f"and OOD>={ood_thresh}%", flush=True)
        print(f"  -> Proceed to Step 287: GCD iteration with best encoding", flush=True)
    else:
        # Partial pass
        loo_only = [(n, r) for n, r in results.items() if r['loo'] >= loo_thresh]
        if loo_only:
            print(f"\n  PARTIAL: {[n for n, _ in loo_only]} meet LOO>={loo_thresh}% "
                  f"but OOD generalization weak", flush=True)
            print(f"  -> Step 287 still worth trying for best LOO encoding", flush=True)
        else:
            print(f"\n  FAIL: No encoding achieves LOO>={loo_thresh}%", flush=True)
            print(f"  -> a%b step is not k-NN discoverable with these encodings", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
