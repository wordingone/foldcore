#!/usr/bin/env python3
"""
Step 289 -- Feature discovery on top of thermometer encoding for a%b.

Spec. Does LOO-scored random feature discovery, applied on top of
thermometer encoding (best from Step 286: 41.8% LOO), close the gap to 75%+?

Hypothesis: probably NOT -- the modular function is discontinuous, and random
nonlinear projections of thermometer vectors won't recover periodicity.
But we need the data point to characterize the boundary.

Feature templates applied to the augmented thermometer vector:
  {cos, abs, mod2, tanh} (same as Step 282 framework)

Steps:
  1. Thermometer encoding (enc_max=TRAIN_MAX) for training pairs (a,b in 1..TRAIN_MAX)
  2. Feature discovery rounds: LOO-scored random feature candidates
  3. Report LOO after each round + final OOD accuracy with best features
  4. Compare: thermometer-only (41.8%) vs thermometer + discovered features
"""

import numpy as np
import time
import math

# -- Config -------------------------------------------------------------------

TRAIN_MAX  = 20
TEST_MAX   = 50
K          = 5
N_CAND     = 600    # feature candidates per round
N_ROUNDS   = 5     # discovery rounds
SEED       = 42

# -- Encoding -----------------------------------------------------------------

def encode_thermometer(a, b, max_val):
    va = np.zeros(max_val, dtype=np.float32)
    va[:a] = 1.0
    vb = np.zeros(max_val, dtype=np.float32)
    vb[:b] = 1.0
    return np.concatenate([va, vb])

# -- Dataset ------------------------------------------------------------------

def build_dataset(max_val):
    X, y = [], []
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            X.append(encode_thermometer(a, b, max_val))
            y.append(a % b)
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

def topk_acc_l2(V, lv, Q, lq, k=K):
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

# -- Feature templates --------------------------------------------------------

def apply_template(name, w, b_bias, X):
    z = X @ w + b_bias
    if name == 'cos':  return np.cos(z)
    if name == 'abs':  return np.abs(z)
    if name == 'mod2': return (np.floor(np.abs(z)).astype(int) % 2).astype(float)
    if name == 'tanh': return np.tanh(z)
    raise ValueError(name)

TEMPLATES = ['cos', 'abs', 'mod2', 'tanh']

# -- Feature discovery --------------------------------------------------------

def discover_features(X_tr, y_tr, n_cand=N_CAND, n_rounds=N_ROUNDS, k=K, seed=0,
                      verbose=True):
    rng = np.random.RandomState(seed)
    X_aug = X_tr.copy()
    baseline = loo_acc_l2(X_aug, y_tr, k)
    print(f"  Baseline LOO: {baseline*100:.1f}%  d={X_aug.shape[1]}", flush=True)
    found = []

    for rnd in range(n_rounds):
        best_delta, best_params, best_col = 0.0, None, None
        d = X_aug.shape[1]

        for name in TEMPLATES:
            for _ in range(n_cand // len(TEMPLATES)):
                w = rng.randn(d)
                b = rng.randn()
                col = apply_template(name, w, b, X_aug).reshape(-1, 1)
                X_cand = np.hstack([X_aug, col])
                acc = loo_acc_l2(X_cand, y_tr, k)
                delta = acc - baseline
                if delta > best_delta:
                    best_delta, best_params, best_col = delta, (name, w, b), col

        if best_params is None or best_delta <= 0:
            if verbose:
                print(f"  round {rnd+1}: no improvement, stopping", flush=True)
            break

        X_aug = np.hstack([X_aug, best_col])
        baseline += best_delta
        found.append(best_params)
        if verbose:
            n, w_, b_ = best_params
            print(f"  round {rnd+1}: +{best_delta*100:.1f}pp from {n}  "
                  f"loo={baseline*100:.1f}%  d={X_aug.shape[1]}", flush=True)

    return found, baseline * 100

def augment(X, features):
    X_aug = X.copy()
    for name, w, b in features:
        col = apply_template(name, w, b, X_aug).reshape(-1, 1)
        X_aug = np.hstack([X_aug, col])
    return X_aug

# -- Main ---------------------------------------------------------------------

def main():
    t_start = time.time()
    print("Step 289 -- Feature discovery on thermometer encoding for a%b", flush=True)
    print(f"TRAIN_MAX={TRAIN_MAX}, TEST_MAX={TEST_MAX}, k={K}, "
          f"n_cand={N_CAND}, n_rounds={N_ROUNDS}", flush=True)

    X_tr, y_tr = build_dataset(TRAIN_MAX)
    print(f"Training: {len(X_tr)} pairs, {len(np.unique(y_tr))} classes, "
          f"d={X_tr.shape[1]}", flush=True)

    print(f"\n[Feature discovery on thermometer encoding]", flush=True)
    t0 = time.time()
    features, loo_final = discover_features(X_tr, y_tr, seed=SEED)
    print(f"  Final LOO: {loo_final:.1f}%  ({len(features)} features added, "
          f"{time.time()-t0:.1f}s)", flush=True)

    # OOD test with discovered features
    print(f"\n[OOD step accuracy with features]", flush=True)
    X_tr_aug = augment(X_tr, features)

    # OOD pairs (a > TRAIN_MAX, b <= TRAIN_MAX)
    X_ood, y_ood = [], []
    rng = np.random.RandomState(0)
    for a in range(TRAIN_MAX + 1, TEST_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            X_ood.append(encode_thermometer(a, b, TRAIN_MAX))
            y_ood.append(a % b)
    X_ood = np.array(X_ood, dtype=np.float32)
    y_ood = np.array(y_ood, dtype=int)
    X_ood_aug = augment(X_ood, features)
    idx = rng.choice(len(y_ood), min(500, len(y_ood)), replace=False)
    acc_ood = topk_acc_l2(X_tr_aug, y_tr, X_ood_aug[idx], y_ood[idx])
    print(f"  OOD step accuracy: {acc_ood*100:.1f}%  ({len(idx)} pairs)", flush=True)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 289 FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Thermometer-only LOO (Step 286):  41.8%", flush=True)
    print(f"  Thermometer + features LOO:        {loo_final:.1f}%", flush=True)
    print(f"  OOD step accuracy:                 {acc_ood*100:.1f}%", flush=True)
    delta = loo_final - 41.8
    print(f"  Delta from encoding-only:         {delta:+.1f}pp", flush=True)

    if loo_final >= 75.0:
        print(f"\n  PASS: feature discovery closes the gap!", flush=True)
    elif delta >= 10.0:
        print(f"\n  PARTIAL: features help significantly (+{delta:.1f}pp) but gap remains",
              flush=True)
    else:
        print(f"\n  FAIL: feature discovery cannot overcome modular discontinuity",
              flush=True)
        print(f"  -> Confirms: a%b is structurally undiscoverable via k-NN", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
