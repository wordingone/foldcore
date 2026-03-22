#!/usr/bin/env python3
"""
Step 289 (curriculum) -- Can curriculum help? 1-digit mod then 2-digit.

Spec/1314. Arc question: emergent step discovery.

Curriculum: train on small mod first (a,b in 1..10), then extend to larger (1..20).
Does the step predictor generalize from small to large?

Hypothesis: curriculum won't help because the underlying discontinuity is not
resolved by staged training. But we need the data point.

Tests:
  1. Direct training on 1..20 (baseline from Step 286 thermometer: 41.8%)
  2. Curriculum: train on 1..10 first, then extend to 1..20
  3. Fine-tune: start with 1..10 features, refine on 1..20 extension pairs
  4. OOD accuracy (a or b > 20) for each approach
"""

import numpy as np
import time

# -- Config -------------------------------------------------------------------

SMALL_MAX  = 10    # curriculum stage 1
FULL_MAX   = 20    # curriculum stage 2 / direct training
TEST_MAX   = 50    # OOD test range
K          = 5
N_CAND     = 400   # reduced for speed
N_ROUNDS   = 4
SEED       = 42

# -- Encoding -----------------------------------------------------------------

def encode_thermometer(a, b, max_val):
    va = np.zeros(max_val, dtype=np.float32)
    va[:a] = 1.0
    vb = np.zeros(max_val, dtype=np.float32)
    vb[:b] = 1.0
    return np.concatenate([va, vb])

def build_dataset(max_val, enc_max=None):
    if enc_max is None: enc_max = max_val
    X, y = [], []
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            X.append(encode_thermometer(a, b, enc_max))
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

# -- Feature discovery --------------------------------------------------------

def apply_template(name, w, b_bias, X):
    z = X @ w + b_bias
    if name == 'cos':  return np.cos(z)
    if name == 'abs':  return np.abs(z)
    if name == 'mod2': return (np.floor(np.abs(z)).astype(int) % 2).astype(float)
    if name == 'tanh': return np.tanh(z)
    raise ValueError(name)

TEMPLATES = ['cos', 'abs', 'mod2', 'tanh']

def discover_features(X_tr, y_tr, n_cand=N_CAND, n_rounds=N_ROUNDS, k=K, seed=0):
    rng = np.random.RandomState(seed)
    X_aug = X_tr.copy()
    baseline = loo_acc_l2(X_aug, y_tr, k)
    found = []
    for rnd in range(n_rounds):
        best_delta, best_params, best_col = 0.0, None, None
        d = X_aug.shape[1]
        for name in TEMPLATES:
            for _ in range(n_cand // len(TEMPLATES)):
                w = rng.randn(d); b = rng.randn()
                col = apply_template(name, w, b, X_aug).reshape(-1, 1)
                X_cand = np.hstack([X_aug, col])
                acc = loo_acc_l2(X_cand, y_tr, k)
                delta = acc - baseline
                if delta > best_delta:
                    best_delta, best_params, best_col = delta, (name, w, b), col
        if best_params is None or best_delta <= 0:
            break
        X_aug = np.hstack([X_aug, best_col])
        baseline += best_delta
        found.append(best_params)
        print(f"    round {rnd+1}: +{best_delta*100:.1f}pp from {best_params[0]}  "
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
    print("Step 289 (curriculum) -- 1-digit mod then 2-digit for a%b", flush=True)
    print(f"SMALL_MAX={SMALL_MAX}, FULL_MAX={FULL_MAX}, TEST_MAX={TEST_MAX}", flush=True)

    rng = np.random.RandomState(SEED)
    ENC_MAX = FULL_MAX   # fixed encoding vocab = FULL_MAX for fair comparison

    # ---- Condition 1: Direct training on FULL_MAX (baseline) ----------------
    print(f"\n[Condition 1] Direct training on 1..{FULL_MAX}", flush=True)
    X_full, y_full = build_dataset(FULL_MAX, enc_max=ENC_MAX)
    print(f"  {len(X_full)} pairs, d={X_full.shape[1]}", flush=True)
    loo_direct = loo_acc_l2(X_full, y_full, k=K)
    print(f"  LOO: {loo_direct*100:.1f}%", flush=True)

    # ---- Condition 2: Stage-1 features (trained on SMALL_MAX) ---------------
    print(f"\n[Condition 2] Stage-1: features from 1..{SMALL_MAX}", flush=True)
    X_small, y_small = build_dataset(SMALL_MAX, enc_max=ENC_MAX)
    print(f"  {len(X_small)} pairs, d={X_small.shape[1]}", flush=True)
    features_small, loo_small = discover_features(X_small, y_small, seed=SEED)
    print(f"  Stage-1 LOO: {loo_small:.1f}%  ({len(features_small)} features)", flush=True)

    # Transfer: apply stage-1 features to FULL_MAX data, then eval LOO
    if features_small:
        X_full_aug1 = augment(X_full, features_small)
        loo_transfer = loo_acc_l2(X_full_aug1, y_full, k=K)
        print(f"  Transfer to 1..{FULL_MAX} LOO: {loo_transfer*100:.1f}%", flush=True)
    else:
        X_full_aug1 = X_full.copy()
        loo_transfer = loo_direct
        print(f"  No features found in stage-1. Transfer = direct.", flush=True)

    # ---- Condition 3: Curriculum (stage-1 features + refine on extension) ---
    print(f"\n[Condition 3] Curriculum: stage-1 features + refine on extension pairs",
          flush=True)
    # Extension pairs: at least one of a,b in SMALL_MAX+1..FULL_MAX
    X_ext, y_ext = [], []
    for a in range(SMALL_MAX + 1, FULL_MAX + 1):
        for b in range(1, FULL_MAX + 1):
            X_ext.append(encode_thermometer(a, b, ENC_MAX))
            y_ext.append(a % b)
    X_ext = np.array(X_ext, dtype=np.float32)
    y_ext = np.array(y_ext, dtype=int)
    print(f"  Extension pairs: {len(X_ext)}", flush=True)

    # Build combined dataset (small + extension)
    X_combined = np.vstack([X_small, X_ext])
    y_combined = np.concatenate([y_small, y_ext])
    if features_small:
        X_combined_aug = augment(X_combined, features_small)
        features_curriculum, loo_curriculum = discover_features(
            X_combined_aug, y_combined, seed=SEED + 1
        )
    else:
        X_combined_aug = X_combined.copy()
        features_curriculum, loo_curriculum = discover_features(
            X_combined_aug, y_combined, seed=SEED + 1
        )
    all_feats = list(features_small) + list(features_curriculum)
    X_full_aug_cur = augment(X_full, all_feats)
    loo_final = loo_acc_l2(X_full_aug_cur, y_full, k=K)
    print(f"  Curriculum LOO on full 1..{FULL_MAX}: {loo_final*100:.1f}%  "
          f"({len(all_feats)} total features)", flush=True)

    # ---- OOD test (a or b > FULL_MAX) for all conditions -------------------
    print(f"\n[OOD test, a or b in {FULL_MAX+1}..{TEST_MAX}]", flush=True)
    X_ood, y_ood = [], []
    for a in range(FULL_MAX + 1, TEST_MAX + 1):
        for b in range(1, FULL_MAX + 1):
            X_ood.append(encode_thermometer(a, b, ENC_MAX))
            y_ood.append(a % b)
    X_ood = np.array(X_ood, dtype=np.float32)
    y_ood = np.array(y_ood, dtype=int)
    idx = rng.choice(len(y_ood), min(500, len(y_ood)), replace=False)
    X_ood_s, y_ood_s = X_ood[idx], y_ood[idx]

    # Cond 1: direct
    acc_ood_direct = topk_acc_l2(X_full, y_full, X_ood_s, y_ood_s)
    print(f"  Direct OOD: {acc_ood_direct*100:.1f}%", flush=True)

    # Cond 2: stage-1 features
    if features_small:
        X_full_aug1_v = augment(X_full, features_small)
        X_ood_aug1 = augment(X_ood_s, features_small)
        acc_ood_s1 = topk_acc_l2(X_full_aug1_v, y_full, X_ood_aug1, y_ood_s)
    else:
        acc_ood_s1 = acc_ood_direct
    print(f"  Stage-1 feats OOD: {acc_ood_s1*100:.1f}%", flush=True)

    # Cond 3: curriculum features
    X_ood_aug_cur = augment(X_ood_s, all_feats)
    acc_ood_cur = topk_acc_l2(X_full_aug_cur, y_full, X_ood_aug_cur, y_ood_s)
    print(f"  Curriculum OOD: {acc_ood_cur*100:.1f}%", flush=True)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"STEP 289 (CURRICULUM) FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Direct LOO (1..{FULL_MAX}):       {loo_direct*100:.1f}%", flush=True)
    print(f"  Stage-1 LOO (1..{SMALL_MAX}):      {loo_small:.1f}%", flush=True)
    print(f"  Transfer LOO:              {loo_transfer*100:.1f}%", flush=True)
    print(f"  Curriculum LOO:            {loo_final*100:.1f}%", flush=True)
    print(f"  Direct OOD:                {acc_ood_direct*100:.1f}%", flush=True)
    print(f"  Curriculum OOD:            {acc_ood_cur*100:.1f}%", flush=True)

    best_loo = max(loo_direct*100, loo_final*100)
    if best_loo >= 75.0:
        print(f"\n  PASS: curriculum closes the gap!", flush=True)
    else:
        delta = loo_final*100 - loo_direct*100
        print(f"\n  FAIL: curriculum adds {delta:+.1f}pp. Modular discontinuity not resolved.",
              flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
