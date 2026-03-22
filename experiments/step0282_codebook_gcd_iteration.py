#!/usr/bin/env python3
"""
Step 282 -- GCD via emergent Euclid's algorithm.

The substrate discovered a modular step (a,b) -> a%b from I/O examples.
Now test: does iterating that step (a,b) -> (b, pred(a%b)) until b=0
converge to the true GCD? WITHOUT the substrate being told the algorithm.

This is the honest emergent decomposition test.

Spec.
"""

import numpy as np
import time
import math

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX  = 20    # training: a,b in 1..TRAIN_MAX
TEST_MAX   = 50    # test: a,b in 1..TEST_MAX (includes OOD pairs > TRAIN_MAX)
K          = 5
N_CAND     = 400   # feature candidates
N_ROUNDS   = 3     # feature discovery rounds
MAX_STEPS  = 20    # max GCD iteration steps
SEED       = 42


# ── k-NN (L2 distance for integer inputs) ────────────────────────────────────

def topk_acc_l2(V, lv, Q, lq, k=K):
    """k-NN with L2 distance, per-class top-k score = -mean-distance."""
    classes = np.unique(lv)
    n_q = Q.shape[0]
    preds = np.zeros(n_q, dtype=int)
    for i in range(n_q):
        dists = np.sum((V - Q[i])**2, axis=1)
        best_c, best_s = -1, 1e18
        for c in classes:
            idx_c = np.where(lv == c)[0]
            k_eff = min(k, len(idx_c))
            top_dists = np.sort(dists[idx_c])[:k_eff]
            sc = top_dists.mean()
            if sc < best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return (preds == lq).mean()

def loo_acc_l2(X, y, k=K):
    """Vectorized LOO accuracy with L2 distance."""
    n = len(y)
    D = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)  # (n, n) pairwise L2^2
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


# ── Feature templates (applied to raw [a, b] or augmented feature vector) ────

def apply_template(name, w, b_bias, X):
    z = X @ w + b_bias
    if name == 'cos':  return np.cos(z)
    if name == 'abs':  return np.abs(z)
    if name == 'mod2': return (np.floor(np.abs(z)).astype(int) % 2).astype(float)
    if name == 'tanh': return np.tanh(z)
    raise ValueError(name)

TEMPLATES = ['cos', 'abs', 'mod2', 'tanh']


def discover_features(X_tr, y_tr, n_cand=N_CAND, n_rounds=N_ROUNDS, k=K, seed=0, verbose=True):
    rng = np.random.RandomState(seed)
    X_aug = X_tr.copy()
    baseline = loo_acc_l2(X_aug, y_tr, k)
    found = []

    for rnd in range(n_rounds):
        best_delta, best_params, best_col = 0.0, None, None
        d = X_aug.shape[1]

        for name in TEMPLATES:
            for _ in range(n_cand):
                w = rng.randn(d)
                b = rng.randn()
                col = apply_template(name, w, b, X_aug).reshape(-1, 1)
                X_cand = np.hstack([X_aug, col])
                acc = loo_acc_l2(X_cand, y_tr, k)
                delta = acc - baseline
                if delta > best_delta:
                    best_delta, best_params, best_col = delta, (name, w, b), col

        if best_params is None or best_delta <= 0:
            if verbose: print(f"    round {rnd+1}: no improvement, stopping", flush=True)
            break

        X_aug = np.hstack([X_aug, best_col])
        baseline += best_delta
        found.append(best_params)
        if verbose:
            n, w_, b_ = best_params
            print(f"    round {rnd+1}: +{best_delta*100:.1f}pp from {n}  "
                  f"loo={baseline*100:.1f}%  d={X_aug.shape[1]}", flush=True)

    return found, baseline * 100


def augment(X, features):
    X_aug = X.copy()
    for name, w, b in features:
        col = apply_template(name, w, b, X_aug).reshape(-1, 1)
        X_aug = np.hstack([X_aug, col])
    return X_aug


# ── Step accuracy: (a,b) -> a%b ───────────────────────────────────────────────

def build_mod_dataset(max_val):
    """All (a,b) pairs for a,b in 1..max_val. Label = a%b (0..max_val-1)."""
    pairs, labels = [], []
    for a in range(1, max_val + 1):
        for b in range(1, max_val + 1):
            pairs.append([float(a), float(b)])
            labels.append(a % b)
    return np.array(pairs, dtype=np.float32), np.array(labels, dtype=int)


# ── GCD iteration ─────────────────────────────────────────────────────────────

def predict_mod(V, V_labels, query, k=K):
    """Predict a%b for a single query (a, b). Returns predicted remainder."""
    dists = np.sum((V - query)**2, axis=1)
    classes = np.unique(V_labels)
    best_c, best_s = -1, 1e18
    for c in classes:
        idx_c = np.where(V_labels == c)[0]
        k_eff = min(k, len(idx_c))
        sc = np.sort(dists[idx_c])[:k_eff].mean()
        if sc < best_s:
            best_s, best_c = sc, c
    return best_c


def gcd_iterate(V, V_labels, a, b, features, max_steps=MAX_STEPS):
    """
    Euclidean algorithm via iterated k-NN modular prediction.
    (a, b) -> (b, pred(a%b)) until b == 0.
    Returns (predicted_gcd, n_steps, trajectory).
    """
    traj = [(a, b)]
    for step in range(max_steps):
        if b == 0:
            return a, step, traj
        # Build feature-augmented query
        q = augment(np.array([[float(a), float(b)]]), features)
        pred_rem = predict_mod(V, V_labels, q[0], k=K)
        a, b = b, pred_rem
        traj.append((a, b))
    return a, max_steps, traj  # didn't converge


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Step 282 -- GCD via emergent Euclid iteration", flush=True)
    print(f"train: a,b in 1-{TRAIN_MAX} ({TRAIN_MAX**2} pairs)  "
          f"test: a,b in 1-{TEST_MAX}  k={K}", flush=True)

    # ── Build training data ───────────────────────────────────────────────────
    X_tr, y_tr = build_mod_dataset(TRAIN_MAX)

    print(f"\n[Step 1] Feature discovery for (a,b) -> a%b", flush=True)
    t0 = time.time()
    features, loo_final = discover_features(X_tr, y_tr, seed=SEED)
    print(f"  Final LOO: {loo_final:.1f}%  ({len(features)} features, "
          f"{time.time()-t0:.1f}s)", flush=True)

    # ── In-distribution step accuracy ─────────────────────────────────────────
    print(f"\n[Step 2] In-distribution step accuracy (a,b in 1-{TRAIN_MAX})", flush=True)
    X_tr_aug = augment(X_tr, features)
    acc_in = topk_acc_l2(X_tr_aug, y_tr, X_tr_aug, y_tr)  # train=test for IID
    print(f"  In-dist step accuracy: {acc_in*100:.1f}%", flush=True)

    # ── OOD step accuracy (a or b > TRAIN_MAX) ───────────────────────────────
    print(f"\n[Step 3] OOD step accuracy (a or b in {TRAIN_MAX+1}-{TEST_MAX})", flush=True)
    X_ood, y_ood = [], []
    for a in range(TRAIN_MAX + 1, TEST_MAX + 1):
        for b in range(1, TEST_MAX + 1):
            X_ood.append([float(a), float(b)])
            y_ood.append(a % b)
    X_ood = np.array(X_ood, dtype=np.float32)
    y_ood = np.array(y_ood, dtype=int)
    X_ood_aug = augment(X_ood, features)
    # subsample for speed
    idx = np.random.RandomState(0).choice(len(X_ood), min(500, len(X_ood)), replace=False)
    acc_ood = topk_acc_l2(X_tr_aug, y_tr, X_ood_aug[idx], y_ood[idx])
    print(f"  OOD step accuracy: {acc_ood*100:.1f}%  ({len(idx)} pairs)", flush=True)

    # ── GCD iteration test ────────────────────────────────────────────────────
    print(f"\n[Step 4] GCD iteration test (a,b in 1-{TEST_MAX}, ~400 pairs)", flush=True)
    V = X_tr_aug
    V_labels = y_tr

    test_pairs = []
    rng = np.random.RandomState(1)
    # Mix: in-distribution and OOD pairs
    for _ in range(200):
        a = rng.randint(2, TEST_MAX + 1)
        b = rng.randint(1, a)  # b < a to ensure progress
        test_pairs.append((a, b))

    correct = 0
    step_counts = []
    errors = []
    for a, b in test_pairs:
        true_gcd = math.gcd(a, b)
        pred_gcd, n_steps, traj = gcd_iterate(V, V_labels, a, b, features)
        step_counts.append(n_steps)
        if pred_gcd == true_gcd:
            correct += 1
        else:
            errors.append((a, b, true_gcd, pred_gcd, n_steps, traj))

    acc_gcd = correct / len(test_pairs)
    in_dist = [(a, b) for a, b in test_pairs if a <= TRAIN_MAX and b <= TRAIN_MAX]
    ood_pairs = [(a, b) for a, b in test_pairs if a > TRAIN_MAX or b > TRAIN_MAX]
    in_correct = sum(1 for a, b in in_dist if math.gcd(a,b) ==
                     gcd_iterate(V, V_labels, a, b, features)[0])
    ood_correct = sum(1 for a, b in ood_pairs if math.gcd(a,b) ==
                      gcd_iterate(V, V_labels, a, b, features)[0])

    print(f"  GCD accuracy (all): {acc_gcd*100:.1f}%  ({correct}/{len(test_pairs)})", flush=True)
    print(f"  In-distribution:    {in_correct}/{len(in_dist)} = "
          f"{in_correct/max(1,len(in_dist))*100:.1f}%", flush=True)
    print(f"  OOD (a or b > {TRAIN_MAX}): {ood_correct}/{len(ood_pairs)} = "
          f"{ood_correct/max(1,len(ood_pairs))*100:.1f}%", flush=True)
    print(f"  Mean steps: {np.mean(step_counts):.1f}  Max: {np.max(step_counts)}", flush=True)

    if errors[:3]:
        print(f"\n  First 3 errors:", flush=True)
        for a, b, tg, pg, ns, traj in errors[:3]:
            print(f"    gcd({a},{b})={tg}, predicted={pg}, steps={ns}, "
                  f"traj={traj[:5]}", flush=True)

    elapsed = time.time() - t_start

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 282 FINAL", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Step LOO (in-dist):    {loo_final:.1f}%", flush=True)
    print(f"  Step acc (OOD):        {acc_ood*100:.1f}%", flush=True)
    print(f"  GCD accuracy (all):    {acc_gcd*100:.1f}%", flush=True)
    print(f"  GCD OOD accuracy:      "
          f"{ood_correct/max(1,len(ood_pairs))*100:.1f}%", flush=True)

    if acc_gcd >= 0.80:
        verdict = "EMERGENT EUCLID: substrate discovered GCD algorithm from I/O"
    elif acc_gcd >= 0.60:
        verdict = "PARTIAL: GCD mostly works, step errors cascade on hard pairs"
    else:
        verdict = "FAILS: step errors cascade, GCD not reliably discovered"
    print(f"\n  Verdict: {verdict}", flush=True)
    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
