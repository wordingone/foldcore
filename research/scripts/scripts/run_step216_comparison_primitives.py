#!/usr/bin/env python3
"""
Step 216 -- Value comparison primitives: can we close the sort gap?

Steps 213-215 showed the substrate fails sorting (45%) with {cos, mod2}.
Hypothesis: the gap is a PRIMITIVE gap, not a MECHANISM gap.
New primitives: relu, pairwise_max, sign_diff.

Tests:
  1. Comparison (a > b): input=(a,b), label=(a>b). Should be trivial for sign_diff.
  2. Sort argmin (len=4, vocab=5): input=(a,b,c,d), label=argmin(). 4 classes.

Kill: {cos,mod2,relu,max,sign_diff} <= {cos,mod2} on sort -> MECHANISM gap.
Pass: expanded menu closes gap on sort -> PRIMITIVE gap.

[researcher] mail 1292.
"""

import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────

K            = 5
N_TRAIN      = 300
N_TEST       = 500
N_CANDIDATES = 200   # per template per round
N_ROUNDS     = 5
SEED         = 42
GAP_KILL     = 5.0   # pp: expanded must beat baseline by this to confirm primitive gap

# ── Tasks ─────────────────────────────────────────────────────────────────────

VOCAB = 5  # integers 0..4

def make_comparison(n, seed=0):
    """(a, b) -> (a > b). Binary."""
    r = np.random.RandomState(seed)
    X = r.randint(0, VOCAB, (n, 2)).astype(np.float32)
    y = (X[:, 0] > X[:, 1]).astype(int)
    return X, y

def make_sort_argmin(n, seed=0):
    """(a,b,c,d) -> argmin. 4 classes."""
    r = np.random.RandomState(seed)
    X = r.randint(0, VOCAB, (n, 4)).astype(np.float32)
    y = X.argmin(axis=1)
    return X, y

def make_sort_argmax(n, seed=0):
    """(a,b,c,d) -> argmax. 4 classes."""
    r = np.random.RandomState(seed)
    X = r.randint(0, VOCAB, (n, 4)).astype(np.float32)
    y = X.argmax(axis=1)
    return X, y

def make_sort_full(n, seed=0):
    """(a,b,c,d) -> sorted tuple as class label. Up to 70 classes."""
    r = np.random.RandomState(seed)
    X = r.randint(0, VOCAB, (n, 4)).astype(np.float32)
    # encode sorted tuple as integer: s[0]*5^3 + s[1]*5^2 + s[2]*5 + s[3]
    S = np.sort(X.astype(int), axis=1)
    y = S[:, 0] * 125 + S[:, 1] * 25 + S[:, 2] * 5 + S[:, 3]
    # re-label to 0..n_classes-1
    unique, inverse = np.unique(y, return_inverse=True)
    return X, inverse


# ── k-NN top-k (vectorized) ───────────────────────────────────────────────────

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n < 1e-10, 1.0, n)

def topk_acc(V, lv, Q, lq, k=K):
    S = norm(Q) @ norm(V).T
    classes = np.unique(lv)
    n_q = Q.shape[0]
    preds = np.zeros(n_q, dtype=int)
    for i in range(n_q):
        best_c, best_s = -1, -1e9
        for c in classes:
            idx_c = np.where(lv == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(S[i, idx_c])[-k_eff:].sum()
            if sc > best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return (preds == lq).mean()

def loo_acc(X_aug, y, k=K):
    Xn = norm(X_aug)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    classes = np.unique(y)
    n = len(y)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        best_c, best_s = -1, -1e9
        for c in classes:
            idx_c = np.where(y == c)[0]
            k_eff = min(k, len(idx_c))
            sc = np.sort(S[i, idx_c])[-k_eff:].sum()
            if sc > best_s:
                best_s, best_c = sc, c
        preds[i] = best_c
    return (preds == y).mean()


# ── Feature templates ─────────────────────────────────────────────────────────
#
# Each generator returns (params, apply_fn) where:
#   params:   anything needed to reproduce the feature
#   apply_fn: (X_aug: np.ndarray(n, d)) -> np.ndarray(n,)  [vectorized]

def gen_cos(d, rng):
    w = rng.randn(d)
    b = rng.uniform(0, 2 * np.pi)
    return ('cos', w, b), lambda X: np.cos(X @ w + b)

def gen_mod2(d, rng):
    w = rng.randn(d)
    b = rng.randn()
    return ('mod2', w, b), lambda X: (np.floor(np.abs(X @ w + b)).astype(int) % 2).astype(float)

def gen_relu(d, rng):
    w = rng.randn(d)
    b = rng.randn()
    return ('relu', w, b), lambda X: np.maximum(0.0, X @ w + b)

def gen_pairwise_max(d, rng):
    i = rng.randint(0, d)
    j = rng.randint(0, d)
    return ('pairmax', i, j), lambda X: np.maximum(X[:, i], X[:, j])

def gen_sign_diff(d, rng):
    i = rng.randint(0, d)
    j = rng.randint(0, d)
    return ('signdiff', i, j), lambda X: np.sign(X[:, i] - X[:, j])

MENU_BASE     = [gen_cos, gen_mod2]
MENU_EXPANDED = [gen_cos, gen_mod2, gen_relu, gen_pairwise_max, gen_sign_diff]


# ── Feature discovery ─────────────────────────────────────────────────────────

def discover(X_tr, y_tr, menu_gens, n_candidates=N_CANDIDATES,
             n_rounds=N_ROUNDS, k=K, seed=0, verbose=True):
    rng = np.random.RandomState(seed)
    X_aug = X_tr.copy()
    baseline = loo_acc(X_aug, y_tr, k)
    found = []   # list of (params, apply_fn) — apply_fn works on the augmented X at discovery time

    for rnd in range(n_rounds):
        best_delta, best_params, best_fn, best_col = 0.0, None, None, None

        for gen in menu_gens:
            for _ in range(n_candidates):
                params, fn = gen(X_aug.shape[1], rng)
                col = fn(X_aug).reshape(-1, 1)
                X_cand = np.hstack([X_aug, col])
                acc = loo_acc(X_cand, y_tr, k)
                delta = acc - baseline
                if delta > best_delta:
                    best_delta = delta
                    best_params, best_fn, best_col = params, fn, col

        if best_params is None or best_delta <= 0:
            if verbose: print(f"    round {rnd+1}: no improvement, stopping", flush=True)
            break

        X_aug = np.hstack([X_aug, best_col])
        baseline += best_delta
        found.append((best_params, best_fn))
        if verbose:
            print(f"    round {rnd+1}: +{best_delta*100:.1f}pp from {best_params[0]}  "
                  f"loo={baseline*100:.1f}%  d={X_aug.shape[1]}", flush=True)

    return found, baseline * 100


def apply_features(X, found):
    """Replay discovered features on new data. found = list of (params, fn)."""
    # NOTE: apply_fn was created on the training augmentation, so we must
    # re-derive it from params and apply to current augmented X.
    X_aug = X.copy()
    for params, _ in found:
        name = params[0]
        if name == 'cos':
            _, w, b = params; col = np.cos(X_aug @ w + b)
        elif name == 'mod2':
            _, w, b = params; col = (np.floor(np.abs(X_aug @ w + b)).astype(int) % 2).astype(float)
        elif name == 'relu':
            _, w, b = params; col = np.maximum(0.0, X_aug @ w + b)
        elif name == 'pairmax':
            _, i, j = params; col = np.maximum(X_aug[:, i], X_aug[:, j])
        elif name == 'signdiff':
            _, i, j = params; col = np.sign(X_aug[:, i] - X_aug[:, j])
        else:
            raise ValueError(name)
        X_aug = np.hstack([X_aug, col.reshape(-1, 1)])
    return X_aug


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(task_name, X_tr, y_tr, X_te, y_te):
    print(f"\n{'-'*60}", flush=True)
    print(f"Task: {task_name}  (n_train={len(X_tr)}, n_test={len(X_te)}, "
          f"d={X_tr.shape[1]}, classes={len(np.unique(y_tr))})", flush=True)

    baseline = topk_acc(X_tr, y_tr, X_te, y_te)
    print(f"  Baseline k-NN (no features): {baseline*100:.1f}%", flush=True)

    print(f"  [{'{cos,mod2}'}]", flush=True)
    t0 = time.time()
    found_base, loo_base = discover(X_tr, y_tr, MENU_BASE, seed=1)
    X_te_base = apply_features(X_te, found_base)
    X_tr_base = apply_features(X_tr, found_base)
    acc_base = topk_acc(X_tr_base, y_tr, X_te_base, y_te)
    print(f"  {'{cos,mod2}'} test: {acc_base*100:.1f}%  "
          f"({len(found_base)} feats, {time.time()-t0:.1f}s)", flush=True)

    print(f"  [{'{cos,mod2,relu,max,sign_diff}'}]", flush=True)
    t0 = time.time()
    found_exp, loo_exp = discover(X_tr, y_tr, MENU_EXPANDED, seed=1)
    X_te_exp = apply_features(X_te, found_exp)
    X_tr_exp = apply_features(X_tr, found_exp)
    acc_exp = topk_acc(X_tr_exp, y_tr, X_te_exp, y_te)
    print(f"  expanded test: {acc_exp*100:.1f}%  "
          f"({len(found_exp)} feats, {time.time()-t0:.1f}s)", flush=True)

    gain = (acc_exp - acc_base) * 100
    gap_from_perfect = (1.0 - acc_exp) * 100

    if gain >= GAP_KILL:
        verdict = f"PRIMITIVE GAP (expanded +{gain:.1f}pp)"
    elif acc_exp > acc_base:
        verdict = f"PARTIAL (expanded +{gain:.1f}pp, gap_to_perfect={gap_from_perfect:.1f}pp)"
    else:
        verdict = f"MECHANISM GAP (expanded did not help, gain={gain:.1f}pp)"

    print(f"  Gain from expansion: {gain:+.1f}pp  -> {verdict}", flush=True)

    return {
        'baseline': baseline * 100,
        'base_menu': acc_base * 100,
        'expanded': acc_exp * 100,
        'gain': gain,
        'gap_to_perfect': gap_from_perfect,
        'verdict': verdict,
        'feats_base': [p[0] for p, _ in found_base],
        'feats_exp': [p[0] for p, _ in found_exp],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Step 216 -- Comparison primitives for sort task", flush=True)
    print(f"k={K}, n_train={N_TRAIN}, n_test={N_TEST}, vocab={VOCAB}, "
          f"candidates={N_CANDIDATES}, rounds={N_ROUNDS}", flush=True)

    tasks = {
        'comparison':  (make_comparison(N_TRAIN, seed=10),  make_comparison(N_TEST, seed=11)),
        'sort_argmin': (make_sort_argmin(N_TRAIN, seed=20), make_sort_argmin(N_TEST, seed=21)),
        'sort_argmax': (make_sort_argmax(N_TRAIN, seed=30), make_sort_argmax(N_TEST, seed=31)),
        'sort_full':   (make_sort_full(N_TRAIN, seed=40),   make_sort_full(N_TEST, seed=41)),
    }

    results = {}
    for task_name, ((X_tr, y_tr), (X_te, y_te)) in tasks.items():
        results[task_name] = run_task(task_name, X_tr, y_tr, X_te, y_te)

    elapsed = time.time() - t_start

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 216 FINAL", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"{'Task':<15} {'Base kNN':>9} {'{cos,mod2}':>10} {'expanded':>10} "
          f"{'gain':>7} {'gap2perf':>9}", flush=True)
    print(f"{'-'*72}", flush=True)

    for task_name, r in results.items():
        print(f"{task_name:<15} {r['baseline']:>8.1f}% {r['base_menu']:>9.1f}% "
              f"{r['expanded']:>9.1f}% {r['gain']:>+6.1f}pp {r['gap_to_perfect']:>8.1f}pp",
              flush=True)

    print(f"\nVerdicts:", flush=True)
    for task_name, r in results.items():
        print(f"  {task_name}: {r['verdict']}", flush=True)

    print(f"\nFeatures selected:", flush=True)
    for task_name, r in results.items():
        print(f"  {task_name}: base={r['feats_base']}, exp={r['feats_exp']}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
