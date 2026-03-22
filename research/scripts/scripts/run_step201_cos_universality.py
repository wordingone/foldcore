#!/usr/bin/env python3
"""
Step 201 -- Cos-only primitive universality test.

Hypothesis (Random Kitchen Sinks): cos alone is universal.
If cos-only matches the full menu {cos, abs, mod2, sign, tanh} on
parity + XOR + multi-rule, the primitive set collapses to ONE element.

Kill: cos-only misses full menu by >3pp on any task = NOT universal.
Pass: cos-only within 3pp on ALL tasks = cos is sufficient.

[researcher] mail 1290.
"""

import numpy as np
import time

# ── Config ────────────────────────────────────────────────────────────────────

K            = 5
D            = 8        # binary input dimension
N_TRAIN      = 200      # training samples
N_TEST       = 400      # test samples
N_CANDIDATES = 300      # random candidates per template per round
N_ROUNDS     = 4        # feature discovery rounds
SEED         = 42
GAP_KILL     = 3.0      # pp: cos-only must be within this of full menu

rng = np.random.RandomState(SEED)


# ── Tasks ─────────────────────────────────────────────────────────────────────

def make_parity(n, d=D, seed=0):
    r = np.random.RandomState(seed)
    X = r.randint(0, 2, (n, d)).astype(np.float32)
    y = X.sum(axis=1).astype(int) % 2
    return X, y

def make_xor(n, d=D, seed=1):
    r = np.random.RandomState(seed)
    X = r.randint(0, 2, (n, d)).astype(np.float32)
    y = (X[:, 0].astype(int) ^ X[:, 1].astype(int))
    return X, y

def make_multi_rule(n, d=D, seed=2):
    """parity(x[:4]) XOR (x[4] AND x[5]) XOR parity(x[6:])"""
    r = np.random.RandomState(seed)
    X = r.randint(0, 2, (n, d)).astype(np.float32)
    p1 = X[:, :4].sum(axis=1).astype(int) % 2
    and_part = (X[:, 4].astype(int) & X[:, 5].astype(int))
    p2 = X[:, 6:].sum(axis=1).astype(int) % 2
    y = p1 ^ and_part ^ p2
    return X, y


# ── k-NN top-k (vectorized) ───────────────────────────────────────────────────

def norm(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n < 1e-10, 1.0, n)

def topk_acc(V, lv, Q, lq, k=K):
    S = norm(Q) @ norm(V).T               # (n_q, n_v)
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
    """Vectorized LOO accuracy on normalized X_aug."""
    Xn = norm(X_aug)
    S = Xn @ Xn.T                         # (n, n)
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

def apply_template(name, w, b, X):
    """Apply template to all rows of X. Returns (n,) array."""
    z = X @ w + b
    if name == 'cos':  return np.cos(z)
    if name == 'abs':  return np.abs(z)
    if name == 'mod2': return (np.floor(np.abs(z)).astype(int) % 2).astype(float)
    if name == 'sign': return np.sign(z)
    if name == 'tanh': return np.tanh(z)
    raise ValueError(name)

FULL_MENU = ['cos', 'abs', 'mod2', 'sign', 'tanh']
COS_ONLY  = ['cos']


# ── Feature discovery ─────────────────────────────────────────────────────────

def discover(X_tr, y_tr, menu, n_candidates=N_CANDIDATES,
             n_rounds=N_ROUNDS, k=K, seed=0, verbose=True):
    """
    LOO-scored random feature discovery.
    Returns augmented X_tr and list of (name, w, b) discovered features.
    """
    r = np.random.RandomState(seed)
    X_aug = X_tr.copy()
    baseline = loo_acc(X_aug, y_tr, k)
    found = []

    for rnd in range(n_rounds):
        best_delta, best_feat, best_col = 0.0, None, None

        for name in menu:
            for _ in range(n_candidates):
                d_cur = X_aug.shape[1]
                w = r.randn(d_cur)
                b = r.uniform(0, 2 * np.pi) if name == 'cos' else r.randn()
                col = apply_template(name, w, b, X_aug).reshape(-1, 1)
                X_cand = np.hstack([X_aug, col])
                acc = loo_acc(X_cand, y_tr, k)
                delta = acc - baseline
                if delta > best_delta:
                    best_delta, best_feat, best_col = delta, (name, w, b), col

        if best_feat is None or best_delta <= 0:
            if verbose: print(f"    round {rnd+1}: no improvement, stopping", flush=True)
            break

        name, w, b = best_feat
        X_aug = np.hstack([X_aug, best_col])
        baseline += best_delta
        found.append((name, w, b))
        if verbose:
            print(f"    round {rnd+1}: +{best_delta*100:.1f}pp from {name}  "
                  f"loo={baseline*100:.1f}%  d={X_aug.shape[1]}", flush=True)

    return X_aug, found


def eval_with_features(X_tr, y_tr, X_te, y_te, features, k=K):
    """Apply discovered features to test data, then classify."""
    X_aug_tr = X_tr.copy()
    X_aug_te = X_te.copy()
    for name, w, b in features:
        X_aug_tr = np.hstack([X_aug_tr, apply_template(name, w, b, X_aug_tr).reshape(-1, 1)])
        X_aug_te = np.hstack([X_aug_te, apply_template(name, w, b, X_aug_te).reshape(-1, 1)])
    return topk_acc(X_aug_tr, y_tr, X_aug_te, y_te, k)


# ── Run one task ──────────────────────────────────────────────────────────────

def run_task(name, X_tr, y_tr, X_te, y_te):
    print(f"\n{'-'*60}", flush=True)
    print(f"Task: {name}", flush=True)

    baseline = topk_acc(X_tr, y_tr, X_te, y_te)
    print(f"  Baseline k-NN: {baseline*100:.1f}%", flush=True)

    print(f"  [Full menu]", flush=True)
    t0 = time.time()
    X_aug_tr, feats_full = discover(X_tr, y_tr, FULL_MENU, seed=1)
    full_acc = eval_with_features(X_tr, y_tr, X_te, y_te, feats_full)
    print(f"  Full menu test acc: {full_acc*100:.1f}%  ({len(feats_full)} features, {time.time()-t0:.1f}s)", flush=True)

    print(f"  [Cos-only]", flush=True)
    t0 = time.time()
    X_aug_tr, feats_cos = discover(X_tr, y_tr, COS_ONLY, seed=1)
    cos_acc = eval_with_features(X_tr, y_tr, X_te, y_te, feats_cos)
    print(f"  Cos-only test acc: {cos_acc*100:.1f}%  ({len(feats_cos)} features, {time.time()-t0:.1f}s)", flush=True)

    gap = (full_acc - cos_acc) * 100
    verdict = "PASS (universal)" if gap <= GAP_KILL else f"FAIL (gap={gap:.1f}pp)"
    print(f"  Gap: {gap:.1f}pp  -> {verdict}", flush=True)

    return {
        'baseline': baseline, 'full': full_acc, 'cos': cos_acc,
        'gap': gap, 'verdict': verdict,
        'feats_full': [f[0] for f in feats_full],
        'feats_cos': [f[0] for f in feats_cos],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Step 201 — Cos-only universality test", flush=True)
    print(f"d={D}, n_train={N_TRAIN}, n_test={N_TEST}, k={K}, "
          f"candidates={N_CANDIDATES}, rounds={N_ROUNDS}", flush=True)

    tasks = {
        'parity':     (make_parity(N_TRAIN, seed=10),     make_parity(N_TEST, seed=11)),
        'xor':        (make_xor(N_TRAIN, seed=20),        make_xor(N_TEST, seed=21)),
        'multi_rule': (make_multi_rule(N_TRAIN, seed=30), make_multi_rule(N_TEST, seed=31)),
    }

    results = {}
    for task_name, ((X_tr, y_tr), (X_te, y_te)) in tasks.items():
        results[task_name] = run_task(task_name, X_tr, y_tr, X_te, y_te)

    elapsed = time.time() - t_start

    print(f"\n{'='*72}", flush=True)
    print(f"STEP 201 FINAL", flush=True)
    print(f"{'='*72}", flush=True)
    print(f"{'Task':<15} {'Baseline':>10} {'Full menu':>10} {'Cos-only':>10} {'Gap':>8} {'Verdict'}", flush=True)
    print(f"{'-'*72}", flush=True)

    all_pass = True
    for task_name, r in results.items():
        print(f"{task_name:<15} {r['baseline']*100:>9.1f}% {r['full']*100:>9.1f}% "
              f"{r['cos']*100:>9.1f}% {r['gap']:>7.1f}pp  {r['verdict']}", flush=True)
        if r['gap'] > GAP_KILL:
            all_pass = False

    print(f"\nPrimitive set collapses to cos: {'YES' if all_pass else 'NO'}", flush=True)
    if all_pass:
        print("  -> Frozen element UNLOCKED. Remove primitive menu, keep cos.", flush=True)
    else:
        print("  -> Binding frozen: primitive menu not reducible to cos alone.", flush=True)

    print(f"\nFeature templates selected:", flush=True)
    for task_name, r in results.items():
        print(f"  {task_name}: full={r['feats_full']}, cos={r['feats_cos']}", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
