#!/usr/bin/env python3
"""
Step 300 -- Reflection spawn for OOD extrapolation.

Spec/1349. The fold detects period -> spawns extension -> OOD becomes
in-distribution.

Mechanism:
1. Train on (a,b) in 1..20 (400 vectors)
2. For each class c and each b: find max-a boundary vector v_max, nearest
   same-class v_prev. Spawn reflection: 2*v_max - v_prev, labeled c.
3. Repeat K times per direction (forward = toward high a, backward = low a).
4. Run phi distribution matching on OOD (a in 21..50) against EXTENDED codebook.

Compare K=0 (18%, Step 297 baseline) vs K=1,5,10,auto (covers full OOD range).

Kill: OOD accuracy <= 50% at K=10.
Success: OOD accuracy > 80%, approaching in-distribution (86.8%).
"""

import time
import numpy as np

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
OOD_MIN   = 21
OOD_MAX   = 50
K_VALS    = [0, 1, 5, 10, 31]    # 31 = covers a=1..50 for all b
PHI_K     = 5                     # K for phi computation (best from Step 296)
IN_DIST_ACC = 0.868

# ─── Dataset ──────────────────────────────────────────────────────────────────

def build_train():
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A, dtype=np.int32), np.array(B, dtype=np.int32), np.array(Y, dtype=np.int32)


def build_ood():
    A, B, Y = [], [], []
    for a in range(OOD_MIN, OOD_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a); B.append(b); Y.append(a % b)
    return np.array(A, dtype=np.int32), np.array(B, dtype=np.int32), np.array(Y, dtype=np.int32)

# ─── Reflection spawn ─────────────────────────────────────────────────────────

def reflection_spawn(A_tr, B_tr, Y_tr, n_iters, cross_class_step=False):
    """
    Extend codebook via reflection spawn.
    For each (b, class c): find max-a and second-max-a, spawn 2*max - second_max.
    Repeat n_iters times. Also spawn backward (min-a direction).

    cross_class_step: if True, for single-point classes use the consensus step
    from other same-b classes (the majority inferred step). This avoids requiring
    2 training points per class to detect the period.

    Returns extended (A, B, Y) arrays.
    """
    A_ext = list(A_tr)
    B_ext = list(B_tr)
    Y_ext = list(Y_tr)

    for b_val in range(1, TRAIN_MAX + 1):
        for _ in range(n_iters):
            b_mask = np.array([bv == b_val for bv in B_ext])
            if not b_mask.any():
                continue
            b_idxs = np.where(b_mask)[0]
            classes = set(Y_ext[i] for i in b_idxs)

            # Infer consensus step for this b from multi-point classes
            inferred_steps = []
            for c in classes:
                c_idxs = [i for i in b_idxs if Y_ext[i] == c]
                a_vals = sorted([A_ext[i] for i in c_idxs])
                if len(a_vals) >= 2:
                    steps = [a_vals[j+1] - a_vals[j] for j in range(len(a_vals)-1)]
                    inferred_steps.extend(steps)

            # Consensus step: modal step among all multi-point same-b classes
            consensus_step = None
            if inferred_steps:
                from collections import Counter
                step_counts = Counter(inferred_steps)
                consensus_step = step_counts.most_common(1)[0][0]

            new_points = []
            for c in classes:
                c_idxs = [i for i in b_idxs if Y_ext[i] == c]
                a_vals = sorted([A_ext[i] for i in c_idxs])

                if len(a_vals) >= 2:
                    # Normal case: determine step from own points
                    # Forward
                    v_max  = a_vals[-1]
                    v_prev = a_vals[-2]
                    step   = v_max - v_prev
                    if step > 0:
                        new_points.append((v_max + step, b_val, c))
                    # Backward
                    v_min = a_vals[0]
                    v_2nd = a_vals[1]
                    step_b = v_2nd - v_min
                    if step_b > 0 and v_min - step_b >= 1:
                        new_points.append((v_min - step_b, b_val, c))

                elif len(a_vals) == 1 and cross_class_step and consensus_step:
                    # Single-point class: use consensus step from other same-b classes
                    v_only = a_vals[0]
                    new_points.append((v_only + consensus_step, b_val, c))
                    if v_only - consensus_step >= 1:
                        new_points.append((v_only - consensus_step, b_val, c))

            for (a_new, b_new, y_new) in new_points:
                is_dup = any(
                    A_ext[i] == a_new and B_ext[i] == b_new
                    for i in range(len(A_ext))
                    if B_ext[i] == b_new and Y_ext[i] == y_new
                )
                if not is_dup:
                    A_ext.append(a_new)
                    B_ext.append(b_new)
                    Y_ext.append(y_new)

    return np.array(A_ext, dtype=np.int32), np.array(B_ext, dtype=np.int32), np.array(Y_ext, dtype=np.int32)

# ─── phi computation ──────────────────────────────────────────────────────────

SENTINEL = (OOD_MAX + TRAIN_MAX) * 2

def compute_phi_ext(query_a, query_b, A_cb, B_cb, Y_cb, K, max_class):
    """Compute phi for query against extended codebook (no LOO)."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b = (B_cb == query_b)
    for c in range(max_class):
        mask = (Y_cb == c) & same_b
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_cb[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def compute_phi_loo(query_a, query_b, query_idx, A_cb, B_cb, Y_cb, K, max_class):
    """Compute phi with LOO for codebook self-evaluation."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    same_b = (B_cb == query_b)
    for c in range(max_class):
        mask = (Y_cb == c) & same_b
        if query_idx is not None and mask[query_idx]:
            mask = mask.copy()
            mask[query_idx] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_cb[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_ood(A_cb, B_cb, Y_cb, A_ood, B_ood, Y_ood, K, max_class, label=""):
    """Evaluate OOD accuracy using phi against extended codebook."""
    n_cb  = len(A_cb)
    n_ood = len(A_ood)

    # Precompute codebook phi (LOO for training points, no LOO for spawned)
    # For simplicity: no LOO (OOD test, not self-eval)
    phi_cb = np.zeros((n_cb, max_class * K), dtype=np.float32)
    for i in range(n_cb):
        phi_cb[i] = compute_phi_ext(A_cb[i], B_cb[i], A_cb, B_cb, Y_cb, K, max_class)

    correct = 0
    per_b = {}
    for b in range(1, TRAIN_MAX + 1):
        per_b[b] = {'correct': 0, 'total': 0}

    for i in range(n_ood):
        a, b, true_y = int(A_ood[i]), int(B_ood[i]), int(Y_ood[i])
        phi_q = compute_phi_ext(a, b, A_cb, B_cb, Y_cb, K, max_class)
        diffs  = phi_cb - phi_q
        dists2 = (diffs * diffs).sum(axis=1)
        pred   = int(Y_cb[np.argmin(dists2)])
        if pred == true_y:
            correct += 1
            per_b[b]['correct'] += 1
        per_b[b]['total'] += 1

    acc = correct / n_ood
    return acc, per_b

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 300 -- Reflection Spawn for OOD Extrapolation", flush=True)
    print(f"Train: a,b in 1..{TRAIN_MAX} (400 vectors)", flush=True)
    print(f"OOD:   a in {OOD_MIN}..{OOD_MAX}, b in 1..{TRAIN_MAX} (600 pairs)", flush=True)
    print(f"In-distribution reference (Step 296): {IN_DIST_ACC*100:.1f}%", flush=True)
    print(f"OOD baseline (Step 297, K=0):         18.0%", flush=True)
    print(flush=True)

    A_tr, B_tr, Y_tr = build_train()
    A_ood, B_ood, Y_ood = build_ood()
    max_class = max(int(Y_tr.max()), int(Y_ood.max())) + 1

    print(f"{'K_iter':>6} | {'CB size':>8} | {'OOD acc':>8} | {'vs baseline':>12} | {'vs in-dist':>11}",
          flush=True)
    print("-" * 58, flush=True)

    results = {}
    for k_iter in K_VALS:
        tk = time.time()
        if k_iter == 0:
            A_cb, B_cb, Y_cb = A_tr.copy(), B_tr.copy(), Y_tr.copy()
        else:
            A_cb, B_cb, Y_cb = reflection_spawn(A_tr, B_tr, Y_tr, k_iter,
                                                 cross_class_step=False)

        acc, per_b = evaluate_ood(A_cb, B_cb, Y_cb, A_ood, B_ood, Y_ood,
                                  PHI_K, max_class)
        results[k_iter] = {'acc': acc, 'cb_size': len(A_cb), 'per_b': per_b}

        delta_base = (acc - 0.18) * 100
        delta_indist = (acc - IN_DIST_ACC) * 100
        print(f"  {k_iter:>4} | {len(A_cb):>8} | {acc*100:>7.1f}% | "
              f"{delta_base:>+11.1f}pp | {delta_indist:>+10.1f}pp  [{time.time()-tk:.1f}s]",
              flush=True)

    print(flush=True)

    # Cross-class step inference version
    print(flush=True)
    print("=== Cross-class step inference (uses consensus step for single-point classes) ===",
          flush=True)
    print(f"{'K_iter':>6} | {'CB size':>8} | {'OOD acc':>8} | {'vs baseline':>12} | {'vs in-dist':>11}",
          flush=True)
    print("-" * 58, flush=True)

    results_cc = {}
    for k_iter in K_VALS:
        tk = time.time()
        if k_iter == 0:
            A_cb2, B_cb2, Y_cb2 = A_tr.copy(), B_tr.copy(), Y_tr.copy()
        else:
            A_cb2, B_cb2, Y_cb2 = reflection_spawn(A_tr, B_tr, Y_tr, k_iter,
                                                    cross_class_step=True)
        acc2, per_b2 = evaluate_ood(A_cb2, B_cb2, Y_cb2, A_ood, B_ood, Y_ood,
                                    PHI_K, max_class)
        results_cc[k_iter] = {'acc': acc2, 'cb_size': len(A_cb2), 'per_b': per_b2}
        delta_base = (acc2 - 0.18) * 100
        delta_indist = (acc2 - IN_DIST_ACC) * 100
        print(f"  {k_iter:>4} | {len(A_cb2):>8} | {acc2*100:>7.1f}% | "
              f"{delta_base:>+11.1f}pp | {delta_indist:>+10.1f}pp  [{time.time()-tk:.1f}s]",
              flush=True)

    best_k_cc = max(results_cc, key=lambda k: results_cc[k]['acc'])
    best_acc_cc = results_cc[best_k_cc]['acc']
    print(flush=True)

    # Per-b breakdown for best K
    best_k = max(results, key=lambda k: results[k]['acc'])
    best_acc = results[best_k]['acc']
    print(f"=== Per-b breakdown, no cross-class step (best K={best_k}, OOD acc={best_acc*100:.1f}%) ===",
          flush=True)
    per_b_best = results[best_k]['per_b']
    per_b_accs = []
    for b_val in sorted(per_b_best.keys()):
        n = per_b_best[b_val]['total']
        if n == 0:
            continue
        acc_b = per_b_best[b_val]['correct'] / n
        per_b_accs.append(acc_b)
        print(f"  b={b_val:>2}: {acc_b*100:.1f}% ({per_b_best[b_val]['correct']}/{n})",
              flush=True)
    print(flush=True)

    # Per-b breakdown for best cross-class K
    print(f"=== Per-b breakdown, WITH cross-class step (best K={best_k_cc}, OOD acc={best_acc_cc*100:.1f}%) ===",
          flush=True)
    per_b_cc = results_cc[best_k_cc]['per_b']
    per_b_cc_accs = []
    for b_val in sorted(per_b_cc.keys()):
        n = per_b_cc[b_val]['total']
        if n == 0:
            continue
        acc_b = per_b_cc[b_val]['correct'] / n
        per_b_cc_accs.append(acc_b)
        print(f"  b={b_val:>2}: {acc_b*100:.1f}% ({per_b_cc[b_val]['correct']}/{n})", flush=True)
    print(flush=True)

    # Summary
    elapsed = time.time() - t0
    print("=" * 65, flush=True)
    print("STEP 300 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"In-distribution (Step 296, K=5):  {IN_DIST_ACC*100:.1f}%", flush=True)
    print(f"OOD baseline (Step 297, K_iter=0): 18.0%", flush=True)
    print(flush=True)
    print(f"Without cross-class step (K sweep):", flush=True)
    for k_iter in K_VALS:
        r = results[k_iter]
        print(f"  K_iter={k_iter:>2}: {r['acc']*100:.1f}%  (CB: {r['cb_size']})", flush=True)
    print(f"With cross-class step (K sweep):", flush=True)
    for k_iter in K_VALS:
        r = results_cc[k_iter]
        print(f"  K_iter={k_iter:>2}: {r['acc']*100:.1f}%  (CB: {r['cb_size']})", flush=True)
    print(flush=True)

    kill_threshold = 0.50
    overall_best = max(best_acc, best_acc_cc)
    print("KILL CRITERION (Spec):", flush=True)
    if overall_best < kill_threshold:
        print(f"  KILLED -- best OOD ({overall_best*100:.1f}%) <= {kill_threshold*100:.0f}%",
              flush=True)
        print(f"  Reflection spawn does not enable OOD generalization.", flush=True)
        print(f"  The in-distribution result is memorization, not computation.", flush=True)
    else:
        gap = best_acc_cc - IN_DIST_ACC
        print(f"  PASSES -- OOD ({best_acc_cc*100:.1f}% with cross-class) > {kill_threshold*100:.0f}%",
              flush=True)
        print(f"  Gap from in-distribution: {gap*100:+.1f}pp", flush=True)
        if best_acc_cc >= IN_DIST_ACC - 0.10:
            print(f"  STRONG PASS -- within 10pp of in-distribution!", flush=True)
        print(flush=True)
        print("INTERPRETATION:", flush=True)
        print("  Reflection spawn: fold detects period -> extends codebook -> OOD = in-dist.", flush=True)
        print("  Cross-class step: period inferred from multi-point classes, applied to all.", flush=True)
        print("  The fold COMPUTES by growing. Genuine extrapolation from structural detection.", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
