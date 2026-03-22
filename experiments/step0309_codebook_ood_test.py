#!/usr/bin/env python3
"""
Step 309 -- OOD test for learned w. Spec.

Three experiments:
1. OOD on a%b: a in 21..50, b in 1..20. Use 308b w at pass 20.
   Compare: Step 300 (95.2%), Step 297 (18%).
2. floor(a/b): fresh training with same mechanism. Compare: Step 302 (86.8%).
3. w stability: normalize w, top-10 rankings at passes 5, 10, 20.

Kill: OOD accuracy < in-dist - 30pp.
Success: OOD within 10pp of in-dist AND floor(a/b) >= 80%.
"""

import time
import numpy as np

K = 5
MAX_CLASS = 20       # a%b classes: 0..19
MAX_CLASS_F = 21     # floor(a/b) classes: 0..20 (floor(20/1)=20)
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
ALPHA = 0.01
LR_W = 0.1
N_PASSES_EXT = 20
PHI_DIM = MAX_CLASS * K       # 100
PHI_DIM_F = MAX_CLASS_F * K   # 105


# ── phi computation ────────────────────────────────────────────────────────

def compute_phi(a_q, b_q, A, B, Y, max_class, phi_dim, excl=-1):
    phi = np.full(phi_dim, SENTINEL, dtype=np.float32)
    for c in range(max_class):
        mask = (B == b_q) & (Y == c)
        if excl >= 0 and excl < len(mask) and mask[excl]:
            mask = mask.copy()
            mask[excl] = False
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - a_q).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(idxs))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def compute_phi_all_loo(A, B, Y, max_class, phi_dim):
    n = len(A)
    result = np.full((n, phi_dim), SENTINEL, dtype=np.float32)
    for i in range(n):
        result[i] = compute_phi(A[i], B[i], A, B, Y, max_class, phi_dim, excl=i)
    return result


def loo_accuracy(A, B, Y, max_class, phi_dim, w=None):
    phi_all = compute_phi_all_loo(A, B, Y, max_class, phi_dim)
    n = len(A)
    correct = 0
    for i in range(n):
        diffs = phi_all - phi_all[i]
        if w is not None:
            dists = (diffs * diffs * w).sum(axis=1)
        else:
            dists = (diffs * diffs).sum(axis=1)
        dists[i] = float('inf')
        nearest = int(np.argmin(dists))
        if Y[nearest] == Y[i]:
            correct += 1
    return correct / n


def ranking_dims(w):
    """Top-10 dim indices by normalized weight."""
    w_norm = w / (np.linalg.norm(w) + 1e-8)
    return list(int(i) for i in np.argsort(w_norm)[-10:][::-1])


# ── 308b-style training extended to n_passes ──────────────────────────────

def run_extended(A0, B0, Y0, train_data, n_passes, max_class, phi_dim, label=""):
    A = A0.copy()
    w = np.ones(phi_dim, dtype=np.float32)
    results = []
    w_rankings = {}

    loo0 = loo_accuracy(A, B0, Y0, max_class, phi_dim, w=w)
    results.append((1, float(loo0), float(np.var(w))))
    print(f"  [{label}] Pass 1: LOO={loo0*100:.1f}%  w_var={np.var(w):.4f}", flush=True)

    for pass_num in range(2, n_passes + 1):
        phi_all_loo = compute_phi_all_loo(A, B0, Y0, max_class, phi_dim)
        n_same = 0
        n_cross = 0

        for idx, (a, b, y) in enumerate(train_data):
            phi_q = phi_all_loo[idx]
            diffs = phi_all_loo - phi_q
            dists = (diffs * diffs * w).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))

            if Y0[nearest] == y:
                n_same += 1
            else:
                n_cross += 1
                diff_sq = (phi_q - phi_all_loo[nearest]) ** 2
                w += LR_W * diff_sq

            A[nearest] = (1.0 - ALPHA) * A[nearest] + ALPHA * a

        loo = loo_accuracy(A, B0, Y0, max_class, phi_dim, w=w)
        w_var = float(np.var(w))
        results.append((pass_num, float(loo), w_var))
        print(f"  [{label}] Pass {pass_num}: LOO={loo*100:.1f}%  w_var={w_var:.1f}"
              f"  same={n_same} cross={n_cross}", flush=True)

        if pass_num in (5, 10, 20):
            w_rankings[pass_num] = ranking_dims(w)

    return results, A, w, w_rankings


# ── OOD evaluation ─────────────────────────────────────────────────────────

def eval_ood(A_cb, B_cb, Y_cb, w, phi_cb, ood_data, max_class, phi_dim):
    """Evaluate OOD queries against trained codebook + w."""
    correct = 0
    for (a_q, b_q, y_true) in ood_data:
        phi_q = compute_phi(float(a_q), b_q, A_cb, B_cb, Y_cb,
                            max_class, phi_dim, excl=-1)
        diffs = phi_cb - phi_q
        dists = (diffs * diffs * w).sum(axis=1)
        nearest = int(np.argmin(dists))
        if Y_cb[nearest] == y_true:
            correct += 1
    return correct / len(ood_data)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 309 -- OOD test for learned w", flush=True)
    print(f"N passes={N_PASSES_EXT}  alpha={ALPHA}  lr_w={LR_W}  K={K}", flush=True)
    print(flush=True)

    # a%b training data
    train_mod, A_list, B_list, Y_list = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_mod.append((float(a), b, y))
            A_list.append(float(a)); B_list.append(b); Y_list.append(y)
    A0 = np.array(A_list, dtype=np.float32)
    B0 = np.array(B_list, dtype=np.int32)
    Y0 = np.array(Y_list, dtype=np.int32)

    # OOD data: a%b, a in 21..50
    ood_data = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    # floor(a/b) training data
    train_floor, Af_list, Bf_list, Yf_list = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a // b
            train_floor.append((float(a), b, y))
            Af_list.append(float(a)); Bf_list.append(b); Yf_list.append(y)
    A0f = np.array(Af_list, dtype=np.float32)
    B0f = np.array(Bf_list, dtype=np.int32)
    Y0f = np.array(Yf_list, dtype=np.int32)

    # ─── Exp 1+3: a%b — 20 passes ─────────────────────────────────────────
    print("=== Exp 1+3: a%b training (20 passes) ===", flush=True)
    t1 = time.time()
    res_mod, A_mod, w_mod, w_rnk_mod = run_extended(
        A0, B0, Y0, train_mod, N_PASSES_EXT, MAX_CLASS, PHI_DIM, label="mod"
    )
    in_dist_loo = res_mod[-1][1]
    print(f"  In-dist LOO pass 20: {in_dist_loo*100:.1f}%  [{time.time()-t1:.1f}s]\n", flush=True)

    # ─── Exp 1: OOD ───────────────────────────────────────────────────────
    print("=== Exp 1: OOD on a%b (a=21..50) ===", flush=True)
    phi_cb = compute_phi_all_loo(A_mod, B0, Y0, MAX_CLASS, PHI_DIM)
    ood_acc = eval_ood(A_mod, B0, Y0, w_mod, phi_cb, ood_data, MAX_CLASS, PHI_DIM)
    delta_ood = ood_acc - in_dist_loo
    kill_ood = ood_acc < in_dist_loo - 0.30
    print(f"  OOD accuracy:          {ood_acc*100:.1f}%", flush=True)
    print(f"  In-dist LOO:           {in_dist_loo*100:.1f}%", flush=True)
    print(f"  Delta (OOD - in-dist): {delta_ood*100:+.1f}pp", flush=True)
    print(f"  Step 300 ref: 95.2%  Step 297 baseline: 18%", flush=True)
    print(f"  Kill criterion (< in-dist - 30pp): {'TRIGGERED' if kill_ood else 'not triggered'}", flush=True)
    print(flush=True)

    # ─── Exp 3: w stability ────────────────────────────────────────────────
    print("=== Exp 3: w stability (normalized top-10 rankings) ===", flush=True)
    for p in [5, 10, 20]:
        if p in w_rnk_mod:
            loo_at_p = res_mod[p - 1][1]
            dims = w_rnk_mod[p]
            classes = [d // K for d in dims]
            print(f"  Pass {p:>2} (LOO={loo_at_p*100:.1f}%): dims={dims}", flush=True)
            print(f"           classes={classes}", flush=True)
    if 10 in w_rnk_mod and 20 in w_rnk_mod:
        overlap = len(set(w_rnk_mod[10]) & set(w_rnk_mod[20]))
        print(f"  Top-10 overlap pass 10 vs 20: {overlap}/10", flush=True)
    if 5 in w_rnk_mod and 20 in w_rnk_mod:
        overlap5 = len(set(w_rnk_mod[5]) & set(w_rnk_mod[20]))
        print(f"  Top-10 overlap pass  5 vs 20: {overlap5}/10", flush=True)
    print(flush=True)

    # ─── Exp 2: floor(a/b) ────────────────────────────────────────────────
    print("=== Exp 2: floor(a/b) fresh training (10 passes) ===", flush=True)
    print(f"  MAX_CLASS_F={MAX_CLASS_F}  PHI_DIM_F={PHI_DIM_F}", flush=True)
    t2 = time.time()
    res_floor, _, w_floor, _ = run_extended(
        A0f, B0f, Y0f, train_floor, 10, MAX_CLASS_F, PHI_DIM_F, label="floor"
    )
    floor_loo = res_floor[-1][1]
    print(f"  floor(a/b) LOO pass 10: {floor_loo*100:.1f}%  [{time.time()-t2:.1f}s]", flush=True)
    print(f"  Step 302 baseline: 86.8%", flush=True)
    print(flush=True)

    # ─── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    ood_within_10 = abs(delta_ood) <= 0.10
    floor_ok = floor_loo >= 0.80
    success = (not kill_ood) and ood_within_10 and floor_ok

    print("=" * 65, flush=True)
    print("STEP 309 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"In-dist a%b LOO (pass 20):  {in_dist_loo*100:.1f}%", flush=True)
    print(f"OOD a%b accuracy:            {ood_acc*100:.1f}%", flush=True)
    print(f"OOD delta:                   {delta_ood*100:+.1f}pp", flush=True)
    print(f"floor(a/b) LOO (pass 10):   {floor_loo*100:.1f}%", flush=True)
    print(flush=True)
    print(f"Kill (OOD < in-dist - 30pp): {'TRIGGERED' if kill_ood else 'not triggered'}", flush=True)
    print(f"OOD within 10pp:             {ood_within_10}", flush=True)
    print(f"floor >= 80%:                {floor_ok}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- w generalizes. Substrate discovers metric structure.", flush=True)
    elif kill_ood:
        print("KILLED -- OOD gap > 30pp. 308b improvement is memorization.", flush=True)
    elif ood_within_10 and not floor_ok:
        print(f"PARTIAL -- OOD generalizes but floor(a/b)={floor_loo*100:.1f}% < 80%.", flush=True)
    elif not ood_within_10 and not kill_ood:
        print(f"PARTIAL -- OOD gap {delta_ood*100:+.1f}pp. Above kill, below success.", flush=True)
    else:
        print("PARTIAL -- Mixed results.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
