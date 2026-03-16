#!/usr/bin/env python3
"""
Step 314 -- Per-b weight learning. Leo mail 1379.

W[b,k] = 20 x 5 = 100 parameters.
For query from b-group b: use W[b,:] as per-slot weights on phi comparison.
Cross-class absorption: W[b_query, k] += LR * sum_c (phi_q[c*K+k] - phi_nearest[c*K+k])^2

Also runs global w[k] (5-dim) as auto_loop baseline (Leo's 88.0%).

Kill: per-b LOO <= 88.0%.
Success: per-b LOO > 88.0% AND >=3 b-groups have cosine < 0.9 between w vectors.
"""

import time
import numpy as np

K = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
PHI_DIM = MAX_CLASS * K   # 100
ALPHA = 0.01
LR_W = 0.1
N_PASSES = 10


# ── phi ────────────────────────────────────────────────────────────────────

def compute_phi(a_q, b_q, A, B, Y, excl=-1):
    phi = np.full(PHI_DIM, SENTINEL, dtype=np.float32)
    for c in range(MAX_CLASS):
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


def compute_phi_all_loo(A, B, Y):
    n = len(A)
    result = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for i in range(n):
        result[i] = compute_phi(A[i], B[i], A, B, Y, excl=i)
    return result


def make_w_full_global(w_k):
    """5-dim w_k → 100-dim w (tiled MAX_CLASS times)."""
    return np.tile(w_k, MAX_CLASS).astype(np.float32)


def make_w_full_per_b(W_bk, b_val):
    """W[b,k] row → 100-dim w for queries from b-group b_val."""
    return np.tile(W_bk[b_val - 1], MAX_CLASS).astype(np.float32)


# ── LOO evaluation ─────────────────────────────────────────────────────────

def loo_accuracy_global(phi_all, A, B, Y, A_cb, w_k):
    """Global w_k: same weight for all queries."""
    w_full = make_w_full_global(w_k)
    phi_sq_w = (phi_all ** 2 * w_full).sum(axis=1)
    C = (phi_all * w_full) @ phi_all.T
    D = phi_sq_w[:, None] + phi_sq_w[None, :] - 2 * C
    np.fill_diagonal(D, float('inf'))
    nearest = np.argmin(D, axis=1)
    return float((Y[nearest] == Y).sum()) / len(Y)


def loo_accuracy_per_b(phi_all, A, B, Y, W_bk):
    """Per-b weights: query from b-group b uses W[b,:]."""
    n = len(Y)
    correct = 0
    for i in range(n):
        w_full_i = make_w_full_per_b(W_bk, int(B[i]))
        diffs = phi_all - phi_all[i]
        dists = (diffs * diffs * w_full_i).sum(axis=1)
        dists[i] = float('inf')
        nearest = int(np.argmin(dists))
        if Y[nearest] == Y[i]:
            correct += 1
    return correct / n


def loo_per_b_group(phi_all, A, B, Y, W_bk):
    """Per-b-group accuracy breakdown."""
    accuracies = {}
    for b_val in range(1, TRAIN_MAX + 1):
        mask = (B == b_val)
        idxs = np.where(mask)[0]
        correct = 0
        for i in idxs:
            w_full_i = make_w_full_per_b(W_bk, b_val)
            diffs = phi_all - phi_all[i]
            dists = (diffs * diffs * w_full_i).sum(axis=1)
            dists[i] = float('inf')
            nearest = int(np.argmin(dists))
            if Y[nearest] == Y[i]:
                correct += 1
        accuracies[b_val] = correct / len(idxs)
    return accuracies


# ── Training loops ─────────────────────────────────────────────────────────

def run_global_loop(A0, B0, Y0, train_data):
    """Global w[k]: 5-dim weight vector, shared across all b-groups."""
    A = A0.copy()
    w_k = np.ones(K, dtype=np.float32)
    results = []

    phi_all = compute_phi_all_loo(A, B0, Y0)
    loo0 = loo_accuracy_global(phi_all, A, B0, Y0, A, w_k)
    results.append((1, loo0))
    print(f"  [global] Pass 1: LOO={loo0*100:.1f}%", flush=True)

    for pass_num in range(2, N_PASSES + 1):
        phi_all = compute_phi_all_loo(A, B0, Y0)
        w_full = make_w_full_global(w_k)
        n_cross = 0

        for idx, (a, b, y) in enumerate(train_data):
            phi_q = phi_all[idx]
            diffs = phi_all - phi_q
            dists = (diffs * diffs * w_full).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))

            if Y0[nearest] != y:
                n_cross += 1
                diff_sq = (phi_q - phi_all[nearest]) ** 2
                # Aggregate by k-slot across all classes
                for k_idx in range(K):
                    slots = [c * K + k_idx for c in range(MAX_CLASS)]
                    w_k[k_idx] += LR_W * diff_sq[slots].sum()

            A[nearest] = (1.0 - ALPHA) * A[nearest] + ALPHA * a

        phi_all = compute_phi_all_loo(A, B0, Y0)
        loo = loo_accuracy_global(phi_all, A, B0, Y0, A, w_k)
        results.append((pass_num, loo))
        print(f"  [global] Pass {pass_num}: LOO={loo*100:.1f}%  w_k={[f'{v:.2f}' for v in w_k[:K]]}"
              f"  cross={n_cross}", flush=True)

    return results, A, w_k


def run_per_b_loop(A0, B0, Y0, train_data):
    """Per-b W[b,k]: 20x5 weight matrix."""
    A = A0.copy()
    W_bk = np.ones((TRAIN_MAX, K), dtype=np.float32)  # (20, 5)
    results = []

    phi_all = compute_phi_all_loo(A, B0, Y0)
    loo0 = loo_accuracy_per_b(phi_all, A, B0, Y0, W_bk)
    results.append((1, loo0))
    print(f"  [per-b]  Pass 1: LOO={loo0*100:.1f}%", flush=True)

    for pass_num in range(2, N_PASSES + 1):
        phi_all = compute_phi_all_loo(A, B0, Y0)
        n_cross = 0

        for idx, (a, b, y) in enumerate(train_data):
            b_idx = b - 1
            w_full_b = make_w_full_per_b(W_bk, b)
            phi_q = phi_all[idx]
            diffs = phi_all - phi_q
            dists = (diffs * diffs * w_full_b).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))

            if Y0[nearest] != y:
                n_cross += 1
                diff_sq = (phi_q - phi_all[nearest]) ** 2
                # Per-b gradient: aggregate by k-slot
                for k_idx in range(K):
                    slots = [c * K + k_idx for c in range(MAX_CLASS)]
                    W_bk[b_idx, k_idx] += LR_W * diff_sq[slots].sum()

            A[nearest] = (1.0 - ALPHA) * A[nearest] + ALPHA * a

        phi_all = compute_phi_all_loo(A, B0, Y0)
        loo = loo_accuracy_per_b(phi_all, A, B0, Y0, W_bk)
        results.append((pass_num, loo))
        print(f"  [per-b]  Pass {pass_num}: LOO={loo*100:.1f}%  cross={n_cross}", flush=True)

    return results, A, W_bk


def cosine_sim(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-8 or nv < 1e-8:
        return 1.0
    return float(np.dot(u, v) / (nu * nv))


def main():
    t0 = time.time()
    print("Step 314 -- Per-b weight learning", flush=True)
    print(f"N passes={N_PASSES}  alpha={ALPHA}  lr_w={LR_W}  K={K}", flush=True)
    print(f"Kill: per-b LOO <= 88.0% | Success: per-b LOO > 88.0% AND >=3 b-groups diverse", flush=True)
    print(flush=True)

    # Training data
    train_data = []
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((float(a), b, y))
            A_list.append(float(a)); B_list.append(b); Y_list.append(y)
    A0 = np.array(A_list, dtype=np.float32)
    B0 = np.array(B_list, dtype=np.int32)
    Y0 = np.array(Y_list, dtype=np.int32)

    # ── Global w[k] loop ───────────────────────────────────────────────────
    print("=== Global w[k] loop (auto_loop baseline) ===", flush=True)
    t1 = time.time()
    res_global, A_global, w_k_final = run_global_loop(A0, B0, Y0, train_data)
    global_final_loo = res_global[-1][1]
    print(f"  Global final LOO: {global_final_loo*100:.1f}%  "
          f"w_k={[f'{v:.2f}' for v in w_k_final]}  [{time.time()-t1:.1f}s]\n", flush=True)

    # ── Per-b W[b,k] loop ─────────────────────────────────────────────────
    print("=== Per-b W[b,k] loop ===", flush=True)
    t2 = time.time()
    res_perb, A_perb, W_bk_final = run_per_b_loop(A0, B0, Y0, train_data)
    perb_final_loo = res_perb[-1][1]
    print(f"  Per-b final LOO: {perb_final_loo*100:.1f}%  [{time.time()-t2:.1f}s]\n", flush=True)

    # ── Per-b analysis ─────────────────────────────────────────────────────
    print("=== Per-b weight structure ===", flush=True)
    phi_final = compute_phi_all_loo(A_perb, B0, Y0)
    per_b_acc = loo_per_b_group(phi_final, A_perb, B0, Y0, W_bk_final)

    print(f"  {'b':>3} | {'LOO':>7} | {'w_k (normalized)':>40} | note", flush=True)
    print("  " + "-" * 75, flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        w_b = W_bk_final[b_val - 1]
        w_norm = w_b / (w_b.sum() + 1e-8)
        n_classes = len(set(Y0[B0 == b_val]))
        note = f"n_cls={n_classes}"
        print(f"  {b_val:>3} | {per_b_acc[b_val]*100:>6.1f}% | "
              f"{[f'{v:.2f}' for v in w_norm]} | {note}", flush=True)

    # Cosine diversity: how many b-group pairs have cosine < 0.9?
    w_vectors = [W_bk_final[b - 1] for b in range(1, TRAIN_MAX + 1)]
    diverse_pairs = 0
    total_pairs = 0
    for i in range(TRAIN_MAX):
        for j in range(i + 1, TRAIN_MAX):
            cs = cosine_sim(w_vectors[i], w_vectors[j])
            total_pairs += 1
            if cs < 0.9:
                diverse_pairs += 1

    # Count b-groups with cosine < 0.9 vs at least one other
    diverse_groups = set()
    for i in range(TRAIN_MAX):
        for j in range(i + 1, TRAIN_MAX):
            cs = cosine_sim(w_vectors[i], w_vectors[j])
            if cs < 0.9:
                diverse_groups.add(i + 1)
                diverse_groups.add(j + 1)

    print(flush=True)
    print(f"  Diverse pairs (cosine<0.9): {diverse_pairs}/{total_pairs}", flush=True)
    print(f"  B-groups with at least one diverse pair: {len(diverse_groups)}", flush=True)
    print(flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    BASELINE_GLOBAL = 0.880   # Leo's auto_loop reference
    kill = perb_final_loo <= BASELINE_GLOBAL
    success = perb_final_loo > BASELINE_GLOBAL and len(diverse_groups) >= 3

    print("=" * 65, flush=True)
    print("STEP 314 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Global w[k] final LOO:   {global_final_loo*100:.1f}%  (Leo ref: 88.0%)", flush=True)
    print(f"Per-b W[b,k] final LOO:  {perb_final_loo*100:.1f}%", flush=True)
    print(f"Diverse b-groups:        {len(diverse_groups)}/20", flush=True)
    print(flush=True)
    print(f"Kill (per-b <= 88.0%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (>88.0% AND >=3 diverse): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Per-b specialization discovers diverse structure.", flush=True)
    elif kill:
        print("KILLED -- Per-b adds nothing over global weights.", flush=True)
    else:
        print("PARTIAL -- Above kill, below success.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
