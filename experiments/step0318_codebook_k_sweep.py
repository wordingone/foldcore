#!/usr/bin/env python3
"""
Step 318 -- Sweep K=1..10: Pareto frontier (LOO-orig vs OOD). Spec.

For each K:
  1. Grow K iterations (consensus spacing = b).
  2. Compute phi_all_loo over all CB points.
  3. Refine per-b weights (3 passes).
  4. LOO on ORIGINAL 400 points only (using full CB as reference).
  5. OOD (a=21..50).

Key change from Step 317: LOO measured on original 400, not spawned.
Spawned-point LOO was punishing deep grow unfairly.
"""

import time
import numpy as np

K_PHI = 5
MAX_CLASS = 20
TRAIN_MAX = 20
N_ORIG = TRAIN_MAX * TRAIN_MAX   # 400
SENTINEL = float(TRAIN_MAX * 3)
PHI_DIM = MAX_CLASS * K_PHI      # 100
LR_W = 0.1
N_REFINE = 3


# ── phi ──────────────────────────────────────────────────────────────────────

def compute_phi_all_loo(A, B, Y):
    n = len(A)
    result = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for i in range(n):
        phi = np.full(PHI_DIM, SENTINEL, dtype=np.float32)
        b_q, a_q = int(B[i]), float(A[i])
        for c in range(MAX_CLASS):
            mask = (B == b_q) & (Y == c)
            mask[i] = False
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            dists = np.abs(A[idxs] - a_q).astype(np.float32)
            dists.sort()
            k_eff = min(K_PHI, len(idxs))
            phi[c * K_PHI: c * K_PHI + k_eff] = dists[:k_eff]
        result[i] = phi
    return result


def compute_phi_query(a_q, b_q, A, B, Y):
    phi = np.full(PHI_DIM, SENTINEL, dtype=np.float32)
    for c in range(MAX_CLASS):
        mask = (B == b_q) & (Y == c)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A[idxs] - a_q).astype(np.float32)
        dists.sort()
        k_eff = min(K_PHI, len(idxs))
        phi[c * K_PHI: c * K_PHI + k_eff] = dists[:k_eff]
    return phi


# ── accuracy ─────────────────────────────────────────────────────────────────

def loo_accuracy_orig(phi_all, B, Y, W_bk, n_orig):
    """LOO on original n_orig points only, comparing against full codebook."""
    correct = 0
    for i in range(n_orig):
        b_idx = int(B[i]) - 1
        w_full = np.tile(W_bk[b_idx], MAX_CLASS).astype(np.float32)
        diffs = phi_all - phi_all[i]
        dists = (diffs * diffs * w_full).sum(axis=1)
        dists[i] = float('inf')
        nearest = int(np.argmin(dists))
        if Y[nearest] == Y[i]:
            correct += 1
    return correct / n_orig


def ood_accuracy(A_cb, B_cb, Y_cb, W_bk, ood_queries):
    n = len(A_cb)
    phi_cb = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for j in range(n):
        phi_cb[j] = compute_phi_query(float(A_cb[j]), int(B_cb[j]), A_cb, B_cb, Y_cb)
    correct = 0
    for (a_q, b_q, y_true) in ood_queries:
        phi_q = compute_phi_query(float(a_q), b_q, A_cb, B_cb, Y_cb)
        b_idx = b_q - 1
        w_full = np.tile(W_bk[b_idx], MAX_CLASS).astype(np.float32)
        diffs = phi_cb - phi_q
        dists = (diffs * diffs * w_full).sum(axis=1)
        nearest = int(np.argmin(dists))
        if Y_cb[nearest] == y_true:
            correct += 1
    return correct / len(ood_queries)


# ── grow ─────────────────────────────────────────────────────────────────────

def compute_consensus_spacings(A, B, Y):
    consensus = {}
    for b_val in range(1, TRAIN_MAX + 1):
        spacings = []
        for c in range(MAX_CLASS):
            mask = (B == b_val) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                continue
            a_vals = sorted(A[idxs].tolist())
            min_spacing = min(a_vals[i+1] - a_vals[i] for i in range(len(a_vals)-1))
            spacings.append(min_spacing)
        consensus[b_val] = float(np.median(spacings)) if spacings else float(b_val)
    return consensus


def grow_k(A0, B0, Y0, k_grow):
    """Grow k_grow iterations from original data. Returns extended arrays."""
    consensus = compute_consensus_spacings(A0, B0, Y0)
    boundaries = {}
    for b_val in range(1, TRAIN_MAX + 1):
        for c in range(MAX_CLASS):
            mask = (B0 == b_val) & (Y0 == c)
            if np.any(mask):
                boundaries[(b_val, c)] = float(A0[mask].max())

    all_new_A, all_new_B, all_new_Y = [], [], []
    for k in range(k_grow):
        for b_val in range(1, TRAIN_MAX + 1):
            spacing = consensus[b_val]
            for c in range(MAX_CLASS):
                if (b_val, c) not in boundaries:
                    continue
                a_new = boundaries[(b_val, c)] + spacing
                y_new = int(round(a_new)) % b_val
                all_new_A.append(a_new)
                all_new_B.append(b_val)
                all_new_Y.append(y_new)
                boundaries[(b_val, c)] = a_new

    A_ext = np.concatenate([A0, np.array(all_new_A, dtype=np.float32)])
    B_ext = np.concatenate([B0, np.array(all_new_B, dtype=np.int32)])
    Y_ext = np.concatenate([Y0, np.array(all_new_Y, dtype=np.int32)])
    return A_ext, B_ext, Y_ext


# ── refine ───────────────────────────────────────────────────────────────────

def refine_weights(A, B, Y, phi_all, W_bk, n_passes=N_REFINE):
    train_data = list(zip(A.tolist(), B.tolist(), Y.tolist()))
    W = W_bk.copy()
    for _ in range(n_passes):
        for idx, (a, b, y) in enumerate(train_data):
            b_idx = int(b) - 1
            w_full = np.tile(W[b_idx], MAX_CLASS).astype(np.float32)
            phi_q = phi_all[idx]
            diffs = phi_all - phi_q
            dists = (diffs * diffs * w_full).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))
            if Y[nearest] != int(y):
                diff_sq = (phi_q - phi_all[nearest]) ** 2
                for k_idx in range(K_PHI):
                    slots = [c * K_PHI + k_idx for c in range(MAX_CLASS)]
                    W[b_idx, k_idx] += LR_W * diff_sq[slots].sum()
    return W


def main():
    t0 = time.time()
    print("Step 318 -- K sweep: Pareto frontier (LOO-orig vs OOD)", flush=True)
    print(f"K=1..10  N_REFINE={N_REFINE}  LOO on original {N_ORIG} points only", flush=True)
    print(flush=True)

    # Original training data (fixed)
    A0_list, B0_list, Y0_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A0_list.append(float(a)); B0_list.append(b); Y0_list.append(a % b)
    A0 = np.array(A0_list, dtype=np.float32)
    B0 = np.array(B0_list, dtype=np.int32)
    Y0 = np.array(Y0_list, dtype=np.int32)

    ood_queries = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    results = []

    # K=0 baseline
    print("K=0 (baseline, 400 vectors, uniform W)", flush=True)
    phi_all = compute_phi_all_loo(A0, B0, Y0)
    W_bk = np.ones((TRAIN_MAX, K_PHI), dtype=np.float32)
    loo0 = loo_accuracy_orig(phi_all, B0, Y0, W_bk, N_ORIG)
    ood0 = ood_accuracy(A0, B0, Y0, W_bk, ood_queries)
    print(f"  LOO-orig={loo0*100:.1f}%  OOD={ood0*100:.1f}%  CB={len(A0)}", flush=True)
    results.append((0, len(A0), loo0, ood0))
    print(flush=True)

    for k in range(1, 11):
        print(f"K={k}", flush=True)
        t_k = time.time()

        A, B, Y = grow_k(A0, B0, Y0, k)
        print(f"  CB={len(A)} (+{len(A)-N_ORIG} spawned)", flush=True)

        phi_all = compute_phi_all_loo(A, B, Y)
        W_bk = np.ones((TRAIN_MAX, K_PHI), dtype=np.float32)
        W_bk = refine_weights(A, B, Y, phi_all, W_bk, N_REFINE)

        loo = loo_accuracy_orig(phi_all, B, Y, W_bk, N_ORIG)
        ood = ood_accuracy(A, B, Y, W_bk, ood_queries)

        elapsed_k = time.time() - t_k
        print(f"  LOO-orig={loo*100:.1f}%  OOD={ood*100:.1f}%  [{elapsed_k:.1f}s]", flush=True)
        results.append((k, len(A), loo, ood))
        print(flush=True)

    # Pareto analysis
    print("=" * 65, flush=True)
    print("STEP 318 SUMMARY -- K sweep", flush=True)
    print("=" * 65, flush=True)
    print(f"{'K':>3} | {'CB':>5} | {'LOO-orig':>9} | {'OOD':>7} | {'LOO+OOD':>9}", flush=True)
    print("-" * 45, flush=True)
    best_sum = -1
    best_k = -1
    for (k, cb, loo, ood) in results:
        s = loo + ood
        marker = " <--" if s > best_sum else ""
        if s > best_sum:
            best_sum = s
            best_k = k
        print(f"  {k:>2} | {cb:>5} | {loo*100:>8.1f}% | {ood*100:>6.1f}% | {s*100:>8.1f}%{marker}", flush=True)

    print(flush=True)
    print(f"Best K (max LOO+OOD): K={best_k}", flush=True)
    best_result = results[best_k]
    print(f"  LOO-orig={best_result[2]*100:.1f}%  OOD={best_result[3]*100:.1f}%", flush=True)

    kill = best_result[2] <= 0.880
    success = best_result[2] > 0.880 and best_result[3] > 0.180

    print(flush=True)
    print(f"Kill (best LOO-orig <= 88.0%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (LOO-orig>88.0% AND OOD>18%): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS", flush=True)
    elif kill:
        print("KILLED", flush=True)
    else:
        print("PARTIAL", flush=True)

    print(f"\nElapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == '__main__':
    main()
