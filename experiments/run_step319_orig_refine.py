#!/usr/bin/env python3
"""
Step 319 -- Grow K=10, refine on original 400 only. Leo mail 1389.

Hypothesis: refine on original 400 only keeps LOO-orig high (~96%)
while K=10 grow gives OOD coverage (~99%).

Step 318 showed: refine on all CB (original+spawned) kills LOO at K>=2.
Fix: only use original 400 for weight gradient updates.
Codebook for phi still uses all 2500 points.

Compare: Step 318 K=1 (96.5% LOO-orig, 48.5% OOD),
         Step 318 K=10 all-refine (57.5% LOO-orig, 99.2% OOD).
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
K_GROW = 10
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
    """LOO on original n_orig points, full CB as reference."""
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


# ── refine on originals only ──────────────────────────────────────────────────

def refine_weights_orig_only(A, B, Y, phi_all, W_bk, n_orig, n_passes=N_REFINE):
    """
    Weight updates use only original n_orig points as query anchors.
    Codebook for NN search is full phi_all (all 2500 points).
    """
    W = W_bk.copy()
    for p in range(n_passes):
        n_cross = 0
        for idx in range(n_orig):
            b_idx = int(B[idx]) - 1
            w_full = np.tile(W[b_idx], MAX_CLASS).astype(np.float32)
            phi_q = phi_all[idx]
            diffs = phi_all - phi_q
            dists = (diffs * diffs * w_full).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))
            if Y[nearest] != Y[idx]:
                n_cross += 1
                diff_sq = (phi_q - phi_all[nearest]) ** 2
                for k_idx in range(K_PHI):
                    slots = [c * K_PHI + k_idx for c in range(MAX_CLASS)]
                    W[b_idx, k_idx] += LR_W * diff_sq[slots].sum()
        print(f"    Refine pass {p+1}: cross={n_cross}", flush=True)
    return W


def main():
    t0 = time.time()
    print("Step 319 -- Grow K=10, refine on original 400 only", flush=True)
    print(f"K_GROW={K_GROW}  N_REFINE={N_REFINE}  Refine anchors: original {N_ORIG} only", flush=True)
    print(f"Compare: Step 318 K=1 (96.5% LOO-orig, 48.5% OOD)", flush=True)
    print(f"         Step 318 K=10 all-refine (57.5% LOO-orig, 99.2% OOD)", flush=True)
    print(flush=True)

    # Original training data
    A0_list, B0_list, Y0_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A0_list.append(float(a)); B0_list.append(b); Y0_list.append(a % b)
    A0 = np.array(A0_list, dtype=np.float32)
    B0 = np.array(B0_list, dtype=np.int32)
    Y0 = np.array(Y0_list, dtype=np.int32)

    ood_queries = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    # Baseline
    print("=== Baseline (400 vectors, uniform W) ===", flush=True)
    phi_base = compute_phi_all_loo(A0, B0, Y0)
    W_bk = np.ones((TRAIN_MAX, K_PHI), dtype=np.float32)
    loo_base = loo_accuracy_orig(phi_base, B0, Y0, W_bk, N_ORIG)
    ood_base = ood_accuracy(A0, B0, Y0, W_bk, ood_queries)
    print(f"  LOO-orig={loo_base*100:.1f}%  OOD={ood_base*100:.1f}%", flush=True)
    print(flush=True)

    # Grow K=10
    print(f"=== GROW (K={K_GROW} iterations) ===", flush=True)
    t_grow = time.time()
    A, B, Y = grow_k(A0, B0, Y0, K_GROW)
    print(f"  CB={len(A)} (+{len(A)-N_ORIG} spawned)  [{time.time()-t_grow:.2f}s]", flush=True)
    print(flush=True)

    # Compute phi over full codebook
    print("Computing phi_all_loo (full CB)...", flush=True)
    t_phi = time.time()
    phi_all = compute_phi_all_loo(A, B, Y)
    print(f"  done [{time.time()-t_phi:.1f}s]", flush=True)
    print(flush=True)

    # Refine on original 400 only
    print(f"=== REFINE on original {N_ORIG} only ({N_REFINE} passes) ===", flush=True)
    t_ref = time.time()
    W_bk = np.ones((TRAIN_MAX, K_PHI), dtype=np.float32)
    W_bk = refine_weights_orig_only(A, B, Y, phi_all, W_bk, N_ORIG, N_REFINE)
    print(f"  [{time.time()-t_ref:.1f}s]", flush=True)
    print(flush=True)

    # Evaluate
    print("=== Evaluation ===", flush=True)
    t_ev = time.time()
    loo = loo_accuracy_orig(phi_all, B, Y, W_bk, N_ORIG)
    print(f"  LOO-orig={loo*100:.1f}%  [{time.time()-t_ev:.1f}s]", flush=True)

    t_ood = time.time()
    ood = ood_accuracy(A, B, Y, W_bk, ood_queries)
    print(f"  OOD={ood*100:.1f}%  [{time.time()-t_ood:.1f}s]", flush=True)
    print(flush=True)

    # Per-b LOO (original points only)
    print("=== Per-b LOO (original points) ===", flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        mask_orig = (B[:N_ORIG] == b_val)
        idxs_b = np.where(mask_orig)[0]
        w_full_b = np.tile(W_bk[b_val - 1], MAX_CLASS).astype(np.float32)
        correct = 0
        for i in idxs_b:
            diffs = phi_all - phi_all[i]
            dists = (diffs * diffs * w_full_b).sum(axis=1)
            dists[i] = float('inf')
            nn = int(np.argmin(dists))
            if Y[nn] == Y[i]:
                correct += 1
        w_b = W_bk[b_val - 1]
        w_norm = w_b / (w_b.sum() + 1e-8)
        print(f"  b={b_val:>2}: LOO={correct/len(idxs_b)*100:>6.1f}%  "
              f"n={len(idxs_b):>2}  w={[f'{v:.2f}' for v in w_norm]}", flush=True)

    # Summary
    elapsed = time.time() - t0
    kill = loo <= 0.880
    success = loo > 0.880 and ood > 0.180

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 319 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"CB size: {len(A)} ({len(A)-N_ORIG} spawned)", flush=True)
    print(f"LOO-orig: {loo*100:.1f}%  (Step 318 K=1: 96.5%, K=10 all-refine: 57.5%)", flush=True)
    print(f"OOD:      {ood*100:.1f}%  (Step 318 K=1: 48.5%, K=10 all-refine: 99.2%)", flush=True)
    print(flush=True)
    print(f"Kill (LOO-orig <= 88.0%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (LOO-orig>88.0% AND OOD>18%): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS", flush=True)
    elif kill:
        print("KILLED", flush=True)
    else:
        print("PARTIAL", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
