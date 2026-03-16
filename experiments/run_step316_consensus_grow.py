#!/usr/bin/env python3
"""
Step 316 -- Consensus spacing in grow step. Leo mail 1383.

Grow phase (one turn):
1. Per-class spacing: for each class with >=2 members, spacing = nearest same-class distance
2. Consensus: median of per-class spacings within each b-group (should = b for a%b)
3. Spawn for ALL classes: a_new = a_max + consensus_spacing (including single-point classes)

Refine phase: per-b W[b,k] learning (same as Step 314).

Compare: Step 315 Turn 1 (94.4% LOO, 48.5% OOD), Step 300 (95.2% OOD).
"""

import time
import numpy as np

K = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
PHI_DIM = MAX_CLASS * K   # 100
LR_W = 0.1
N_REFINE = 3


# ── phi ────────────────────────────────────────────────────────────────────

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
            k_eff = min(K, len(idxs))
            phi[c * K: c * K + k_eff] = dists[:k_eff]
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
        k_eff = min(K, len(idxs))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


def loo_accuracy_per_b(phi_all, B, Y, W_bk):
    n = len(Y)
    correct = 0
    for i in range(n):
        b_idx = int(B[i]) - 1
        w_full = np.tile(W_bk[b_idx], MAX_CLASS).astype(np.float32)
        diffs = phi_all - phi_all[i]
        dists = (diffs * diffs * w_full).sum(axis=1)
        dists[i] = float('inf')
        nearest = int(np.argmin(dists))
        if Y[nearest] == Y[i]:
            correct += 1
    return correct / n


def ood_accuracy_fast(A_cb, B_cb, Y_cb, W_bk, ood_queries):
    """OOD: phi_q vs full codebook phi (no LOO), per-b weights."""
    correct = 0
    # Precompute codebook phi (no LOO)
    n = len(A_cb)
    phi_cb = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for j in range(n):
        phi_cb[j] = compute_phi_query(float(A_cb[j]), int(B_cb[j]), A_cb, B_cb, Y_cb)

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


# ── Consensus GROW step ────────────────────────────────────────────────────

def compute_consensus_spacings(A, B, Y):
    """
    For each b-group: compute per-class nearest-same-class spacing.
    consensus_spacing[b] = median of per-class spacings (for classes with >=2 members).
    """
    consensus = {}
    for b_val in range(1, TRAIN_MAX + 1):
        spacings = []
        for c in range(MAX_CLASS):
            mask = (B == b_val) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) < 2:
                continue
            a_vals = sorted(A[idxs].tolist())
            # Nearest same-class spacing: min consecutive difference
            min_spacing = min(a_vals[i+1] - a_vals[i] for i in range(len(a_vals)-1))
            spacings.append(min_spacing)
        if spacings:
            consensus[b_val] = float(np.median(spacings))
        else:
            # Fallback: no multi-member class → use b itself (can't compute from data)
            consensus[b_val] = float(b_val)
    return consensus


def grow_consensus(A, B, Y):
    """
    Consensus-spacing grow: for each (b, class), spawn at a_max + consensus_spacing[b].
    """
    consensus = compute_consensus_spacings(A, B, Y)
    new_A, new_B, new_Y = [], [], []
    spawned_info = []

    for b_val in range(1, TRAIN_MAX + 1):
        spacing = consensus[b_val]
        for c in range(MAX_CLASS):
            mask = (B == b_val) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            a_max = float(A[idxs].max())
            a_new = a_max + spacing
            y_new = int(round(a_new)) % b_val
            new_A.append(a_new)
            new_B.append(b_val)
            new_Y.append(y_new)
            spawned_info.append((b_val, c, a_max, spacing, a_new, y_new))

    A_ext = np.concatenate([A, np.array(new_A, dtype=np.float32)])
    B_ext = np.concatenate([B, np.array(new_B, dtype=np.int32)])
    Y_ext = np.concatenate([Y, np.array(new_Y, dtype=np.int32)])
    return A_ext, B_ext, Y_ext, consensus, spawned_info


# ── REFINE step ────────────────────────────────────────────────────────────

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
                for k_idx in range(K):
                    slots = [c * K + k_idx for c in range(MAX_CLASS)]
                    W[b_idx, k_idx] += LR_W * diff_sq[slots].sum()
    return W


def main():
    t0 = time.time()
    print("Step 316 -- Consensus spacing grow + per-b refine (one turn)", flush=True)
    print(f"Compare: Step 315 Turn 1 (94.4% LOO, 48.5% OOD), Step 300 (95.2% OOD)", flush=True)
    print(flush=True)

    # Initial codebook
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_list.append(float(a)); B_list.append(b); Y_list.append(a % b)
    A = np.array(A_list, dtype=np.float32)
    B = np.array(B_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)

    ood_queries = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    # Baseline
    print("=== Baseline (400 vectors, uniform W) ===", flush=True)
    phi_base = compute_phi_all_loo(A, B, Y)
    W_bk = np.ones((TRAIN_MAX, K), dtype=np.float32)
    loo_base = loo_accuracy_per_b(phi_base, B, Y, W_bk)
    print(f"  LOO={loo_base*100:.1f}%", flush=True)
    print(flush=True)

    # GROW with consensus
    print("=== GROW (consensus spacing) ===", flush=True)
    A, B, Y, consensus, spawned_info = grow_consensus(A, B, Y)
    print(f"  CB={len(A)} (+{len(spawned_info)} spawned)", flush=True)
    print(f"  Consensus spacings (b: spacing):", flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        c_sp = consensus[b_val]
        print(f"    b={b_val:>2}: consensus={c_sp:.1f}  (expected={b_val})", flush=True)
    print(flush=True)

    # Recompute phi
    print("Recomputing phi after grow...", flush=True)
    t_phi = time.time()
    phi_all = compute_phi_all_loo(A, B, Y)
    print(f"  done [{time.time()-t_phi:.1f}s]", flush=True)

    # REFINE
    print(f"REFINE ({N_REFINE} passes)...", flush=True)
    t_ref = time.time()
    W_bk = refine_weights(A, B, Y, phi_all, W_bk, N_REFINE)
    print(f"  done [{time.time()-t_ref:.1f}s]", flush=True)
    print(flush=True)

    # EVALUATE
    print("=== Evaluation ===", flush=True)
    t_ev = time.time()
    loo = loo_accuracy_per_b(phi_all, B, Y, W_bk)
    print(f"  LOO={loo*100:.1f}%  [{time.time()-t_ev:.1f}s]", flush=True)

    t_ood = time.time()
    ood = ood_accuracy_fast(A, B, Y, W_bk, ood_queries)
    print(f"  OOD={ood*100:.1f}%  [{time.time()-t_ood:.1f}s]", flush=True)
    print(flush=True)

    # Per-b breakdown
    print("=== Per-b breakdown ===", flush=True)
    print(f"  {'b':>3} | {'LOO':>7} | n_cb | consensus | w_k (normalized)", flush=True)
    print("  " + "-" * 65, flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        mask = (B == b_val)
        idxs_b = np.where(mask)[0]
        correct = 0
        w_full_b = np.tile(W_bk[b_val - 1], MAX_CLASS).astype(np.float32)
        for i in idxs_b:
            diffs = phi_all - phi_all[i]
            dists = (diffs * diffs * w_full_b).sum(axis=1)
            dists[i] = float('inf')
            nn = int(np.argmin(dists))
            if Y[nn] == Y[i]:
                correct += 1
        acc_b = correct / len(idxs_b)
        w_b = W_bk[b_val - 1]
        w_norm = w_b / (w_b.sum() + 1e-8)
        print(f"  {b_val:>3} | {acc_b*100:>6.1f}% | {len(idxs_b):>4} | "
              f"{consensus[b_val]:>9.1f} | {[f'{v:.2f}' for v in w_norm]}", flush=True)

    # Summary
    elapsed = time.time() - t0
    kill = loo <= 0.880
    success = loo > 0.880 and ood > 0.180

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 316 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Baseline LOO:        {loo_base*100:.1f}%", flush=True)
    print(f"Post-grow-refine LOO: {loo*100:.1f}%  (Step 315 Turn 1: 94.4%)", flush=True)
    print(f"Post-grow-refine OOD: {ood*100:.1f}%  (Step 315 Turn 1: 48.5%, Step 300: 95.2%)", flush=True)
    print(flush=True)
    print(f"Kill (LOO <= 88.0%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (LOO>88.0% AND OOD>18%): {'YES' if success else 'NO'}", flush=True)
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
