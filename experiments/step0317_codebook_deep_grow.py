#!/usr/bin/env python3
"""
Step 317 -- K=10 deep grow + per-b refine. Spec.

GROW: K=10 iterations of reflection spawn with consensus spacing.
Each iteration extends from the new boundary — grows codebook toward OOD territory.
REFINE: per-b W[b,k] learning (3 passes).
EVALUATE: LOO + OOD (a=21..50).

Compare: Step 316 (48.5% OOD, K=1), Step 300 (95.2% OOD, K=31).
"""

import time
import numpy as np

K_PHI = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
PHI_DIM = MAX_CLASS * K_PHI   # 100
LR_W = 0.1
K_GROW = 10   # grow iterations
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


# ── accuracy ───────────────────────────────────────────────────────────────

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


def ood_accuracy(A_cb, B_cb, Y_cb, W_bk, ood_queries):
    """OOD: phi_q vs codebook phi (no LOO), per-b weights."""
    # Precompute codebook phi (no LOO)
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


# ── deep GROW ──────────────────────────────────────────────────────────────

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


def grow_deep(A, B, Y, k_grow):
    """
    K iterations of consensus-spaced reflection spawn.
    Each iteration uses the current boundary (extends iteratively).
    """
    # Compute consensus from original training data (stable)
    consensus = compute_consensus_spacings(A, B, Y)

    # Track current boundary for each (b, class)
    boundaries = {}
    for b_val in range(1, TRAIN_MAX + 1):
        for c in range(MAX_CLASS):
            mask = (B == b_val) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) > 0:
                boundaries[(b_val, c)] = float(A[idxs].max())

    all_new_A, all_new_B, all_new_Y = [], [], []
    spawn_counts = []

    for k in range(k_grow):
        new_A, new_B, new_Y = [], [], []
        for b_val in range(1, TRAIN_MAX + 1):
            spacing = consensus[b_val]
            for c in range(MAX_CLASS):
                if (b_val, c) not in boundaries:
                    continue
                a_max = boundaries[(b_val, c)]
                a_new = a_max + spacing
                y_new = int(round(a_new)) % b_val
                new_A.append(a_new)
                new_B.append(b_val)
                new_Y.append(y_new)
                # Update boundary for next iteration
                boundaries[(b_val, c)] = a_new
        all_new_A.extend(new_A)
        all_new_B.extend(new_B)
        all_new_Y.extend(new_Y)
        spawn_counts.append(len(new_A))

    A_ext = np.concatenate([A, np.array(all_new_A, dtype=np.float32)])
    B_ext = np.concatenate([B, np.array(all_new_B, dtype=np.int32)])
    Y_ext = np.concatenate([Y, np.array(all_new_Y, dtype=np.int32)])
    return A_ext, B_ext, Y_ext, consensus, spawn_counts


# ── REFINE ─────────────────────────────────────────────────────────────────

def refine_weights(A, B, Y, phi_all, W_bk, n_passes=N_REFINE):
    train_data = list(zip(A.tolist(), B.tolist(), Y.tolist()))
    W = W_bk.copy()
    for p in range(n_passes):
        n_cross = 0
        for idx, (a, b, y) in enumerate(train_data):
            b_idx = int(b) - 1
            w_full = np.tile(W[b_idx], MAX_CLASS).astype(np.float32)
            phi_q = phi_all[idx]
            diffs = phi_all - phi_q
            dists = (diffs * diffs * w_full).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))
            if Y[nearest] != int(y):
                n_cross += 1
                diff_sq = (phi_q - phi_all[nearest]) ** 2
                for k_idx in range(K_PHI):
                    slots = [c * K_PHI + k_idx for c in range(MAX_CLASS)]
                    W[b_idx, k_idx] += LR_W * diff_sq[slots].sum()
        print(f"    Refine pass {p+1}: cross={n_cross}", flush=True)
    return W


def main():
    t0 = time.time()
    print("Step 317 -- K=10 deep grow + per-b refine", flush=True)
    print(f"K_GROW={K_GROW}  N_REFINE={N_REFINE}  LR_W={LR_W}", flush=True)
    print(f"Compare: Step 316 (48.5% OOD, K=1), Step 300 (95.2% OOD, K=31)", flush=True)
    print(flush=True)

    # Training data
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_list.append(float(a)); B_list.append(b); Y_list.append(a % b)
    A = np.array(A_list, dtype=np.float32)
    B = np.array(B_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)

    ood_queries = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    # GROW K=10
    print(f"=== GROW (K={K_GROW} iterations) ===", flush=True)
    t_grow = time.time()
    A, B, Y, consensus, spawn_counts = grow_deep(A, B, Y, K_GROW)
    print(f"  Total spawned: {sum(spawn_counts)} vectors -> CB={len(A)}", flush=True)
    print(f"  Per-iteration: {spawn_counts}", flush=True)
    print(f"  Consensus spacings: {[consensus[b] for b in range(1, 6)]}... (all = b)", flush=True)

    # A-range per b-group after grow
    print(f"  Extended a-range sample (b=5): "
          f"{sorted(A[B==5])[-5:]}", flush=True)
    print(f"  Extended a-range sample (b=20): "
          f"{sorted(A[B==20])[-5:]}", flush=True)
    print(f"  [{time.time()-t_grow:.2f}s]", flush=True)
    print(flush=True)

    # Compute phi
    print("Computing phi_all_loo...", flush=True)
    t_phi = time.time()
    phi_all = compute_phi_all_loo(A, B, Y)
    print(f"  done [{time.time()-t_phi:.1f}s]", flush=True)

    # REFINE
    print(f"REFINE ({N_REFINE} passes)...", flush=True)
    t_ref = time.time()
    W_bk = np.ones((TRAIN_MAX, K_PHI), dtype=np.float32)
    W_bk = refine_weights(A, B, Y, phi_all, W_bk, N_REFINE)
    print(f"  [{time.time()-t_ref:.1f}s]", flush=True)
    print(flush=True)

    # EVALUATE
    print("=== Evaluation ===", flush=True)
    t_ev = time.time()
    loo = loo_accuracy_per_b(phi_all, B, Y, W_bk)
    print(f"  LOO={loo*100:.1f}%  [{time.time()-t_ev:.1f}s]", flush=True)

    t_ood = time.time()
    ood = ood_accuracy(A, B, Y, W_bk, ood_queries)
    print(f"  OOD={ood*100:.1f}%  [{time.time()-t_ood:.1f}s]", flush=True)
    print(flush=True)

    # Per-b breakdown
    print("=== Per-b LOO ===", flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        mask = (B == b_val)
        idxs_b = np.where(mask)[0]
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
              f"n={len(idxs_b):>4}  w={[f'{v:.2f}' for v in w_norm]}", flush=True)

    # OOD per-b coverage
    print(flush=True)
    print("=== OOD coverage: max spawned a per b-group ===", flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        a_max_in_cb = float(A[B == b_val].max())
        ood_covered = sum(1 for (a, bq, _) in ood_queries if bq == b_val and a <= a_max_in_cb)
        ood_total = sum(1 for (_, bq, _) in ood_queries if bq == b_val)
        print(f"  b={b_val:>2}: max_a={a_max_in_cb:.0f}  OOD_covered={ood_covered}/{ood_total}", flush=True)

    # Summary
    elapsed = time.time() - t0
    kill = loo <= 0.880
    success = loo > 0.880 and ood > 0.180

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 317 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"CB size: {len(A)} ({len(A)-400} spawned)", flush=True)
    print(f"LOO: {loo*100:.1f}%  (Step 316: 94.4%)", flush=True)
    print(f"OOD: {ood*100:.1f}%  (Step 316: 48.5%, Step 300: 95.2%)", flush=True)
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
