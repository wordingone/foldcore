#!/usr/bin/env python3
"""
Step 315 -- Grow-and-refine loop. Spec.

Each turn:
1. GROW: for each b-group, reflect boundary of each class outward.
   Spawn: a_new = 2*a_max - a_prev (reflection). Label = a_new % b.
   Fallback (single-member class): a_new = a_max + b.
2. REFINE: learn per-b W[b,k] on extended codebook (N_REFINE passes).
3. EVALUATE: LOO + OOD (a=21..50).

Kill: combined LOO <= max(88.0%, refine-alone).
Success: combined LOO > 88.0% AND OOD > 18%.
Compare: refine-alone=88.0%, grow-alone=95.2% OOD (Step 300).
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
N_TURNS = 5
N_REFINE = 3  # refine passes per turn


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
    """phi for OOD query (not in codebook, no LOO needed)."""
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


def ood_accuracy(A, B, Y, W_bk, ood_queries):
    correct = 0
    for (a_q, b_q, y_true) in ood_queries:
        phi_q = compute_phi_query(float(a_q), b_q, A, B, Y)
        b_idx = b_q - 1
        w_full = np.tile(W_bk[b_idx], MAX_CLASS).astype(np.float32)
        # Distance to all codebook entries
        # Build phi_all lazily for this query
        dists = np.array([
            ((phi_q - compute_phi_query(float(A[j]), int(B[j]), A, B, Y)) ** 2 * w_full).sum()
            for j in range(len(A))
        ])
        # Actually: use precomputed phi_all if available — but for OOD we don't need LOO
        # Use direct phi_q vs codebook phi (no LOO exclusion for training entries)
        nearest = int(np.argmin(dists))
        if Y[nearest] == y_true:
            correct += 1
    return correct / len(ood_queries)


def ood_accuracy_fast(A_cb, B_cb, Y_cb, phi_cb_nolou, W_bk, ood_queries):
    """Fast OOD: precomputed codebook phi (no LOO), per-b weights."""
    correct = 0
    for (a_q, b_q, y_true) in ood_queries:
        phi_q = compute_phi_query(float(a_q), b_q, A_cb, B_cb, Y_cb)
        b_idx = b_q - 1
        w_full = np.tile(W_bk[b_idx], MAX_CLASS).astype(np.float32)
        diffs = phi_cb_nolou - phi_q
        dists = (diffs * diffs * w_full).sum(axis=1)
        nearest = int(np.argmin(dists))
        if Y_cb[nearest] == y_true:
            correct += 1
    return correct / len(ood_queries)


def compute_phi_all_nolou(A, B, Y):
    """phi without LOO exclusion (for OOD reference)."""
    n = len(A)
    result = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for i in range(n):
        phi = np.full(PHI_DIM, SENTINEL, dtype=np.float32)
        b_q, a_q = int(B[i]), float(A[i])
        for c in range(MAX_CLASS):
            mask = (B == b_q) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            dists = np.abs(A[idxs] - a_q).astype(np.float32)
            dists.sort()
            k_eff = min(K, len(idxs))
            phi[c * K: c * K + k_eff] = dists[:k_eff]
        result[i] = phi
    return result


# ── GROW step ──────────────────────────────────────────────────────────────

def grow_codebook(A, B, Y):
    """
    For each b-group, for each class present:
    Sort a-values. Spawn: 2*a_max - a_prev (fallback: a_max + b).
    New vector: (a_new, b, a_new % b).
    """
    new_A, new_B, new_Y = [], [], []
    for b_val in range(1, TRAIN_MAX + 1):
        for c in range(MAX_CLASS):
            mask = (B == b_val) & (Y == c)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            a_vals = sorted(A[idxs].tolist())
            a_max = a_vals[-1]
            if len(a_vals) >= 2:
                a_prev = a_vals[-2]
                a_new = 2 * a_max - a_prev
            else:
                a_new = a_max + b_val
            y_new = int(round(a_new)) % b_val
            new_A.append(float(a_new))
            new_B.append(b_val)
            new_Y.append(y_new)

    A_ext = np.concatenate([A, np.array(new_A, dtype=np.float32)])
    B_ext = np.concatenate([B, np.array(new_B, dtype=np.int32)])
    Y_ext = np.concatenate([Y, np.array(new_Y, dtype=np.int32)])
    return A_ext, B_ext, Y_ext, len(new_A)


# ── REFINE step ────────────────────────────────────────────────────────────

def refine_weights(A, B, Y, phi_all, W_bk, n_passes=N_REFINE):
    """Per-b weight learning passes."""
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
    print("Step 315 -- Grow-and-refine loop", flush=True)
    print(f"N turns={N_TURNS}  refine passes/turn={N_REFINE}  alpha={ALPHA}  lr_w={LR_W}", flush=True)
    print(f"Kill: LOO <= 88.0% | Success: LOO > 88.0% AND OOD > 18%", flush=True)
    print(f"Compare: refine=88.0%, grow OOD=95.2% (Step 300)", flush=True)
    print(flush=True)

    # Initial codebook: 400 training vectors
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_list.append(float(a)); B_list.append(b); Y_list.append(a % b)
    A = np.array(A_list, dtype=np.float32)
    B = np.array(B_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)

    # OOD queries
    ood_queries = [(a, b, a % b) for a in range(21, 51) for b in range(1, TRAIN_MAX + 1)]

    # Initial W: uniform
    W_bk = np.ones((TRAIN_MAX, K), dtype=np.float32)

    # Baseline (turn 0)
    print("=== Turn 0 (baseline, 400 vectors, uniform W) ===", flush=True)
    t_ev = time.time()
    phi_all = compute_phi_all_loo(A, B, Y)
    loo0 = loo_accuracy_per_b(phi_all, B, Y, W_bk)
    phi_nolou = compute_phi_all_nolou(A, B, Y)
    ood0 = ood_accuracy_fast(A, B, Y, phi_nolou, W_bk, ood_queries)
    print(f"  CB={len(A)}  LOO={loo0*100:.1f}%  OOD={ood0*100:.1f}%  [{time.time()-t_ev:.1f}s]", flush=True)
    print(flush=True)

    best_loo = loo0
    results = [(0, len(A), loo0, ood0)]

    for turn in range(1, N_TURNS + 1):
        print(f"=== Turn {turn} ===", flush=True)

        # GROW
        A, B, Y, n_spawned = grow_codebook(A, B, Y)
        print(f"  GROW: spawned {n_spawned} vectors -> CB={len(A)}", flush=True)

        # Recompute phi for grown codebook
        t_phi = time.time()
        phi_all = compute_phi_all_loo(A, B, Y)
        print(f"  phi recomputed [{time.time()-t_phi:.1f}s]", flush=True)

        # REFINE
        t_ref = time.time()
        W_bk = refine_weights(A, B, Y, phi_all, W_bk, N_REFINE)
        print(f"  REFINE: {N_REFINE} passes [{time.time()-t_ref:.1f}s]", flush=True)

        # EVALUATE
        t_ev = time.time()
        loo = loo_accuracy_per_b(phi_all, B, Y, W_bk)
        phi_nolou = compute_phi_all_nolou(A, B, Y)
        ood = ood_accuracy_fast(A, B, Y, phi_nolou, W_bk, ood_queries)
        print(f"  EVAL: LOO={loo*100:.1f}%  OOD={ood*100:.1f}%  "
              f"delta_LOO={( loo - results[-1][2])*100:+.1f}pp  [{time.time()-t_ev:.1f}s]", flush=True)
        print(flush=True)

        results.append((turn, len(A), loo, ood))
        best_loo = max(best_loo, loo)

        if loo <= best_loo - 0.05 and turn >= 2:
            print(f"  Saturated (LOO dropped from peak {best_loo*100:.1f}%). Stopping.", flush=True)
            break

    # Per-b breakdown at final state
    phi_final = compute_phi_all_loo(A, B, Y)
    print("=== Final per-b breakdown ===", flush=True)
    print(f"  {'b':>3} | {'LOO':>7} | n_cb | w_k (normalized)", flush=True)
    print("  " + "-" * 60, flush=True)
    for b_val in range(1, TRAIN_MAX + 1):
        mask = (B == b_val)
        idxs = np.where(mask)[0]
        correct = sum(1 for i in idxs if (
            Y[np.argmin(np.where(
                np.arange(len(A)) == i,
                np.inf,
                ((phi_final - phi_final[i]) ** 2 * np.tile(W_bk[b_val-1], MAX_CLASS)).sum(axis=1)
            ))] == Y[i]
        ))
        acc_b = correct / len(idxs) if len(idxs) > 0 else 0
        w_b = W_bk[b_val - 1]
        w_norm = w_b / (w_b.sum() + 1e-8)
        print(f"  {b_val:>3} | {acc_b*100:>6.1f}% | {len(idxs):>4} | "
              f"{[f'{v:.2f}' for v in w_norm]}", flush=True)

    # Summary
    elapsed = time.time() - t0
    final_loo = results[-1][2]
    final_ood = results[-1][3]
    kill = final_loo <= 0.880
    success = final_loo > 0.880 and final_ood > 0.180

    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 315 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Turn':>5} | {'CB':>5} | {'LOO':>7} | {'OOD':>7}", flush=True)
    print("-" * 35, flush=True)
    for turn, cb, loo, ood in results:
        print(f"  {turn:>3} | {cb:>5} | {loo*100:>6.1f}% | {ood*100:>6.1f}%", flush=True)
    print(flush=True)
    print(f"Kill (LOO <= 88.0%): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (LOO>88.0% AND OOD>18%): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Grow+refine exceeds both components alone.", flush=True)
    elif kill:
        print("KILLED -- Combined loop adds nothing.", flush=True)
    else:
        print("PARTIAL -- Some improvement but below success threshold.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
