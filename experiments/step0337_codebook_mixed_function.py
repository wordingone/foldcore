#!/usr/bin/env python3
"""
Step 337 — Mixed-function classification with CL+phi.

DATASET:
- b in 1..10:  y = a % b          (modular, periodic within-class spacing)
- b in 11..20: y = floor(a / b)   (division, monotonic within-class spacing)
- a in 1..20 for all, giving 400 points total

EXPERIMENTS:
A. Baseline: CL+phi with GLOBAL parameters (single CL filter, uniform weights)
B. Stage 7:  CL+phi with PER-ENTRY parameters stored in codebook
             Each entry stores [K_local, weights_local]; updated by CL dynamics
C. Oracle:   Separate CL+phi per function, then combine

Kill: B must beat A.
Also measure: oracle gap (how close B gets to C).

Why it should work:
- a%b and floor(a/b) have fundamentally different distance structure
- optimal phi weights and K differ per function
- per-entry specialization has genuine structure to capture
- CL proximity grouping should naturally separate b<=10 from b>=11 regions
"""

import numpy as np
import time

# ─── Config ────────────────────────────────────────────────────────────────────

TRAIN_MAX    = 20
K_VALS       = [3, 5, 7]        # K candidates for per-entry selection
K_DEFAULT    = 5                # Default K for global baseline
SENTINEL     = TRAIN_MAX * 3
SPAWN_THRESH = 4.0              # From Step 333 best
N_EPOCHS     = 10
LR_W         = 0.1
SEED         = 42

# ─── Dataset ───────────────────────────────────────────────────────────────────

def build_dataset():
    """
    400 entries: a,b in 1..20
    b in 1..10:  y = a % b
    b in 11..20: y = floor(a / b)
    """
    A, B, Y, FUNC = [], [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(a)
            B.append(b)
            if b <= 10:
                Y.append(a % b)
                FUNC.append(0)  # modular
            else:
                Y.append(a // b)
                FUNC.append(1)  # division
    return np.array(A), np.array(B), np.array(Y), np.array(FUNC)


# ─── Phi computation within CL group ──────────────────────────────────────────

def compute_phi_group(query_a, A_group, Y_group, exclude_idx, K, max_class):
    """phi for query_a restricted to group members (LOO-safe)."""
    phi = np.full(max_class * K, float(SENTINEL), dtype=np.float32)
    for c in range(max_class):
        class_mask = Y_group == c
        if exclude_idx is not None and exclude_idx < len(class_mask):
            if Y_group[exclude_idx] == c:
                class_mask = class_mask.copy()
                class_mask[exclude_idx] = False
        idxs = np.where(class_mask)[0]
        if len(idxs) == 0:
            continue
        dists = np.abs(A_group[idxs] - query_a).astype(np.float32)
        dists.sort()
        k_eff = min(K, len(dists))
        phi[c * K: c * K + k_eff] = dists[:k_eff]
    return phi


# ─── Competitive learning ──────────────────────────────────────────────────────

def build_cl_codebook(X, spawn_thresh=SPAWN_THRESH, lr=0.1, n_epochs=3, seed=SEED):
    """Build CL codebook in [a,b] space. Returns codebook, assignments."""
    rng = np.random.RandomState(seed)
    n = len(X)
    order = np.arange(n)
    codebook = [X[0].copy()]

    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            x = X[i]
            cb = np.array(codebook)
            dists = np.linalg.norm(cb - x, axis=1)
            winner = np.argmin(dists)
            if dists[winner] > spawn_thresh:
                codebook.append(x.copy())
            else:
                codebook[winner] = codebook[winner] + lr * (x - codebook[winner])

    codebook = np.array(codebook)
    dists_all = np.linalg.norm(codebook[:, None, :] - X[None, :, :], axis=2).T
    assignments = np.argmin(dists_all, axis=1).astype(np.int32)
    return codebook, assignments


# ─── Experiment A: Global CL+phi (baseline) ──────────────────────────────────

def exp_a_global(A, B, Y, assignments, K, max_class):
    """CL filter + uniform weights, single global K."""
    n = len(A)
    n_cb = assignments.max() + 1

    # Precompute phi for all entries within their groups (LOO)
    all_phi = np.zeros((n, max_class * K), dtype=np.float32)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idx = np.where(group_mask)[0]
        pos_in_group = np.where(group_idx == i)[0]
        exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
        all_phi[i] = compute_phi_group(
            A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    correct = 0
    w_expanded = np.ones(max_class * K, dtype=np.float64)
    for i in range(n):
        g = assignments[i]
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue
        phi_q = all_phi[i]
        phi_group = all_phi[group_idxs]
        diffs = phi_group - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        if Y[best_j] == Y[i]:
            correct += 1

    return correct / n


# ─── Experiment B: Per-entry CL+phi (Stage 7) ────────────────────────────────

def exp_b_per_entry(A, B, Y, assignments, K_vals, max_class, n_epochs=N_EPOCHS,
                    lr_w=LR_W, seed=SEED):
    """
    Per-CL-entry parameters: K_local and weights_local.
    Each codebook entry stores its own K and weight vector.
    Updated by CL dynamics (winner gets weight update).
    """
    rng = np.random.RandomState(seed)
    n = len(A)
    n_cb = assignments.max() + 1

    # Initialize per-entry parameters
    K_max = max(K_vals)
    cb_K = np.full(n_cb, K_vals[len(K_vals) // 2], dtype=np.int32)  # init to middle K
    cb_weights = {k: np.ones(k, dtype=np.float64) for k in K_vals}
    # Per-entry weights dict indexed by (entry, K)
    entry_weights = {}
    for g in range(n_cb):
        for k in K_vals:
            entry_weights[(g, k)] = np.ones(k, dtype=np.float64)

    # Precompute phi for multiple K values
    all_phi = {}
    for K in K_vals:
        all_phi[K] = np.zeros((n, max_class * K), dtype=np.float32)
        for i in range(n):
            g = assignments[i]
            group_mask = assignments == g
            group_idx = np.where(group_mask)[0]
            pos_in_group = np.where(group_idx == i)[0]
            exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
            all_phi[K][i] = compute_phi_group(
                A[i], A[group_mask], Y[group_mask], exc, K, max_class)

    order = np.arange(n)
    for epoch in range(n_epochs):
        rng.shuffle(order)
        for i in order:
            g = int(assignments[i])
            group_mask = assignments == g
            group_idxs = np.where(group_mask)[0]
            if len(group_idxs) <= 1:
                continue

            K_cur = cb_K[g]
            w = entry_weights[(g, K_cur)]
            w_expanded = np.tile(w, max_class)

            phi_q = all_phi[K_cur][i]
            phi_group = all_phi[K_cur][group_idxs]
            diffs = phi_group - phi_q
            dists = (diffs ** 2 * w_expanded).sum(axis=1)

            self_pos = np.where(group_idxs == i)[0]
            if len(self_pos) > 0:
                dists[self_pos[0]] = float('inf')

            best_local = np.argmin(dists)
            best_j = group_idxs[best_local]

            if Y[best_j] != Y[i]:
                # Wrong prediction with current K: try other K values
                best_acc_k = -1
                best_k = K_cur
                for K_try in K_vals:
                    w_try = entry_weights[(g, K_try)]
                    w_exp_try = np.tile(w_try, max_class)
                    phi_q_try = all_phi[K_try][i]
                    phi_grp_try = all_phi[K_try][group_idxs]
                    diffs_try = phi_grp_try - phi_q_try
                    dists_try = (diffs_try ** 2 * w_exp_try).sum(axis=1)
                    if len(self_pos) > 0:
                        dists_try[self_pos[0]] = float('inf')
                    bj_try = group_idxs[np.argmin(dists_try)]
                    acc_try = 1 if Y[bj_try] == Y[i] else 0
                    if acc_try > best_acc_k:
                        best_acc_k = acc_try
                        best_k = K_try

                # Update K if a better one found
                cb_K[g] = best_k
                K_cur = best_k
                w = entry_weights[(g, K_cur)]
                w_expanded = np.tile(w, max_class)

                # Update weights for current K
                phi_q = all_phi[K_cur][i]
                phi_group = all_phi[K_cur][group_idxs]
                diffs = phi_group - phi_q
                dists = (diffs ** 2 * w_expanded).sum(axis=1)
                if len(self_pos) > 0:
                    dists[self_pos[0]] = float('inf')
                best_local = np.argmin(dists)
                best_j_new = group_idxs[best_local]

                if Y[best_j_new] != Y[i]:
                    # Still wrong: upweight k-positions where phi_q differs from phi_wrong
                    diff_sq = (phi_q - all_phi[K_cur][best_j_new]) ** 2
                    per_k_signal = np.zeros(K_cur)
                    for k in range(K_cur):
                        indices = [c * K_cur + k for c in range(max_class)]
                        per_k_signal[k] = diff_sq[indices].mean()
                    entry_weights[(g, K_cur)] += lr_w * per_k_signal
                    entry_weights[(g, K_cur)] = np.maximum(entry_weights[(g, K_cur)], 0.01)
                    s = entry_weights[(g, K_cur)].sum()
                    if s > 0:
                        entry_weights[(g, K_cur)] = entry_weights[(g, K_cur)] / s * K_cur

    # Evaluate with per-entry parameters
    correct = 0
    for i in range(n):
        g = int(assignments[i])
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue

        K_cur = cb_K[g]
        w = entry_weights[(g, K_cur)]
        w_expanded = np.tile(w, max_class)

        phi_q = all_phi[K_cur][i]
        phi_group = all_phi[K_cur][group_idxs]
        diffs = phi_group - phi_q
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        if Y[best_j] == Y[i]:
            correct += 1

    return correct / n, cb_K, entry_weights


# ─── Experiment C: Oracle (separate CL+phi per function) ─────────────────────

def exp_c_oracle(A, B, Y, FUNC, K, max_class):
    """Oracle: separate CL+phi for each function type."""
    correct_total = 0
    n = len(A)

    for func_id in [0, 1]:
        mask = FUNC == func_id
        A_f = A[mask]
        B_f = B[mask]
        Y_f = Y[mask]

        X_f = np.column_stack([A_f, B_f]).astype(np.float32)
        codebook_f, assignments_f = build_cl_codebook(X_f, seed=SEED + func_id)

        n_f = len(A_f)
        max_class_f = int(Y_f.max()) + 1

        # Precompute phi
        all_phi_f = np.zeros((n_f, max_class * K), dtype=np.float32)
        for i in range(n_f):
            g = assignments_f[i]
            group_mask = assignments_f == g
            group_idx = np.where(group_mask)[0]
            pos_in_group = np.where(group_idx == i)[0]
            exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
            all_phi_f[i] = compute_phi_group(
                A_f[i], A_f[group_mask], Y_f[group_mask], exc, K, max_class)

        w_expanded = np.ones(max_class * K, dtype=np.float64)
        for i in range(n_f):
            g = assignments_f[i]
            group_mask_f = assignments_f == g
            group_idxs_f = np.where(group_mask_f)[0]
            if len(group_idxs_f) <= 1:
                continue
            phi_q = all_phi_f[i]
            phi_group = all_phi_f[group_idxs_f]
            diffs = phi_group - phi_q
            dists = (diffs ** 2 * w_expanded).sum(axis=1)
            self_pos = np.where(group_idxs_f == i)[0]
            if len(self_pos) > 0:
                dists[self_pos[0]] = float('inf')
            best_j = group_idxs_f[np.argmin(dists)]
            if Y_f[best_j] == Y_f[i]:
                correct_total += 1

    return correct_total / n


# ─── CL separation analysis ───────────────────────────────────────────────────

def analyze_cl_separation(assignments, FUNC, n_cb):
    """How well does CL naturally separate b<=10 (func=0) from b>=11 (func=1)?"""
    entry_purity = []
    for g in range(n_cb):
        mask = assignments == g
        if mask.sum() < 2:
            continue
        func_in_group = FUNC[mask]
        majority = np.bincount(func_in_group, minlength=2).argmax()
        purity = (func_in_group == majority).mean()
        entry_purity.append(purity)

    # Pair purity
    same_func_pairs = 0
    total_pairs = 0
    for g in range(n_cb):
        mask = assignments == g
        func_g = FUNC[mask]
        n_g = mask.sum()
        if n_g < 2:
            continue
        for i in range(n_g):
            for j in range(i + 1, n_g):
                total_pairs += 1
                if func_g[i] == func_g[j]:
                    same_func_pairs += 1

    pair_purity = same_func_pairs / total_pairs if total_pairs > 0 else 0.0
    avg_entry_purity = np.mean(entry_purity) if entry_purity else 0.0
    return avg_entry_purity, pair_purity


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    np.random.seed(SEED)

    print("Step 337 — Mixed-function classification with CL+phi", flush=True)
    print(f"K_default={K_DEFAULT}, K_vals={K_vals}, spawn_thresh={SPAWN_THRESH}", flush=True)
    print(f"Dataset: b<=10 -> y=a%b (modular), b>=11 -> y=floor(a/b) (division)", flush=True)
    print("Kill: Experiment B must beat Experiment A", flush=True)
    print(flush=True)

    A, B, Y, FUNC = build_dataset()
    n = len(A)
    max_class = int(Y.max()) + 1
    X = np.column_stack([A, B]).astype(np.float32)
    print(f"Dataset: {n} entries, max_class={max_class}", flush=True)
    n_mod = (FUNC == 0).sum()
    n_div = (FUNC == 1).sum()
    print(f"  Modular (b<=10): {n_mod}  Division (b>=11): {n_div}", flush=True)
    print(flush=True)

    # Build global CL codebook
    print(f"Building global CL codebook (spawn_thresh={SPAWN_THRESH})...", flush=True)
    codebook, assignments = build_cl_codebook(X)
    n_cb = len(codebook)
    group_sizes = [(assignments == c).sum() for c in range(n_cb)]
    print(f"  Codebook size: {n_cb}", flush=True)
    print(f"  Group sizes: min={min(group_sizes)} max={max(group_sizes)} "
          f"mean={np.mean(group_sizes):.1f}", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # CL separation analysis
    print("Analyzing CL separation of functions...", flush=True)
    avg_purity, pair_purity = analyze_cl_separation(assignments, FUNC, n_cb)
    print(f"  Avg entry function-purity: {avg_purity*100:.1f}%", flush=True)
    print(f"  Same-function pair purity: {pair_purity*100:.1f}%", flush=True)
    print(f"  (100% = CL naturally separates functions; ~50% = no separation)", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Experiment A: Global CL+phi (baseline)
    print("Experiment A: Global CL+phi (uniform weights, K=5)...", flush=True)
    acc_a = exp_a_global(A, B, Y, assignments, K_DEFAULT, max_class)
    print(f"  A. Global CL+phi LOO: {acc_a*100:.2f}%", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Experiment B: Per-entry parameters (Stage 7)
    print(f"Experiment B: Per-entry CL+phi (K_vals={K_vals}, {N_EPOCHS} epochs)...", flush=True)
    acc_b, cb_K_final, entry_weights_final = exp_b_per_entry(
        A, B, Y, assignments, K_vals, max_class)
    delta_ba = acc_b - acc_a
    print(f"  B. Per-entry CL+phi LOO: {acc_b*100:.2f}%  (delta vs A: {delta_ba*100:+.2f}pp)", flush=True)

    # Per-entry K distribution
    K_counts = {k: 0 for k in K_vals}
    for g in range(n_cb):
        k = int(cb_K_final[g])
        if k in K_counts:
            K_counts[k] += 1
    print(f"  Final K distribution per entry: {K_counts}", flush=True)

    # Modular vs division accuracy breakdown (using best K per entry)
    correct_mod = 0
    total_mod = 0
    correct_div = 0
    total_div = 0
    for i in range(n):
        g = int(assignments[i])
        group_mask = assignments == g
        group_idxs = np.where(group_mask)[0]
        if len(group_idxs) <= 1:
            continue
        K_cur = int(cb_K_final[g])
        w = entry_weights_final[(g, K_cur)]
        w_expanded = np.tile(w, max_class)
        # Recompute phi for eval (not stored, recompute inline)
        group_idx = np.where(group_mask)[0]
        pos_in_group = np.where(group_idx == i)[0]
        exc = int(pos_in_group[0]) if len(pos_in_group) > 0 else None
        phi_i = compute_phi_group(A[i], A[group_mask], Y[group_mask], exc, K_cur, max_class)
        all_phi_g = np.zeros((len(group_idxs), max_class * K_cur), dtype=np.float32)
        for jj, j in enumerate(group_idxs):
            g2 = assignments[j]
            gm2 = assignments == g2
            gi2 = np.where(gm2)[0]
            pig2 = np.where(gi2 == j)[0]
            exc2 = int(pig2[0]) if len(pig2) > 0 else None
            all_phi_g[jj] = compute_phi_group(A[j], A[gm2], Y[gm2], exc2, K_cur, max_class)
        diffs = all_phi_g - phi_i
        dists = (diffs ** 2 * w_expanded).sum(axis=1)
        self_pos = np.where(group_idxs == i)[0]
        if len(self_pos) > 0:
            dists[self_pos[0]] = float('inf')
        best_j = group_idxs[np.argmin(dists)]
        pred_correct = (Y[best_j] == Y[i])
        if FUNC[i] == 0:
            total_mod += 1
            correct_mod += pred_correct
        else:
            total_div += 1
            correct_div += pred_correct

    if total_mod > 0:
        print(f"  B breakdown — modular: {correct_mod/total_mod*100:.2f}%  "
              f"division: {correct_div/total_div*100:.2f}%", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)
    print(flush=True)

    # Experiment C: Oracle
    print("Experiment C: Oracle (separate CL+phi per function)...", flush=True)
    acc_c = exp_c_oracle(A, B, Y, FUNC, K_DEFAULT, max_class)
    delta_ca = acc_c - acc_a
    oracle_gap = acc_c - acc_b
    print(f"  C. Oracle LOO: {acc_c*100:.2f}%  (delta vs A: {delta_ca*100:+.2f}pp)", flush=True)
    print(f"  Oracle gap (C - B): {oracle_gap*100:+.2f}pp", flush=True)
    print(f"  Elapsed: {time.time()-t0:.2f}s", flush=True)

    elapsed = time.time() - t0

    # Summary
    print(flush=True)
    print("=" * 65, flush=True)
    print("STEP 337 RESULTS", flush=True)
    print("=" * 65, flush=True)
    print(f"CL separation (function purity):  {avg_purity*100:.1f}%  pair: {pair_purity*100:.1f}%", flush=True)
    print(flush=True)
    print(f"A. Global CL+phi (K=5, uniform):  {acc_a*100:.2f}%", flush=True)
    print(f"B. Per-entry CL+phi (Stage 7):    {acc_b*100:.2f}%  ({delta_ba*100:+.2f}pp vs A)", flush=True)
    print(f"C. Oracle (separate per function): {acc_c*100:.2f}%  ({delta_ca*100:+.2f}pp vs A)", flush=True)
    print(f"   Oracle gap (C - B):            {oracle_gap*100:+.2f}pp", flush=True)

    kill    = acc_b <= acc_a
    success = acc_b > acc_a

    print(flush=True)
    print("=" * 65, flush=True)
    print("KILL CHECK", flush=True)
    print("=" * 65, flush=True)
    print(f"Experiment A (baseline): {acc_a*100:.2f}%", flush=True)
    print(f"Experiment B (Stage 7):  {acc_b*100:.2f}%", flush=True)
    print(f"Delta B vs A: {delta_ba*100:+.2f}pp", flush=True)
    print(f"Kill (B <= A): {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (B > A): {'YES' if success else 'NO'}", flush=True)

    if kill:
        print("\nKILLED — per-entry parameters don't improve over global on mixed functions", flush=True)
    else:
        print(f"\nSUCCESS — Stage 7 works: per-entry specialization adds {delta_ba*100:.2f}pp "
              f"on mixed-function dataset", flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


K_vals = K_VALS  # module-level alias for use in main

if __name__ == '__main__':
    main()
