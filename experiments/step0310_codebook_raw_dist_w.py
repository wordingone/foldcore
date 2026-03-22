#!/usr/bin/env python3
"""
Step 310 -- Raw distances + learned weights. Spec.

Features: phi_raw(x)_i = |a_x - A[i]| if B[i]==b_x else SENTINEL
w: N-dimensional (one weight per codebook entry)
Distance: d_w(i,j) = sum_k w_k * (phi_raw(i)_k - phi_raw(j)_k)^2
Training: cross-class match -> w += lr * diff_sq (same as 308b)
No codebook absorption (codebook = fixed training data).

Kill: LOO < 50% at pass 10.
Success: LOO > 70% AND w shows class-correlated structure.
Compare: Step 296 phi=86.8%, Step 308b phi+w=91.2%, baseline 1-NN=5%.
"""

import time
import numpy as np

TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)   # 60
LR_W = 0.1
N_PASSES = 10


def build_phi_raw(A, B):
    """
    phi_raw[i][j] = |A[i] - A[j]| if B[i]==B[j] else SENTINEL.
    Self-diagonal: SENTINEL (LOO exclusion).
    Returns: (N, N) float32 array.
    """
    diff = np.abs(A[:, None] - A[None, :]).astype(np.float32)
    b_match = (B[:, None] == B[None, :])
    phi = np.where(b_match, diff, SENTINEL)
    np.fill_diagonal(phi, SENTINEL)
    return phi


def weighted_dist_matrix(phi, w):
    """
    D[i,j] = sum_k w_k * (phi[i,k] - phi[j,k])^2
    Uses quadratic expansion: D = a + a.T - 2*C
    where a[i] = sum_k w_k * phi[i,k]^2, C = (phi*w) @ phi.T
    Self-distances set to inf.
    """
    a = (phi ** 2 * w).sum(axis=1)          # (N,)
    C = (phi * w) @ phi.T                    # (N, N)
    D = a[:, None] + a[None, :] - 2 * C
    np.fill_diagonal(D, float('inf'))
    return D


def loo_accuracy(D, Y):
    nearest = np.argmin(D, axis=1)
    return float((Y[nearest] == Y).sum()) / len(Y)


def main():
    t0 = time.time()
    print("Step 310 -- Raw distances + learned weights", flush=True)
    print(f"N passes={N_PASSES}  lr_w={LR_W}", flush=True)
    print(f"Compare: phi=86.8%, phi+w=91.2%, 1-NN baseline=5%", flush=True)
    print(flush=True)

    # Fixed codebook = all training data
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A_list.append(float(a))
            B_list.append(b)
            Y_list.append(a % b)
    A = np.array(A_list, dtype=np.float32)
    B = np.array(B_list, dtype=np.int32)
    Y = np.array(Y_list, dtype=np.int32)
    n = len(A)

    # phi_raw is fixed (no absorption)
    phi = build_phi_raw(A, B)

    w = np.ones(n, dtype=np.float32)
    results = []

    # Pass 1: uniform w baseline
    D = weighted_dist_matrix(phi, w)
    loo0 = loo_accuracy(D, Y)
    results.append((1, loo0, float(np.var(w))))
    print(f"Pass 1 (uniform w): LOO={loo0*100:.1f}%  w_var={np.var(w):.4f}", flush=True)

    for pass_num in range(2, N_PASSES + 1):
        D = weighted_dist_matrix(phi, w)
        nearest = np.argmin(D, axis=1)  # (N,) — NN for each entry

        n_same = 0
        n_cross = 0
        for i in range(n):
            nn = nearest[i]
            if Y[nn] == Y[i]:
                n_same += 1
            else:
                n_cross += 1
                diff_sq = (phi[i] - phi[nn]) ** 2
                w += LR_W * diff_sq

        # Recompute LOO with updated w
        D_new = weighted_dist_matrix(phi, w)
        loo = loo_accuracy(D_new, Y)
        w_var = float(np.var(w))
        results.append((pass_num, loo, w_var))
        print(f"Pass {pass_num}: LOO={loo*100:.1f}%  w_var={w_var:.1f}"
              f"  same={n_same} cross={n_cross}", flush=True)

    final_loo = results[-1][1]
    print(flush=True)

    # ── w structure analysis ──────────────────────────────────────────────
    print("=== w structure analysis ===", flush=True)

    # Per-class mean w
    class_mean_w = {}
    for c in range(TRAIN_MAX):
        mask = (Y == c)
        if mask.any():
            class_mean_w[c] = float(w[mask].mean())

    print("  Per-class mean w (sorted by w):", flush=True)
    for c, mw in sorted(class_mean_w.items(), key=lambda x: -x[1])[:10]:
        count = int((Y == c).sum())
        print(f"    class {c:>2}: mean_w={mw:.1f}  n_entries={count}", flush=True)

    # Top-10 individual weights
    top10 = np.argsort(w)[-10:][::-1]
    print(f"  Top-10 dims (class, a, b, w):", flush=True)
    for idx in top10:
        print(f"    dim {int(idx):>3}: class={Y[idx]:>2}, a={int(A[idx]):>2}, b={B[idx]:>2}, w={w[idx]:.1f}", flush=True)

    # R^2: variance of w explained by class membership
    w_mean = w.mean()
    ss_total = ((w - w_mean) ** 2).sum()
    class_means_arr = np.array([class_mean_w.get(int(Y[i]), w_mean) for i in range(n)])
    ss_between = ((class_means_arr - w_mean) ** 2).sum()
    r2 = float(ss_between / ss_total) if ss_total > 0 else 0.0
    w_class_correlated = r2 > 0.1
    print(f"  w variance explained by class (R^2): {r2:.3f}", flush=True)

    # Check if w correlates with b-group (alternative structure)
    b_mean_w = {}
    for b_val in range(1, TRAIN_MAX + 1):
        mask = (B == b_val)
        if mask.any():
            b_mean_w[b_val] = float(w[mask].mean())
    b_means_arr = np.array([b_mean_w[int(B[i])] for i in range(n)])
    ss_between_b = ((b_means_arr - w_mean) ** 2).sum()
    r2_b = float(ss_between_b / ss_total) if ss_total > 0 else 0.0
    print(f"  w variance explained by b-group (R^2): {r2_b:.3f}", flush=True)
    print(flush=True)

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    success = final_loo > 0.70 and w_class_correlated
    killed = final_loo < 0.50

    print("=" * 65, flush=True)
    print("STEP 310 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"Final LOO (pass {N_PASSES}):          {final_loo*100:.1f}%", flush=True)
    print(f"w class-correlated (R^2={r2:.3f}): {'yes' if w_class_correlated else 'no'}", flush=True)
    print(f"w b-correlated (R^2={r2_b:.3f}):    {'yes' if r2_b > 0.1 else 'no'}", flush=True)
    print(f"Kill (LOO < 50%):            {'TRIGGERED' if killed else 'not triggered'}", flush=True)
    print(f"Success (LOO>70% AND class-corr): {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Substrate discovers class structure from raw distances.", flush=True)
    elif killed:
        print("KILLED -- LOO < 50%. Raw distance representation insufficient.", flush=True)
    else:
        print("PARTIAL -- Above kill, below success.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
