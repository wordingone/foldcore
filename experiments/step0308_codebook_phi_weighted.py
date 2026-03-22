#!/usr/bin/env python3
"""
Step 308 -- Frozen phi-absorption (308a) vs Learned weights (308b).

Spec. Two experiments with the same codebook structure.

BUG FIX vs initial run: phi must use LOO exclusion (exclude self from own
phi computation), matching Step 296's protocol. Without exclusion, every
vector has 0 at its own class slot 0 -> all unique -> NN fails (12% LOO).
With proper LOO exclusion: Step 296 gets 86.8%.

308a CONTROL: Pass 1 = all spawn. Passes 2-10: phi-NN absorption, alpha=0.01.
308b TEST: Same + learnable w. Cross-class -> upweight differing phi dims.

Kill: 308b LOO < 308a LOO - 20pp.
Success: 308b LOO >= 308a LOO - 5pp AND w converges (variance decreases).
"""

import time
import numpy as np

K = 5
MAX_CLASS = 20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
ALPHA = 0.01
LR_W = 0.1
N_PASSES = 10
PHI_DIM = MAX_CLASS * K
STEP296_REF = 0.868


# ─── phi computation ─────────────────────────────────────────────────────────

def compute_phi(a_q, b_q, A, B, Y, excl=-1):
    """
    phi for query (a_q, b_q).
    Per-class sorted top-K distances to same-b codebook vectors.
    excl >= 0: exclude that index (LOO). excl = -1: no exclusion.
    """
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
    """
    phi for all n codebook vectors with LOO exclusion.
    phi[i] = compute_phi(A[i], B[i], A, B, Y, excl=i)
    This is Step 296's protocol. O(n^2).
    """
    n = len(A)
    result = np.full((n, PHI_DIM), SENTINEL, dtype=np.float32)
    for i in range(n):
        result[i] = compute_phi(A[i], B[i], A, B, Y, excl=i)
    return result


# ─── LOO accuracy ─────────────────────────────────────────────────────────────

def loo_accuracy(A, B, Y, w=None):
    """
    Step 296 LOO protocol: phi[i] excludes i. NN among j != i.
    This gives 86.8% on unmodified training data.
    """
    phi_all = compute_phi_all_loo(A, B, Y)
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


# ─── 308a: Frozen phi absorption ─────────────────────────────────────────────

def run_308a(A, B, Y, train_data):
    A = A.copy()
    results = []

    loo0 = loo_accuracy(A, B, Y)
    results.append((1, float(loo0), 0, 0))
    print(f"    Pass 1 (spawn all): CB={len(A)}, LOO={loo0*100:.1f}%", flush=True)

    for pass_num in range(2, N_PASSES + 1):
        # Use LOO phi for absorption: exclude self when computing phi query
        phi_all_loo = compute_phi_all_loo(A, B, Y)
        n_same = 0
        n_cross = 0

        for idx, (a, b, y) in enumerate(train_data):
            phi_q = phi_all_loo[idx]   # phi of training example, excl self
            diffs = phi_all_loo - phi_q
            dists = (diffs * diffs).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))

            if Y[nearest] == y:
                n_same += 1
            else:
                n_cross += 1

            # Blend nearest toward input (a-space)
            A[nearest] = (1.0 - ALPHA) * A[nearest] + ALPHA * a

        loo = loo_accuracy(A, B, Y)
        results.append((pass_num, float(loo), n_same, n_cross))
        print(f"    Pass {pass_num}: LOO={loo*100:.1f}%  same={n_same} cross={n_cross}",
              flush=True)

    return results, A


# ─── 308b: Learned weights ────────────────────────────────────────────────────

def run_308b(A, B, Y, train_data):
    A = A.copy()
    w = np.ones(PHI_DIM, dtype=np.float32)
    results = []

    loo0 = loo_accuracy(A, B, Y, w=w)
    results.append((1, float(loo0), float(np.var(w)), 0, 0))
    print(f"    Pass 1 (spawn all): CB={len(A)}, LOO={loo0*100:.1f}%, w_var={np.var(w):.6f}",
          flush=True)

    for pass_num in range(2, N_PASSES + 1):
        phi_all_loo = compute_phi_all_loo(A, B, Y)
        n_same = 0
        n_cross = 0

        for idx, (a, b, y) in enumerate(train_data):
            phi_q = phi_all_loo[idx]
            diffs = phi_all_loo - phi_q
            dists = (diffs * diffs * w).sum(axis=1)
            dists[idx] = float('inf')
            nearest = int(np.argmin(dists))

            if Y[nearest] == y:
                n_same += 1
            else:
                n_cross += 1
                # Upweight dimensions where cross-class pair differs
                diff_sq = (phi_q - phi_all_loo[nearest]) ** 2
                w += LR_W * diff_sq

            A[nearest] = (1.0 - ALPHA) * A[nearest] + ALPHA * a

        loo = loo_accuracy(A, B, Y, w=w)
        w_var = float(np.var(w))
        results.append((pass_num, float(loo), w_var, n_same, n_cross))
        print(f"    Pass {pass_num}: LOO={loo*100:.1f}%  w_var={w_var:.4f}  "
              f"same={n_same} cross={n_cross}", flush=True)

    return results, A, w


def main():
    t0 = time.time()
    print("Step 308 -- Frozen phi (308a) vs Learned weights (308b)", flush=True)
    print(f"N passes={N_PASSES}  alpha={ALPHA}  lr_w={LR_W}  K={K}", flush=True)
    print(f"Step 296 LOO ref: {STEP296_REF*100:.1f}%  (this should match pass 1)", flush=True)
    print(flush=True)

    train_data = []
    A_list, B_list, Y_list = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            y = a % b
            train_data.append((float(a), b, y))
            A_list.append(float(a))
            B_list.append(b)
            Y_list.append(y)
    A0 = np.array(A_list, dtype=np.float32)
    B0 = np.array(B_list, dtype=np.int32)
    Y0 = np.array(Y_list, dtype=np.int32)

    # ─── 308a ──────────────────────────────────────────────────────────────
    print("=== 308a: CONTROL -- Frozen phi ===", flush=True)
    t_a = time.time()
    res_a, A_final_a = run_308a(A0, B0, Y0, train_data)
    final_loo_a = res_a[-1][1]
    print(f"  308a final LOO: {final_loo_a*100:.1f}%  [{time.time()-t_a:.1f}s]\n", flush=True)

    # ─── 308b ──────────────────────────────────────────────────────────────
    print("=== 308b: TEST -- Learned weights ===", flush=True)
    t_b = time.time()
    res_b, A_final_b, w_final = run_308b(A0, B0, Y0, train_data)
    final_loo_b = res_b[-1][1]
    print(f"  308b final LOO: {final_loo_b*100:.1f}%  [{time.time()-t_b:.1f}s]\n", flush=True)

    # ─── w convergence ─────────────────────────────────────────────────────
    print("=== w analysis ===", flush=True)
    w_vars = [r[2] for r in res_b]
    w_converged = len(w_vars) > 2 and w_vars[-1] < w_vars[2]
    print(f"  w_var trend: {['increasing', 'CONVERGING'][int(w_converged)]}", flush=True)
    print(f"  w mean: {np.mean(w_final):.3f}  var: {np.var(w_final):.4f}", flush=True)
    top5 = [(i, i // K, i % K, float(w_final[i])) for i in np.argsort(w_final)[-5:][::-1]]
    print(f"  Top-5 dims (class, k, w):", flush=True)
    for idx, c, k, wv in top5:
        print(f"    dim {idx:>3}: class={c:>2}, k={k}, w={wv:.3f}", flush=True)
    print(flush=True)

    # ─── Comparison table ──────────────────────────────────────────────────
    print("=== LOO per pass ===", flush=True)
    print(f"{'Pass':>5} | {'308a LOO':>9} | {'308b LOO':>9} | {'delta':>7}", flush=True)
    print("-" * 42, flush=True)
    for i in range(min(len(res_a), len(res_b))):
        la = res_a[i][1]
        lb = res_b[i][1]
        print(f"  {res_a[i][0]:>3} | {la*100:>8.1f}% | {lb*100:>8.1f}% | {(lb-la)*100:>+6.1f}pp",
              flush=True)
    print(flush=True)

    elapsed = time.time() - t0
    delta = final_loo_b - final_loo_a
    print("=" * 65, flush=True)
    print("STEP 308 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"308a (frozen phi): {final_loo_a*100:.1f}%", flush=True)
    print(f"308b (learned w):  {final_loo_b*100:.1f}%", flush=True)
    print(f"Delta: {delta*100:+.1f}pp  (kill: -20pp, success: -5pp)", flush=True)
    print(f"w converging: {w_converged}", flush=True)
    print(flush=True)

    if delta >= -0.05 and w_converged:
        print("SUCCESS -- within 5pp AND w converging.", flush=True)
        print("Absorption dynamics discover a metric.", flush=True)
    elif delta >= -0.05:
        print(f"PARTIAL -- within 5pp but w diverging.", flush=True)
    elif delta >= -0.20:
        print(f"PARTIAL -- {delta*100:+.1f}pp. Above kill criterion.", flush=True)
    else:
        print(f"KILLED -- {delta*100:+.1f}pp < -20pp kill criterion.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
