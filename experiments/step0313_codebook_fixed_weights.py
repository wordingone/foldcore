#!/usr/bin/env python3
"""
Step 313 -- Fixed decreasing weights on phi. Spec.

Weight schedules applied to per-class sorted top-K phi distances:
  1. uniform:   w_k = 1          (Step 296 baseline)
  2. 1/(k+1):   w_k = 1/(k+1)
  3. exp(-k):   w_k = exp(-k)
  4. k=0 only:  w_k = 1 if k==0 else 0

Three evaluations per schedule:
  1. LOO in-dist on a%b (compare to phi=86.8%, learned w=91.2%)
  2. OOD a%b (a=21..50, compare to phi OOD=18%, learned w OOD=17.3%)
  3. floor(a/b) LOO (compare to phi=86.8%, learned w=86.5%)

Kill: all schedules <= 86.8% in-dist.
Success: any schedule > 86.8% in-dist AND > 30% OOD.
"""

import time
import numpy as np

K = 5
MAX_CLASS_MOD = 20    # a%b classes 0..19
MAX_CLASS_FLOOR = 21  # floor(a/b) classes 0..20
TRAIN_MAX = 20
SENTINEL = float(TRAIN_MAX * 3)
PHI_DIM_MOD = MAX_CLASS_MOD * K    # 100
PHI_DIM_FLOOR = MAX_CLASS_FLOOR * K  # 105

# Weight schedules: (name, w_k values for k=0..K-1)
SCHEDULES = [
    ("uniform",   np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)),
    ("1/(k+1)",   np.array([1.0, 0.5, 1/3, 0.25, 0.2], dtype=np.float32)),
    ("exp(-k)",   np.array([np.exp(-k) for k in range(K)], dtype=np.float32)),
    ("k=0 only",  np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)),
]


def make_w_full(w_k, max_class, phi_dim):
    """Expand per-slot weights to full phi vector (repeated max_class times)."""
    w = np.tile(w_k, max_class).astype(np.float32)
    assert len(w) == phi_dim
    return w


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


def loo_accuracy_weighted(phi_all, Y, w_full):
    """Vectorized weighted LOO NN accuracy."""
    n = len(Y)
    # Distance matrix: D[i,j] = sum(w * (phi[i] - phi[j])^2)
    # Use: D[i,j] = sum(w*phi[i]^2) + sum(w*phi[j]^2) - 2*sum(w*phi[i]*phi[j])
    phi_sq_w = (phi_all ** 2 * w_full).sum(axis=1)   # (N,)
    phi_w = phi_all * w_full                           # (N, D)
    cross = phi_w @ phi_all.T                          # (N, N)
    D = phi_sq_w[:, None] + phi_sq_w[None, :] - 2 * cross
    np.fill_diagonal(D, float('inf'))
    nearest = np.argmin(D, axis=1)
    return float((Y[nearest] == Y).sum()) / n


def ood_accuracy_weighted(phi_cb, Y_cb, ood_phi, ood_y, w_full):
    """Evaluate OOD queries against codebook phi with weights."""
    correct = 0
    for phi_q, y_true in zip(ood_phi, ood_y):
        diffs = phi_cb - phi_q
        dists = (diffs * diffs * w_full).sum(axis=1)
        nearest = int(np.argmin(dists))
        if Y_cb[nearest] == y_true:
            correct += 1
    return correct / len(ood_y)


def build_mod_data():
    """a%b training data, a in 1..20, b in 1..20."""
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(float(a)); B.append(b); Y.append(a % b)
    return (np.array(A, dtype=np.float32),
            np.array(B, dtype=np.int32),
            np.array(Y, dtype=np.int32))


def build_floor_data():
    """floor(a/b) training data, a in 1..20, b in 1..20."""
    A, B, Y = [], [], []
    for a in range(1, TRAIN_MAX + 1):
        for b in range(1, TRAIN_MAX + 1):
            A.append(float(a)); B.append(b); Y.append(a // b)
    return (np.array(A, dtype=np.float32),
            np.array(B, dtype=np.int32),
            np.array(Y, dtype=np.int32))


def build_ood_data():
    """a%b OOD queries, a in 21..50, b in 1..20."""
    queries = []
    for a in range(21, 51):
        for b in range(1, TRAIN_MAX + 1):
            queries.append((a, b, a % b))
    return queries


def main():
    t0 = time.time()
    print("Step 313 -- Fixed decreasing weights on phi", flush=True)
    print(f"K={K}  Schedules: {[s[0] for s in SCHEDULES]}", flush=True)
    print(flush=True)

    # ── Build datasets ─────────────────────────────────────────────────────
    A_mod, B_mod, Y_mod = build_mod_data()
    A_floor, B_floor, Y_floor = build_floor_data()
    ood_queries = build_ood_data()
    ood_y = np.array([q[2] for q in ood_queries], dtype=np.int32)

    # ── Precompute phi (once per dataset) ─────────────────────────────────
    print("Precomputing phi (a%b, LOO)...", flush=True)
    t_phi = time.time()
    phi_mod = compute_phi_all_loo(A_mod, B_mod, Y_mod, MAX_CLASS_MOD, PHI_DIM_MOD)
    print(f"  done [{time.time()-t_phi:.2f}s]", flush=True)

    print("Precomputing phi (floor(a/b), LOO)...", flush=True)
    t_phi2 = time.time()
    phi_floor = compute_phi_all_loo(A_floor, B_floor, Y_floor, MAX_CLASS_FLOOR, PHI_DIM_FLOOR)
    print(f"  done [{time.time()-t_phi2:.2f}s]", flush=True)

    # OOD phi: queries against the a%b codebook (no LOO — OOD queries not in codebook)
    print("Computing OOD phi...", flush=True)
    t_ood = time.time()
    ood_phi = np.array([
        compute_phi(float(a), b, A_mod, B_mod, Y_mod, MAX_CLASS_MOD, PHI_DIM_MOD, excl=-1)
        for (a, b, _) in ood_queries
    ], dtype=np.float32)
    print(f"  done [{time.time()-t_ood:.2f}s]\n", flush=True)

    # ── Evaluate all schedules ─────────────────────────────────────────────
    results = []
    for name, w_k in SCHEDULES:
        w_mod = make_w_full(w_k, MAX_CLASS_MOD, PHI_DIM_MOD)
        w_floor = make_w_full(w_k, MAX_CLASS_FLOOR, PHI_DIM_FLOOR)

        acc_indist = loo_accuracy_weighted(phi_mod, Y_mod, w_mod)
        acc_ood = ood_accuracy_weighted(phi_mod, Y_mod, ood_phi, ood_y, w_mod)
        acc_floor = loo_accuracy_weighted(phi_floor, Y_floor, w_floor)

        print(f"[{name}]", flush=True)
        print(f"  in-dist:  {acc_indist*100:.1f}%  (baseline: 86.8%  delta: {(acc_indist-0.868)*100:+.1f}pp)", flush=True)
        print(f"  OOD:      {acc_ood*100:.1f}%  (baseline: 18%    delta: {(acc_ood-0.18)*100:+.1f}pp)", flush=True)
        print(f"  floor:    {acc_floor*100:.1f}%  (baseline: 86.8%  delta: {(acc_floor-0.868)*100:+.1f}pp)", flush=True)
        print(flush=True)

        results.append((name, acc_indist, acc_ood, acc_floor))

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    kill = all(r[1] <= 0.868 for r in results)
    success = any(r[1] > 0.868 and r[2] > 0.30 for r in results)

    print("=" * 65, flush=True)
    print("STEP 313 SUMMARY", flush=True)
    print("=" * 65, flush=True)
    print(f"{'Schedule':>10} | {'In-dist':>8} | {'OOD':>8} | {'Floor':>8}", flush=True)
    print("-" * 46, flush=True)
    for name, ind, ood, fl in results:
        print(f"  {name:>9} | {ind*100:>7.1f}% | {ood*100:>7.1f}% | {fl*100:>7.1f}%", flush=True)
    print(flush=True)
    print(f"Kill (all in-dist <= 86.8%):                {'TRIGGERED' if kill else 'not triggered'}", flush=True)
    print(f"Success (in-dist > 86.8% AND OOD > 30%):   {'YES' if success else 'NO'}", flush=True)
    print(flush=True)

    if success:
        print("SUCCESS -- Fixed prescribed weights generalize.", flush=True)
        print("Substrate discovery persists as physics.", flush=True)
    elif kill:
        print("KILLED -- No schedule beats uniform phi.", flush=True)
    else:
        print("PARTIAL -- Some in-dist improvement but OOD fails.", flush=True)

    print(f"\nElapsed: {elapsed:.1f}s", flush=True)


if __name__ == '__main__':
    main()
