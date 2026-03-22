#!/usr/bin/env python3
"""
Step 78 — Map eigenform landscape of Phi(M) = tanh(alpha*M + beta*M^2/k).

Spec: understand basin structure before changing approach.
10,000 random 4x4 matrices -> converge -> cluster fixed points.
Pure math, no dataset.
"""
import sys, random, math, time
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine, mzero

K=4; ALPHA=1.2; BETA=0.8; DT=0.03; MAX_NORM=3.0
CONVERGE_TOL=0.001; MAX_STEPS=1000
N_LANDSCAPE=10000; N_PERTURB=100; PERTURB_FROB=0.1
BASIN_COS_THRESH=0.99   # two fixed points are "same" if |cosine| > 0.99
SEED=42


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA/K)))

def iterate_to_convergence(M):
    for step in range(MAX_STEPS):
        phi_M = phi(M)
        delta = frob(msub(phi_M, M))
        M = madd(M, mscale(msub(phi_M, M), DT))
        M = mclip(M, MAX_NORM)
        if delta < CONVERGE_TOL:
            return M, step, True
    return M, MAX_STEPS, False

def rand_matrix(rng, lo=-2.0, hi=2.0):
    return [[rng.uniform(lo, hi) for _ in range(K)] for _ in range(K)]

def mat_neg(M):
    return [[-M[i][j] for j in range(K)] for i in range(K)]

def mat_transpose(M):
    return [[M[j][i] for j in range(K)] for i in range(K)]

def same_basin(M1, M2, thresh=BASIN_COS_THRESH):
    return abs(mcosine(M1, M2)) > thresh


def main():
    t0 = time.time()
    rng = random.Random(SEED)

    # ── §1: Check zero is unstable ────────────────────────────────────────────
    print("§1 Zero matrix stability check")
    M_zero = mzero(K)
    phi_zero = phi(M_zero)
    print(f"  frob(Phi(0)) = {frob(phi_zero):.6f}  (should be 0 since tanh(0)=0)")
    # Perturb zero slightly
    M_eps = [[0.001 if i==j else 0.0 for j in range(K)] for i in range(K)]
    M_conv, steps, conv = iterate_to_convergence(M_eps)
    print(f"  Small perturbation from 0: converged={conv} in {steps} steps, frob={frob(M_conv):.4f}")
    print(f"  (alpha=1.2 > 1 means zero IS a fixed point but unstable — repelled by perturbation)")
    print()

    # ── §2: Landscape survey — 10,000 random matrices ────────────────────────
    print(f"§2 Landscape survey ({N_LANDSCAPE} random matrices, uniform [-2,2])...")
    t1 = time.time()
    converged_pts = []   # list of (M_fixed, n_steps)
    n_converged = 0
    for i in range(N_LANDSCAPE):
        M = rand_matrix(rng)
        M_f, steps, conv = iterate_to_convergence(M)
        if conv:
            n_converged += 1
            converged_pts.append(M_f)
    print(f"  Converged: {n_converged}/{N_LANDSCAPE} ({n_converged/N_LANDSCAPE*100:.1f}%)")
    print(f"  Time: {time.time()-t1:.1f}s")
    print()

    # ── §3: Cluster fixed points ──────────────────────────────────────────────
    print("§3 Clustering fixed points (cosine threshold 0.99)...")
    clusters = []   # list of [M_representative, count, frob_values]
    for M_f in converged_pts:
        matched = False
        for c in clusters:
            if same_basin(M_f, c[0]):
                c[1] += 1
                c[2].append(frob(M_f))
                matched = True
                break
        if not matched:
            clusters.append([M_f, 1, [frob(M_f)]])
    clusters.sort(key=lambda c: -c[1])

    print(f"  Distinct fixed points: {len(clusters)}")
    print()
    print(f"  {'Basin':<8} {'Count':<8} {'Size%':<8} {'frob_mean':<12} {'frob_std':<12}")
    print(f"  {'-'*6}   {'-'*5}   {'-'*5}   {'-'*9}   {'-'*9}")
    for i, (M_rep, count, frob_vals) in enumerate(clusters[:20]):
        mean_f = sum(frob_vals) / len(frob_vals)
        std_f  = math.sqrt(sum((f-mean_f)**2 for f in frob_vals)/len(frob_vals)) if len(frob_vals)>1 else 0.0
        print(f"  {i:<8} {count:<8} {count/n_converged*100:<8.1f} {mean_f:<12.4f} {std_f:.4f}")
    if len(clusters) > 20:
        remaining = sum(c[1] for c in clusters[20:])
        print(f"  ... {len(clusters)-20} more basins covering {remaining} points")
    print()

    # ── §4: Symmetry analysis ─────────────────────────────────────────────────
    print("§4 Symmetry analysis on top-10 fixed points:")
    print(f"  {'Basin':<8} {'frob(M*)':<10} {'-M* fixed?':<14} {'M*^T fixed?':<14} {'cos(M*, -M*)':<14} {'cos(M*, M*T)'}")
    print(f"  {'-'*6}   {'-'*8}   {'-'*12}   {'-'*12}   {'-'*12}   {'-'*12}")
    for i, (M_rep, count, frob_vals) in enumerate(clusters[:10]):
        M_neg = mat_neg(M_rep)
        M_tr  = mat_transpose(M_rep)
        # Check if -M* and M*^T are also fixed points (frob(Phi(M)-M) < tol)
        neg_residual = frob(msub(phi(M_neg), M_neg))
        tr_residual  = frob(msub(phi(M_tr),  M_tr))
        cos_neg = mcosine(M_rep, M_neg)
        cos_tr  = mcosine(M_rep, M_tr)
        neg_fixed = neg_residual < 0.01
        tr_fixed  = tr_residual  < 0.01
        mean_f = sum(frob_vals)/len(frob_vals)
        print(f"  {i:<8} {mean_f:<10.4f} {'YES' if neg_fixed else 'no':<14} "
              f"{'YES' if tr_fixed else 'no':<14} {cos_neg:<14.4f} {cos_tr:.4f}")
    print()

    # ── §5: Perturbation basin stability ─────────────────────────────────────
    print(f"§5 Perturbation stability ({N_PERTURB} matrices, frob(perturbation)={PERTURB_FROB}):")
    rng2 = random.Random(SEED + 1)
    n_cross = 0
    for _ in range(N_PERTURB):
        M0 = rand_matrix(rng2)
        M0_conv, _, conv0 = iterate_to_convergence(M0)
        if not conv0:
            continue
        # Add perturbation of frob=PERTURB_FROB
        dM = rand_matrix(rng2, -1, 1)
        scale = PERTURB_FROB / (frob(dM) + 1e-15)
        M_perturbed = madd(M0_conv, mscale(dM, scale))
        M_p_conv, _, conv_p = iterate_to_convergence(M_perturbed)
        if conv_p and not same_basin(M0_conv, M_p_conv):
            n_cross += 1
    print(f"  Basin crossings: {n_cross}/{N_PERTURB} ({n_cross/N_PERTURB*100:.1f}%)")
    print(f"  (Low = stable storage; High = flat landscape)")
    print()

    elapsed = time.time() - t0
    print("=" * 65)
    print("  SUMMARY — Step 78")
    print("=" * 65)
    print(f"  Distinct fixed points:      {len(clusters)}")
    top3 = sum(c[1] for c in clusters[:3])
    print(f"  Top-3 basins cover:         {top3/n_converged*100:.1f}% of converged points")
    print(f"  Basin crossing rate:        {n_cross/N_PERTURB*100:.1f}%  (frob perturbation={PERTURB_FROB})")
    frob_all = [frob(M) for M in converged_pts]
    mean_frob = sum(frob_all)/len(frob_all)
    std_frob  = math.sqrt(sum((f-mean_frob)**2 for f in frob_all)/len(frob_all))
    print(f"  Frob of fixed points:       mean={mean_frob:.4f}  std={std_frob:.4f}")
    print(f"  Elapsed:                    {elapsed:.1f}s")
    print()
    # Interpret
    if len(clusters) < 5:
        print("  -> FEW BASINS: k=4 is too small. Must increase k.")
    elif top3/n_converged > 0.8:
        print("  -> DOMINATED LANDSCAPE: few attractors hold most of the volume.")
        print("     Classification capacity limited by attractor count, not k.")
    else:
        print("  -> RICH LANDSCAPE: many basins with reasonable coverage.")
        print("     Readout mechanism is the bottleneck, not the substrate.")
    if n_cross/N_PERTURB < 0.05:
        print("  -> STABLE STORAGE: small perturbations rarely cross basins.")
    else:
        print("  -> FRAGILE LANDSCAPE: perturbations frequently cross basins.")
    print("=" * 65)


if __name__ == '__main__':
    main()
