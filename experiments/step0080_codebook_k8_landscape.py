#!/usr/bin/env python3
"""
Step 80 -- k=8 landscape + basin routing.

Spec: k=4 exhausted. Does k=8 change the picture?

Part A: k=8 eigenform landscape
  - N=1000 random 8x8 matrices, uniform [-2,2]
  - Phi(M) = tanh(1.2*M + 0.8*M^2/8), dt=0.03, max_steps=500
  - Report: convergence rate, distinct eigenforms, frob, symmetries
  - Perturbation sweep: 50 converged seeds x eps=[0.1, 0.5, 1.0]

Part B: basin routing at k=8 (if Part A shows useful crossing regime)
  - 3 clusters in R^64 (d=k^2=64), sigma=0.3
  - 3 eigenform seeds, perturb with class inputs at critical eps
  - Report convergence and basin assignment by class

Runtime note: k=8 is ~8x more expensive per step vs k=4. Kept N=1000, max_steps=500.
"""
import sys, random, math, time, collections
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 8
ALPHA = 1.2
BETA = 0.8
DT = 0.03
MAX_NORM = 5.0
CONVERGE_TOL = 0.001
MAX_STEPS = 500
BASIN_COS = 0.99

N_MATRICES = 1000
N_PERTURB_SEEDS = 50
EPSILONS = [0.1, 0.5, 1.0]

# Part B
D = K * K  # 64
N_CLUSTERS = 3
N_TEST = 30
SIGMA = 0.3
SEED = 42


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA / K)))


def converge(M, max_steps=MAX_STEPS, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        phi_M = phi(M)
        d = frob(msub(phi_M, M))
        M = madd(M, mscale(msub(phi_M, M), DT))
        M = mclip(M, MAX_NORM)
        if d < tol:
            return M, True
    return M, False


def unit_frob(M):
    f = frob(M)
    return mscale(M, 1.0 / (f + 1e-15))


def same_basin(A, B):
    return abs(mcosine(A, B)) > BASIN_COS


def rand_mat_uniform(rng, lo=-2.0, hi=2.0):
    return [[rng.uniform(lo, hi) for _ in range(K)] for _ in range(K)]


def rand_mat_gauss(rng):
    return [[rng.gauss(0, 1) for _ in range(K)] for _ in range(K)]


def make_projection(d, k2, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0 / math.sqrt(d)) for _ in range(d)] for _ in range(k2)]


def project(P, x):
    flat = [sum(P[i][j] * x[j] for j in range(len(x))) for i in range(len(P))]
    return [flat[i * K:(i + 1) * K] for i in range(K)]


def generate_test(seed=SEED):
    rng = random.Random(seed)
    means = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(N_CLUSTERS)]
    test = []
    for c, mu in enumerate(means):
        test.append([([mu[j] + rng.gauss(0, SIGMA) for j in range(D)], c) for _ in range(N_TEST)])
    return test


def find_eigenforms(n, max_attempts=500, seed=SEED + 10):
    rng = random.Random(seed)
    found = []
    for _ in range(max_attempts):
        M0 = rand_mat_uniform(rng)
        M_f, conv = converge(M0)
        if conv and all(not same_basin(M_f, s) for s in found):
            found.append(M_f)
            if len(found) == n:
                break
    return found


def main():
    t0 = time.time()
    print("Step 80 -- k=8 landscape + basin routing", flush=True)
    print(f"K={K}, ALPHA={ALPHA}, BETA={BETA}, DT={DT}, N_MATRICES={N_MATRICES}, max_steps={MAX_STEPS}")
    print()

    # ── Part A: k=8 Landscape ─────────────────────────────────────────────────
    print("=== Part A: k=8 Eigenform Landscape ===", flush=True)

    rng = random.Random(SEED)
    n_conv = 0
    eigenforms = []
    frob_vals = []

    print(f"Running {N_MATRICES} random matrices...", flush=True)
    for i in range(N_MATRICES):
        M0 = rand_mat_uniform(rng)
        M_f, conv = converge(M0)
        if conv:
            n_conv += 1
            frob_vals.append(frob(M_f))
            # Check if new basin
            is_new = all(not same_basin(M_f, ef) for ef in eigenforms)
            if is_new:
                eigenforms.append(M_f)

    conv_rate = n_conv / N_MATRICES * 100
    n_basins = len(eigenforms)
    frob_mean = sum(frob_vals) / len(frob_vals) if frob_vals else 0.0
    frob_min = min(frob_vals) if frob_vals else 0.0
    frob_max = max(frob_vals) if frob_vals else 0.0

    print(f"  Convergence rate: {n_conv}/{N_MATRICES} ({conv_rate:.1f}%)")
    print(f"  Distinct eigenforms: {n_basins}")
    print(f"  Frob norm: mean={frob_mean:.4f}, min={frob_min:.4f}, max={frob_max:.4f}")
    print()

    # Symmetry check
    if eigenforms:
        print("Symmetry check (first 5 eigenforms):")
        for i, ef in enumerate(eigenforms[:5]):
            # Z2: -M*
            neg_ef = [[-ef[r][c] for c in range(K)] for r in range(K)]
            neg_conv, neg_ok = converge(neg_ef)
            neg_same = same_basin(neg_conv, neg_ef) if neg_ok else False

            # Transpose: M*^T
            tr_ef = [[ef[c][r] for c in range(K)] for r in range(K)]
            tr_cos = mcosine(ef, tr_ef)

            print(f"  EF {i}: frob={frob(ef):.4f}  -M* is eigenform: {neg_ok}  cos(M,M^T)={tr_cos:.3f}")
        print()

    # Perturbation sweep
    print(f"Perturbation sweep ({N_PERTURB_SEEDS} seeds x eps={EPSILONS}):")
    print(f"  {'eps':<6} {'conv%':<8} {'same%':<10} {'cross%':<10}")
    print(f"  {'-'*4}   {'-'*6}   {'-'*8}   {'-'*8}")

    rng_p = random.Random(SEED + 99)
    perturb_seeds = eigenforms[:N_PERTURB_SEEDS] if len(eigenforms) >= N_PERTURB_SEEDS else eigenforms

    critical_eps = None
    for eps in EPSILONS:
        n_ptotal = 0; n_pconv = 0; n_psame = 0; n_pcross = 0
        for seed_M in perturb_seeds:
            D_mat = rand_mat_gauss(rng_p)
            D_unit = unit_frob(D_mat)
            M_perturbed = madd(seed_M, mscale(D_unit, eps))
            M_pconv, pconv = converge(M_perturbed)
            n_ptotal += 1
            if pconv:
                n_pconv += 1
                if same_basin(M_pconv, seed_M):
                    n_psame += 1
                else:
                    n_pcross += 1

        pconv_pct = n_pconv / n_ptotal * 100 if n_ptotal > 0 else 0
        psame_pct = n_psame / n_pconv * 100 if n_pconv > 0 else 0
        pcross_pct = n_pcross / n_pconv * 100 if n_pconv > 0 else 0
        note = ""
        if n_pconv > 0 and n_pcross > 0 and critical_eps is None:
            critical_eps = eps
            note = "<-- CROSSING STARTS"
        print(f"  {eps:<6} {pconv_pct:<8.1f} {psame_pct:<10.1f} {pcross_pct:<10.1f} {note}")

    print()

    if critical_eps is None:
        print("  No crossing found at any tested epsilon.")
        print()
        print("Part A verdict: BARREN — no useful crossing regime at k=8.")
        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")
        return

    print(f"  Critical epsilon: {critical_eps}")
    print()

    # ── Part B: Basin routing ──────────────────────────────────────────────────
    print("=== Part B: Basin routing at k=8 ===", flush=True)
    print(f"d={D}, {N_CLUSTERS} clusters, sigma={SIGMA}, eps={critical_eps}")
    print()

    # Find 3 distinct eigenform seeds for routing
    print("Finding 3 eigenform seeds for basin routing...", flush=True)
    routing_seeds = find_eigenforms(3)
    print(f"  Found {len(routing_seeds)}")
    for i, s in enumerate(routing_seeds):
        print(f"  Seed {i}: frob={frob(s):.4f}")

    if len(routing_seeds) < 3:
        print("  ERROR: could not find 3 distinct eigenforms for Part B.")
        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")
        return
    print()

    # Generate test data and project into k^2=64 dimensional matrix space
    P = make_projection(D, K * K, seed=SEED + 1)
    test_clusters = generate_test()

    results = [[collections.Counter() for _ in range(N_CLUSTERS)] for _ in range(len(routing_seeds))]
    conv_total = [[0, 0] for _ in range(len(routing_seeds))]

    for cls_idx, class_data in enumerate(test_clusters):
        for x, c in class_data:
            R = project(P, x)
            R_unit = unit_frob(R)
            for seed_idx, M_star in enumerate(routing_seeds):
                M_prime = madd(M_star, mscale(R_unit, critical_eps))
                M_final, conv = converge(M_prime)
                conv_total[seed_idx][1] += 1
                if conv:
                    conv_total[seed_idx][0] += 1
                    if same_basin(M_final, M_star):
                        results[seed_idx][cls_idx]['same'] += 1
                    else:
                        matched = False
                        for si, other_seed in enumerate(routing_seeds):
                            if si != seed_idx and same_basin(M_final, other_seed):
                                results[seed_idx][cls_idx][f'seed{si}'] += 1
                                matched = True
                                break
                        if not matched:
                            results[seed_idx][cls_idx]['other'] += 1
                else:
                    results[seed_idx][cls_idx]['noconv'] += 1

    print("Convergence rates:")
    for i, (conv, total) in enumerate(conv_total):
        print(f"  Seed {i}: {conv}/{total} ({conv/total*100:.1f}%)")
    print()

    print("Basin assignment by (seed, class):")
    print(f"  {'':15}", end="")
    for c in range(N_CLUSTERS):
        print(f"  {'class '+str(c):<25}", end="")
    print()

    class_dependent = False
    for seed_idx in range(len(routing_seeds)):
        print(f"  Seed {seed_idx:<10}", end="")
        top_basins = []
        for cls_idx in range(N_CLUSTERS):
            ctr = results[seed_idx][cls_idx]
            if ctr:
                top = ctr.most_common(1)[0]
                top_basins.append(top[0])
                print(f"  {top[0]:<10}({top[1]:>3}/{N_TEST})", end="")
            else:
                top_basins.append(None)
                print(f"  {'none':<25}", end="")
        print()
        if len(set(str(b) for b in top_basins)) > 1:
            class_dependent = True

    print()
    verdict = "CLASS-DEPENDENT: different classes -> different basins" if class_dependent \
              else "NOT CLASS-DEPENDENT: all classes -> same basin pattern"
    print(f"  -> {verdict}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
