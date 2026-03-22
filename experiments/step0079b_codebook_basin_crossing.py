#!/usr/bin/env python3
"""
Step 79b — Basin crossing threshold sweep + class-dependent crossing.

Spec: additive perturbation stays in basins (Step 78).
Find critical epsilon where crossing starts, then test if it's class-dependent.

Part 1: epsilon sweep [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
Part 2: class-dependent crossing at critical epsilon
"""
import sys, random, math, time, collections
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mzero, mcosine

K=4; ALPHA=1.2; BETA=0.8; DT=0.03; MAX_NORM=3.0
CONVERGE_TOL=0.001; MAX_STEPS=1000
D=16; N_CLUSTERS=3; N_TEST=30; SIGMA=0.3; SEED=42
BASIN_COS=0.99
EPSILONS=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
N_RANDOM_DIRS=30


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M,M), BETA/K)))

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
    return mscale(M, 1.0/(f+1e-15))

def same_basin(A, B):
    return abs(mcosine(A, B)) > BASIN_COS

def rand_mat(rng):
    return [[rng.gauss(0,1) for _ in range(K)] for _ in range(K)]

def make_projection(d, k2, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0/math.sqrt(d)) for _ in range(d)] for _ in range(k2)]

def project(P, x):
    flat = [sum(P[i][j]*x[j] for j in range(len(x))) for i in range(len(P))]
    return [flat[i*K:(i+1)*K] for i in range(K)]

def generate_test(seed=SEED):
    rng = random.Random(seed)
    means = [[rng.gauss(0,1) for _ in range(D)] for _ in range(N_CLUSTERS)]
    test = []
    for c, mu in enumerate(means):
        test.append([([mu[j]+rng.gauss(0,SIGMA) for j in range(D)], c) for _ in range(N_TEST)])
    return test

def find_eigenforms(n=3, seed=SEED+10):
    rng = random.Random(seed)
    found = []
    for _ in range(500):
        M0 = [[rng.uniform(-2,2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge(M0)
        if conv and all(not same_basin(M_f, s) for s in found):
            found.append(M_f)
            if len(found) == n:
                break
    return found


def main():
    t0 = time.time()
    print("Step 79b — basin crossing threshold + class-dependent test", flush=True)

    print("Finding 3 eigenform seeds...", flush=True)
    seeds = find_eigenforms(3)
    print(f"  Found {len(seeds)}")
    for i, s in enumerate(seeds):
        print(f"  Seed {i}: frob={frob(s):.4f}")
    print()

    rng = random.Random(SEED+77)

    # ── Part 1: epsilon sweep ──────────────────────────────────────────────────
    print("Part 1: Epsilon sweep (30 random directions per epsilon per seed)")
    print(f"  {'eps':<6} {'conv%':<8} {'same%':<10} {'cross%':<10} {'notes'}")
    print(f"  {'-'*4}   {'-'*6}   {'-'*8}   {'-'*8}")

    critical_eps = None
    for eps in EPSILONS:
        n_conv = 0; n_same = 0; n_cross = 0; n_total = 0
        for seed_M in seeds:
            for _ in range(N_RANDOM_DIRS):
                D_mat = rand_mat(rng)
                D_unit = unit_frob(D_mat)
                M_perturbed = madd(seed_M, mscale(D_unit, eps))
                M_conv, conv = converge(M_perturbed)
                n_total += 1
                if conv:
                    n_conv += 1
                    if same_basin(M_conv, seed_M):
                        n_same += 1
                    else:
                        n_cross += 1
        pconv = n_conv/n_total*100
        psame = n_same/n_conv*100 if n_conv>0 else 0
        pcross = n_cross/n_conv*100 if n_conv>0 else 0
        note = ""
        if n_conv > 0 and n_cross > 0 and critical_eps is None:
            critical_eps = eps
            note = "<-- CROSSING STARTS"
        print(f"  {eps:<6} {pconv:<8.1f} {psame:<10.1f} {pcross:<10.1f} {note}")
    print()

    if critical_eps is None:
        critical_eps = EPSILONS[-1]
        print(f"  No crossing found. Using eps={critical_eps} for Part 2.")
    else:
        print(f"  Critical epsilon: {critical_eps}")
    print()

    # ── Part 2: Class-dependent crossing ──────────────────────────────────────
    print(f"Part 2: Class-dependent crossing at eps={critical_eps}")
    print("  (M' = M* + eps * unit_frob(R_input), iterate to convergence)")
    print()

    P = make_projection(D, K*K, seed=SEED+1)
    test_clusters = generate_test()

    # For each (seed, class): record basin assignment
    results = [[collections.Counter() for _ in range(N_CLUSTERS)] for _ in range(len(seeds))]
    conv_total = [[0,0] for _ in range(len(seeds))]  # [conv, total]

    for cls_idx, class_data in enumerate(test_clusters):
        for x, c in class_data:
            R = project(P, x)
            R_unit = unit_frob(R)
            for seed_idx, M_star in enumerate(seeds):
                M_prime = madd(M_star, mscale(R_unit, critical_eps))
                M_final, conv = converge(M_prime)
                conv_total[seed_idx][1] += 1
                if conv:
                    conv_total[seed_idx][0] += 1
                    # Assign basin: 0=same, negative=cross (index of which basin)
                    if same_basin(M_final, M_star):
                        results[seed_idx][cls_idx]['same'] += 1
                    else:
                        # Find which other known basin
                        matched = False
                        for si, other_seed in enumerate(seeds):
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
    for seed_idx in range(len(seeds)):
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
    if class_dependent:
        print("  -> BASIN ASSIGNMENT IS CLASSIFICATION. Vectors cannot replicate this.")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
