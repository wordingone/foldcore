#!/usr/bin/env python3
"""
Step 79 — Basin-assignment classification probe.

Spec: does cross-application with input produce
class-dependent basin assignments on toy data (Step 74: 3 clusters, d=16)?

Part 1: For each (seed, input_class) pair, what basin does Psi(M*, R) converge to?
Part 2: If class-dependent patterns, use basin assignment as classifier.
"""
import sys, random, math, time, collections
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mzero, mrand, mcosine

K=4; ALPHA=1.2; BETA=0.8; DT=0.03; MAX_NORM=3.0
CONVERGE_TOL=0.001; MAX_STEPS=1000
D=16; N_CLUSTERS=3; N_TRAIN=100; N_TEST=30; SIGMA=0.3; SEED=42
BASIN_COS=0.99   # same basin if |cos| > 0.99


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M,M), BETA/K)))

def cross_apply(Mi, Mj):
    return mtanh(madd(mscale(madd(Mi,Mj), ALPHA/2), mscale(mmul(Mi,Mj), BETA/K)))

def converge(M, max_steps=MAX_STEPS, tol=CONVERGE_TOL):
    for step in range(max_steps):
        phi_M = phi(M)
        d = frob(msub(phi_M, M))
        M = madd(M, mscale(msub(phi_M, M), DT))
        M = mclip(M, MAX_NORM)
        if d < tol:
            return M, step, True
    return M, max_steps, False

def same_basin(A, B):
    return abs(mcosine(A, B)) > BASIN_COS

def make_projection(d, k2, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0/math.sqrt(d)) for _ in range(d)] for _ in range(k2)]

def project(P, x):
    flat = [sum(P[i][j]*x[j] for j in range(len(x))) for i in range(len(P))]
    return [flat[i*K:(i+1)*K] for i in range(K)]

def generate_data(seed=SEED):
    rng = random.Random(seed)
    means = [[rng.gauss(0,1) for _ in range(D)] for _ in range(N_CLUSTERS)]
    train = []; test = []
    for c, mu in enumerate(means):
        train.append([([mu[j]+rng.gauss(0,SIGMA) for j in range(D)], c) for _ in range(N_TRAIN)])
        test.append([([mu[j]+rng.gauss(0,SIGMA) for j in range(D)], c) for _ in range(N_TEST)])
    return train, test


# ─── Find 3 distinct eigenform seeds ─────────────────────────────────────────

def find_eigenform_seeds(n_needed=3, max_attempts=500, seed=SEED+10):
    rng = random.Random(seed)
    seeds = []
    attempts = 0
    while len(seeds) < n_needed and attempts < max_attempts:
        attempts += 1
        M0 = [[rng.uniform(-2,2) for _ in range(K)] for _ in range(K)]
        M_conv, steps, conv = converge(M0)
        if not conv:
            continue
        # Check if distinct from existing seeds
        is_new = all(not same_basin(M_conv, s) for s in seeds)
        if is_new:
            seeds.append(M_conv)
    return seeds, attempts


def main():
    t0 = time.time()
    print("Step 79 — Basin-assignment classification probe", flush=True)
    print(f"3 clusters x d={D} sigma={SIGMA}, {N_TRAIN} train {N_TEST} test/class")
    print()

    # ── Generate 3 eigenform seeds ─────────────────────────────────────────
    print("Finding 3 distinct eigenform seeds...", flush=True)
    seeds, attempts = find_eigenform_seeds(3)
    print(f"  Found {len(seeds)} seeds in {attempts} attempts")
    for i, s in enumerate(seeds):
        print(f"  Seed {i}: frob={frob(s):.4f}  residual={frob(msub(phi(s),s)):.6f}")
    if len(seeds) < 3:
        print("  ERROR: could not find 3 distinct eigenforms. Aborting.")
        return
    print()

    # ── Generate toy data and project ──────────────────────────────────────
    train_clusters, test_clusters = generate_data()
    P = make_projection(D, K*K, seed=SEED+1)
    def embed(x): return project(P, x)

    # ── Part 1: Basin probe ─────────────────────────────────────────────────
    print("Part 1: Basin probe — Psi(seed_j, R_input) -> which basin?", flush=True)
    print()

    # Known basins: collect from seeds + their negatives
    known_basins = []
    for s in seeds:
        known_basins.append(s)
        neg = [[-s[i][j] for j in range(K)] for i in range(K)]
        known_basins.append(neg)

    def assign_basin(M_conv):
        """Return index into known_basins, or -1 if new."""
        for idx, kb in enumerate(known_basins):
            if same_basin(M_conv, kb):
                return idx
        # New basin — add it
        known_basins.append(M_conv)
        return len(known_basins) - 1

    # Results: results[seed_idx][class_idx] = Counter of basin_ids
    results = [[collections.Counter() for _ in range(N_CLUSTERS)] for _ in range(len(seeds))]
    # Also track convergence rate
    conv_counts = [[0,0] for _ in range(len(seeds))]  # [converged, total]

    for cls_idx, class_data in enumerate(test_clusters):
        for x, c in class_data:
            R = embed(x)
            for seed_idx, M_star in enumerate(seeds):
                C0 = cross_apply(M_star, R)
                M_final, steps, conv = converge(C0)
                conv_counts[seed_idx][1] += 1
                if conv:
                    conv_counts[seed_idx][0] += 1
                    basin_id = assign_basin(M_final)
                    results[seed_idx][cls_idx][basin_id] += 1
                else:
                    results[seed_idx][cls_idx][-1] += 1  # -1 = no convergence

    print("Convergence rates (converged / total):")
    for i, (conv, total) in enumerate(conv_counts):
        print(f"  Seed {i}: {conv}/{total} ({conv/total*100:.1f}%)")
    print()

    print("Basin assignment table (seed x class -> top basin id : count):")
    print(f"  {'':12}", end="")
    for c in range(N_CLUSTERS):
        print(f"  {'class '+str(c):<20}", end="")
    print()
    print(f"  {'-'*10}", end="")
    for c in range(N_CLUSTERS):
        print(f"  {'-'*18}", end="")
    print()

    class_discriminable = True
    for seed_idx in range(len(seeds)):
        print(f"  Seed {seed_idx:<7}", end="")
        top_basins = []
        for cls_idx in range(N_CLUSTERS):
            ctr = results[seed_idx][cls_idx]
            if ctr:
                top = ctr.most_common(1)[0]
                top_basins.append(top[0])
                print(f"  basin {top[0]:>2} ({top[1]:>3}/{N_TEST})", end="")
            else:
                top_basins.append(-1)
                print(f"  {'none':<20}", end="")
        print()
        # Check if all 3 classes map to same basin → not discriminable
        if len(set(top_basins)) == 1:
            class_discriminable = False

    print()
    if class_discriminable:
        print("  -> DISCRIMINABLE: different classes map to different basins for at least one seed")
    else:
        print("  -> NOT DISCRIMINABLE: all classes map to same basin")
    print()

    # ── Part 2: Basin-assignment classifier ────────────────────────────────
    print("Part 2: Basin-assignment classifier (train on class->basin mapping)...", flush=True)

    # Training: for each seed, record which basin each training class maps to most often
    train_basin_maps = []  # [seed_idx] -> {basin_id: class_label}
    for seed_idx, M_star in enumerate(seeds):
        train_results = [collections.Counter() for _ in range(N_CLUSTERS)]
        for cls_idx, class_data in enumerate(train_clusters):
            for x, c in class_data[:20]:  # use first 20 to calibrate
                R = embed(x)
                C0 = cross_apply(M_star, R)
                M_final, _, conv = converge(C0)
                if conv:
                    bid = assign_basin(M_final)
                    train_results[cls_idx][bid] += 1
        # Build basin->class mapping (majority vote per basin id seen)
        basin_to_class = {}
        for cls_idx in range(N_CLUSTERS):
            if train_results[cls_idx]:
                top_bid = train_results[cls_idx].most_common(1)[0][0]
                if top_bid not in basin_to_class:
                    basin_to_class[top_bid] = cls_idx
        train_basin_maps.append(basin_to_class)
        print(f"  Seed {seed_idx} basin->class: {basin_to_class}", flush=True)

    # Test: classify each test sample
    def classify_basin(R):
        votes = collections.Counter()
        for seed_idx, M_star in enumerate(seeds):
            C0 = cross_apply(M_star, R)
            M_final, _, conv = converge(C0)
            if conv:
                bid = assign_basin(M_final)
                if bid in train_basin_maps[seed_idx]:
                    votes[train_basin_maps[seed_idx][bid]] += 1
        if not votes:
            return None
        return votes.most_common(1)[0][0]

    print()
    correct = 0; total = 0
    per_class_acc = []
    for cls_idx, class_data in enumerate(test_clusters):
        cls_correct = sum(1 for x, c in class_data if classify_basin(embed(x)) == c)
        per_class_acc.append(cls_correct/len(class_data))
        correct += cls_correct; total += len(class_data)
    aa = correct/total

    elapsed = time.time() - t0
    print("=" * 60)
    print("  RESULTS — Step 79")
    print("=" * 60)
    print(f"  Basin-assignment AA:  {aa*100:.1f}%  ({correct}/{total})")
    print(f"  Per-class: {[f'{a:.0%}' for a in per_class_acc]}")
    print(f"  Known basins at end: {len(known_basins)}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print()
    print(f"  Baseline (Step 74, cosine): 100%  CB=5  <1s")
    print("=" * 60)


if __name__ == '__main__':
    main()
