#!/usr/bin/env python3
"""
Step 87 -- High-convergence composition + chaining test.

Spec:
Part 1: Top-8 orbit EFs, 8x8 at 5000 steps. Does convergence improve vs 1000 steps?
Part 2: Pick 4 EFs with all-pairwise convergence. All 24 permutations of length-4 chain.
  result = ((A o B) o C) o D at 5000 steps per intermediate.
"""
import sys, random, math, time, collections, itertools
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 4
ALPHA = 1.2
BETA = 0.8
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
FIND_STEPS = 1000
COMPOSE_STEPS_FAST = 1000   # for orbit pre-scan
COMPOSE_STEPS_FULL = 5000   # for Part 1+2
KNOWN_COS = 0.95
BASIN_COS = 0.99

N_TARGET_EFS = 31
SEED = 42

# Top-8 orbit EFs from Step 84 analysis (one per orbit cluster, hub first):
# Clusters: {0,20,24}, {2,5,15}, {3,12,14}, {4,10,16}, {6,8,13}, {7,11,23}, {1,17,21}, {9,25}
# Hub: EF_10 (orbit=10), picking cluster representatives
TOP8_EF_IDXS = [10, 0, 2, 3, 6, 7, 1, 9]


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA / K)))


def psi(Mi, Mj):
    avg = mscale(madd(Mi, Mj), ALPHA / 2.0)
    prod = mscale(mmul(Mi, Mj), BETA / K)
    return mtanh(madd(avg, prod))


def converge(M, max_steps, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        phi_M = phi(M)
        d = frob(msub(phi_M, M))
        M = madd(M, mscale(msub(phi_M, M), DT))
        M = mclip(M, MAX_NORM)
        if d < tol:
            return M, True
    return M, False


def same_basin(A, B):
    return abs(mcosine(A, B)) > BASIN_COS


def copy_mat(M):
    return [[M[i][j] for j in range(K)] for i in range(K)]


def find_nearest(M, ef_list, threshold=KNOWN_COS):
    best_idx, best_cos = -1, -1.0
    for idx, ef in enumerate(ef_list):
        c = abs(mcosine(M, ef))
        if c > best_cos:
            best_cos = c
            best_idx = idx
    if best_cos >= threshold:
        return best_idx, best_cos
    return -1, best_cos


def find_eigenforms(n_needed, seed=SEED + 10, max_attempts=8000):
    rng = random.Random(seed)
    found = []
    for _ in range(max_attempts):
        M0 = [[rng.uniform(-2, 2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge(M0, FIND_STEPS)
        if conv and all(not same_basin(M_f, s) for s in found):
            found.append(M_f)
            if len(found) == n_needed:
                break
    return found


def compose_pair(Mi, Mj, steps=COMPOSE_STEPS_FULL):
    C = psi(Mi, Mj)
    C_f, conv = converge(C, steps)
    return (C_f if conv else None), conv


def ef_fingerprint(M, all_known):
    idx, cos_val = find_nearest(M, all_known, KNOWN_COS)
    if idx >= 0:
        return idx
    all_known.append(copy_mat(M))
    return len(all_known) - 1


def main():
    t0 = time.time()
    print("Step 87 -- High-convergence composition + chaining test", flush=True)
    print(f"COMPOSE_STEPS_FULL={COMPOSE_STEPS_FULL}", flush=True)
    print()

    # ── Find base eigenforms ──────────────────────────────────────────────────
    print("Finding base eigenforms...", flush=True)
    base_efs = find_eigenforms(N_TARGET_EFS)
    N = len(base_efs)
    print(f"  Found {N} base eigenforms")
    print()

    # ── Select top-8 orbit EFs ────────────────────────────────────────────────
    # Use hardcoded cluster representatives from Step 84 orbit analysis
    # (one EF per orbit cluster, hub EF_10 first)
    ef8_idxs = [idx for idx in TOP8_EF_IDXS if idx < N]
    print(f"  Top-8 orbit EFs (from Step 84 clusters): {ef8_idxs}")
    print(f"  (Hub=EF_10, one representative per orbit cluster)")
    print()

    ef8_mats = [base_efs[idx] for idx in ef8_idxs]
    n8 = len(ef8_idxs)

    # Registry for fingerprinting
    all_known = [copy_mat(ef) for ef in base_efs]

    # ── Part 1: 8x8 at 5000 steps ─────────────────────────────────────────────
    print(f"=== Part 1: {n8}x{n8} composition table (COMPOSE_STEPS={COMPOSE_STEPS_FULL}) ===", flush=True)

    table8 = [[-1] * n8 for _ in range(n8)]
    conv8 = [[False] * n8 for _ in range(n8)]
    novel8_count = 0

    for i in range(n8):
        for j in range(n8):
            M_f, conv = compose_pair(ef8_mats[i], ef8_mats[j])
            conv8[i][j] = conv
            if conv:
                fp = ef_fingerprint(M_f, all_known)
                table8[i][j] = fp
                if fp >= N:
                    novel8_count += 1

    n_conv8 = sum(conv8[i][j] for i in range(n8) for j in range(n8))
    n_known8 = sum(1 for i in range(n8) for j in range(n8)
                   if conv8[i][j] and table8[i][j] < N)
    print(f"  Convergence: {n_conv8}/{n8*n8} ({n_conv8/(n8*n8)*100:.1f}%)")
    print(f"  Known EF results: {n_known8}, Novel: {novel8_count}")
    print()

    # Print table
    labels8 = [f"E{idx}" for idx in ef8_idxs]
    header = "  " + " " * 5 + "".join(f"  {l:>3}" for l in labels8)
    print(header)
    for i in range(n8):
        row = f"  {labels8[i]:>4}|"
        for j in range(n8):
            if conv8[i][j]:
                fp = table8[i][j]
                entry = f"E{fp}" if fp < N else "N"
            else:
                entry = "?"
            row += f"  {entry:>3}"
        print(row)
    print()

    # Check commutativity within 8x8
    print("  Commutativity check (x o y == y o x):")
    n_comm = 0
    n_comm_total = 0
    non_comm_pairs = []
    for i in range(n8):
        for j in range(i + 1, n8):
            if conv8[i][j] and conv8[j][i]:
                n_comm_total += 1
                if table8[i][j] == table8[j][i]:
                    n_comm += 1
                else:
                    non_comm_pairs.append((ef8_idxs[i], ef8_idxs[j],
                                           table8[i][j], table8[j][i]))
    if n_comm_total > 0:
        print(f"    {n_comm}/{n_comm_total} commutative ({n_comm/n_comm_total*100:.1f}%)")
    if non_comm_pairs:
        print(f"    Non-commutative pairs:")
        for a, b, ab, ba in non_comm_pairs:
            print(f"      EF_{a} o EF_{b} = EF_{ab}, EF_{b} o EF_{a} = EF_{ba}")
    print()

    # ── Part 2: Find 4 EFs with all-pairwise convergence ─────────────────────
    print("=== Part 2: Length-4 permutation chains ===", flush=True)

    # Find 4 EFs from the 8 where ALL 4x4=16 pairwise compositions converge
    chosen4 = None
    for combo in itertools.combinations(range(n8), 4):
        all_conv = all(conv8[i][j] for i in combo for j in combo)
        if all_conv:
            chosen4 = list(combo)
            break

    if chosen4 is None:
        # Relax: find 4 with max pairwise convergence
        best_combo = None
        best_count = 0
        for combo in itertools.combinations(range(n8), 4):
            count = sum(1 for i in combo for j in combo if conv8[i][j])
            if count > best_count:
                best_count = count
                best_combo = list(combo)
        chosen4 = best_combo
        c4_conv = best_count
        print(f"  No fully-connected 4-EF set. Using best: slots {chosen4}")
        print(f"  Pairwise convergence within group: {c4_conv}/16 ({c4_conv/16*100:.1f}%)")
    else:
        print(f"  Found fully-connected 4-EF set: slots {chosen4}")
        c4_conv = 16

    if chosen4 is None:
        print("  Cannot find 4-EF group. Skipping Part 2.")
    else:
        c4_ef_idxs = [ef8_idxs[s] for s in chosen4]
        c4_mats = [ef8_mats[s] for s in chosen4]
        labels4 = [f"EF_{idx}" for idx in c4_ef_idxs]
        print(f"  Using EFs: {labels4}")
        print()

        # All 24 permutations of [A,B,C,D]
        print(f"  Computing all 24 permutations of {labels4}...")
        print(f"  result = ((A o B) o C) o D at {COMPOSE_STEPS_FULL} steps each...", flush=True)
        print()

        perm_results = {}
        n_perm_conv = 0
        n_perm_total = 0

        for perm in itertools.permutations(range(4)):
            n_perm_total += 1
            idxs = [chosen4[p] for p in perm]
            mats = [ef8_mats[idx] for idx in idxs]
            ef_names = [f"EF_{ef8_idxs[idx]}" for idx in idxs]

            # Chain: ((m0 o m1) o m2) o m3
            M_cur, conv = compose_pair(mats[0], mats[1])
            if not conv:
                perm_results[perm] = None
                continue

            ok = True
            for step in range(2, 4):
                M_cur, conv = compose_pair(M_cur, mats[step])
                if not conv:
                    ok = False
                    break

            if ok and M_cur is not None:
                n_perm_conv += 1
                fp = ef_fingerprint(M_cur, all_known)
                perm_results[perm] = fp
                fp_label = f"EF_{fp}" if fp < N else f"N{fp-N}"
                print(f"    ({' o '.join(ef_names)}) -> {fp_label}")
            else:
                perm_results[perm] = None
                print(f"    ({' o '.join(ef_names)}) -> NO CONV")

        print()
        conv_fps = [fp for fp in perm_results.values() if fp is not None]
        distinct_perms = len(set(conv_fps))
        print(f"  Convergence: {n_perm_conv}/{n_perm_total} ({n_perm_conv/n_perm_total*100:.1f}%)")
        print(f"  Distinct results: {distinct_perms} from {n_perm_conv} converged permutations")
        print()

        if n_perm_conv > 0:
            # Order sensitivity: group by frozenset of 4 (all same elements, only 1 combo)
            # Check if different orderings give different results
            result_set = set(conv_fps)
            if distinct_perms > 1:
                print(f"  ORDER IS ENCODED: {distinct_perms} distinct results from permutations")
            else:
                print(f"  ORDER NOT ENCODED: all permutations give same result (EF_{conv_fps[0]})")

            # Show which permutations give which result
            fp_to_perms = collections.defaultdict(list)
            for perm, fp in perm_results.items():
                if fp is not None:
                    fp_label = f"EF_{fp}" if fp < N else f"N{fp-N}"
                    fp_to_perms[fp_label].append(
                        tuple(f"EF_{ef8_idxs[chosen4[p]]}" for p in perm))
            print(f"  Equivalence classes:")
            for fp_label, perms in sorted(fp_to_perms.items()):
                print(f"    {fp_label}: {len(perms)} permutations")
                for p in perms[:3]:  # show up to 3
                    print(f"      {p}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print()
    print("=== Summary ===")
    print(f"  Part 1 — {n8}x{n8} at {COMPOSE_STEPS_FULL} steps:")
    print(f"    Convergence: {n_conv8}/{n8*n8} ({n_conv8/(n8*n8)*100:.1f}%)")
    if n_comm_total > 0:
        print(f"    Commutative: {n_comm}/{n_comm_total} ({n_comm/n_comm_total*100:.1f}%)")
    print(f"    Non-commutative pairs: {len(non_comm_pairs)}")
    if chosen4 is not None:
        print(f"  Part 2 — 24 permutations at {COMPOSE_STEPS_FULL} steps:")
        print(f"    Convergence: {n_perm_conv}/{n_perm_total} ({n_perm_conv/n_perm_total*100:.1f}%)")
        if n_perm_conv > 0:
            print(f"    Distinct results: {distinct_perms}")
            print(f"    Order encoded: {distinct_perms > 1}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
