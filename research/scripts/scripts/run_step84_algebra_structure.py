#!/usr/bin/env python3
"""
Step 84 -- Eigenform algebra structure: families, closed sub-algebras, orbits.

[researcher] directive (mail 1202): is the composition table a STRUCTURE or a HASH?

Recomputes 26x26 table (same seed/params as Step 83), then analyzes:
Part 1: Family structure (M*, -M*, M*^T, -M*^T) and whether composition respects families.
Part 2: Largest closed sub-algebra within base EFs.
Part 3: One-step orbit size per EF — find hubs.
"""
import sys, random, math, time, collections
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 4
ALPHA = 1.2
BETA = 0.8
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
FIND_STEPS = 1000
COMPOSE_STEPS = 1000
KNOWN_COS = 0.95
BASIN_COS = 0.99
FAMILY_COS = 0.95   # threshold for "same family"

N_TARGET_EFS = 31
SEED = 42


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


def neg(M):
    return [[-M[i][j] for j in range(K)] for i in range(K)]


def transpose(M):
    return [[M[j][i] for j in range(K)] for i in range(K)]


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


def compose_pair(Mi, Mj):
    C = psi(Mi, Mj)
    C_f, conv = converge(C, COMPOSE_STEPS)
    return (C_f if conv else None), conv


def main():
    t0 = time.time()
    print("Step 84 -- Eigenform algebra structure", flush=True)
    print()

    # ── Find eigenforms (same seed as Step 83) ────────────────────────────────
    print(f"Finding base eigenforms...", flush=True)
    base_efs = find_eigenforms(N_TARGET_EFS)
    N = len(base_efs)
    print(f"  Found {N} base eigenforms")
    print()

    # ── Build 26x26 table ─────────────────────────────────────────────────────
    print(f"Building {N}x{N} composition table ({N*N} pairs)...", flush=True)

    # Track novel EFs generated
    all_efs = [copy_mat(ef) for ef in base_efs]
    novel_count = 0

    def get_or_register(M_f):
        idx, cos_val = find_nearest(M_f, all_efs, KNOWN_COS)
        if idx >= 0:
            return idx, False
        all_efs.append(copy_mat(M_f))
        nonlocal novel_count
        novel_count += 1
        return len(all_efs) - 1, True

    table = [[-1] * N for _ in range(N)]  # -1 = noconv
    for i in range(N):
        for j in range(N):
            M_f, conv = compose_pair(base_efs[i], base_efs[j])
            if conv:
                ef_idx, is_novel = get_or_register(M_f)
                table[i][j] = ef_idx  # base EF index or N+k for novel

    # Count stats
    n_conv = sum(1 for i in range(N) for j in range(N) if table[i][j] >= 0)
    n_known = sum(1 for i in range(N) for j in range(N) if 0 <= table[i][j] < N)
    print(f"  Convergence: {n_conv}/{N*N} ({n_conv/(N*N)*100:.1f}%)")
    print(f"  Known EF results: {n_known}, Novel: {novel_count}")
    print()

    # ── Part 1: Family structure ──────────────────────────────────────────────
    print("=== Part 1: Family structure ===", flush=True)

    # Assign EF families: M* and -M* are related (neg), M* and M*^T related (tr)
    # Use union-find to group families
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Detect family relations
    family_rel = {}  # (i, j) -> 'neg' or 'tr'
    for i in range(N):
        for j in range(i + 1, N):
            neg_i = neg(base_efs[i])
            tr_i = transpose(base_efs[i])
            cos_neg = abs(mcosine(neg_i, base_efs[j]))
            cos_tr = abs(mcosine(tr_i, base_efs[j]))
            # Check if they're "same" (cos > FAMILY_COS means related)
            # But we need exact: cos(neg(M_i), M_j) ≈ 1 means M_j ≈ -M_i (same family via neg)
            # cos(tr(M_i), M_j) ≈ 1 means M_j ≈ M_i^T
            c_neg = mcosine(neg_i, base_efs[j])
            c_tr = mcosine(tr_i, base_efs[j])
            if abs(c_neg) > FAMILY_COS:
                union(i, j)
                family_rel[(i, j)] = f'neg(cos={c_neg:.3f})'
            elif abs(c_tr) > FAMILY_COS:
                union(i, j)
                family_rel[(i, j)] = f'tr(cos={c_tr:.3f})'

    # Group families
    families = collections.defaultdict(list)
    for i in range(N):
        families[find(i)].append(i)
    n_families = len(families)
    print(f"  Families (related by neg/tr): {n_families} families from {N} EFs")
    fam_sizes = sorted([len(v) for v in families.values()], reverse=True)
    print(f"  Family sizes: {fam_sizes}")
    for rep, members in sorted(families.items()):
        if len(members) > 1:
            print(f"    Family {rep}: {members}")
    print()

    # Build family index map: EF_i -> family_id
    ef_to_fam = {i: find(i) for i in range(N)}
    fam_ids = sorted(set(ef_to_fam.values()))
    fam_name = {f: chr(65 + idx) for idx, f in enumerate(fam_ids)}  # A, B, C, ...

    # Check: does composition respect families?
    # M_a o M_b = M_c => does (-M_a) o M_b = -M_c? and M_a^T o M_b = M_c^T?
    print("  Negation distributes? (M_a o M_b = M_c => (-M_a) o M_b = -M_c)")
    neg_tests = 0; neg_ok = 0; neg_tested = 0
    for i in range(N):
        for j in range(N):
            if 0 <= table[i][j] < N:
                neg_tests += 1
                # Find which EF is -M_i
                neg_i = neg(base_efs[i])
                neg_i_idx, _ = find_nearest(neg_i, base_efs, KNOWN_COS)
                if neg_i_idx < 0:
                    continue  # -M_i not in base set
                # Compose neg_i with M_j
                M_f, conv = compose_pair(base_efs[neg_i_idx], base_efs[j])
                if not conv:
                    continue
                # Expected: -M_c where M_c = base_efs[table[i][j]]
                neg_c = neg(base_efs[table[i][j]])
                neg_c_idx, _ = find_nearest(neg_c, base_efs, KNOWN_COS)
                result_idx, _ = find_nearest(M_f, base_efs, KNOWN_COS)
                neg_tested += 1
                if result_idx >= 0 and result_idx == neg_c_idx:
                    neg_ok += 1
    if neg_tested > 0:
        print(f"    Tested: {neg_tested}, passes: {neg_ok} ({neg_ok/neg_tested*100:.1f}%)")
    else:
        print(f"    Not enough data to test (neg counterparts not in base set)")
    print()

    # Family composition table (family-level)
    print("  Family-level composition table:")
    fam_list = sorted(set(ef_to_fam.values()))
    fam_table = {}  # (fa, fb) -> set of result family IDs
    for i in range(N):
        for j in range(N):
            if 0 <= table[i][j] < N:
                fa = ef_to_fam[i]
                fb = ef_to_fam[j]
                fc = ef_to_fam[table[i][j]]
                key = (fa, fb)
                if key not in fam_table:
                    fam_table[key] = set()
                fam_table[key].add(fc)

    # Print family table
    header = "  " + " " * 4 + "".join(f" {fam_name[f]:>3}" for f in fam_list)
    print(header)
    for fa in fam_list:
        row = f"  {fam_name[fa]:>3}|"
        for fb in fam_list:
            key = (fa, fb)
            if key in fam_table:
                results = fam_table[key]
                if len(results) == 1:
                    row += f" {fam_name[list(results)[0]]:>3}"
                else:
                    row += f" {'+':>3}"  # multiple results
            else:
                row += f" {'?':>3}"
        print(row)
    consistent_fam = all(len(v) == 1 for v in fam_table.values())
    print(f"  Family composition consistent (single result per pair): {consistent_fam}")
    print()

    # ── Part 2: Closed sub-algebras ───────────────────────────────────────────
    print("=== Part 2: Closed sub-algebras (within base EFs) ===", flush=True)

    # Find largest S ⊆ {0..N-1} where: for all (i,j) in S×S, if 0 <= table[i][j] < N: table[i][j] in S
    # Greedy: start with all N, iteratively remove EFs that cause violations
    # A violation: EF k ∈ S such that exists (i,j) in S×S with table[i][j] = k AND k ∉ S (impossible by def)
    # Actually: violation is (i,j) in S×S where 0<=table[i][j]<N but table[i][j] ∉ S
    # Need to remove source EFs that produce external results

    def find_closed_subset(candidates):
        S = set(candidates)
        changed = True
        while changed:
            changed = False
            to_remove = set()
            for i in list(S):
                for j in list(S):
                    r = table[i][j]
                    if 0 <= r < N and r not in S:
                        # Composition of i and j escapes S → remove i or j
                        # Remove i (row producer) greedily
                        to_remove.add(i)
            if to_remove:
                S -= to_remove
                changed = True
        return S

    closed_full = find_closed_subset(range(N))
    print(f"  Greedy closure from all {N} EFs: {len(closed_full)} EFs → {sorted(closed_full)}")

    # Also try: find if the DIAGONAL-only set (all idempotent singletons) is trivially closed
    # Try each EF as a seed for a closed set
    best_closed = set()
    best_size = 0
    for seed_ef in range(N):
        # Start with just this EF, expand by adding any results
        S = {seed_ef}
        changed = True
        while changed:
            changed = False
            for i in list(S):
                for j in list(S):
                    r = table[i][j]
                    if 0 <= r < N and r not in S:
                        S.add(r)
                        changed = True
        # Now check if S is truly closed (no escapes)
        closed_S = find_closed_subset(S)
        if len(closed_S) > best_size and len(closed_S) >= 3:
            best_size = len(closed_S)
            best_closed = closed_S

    if best_size >= 3:
        print(f"  Largest closed sub-algebra found: {best_size} EFs → {sorted(best_closed)}")
        print(f"  Sub-table:")
        members = sorted(best_closed)
        header2 = "  " + " " * 4 + "".join(f" {j:>3}" for j in members)
        print(header2)
        for i in members:
            row = f"  {i:>3}|"
            for j in members:
                entry = table[i][j]
                if entry >= 0:
                    row += f" {entry:>3}" if entry < N else f" {'N':>3}"
                else:
                    row += f" {'?':>3}"
            print(row)
    else:
        print(f"  No closed sub-algebra with ≥3 base EFs found.")
    print()

    # ── Part 3: Orbit structure ───────────────────────────────────────────────
    print("=== Part 3: One-step orbit sizes ===", flush=True)

    orbit_sizes = []
    for i in range(N):
        # Outgoing orbit: all distinct EFs reachable from EF_i in one step
        reached = set()
        for j in range(N):
            r = table[i][j]
            if r >= 0:
                reached.add(r)
        orbit_sizes.append((i, len(reached), reached))

    orbit_sizes_sorted = sorted(orbit_sizes, key=lambda x: -x[1])
    print(f"  {'EF':>4}  {'orbit':>6}  distinct reachable")
    print(f"  {'--':>4}  {'------':>6}  ------------------")
    for i, sz, reached in orbit_sizes_sorted:
        known_in = sorted(r for r in reached if r < N)
        novel_in = sum(1 for r in reached if r >= N)
        print(f"  {i:>4}  {sz:>6}  known={known_in}  novel={novel_in}")

    hub_ef, hub_size, _ = orbit_sizes_sorted[0]
    min_ef, min_size, _ = orbit_sizes_sorted[-1]
    print()
    print(f"  Hub (most connected): EF_{hub_ef} reaches {hub_size} EFs in one step")
    print(f"  Leaf (least connected): EF_{min_ef} reaches {min_size} EFs in one step")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  Families: {n_families} families (sizes: {fam_sizes})")
    print(f"  Family composition consistent: {consistent_fam}")
    if neg_tested > 0:
        print(f"  Negation distributes: {neg_ok}/{neg_tested} ({neg_ok/neg_tested*100:.1f}%)")
    print(f"  Largest closed sub-algebra: {best_size} EFs")
    print(f"  Hub EF: EF_{hub_ef} (orbit={hub_size})")
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
