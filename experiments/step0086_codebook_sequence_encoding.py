#!/usr/bin/env python3
"""
Step 86 -- Sequence encoding test.

Spec: can eigenform composition encode ORDER?
Vectors can't distinguish [A,B] from [B,A]. Can composition?

Part 1: 6 EFs from different families, 120 ordered permutations P(6,3)
  - For [x,y,z]: compute (x o y) o z
  - How many distinct results from 120 ordered perms vs 20 unordered combos?

Part 2: 3 EFs, 81 sequences of length 4 (3^4)
  - ((x1 o x2) o x3) o x4
  - How many distinct results?
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
COMPOSE_STEPS = 1000   # keep fast per spec "under 5 min"
KNOWN_COS = 0.95
BASIN_COS = 0.99
FAMILY_COS = 0.95

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


def copy_mat(M):
    return [[M[i][j] for j in range(K)] for i in range(K)]


def neg(M):
    return [[-M[i][j] for j in range(K)] for i in range(K)]


def transpose(M):
    return [[M[j][i] for j in range(K)] for i in range(K)]


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


def detect_families(base_efs):
    N = len(base_efs)
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

    for i in range(N):
        for j in range(i + 1, N):
            c_neg = mcosine(neg(base_efs[i]), base_efs[j])
            c_tr = mcosine(transpose(base_efs[i]), base_efs[j])
            if abs(c_neg) > FAMILY_COS or abs(c_tr) > FAMILY_COS:
                union(i, j)

    families = collections.defaultdict(list)
    for i in range(N):
        families[find(i)].append(i)
    ef_to_fam = {i: find(i) for i in range(N)}
    return families, ef_to_fam


def ef_fingerprint(M, all_known):
    """Get a stable identity for M: index into all_known or register new."""
    idx, cos_val = find_nearest(M, all_known, KNOWN_COS)
    if idx >= 0:
        return idx
    all_known.append(copy_mat(M))
    return len(all_known) - 1


def main():
    t0 = time.time()
    print("Step 86 -- Sequence encoding test", flush=True)
    print()

    # ── Find base eigenforms ──────────────────────────────────────────────────
    print("Finding base eigenforms...", flush=True)
    base_efs = find_eigenforms(N_TARGET_EFS)
    N = len(base_efs)
    print(f"  Found {N} base eigenforms")

    # ── Detect families, pick 6 from different families ───────────────────────
    families, ef_to_fam = detect_families(base_efs)
    fam_ids_sorted = sorted(families.keys())

    # For each family, pick first member. Then select 6 families with largest size
    # (larger families = more EFs tested in composition, better quality indicator)
    # Prefer pair-families (size 2) over singletons
    fam_by_size = sorted(fam_ids_sorted, key=lambda f: -len(families[f]))
    selected_fams = fam_by_size[:6]
    ef_pool = [sorted(families[f])[0] for f in selected_fams]  # one rep per family

    print(f"\n  Selected 6 EFs from different families:")
    for i, (ef_idx, fam_id) in enumerate(zip(ef_pool, selected_fams)):
        print(f"    Slot {i}: EF_{ef_idx} (family_id={fam_id}, size={len(families[fam_id])})")
    print()

    # Check pairwise convergence for selected 6
    print("  Pairwise convergence check (6x6)...", flush=True)
    pair_conv = [[False]*6 for _ in range(6)]
    for i in range(6):
        for j in range(6):
            if i == j:
                pair_conv[i][j] = True
                continue
            _, conv = compose_pair(base_efs[ef_pool[i]], base_efs[ef_pool[j]])
            pair_conv[i][j] = conv

    n_pair_conv = sum(pair_conv[i][j] for i in range(6) for j in range(6))
    print(f"  Pairwise convergence: {n_pair_conv}/36 ({n_pair_conv/36*100:.1f}%)")
    print()

    # Registry for all EFs encountered (for fingerprinting)
    all_known = [copy_mat(ef) for ef in base_efs]

    # ── Part 1: P(6,3) = 120 ordered permutations of 3-from-6 ─────────────────
    print("=== Part 1: Sequence discrimination (P(6,3) = 120 permutations) ===", flush=True)

    ordered_results = {}   # (i,j,k) -> fingerprint or None
    unordered_results = {} # frozenset(i,j,k) -> set of fingerprints

    ef_labels = [f"EF{ef_pool[slot]}" for slot in range(6)]

    n_perm_total = 0
    n_perm_conv = 0

    for perm in itertools.permutations(range(6), 3):
        i, j, k = perm
        n_perm_total += 1
        Mi = base_efs[ef_pool[i]]
        Mj = base_efs[ef_pool[j]]
        Mk = base_efs[ef_pool[k]]

        # Step 1: x o y
        M_ij, conv1 = compose_pair(Mi, Mj)
        if not conv1:
            ordered_results[perm] = None
            continue

        # Step 2: (x o y) o z
        M_ijk, conv2 = compose_pair(M_ij, Mk)
        if not conv2:
            ordered_results[perm] = None
            continue

        n_perm_conv += 1
        fp = ef_fingerprint(M_ijk, all_known)
        ordered_results[perm] = fp

        # Track unordered
        combo = frozenset(perm)
        if combo not in unordered_results:
            unordered_results[combo] = set()
        unordered_results[combo].add(fp)

    # Stats for ordered
    conv_fps = [fp for fp in ordered_results.values() if fp is not None]
    distinct_ordered = len(set(conv_fps))
    fp_counts = collections.Counter(conv_fps)
    most_common_fp, most_common_count = fp_counts.most_common(1)[0] if fp_counts else (None, 0)
    collision_rate = 1.0 - distinct_ordered / n_perm_conv if n_perm_conv > 0 else 0.0

    print(f"  Ordered permutations: {n_perm_total}")
    print(f"  Convergence: {n_perm_conv}/{n_perm_total} ({n_perm_conv/n_perm_total*100:.1f}%)")
    print(f"  Distinct results (ordered): {distinct_ordered}")
    print(f"  Most common result: EF_{most_common_fp} appears {most_common_count} times")
    print(f"  Collision rate: {collision_rate:.1%} (1 - distinct/converged)")
    print()

    # Stats for unordered (C(6,3) = 20 combinations)
    unordered_combos = list(itertools.combinations(range(6), 3))
    distinct_unordered = set()
    for combo in unordered_combos:
        fs = frozenset(combo)
        if fs in unordered_results:
            for fp in unordered_results[fs]:
                distinct_unordered.add(fp)
    n_unord_combos = len([c for c in unordered_combos if frozenset(c) in unordered_results])

    print(f"  Unordered combinations: C(6,3)=20 (with results for {n_unord_combos})")
    print(f"  Distinct results (unordered): {len(distinct_unordered)}")
    print()

    # Order-sensitivity: for each unordered combo, do different orderings give different results?
    print("  Order-sensitivity per combination (does permutation matter?):")
    order_sensitive = 0
    order_insensitive = 0
    for combo in unordered_combos:
        fs = frozenset(combo)
        if fs not in unordered_results:
            continue
        fps_for_combo = unordered_results[fs]
        if len(fps_for_combo) > 1:
            order_sensitive += 1
            print(f"    {tuple(ef_labels[x] for x in sorted(fs))}: {len(fps_for_combo)} distinct outputs (ORDER MATTERS)")
        else:
            order_insensitive += 1
            print(f"    {tuple(ef_labels[x] for x in sorted(fs))}: 1 output (order-insensitive)")

    total_tested = order_sensitive + order_insensitive
    print()
    if total_tested > 0:
        print(f"  Order-sensitive combos: {order_sensitive}/{total_tested} ({order_sensitive/total_tested*100:.1f}%)")
    print()

    # Verdict: does ordered > unordered?
    print(f"  VERDICT:")
    print(f"    Ordered   perms: {distinct_ordered} distinct results from {n_perm_conv} converged sequences")
    print(f"    Unordered combos: {len(distinct_unordered)} distinct results from {n_unord_combos} converged combos")
    if distinct_ordered > len(distinct_unordered):
        ratio = distinct_ordered / max(len(distinct_unordered), 1)
        print(f"    -> ORDER IS ENCODED: {ratio:.1f}x more distinct results from ordered sequences")
    elif distinct_ordered == len(distinct_unordered):
        print(f"    -> ORDER NOT ENCODED: same number of distinct results regardless of order")
    else:
        print(f"    -> UNEXPECTED: fewer ordered results than unordered (check data)")
    print()

    # ── Part 2: Length-4 sequences, 3^4 = 81 ─────────────────────────────────
    print("=== Part 2: Length-4 sequences (3^4 = 81) ===", flush=True)

    # Pick 3 EFs from selected 6 (first 3)
    ef3 = ef_pool[:3]
    print(f"  Using EFs: {[f'EF_{x}' for x in ef3]}")
    print(f"  Sequences: 3^4 = 81, left-to-right: ((x1 o x2) o x3) o x4", flush=True)
    print()

    seq4_results = {}
    n_seq4_conv = 0
    n_seq4_total = 0

    for seq in itertools.product(range(3), repeat=4):
        n_seq4_total += 1
        M_list = [base_efs[ef3[s]] for s in seq]

        # ((x1 o x2) o x3) o x4
        M_cur, conv = compose_pair(M_list[0], M_list[1])
        if not conv:
            seq4_results[seq] = None
            continue

        ok = True
        for step in range(2, 4):
            M_cur, conv = compose_pair(M_cur, M_list[step])
            if not conv:
                ok = False
                break

        if ok and M_cur is not None:
            n_seq4_conv += 1
            fp = ef_fingerprint(M_cur, all_known)
            seq4_results[seq] = fp
        else:
            seq4_results[seq] = None

    conv4_fps = [fp for fp in seq4_results.values() if fp is not None]
    distinct4 = len(set(conv4_fps))
    fp4_counts = collections.Counter(conv4_fps)

    print(f"  Convergence: {n_seq4_conv}/{n_seq4_total} ({n_seq4_conv/n_seq4_total*100:.1f}%)")
    print(f"  Distinct results: {distinct4} from {n_seq4_conv} converged sequences")
    if fp4_counts:
        top5 = fp4_counts.most_common(5)
        print(f"  Top result frequencies: {[(f'EF_{fp}', cnt) for fp, cnt in top5]}")
    print()

    # Compare: if we had used unordered bags of 4 (multisets)
    # Unordered = {0,0,0,0}, {0,0,0,1}, ... = C(3+4-1, 4) = C(6,4) = 15 multisets
    unordered4 = collections.defaultdict(set)
    for seq, fp in seq4_results.items():
        if fp is not None:
            bag = tuple(sorted(seq))
            unordered4[bag].add(fp)
    distinct4_unord = len(set(fp for fps in unordered4.values() for fp in fps))
    print(f"  Distinct results (unordered bags): {distinct4_unord} from {len(unordered4)} bags with results")
    print()

    # Order-sensitive bags
    order_sensitive4 = sum(1 for fps in unordered4.values() if len(fps) > 1)
    print(f"  Order-sensitive bags (same elements, different result): {order_sensitive4}/{len(unordered4)}")
    print()

    if distinct4 > distinct4_unord:
        print(f"  LENGTH-4 VERDICT: ORDER ENCODED ({distinct4} ordered vs {distinct4_unord} unordered)")
    else:
        print(f"  LENGTH-4 VERDICT: order not distinguishable at this level")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  Part 1 (length-3, P(6,3)=120 perms):")
    print(f"    Convergence:         {n_perm_conv}/{n_perm_total} ({n_perm_conv/n_perm_total*100:.1f}%)")
    print(f"    Ordered distinct:    {distinct_ordered}")
    print(f"    Unordered distinct:  {len(distinct_unordered)}")
    print(f"    Order-sensitive combos: {order_sensitive}/{total_tested}" if total_tested > 0 else "    Order-sensitive: N/A")
    print(f"  Part 2 (length-4, 3^4=81 seqs):")
    print(f"    Convergence:         {n_seq4_conv}/{n_seq4_total} ({n_seq4_conv/n_seq4_total*100:.1f}%)")
    print(f"    Ordered distinct:    {distinct4}")
    print(f"    Unordered distinct:  {distinct4_unord}")
    print(f"    Order-sensitive bags: {order_sensitive4}/{len(unordered4)}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
