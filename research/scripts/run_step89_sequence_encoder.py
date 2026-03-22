#!/usr/bin/env python3
"""
Step 89 -- Chaining + sequence encoding at alpha=1.1, beta=0.5.

[researcher] directive (mail 1212): decisive test of eigenform algebra as sequence encoder.

Part 1: Length-4 chains (24 permutations, same 10 EFs as Step 88)
Part 2: 3^3=27 sequences from 3 non-commutative EFs

At α=1.2: 0% chain convergence, commutative.
At α=1.1, β=0.5: if chains converge AND produce order-dependent results → sequence encoder.
"""
import sys, random, math, time, collections, itertools
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 4
# α=1.1, β=0.5 — new regime from Step 88
ALPHA = 1.1
BETA = 0.5
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
FIND_STEPS = 2000
COMPOSE_STEPS = 2000
BASIN_COS = 0.99
KNOWN_COS = 0.95

SEED = 42
EF_SEED = SEED + 100 + 1   # same as Step 88 rank=1 run


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


def find_eigenforms(n_needed, seed=EF_SEED, max_attempts=3000):
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


def ef_fingerprint(M, all_known):
    idx, cos_val = find_nearest(M, all_known, KNOWN_COS)
    if idx >= 0:
        return idx
    all_known.append(copy_mat(M))
    return len(all_known) - 1


def fp_label(fp, N_base):
    return f"EF_{fp}" if fp < N_base else f"N{fp - N_base}"


def main():
    t0 = time.time()
    print(f"Step 89 -- Sequence encoding at alpha={ALPHA}, beta={BETA}", flush=True)
    print(f"COMPOSE_STEPS={COMPOSE_STEPS}, K={K}", flush=True)
    print()

    # ── Reproduce 10 EFs from Step 88 ────────────────────────────────────────
    print("Finding 10 eigenforms (same seed as Step 88)...", flush=True)
    eigenforms = find_eigenforms(10)
    N = len(eigenforms)
    print(f"  Found {N} eigenforms")
    if N < 4:
        print("  Need at least 4. Aborting.")
        return
    print()

    all_known = [copy_mat(ef) for ef in eigenforms]

    # ── Compute full NxN table at 2000 steps ──────────────────────────────────
    print(f"  Building {N}x{N} composition table ({N*N} pairs)...", flush=True)
    table = [[-1] * N for _ in range(N)]
    conv_mat = [[False] * N for _ in range(N)]

    for i in range(N):
        for j in range(N):
            M_f, conv = compose_pair(eigenforms[i], eigenforms[j])
            conv_mat[i][j] = conv
            if conv:
                fp = ef_fingerprint(M_f, all_known)
                table[i][j] = fp

    n_conv = sum(conv_mat[i][j] for i in range(N) for j in range(N))
    print(f"  Convergence: {n_conv}/{N*N} ({n_conv/(N*N)*100:.1f}%)")

    # Find non-commutative pairs
    non_comm = []
    for i in range(N):
        for j in range(i + 1, N):
            if conv_mat[i][j] and conv_mat[j][i]:
                if table[i][j] != table[j][i]:
                    non_comm.append((i, j, table[i][j], table[j][i]))

    print(f"  Non-commutative pairs: {len(non_comm)}")
    for a, b, ab, ba in non_comm:
        print(f"    EF_{a} o EF_{b} = {fp_label(ab, N)}, EF_{b} o EF_{a} = {fp_label(ba, N)}")
    print()

    # Print composition table
    print(f"  Composition table:")
    header = "  " + " " * 5 + "".join(f" {j:>3}" for j in range(N))
    print(header)
    for i in range(N):
        row = f"  {i:>3}|"
        for j in range(N):
            if conv_mat[i][j]:
                fp = table[i][j]
                entry = str(fp) if fp < N else "N"
            else:
                entry = "?"
            row += f" {entry:>3}"
        print(row)
    print()

    # ── Part 1: Length-4 chains — find best 4-EF subset ──────────────────────
    print("=== Part 1: Length-4 chains (24 permutations) ===", flush=True)

    # Find the largest subset where all pairwise compositions converge
    best4 = None
    best4_score = -1
    for combo in itertools.combinations(range(N), 4):
        score = sum(1 for i in combo for j in combo if conv_mat[i][j])
        if score > best4_score:
            best4_score = score
            best4 = list(combo)

    print(f"  Best 4-EF subset: EFs {best4} ({best4_score}/16 pairwise converge)")
    ef4_mats = [eigenforms[idx] for idx in best4]
    labels4 = [f"EF_{idx}" for idx in best4]
    print(f"  Labels: {labels4}")
    print()

    # Run all 24 permutations
    print(f"  Computing 24 permutations, left-to-right (2000 steps each)...", flush=True)
    perm_results = {}
    n_chain_conv = 0

    for perm in itertools.permutations(range(4)):
        idxs = [best4[p] for p in perm]
        mats = [eigenforms[idx] for idx in idxs]
        names = [f"EF_{idx}" for idx in idxs]

        # ((m0 o m1) o m2) o m3
        M_cur, conv = compose_pair(mats[0], mats[1])
        ok = conv
        if ok:
            for step in range(2, 4):
                M_cur, conv = compose_pair(M_cur, mats[step])
                if not conv:
                    ok = False
                    break

        if ok and M_cur is not None:
            n_chain_conv += 1
            fp = ef_fingerprint(M_cur, all_known)
            perm_results[perm] = fp
            print(f"    ({' o '.join(names)}) -> {fp_label(fp, N)}")
        else:
            perm_results[perm] = None
            print(f"    ({' o '.join(names)}) -> NO CONV")

    print()
    conv_fps = [fp for fp in perm_results.values() if fp is not None]
    distinct_count = len(set(conv_fps))
    print(f"  Chain convergence: {n_chain_conv}/24 ({n_chain_conv/24*100:.1f}%)")
    print(f"  Distinct results: {distinct_count}")

    if n_chain_conv > 0:
        # Show equivalence classes
        fp_groups = collections.defaultdict(list)
        for perm, fp in perm_results.items():
            if fp is not None:
                fp_groups[fp].append(tuple(best4[p] for p in perm))

        print(f"  Result groups:")
        for fp, perms in sorted(fp_groups.items()):
            print(f"    {fp_label(fp, N)}: {len(perms)} permutations -> {perms}")

        # Check order sensitivity: pick 2 permutations that differ only in first 2 positions
        order_sensitive = False
        for perm1, perm2 in itertools.combinations(
                [p for p, fp in perm_results.items() if fp is not None], 2):
            if perm_results[perm1] != perm_results[perm2]:
                order_sensitive = True
                break

        print(f"\n  ORDER ENCODED: {distinct_count > 1}")
        if distinct_count > 1:
            print(f"  -> {distinct_count} distinct outputs from {n_chain_conv} converged chains")
    print()

    # ── Part 2: 3^3=27 sequences from non-commutative EFs ─────────────────────
    print("=== Part 2: 3^3=27 sequences from non-commutative EFs ===", flush=True)

    # Pick 3 EFs that have non-commutative relationships
    if len(non_comm) >= 1:
        # Use EFs involved in non-commutative pairs
        nc_ef_set = set()
        for a, b, _, _ in non_comm:
            nc_ef_set.add(a)
            nc_ef_set.add(b)
        nc_efs = sorted(nc_ef_set)[:3]  # take first 3
        if len(nc_efs) < 3:
            # Pad with other EFs
            for i in range(N):
                if i not in nc_ef_set:
                    nc_efs.append(i)
                if len(nc_efs) >= 3:
                    break
    else:
        # No non-commutative pairs — just use first 3
        nc_efs = [0, 1, 2]

    print(f"  Using EFs for sequence test: {nc_efs}")
    nc_mats = [eigenforms[idx] for idx in nc_efs]
    nc_labels = [f"EF_{idx}" for idx in nc_efs]
    print(f"  Labels: {nc_labels}")
    print()

    # Verify non-commutativity within this set
    print(f"  Pairwise check (commutativity within set):")
    for i in range(3):
        for j in range(i + 1, 3):
            if conv_mat[nc_efs[i]][nc_efs[j]] and conv_mat[nc_efs[j]][nc_efs[i]]:
                comm = table[nc_efs[i]][nc_efs[j]] == table[nc_efs[j]][nc_efs[i]]
                r_ij = fp_label(table[nc_efs[i]][nc_efs[j]], N) if conv_mat[nc_efs[i]][nc_efs[j]] else "?"
                r_ji = fp_label(table[nc_efs[j]][nc_efs[i]], N) if conv_mat[nc_efs[j]][nc_efs[i]] else "?"
                print(f"    {nc_labels[i]} o {nc_labels[j]} = {r_ij}, {nc_labels[j]} o {nc_labels[i]} = {r_ji}  {'(commutes)' if comm else '(NON-COMMUTATIVE)'}")
    print()

    # Build 3x3 sub-table for reference
    print(f"  3x3 sub-table:")
    header3 = "  " + " " * 6 + "".join(f" {l:>6}" for l in nc_labels)
    print(header3)
    for i in range(3):
        row = f"  {nc_labels[i]:>5}|"
        for j in range(3):
            ei, ej = nc_efs[i], nc_efs[j]
            if conv_mat[ei][ej]:
                entry = fp_label(table[ei][ej], N)
            else:
                entry = "?"
            row += f" {entry:>6}"
        print(row)
    print()

    # 3^3 = 27 length-3 sequences
    print(f"  Computing 3^3=27 length-3 sequences (left-to-right)...", flush=True)
    seq3_results = {}
    n_seq3_conv = 0

    for seq in itertools.product(range(3), repeat=3):
        mats_seq = [nc_mats[s] for s in seq]
        names_seq = [nc_labels[s] for s in seq]

        # (m0 o m1) o m2
        M_cur, conv = compose_pair(mats_seq[0], mats_seq[1])
        if conv:
            M_cur, conv = compose_pair(M_cur, mats_seq[2])

        if conv and M_cur is not None:
            n_seq3_conv += 1
            fp = ef_fingerprint(M_cur, all_known)
            seq3_results[seq] = fp
        else:
            seq3_results[seq] = None

    conv3_fps = [fp for fp in seq3_results.values() if fp is not None]
    distinct3 = len(set(conv3_fps))
    print(f"  Convergence: {n_seq3_conv}/27 ({n_seq3_conv/27*100:.1f}%)")
    print(f"  Distinct results: {distinct3} from {n_seq3_conv} converged sequences")

    # Show all results
    print()
    fp_groups3 = collections.defaultdict(list)
    for seq, fp in seq3_results.items():
        if fp is not None:
            fp_groups3[fp].append(tuple(nc_labels[s] for s in seq))

    print(f"  Results by output:")
    for fp in sorted(fp_groups3.keys()):
        print(f"    {fp_label(fp, N)}: {len(fp_groups3[fp])} sequences")
        for s in fp_groups3[fp][:4]:
            print(f"      {s}")
        if len(fp_groups3[fp]) > 4:
            print(f"      ... ({len(fp_groups3[fp])-4} more)")
    print()

    # Check: are reversed sequences [A,B,C] vs [C,B,A] always different?
    print(f"  Reversed-sequence test ([A,B,C] vs [C,B,A]):")
    reversed_diff = 0
    reversed_total = 0
    for seq in itertools.product(range(3), repeat=3):
        rev = tuple(reversed(seq))
        if seq < rev:  # avoid duplicates
            fp_fwd = seq3_results.get(seq)
            fp_rev = seq3_results.get(rev)
            if fp_fwd is not None and fp_rev is not None:
                reversed_total += 1
                if fp_fwd != fp_rev:
                    reversed_diff += 1
                    fwd_names = tuple(nc_labels[s] for s in seq)
                    rev_names = tuple(nc_labels[s] for s in rev)
                    print(f"    {fwd_names} -> {fp_label(fp_fwd, N)} | {rev_names} -> {fp_label(fp_rev, N)}  DIFFERENT")
                else:
                    fwd_names = tuple(nc_labels[s] for s in seq)
                    print(f"    {fwd_names} <-> reverse: SAME ({fp_label(fp_fwd, N)})")

    if reversed_total > 0:
        print(f"  Reversed pairs different: {reversed_diff}/{reversed_total} ({reversed_diff/reversed_total*100:.1f}%)")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  Parameters: alpha={ALPHA}, beta={BETA}, k={K}")
    print(f"  Pairwise convergence: {n_conv}/{N*N} ({n_conv/(N*N)*100:.1f}%)")
    print(f"  Non-commutative pairs: {len(non_comm)}")
    print()
    print(f"  Part 1 — Length-4 chains (24 permutations):")
    print(f"    Chain convergence: {n_chain_conv}/24 ({n_chain_conv/24*100:.1f}%)")
    if n_chain_conv > 0:
        print(f"    Distinct results: {distinct_count}")
        print(f"    Order encoded: {distinct_count > 1}")
    print()
    print(f"  Part 2 — 3^3=27 length-3 sequences:")
    print(f"    Convergence: {n_seq3_conv}/27 ({n_seq3_conv/27*100:.1f}%)")
    print(f"    Distinct results: {distinct3}")
    if reversed_total > 0:
        print(f"    Reversed pairs different: {reversed_diff}/{reversed_total}")
    print()
    if n_chain_conv > 0 and distinct_count > 1:
        print(f"  VERDICT: SEQUENCE ENCODER CONFIRMED")
        print(f"    Length-4: {distinct_count} distinct from {n_chain_conv} chains")
    elif n_chain_conv > 0 and distinct_count == 1:
        print(f"  VERDICT: chains converge but all to same result — order not encoded")
    else:
        print(f"  VERDICT: chains still fail to converge")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
