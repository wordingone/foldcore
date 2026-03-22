#!/usr/bin/env python3
"""
Step 85 -- Sub-algebra verification and second-generation composition.

[researcher] directive (mail 1204):
Part 1: Verify {C,D,E,J,M,N} sub-algebra (6x6 table, full coverage)
  - Pick 1 representative EF from each family C,D,E,J,M,N
  - All 36 pairwise compositions (6x6), COMPOSE_STEPS=5000
  - Check: converges? which family? commutative?
  - Test ALL associative triples

Part 2: Second-generation composition (novel EFs)
  - Take 10 novel EFs (N0-N9 from Step 83)
  - Compose each with itself (idempotent?)
  - Compose each with 5 random base EFs
  - Does it converge? Rate? 3rd-generation EFs or cycle back?

Is the algebra finite or unbounded?
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
COMPOSE_STEPS = 5000   # [researcher]: longer iterations for full coverage
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


def compose_pair(Mi, Mj, steps=COMPOSE_STEPS):
    C = psi(Mi, Mj)
    C_f, conv = converge(C, steps)
    return (C_f if conv else None), conv


def detect_families(base_efs):
    """Union-find family detection (same as Step 84)."""
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
            neg_i = neg(base_efs[i])
            tr_i = transpose(base_efs[i])
            c_neg = mcosine(neg_i, base_efs[j])
            c_tr = mcosine(tr_i, base_efs[j])
            if abs(c_neg) > FAMILY_COS:
                union(i, j)
            elif abs(c_tr) > FAMILY_COS:
                union(i, j)

    families = collections.defaultdict(list)
    for i in range(N):
        families[find(i)].append(i)

    fam_ids = sorted(set(find(i) for i in range(N)))
    fam_name = {f: chr(65 + idx) for idx, f in enumerate(fam_ids)}
    ef_to_fam = {i: find(i) for i in range(N)}
    return families, ef_to_fam, fam_name


def get_family_of(M_f, base_efs, ef_to_fam, fam_name):
    """Identify which family a converged result belongs to."""
    idx, cos_val = find_nearest(M_f, base_efs, KNOWN_COS)
    if idx >= 0:
        fam_id = ef_to_fam[idx]
        return fam_name[fam_id], idx, cos_val
    return None, -1, cos_val


def main():
    t0 = time.time()
    print("Step 85 -- Sub-algebra verification + second-generation composition", flush=True)
    print()

    # ── Reproduce base EFs (same seed as Steps 83/84) ────────────────────────
    print("Finding base eigenforms (same seed as Step 83/84)...", flush=True)
    base_efs = find_eigenforms(N_TARGET_EFS)
    N = len(base_efs)
    print(f"  Found {N} base eigenforms")
    print()

    # ── Detect families ───────────────────────────────────────────────────────
    print("Detecting families...", flush=True)
    families, ef_to_fam, fam_name = detect_families(base_efs)
    fam_ids_sorted = sorted(families.keys())

    print(f"  {len(families)} families:")
    for fid in fam_ids_sorted:
        members = families[fid]
        print(f"    Family {fam_name[fid]}: EFs {members}")
    print()

    # ── Identify {C,D,E,J,M,N} representatives ────────────────────────────────
    # Target families by name ([researcher] specified C,D,E,J,M,N)
    target_names = {'C', 'D', 'E', 'J', 'M', 'N'}
    target_reps = {}  # fam_name -> (fam_id, representative_ef_idx)
    for fid in fam_ids_sorted:
        name = fam_name[fid]
        if name in target_names:
            # Pick first member as representative
            rep_idx = sorted(families[fid])[0]
            target_reps[name] = (fid, rep_idx)

    if len(target_reps) < len(target_names):
        found = set(target_reps.keys())
        missing = target_names - found
        print(f"  WARNING: Could not find families {missing}. Available: {list(fam_name.values())}")
        # If we can't find 6, use the first 6 multi-member families as substitutes
        print("  Using first 6 available families as substitutes.")
        for fid in fam_ids_sorted:
            if len(target_reps) >= 6:
                break
            name = fam_name[fid]
            if name not in target_reps:
                rep_idx = sorted(families[fid])[0]
                target_reps[name] = (fid, rep_idx)
                print(f"    Substitute: Family {name} -> EF_{rep_idx}")

    ordered_names = sorted(target_reps.keys())
    print(f"  Sub-algebra representatives: {ordered_names}")
    for name in ordered_names:
        fid, rep_idx = target_reps[name]
        members = families[fid]
        print(f"    Family {name} (id={fid}, members={members}): representative = EF_{rep_idx}")
    print()

    # ── Part 1: 6x6 composition table ─────────────────────────────────────────
    print("=== Part 1: {C,D,E,J,M,N} Sub-algebra (6x6 table) ===", flush=True)
    print(f"  COMPOSE_STEPS={COMPOSE_STEPS}", flush=True)
    print()

    rep_efs = [(name, target_reps[name][0], target_reps[name][1]) for name in ordered_names]
    M_reps = [base_efs[r[2]] for r in rep_efs]

    n_reps = len(rep_efs)
    sub_table = [[-1] * n_reps for _ in range(n_reps)]     # result family name or None
    sub_table_idx = [[-1] * n_reps for _ in range(n_reps)] # result EF idx (-1=noconv, -2=novel)
    sub_conv = [[False] * n_reps for _ in range(n_reps)]

    # All novel EFs encountered (for family assignment)
    all_efs_ext = [copy_mat(ef) for ef in base_efs]
    novel_count = 0

    def get_or_register(M_f):
        nonlocal novel_count
        idx, cos_val = find_nearest(M_f, all_efs_ext, KNOWN_COS)
        if idx >= 0:
            return idx, False
        all_efs_ext.append(copy_mat(M_f))
        novel_count += 1
        return len(all_efs_ext) - 1, True

    print(f"  Computing {n_reps}x{n_reps} = {n_reps*n_reps} compositions...", flush=True)
    for i in range(n_reps):
        for j in range(n_reps):
            name_i, fid_i, idx_i = rep_efs[i]
            name_j, fid_j, idx_j = rep_efs[j]
            M_f, conv = compose_pair(M_reps[i], M_reps[j])
            sub_conv[i][j] = conv
            if conv:
                ef_idx, is_novel = get_or_register(M_f)
                sub_table_idx[i][j] = ef_idx
                if is_novel:
                    sub_table[i][j] = 'N'  # novel
                else:
                    # Find its family
                    if ef_idx < N:
                        fam_id = ef_to_fam[ef_idx]
                        sub_table[i][j] = fam_name[fam_id]
                    else:
                        sub_table[i][j] = 'N'
            print(f"    {name_i} o {name_j} -> {'?' if not conv else sub_table[i][j]} (EF_{sub_table_idx[i][j]})", flush=True)

    print()

    # Print 6x6 table
    print("  6x6 composition table (row o col = family of result):")
    header = "  " + " " * 4 + "".join(f"  {name:>3}" for name in ordered_names)
    print(header)
    for i in range(n_reps):
        row = f"  {ordered_names[i]:>3}|"
        for j in range(n_reps):
            entry = sub_table[i][j] if sub_conv[i][j] else '?'
            row += f"  {str(entry):>3}"
        print(row)
    print()

    # Convergence stats
    n_conv_sub = sum(sub_conv[i][j] for i in range(n_reps) for j in range(n_reps))
    n_novel_sub = sum(1 for i in range(n_reps) for j in range(n_reps)
                      if sub_conv[i][j] and sub_table[i][j] == 'N')
    print(f"  Convergence: {n_conv_sub}/{n_reps*n_reps} ({n_conv_sub/(n_reps*n_reps)*100:.1f}%)")
    print(f"  Novel results: {n_novel_sub}")
    print()

    # Commutativity check
    print("  Commutativity check (i o j == j o i):")
    comm_pairs = 0
    comm_total = 0
    non_comm = []
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            if sub_conv[i][j] and sub_conv[j][i]:
                comm_total += 1
                if sub_table[i][j] == sub_table[j][i]:
                    comm_pairs += 1
                else:
                    non_comm.append((ordered_names[i], ordered_names[j],
                                     sub_table[i][j], sub_table[j][i]))
    if comm_total > 0:
        print(f"    Commutative: {comm_pairs}/{comm_total} pairs ({comm_pairs/comm_total*100:.1f}%)")
    if non_comm:
        print(f"    Non-commutative pairs:")
        for a, b, ab, ba in non_comm:
            print(f"      {a} o {b} = {ab}, {b} o {a} = {ba}")
    print()

    # Closure check
    result_fams = set(sub_table[i][j] for i in range(n_reps) for j in range(n_reps)
                      if sub_conv[i][j] and sub_table[i][j] != 'N')
    closed = result_fams.issubset(set(ordered_names))
    print(f"  Closed under composition: {closed}")
    print(f"    Result families: {sorted(result_fams)}")
    print(f"    Expected: {sorted(ordered_names)}")
    print()

    # ALL associative triples
    print("  Associativity: testing ALL triples (a o b) o c vs a o (b o c)...", flush=True)
    n_assoc_ok = 0
    n_assoc_fail = 0
    n_assoc_skip = 0
    fail_examples = []

    for i in range(n_reps):
        for j in range(n_reps):
            for k_idx in range(n_reps):
                # (i o j) o k vs i o (j o k)
                ij = sub_table_idx[i][j]
                jk = sub_table_idx[j][k_idx]
                if ij < 0 or jk < 0:
                    n_assoc_skip += 1
                    continue

                # For (ij) o k: need to compose all_efs_ext[ij] with M_reps[k_idx]
                M_ij = all_efs_ext[ij]
                M_f_lhs, conv_lhs = compose_pair(M_ij, M_reps[k_idx])
                if not conv_lhs:
                    n_assoc_skip += 1
                    continue

                M_jk = all_efs_ext[jk]
                M_f_rhs, conv_rhs = compose_pair(M_reps[i], M_jk)
                if not conv_rhs:
                    n_assoc_skip += 1
                    continue

                # Compare results
                cos_check = abs(mcosine(M_f_lhs, M_f_rhs))
                if cos_check > KNOWN_COS:
                    n_assoc_ok += 1
                else:
                    n_assoc_fail += 1
                    if len(fail_examples) < 3:
                        fail_examples.append((ordered_names[i], ordered_names[j],
                                              ordered_names[k_idx], cos_check))

    n_assoc_total = n_assoc_ok + n_assoc_fail
    if n_assoc_total > 0:
        print(f"    Tested: {n_assoc_total} triples, skipped: {n_assoc_skip}")
        print(f"    Associative: {n_assoc_ok}/{n_assoc_total} ({n_assoc_ok/n_assoc_total*100:.1f}%)")
    else:
        print(f"    No testable triples (all skipped)")
    if fail_examples:
        print(f"    Failure examples (cos < {KNOWN_COS}):")
        for a, b, c_, cos in fail_examples:
            print(f"      ({a} o {b}) o {c_} vs {a} o ({b} o {c_}): cos={cos:.4f}")
    print()

    # ── Part 2: Second-generation novel EF composition ────────────────────────
    print("=== Part 2: Second-generation composition (N0-N9) ===", flush=True)
    print()

    # Reproduce novel EFs N0-N9 from Step 83 (same seed, same process)
    # Re-run Step 83 composition loop until we have 10 novel EFs registered
    print("  Reproducing N0-N9 from Step 83 composition...", flush=True)
    all_efs_step83 = [copy_mat(ef) for ef in base_efs]
    novel_efs_83 = []
    novel_start_idx = N

    def get_or_register_83(M_f):
        idx, cos_val = find_nearest(M_f, all_efs_step83, KNOWN_COS)
        if idx >= 0:
            return idx, False
        all_efs_step83.append(copy_mat(M_f))
        novel_efs_83.append(copy_mat(M_f))
        return len(all_efs_step83) - 1, True

    # Run enough of the 26x26 table to get N0-N9
    found_10 = False
    for i in range(N):
        if found_10:
            break
        for j in range(N):
            M_f, conv = compose_pair(base_efs[i], base_efs[j], steps=1000)
            if conv:
                _, is_novel = get_or_register_83(M_f)
            if len(novel_efs_83) >= 10:
                found_10 = True
                break

    n_novel_found = len(novel_efs_83)
    print(f"  Found {n_novel_found} novel EFs (N0..N{n_novel_found-1})")
    if n_novel_found < 10:
        print(f"  WARNING: only {n_novel_found} novel EFs found (needed 10). Using all available.")
    print()

    novel_10 = novel_efs_83[:10]
    # 5 random base EFs for composition (fixed seed)
    rng_p2 = random.Random(SEED + 500)
    base_sample_idxs = rng_p2.sample(range(N), min(5, N))
    print(f"  5 base EFs for cross-composition: {base_sample_idxs}")
    print()

    # Track all EFs encountered in Part 2
    all_efs_p2 = [copy_mat(ef) for ef in base_efs] + [copy_mat(ef) for ef in novel_efs_83]
    p2_novel_count = 0
    total_base_p2 = len(all_efs_p2)

    def classify_p2(M_f):
        nonlocal p2_novel_count
        # Check vs base
        idx_base, cos_base = find_nearest(M_f, base_efs, KNOWN_COS)
        if idx_base >= 0:
            return f"base({idx_base})", cos_base

        # Check vs novel_efs_83 (N0-N9+)
        for k_n, nef in enumerate(novel_efs_83):
            if abs(mcosine(M_f, nef)) > KNOWN_COS:
                return f"N{k_n}", abs(mcosine(M_f, nef))

        # Truly new
        p2_novel_count += 1
        return f"3rd-gen", 0.0

    print("  Self-composition (N_i o N_i = idempotent?):")
    n_self_conv = 0
    n_self_idemp = 0
    for k_n, nef in enumerate(novel_10):
        M_f, conv = compose_pair(nef, nef)
        if conv:
            n_self_conv += 1
            label, cos_val = classify_p2(M_f)
            is_same = abs(mcosine(M_f, nef)) > KNOWN_COS
            idemp_str = "IDEMPOTENT" if is_same else f"-> {label}"
            if is_same:
                n_self_idemp += 1
            print(f"    N{k_n} o N{k_n} -> {idemp_str} (cos_self={abs(mcosine(M_f, nef)):.4f})")
        else:
            print(f"    N{k_n} o N{k_n} -> NO CONVERGENCE")
    if n_self_conv > 0:
        print(f"  Self-composition: {n_self_conv}/{len(novel_10)} converge, {n_self_idemp} idempotent")
    print()

    print("  Cross-composition (N_i o base_j):")
    n_cross_conv = 0
    n_cross_total = 0
    gen_counts = {'base': 0, '1st-novel': 0, '3rd-gen': 0}
    results_matrix = []

    for k_n, nef in enumerate(novel_10):
        row_results = []
        for b_idx in base_sample_idxs:
            n_cross_total += 1
            M_f, conv = compose_pair(nef, base_efs[b_idx])
            if conv:
                n_cross_conv += 1
                label, cos_val = classify_p2(M_f)
                row_results.append(label)
                if label.startswith('base'):
                    gen_counts['base'] += 1
                elif label.startswith('N'):
                    gen_counts['1st-novel'] += 1
                else:
                    gen_counts['3rd-gen'] += 1
            else:
                row_results.append('?')
        results_matrix.append(row_results)
        print(f"    N{k_n} o {base_sample_idxs} -> {row_results}")

    print()
    if n_cross_total > 0:
        print(f"  Cross-composition convergence: {n_cross_conv}/{n_cross_total} ({n_cross_conv/n_cross_total*100:.1f}%)")
        print(f"  Result distribution:")
        print(f"    -> base EF:     {gen_counts['base']} ({gen_counts['base']/n_cross_conv*100:.1f}% of converged)" if n_cross_conv > 0 else "    -> base EF:     0")
        print(f"    -> 1st-novel:   {gen_counts['1st-novel']} ({gen_counts['1st-novel']/n_cross_conv*100:.1f}% of converged)" if n_cross_conv > 0 else "    -> 1st-novel:   0")
        print(f"    -> 3rd-gen new: {gen_counts['3rd-gen']} ({gen_counts['3rd-gen']/n_cross_conv*100:.1f}% of converged)" if n_cross_conv > 0 else "    -> 3rd-gen new: 0")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  Part 1 — {n_reps}x{n_reps} sub-algebra:")
    print(f"    Convergence: {n_conv_sub}/{n_reps*n_reps} ({n_conv_sub/(n_reps*n_reps)*100:.1f}%)")
    print(f"    Novel results: {n_novel_sub}")
    print(f"    Closed: {closed}")
    if comm_total > 0:
        print(f"    Commutative: {comm_pairs}/{comm_total} ({comm_pairs/comm_total*100:.1f}%)")
    if n_assoc_total > 0:
        print(f"    Associative: {n_assoc_ok}/{n_assoc_total} ({n_assoc_ok/n_assoc_total*100:.1f}%)")
    print(f"  Part 2 — Novel EF second-generation:")
    print(f"    Self-idempotent: {n_self_idemp}/{len(novel_10)}")
    print(f"    Cross-conv rate: {n_cross_conv/n_cross_total*100:.1f}%" if n_cross_total > 0 else "    Cross-conv: N/A")
    if n_cross_conv > 0:
        verdict = "FINITE (cycles back)" if gen_counts['3rd-gen'] == 0 else f"UNBOUNDED? ({gen_counts['3rd-gen']} 3rd-gen)"
        print(f"    Algebra verdict: {verdict}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
