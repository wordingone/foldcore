#!/usr/bin/env python3
"""
Step 90 -- k=16 landscape scan, final scale test.

[researcher] directive (mail 1214):
alpha=1.1, beta=2.0 (beta/k = 2.0/16 = 0.125, same ratio as 0.5/4)

1. Generate 500 random 16x16 matrices, iterate Phi for 2000 steps
2. Report: convergence rate, n distinct EFs, frob distribution

If convergence > 5%:
3. Take 6 EFs, compute 6x6 composition table
4. Report: convergence, commutativity, non-trivial structure?

NOTE: k=16 matrices have 16^3=4096 ops per mmul vs 64 for k=4 (~64x slower).
Using 150 matrices for landscape scan to fit time budget.
"""
import sys, random, math, time
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 16
ALPHA = 1.1
BETA = 2.0
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
LANDSCAPE_STEPS = 2000
LANDSCAPE_N = 150    # reduced from 500 due to k=16 compute cost
COMPOSE_STEPS = 2000
BASIN_COS = 0.99
KNOWN_COS = 0.95

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


def ef_fingerprint(M, all_known):
    idx, cos_val = find_nearest(M, all_known, KNOWN_COS)
    if idx >= 0:
        return idx
    all_known.append(copy_mat(M))
    return len(all_known) - 1


def fp_label(fp, N_base):
    return f"EF_{fp}" if fp < N_base else f"N{fp-N_base}"


def main():
    t0 = time.time()
    print(f"Step 90 -- k=16 landscape scan", flush=True)
    print(f"K={K}, alpha={ALPHA}, beta={BETA}, beta/K={BETA/K:.4f}", flush=True)
    print(f"LANDSCAPE_STEPS={LANDSCAPE_STEPS}, N={LANDSCAPE_N} (reduced from 500 for k=16 compute cost)", flush=True)
    print(f"k=4 reference: alpha=1.1, beta=0.5, beta/K=0.125 -- same ratio", flush=True)
    print()

    # ── Landscape scan ────────────────────────────────────────────────────────
    print(f"=== Part 1: Landscape scan ({LANDSCAPE_N} matrices) ===", flush=True)
    rng = random.Random(SEED + 1)
    eigenforms = []
    frob_vals = []
    n_conv = 0
    t_scan = time.time()

    for mat_idx in range(LANDSCAPE_N):
        M0 = [[rng.uniform(-2, 2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge(M0, LANDSCAPE_STEPS)
        if conv:
            n_conv += 1
            frob_vals.append(frob(M_f))
            if all(not same_basin(M_f, ef) for ef in eigenforms):
                eigenforms.append(copy_mat(M_f))
        if mat_idx == 9:
            elapsed_10 = time.time() - t_scan
            print(f"  First 10 matrices done in {elapsed_10:.1f}s → projected: {elapsed_10/10*LANDSCAPE_N:.0f}s for all {LANDSCAPE_N}", flush=True)

    scan_elapsed = time.time() - t_scan
    conv_rate = n_conv / LANDSCAPE_N
    n_efs = len(eigenforms)
    mean_frob = sum(frob_vals) / len(frob_vals) if frob_vals else 0.0

    print(f"  Convergence: {n_conv}/{LANDSCAPE_N} ({conv_rate*100:.1f}%)")
    print(f"  Distinct eigenforms: {n_efs}")
    print(f"  Mean frob: {mean_frob:.4f}")
    if frob_vals:
        frob_sorted = sorted(frob_vals)
        print(f"  Frob range: [{frob_sorted[0]:.4f}, {frob_sorted[-1]:.4f}]")
        print(f"  Frob median: {frob_sorted[len(frob_sorted)//2]:.4f}")
    print(f"  Scan time: {scan_elapsed:.1f}s")
    print()

    if conv_rate < 0.05:
        print(f"  Convergence rate {conv_rate*100:.1f}% < 5%. Skipping composition table.")
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        return

    # ── Composition table (6 EFs) ─────────────────────────────────────────────
    print(f"=== Part 2: 6x6 composition table ===", flush=True)

    # Find 6 eigenforms
    n_ef_target = min(6, n_efs)
    ef6 = eigenforms[:n_ef_target]
    N6 = len(ef6)
    print(f"  Using {N6} eigenforms for composition table", flush=True)

    if N6 < 2:
        print(f"  Not enough eigenforms for composition. Done.")
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        return

    all_known = [copy_mat(ef) for ef in ef6]
    novel_count = 0

    def get_or_register(M_f):
        nonlocal novel_count
        idx, cos_val = find_nearest(M_f, all_known, KNOWN_COS)
        if idx >= 0:
            return idx, False
        all_known.append(copy_mat(M_f))
        novel_count += 1
        return len(all_known) - 1, True

    table = [[-1] * N6 for _ in range(N6)]
    conv_mat = [[False] * N6 for _ in range(N6)]
    t_comp = time.time()

    for i in range(N6):
        for j in range(N6):
            C = psi(ef6[i], ef6[j])
            C_f, conv = converge(C, COMPOSE_STEPS)
            conv_mat[i][j] = conv
            if conv:
                fp, is_novel = get_or_register(C_f)
                table[i][j] = fp

    comp_elapsed = time.time() - t_comp
    n_conv_comp = sum(conv_mat[i][j] for i in range(N6) for j in range(N6))
    n_known_comp = n_conv_comp - novel_count

    print(f"  Composition convergence: {n_conv_comp}/{N6*N6} ({n_conv_comp/(N6*N6)*100:.1f}%)")
    print(f"  Known EF results: {n_known_comp}, Novel: {novel_count}")
    print(f"  Composition time: {comp_elapsed:.1f}s")
    print()

    # Print table
    labels = [f"E{i}" for i in range(N6)]
    print(f"  Composition table:")
    header = "  " + " " * 4 + "".join(f" {l:>4}" for l in labels)
    print(header)
    for i in range(N6):
        row = f"  {labels[i]:>3}|"
        for j in range(N6):
            if conv_mat[i][j]:
                fp = table[i][j]
                entry = str(fp) if fp < N6 else "N"
            else:
                entry = "?"
            row += f" {entry:>4}"
        print(row)
    print()

    # Commutativity
    n_comm = 0
    n_comm_total = 0
    non_comm = []
    for i in range(N6):
        for j in range(i + 1, N6):
            if conv_mat[i][j] and conv_mat[j][i]:
                n_comm_total += 1
                if table[i][j] == table[j][i]:
                    n_comm += 1
                else:
                    non_comm.append((i, j, table[i][j], table[j][i]))

    if n_comm_total > 0:
        print(f"  Commutative: {n_comm}/{n_comm_total} ({n_comm/n_comm_total*100:.1f}%)")
    print(f"  Non-commutative pairs: {len(non_comm)}")
    for a, b, ab, ba in non_comm:
        print(f"    EF_{a} o EF_{b} = {fp_label(ab, N6)}, EF_{b} o EF_{a} = {fp_label(ba, N6)}")
    print()

    # Check for non-trivial structure (result != either input, both orders different)
    print(f"  Non-trivial compositions (result != either input AND both orders converge to DIFFERENT non-input):")
    n_nontrivial = 0
    for i in range(N6):
        for j in range(i + 1, N6):
            if not (conv_mat[i][j] and conv_mat[j][i]):
                continue
            r_ij = table[i][j]
            r_ji = table[j][i]
            # Both results must be non-input
            if r_ij == i or r_ij == j:
                continue
            if r_ji == i or r_ji == j:
                continue
            # And both must be valid (non-novel or novel, but not -1)
            if r_ij < 0 or r_ji < 0:
                continue
            n_nontrivial += 1
            print(f"    EF_{i} o EF_{j} = {fp_label(r_ij, N6)}, EF_{j} o EF_{i} = {fp_label(r_ji, N6)}")

    if n_nontrivial == 0:
        print(f"    None found — all compositions are projections (result = one of the inputs)")
    print()

    # Check absorbers
    print(f"  Absorber check (left/right zeros):")
    absorbers = []
    for k_idx in range(N6):
        left_zero = all(conv_mat[k_idx][j] and table[k_idx][j] == k_idx for j in range(N6))
        right_zero = all(conv_mat[i][k_idx] and table[i][k_idx] == k_idx for i in range(N6))
        if left_zero or right_zero:
            absorbers.append((k_idx, left_zero, right_zero))
            print(f"    EF_{k_idx}: left_zero={left_zero}, right_zero={right_zero}")
    if not absorbers:
        print(f"    No absorbers found")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  K={K}, alpha={ALPHA}, beta={BETA}, beta/K={BETA/K:.4f}")
    print(f"  Landscape ({LANDSCAPE_N} matrices): {conv_rate*100:.1f}% convergence, {n_efs} distinct EFs")
    print(f"  Composition ({N6}x{N6}): {n_conv_comp}/{N6*N6} ({n_conv_comp/(N6*N6)*100:.1f}%)")
    if n_comm_total > 0:
        print(f"  Commutative: {n_comm}/{n_comm_total} ({n_comm/n_comm_total*100:.1f}%)")
    print(f"  Non-commutative pairs: {len(non_comm)}")
    print(f"  Non-trivial compositions: {n_nontrivial}")
    print(f"  Absorbers: {len(absorbers)}")
    if n_nontrivial > 0:
        print(f"  VERDICT: RICHER than k=4 (non-trivial structure exists)")
    elif len(non_comm) > 0:
        print(f"  VERDICT: non-commutative but still projections (same trivial pattern)")
    else:
        print(f"  VERDICT: commutative projections — same trivial pattern as k=4")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
