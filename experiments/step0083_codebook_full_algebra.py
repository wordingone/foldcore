#!/usr/bin/env python3
"""
Step 83 -- Full composition algebra + generators.

Spec: map the full eigenform algebra.

Part 1: 31x31 composition table. Track novel eigenforms (cos < 0.95 to known = novel).
Part 2: Generator closure — start with EF_0, compose until no new EFs appear.
Part 3: Consistency check — 5 known pairs x 10 noisy runs each.
"""
import sys, random, math, time
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 4
ALPHA = 1.2
BETA = 0.8
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
FIND_STEPS = 1000     # for finding eigenforms (k=4 needs ~1000 steps)
COMPOSE_STEPS = 1000  # for composition convergence
KNOWN_COS = 0.95      # cosine threshold to call a result "known" (Spec: 0.95)
BASIN_COS = 0.99      # same-basin for finding distinct EFs

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


def add_noise(M, eps, rng):
    return [[M[i][j] + rng.gauss(0, eps) for j in range(K)] for i in range(K)]


def find_nearest(M, ef_list, threshold=KNOWN_COS):
    """Return (idx, cos) of nearest in ef_list, or (-1, best_cos) if none exceed threshold."""
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
    """Returns (result_mat_or_None, converged)."""
    C = psi(Mi, Mj)
    C_f, conv = converge(C, COMPOSE_STEPS)
    return (C_f if conv else None), conv


def main():
    t0 = time.time()
    print("Step 83 -- Full eigenform algebra + generators", flush=True)
    print(f"K={K}, target={N_TARGET_EFS} EFs, KNOWN_COS={KNOWN_COS}, COMPOSE_STEPS={COMPOSE_STEPS}")
    print()

    # ── Find 31 eigenforms ────────────────────────────────────────────────────
    print(f"Finding {N_TARGET_EFS} distinct eigenforms (max 8000 attempts)...", flush=True)
    base_efs = find_eigenforms(N_TARGET_EFS)
    N = len(base_efs)
    print(f"  Found {N} base eigenforms in scan")
    if N < 2:
        print("  Not enough eigenforms. Aborting.")
        return
    print()

    # Combined EF list: base_efs + novel ones discovered during composition
    all_efs = [copy_mat(ef) for ef in base_efs]   # all known EFs (base + novel)
    novel_efs = []   # only the novel ones
    novel_start_idx = N   # novel EFs start at this index in all_efs

    def get_or_register(M_f):
        """Find existing EF match or register as novel. Returns (label_str, ef_idx)."""
        idx, cos_val = find_nearest(M_f, all_efs, KNOWN_COS)
        if idx >= 0:
            return idx, cos_val, False  # known
        # Novel: register
        all_efs.append(copy_mat(M_f))
        novel_efs.append(copy_mat(M_f))
        new_idx = len(all_efs) - 1
        return new_idx, cos_val, True  # novel

    def ef_label(idx):
        if idx < novel_start_idx:
            return str(idx)
        return f"N{idx - novel_start_idx}"

    # ── Part 1: Full NxN composition table ───────────────────────────────────
    print(f"=== Part 1: {N}x{N} composition table ({N*N} pairs) ===", flush=True)

    table = [[-2] * N for _ in range(N)]  # -2 = not computed, -1 = noconv
    n_conv = 0
    n_known_hit = 0
    n_novel_hit = 0

    for i in range(N):
        for j in range(N):
            M_f, conv = compose_pair(base_efs[i], base_efs[j])
            if conv:
                n_conv += 1
                ef_idx, cos_val, is_novel = get_or_register(M_f)
                table[i][j] = ef_idx
                if is_novel:
                    n_novel_hit += 1
                else:
                    n_known_hit += 1
            else:
                table[i][j] = -1

    conv_rate = n_conv / (N * N) * 100
    print(f"  Convergence: {n_conv}/{N*N} ({conv_rate:.1f}%)")
    print(f"  Hit known EF:  {n_known_hit}")
    print(f"  Novel EFs generated: {len(novel_efs)} (IDs: N0..N{len(novel_efs)-1})")
    print()

    # Print full table (truncated if N large)
    print("  Composition table (row=i, col=j, entry=label or ? if noconv):")
    header = "  " + " " * 4 + "".join(f" {ef_label(j):>3}" for j in range(N))
    print(header)
    for i in range(N):
        row = f"  {ef_label(i):>3}|"
        for j in range(N):
            entry = table[i][j]
            row += f" {ef_label(entry) if entry >= 0 else '?':>3}"
        print(row)
    print()

    # ── Part 2: Generator closure ─────────────────────────────────────────────
    print("=== Part 2: Generator closure from EF_0 ===", flush=True)

    # Closure starting from EF_0
    reachable = {0}  # indices into base_efs + all_efs
    round_num = 0
    max_rounds = 5

    while round_num < max_rounds:
        round_num += 1
        new_found = set()
        reachable_list = sorted(reachable)
        for i in reachable_list:
            for j in range(N):  # compose with all base EFs
                M_i = all_efs[i]
                M_j = base_efs[j]
                M_f, conv = compose_pair(M_i, M_j)
                if conv:
                    ef_idx, _, is_novel = get_or_register(M_f)
                    if ef_idx not in reachable:
                        new_found.add(ef_idx)
        if not new_found:
            print(f"  Round {round_num}: no new EFs. Closure reached.")
            break
        reachable.update(new_found)
        novel_in_round = [ef_label(x) for x in new_found if x >= novel_start_idx]
        known_in_round = [ef_label(x) for x in new_found if x < novel_start_idx]
        print(f"  Round {round_num}: +{len(new_found)} new EFs (known: {known_in_round}, novel: {novel_in_round})")
    else:
        print(f"  Max rounds reached. Reachable set may not be fully closed.")

    print(f"  EF_0 generates {len(reachable)} EFs total (of {N} base + {len(novel_efs)} novel)")
    print(f"  Reachable set: {sorted(ef_label(x) for x in reachable)}")
    print()

    # ── Part 3: Consistency check ─────────────────────────────────────────────
    print("=== Part 3: Consistency check (5 known pairs, 10 noisy runs each) ===", flush=True)

    # Find 5 pairs that produced known eigenforms
    known_pairs = []
    for i in range(N):
        for j in range(N):
            entry = table[i][j]
            if entry >= 0 and entry < novel_start_idx and len(known_pairs) < 5:
                known_pairs.append((i, j, entry))

    if not known_pairs:
        print("  No known-result pairs found. Skipping.")
    else:
        rng_noise = random.Random(SEED + 200)
        for (i, j, expected) in known_pairs:
            results = []
            for run in range(10):
                Mi_noisy = add_noise(base_efs[i], 0.001, rng_noise)
                Mj_noisy = add_noise(base_efs[j], 0.001, rng_noise)
                M_f, conv = compose_pair(Mi_noisy, Mj_noisy)
                if conv:
                    ef_idx, _, _ = get_or_register(M_f)
                    results.append(ef_label(ef_idx))
                else:
                    results.append("?")
            consistent = len(set(results)) == 1
            print(f"  EF_{i} o EF_{j} -> expected {ef_label(expected)}: [{', '.join(results)}]  consistent={consistent}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=== Summary ===")
    print(f"  Base EFs found:       {N}")
    print(f"  Composition conv%:    {conv_rate:.1f}%")
    print(f"  Novel EFs generated:  {len(novel_efs)}")
    print(f"  EF_0 generates:       {len(reachable)} (rounds={round_num})")
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
