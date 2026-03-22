#!/usr/bin/env python3
"""
Step 88 -- Parameter sweep for denser eigenform landscape.

Spec:
Part 1: Sweep 16 (alpha, beta) combinations.
  - 1000 random 4x4 matrices, 2000 steps, dt=0.03
  - Report: convergence rate, distinct EFs, mean frob

Part 2: Top 3 (alpha, beta) by convergence rate.
  - Find 10 eigenforms, compute 10x10 composition table (2000 steps)
  - Report: pairwise composition convergence, novel EFs, closed sub-algebras > 3
"""
import sys, random, math, time, collections
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K = 4
DT = 0.03
MAX_NORM = 3.0
CONVERGE_TOL = 0.001
LANDSCAPE_STEPS = 2000   # per spec
LANDSCAPE_N = 1000       # matrices to test
COMPOSE_STEPS = 2000
BASIN_COS = 0.99
KNOWN_COS = 0.95

SEED = 42

# Sweep grid
ALPHAS = [1.1, 1.5, 2.0, 3.0]
BETAS  = [0.5, 1.0, 2.0, 4.0]


def make_phi(alpha, beta):
    def phi(M):
        return mtanh(madd(mscale(M, alpha), mscale(mmul(M, M), beta / K)))
    return phi


def make_psi(alpha, beta):
    def psi(Mi, Mj):
        avg = mscale(madd(Mi, Mj), alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), beta / K)
        return mtanh(madd(avg, prod))
    return psi


def converge_with(phi, M, max_steps, tol=CONVERGE_TOL):
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


def scan_landscape(alpha, beta, n_samples, seed):
    """Test n_samples random matrices under Phi(alpha, beta). Return stats."""
    phi = make_phi(alpha, beta)
    rng = random.Random(seed)
    n_conv = 0
    frob_vals = []
    eigenforms = []  # distinct EFs found

    for _ in range(n_samples):
        M0 = [[rng.uniform(-2, 2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge_with(phi, M0, LANDSCAPE_STEPS)
        if conv:
            n_conv += 1
            frob_vals.append(frob(M_f))
            # Check if distinct
            if all(not same_basin(M_f, ef) for ef in eigenforms):
                eigenforms.append(copy_mat(M_f))

    conv_rate = n_conv / n_samples
    mean_frob = sum(frob_vals) / len(frob_vals) if frob_vals else 0.0
    return conv_rate, len(eigenforms), mean_frob


def find_eigenforms_with(phi, n_needed, seed, max_attempts=3000):
    rng = random.Random(seed)
    found = []
    for _ in range(max_attempts):
        M0 = [[rng.uniform(-2, 2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge_with(phi, M0, LANDSCAPE_STEPS)
        if conv and all(not same_basin(M_f, s) for s in found):
            found.append(M_f)
            if len(found) == n_needed:
                break
    return found


def compose_pair_with(phi, psi, Mi, Mj):
    C = psi(Mi, Mj)
    C_f, conv = converge_with(phi, C, COMPOSE_STEPS)
    return (C_f if conv else None), conv


def find_closed_subsets(N, table):
    """Find all closed sub-algebras of size >= 3."""
    closed_sets = []
    for size in range(3, N + 1):
        for combo in _combinations(range(N), size):
            S = set(combo)
            closed = True
            for i in S:
                for j in S:
                    r = table[i][j]
                    if r >= 0 and r not in S:
                        closed = False
                        break
                if not closed:
                    break
            if closed:
                # Check at least one non-trivial composition
                has_data = any(table[i][j] >= 0 for i in S for j in S if i != j)
                if has_data:
                    closed_sets.append(sorted(S))
    return closed_sets


def _combinations(pool, r):
    """itertools.combinations equivalent."""
    import itertools
    return list(itertools.combinations(pool, r))


def main():
    t0 = time.time()
    print("Step 88 -- Parameter sweep for denser eigenform landscape", flush=True)
    print(f"K={K}, dt={DT}, LANDSCAPE_STEPS={LANDSCAPE_STEPS}, N={LANDSCAPE_N}", flush=True)
    print()

    # ── Part 1: Landscape scan ─────────────────────────────────────────────────
    print("=== Part 1: Landscape scan (16 combinations) ===", flush=True)
    print(f"  {'alpha':>6} {'beta':>6}  {'conv%':>7}  {'n_EFs':>6}  {'frob':>7}  elapsed", flush=True)
    print(f"  {'------':>6} {'------':>6}  {'-------':>7}  {'------':>6}  {'-------':>7}  -------")

    results = []
    for alpha in ALPHAS:
        for beta in BETAS:
            ts = time.time()
            conv_rate, n_efs, mean_frob = scan_landscape(alpha, beta, LANDSCAPE_N, SEED + 1)
            elapsed_s = time.time() - ts
            results.append((alpha, beta, conv_rate, n_efs, mean_frob))
            print(f"  {alpha:>6.1f} {beta:>6.1f}  {conv_rate*100:>6.1f}%  {n_efs:>6}  {mean_frob:>7.4f}  {elapsed_s:.1f}s", flush=True)

    print()

    # Sort by composition-relevant metric: conv_rate * sqrt(n_efs) (want high conv + many EFs)
    results_sorted = sorted(results, key=lambda x: x[2] * math.sqrt(x[3]), reverse=True)
    top3 = results_sorted[:3]

    print("  Top 3 by conv_rate * sqrt(n_EFs):")
    for alpha, beta, conv_rate, n_efs, mean_frob in top3:
        score = conv_rate * math.sqrt(n_efs)
        print(f"    alpha={alpha}, beta={beta}: conv={conv_rate*100:.1f}%, n_EFs={n_efs}, score={score:.3f}")
    print()

    # Also show top 3 by raw convergence rate
    top3_conv = sorted(results, key=lambda x: x[2], reverse=True)[:3]
    print("  Top 3 by raw convergence rate:")
    for alpha, beta, conv_rate, n_efs, mean_frob in top3_conv:
        print(f"    alpha={alpha}, beta={beta}: conv={conv_rate*100:.1f}%, n_EFs={n_efs}")
    print()

    # Use top 3 by raw convergence rate for Part 2 (per the spec)
    top3_for_part2 = top3_conv

    # ── Part 2: Composition tables for top 3 ──────────────────────────────────
    print("=== Part 2: Composition tables for top 3 (10x10, 2000 steps) ===", flush=True)
    print()

    for rank, (alpha, beta, conv_rate_p1, n_efs_p1, _) in enumerate(top3_for_part2, 1):
        print(f"  [{rank}] alpha={alpha}, beta={beta} (landscape conv={conv_rate_p1*100:.1f}%)", flush=True)

        phi_fn = make_phi(alpha, beta)
        psi_fn = make_psi(alpha, beta)

        # Find 10 eigenforms
        eigenforms = find_eigenforms_with(phi_fn, 10, SEED + 100 + rank)
        N_ef = len(eigenforms)
        if N_ef < 2:
            print(f"    Only found {N_ef} eigenforms. Skipping composition.")
            print()
            continue
        print(f"    Found {N_ef} eigenforms")

        # Build NxN composition table
        all_efs = [copy_mat(ef) for ef in eigenforms]
        novel_count = 0

        def get_or_register(M_f):
            nonlocal novel_count
            idx, cos_val = find_nearest(M_f, all_efs, KNOWN_COS)
            if idx >= 0:
                return idx, False
            all_efs.append(copy_mat(M_f))
            novel_count += 1
            return len(all_efs) - 1, True

        table = [[-1] * N_ef for _ in range(N_ef)]
        n_conv_comp = 0
        for i in range(N_ef):
            for j in range(N_ef):
                M_f, conv = compose_pair_with(phi_fn, psi_fn, eigenforms[i], eigenforms[j])
                if conv:
                    n_conv_comp += 1
                    ef_idx, is_novel = get_or_register(M_f)
                    table[i][j] = ef_idx

        comp_conv_rate = n_conv_comp / (N_ef * N_ef)
        n_known_comp = n_conv_comp - novel_count
        print(f"    Composition convergence: {n_conv_comp}/{N_ef*N_ef} ({comp_conv_rate*100:.1f}%)")
        print(f"    Known EF results: {n_known_comp}, Novel: {novel_count}")

        # Commutativity
        n_comm = 0
        n_comm_total = 0
        non_comm = []
        for i in range(N_ef):
            for j in range(i + 1, N_ef):
                if table[i][j] >= 0 and table[j][i] >= 0:
                    n_comm_total += 1
                    if table[i][j] == table[j][i]:
                        n_comm += 1
                    else:
                        non_comm.append((i, j))
        if n_comm_total > 0:
            print(f"    Commutative: {n_comm}/{n_comm_total} ({n_comm/n_comm_total*100:.1f}%)")
        print(f"    Non-commutative pairs: {len(non_comm)}")
        if non_comm:
            for a, b in non_comm[:3]:
                print(f"      EF_{a} o EF_{b} = EF_{table[a][b]}, EF_{b} o EF_{a} = EF_{table[b][a]}")

        # Find closed sub-algebras within base EFs only
        print(f"    Searching for closed sub-algebras >= 3...", flush=True)
        base_table = [[-1] * N_ef for _ in range(N_ef)]
        for i in range(N_ef):
            for j in range(N_ef):
                r = table[i][j]
                if 0 <= r < N_ef:
                    base_table[i][j] = r

        import itertools
        best_closed = []
        best_size = 0
        for size in range(N_ef, 2, -1):
            if best_size >= size:
                break
            for combo in itertools.combinations(range(N_ef), size):
                S = set(combo)
                closed = True
                for i in S:
                    for j in S:
                        r = base_table[i][j]
                        if r >= 0 and r not in S:
                            closed = False
                            break
                    if not closed:
                        break
                if closed:
                    has_data = any(base_table[i][j] >= 0 for i in S for j in S if i != j)
                    if has_data:
                        best_closed.append(sorted(S))
                        if len(S) > best_size:
                            best_size = len(S)

        if best_closed:
            largest = max(best_closed, key=len)
            print(f"    Largest closed sub-algebra: {len(largest)} EFs -> {largest}")
            all_sizes = sorted(set(len(s) for s in best_closed), reverse=True)
            print(f"    All closed sub-algebra sizes: {all_sizes}")
        else:
            print(f"    No closed sub-algebra >= 3 found within base EFs")
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  Landscape scan results (sorted by conv rate):")
    for alpha, beta, conv_rate, n_efs, mean_frob in sorted(results, key=lambda x: -x[2]):
        print(f"    alpha={alpha}, beta={beta}: conv={conv_rate*100:.1f}%, n_EFs={n_efs}, frob={mean_frob:.3f}")
    print(f"  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
