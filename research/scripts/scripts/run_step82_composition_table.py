#!/usr/bin/env python3
"""
Step 82 -- Eigenform algebra: composition table.

[researcher] directive (mail 1198): test whether eigenforms compose into an algebra.
Classification is the wrong benchmark. The question: does Psi(M_i*, M_j*) -> known eigenform?

Part 1: Build composition table for 10 eigenforms.
  C = Psi(M_i*, M_j*) = tanh(alpha*(M_i*+M_j*)/2 + beta*M_i**M_j*/k)
  Iterate C under Phi for 2000 steps.
  Record: converged? which known eigenform?

Part 2: Algebraic structure.
  - Closed? Identity? Associative? Commutative?
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
EIGENFORM_STEPS = 1000   # for finding true eigenforms
COMPOSE_STEPS = 2000     # for composition convergence
BASIN_COS = 0.99

N_EIGENFORMS = 10
SEED = 42


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA / K)))


def psi(Mi, Mj):
    """Cross-apply: Psi(Mi, Mj) = tanh(alpha*(Mi+Mj)/2 + beta*Mi*Mj/k)"""
    avg = mscale(madd(Mi, Mj), ALPHA / 2.0)
    prod = mscale(mmul(Mi, Mj), BETA / K)
    return mtanh(madd(avg, prod))


def converge(M, max_steps=EIGENFORM_STEPS, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        phi_M = phi(M)
        d = frob(msub(phi_M, M))
        M = madd(M, mscale(msub(phi_M, M), DT))
        M = mclip(M, MAX_NORM)
        if d < tol:
            return M, True
    return M, False


def converge_from(M0, max_steps=COMPOSE_STEPS, tol=CONVERGE_TOL):
    """Iterate Phi dynamics starting from arbitrary M0."""
    M = M0
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


def find_eigenforms(n_needed, seed=SEED + 10, max_attempts=3000):
    rng = random.Random(seed)
    found = []
    for _ in range(max_attempts):
        M0 = [[rng.uniform(-2, 2) for _ in range(K)] for _ in range(K)]
        M_f, conv = converge(M0)
        if conv and all(not same_basin(M_f, s) for s in found):
            found.append(M_f)
            if len(found) == n_needed:
                break
    return found


def nearest_eigenform(M, eigenforms):
    """Return index of nearest eigenform by |cosine|, or -1 if none match BASIN_COS."""
    best_idx = -1
    best_cos = -1.0
    for idx, ef in enumerate(eigenforms):
        c = abs(mcosine(M, ef))
        if c > best_cos:
            best_cos = c
            best_idx = idx
    if best_cos > BASIN_COS:
        return best_idx, best_cos
    return -1, best_cos


def label(idx):
    return str(idx) if idx >= 0 else "?"


def main():
    t0 = time.time()
    print("Step 82 -- Eigenform algebra: composition table", flush=True)
    print(f"K={K}, N_EIGENFORMS={N_EIGENFORMS}, COMPOSE_STEPS={COMPOSE_STEPS}")
    print()

    # ── Find eigenforms ───────────────────────────────────────────────────────
    print(f"Finding {N_EIGENFORMS} distinct eigenforms...", flush=True)
    eigenforms = find_eigenforms(N_EIGENFORMS)
    N = len(eigenforms)
    print(f"  Found {N}")
    for i, ef in enumerate(eigenforms):
        print(f"  EF {i:2d}: frob={frob(ef):.4f}")
    print()

    if N < 2:
        print("  Not enough eigenforms to build composition table.")
        return

    # ── Part 1: Composition table ─────────────────────────────────────────────
    print("=== Part 1: Composition table ===", flush=True)
    print(f"Psi(M_i*, M_j*) -> Phi-convergence -> which eigenform?")
    print()

    table = [[-2] * N for _ in range(N)]   # -2 = not computed, -1 = noconv
    n_conv = 0
    n_total = N * N
    converge_cos = []

    for i in range(N):
        for j in range(N):
            C = psi(eigenforms[i], eigenforms[j])
            C_final, conv = converge_from(C)
            if conv:
                n_conv += 1
                idx, cos_val = nearest_eigenform(C_final, eigenforms)
                table[i][j] = idx
                converge_cos.append(cos_val)
            else:
                table[i][j] = -1

    conv_rate = n_conv / n_total * 100
    cos_mean = sum(converge_cos) / len(converge_cos) if converge_cos else 0.0
    print(f"  Convergence: {n_conv}/{n_total} ({conv_rate:.1f}%)")
    print(f"  Mean cosine to nearest EF (when converged): {cos_mean:.4f}")
    print()

    # Print the table
    print("  Composition table (row=i, col=j, entry=Psi(EF_i, EF_j) -> EF_? or ? if noconv):")
    header = "  " + " " * 4 + "".join(f"  {j:2d}" for j in range(N))
    print(header)
    for i in range(N):
        row = f"  {i:2d} |"
        for j in range(N):
            entry = table[i][j]
            row += f"  {label(entry):2s}"
        print(row)
    print()

    if n_conv == 0:
        print("  No convergence. Composition under Psi does not produce known eigenforms.")
        elapsed = time.time() - t0
        print(f"\n  Elapsed: {elapsed:.1f}s")
        return

    # ── Part 2: Algebraic structure ───────────────────────────────────────────
    print("=== Part 2: Algebraic structure ===", flush=True)

    # Closed: all converged entries are known eigenforms
    known_ids = set(range(N))
    conv_entries = [table[i][j] for i in range(N) for j in range(N) if table[i][j] >= 0]
    closed = all(e in known_ids for e in conv_entries)
    novel_ids = set(e for e in conv_entries if e not in known_ids)
    print(f"  Closed: {closed}  (all converged results are among the {N} known EFs)")
    if novel_ids:
        print(f"    Novel eigenform IDs produced: {novel_ids}")
    print()

    # Identity: M_id where Psi(M_id, M_j) -> M_j for all j (or vice versa)
    left_identity = []
    right_identity = []
    for i in range(N):
        # Left identity: table[i][j] == j for all j where table[i][j] >= 0
        l_ok = all(table[i][j] == j for j in range(N) if table[i][j] >= 0)
        r_ok = all(table[j][i] == j for j in range(N) if table[j][i] >= 0)
        if l_ok and n_conv > 0:
            left_identity.append(i)
        if r_ok and n_conv > 0:
            right_identity.append(i)
    print(f"  Left identity candidates:  {left_identity}")
    print(f"  Right identity candidates: {right_identity}")
    identity = list(set(left_identity) & set(right_identity))
    print(f"  Two-sided identity: {identity}")
    print()

    # Commutative: table[i][j] == table[j][i]
    n_comm_pairs = 0
    n_comm_total = 0
    for i in range(N):
        for j in range(i + 1, N):
            if table[i][j] >= 0 and table[j][i] >= 0:
                n_comm_total += 1
                if table[i][j] == table[j][i]:
                    n_comm_pairs += 1
    comm_rate = n_comm_pairs / n_comm_total * 100 if n_comm_total > 0 else 0.0
    print(f"  Commutative: {n_comm_pairs}/{n_comm_total} pairs commute ({comm_rate:.1f}%)")
    print()

    # Associative: sample 20 random triples (A,B,C), check (A*B)*C == A*(B*C)
    rng = random.Random(SEED + 77)
    n_assoc_tests = 20
    n_assoc_ok = 0
    n_assoc_testable = 0
    print(f"  Associativity: testing {n_assoc_tests} random triples (A,B,C)...")
    for _ in range(n_assoc_tests):
        a, b, c = rng.randint(0, N - 1), rng.randint(0, N - 1), rng.randint(0, N - 1)
        ab = table[a][b]
        bc = table[b][c]
        if ab >= 0 and bc >= 0:
            n_assoc_testable += 1
            lhs = table[ab][c]   # (A*B)*C
            rhs = table[a][bc]   # A*(B*C)
            if lhs >= 0 and rhs >= 0:
                if lhs == rhs:
                    n_assoc_ok += 1
                # else not associative for this triple

    assoc_rate = n_assoc_ok / n_assoc_testable * 100 if n_assoc_testable > 0 else 0.0
    print(f"    Testable triples: {n_assoc_testable}, associative: {n_assoc_ok} ({assoc_rate:.1f}%)")
    print()

    # Summary
    print("=== Summary ===")
    print(f"  Convergence under Psi: {conv_rate:.1f}%")
    print(f"  Closed algebra:        {closed}")
    print(f"  Identity element:      {identity if identity else 'none'}")
    print(f"  Commutative:           {comm_rate:.1f}% of pairs")
    print(f"  Associative:           {assoc_rate:.1f}% of testable triples")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
