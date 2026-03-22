#!/usr/bin/env python3
"""
Step 94 -- Scale the alphabet: 8 EFs, test discrimination.
[researcher] mail 1225. k=8, spectral Phi, Formula C.

1. 8^3 = 512 length-3 sequences — how many distinct?
2. 56 ordered pairs (8x7, no self) — how many distinct?
3. 100 random length-6 sequences — how many distinct?
"""
import random, math, time, itertools

SEED = 42
K = 8
LANDSCAPE_STEPS = 200
COMPOSE_STEPS = 200
CONVERGE_TOL = 0.01
BASIN_COS = 0.95
KNOWN_COS = 0.90
N_EFS = 8


def target_norm(k): return math.sqrt(k)


def mmul(A, B, k):
    return [[sum(A[i][l]*B[l][j] for l in range(k)) for j in range(k)] for i in range(k)]


def mmt(M, k):
    return [[sum(M[i][l]*M[j][l] for l in range(k)) for j in range(k)] for i in range(k)]


def madd(A, B, k):
    return [[A[i][j]+B[i][j] for j in range(k)] for i in range(k)]


def msub(A, B, k):
    return [[A[i][j]-B[i][j] for j in range(k)] for i in range(k)]


def mscale(M, s, k):
    return [[M[i][j]*s for j in range(k)] for i in range(k)]


def frob(M, k):
    return math.sqrt(sum(M[i][j]**2 for i in range(k) for j in range(k)))


def cosine(A, B, k):
    dot = sum(A[i][j]*B[i][j] for i in range(k) for j in range(k))
    na, nb = frob(A, k), frob(B, k)
    return dot/(na*nb) if na > 1e-10 and nb > 1e-10 else 0.0


def copy_mat(M, k):
    return [[M[i][j] for j in range(k)] for i in range(k)]


def phi(M, k):
    C = mmt(M, k)
    n = frob(C, k)
    if n < 1e-10: return copy_mat(M, k)
    return mscale(C, target_norm(k)/n, k)


def converge(M, k, max_steps, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        p = phi(M, k)
        d = frob(msub(p, M, k), k)
        M = p
        if d < tol: return M, True
    return M, False


def psi_C(A, B, k):
    AB = mmul(A, B, k)
    n = frob(AB, k)
    if n < 1e-10: return madd(A, B, k)
    scaled = mscale(AB, target_norm(k)/n, k)
    return msub(madd(A, B, k), scaled, k)


def compose(A, B, k):
    start = psi_C(A, B, k)
    return converge(start, k, COMPOSE_STEPS)


def same_basin(A, B, k):
    return abs(cosine(A, B, k)) > BASIN_COS


def find_nearest(M, efs, k, thr=KNOWN_COS):
    best_i, best_c = -1, -1.0
    for i, ef in enumerate(efs):
        c = abs(cosine(M, ef, k))
        if c > best_c: best_c = c; best_i = i
    return (best_i, best_c) if best_c >= thr else (-1, best_c)


def find_efs(n, k, seed):
    rng = random.Random(seed)
    found = []
    for _ in range(2000):
        M0 = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(k)]
        M_f, conv = converge(M0, k, LANDSCAPE_STEPS)
        if conv and all(not same_basin(M_f, ef, k) for ef in found):
            found.append(M_f)
            if len(found) == n: break
    return found


def register_result(M_f, registry, k):
    idx, _ = find_nearest(M_f, registry, k, KNOWN_COS)
    if idx >= 0: return idx
    registry.append(copy_mat(M_f, k))
    return len(registry) - 1


def chain(seq, k):
    """Left-to-right: ((seq[0] o seq[1]) o seq[2]) o ..."""
    M, conv = compose(seq[0], seq[1], k)
    if not conv: return None, False
    for i in range(2, len(seq)):
        M, conv = compose(M, seq[i], k)
        if not conv: return None, False
    return M, True


def run_test(name, seqs, efs, k):
    registry = []
    n_conv = 0
    n_total = len(seqs)
    for seq_idx in seqs:
        seq = [efs[i] for i in seq_idx]
        M_f, conv = chain(seq, k)
        if conv:
            n_conv += 1
            register_result(M_f, registry, k)
    n_distinct = len(registry)
    pct = n_distinct / n_conv * 100 if n_conv > 0 else 0
    print(f"  {name}: conv={n_conv}/{n_total}, distinct={n_distinct} ({pct:.1f}% of converged)")
    return n_conv, n_distinct


def main():
    t0 = time.time()
    print(f"Step 94 -- Scale alphabet: {N_EFS} EFs, k={K}", flush=True)
    print()

    print(f"Finding {N_EFS} eigenforms...", flush=True)
    efs = find_efs(N_EFS, K, SEED + K)
    print(f"  Found {len(efs)} eigenforms")
    print()

    N = len(efs)

    # Test 1: All 8^3 = 512 length-3 sequences
    print(f"=== Test 1: All {N}^3 = {N**3} length-3 sequences ===", flush=True)
    seqs_l3 = list(itertools.product(range(N), repeat=3))
    c1, d1 = run_test(f"{N}^3 length-3", seqs_l3, efs, K)
    print(f"  Reference: {N**3} sequences, {d1} distinct (ratio {d1/N**3:.3f})")
    print()

    # Test 2: 56 ordered pairs (all i!=j)
    print(f"=== Test 2: {N*(N-1)} ordered pairs (i!=j) ===", flush=True)
    seqs_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    c2, d2 = run_test(f"{N}x{N-1} ordered pairs", seqs_pairs, efs, K)
    print(f"  Full non-commutativity would give {N*(N-1)} distinct")
    print()

    # Test 3: 100 random length-6 sequences
    print(f"=== Test 3: 100 random length-6 sequences ===", flush=True)
    rng = random.Random(SEED + 100)
    seqs_l6 = [tuple(rng.randint(0, N-1) for _ in range(6)) for _ in range(100)]
    c3, d3 = run_test("100 random length-6", seqs_l6, efs, K)
    print()

    # Summary
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  k={K}, {N} EFs, spectral Phi, Formula C")
    print(f"  Length-3 ({N}^3={N**3} seqs): {d1} distinct ({d1/N**3*100:.1f}%)")
    print(f"  Ordered pairs ({N*(N-1)}): {d2} distinct ({d2/(N*(N-1))*100:.1f}%)")
    print(f"  Length-6 (100 random): {d3} distinct ({d3/100*100:.1f}%)")
    print(f"Elapsed: {elapsed:.1f}s")

    # Verdict
    scales = d1 >= 200
    print(f"\n  VERDICT: Alphabet scaling {'WORKS' if scales else 'LIMITED'}")
    if d1 >= 200:
        print(f"  {d1}/512 distinct length-3 sequences — deep quotient structure absent at larger alphabet")
    else:
        print(f"  {d1}/512 distinct length-3 sequences — quotient structure persists at scale")
    if d2 == N*(N-1):
        print(f"  Fully non-commutative: all {d2} ordered pairs distinct")
    else:
        print(f"  Partial commutativity: {d2}/{N*(N-1)} distinct ordered pairs")


if __name__ == '__main__':
    main()
