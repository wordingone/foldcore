#!/usr/bin/env python3
"""
Step 93 -- Sequence encoding with Formula C + spectral eigenform.
Spec. k=8.

Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k)
Formula C: Psi(A,B) = Phi(A + B - A*B/||A*B||*target)

Part 1: 24 permutations of 4 EFs, left-to-right chains ((A o B) o C) o D
Part 2: 81 length-4 sequences from 3 EFs (3^4 with repetition)
Part 3: Associativity (10 triples) + noise consistency (5 pairs)
"""
import random, math, time, itertools

SEED = 42
K = 8
LANDSCAPE_STEPS = 200
COMPOSE_STEPS = 200
CONVERGE_TOL = 0.01
BASIN_COS = 0.95
KNOWN_COS = 0.90
N_EFS = 6


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


def madd_noise(M, eps, rng, k):
    return [[M[i][j] + rng.uniform(-eps, eps) for j in range(k)] for i in range(k)]


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


def mat_key(M, k, decimals=3):
    """Round matrix to create a dict key for deduplication."""
    return tuple(round(M[i][j], decimals) for i in range(k) for j in range(k))


def register_result(M_f, registry, k):
    """Register result and return index. registry = list of matrices."""
    idx, _ = find_nearest(M_f, registry, k, KNOWN_COS)
    if idx >= 0: return idx
    registry.append(copy_mat(M_f, k))
    return len(registry) - 1


def chain(seq, k):
    """Left-to-right chain: ((seq[0] o seq[1]) o seq[2]) o ..."""
    M, conv = compose(seq[0], seq[1], k)
    if not conv: return None, False
    for i in range(2, len(seq)):
        M, conv = compose(M, seq[i], k)
        if not conv: return None, False
    return M, True


def main():
    t0 = time.time()
    print(f"Step 93 -- Sequence encoding with Formula C, k={K}", flush=True)
    print(f"Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k)", flush=True)
    print(f"Formula C: Psi(A,B) = Phi(A + B - A*B/||A*B||*target)", flush=True)
    print()

    print(f"Finding {N_EFS} eigenforms...", flush=True)
    efs = find_efs(N_EFS, K, SEED + K)
    print(f"  Found {len(efs)} eigenforms")
    print()

    # ── Part 1: 24 permutations of 4 EFs ──────────────────────────────────────
    print("=== Part 1: Length-4 chains (24 permutations of EF_0..EF_3) ===", flush=True)
    ef4 = efs[:4]
    labels = ['A', 'B', 'C', 'D']
    perms = list(itertools.permutations(range(4)))  # 24 permutations

    registry_p1 = []
    chain_results = {}
    n_conv_p1 = 0

    for perm in perms:
        seq = [ef4[i] for i in perm]
        M_f, conv = chain(seq, K)
        chain_results[perm] = (M_f, conv)
        if conv:
            n_conv_p1 += 1
            register_result(M_f, registry_p1, K)

    n_distinct_p1 = len(registry_p1)
    print(f"  Chain convergence: {n_conv_p1}/24 ({n_conv_p1/24*100:.1f}%)")
    print(f"  Distinct results: {n_distinct_p1} (out of 24 permutations)")

    # Check reversed pairs
    reversed_differ = 0
    reversed_total = 0
    for perm in perms:
        rev = tuple(reversed(perm))
        if rev < perm: continue  # avoid double-counting
        M1, c1 = chain_results[perm]
        M2, c2 = chain_results[rev]
        if c1 and c2:
            reversed_total += 1
            cos = abs(cosine(M1, M2, K))
            if cos < BASIN_COS:
                reversed_differ += 1
    print(f"  Reversed pairs that differ: {reversed_differ}/{reversed_total}")

    # Show result distribution
    if n_conv_p1 > 0:
        result_idx = {}
        for perm in perms:
            M_f, conv = chain_results[perm]
            if conv:
                idx = register_result(M_f, registry_p1, K)
                if idx not in result_idx: result_idx[idx] = []
                result_idx[idx].append(perm)
        print(f"  Result distribution:")
        for idx in sorted(result_idx.keys()):
            label = f"R{idx}" if idx >= 4 else f"EF_{idx}"
            perms_str = ", ".join("".join(labels[i] for i in p) for p in result_idx[idx][:6])
            if len(result_idx[idx]) > 6: perms_str += f" ... (+{len(result_idx[idx])-6} more)"
            print(f"    {label}: {len(result_idx[idx])} chains -> {perms_str}")
    print()

    # ── Part 2: 81 sequences (3^4) from 3 EFs ─────────────────────────────────
    print("=== Part 2: 3^4 = 81 length-4 sequences from EF_0, EF_1, EF_2 ===", flush=True)
    ef3 = efs[:3]
    seqs = list(itertools.product(range(3), repeat=4))  # 81 sequences

    registry_p2 = []
    n_conv_p2 = 0
    seq_results = {}

    for seq_idx in seqs:
        seq = [ef3[i] for i in seq_idx]
        M_f, conv = chain(seq, K)
        seq_results[seq_idx] = (M_f, conv)
        if conv:
            n_conv_p2 += 1
            register_result(M_f, registry_p2, K)

    n_distinct_p2 = len(registry_p2)
    print(f"  Sequence convergence: {n_conv_p2}/81 ({n_conv_p2/81*100:.1f}%)")
    print(f"  Distinct results: {n_distinct_p2} / 81 (theoretical max)")

    injectivity = n_conv_p2 > 0 and n_distinct_p2 == n_conv_p2
    print(f"  Near-injective (>50 distinct from 81)? {'YES' if n_distinct_p2 > 50 else 'NO'} ({n_distinct_p2}/81)")
    print(f"  Perfectly injective? {'YES' if injectivity else 'NO'}")

    # Distribution: how many seqs per result
    result_counts_p2 = {}
    for seq_idx in seqs:
        M_f, conv = seq_results[seq_idx]
        if conv:
            idx = register_result(M_f, registry_p2, K)
            result_counts_p2[idx] = result_counts_p2.get(idx, 0) + 1
    singletons = sum(1 for c in result_counts_p2.values() if c == 1)
    max_collisions = max(result_counts_p2.values()) if result_counts_p2 else 0
    print(f"  Singleton results (unique sequences): {singletons}")
    print(f"  Max collisions (sequences mapping to same result): {max_collisions}")
    print()

    # ── Part 3: Associativity + noise ─────────────────────────────────────────
    print("=== Part 3: Associativity + noise consistency ===", flush=True)
    rng = random.Random(SEED)

    # Associativity: (A o B) o C vs A o (B o C)
    print("  Associativity test (10 triples):")
    assoc_matches = 0
    assoc_total = 0
    for trial in range(10):
        i, j, l = [rng.randint(0, N_EFS-1) for _ in range(3)]
        A, B, C = efs[i], efs[j], efs[l]

        AB, c1 = compose(A, B, K)
        if not c1: continue
        AB_C, c2 = compose(AB, C, K)
        if not c2: continue

        BC, c3 = compose(B, C, K)
        if not c3: continue
        A_BC, c4 = compose(A, BC, K)
        if not c4: continue

        assoc_total += 1
        cos = abs(cosine(AB_C, A_BC, K))
        matches = cos > BASIN_COS
        if matches: assoc_matches += 1
        eq_str = "==" if matches else "!="
        print(f"    EF_{i} o EF_{j} o EF_{l}: (A o B) o C {eq_str} A o (B o C) [cos={cos:.4f}]")

    if assoc_total > 0:
        print(f"  Associative: {assoc_matches}/{assoc_total} triples ({assoc_matches/assoc_total*100:.1f}%)")
    print()

    # Noise consistency
    print("  Noise consistency (eps=0.001, 5 compositions):")
    noise_consistent = 0
    noise_total = 0
    rng2 = random.Random(SEED + 1)
    for trial in range(5):
        i, j = rng2.randint(0, N_EFS-1), rng2.randint(0, N_EFS-1)
        A, B = efs[i], efs[j]
        A_noisy = madd_noise(A, 0.001, rng2, K)
        B_noisy = madd_noise(B, 0.001, rng2, K)

        M_clean, c1 = compose(A, B, K)
        M_noisy, c2 = compose(A_noisy, B_noisy, K)
        if not (c1 and c2): continue

        noise_total += 1
        cos = abs(cosine(M_clean, M_noisy, K))
        consistent = cos > KNOWN_COS
        if consistent: noise_consistent += 1
        print(f"    EF_{i} o EF_{j}: clean vs noisy [cos={cos:.4f}] => {'SAME' if consistent else 'DIFFERENT'}")

    if noise_total > 0:
        print(f"  Noise-consistent: {noise_consistent}/{noise_total} compositions")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print("=== Summary ===")
    print(f"  k={K}, spectral Phi, Formula C")
    print(f"  Part 1 (24 permutations of 4 EFs):")
    print(f"    Convergence: {n_conv_p1}/24, Distinct: {n_distinct_p1}, Reversed differ: {reversed_differ}/{reversed_total}")
    print(f"  Part 2 (81 sequences from 3 EFs):")
    print(f"    Convergence: {n_conv_p2}/81, Distinct: {n_distinct_p2}, Near-injective: {'YES' if n_distinct_p2 > 50 else 'NO'}")
    print(f"  Part 3: Associative: {assoc_matches}/{assoc_total if assoc_total else 'N/A'}, Noise-consistent: {noise_consistent}/{noise_total if noise_total else 'N/A'}")
    print(f"Elapsed: {elapsed:.1f}s")

    verdict = []
    if n_distinct_p1 >= 12: verdict.append("ORDER-SENSITIVE chains")
    if n_distinct_p2 > 50: verdict.append("NEAR-INJECTIVE sequence encoding")
    if reversed_differ == reversed_total and reversed_total > 0: verdict.append("PERFECT reversal discrimination")
    if verdict:
        print(f"\n  VERDICT: {' + '.join(verdict)}")
    else:
        print(f"\n  VERDICT: Limited sequence discrimination (need deeper analysis)")


if __name__ == '__main__':
    main()
