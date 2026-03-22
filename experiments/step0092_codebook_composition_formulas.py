#!/usr/bin/env python3
"""
Step 92 -- Three composition formulas for spectral eigenforms.
Spec. k=8. Same Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k).
Formula A: Psi(A,B) = Phi(A*B*A^T)
Formula B: Psi(A,B) = Phi(A*B + B*A)
Formula C: Psi(A,B) = Phi(A + B - A*B/||A*B||*target)
Key test: A o B != A AND A o B != B (genuine mixing)
"""
import random, math, time

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


def mmt(M, k):  # M * M^T
    return [[sum(M[i][l]*M[j][l] for l in range(k)) for j in range(k)] for i in range(k)]


def mt(M, k):  # M^T
    return [[M[j][i] for j in range(k)] for i in range(k)]


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
    return dot/(na*nb) if na>1e-10 and nb>1e-10 else 0.0


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


def same_basin(A, B, k):
    return abs(cosine(A, B, k)) > BASIN_COS


def find_nearest(M, efs, k, thr=KNOWN_COS):
    best_i, best_c = -1, -1.0
    for i, ef in enumerate(efs):
        c = abs(cosine(M, ef, k))
        if c > best_c: best_c = c; best_i = i
    return (best_i, best_c) if best_c >= thr else (-1, best_c)


def fp_label(fp, N): return f"EF_{fp}" if fp < N else f"N{fp-N}"


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


def run_composition(name, psi_fn, efs, k):
    N = len(efs)
    all_known = [copy_mat(ef, k) for ef in efs]
    novel_count = 0

    def register(M_f):
        nonlocal novel_count
        idx, _ = find_nearest(M_f, all_known, k, KNOWN_COS)
        if idx >= 0: return idx, False
        all_known.append(copy_mat(M_f, k))
        novel_count += 1
        return len(all_known)-1, True

    table = [[-1]*N for _ in range(N)]
    conv_mat = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            start = psi_fn(efs[i], efs[j], k)
            if start is None:
                continue
            M_f, conv = converge(start, k, COMPOSE_STEPS)
            conv_mat[i][j] = conv
            if conv:
                fp, _ = register(M_f)
                table[i][j] = fp

    n_conv = sum(conv_mat[i][j] for i in range(N) for j in range(N))
    n_known = n_conv - novel_count

    # Commutativity
    n_comm, n_comm_tot, non_comm = 0, 0, []
    for i in range(N):
        for j in range(i+1, N):
            if conv_mat[i][j] and conv_mat[j][i]:
                n_comm_tot += 1
                if table[i][j] == table[j][i]: n_comm += 1
                else: non_comm.append((i, j, table[i][j], table[j][i]))

    # Non-trivial (result != either input)
    nontrivial = [(i, j, table[i][j], table[j][i])
                  for i in range(N) for j in range(i+1, N)
                  if conv_mat[i][j] and conv_mat[j][i]
                  and table[i][j] >= 0 and table[j][i] >= 0
                  and table[i][j] != i and table[i][j] != j
                  and table[j][i] != i and table[j][i] != j]

    print(f"\n  === Formula {name} ===")
    print(f"  Conv: {n_conv}/{N*N} ({n_conv/(N*N)*100:.1f}%), Known: {n_known}, Novel: {novel_count}")
    # Print table
    row_hdr = "  " + " "*5 + "".join(f" {j:>3}" for j in range(N))
    print(row_hdr)
    for i in range(N):
        row = f"  {i:>3}|"
        for j in range(N):
            entry = fp_label(table[i][j], N) if conv_mat[i][j] else "?"
            row += f" {entry:>3}"
        print(row)
    if n_comm_tot > 0:
        print(f"  Commutative: {n_comm}/{n_comm_tot} ({n_comm/n_comm_tot*100:.1f}%)")
    print(f"  Non-commutative pairs: {len(non_comm)}")
    print(f"  Non-trivial (A o B != A, != B, both orders converge): {len(nontrivial)}")
    for i, j, ab, ba in nontrivial[:5]:
        print(f"    EF_{i} o EF_{j} = {fp_label(ab, N)},  EF_{j} o EF_{i} = {fp_label(ba, N)}")
    verdict = "GENUINE MIXING" if nontrivial else ("NON-COMMUTATIVE PROJECTION" if non_comm else "TRIVIAL PROJECTION")
    print(f"  VERDICT: {verdict}")
    return len(nontrivial), len(non_comm), novel_count


def psi_A(A, B, k):  # A*B*A^T
    AB = mmul(A, B, k)
    return mmul(AB, mt(A, k), k)


def psi_B(A, B, k):  # A*B + B*A
    return madd(mmul(A, B, k), mmul(B, A, k), k)


def psi_C(A, B, k):  # A + B - A*B/||A*B||*target
    AB = mmul(A, B, k)
    n = frob(AB, k)
    if n < 1e-10: return madd(A, B, k)
    scaled = mscale(AB, target_norm(k)/n, k)
    return msub(madd(A, B, k), scaled, k)


def main():
    t0 = time.time()
    print(f"Step 92 -- Three composition formulas, k={K}", flush=True)
    print(f"Finding {N_EFS} eigenforms...", flush=True)
    efs = find_efs(N_EFS, K, SEED+K)
    print(f"  Found {len(efs)} eigenforms")

    results = {}
    for name, fn in [("A: Phi(A*B*A^T)", psi_A), ("B: Phi(A*B+B*A)", psi_B), ("C: Phi(A+B-AB_norm)", psi_C)]:
        nt, nc, nov = run_composition(name, fn, efs, K)
        results[name] = (nt, nc, nov)

    print(f"\n=== Summary ===")
    for name, (nt, nc, nov) in results.items():
        print(f"  {name}: non-trivial={nt}, non-comm={nc}, novel={nov}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
