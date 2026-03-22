#!/usr/bin/env python3
"""
Step 91 -- New eigenform equation: normalized spectral self-interaction.

Spec:
Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k)

Properties:
- Scale-independent: normalization prevents saturation at any k
- M*M^T is always symmetric PSD
- Fixed points: M where M*M^T / ||M*M^T|| proportional to M

Test for k in [4, 8, 16]. If convergence > 20%, compute 6x6 composition table.
Composition: Psi(A,B) = iterate Phi from A*B to fixed point.
"""
import sys, random, math, time

SEED = 42
LANDSCAPE_N = 500
LANDSCAPE_STEPS = 200
CONVERGE_TOL = 0.01   # Avir spec: frob(Phi(M) - M) < 0.01
BASIN_COS = 0.95      # for distinct EF detection
KNOWN_COS = 0.90      # for composition result identification
COMPOSE_STEPS = 200
K_VALUES = [4, 8, 16]


def target_norm(k):
    return math.sqrt(k)


def mat_mul_transpose(M, k):
    """M * M^T"""
    return [[sum(M[i][l] * M[j][l] for l in range(k)) for j in range(k)] for i in range(k)]


def mat_frob(M, k):
    return math.sqrt(sum(M[i][j] ** 2 for i in range(k) for j in range(k)))


def mat_sub(A, B, k):
    return [[A[i][j] - B[i][j] for j in range(k)] for i in range(k)]


def mat_scale(M, s, k):
    return [[M[i][j] * s for j in range(k)] for i in range(k)]


def mat_mul(A, B, k):
    """A * B"""
    return [[sum(A[i][l] * B[l][j] for l in range(k)) for j in range(k)] for i in range(k)]


def mat_cosine(A, B, k):
    dot = sum(A[i][j] * B[i][j] for i in range(k) for j in range(k))
    na = mat_frob(A, k)
    nb = mat_frob(B, k)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return dot / (na * nb)


def copy_mat(M, k):
    return [[M[i][j] for j in range(k)] for i in range(k)]


def phi(M, k):
    """Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k)"""
    MMt = mat_mul_transpose(M, k)
    norm = mat_frob(MMt, k)
    if norm < 1e-10:
        return copy_mat(M, k)
    return mat_scale(MMt, target_norm(k) / norm, k)


def converge(M, k, max_steps, tol=CONVERGE_TOL):
    """Iterate Phi directly (no dt — direct replacement)."""
    for _ in range(max_steps):
        phi_M = phi(M, k)
        diff = mat_frob(mat_sub(phi_M, M, k), k)
        M = phi_M
        if diff < tol:
            return M, True
    return M, False


def same_basin(A, B, k):
    return abs(mat_cosine(A, B, k)) > BASIN_COS


def find_nearest(M, ef_list, k, threshold=KNOWN_COS):
    best_idx, best_cos = -1, -1.0
    for idx, ef in enumerate(ef_list):
        c = abs(mat_cosine(M, ef, k))
        if c > best_cos:
            best_cos = c
            best_idx = idx
    if best_cos >= threshold:
        return best_idx, best_cos
    return -1, best_cos


def ef_fingerprint(M, all_known, k):
    idx, _ = find_nearest(M, all_known, k, KNOWN_COS)
    if idx >= 0:
        return idx
    all_known.append(copy_mat(M, k))
    return len(all_known) - 1


def fp_label(fp, N_base):
    return f"EF_{fp}" if fp < N_base else f"N{fp-N_base}"


def scan_landscape(k, seed):
    rng = random.Random(seed)
    eigenforms = []
    frob_vals = []
    n_conv = 0

    for _ in range(LANDSCAPE_N):
        M0 = [[rng.uniform(-1, 1) for _ in range(k)] for _ in range(k)]
        M_f, conv = converge(M0, k, LANDSCAPE_STEPS)
        if conv:
            n_conv += 1
            frob_vals.append(mat_frob(M_f, k))
            if all(not same_basin(M_f, ef, k) for ef in eigenforms):
                eigenforms.append(copy_mat(M_f, k))

    conv_rate = n_conv / LANDSCAPE_N
    mean_frob = sum(frob_vals) / len(frob_vals) if frob_vals else 0.0
    return conv_rate, len(eigenforms), mean_frob, eigenforms


def compose_pair(A, B, k):
    """Psi(A,B) = iterate Phi from A*B."""
    AB = mat_mul(A, B, k)
    return converge(AB, k, COMPOSE_STEPS)


def main():
    t0 = time.time()
    print("Step 91 -- Normalized spectral eigenform: Phi(M) = M*M^T / ||M*M^T||_F * sqrt(k)", flush=True)
    print(f"N={LANDSCAPE_N}, LANDSCAPE_STEPS={LANDSCAPE_STEPS}, tol={CONVERGE_TOL}", flush=True)
    print()

    results = {}

    for k in K_VALUES:
        print(f"=== k={k} ({k}x{k} matrices, target_norm={target_norm(k):.3f}) ===", flush=True)
        ts = time.time()
        conv_rate, n_efs, mean_frob, eigenforms = scan_landscape(k, SEED + k)
        elapsed_k = time.time() - ts
        print(f"  Convergence: {int(conv_rate*LANDSCAPE_N)}/{LANDSCAPE_N} ({conv_rate*100:.1f}%)")
        print(f"  Distinct eigenforms: {n_efs}")
        print(f"  Mean frob: {mean_frob:.4f}  (target={target_norm(k):.3f})")
        print(f"  Elapsed: {elapsed_k:.1f}s")
        results[k] = (conv_rate, n_efs, mean_frob, eigenforms)
        print()

    # Summary table
    print("=== Landscape summary ===")
    print(f"  {'k':>4}  {'conv%':>7}  {'n_EFs':>6}  {'mean_frob':>10}")
    print(f"  {'----':>4}  {'-------':>7}  {'------':>6}  {'----------':>10}")
    for k in K_VALUES:
        conv_rate, n_efs, mean_frob, _ = results[k]
        print(f"  {k:>4}  {conv_rate*100:>6.1f}%  {n_efs:>6}  {mean_frob:>10.4f}")
    print()

    # Composition tables for any k with > 20% convergence
    for k in K_VALUES:
        conv_rate, n_efs, mean_frob, eigenforms = results[k]
        if conv_rate < 0.20:
            print(f"  k={k}: convergence {conv_rate*100:.1f}% < 20%, skipping composition.", flush=True)
            continue

        print(f"=== k={k}: Composition table (6x6) ===", flush=True)
        n6 = min(6, n_efs)
        ef6 = eigenforms[:n6]
        all_known = [copy_mat(ef, k) for ef in ef6]
        novel_count = 0

        def get_or_reg(M_f):
            nonlocal novel_count
            idx, _ = find_nearest(M_f, all_known, k, KNOWN_COS)
            if idx >= 0:
                return idx, False
            all_known.append(copy_mat(M_f, k))
            novel_count += 1
            return len(all_known) - 1, True

        table = [[-1] * n6 for _ in range(n6)]
        conv_mat = [[False] * n6 for _ in range(n6)]

        for i in range(n6):
            for j in range(n6):
                M_f, conv = compose_pair(ef6[i], ef6[j], k)
                conv_mat[i][j] = conv
                if conv:
                    fp, _ = get_or_reg(M_f)
                    table[i][j] = fp

        n_conv_comp = sum(conv_mat[i][j] for i in range(n6) for j in range(n6))
        n_known_comp = n_conv_comp - novel_count
        print(f"  Composition convergence: {n_conv_comp}/{n6*n6} ({n_conv_comp/(n6*n6)*100:.1f}%)")
        print(f"  Known: {n_known_comp}, Novel: {novel_count}")

        # Print table
        labels = [f"E{i}" for i in range(n6)]
        header = "  " + " " * 4 + "".join(f" {l:>3}" for l in labels)
        print(header)
        for i in range(n6):
            row = f"  {labels[i]:>3}|"
            for j in range(n6):
                if conv_mat[i][j]:
                    fp = table[i][j]
                    entry = str(fp) if fp < n6 else "N"
                else:
                    entry = "?"
                row += f" {entry:>3}"
            print(row)
        print()

        # Commutativity
        n_comm = 0
        n_comm_total = 0
        non_comm = []
        for i in range(n6):
            for j in range(i + 1, n6):
                if conv_mat[i][j] and conv_mat[j][i]:
                    n_comm_total += 1
                    if table[i][j] == table[j][i]:
                        n_comm += 1
                    else:
                        non_comm.append((i, j, table[i][j], table[j][i]))
        if n_comm_total > 0:
            print(f"  Commutative: {n_comm}/{n_comm_total} ({n_comm/n_comm_total*100:.1f}%)")
        print(f"  Non-commutative pairs: {len(non_comm)}")
        for a, b, ab, ba in non_comm[:5]:
            print(f"    EF_{a} o EF_{b} = {fp_label(ab, n6)}, EF_{b} o EF_{a} = {fp_label(ba, n6)}")

        # Non-trivial (result != either input)
        n_nontrivial = 0
        for i in range(n6):
            for j in range(i + 1, n6):
                if not (conv_mat[i][j] and conv_mat[j][i]):
                    continue
                r_ij, r_ji = table[i][j], table[j][i]
                if (r_ij != i and r_ij != j and r_ij >= 0 and
                        r_ji != i and r_ji != j and r_ji >= 0):
                    n_nontrivial += 1
                    print(f"  NON-TRIVIAL: EF_{i} o EF_{j} = {fp_label(r_ij, n6)}, EF_{j} o EF_{i} = {fp_label(r_ji, n6)}")
        if n_nontrivial == 0:
            print(f"  Non-trivial compositions: 0 (all are projections/absorptions)")

        # Absorbers
        absorbers = []
        for ki in range(n6):
            lz = all(not conv_mat[ki][j] or table[ki][j] == ki for j in range(n6))
            rz = all(not conv_mat[i][ki] or table[i][ki] == ki for i in range(n6))
            if lz or rz:
                absorbers.append((ki, lz, rz))
        print(f"  Absorbers: {len(absorbers)} {[(f'EF_{a}', l, r) for a,l,r in absorbers]}")
        print()

    elapsed = time.time() - t0
    print("=== Final summary ===")
    for k in K_VALUES:
        conv_rate, n_efs, mean_frob, _ = results[k]
        verdict = "VIABLE" if conv_rate >= 0.20 else ("SPARSE" if conv_rate >= 0.05 else "BARREN")
        print(f"  k={k:>2}: {conv_rate*100:>5.1f}% convergence, {n_efs:>4} EFs — {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
