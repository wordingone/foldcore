#!/usr/bin/env python3
"""
Step 81 -- Coupled RK cells, true eigenform init.

[researcher] directive (mail 1196): test whether COUPLING between true eigenforms
provides emergent classification that individual eigenforms can't.

Setup: 9 cells (3 per class), each a distinct true eigenform.
Input perturbs all cells; coupling + eigenform recovery runs 10 steps.
Classification by per-class average response (smallest = most stable,
largest = most activated).
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
MAX_STEPS = 1000  # true eigenform init: full convergence
BASIN_COS = 0.99

# Step 81 params
N_CELLS = 9
N_PER_CLASS = 3
EPS = 0.05
COUPLING_STEPS = 10
COUPLING_LR = 0.01

# Toy data
D = 16
N_CLUSTERS = 3
N_TRAIN = 100
N_TEST = 30
SIGMA = 0.3
SEED = 42


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA / K)))


def converge(M, max_steps=MAX_STEPS, tol=CONVERGE_TOL):
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


def make_projection(d, k2, seed):
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0 / math.sqrt(d)) for _ in range(d)] for _ in range(k2)]


def project(P, x):
    flat = [sum(P[i][j] * x[j] for j in range(len(x))) for i in range(len(P))]
    return [flat[i * K:(i + 1) * K] for i in range(K)]


def generate_data(seed=SEED):
    rng = random.Random(seed)
    means = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(N_CLUSTERS)]
    train = []
    test = []
    for c, mu in enumerate(means):
        train.append([([mu[j] + rng.gauss(0, SIGMA) for j in range(D)], c) for _ in range(N_TRAIN)])
        test.append([([mu[j] + rng.gauss(0, SIGMA) for j in range(D)], c) for _ in range(N_TEST)])
    return train, test


def find_eigenforms(n_needed, seed=SEED + 10, max_attempts=2000):
    """Find n_needed distinct eigenforms via full convergence."""
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


def copy_mat(M):
    return [[M[i][j] for j in range(K)] for i in range(K)]


def coupled_classify(cells, R, eps=EPS, coupling_steps=COUPLING_STEPS, coupling_lr=COUPLING_LR):
    """
    Run coupled dynamics for one input R.
    Returns response_i = frob(M_i_final - M_i_original) for each cell.
    """
    # Save originals
    originals = [copy_mat(c) for c in cells]

    # 1. Perturb all cells
    states = [madd(copy_mat(c), mscale(R, eps)) for c in cells]

    # 2. Run coupling steps
    for _ in range(coupling_steps):
        new_states = []
        for i in range(N_CELLS):
            # Coupling: C_i = sum_j w_ij * (M_j - M_i)
            coupling = [[0.0] * K for _ in range(K)]
            w_total = 0.0
            for j in range(N_CELLS):
                if j == i:
                    continue
                diff = msub(states[j], states[i])
                w = 1.0 / (1.0 + frob(msub(states[i], states[j])))
                coupling = madd(coupling, mscale(diff, w))
                w_total += w
            # Update: M_i += lr * C_i
            M_coupled = madd(states[i], mscale(coupling, coupling_lr))
            # Eigenform recovery: M_i += dt * (phi(M_i) - M_i)
            phi_M = phi(M_coupled)
            M_updated = madd(M_coupled, mscale(msub(phi_M, M_coupled), DT))
            M_updated = mclip(M_updated, MAX_NORM)
            new_states.append(M_updated)
        states = new_states

    # 3. Compute responses
    responses = [frob(msub(states[i], originals[i])) for i in range(N_CELLS)]
    return responses


def classify(responses):
    """
    Returns (pred_stable, pred_activated):
    - pred_stable: class with smallest avg response
    - pred_activated: class with largest avg response
    """
    class_responses = []
    for cls in range(N_CLUSTERS):
        cell_start = cls * N_PER_CLASS
        avg = sum(responses[cell_start:cell_start + N_PER_CLASS]) / N_PER_CLASS
        class_responses.append(avg)
    pred_stable = class_responses.index(min(class_responses))
    pred_activated = class_responses.index(max(class_responses))
    return pred_stable, pred_activated


def main():
    t0 = time.time()
    print("Step 81 -- Coupled RK cells, true eigenform init", flush=True)
    print(f"K={K}, N_CELLS={N_CELLS}, N_PER_CLASS={N_PER_CLASS}, eps={EPS}, coupling_steps={COUPLING_STEPS}")
    print()

    # ── Find 9 distinct eigenforms ────────────────────────────────────────────
    print(f"Finding {N_CELLS} distinct eigenforms (true convergence, max_steps={MAX_STEPS})...", flush=True)
    eigenforms = find_eigenforms(N_CELLS)
    print(f"  Found {len(eigenforms)}")
    if len(eigenforms) < N_CELLS:
        print(f"  ERROR: only found {len(eigenforms)}, need {N_CELLS}. Aborting.")
        return
    for i, ef in enumerate(eigenforms):
        cls = i // N_PER_CLASS
        print(f"  Cell {i} (class {cls}): frob={frob(ef):.4f}")
    print()

    # ── Load data and projection ──────────────────────────────────────────────
    train_clusters, test_clusters = generate_data()
    P = make_projection(D, K * K, seed=SEED + 1)
    def embed(x): return project(P, x)

    # ── Evaluate on test set ──────────────────────────────────────────────────
    correct_stable = 0
    correct_activated = 0
    total = 0

    per_class_stable = [0] * N_CLUSTERS
    per_class_activated = [0] * N_CLUSTERS
    per_class_total = [0] * N_CLUSTERS

    # Sample mean responses per class pair for diagnostics
    response_log = []  # (true_class, avg_response_per_class)

    for cls_idx, class_data in enumerate(test_clusters):
        for x, c in class_data:
            R = embed(x)
            responses = coupled_classify(eigenforms, R)
            pred_stable, pred_activated = classify(responses)

            correct_stable += int(pred_stable == c)
            correct_activated += int(pred_activated == c)
            total += 1
            per_class_stable[c] += int(pred_stable == c)
            per_class_activated[c] += int(pred_activated == c)
            per_class_total[c] += 1

            # Aggregate class responses for diagnostics
            class_avgs = []
            for cls in range(N_CLUSTERS):
                start = cls * N_PER_CLASS
                avg = sum(responses[start:start + N_PER_CLASS]) / N_PER_CLASS
                class_avgs.append(avg)
            response_log.append((c, class_avgs))

    aa_stable = correct_stable / total * 100
    aa_activated = correct_activated / total * 100

    # Mean response per (true_class, cell_class) for diagnostics
    print("Response diagnostics (mean response of cell-class for each true-class input):")
    print(f"  {'':12}", end="")
    for cc in range(N_CLUSTERS):
        print(f"  {'cells_cls'+str(cc):<12}", end="")
    print()
    for tc in range(N_CLUSTERS):
        entries = [(c, avgs) for c, avgs in response_log if c == tc]
        print(f"  true_cls {tc}  ", end="")
        for cc in range(N_CLUSTERS):
            mean_r = sum(avgs[cc] for _, avgs in entries) / len(entries) if entries else 0.0
            print(f"  {mean_r:<12.4f}", end="")
        print()
    print()

    print("=" * 60)
    print("  RESULTS -- Step 81")
    print("=" * 60)
    print(f"  Stable   (min response): AA={aa_stable:.1f}%  ({correct_stable}/{total})")
    print(f"  Activated (max response): AA={aa_activated:.1f}%  ({correct_activated}/{total})")
    print()
    print(f"  Per-class stable:    {[f'{per_class_stable[c]/per_class_total[c]:.0%}' for c in range(N_CLUSTERS)]}")
    print(f"  Per-class activated: {[f'{per_class_activated[c]/per_class_total[c]:.0%}' for c in range(N_CLUSTERS)]}")
    print()
    print(f"  Baselines:")
    print(f"    EigenFold individual (Step 75b): 22.2%")
    print(f"    Vector cosine        (Step 76):  46.2%")
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == '__main__':
    main()
