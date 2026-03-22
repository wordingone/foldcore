#!/usr/bin/env python3
"""
Step 74 — EigenFold proof-of-concept.

Spec: matrix codebook, eigenform stability as classifier.
3 clusters in R^16, presented sequentially. Pure Python, rk.py only.

Spec:
- Codebook: list of 4×4 matrices, starts empty
- Eigenform: Φ(M) = tanh(1.2·M + 0.8·M²/4)
- Input projection: random frozen P: R^16 → R^16, reshape to 4×4
- δ_i = frob(Ψ(M_i, R) - M_i)  [cross-application perturbation]
- Classify: smallest δ_i = most stable = best match
- Update: M_winner += λ·δ_winner  (λ=0.1)
- Recovery: 5 steps of eigenform dynamics, clip to max_norm=3
- Spawn: if min(δ_i) > threshold, init via 20 eigenform steps from R
"""
import sys, random, math, time
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import mzero, madd, msub, mscale, mmul, mtanh, frob, mcosine, mclip, mrand

# ─── Hyperparameters ─────────────────────────────────────────────────────────
K          = 4      # matrix dimension
D          = 16     # input dimension
ALPHA      = 1.2
BETA       = 0.8
LAM        = 0.1    # winner update step size
RECOVERY   = 5      # eigenform recovery steps after update
DT         = 0.03   # eigenform step size
MAX_NORM   = 3.0
INIT_STEPS = 20     # eigenform steps when initializing new element
THRESHOLD  = 1.0    # spawn if min(δ) > this

N_CLUSTERS = 3
N_TRAIN    = 100
N_TEST     = 30
SIGMA      = 0.3
SEED       = 42


# ─── Eigenform math ──────────────────────────────────────────────────────────

def phi(M):
    """Φ(M) = tanh(α·M + β·M²/k)"""
    linear = mscale(M, ALPHA)
    quad   = mscale(mmul(M, M), BETA / K)
    return mtanh(madd(linear, quad))


def cross_apply(Mi, Mj):
    """Ψ(Mi,Mj) = tanh(α·(Mi+Mj)/2 + β·Mi·Mj/k)"""
    avg  = mscale(madd(Mi, Mj), ALPHA / 2.0)
    prod = mscale(mmul(Mi, Mj), BETA / K)
    return mtanh(madd(avg, prod))


def eigenform_steps(M, n=INIT_STEPS):
    """Drive M toward eigenform for n steps."""
    for _ in range(n):
        M = madd(M, mscale(msub(phi(M), M), DT))
        M = mclip(M, MAX_NORM)
    return M


# ─── Projection ──────────────────────────────────────────────────────────────

def make_projection(d, k2, seed):
    """Random frozen (k², d) projection matrix."""
    rng = random.Random(seed)
    return [[rng.gauss(0, 1.0 / math.sqrt(d)) for _ in range(d)] for _ in range(k2)]


def project(P, x):
    """x ∈ R^d → k×k matrix R."""
    k = int(math.sqrt(len(P)))
    flat = [sum(P[i][j] * x[j] for j in range(len(x))) for i in range(len(P))]
    return [flat[i * k:(i + 1) * k] for i in range(k)]


# ─── EigenFold codebook ──────────────────────────────────────────────────────

class EigenFold:
    def __init__(self, P):
        self.P        = P
        self.elements = []   # list of (M, label)

    def _delta(self, M, R):
        return frob(msub(cross_apply(M, R), M))

    def classify(self, x):
        if not self.elements:
            return None, None
        R      = project(self.P, x)
        deltas = [self._delta(M, R) for M, _ in self.elements]
        best   = min(range(len(deltas)), key=lambda i: deltas[i])
        return self.elements[best][1], deltas[best]

    def train(self, x, label):
        R = project(self.P, x)

        if not self.elements:
            self.elements.append((eigenform_steps(R), label))
            return

        deltas  = [self._delta(M, R) for M, _ in self.elements]
        min_d   = min(deltas)
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])

        if min_d > THRESHOLD:
            self.elements.append((eigenform_steps(R), label))
        else:
            M_w, lbl_w = self.elements[win_idx]
            delta_vec   = msub(cross_apply(M_w, R), M_w)
            M_w         = madd(M_w, mscale(delta_vec, LAM))
            for _ in range(RECOVERY):
                M_w = madd(M_w, mscale(msub(phi(M_w), M_w), DT))
            M_w = mclip(M_w, MAX_NORM)
            self.elements[win_idx] = (M_w, lbl_w)


# ─── Data generation ─────────────────────────────────────────────────────────

def generate_data(seed=SEED):
    rng   = random.Random(seed)
    means = [[rng.gauss(0, 1.0) for _ in range(D)] for _ in range(N_CLUSTERS)]
    train_clusters = []
    test_clusters  = []
    for c, mu in enumerate(means):
        tr = [([mu[j] + rng.gauss(0, SIGMA) for j in range(D)], c) for _ in range(N_TRAIN)]
        te = [([mu[j] + rng.gauss(0, SIGMA) for j in range(D)], c) for _ in range(N_TEST)]
        train_clusters.append(tr)
        test_clusters.append(te)
    return train_clusters, test_clusters


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("  EigenFold — Step 74")
    print(f"  k={K} d={D} alpha={ALPHA} beta={BETA} lam={LAM} threshold={THRESHOLD}")
    print(f"  3 clusters x {N_TRAIN} train / {N_TEST} test, sigma={SIGMA}")
    print("=" * 60)
    print()

    train_clusters, test_clusters = generate_data()
    P  = make_projection(D, K * K, seed=SEED + 1)
    ef = EigenFold(P)

    acc_matrix = [[None] * N_CLUSTERS for _ in range(N_CLUSTERS)]

    for seen, cluster_data in enumerate(train_clusters):
        print(f"  Training cluster {seen} ({len(cluster_data)} samples)...")
        for x, lbl in cluster_data:
            ef.train(x, lbl)
        print(f"    Codebook size: {len(ef.elements)}")
        print(f"    Labels: {[lbl for _, lbl in ef.elements]}")

        # Evaluate all clusters seen so far
        for c, held in enumerate(test_clusters):
            correct = sum(1 for x, lbl in held
                         if ef.classify(x)[0] == lbl)
            acc_matrix[c][seen] = correct / len(held)

        known = [acc_matrix[c][seen] for c in range(seen + 1)]
        aa    = sum(known) / len(known)
        print(f"    Acc (all): {[f'{acc_matrix[c][seen]:.0%}' for c in range(N_CLUSTERS)]}")
        print(f"    AA (seen {seen+1}): {aa:.1%}")
        print()

    # Final summary
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Cluster':<10} {'After own':<12} {'Final':<12} {'Forgetting':<12}")
    print(f"  {'-'*8}   {'-'*9}   {'-'*9}   {'-'*9}")

    total_fgt = 0.0
    for c in range(N_CLUSTERS - 1):
        after_own = acc_matrix[c][c]
        final     = acc_matrix[c][N_CLUSTERS - 1]
        fgt       = max(0.0, after_own - final)
        total_fgt += fgt
        print(f"  {c:<10} {after_own:<12.1%} {final:<12.1%} {fgt:.1%}")

    final_aa = sum(acc_matrix[c][N_CLUSTERS - 1] for c in range(N_CLUSTERS)) / N_CLUSTERS
    avg_fgt  = total_fgt / (N_CLUSTERS - 1)

    print()
    print(f"  Final AA:       {final_aa:.1%}")
    print(f"  Avg forgetting: {avg_fgt:.1%}")
    print(f"  Codebook size:  {len(ef.elements)}")
    print(f"  Elapsed:        {time.time()-t0:.1f}s")
    print()
    print(f"  Baselines (FluxCore v17 many-to-few, P-MNIST):")
    print(f"    Coverage 33/33, energy=0.081, 25s")
    print("=" * 60)


if __name__ == '__main__':
    main()
