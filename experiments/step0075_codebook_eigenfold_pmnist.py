#!/usr/bin/env python3
"""
Step 75 — EigenFold on P-MNIST, 2 tasks.

Spec: P-MNIST, 2 tasks, 6K/task.
Projection: R^784 -> R^384 (permuted, unit-normalized, numpy) -> R^16 (P2) -> 4x4 matrix.
EigenFold spec unchanged from Step 74.
Sweep thresholds: 0.5, 1.0, 2.0 — skip later values if codebook behavior is stable.
"""
import sys, random, math, time
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip

# ─── EigenFold hyperparameters ───────────────────────────────────────────────
K          = 4      # matrix dimension
D_MID      = 384    # after first projection
D_MAT      = K * K  # 16 — second projection output
ALPHA      = 1.2
BETA       = 0.8
LAM        = 0.1
RECOVERY   = 5
DT         = 0.03
MAX_NORM   = 3.0
INIT_STEPS = 20

# P-MNIST config
N_TASKS       = 2
N_TRAIN_TASK  = 6000
N_TEST_TASK   = 10000
N_CLASSES     = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED          = 42


# ─── Eigenform math (rk.py) ──────────────────────────────────────────────────

def phi(M):
    linear = mscale(M, ALPHA)
    quad   = mscale(mmul(M, M), BETA / K)
    return mtanh(madd(linear, quad))


def cross_apply(Mi, Mj):
    avg  = mscale(madd(Mi, Mj), ALPHA / 2.0)
    prod = mscale(mmul(Mi, Mj), BETA / K)
    return mtanh(madd(avg, prod))


def eigenform_steps(M, n=INIT_STEPS):
    for _ in range(n):
        M = madd(M, mscale(msub(phi(M), M), DT))
        M = mclip(M, MAX_NORM)
    return M


def delta(M, R):
    return frob(msub(cross_apply(M, R), M))


# ─── Projection ──────────────────────────────────────────────────────────────

def make_proj1(seed=12345):
    """R^784 -> R^384, frozen Gaussian."""
    rng = np.random.RandomState(seed)
    return (rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784))


def make_proj2(seed=99999, scale=10.0):
    """R^384 -> R^16, frozen Gaussian. Scale=10 gives frob(R) ~2.0 (eigenform range)."""
    rng = np.random.RandomState(seed)
    return (rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID) * scale)


def to_matrix(vec384, P2):
    """vec384: (384,) numpy float32 -> 4x4 Python list."""
    flat = (P2 @ vec384).tolist()  # (16,)
    return [flat[i * K:(i + 1) * K] for i in range(K)]


def embed_batch(X_flat, perm, P1, P2):
    """
    X_flat: (n, 784) numpy
    Returns list of 4x4 Python matrices (one per sample).
    """
    X_perm = X_flat[:, perm]             # (n, 784)
    mid    = X_perm @ P1.T              # (n, 384)
    norms  = np.linalg.norm(mid, axis=1, keepdims=True) + 1e-15
    mid    = (mid / norms).astype(np.float32)
    mats   = [to_matrix(mid[i], P2) for i in range(len(mid))]
    return mats


# ─── EigenFold codebook ──────────────────────────────────────────────────────

class EigenFold:
    def __init__(self, threshold):
        self.threshold = threshold
        self.elements  = []   # list of (M, label)

    def classify(self, R):
        if not self.elements:
            return None
        best_i = min(range(len(self.elements)), key=lambda i: delta(self.elements[i][0], R))
        return self.elements[best_i][1]

    def train(self, R, label):
        if not self.elements:
            self.elements.append((eigenform_steps(R), label))
            return
        deltas  = [delta(M, R) for M, _ in self.elements]
        min_d   = min(deltas)
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        if min_d > self.threshold:
            self.elements.append((eigenform_steps(R), label))
        else:
            M_w, lbl_w = self.elements[win_idx]
            dv  = msub(cross_apply(M_w, R), M_w)
            M_w = madd(M_w, mscale(dv, LAM))
            for _ in range(RECOVERY):
                M_w = madd(M_w, mscale(msub(phi(M_w), M_w), DT))
            M_w = mclip(M_w, MAX_NORM)
            self.elements[win_idx] = (M_w, lbl_w)


# ─── MNIST loading ───────────────────────────────────────────────────────────

def load_mnist():
    import torchvision
    train_ds = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    test_ds  = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = train_ds.targets.numpy()
    X_te = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = test_ds.targets.numpy()
    return X_tr, y_tr, X_te, y_te


def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        cls_idx = np.where(y == c)[0]
        chosen  = rng.choice(cls_idx, n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


def make_permutation(seed):
    perm = list(range(784))
    rng  = random.Random(seed)
    rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)


# ─── Run one threshold config ─────────────────────────────────────────────────

def run_threshold(threshold, X_tr, y_tr, X_te, y_te, P1, P2, perms):
    ef = EigenFold(threshold)
    acc_matrix = [[None] * N_TASKS for _ in range(N_TASKS)]

    for task_id in range(N_TASKS):
        perm = perms[task_id]
        t0   = time.time()

        # Train
        Xtr_t, ytr_t = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS, seed=task_id * 100)
        mats_tr = embed_batch(Xtr_t, perm, P1, P2)
        for R, lbl in zip(mats_tr, ytr_t.tolist()):
            ef.train(R, lbl)
        t_train = time.time() - t0

        # Evaluate all tasks seen so far
        for et in range(task_id + 1):
            perm_et = perms[et]
            mats_te = embed_batch(X_te, perm_et, P1, P2)
            correct = sum(1 for R, lbl in zip(mats_te, y_te.tolist())
                         if ef.classify(R) == lbl)
            acc_matrix[et][task_id] = correct / len(y_te)

        cur_aa = sum(acc_matrix[et][task_id] for et in range(task_id + 1)) / (task_id + 1)
        print(f"    task {task_id+1}/{N_TASKS}  AA={cur_aa*100:.1f}%  CB={len(ef.elements)}  train={t_train:.0f}s",
              flush=True)

    final  = [acc_matrix[i][N_TASKS - 1] for i in range(N_TASKS)]
    aa     = sum(final) / N_TASKS
    fgts   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS - 1]) for i in range(N_TASKS - 1)]
    avg_f  = sum(fgts) / len(fgts) if fgts else 0.0
    return aa, avg_f, len(ef.elements)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_total = time.time()
    print("=" * 65)
    print("  EigenFold Step 75 — P-MNIST 2 tasks")
    print(f"  k={K} d={D_MID}->{D_MAT} alpha={ALPHA} beta={BETA} lam={LAM}")
    print(f"  {N_TASKS} tasks x {N_TRAIN_TASK} train / {N_TEST_TASK} test")
    print("=" * 65)

    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P1   = make_proj1()
    P2   = make_proj2()
    perms = [make_permutation(seed=i * 7 + 1) for i in range(N_TASKS)]
    print(f"  Ready. Starting threshold sweep.", flush=True)

    THRESHOLDS = [0.05, 0.10, 0.20]
    results = []

    for thr in THRESHOLDS:
        print(f"\n  threshold={thr}")
        aa, fgt, cb = run_threshold(thr, X_tr, y_tr, X_te, y_te, P1, P2, perms)
        results.append((thr, aa, fgt, cb))
        print(f"    -> AA={aa*100:.1f}%  Fgt={fgt*100:.1f}pp  CB={cb}")

        # Stop early if behavior is clearly degenerate
        if cb > 500:
            print("    (codebook runaway — skipping remaining thresholds)")
            break

    elapsed = time.time() - t_total
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  {'Threshold':<12} {'AA':<10} {'Forgetting':<14} {'CB':<8}")
    print(f"  {'-'*10}   {'-'*7}   {'-'*11}   {'-'*6}")
    for thr, aa, fgt, cb in results:
        print(f"  {thr:<12} {aa*100:<10.1f} {fgt*100:<14.1f} {cb:<8}")
    print()
    print(f"  Total elapsed: {elapsed:.0f}s")
    print()
    print(f"  Baseline (FluxCore v17, P-MNIST 10 tasks): AA=84.1% Fgt=11.4pp")
    print("=" * 65)


if __name__ == '__main__':
    main()
