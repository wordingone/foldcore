#!/usr/bin/env python3
"""
Step 75b — EigenFold P-MNIST, calibrated threshold.

[researcher] directive (mail 1179):
1. Delta diagnostic on 200 samples (10x-scaled P2)
2. Pick threshold at p90 of delta distribution
3. Run 2 tasks with that threshold
"""
import sys, random, math, time
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip

K=4; D_MID=384; D_MAT=16; ALPHA=1.2; BETA=0.8
LAM=0.1; RECOVERY=5; DT=0.03; MAX_NORM=3.0; INIT_STEPS=20
N_TASKS=2; N_TRAIN_TASK=6000; N_TEST_TASK=10000; N_CLASSES=10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED=42


def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA/K)))

def cross_apply(Mi, Mj):
    return mtanh(madd(mscale(madd(Mi, Mj), ALPHA/2), mscale(mmul(Mi, Mj), BETA/K)))

def eigenform_steps(M, n=INIT_STEPS):
    for _ in range(n):
        M = madd(M, mscale(msub(phi(M), M), DT))
        M = mclip(M, MAX_NORM)
    return M

def delta(M, R):
    return frob(msub(cross_apply(M, R), M))

def make_proj1(seed=12345):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784)

def make_proj2(seed=99999, scale=10.0):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID) * scale

def to_matrix(vec384, P2):
    flat = (P2 @ vec384).tolist()
    return [flat[i*K:(i+1)*K] for i in range(K)]

def embed_batch(X_flat, perm, P1, P2):
    X_perm = X_flat[:, perm]
    mid = X_perm @ P1.T
    norms = np.linalg.norm(mid, axis=1, keepdims=True) + 1e-15
    mid = (mid / norms).astype(np.float32)
    return [to_matrix(mid[i], P2) for i in range(len(mid))]

def make_permutation(seed):
    perm = list(range(784)); rng = random.Random(seed); rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_tr = tr.targets.numpy()
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te = te.targets.numpy()
    return X_tr, y_tr, X_te, y_te

def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


class EigenFold:
    def __init__(self, threshold):
        self.threshold = threshold
        self.elements  = []

    def classify(self, R):
        if not self.elements:
            return None
        return self.elements[min(range(len(self.elements)),
                                  key=lambda i: delta(self.elements[i][0], R))][1]

    def train(self, R, label):
        if not self.elements:
            self.elements.append((eigenform_steps(R), label))
            return
        deltas  = [delta(M, R) for M, _ in self.elements]
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        if min(deltas) > self.threshold:
            self.elements.append((eigenform_steps(R), label))
        else:
            M_w, lbl_w = self.elements[win_idx]
            dv  = msub(cross_apply(M_w, R), M_w)
            M_w = madd(M_w, mscale(dv, LAM))
            for _ in range(RECOVERY):
                M_w = madd(M_w, mscale(msub(phi(M_w), M_w), DT))
            M_w = mclip(M_w, MAX_NORM)
            self.elements[win_idx] = (M_w, lbl_w)


def run_one(threshold, train_mats_per_task, test_mats, y_te):
    ef = EigenFold(threshold)
    acc_matrix = [[None]*N_TASKS for _ in range(N_TASKS)]
    for task_id, (mats_tr, labels_tr) in enumerate(train_mats_per_task):
        t0 = time.time()
        for R, lbl in zip(mats_tr, labels_tr):
            ef.train(R, lbl)
            if len(ef.elements) > 2000:   # hard cap — runaway guard
                break
        t_tr = time.time() - t0
        for et in range(task_id + 1):
            correct = sum(1 for R, lbl in zip(test_mats[et], y_te.tolist())
                         if ef.classify(R) == lbl)
            acc_matrix[et][task_id] = correct / len(y_te)
        aa = sum(acc_matrix[et][task_id] for et in range(task_id+1)) / (task_id+1)
        print(f"    task {task_id+1}  AA={aa*100:.1f}%  CB={len(ef.elements)}  train={t_tr:.0f}s",
              flush=True)
    final  = [acc_matrix[i][N_TASKS-1] for i in range(N_TASKS)]
    fgts   = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS-1]) for i in range(N_TASKS-1)]
    return sum(final)/N_TASKS, sum(fgts)/len(fgts) if fgts else 0.0, len(ef.elements)


def main():
    t0 = time.time()
    print("EigenFold Step 75b — calibrated threshold, P-MNIST 2 tasks", flush=True)
    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P1 = make_proj1(); P2 = make_proj2(scale=10.0)
    perms = [make_permutation(seed=i*7+1) for i in range(N_TASKS)]

    # ── Diagnostic: delta distribution on 200 samples ────────────────────────
    print("\nDiagnostic: delta distribution (200 samples, codebook=[first 20])...", flush=True)
    X_diag, y_diag = stratified_sample(X_tr, y_tr, 2, seed=777)   # 20 samples, 2/class
    mats_diag = embed_batch(X_diag, perms[0], P1, P2)
    # Build a small seed codebook from first 20, then measure deltas for next 180
    X_main, _ = stratified_sample(X_tr, y_tr, 20, seed=888)       # 200 samples, 20/class
    mats_main = embed_batch(X_main, perms[0], P1, P2)
    # Seed codebook: eigenform-init first 20
    seed_cb = [eigenform_steps(R) for R in mats_diag[:20]]
    all_deltas = []
    for R in mats_main:
        d_min = min(delta(M, R) for M in seed_cb)
        all_deltas.append(d_min)
    all_deltas.sort()
    n = len(all_deltas)
    p25 = all_deltas[n//4]; p50 = all_deltas[n//2]
    p75 = all_deltas[3*n//4]; p90 = all_deltas[int(0.9*n)]
    print(f"  min={all_deltas[0]:.4f}  p25={p25:.4f}  p50={p50:.4f}  "
          f"p75={p75:.4f}  p90={p90:.4f}  max={all_deltas[-1]:.4f}")
    threshold = p90
    print(f"  -> Using threshold={threshold:.4f} (p90)", flush=True)

    # ── Pre-embed all data ────────────────────────────────────────────────────
    print("\nEmbedding all data...", flush=True)
    train_mats = []
    for task_id in range(N_TASKS):
        Xtr_t, ytr_t = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS, seed=task_id*100)
        mats = embed_batch(Xtr_t, perms[task_id], P1, P2)
        train_mats.append((mats, ytr_t.tolist()))
    test_mats = [embed_batch(X_te, perms[t], P1, P2) for t in range(N_TASKS)]
    print(f"  Done. Running EigenFold...", flush=True)

    # ── Main run ─────────────────────────────────────────────────────────────
    print(f"\n  threshold={threshold:.4f}")
    aa, fgt, cb = run_one(threshold, train_mats, test_mats, y_te)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print("  RESULTS — Step 75b")
    print("=" * 60)
    print(f"  Threshold (p90):  {threshold:.4f}")
    print(f"  Final AA:         {aa*100:.1f}%")
    print(f"  Avg forgetting:   {fgt*100:.1f}pp")
    print(f"  Codebook size:    {cb}")
    print(f"  Total elapsed:    {elapsed:.0f}s")
    print()
    print(f"  Baseline k-NN (CB=6000): AA~67.9%  Fgt=?")
    print("=" * 60)


if __name__ == '__main__':
    main()
