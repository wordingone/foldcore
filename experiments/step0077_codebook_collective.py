#!/usr/bin/env python3
"""
Step 77 — Collective classification test.

Spec: does coupling between elements add value?
System A: EigenFold + coupling (collective inference)
System B: Vector cosine baseline (from Step 76)
System C: EigenFold no coupling (ablation, same as Step 76 System A)

Coupling (inference only, not training):
  M_i' = M_i + 0.1 * sum_j w_ij * (Psi(M_i, M_j) - M_i)
  where j are the 3 nearest neighbors by mcosine, uniform weights.
  Classify with delta_i' = frob(Psi(M_i', R) - M_i').
"""
import sys, random, math, time
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip, mcosine

K=4; D_MID=384; D_MAT=16; ALPHA=1.2; BETA=0.8
LAM_EF=0.1; RECOVERY=5; DT=0.03; MAX_NORM=3.0; INIT_STEPS=20
N_TASKS=2; N_TRAIN_TASK=6000; N_TEST_TASK=10000; N_CLASSES=10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED=42; PROJ2_SCALE=10.0
COUPLING_K=3; COUPLING_LR=0.1
LAM_VEC=0.015


# ─── Eigenform math ──────────────────────────────────────────────────────────

def phi(M):
    return mtanh(madd(mscale(M, ALPHA), mscale(mmul(M, M), BETA/K)))

def cross_apply(Mi, Mj):
    return mtanh(madd(mscale(madd(Mi, Mj), ALPHA/2), mscale(mmul(Mi, Mj), BETA/K)))

def eigenform_steps(M, n=INIT_STEPS):
    for _ in range(n):
        M = madd(M, mscale(msub(phi(M), M), DT))
        M = mclip(M, MAX_NORM)
    return M

def ef_delta(M, R):
    return frob(msub(cross_apply(M, R), M))


# ─── Coupling step (inference-time) ──────────────────────────────────────────

def coupled_classify(elements, R):
    """One coupling step across all elements, then classify vs R."""
    if not elements: return None
    n = len(elements)
    mats = [M for M, _ in elements]

    # Skip coupling for tiny codebooks
    if n <= 1:
        return elements[min(range(n), key=lambda i: ef_delta(mats[i], R))][1]

    # Compute cosines between all pairs
    k_nn = min(COUPLING_K, n - 1)
    coupled = []
    for i in range(n):
        # Find k nearest neighbors (by mcosine, excluding self)
        cos_scores = [(mcosine(mats[i], mats[j]), j) for j in range(n) if j != i]
        cos_scores.sort(reverse=True)
        neighbors = [j for _, j in cos_scores[:k_nn]]

        # Uniform weights
        w = 1.0 / k_nn
        coupling_sum = mscale(msub(cross_apply(mats[i], mats[neighbors[0]]), mats[i]), w)
        for j in neighbors[1:]:
            coupling_sum = madd(coupling_sum,
                                mscale(msub(cross_apply(mats[i], mats[j]), mats[i]), w))

        M_prime = madd(mats[i], mscale(coupling_sum, COUPLING_LR))
        coupled.append(M_prime)

    # Classify using coupled state
    best = min(range(n), key=lambda i: ef_delta(coupled[i], R))
    return elements[best][1]


# ─── Projections & data loading ───────────────────────────────────────────────

def make_proj1(seed=12345):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784)

def make_proj2(seed=99999):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID) * PROJ2_SCALE

def to_matrix(vec16):
    return [vec16[i*K:(i+1)*K].tolist() for i in range(K)]

def make_permutation(seed):
    perm = list(range(784)); rng = random.Random(seed); rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)

def embed_all(X_flat, perm, P1, P2):
    X_perm = X_flat[:, perm]
    mid = X_perm @ P1.T
    norms = np.linalg.norm(mid, axis=1, keepdims=True) + 1e-15
    mid = (mid / norms).astype(np.float32)
    proj16 = (mid @ P2.T)
    proj16_u = proj16 / (np.linalg.norm(proj16, axis=1, keepdims=True) + 1e-15)
    mats = [to_matrix(proj16[i]) for i in range(len(proj16))]
    vecs = [proj16_u[i].tolist() for i in range(len(proj16))]
    return mats, vecs

def load_mnist():
    import torchvision
    tr = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=True,  download=True)
    te = torchvision.datasets.MNIST('C:/Users/Admin/mnist_data', train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()

def stratified_sample(X, y, n_per_class, seed):
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(N_CLASSES):
        chosen = rng.choice(np.where(y == c)[0], n_per_class, replace=False)
        idx.extend(chosen.tolist())
    rng.shuffle(idx)
    return X[idx], y[idx]


# ─── EigenFold codebook (shared for A and C) ─────────────────────────────────

class EigenFold:
    def __init__(self, threshold):
        self.threshold = threshold
        self.elements  = []

    def classify_nocoupling(self, R):
        if not self.elements: return None
        return self.elements[min(range(len(self.elements)),
                                  key=lambda i: ef_delta(self.elements[i][0], R))][1]

    def classify_coupled(self, R):
        return coupled_classify(self.elements, R)

    def train(self, R, label):
        if not self.elements:
            self.elements.append((eigenform_steps(R), label)); return
        deltas  = [ef_delta(M, R) for M, _ in self.elements]
        win_idx = min(range(len(deltas)), key=lambda i: deltas[i])
        if deltas[win_idx] > self.threshold:
            self.elements.append((eigenform_steps(R), label))
        else:
            M_w, lbl_w = self.elements[win_idx]
            dv  = msub(cross_apply(M_w, R), M_w)
            M_w = madd(M_w, mscale(dv, LAM_EF))
            for _ in range(RECOVERY):
                M_w = madd(M_w, mscale(msub(phi(M_w), M_w), DT))
            M_w = mclip(M_w, MAX_NORM)
            self.elements[win_idx] = (M_w, lbl_w)


# ─── Vector codebook ─────────────────────────────────────────────────────────

class VectorCodebook:
    def __init__(self, threshold):
        self.threshold = threshold
        self.vecs      = []
        self.vecs_np   = None

    def _rebuild(self):
        self.vecs_np = np.array([v for v,_ in self.vecs], dtype=np.float32)

    def classify(self, r):
        if not self.vecs: return None
        if self.vecs_np is None: self._rebuild()
        r_np = np.array(r, dtype=np.float32)
        return self.vecs[int(np.argmax(self.vecs_np @ r_np))][1]

    def train(self, r, label):
        if not self.vecs:
            self.vecs.append((list(r), label)); self.vecs_np = None; return
        if self.vecs_np is None: self._rebuild()
        r_np  = np.array(r, dtype=np.float32)
        sims  = self.vecs_np @ r_np
        win   = int(np.argmax(sims))
        if float(sims[win]) < self.threshold:
            self.vecs.append((list(r), label)); self.vecs_np = None
        else:
            v_w, lbl_w = self.vecs[win]
            v_new = [v + LAM_VEC * ri for v, ri in zip(v_w, r)]
            n = math.sqrt(sum(x*x for x in v_new) + 1e-15)
            v_new = [x/n for x in v_new]
            self.vecs[win] = (v_new, lbl_w)
            self.vecs_np[win] = np.array(v_new, dtype=np.float32)


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate(seed_mats, seed_vecs, probe_mats, probe_vecs):
    seed_cb = [eigenform_steps(R) for R in seed_mats]
    ef_dels = sorted([min(ef_delta(M, R) for M in seed_cb) for R in probe_mats])
    seed_vecs_np = np.array(seed_vecs, dtype=np.float32)
    cos_vals = sorted([float(np.max(seed_vecs_np @ np.array(r, dtype=np.float32)))
                       for r in probe_vecs])
    n = len(ef_dels)
    thr_ef  = ef_dels[int(0.90 * n)]
    thr_vec = cos_vals[int(0.10 * n)]
    print(f"  EigenFold delta: p25={ef_dels[n//4]:.3f} p50={ef_dels[n//2]:.3f} p90={thr_ef:.3f}")
    print(f"  Vector cosine:   p10={thr_vec:.3f} p50={cos_vals[n//2]:.3f} p90={cos_vals[int(0.9*n)]:.3f}")
    return thr_ef, thr_vec


# ─── Run one system ───────────────────────────────────────────────────────────

def run_system(name, classify_fn, train_fn, cb_size_fn,
               train_data, test_data, y_te):
    acc_matrix = [[None]*N_TASKS for _ in range(N_TASKS)]
    for task_id, (inputs_tr, labels_tr) in enumerate(train_data):
        t0 = time.time()
        for inp, lbl in zip(inputs_tr, labels_tr):
            train_fn(inp, lbl)
        t_tr = time.time() - t0
        for et in range(task_id + 1):
            correct = sum(1 for inp, lbl in zip(test_data[et], y_te.tolist())
                         if classify_fn(inp) == lbl)
            acc_matrix[et][task_id] = correct / len(y_te)
        aa = sum(acc_matrix[et][task_id] for et in range(task_id+1)) / (task_id+1)
        print(f"    [{name}] task {task_id+1}  AA={aa*100:.1f}%  CB={cb_size_fn()}  train={t_tr:.0f}s",
              flush=True)
    final = [acc_matrix[i][N_TASKS-1] for i in range(N_TASKS)]
    fgts  = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS-1]) for i in range(N_TASKS-1)]
    return sum(final)/N_TASKS, sum(fgts)/len(fgts) if fgts else 0.0, cb_size_fn()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 77 — Collective classification test", flush=True)
    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P1 = make_proj1(); P2 = make_proj2()
    perms = [make_permutation(seed=i*7+1) for i in range(N_TASKS)]

    print("Embedding...", flush=True)
    train_ef = []; train_vec = []
    for t in range(N_TASKS):
        Xtr_t, ytr_t = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS, seed=t*100)
        mats, vecs = embed_all(Xtr_t, perms[t], P1, P2)
        train_ef.append((mats, ytr_t.tolist()))
        train_vec.append((vecs, ytr_t.tolist()))
    test_ef = []; test_vec = []
    for t in range(N_TASKS):
        mats, vecs = embed_all(X_te, perms[t], P1, P2)
        test_ef.append(mats); test_vec.append(vecs)

    print("Calibrating...", flush=True)
    X_seed, _ = stratified_sample(X_tr, y_tr, 2,  seed=777)
    X_prob, _ = stratified_sample(X_tr, y_tr, 20, seed=888)
    sm, sv = embed_all(X_seed, perms[0], P1, P2)
    pm, pv = embed_all(X_prob, perms[0], P1, P2)
    thr_ef, thr_vec = calibrate(sm, sv, pm, pv)
    print(f"  Thresholds: EigenFold={thr_ef:.4f}  Vector={thr_vec:.4f}", flush=True)

    # Single trained EigenFold codebook — reused for both A and C
    print("\nTraining EigenFold codebook (used for A and C)...", flush=True)
    ef = EigenFold(thr_ef)
    t_train = time.time()
    for mats, labels in train_ef:
        for R, lbl in zip(mats, labels):
            ef.train(R, lbl)
    t_train = time.time() - t_train
    print(f"  Codebook trained: CB={len(ef.elements)} in {t_train:.0f}s", flush=True)

    # System C: no coupling (re-run evaluation only)
    print("\nSystem C: EigenFold no coupling...", flush=True)
    acc_c = [[None]*N_TASKS for _ in range(N_TASKS)]
    # Need per-task trained states — retrain fresh for proper per-task eval
    ef_c = EigenFold(thr_ef)
    results_c = []
    for task_id, (mats_tr, labels_tr) in enumerate(train_ef):
        for R, lbl in zip(mats_tr, labels_tr):
            ef_c.train(R, lbl)
        for et in range(task_id + 1):
            correct = sum(1 for R, lbl in zip(test_ef[et], y_te.tolist())
                         if ef_c.classify_nocoupling(R) == lbl)
            acc_c[et][task_id] = correct / len(y_te)
        aa = sum(acc_c[et][task_id] for et in range(task_id+1)) / (task_id+1)
        print(f"    [EF no-coupling] task {task_id+1}  AA={aa*100:.1f}%  CB={len(ef_c.elements)}",
              flush=True)
    final_c = [acc_c[i][N_TASKS-1] for i in range(N_TASKS)]
    fgt_c = max(0., acc_c[0][0] - acc_c[0][N_TASKS-1])
    aa_c = sum(final_c)/N_TASKS; cb_c = len(ef_c.elements)

    # System A: with coupling (same trained codebook as C)
    print("\nSystem A: EigenFold + coupling...", flush=True)
    ef_a = EigenFold(thr_ef)
    acc_a = [[None]*N_TASKS for _ in range(N_TASKS)]
    for task_id, (mats_tr, labels_tr) in enumerate(train_ef):
        t_task = time.time()
        for R, lbl in zip(mats_tr, labels_tr):
            ef_a.train(R, lbl)
        t_tr = time.time() - t_task
        for et in range(task_id + 1):
            correct = sum(1 for R, lbl in zip(test_ef[et], y_te.tolist())
                         if ef_a.classify_coupled(R) == lbl)
            acc_a[et][task_id] = correct / len(y_te)
        aa = sum(acc_a[et][task_id] for et in range(task_id+1)) / (task_id+1)
        print(f"    [EF+coupling] task {task_id+1}  AA={aa*100:.1f}%  CB={len(ef_a.elements)}  train={t_tr:.0f}s",
              flush=True)
    final_a = [acc_a[i][N_TASKS-1] for i in range(N_TASKS)]
    fgt_a = max(0., acc_a[0][0] - acc_a[0][N_TASKS-1])
    aa_a = sum(final_a)/N_TASKS; cb_a = len(ef_a.elements)

    # System B: vector cosine baseline
    print("\nSystem B: Vector cosine baseline...", flush=True)
    vc = VectorCodebook(thr_vec)
    acc_b = [[None]*N_TASKS for _ in range(N_TASKS)]
    for task_id, (vecs_tr, labels_tr) in enumerate(train_vec):
        t_task = time.time()
        for v, lbl in zip(vecs_tr, labels_tr):
            vc.train(v, lbl)
        t_tr = time.time() - t_task
        for et in range(task_id + 1):
            correct = sum(1 for v, lbl in zip(test_vec[et], y_te.tolist())
                         if vc.classify(v) == lbl)
            acc_b[et][task_id] = correct / len(y_te)
        aa = sum(acc_b[et][task_id] for et in range(task_id+1)) / (task_id+1)
        print(f"    [Vector] task {task_id+1}  AA={aa*100:.1f}%  CB={len(vc.vecs)}  train={t_tr:.0f}s",
              flush=True)
    final_b = [acc_b[i][N_TASKS-1] for i in range(N_TASKS)]
    fgt_b = max(0., acc_b[0][0] - acc_b[0][N_TASKS-1])
    aa_b = sum(final_b)/N_TASKS; cb_b = len(vc.vecs)

    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print("  RESULTS — Step 77")
    print("=" * 65)
    print(f"  {'System':<28} {'AA':<9} {'Fgt':<10} {'CB':<8}")
    print(f"  {'-'*26}   {'-'*6}   {'-'*7}   {'-'*6}")
    print(f"  {'C: EigenFold (no coupling)':<28} {aa_c*100:<9.1f} {fgt_c*100:<10.1f} {cb_c}")
    print(f"  {'A: EigenFold + coupling':<28} {aa_a*100:<9.1f} {fgt_a*100:<10.1f} {cb_a}")
    print(f"  {'B: Vector cosine':<28} {aa_b*100:<9.1f} {fgt_b*100:<10.1f} {cb_b}")
    print()
    da_c = aa_a - aa_c
    da_b = aa_a - aa_b
    print(f"  A vs C (coupling benefit):  {da_c*100:+.1f}pp")
    print(f"  A vs B (matrix vs vector):  {da_b*100:+.1f}pp")
    verdict = ("A > B: matrices beat vectors via collective dynamics"
               if da_b > 0.02 else
               "A ~ B: coupling doesn't overcome vector advantage" if abs(da_b) <= 0.02
               else "B wins: collective dynamics insufficient")
    print(f"  Verdict: {verdict}")
    print(f"  Total elapsed: {elapsed:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
