#!/usr/bin/env python3
"""
Step 76 — EigenFold vs cosine nearest-prototype head-to-head.

[researcher] directive (mail 1181): same data, same d=16 features, same CB budget.
System A: EigenFold matrix codebook (4x4, frob perturbation).
System B: Vector codebook (16-dim, cosine similarity).
Both spawn at ~same rate. Report AA, forgetting, CB, time.
"""
import sys, random, math, time
import numpy as np

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import madd, msub, mscale, mmul, mtanh, frob, mclip

K=4; D_MID=384; D_MAT=16; ALPHA=1.2; BETA=0.8
LAM_EF=0.1; RECOVERY=5; DT=0.03; MAX_NORM=3.0; INIT_STEPS=20
N_TASKS=2; N_TRAIN_TASK=6000; N_TEST_TASK=10000; N_CLASSES=10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES
SEED=42
PROJ2_SCALE=10.0
DIAG_N = 200  # samples for threshold calibration


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


# ─── Projections & embedding ─────────────────────────────────────────────────

def make_proj1(seed=12345):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784)

def make_proj2(seed=99999):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID) * PROJ2_SCALE

def to_matrix(vec16):
    """(16,) numpy -> 4x4 Python list."""
    return [vec16[i*K:(i+1)*K].tolist() for i in range(K)]

def make_permutation(seed):
    perm = list(range(784)); rng = random.Random(seed); rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)

def embed_all(X_flat, perm, P1, P2):
    """Returns (mats_4x4, vecs_16) for each sample."""
    X_perm = X_flat[:, perm]
    mid = X_perm @ P1.T
    norms = np.linalg.norm(mid, axis=1, keepdims=True) + 1e-15
    mid = (mid / norms).astype(np.float32)
    proj16 = (mid @ P2.T)                        # (n, 16)
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


# ─── System A: EigenFold ──────────────────────────────────────────────────────

class EigenFold:
    def __init__(self, threshold):
        self.threshold = threshold
        self.elements  = []

    def classify(self, R):
        if not self.elements: return None
        return self.elements[min(range(len(self.elements)),
                                  key=lambda i: ef_delta(self.elements[i][0], R))][1]

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


# ─── System B: Vector codebook ───────────────────────────────────────────────

LAM_VEC = 0.015

class VectorCodebook:
    def __init__(self, threshold):
        self.threshold = threshold   # spawn if max_cosine < threshold
        self.vecs   = []             # list of (v: list[16], label)
        self.vecs_np = None          # cached numpy (N, 16) for fast classify

    def _cosine(self, v, w):
        return sum(a*b for a,b in zip(v,w))

    def _rebuild(self):
        self.vecs_np = np.array([v for v,_ in self.vecs], dtype=np.float32)

    def classify(self, r):
        if not self.vecs: return None
        if self.vecs_np is None: self._rebuild()
        r_np = np.array(r, dtype=np.float32)
        sims = self.vecs_np @ r_np
        return self.vecs[int(np.argmax(sims))][1]

    def train(self, r, label):
        if not self.vecs:
            self.vecs.append((list(r), label))
            self.vecs_np = None; return
        if self.vecs_np is None: self._rebuild()
        r_np = np.array(r, dtype=np.float32)
        sims = self.vecs_np @ r_np
        win_idx = int(np.argmax(sims))
        max_cos  = float(sims[win_idx])
        if max_cos < self.threshold:
            self.vecs.append((list(r), label))
            self.vecs_np = None
        else:
            v_w, lbl_w = self.vecs[win_idx]
            v_new = [v + LAM_VEC * ri for v, ri in zip(v_w, r)]
            n = math.sqrt(sum(x*x for x in v_new) + 1e-15)
            v_new = [x/n for x in v_new]
            self.vecs[win_idx] = (v_new, lbl_w)
            self.vecs_np[win_idx] = np.array(v_new, dtype=np.float32)


# ─── Calibration ─────────────────────────────────────────────────────────────

def calibrate_ef(seed_mats, probe_mats, target_pct=0.90):
    """p90 of min-delta against seed codebook."""
    seed_cb = [eigenform_steps(R) for R in seed_mats]
    deltas = [min(ef_delta(M, R) for M in seed_cb) for R in probe_mats]
    deltas.sort()
    n = len(deltas)
    print(f"  EigenFold delta: min={deltas[0]:.3f} p25={deltas[n//4]:.3f} "
          f"p50={deltas[n//2]:.3f} p90={deltas[int(0.9*n)]:.3f} max={deltas[-1]:.3f}")
    return deltas[int(target_pct * n)]

def calibrate_vec(seed_vecs_np, probe_vecs, target_pct=0.10):
    """p10 of max-cosine against seed codebook (spawn if max_cos < thr)."""
    sims_per_probe = [float(np.max(seed_vecs_np @ np.array(r, dtype=np.float32)))
                      for r in probe_vecs]
    sims_per_probe.sort()
    n = len(sims_per_probe)
    print(f"  Vector cosine:   min={sims_per_probe[0]:.3f} p10={sims_per_probe[int(0.1*n)]:.3f} "
          f"p50={sims_per_probe[n//2]:.3f} p90={sims_per_probe[int(0.9*n)]:.3f} max={sims_per_probe[-1]:.3f}")
    return sims_per_probe[int(target_pct * n)]


# ─── Benchmark runner ─────────────────────────────────────────────────────────

def run_system(system, train_data, test_data, y_te, name):
    acc_matrix = [[None]*N_TASKS for _ in range(N_TASKS)]
    for task_id, (inputs_tr, labels_tr) in enumerate(train_data):
        t0 = time.time()
        for inp, lbl in zip(inputs_tr, labels_tr):
            system.train(inp, lbl)
        t_tr = time.time() - t0
        for et in range(task_id + 1):
            correct = sum(1 for inp, lbl in zip(test_data[et], y_te.tolist())
                         if system.classify(inp) == lbl)
            acc_matrix[et][task_id] = correct / len(y_te)
        aa = sum(acc_matrix[et][task_id] for et in range(task_id+1)) / (task_id+1)
        cb = len(system.elements if hasattr(system, 'elements') else system.vecs)
        print(f"    [{name}] task {task_id+1}  AA={aa*100:.1f}%  CB={cb}  train={t_tr:.0f}s",
              flush=True)
    final = [acc_matrix[i][N_TASKS-1] for i in range(N_TASKS)]
    fgts  = [max(0., acc_matrix[i][i] - acc_matrix[i][N_TASKS-1]) for i in range(N_TASKS-1)]
    cb    = len(system.elements if hasattr(system, 'elements') else system.vecs)
    return sum(final)/N_TASKS, sum(fgts)/len(fgts) if fgts else 0.0, cb


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Step 76 — EigenFold vs cosine nearest-prototype", flush=True)
    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    P1 = make_proj1(); P2 = make_proj2()
    perms = [make_permutation(seed=i*7+1) for i in range(N_TASKS)]

    # ── Embed all data ─────────────────────────────────────────────────────
    print("Embedding all data...", flush=True)
    train_data_ef  = []   # list of (mats, labels)
    train_data_vec = []   # list of (vecs, labels)
    for t in range(N_TASKS):
        Xtr_t, ytr_t = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS, seed=t*100)
        mats, vecs = embed_all(Xtr_t, perms[t], P1, P2)
        train_data_ef.append((mats, ytr_t.tolist()))
        train_data_vec.append((vecs, ytr_t.tolist()))

    test_ef  = []; test_vec = []
    for t in range(N_TASKS):
        mats, vecs = embed_all(X_te, perms[t], P1, P2)
        test_ef.append(mats); test_vec.append(vecs)

    # ── Calibrate thresholds ───────────────────────────────────────────────
    print("\nCalibrating thresholds (200 samples each)...", flush=True)
    # Seed: 20 samples. Probe: 200 samples.
    X_seed, _ = stratified_sample(X_tr, y_tr, 2,  seed=777)  # 20 samples
    X_prob, _ = stratified_sample(X_tr, y_tr, 20, seed=888)  # 200 samples
    seed_mats, seed_vecs = embed_all(X_seed, perms[0], P1, P2)
    prob_mats, prob_vecs = embed_all(X_prob, perms[0], P1, P2)
    seed_vecs_np = np.array(seed_vecs, dtype=np.float32)

    thr_ef  = calibrate_ef(seed_mats, prob_mats, target_pct=0.90)
    thr_vec = calibrate_vec(seed_vecs_np, prob_vecs, target_pct=0.10)
    print(f"  EigenFold threshold: {thr_ef:.4f}")
    print(f"  Vector threshold:    {thr_vec:.4f}")

    # ── Run both systems ───────────────────────────────────────────────────
    print("\nRunning System A — EigenFold...", flush=True)
    t_ef = time.time()
    ef = EigenFold(thr_ef)
    aa_ef, fgt_ef, cb_ef = run_system(ef, train_data_ef, test_ef, y_te, "EigenFold")
    t_ef = time.time() - t_ef

    print("\nRunning System B — Vector codebook...", flush=True)
    t_vec = time.time()
    vc = VectorCodebook(thr_vec)
    aa_vc, fgt_vc, cb_vc = run_system(vc, train_data_vec, test_vec, y_te, "Vector")
    t_vec = time.time() - t_vec

    elapsed = time.time() - t0
    print()
    print("=" * 65)
    print("  RESULTS — Step 76")
    print("=" * 65)
    print(f"  {'System':<22} {'AA':<10} {'Forgetting':<14} {'CB':<8} {'Time'}")
    print(f"  {'-'*20}   {'-'*7}   {'-'*11}   {'-'*6}   {'-'*6}")
    print(f"  {'A: EigenFold':<22} {aa_ef*100:<10.1f} {fgt_ef*100:<14.1f} {cb_ef:<8} {t_ef:.0f}s")
    print(f"  {'B: Vector cosine':<22} {aa_vc*100:<10.1f} {fgt_vc*100:<14.1f} {cb_vc:<8} {t_vec:.0f}s")
    print()
    delta_aa = aa_ef - aa_vc
    verdict = "EigenFold WINS" if delta_aa > 0.02 else ("DRAW" if abs(delta_aa) <= 0.02 else "Vector WINS")
    print(f"  Delta AA: {delta_aa*100:+.1f}pp  ->  {verdict}")
    print(f"  Total elapsed: {elapsed:.0f}s")
    print("=" * 65)


if __name__ == '__main__':
    main()
