#!/usr/bin/env python3
"""
Step 95 -- P-MNIST benchmark with spectral substrate.
[researcher] mail 1227. k=8, spectral Phi, Formula C composition.

Classification via compositional encoding:
  Train: for each class c, compose all training sample eigenforms left-to-right
         into a single class eigenform C_c.
  Test:  compute eigenform of test sample, classify by nearest C_c (cosine).

Compare to vector cosine baseline: 46.2% AA (Step 76).
"""
import sys, random, math, time
import numpy as np

K = 8
N_TASKS = 2
N_TRAIN_TASK = 6000
N_TEST_TASK = 10000
N_CLASSES = 10
TRAIN_PER_CLS = N_TRAIN_TASK // N_CLASSES  # 600
COMPOSE_STEPS = 200
CONVERGE_TOL = 0.01
D_MID = 384
D_MAT = K * K  # 64
SEED = 42

TARGET = math.sqrt(K)


# ─── Spectral substrate (numpy) ───────────────────────────────────────────────

def phi_np(M):
    C = M @ M.T
    n = np.linalg.norm(C, 'fro')
    if n < 1e-10: return M.copy()
    return C * (TARGET / n)


def converge_np(M, max_steps=COMPOSE_STEPS, tol=CONVERGE_TOL):
    for _ in range(max_steps):
        p = phi_np(M)
        d = np.linalg.norm(p - M, 'fro')
        M = p
        if d < tol: return M, True
    return M, False


def psi_C_np(A, B):
    AB = A @ B
    n = np.linalg.norm(AB, 'fro')
    if n < 1e-10: return A + B
    return A + B - AB * (TARGET / n)


def compose_np(A, B):
    start = psi_C_np(A, B)
    return converge_np(start)


def cosine_np(A, B):
    dot = float(np.sum(A * B))
    na = float(np.linalg.norm(A, 'fro'))
    nb = float(np.linalg.norm(B, 'fro'))
    if na < 1e-10 or nb < 1e-10: return 0.0
    return dot / (na * nb)


# ─── Data loading & embedding ─────────────────────────────────────────────────

def make_proj1(seed=12345):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MID, 784).astype(np.float32) / math.sqrt(784)


def make_proj2(seed=22222):
    rng = np.random.RandomState(seed)
    return rng.randn(D_MAT, D_MID).astype(np.float32) / math.sqrt(D_MID)


def make_permutation(seed):
    perm = list(range(784))
    rng = random.Random(seed)
    rng.shuffle(perm)
    return np.array(perm, dtype=np.int64)


def embed_task(X_flat, perm, P1, P2):
    """Project and reshape to (n, K, K) matrices."""
    X_perm = X_flat[:, perm]
    mid = X_perm @ P1.T  # (n, D_MID)
    mid = mid / (np.linalg.norm(mid, axis=1, keepdims=True) + 1e-15)
    proj = mid @ P2.T    # (n, K*K)
    return proj.reshape(-1, K, K).astype(np.float64)


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


# ─── Compositional classifier ─────────────────────────────────────────────────

class CompositionalClassifier:
    """One class eigenform per class, built by left-to-right chaining."""

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.class_ef = [None] * n_classes  # one eigenform per class

    def train_class(self, mats, label):
        """Chain all mats[label] samples into a class eigenform."""
        # First convert each sample matrix to eigenform
        efs = []
        for M in mats:
            M_f, conv = converge_np(M)
            if conv: efs.append(M_f)
        if not efs: return
        # Chain left to right
        C = efs[0]
        for i in range(1, len(efs)):
            C, conv = compose_np(C, efs[i])
            if not conv:
                C, _ = converge_np(C)
        C, _ = converge_np(C)
        self.class_ef[label] = C

    def diagnose(self):
        """Report cosine similarity between class eigenforms."""
        efs = [ef for ef in self.class_ef if ef is not None]
        if len(efs) < 2: return
        cos_vals = []
        for i in range(len(efs)):
            for j in range(i+1, len(efs)):
                cos_vals.append(abs(cosine_np(efs[i], efs[j])))
        cos_vals.sort()
        n = len(cos_vals)
        print(f"  Class EF pairwise cosine: min={cos_vals[0]:.3f} "
              f"med={cos_vals[n//2]:.3f} max={cos_vals[-1]:.3f}")

    def classify(self, M):
        """Classify by cosine similarity to class eigenforms."""
        M_f, conv = converge_np(M)
        if not conv: M_f = M
        best_cls, best_cos = 0, -2.0
        for c in range(self.n_classes):
            if self.class_ef[c] is None: continue
            cos = cosine_np(M_f, self.class_ef[c])
            if cos > best_cos:
                best_cos = cos
                best_cls = c
        return best_cls


# ─── Grouped multi-prototype compositional classifier ─────────────────────────

class GroupedCompositionalClassifier:
    """K groups of samples per class, each group chains into one eigenform.
    Classify by nearest prototype among all class*K_GROUPS eigenforms."""

    def __init__(self, n_classes, k_groups=10):
        self.n_classes = n_classes
        self.k_groups = k_groups
        self.prototypes = []  # list of (ef, label)

    def train_class(self, mats, label):
        n = len(mats)
        group_size = max(1, n // self.k_groups)
        for g in range(self.k_groups):
            start = g * group_size
            end = start + group_size if g < self.k_groups - 1 else n
            group = mats[start:end]
            if len(group) == 0: continue
            efs = []
            for M in group:
                M_f, conv = converge_np(M)
                if conv: efs.append(M_f)
            if not efs: continue
            C = efs[0]
            for i in range(1, len(efs)):
                C, conv = compose_np(C, efs[i])
                if not conv: C, _ = converge_np(C)
            C, _ = converge_np(C)
            self.prototypes.append((C, label))

    def classify(self, M):
        M_f, conv = converge_np(M)
        if not conv: M_f = M
        best_cls, best_cos = 0, -2.0
        for ef, lbl in self.prototypes:
            cos = cosine_np(M_f, ef)
            if cos > best_cos:
                best_cos = cos
                best_cls = lbl
        return best_cls


# ─── Vector cosine baseline ────────────────────────────────────────────────────

class VectorMeanClassifier:
    """Mean-of-class vector, classify by cosine."""
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.means = [None] * n_classes

    def train_class(self, mats, label):
        vecs = mats.reshape(len(mats), -1).astype(np.float64)
        mean = vecs.mean(axis=0)
        n = np.linalg.norm(mean)
        self.means[label] = mean / (n + 1e-15)

    def classify(self, M):
        v = M.reshape(-1)
        n = np.linalg.norm(v)
        v = v / (n + 1e-15)
        best_cls, best_cos = 0, -2.0
        for c in range(self.n_classes):
            if self.means[c] is None: continue
            cos = float(np.dot(v, self.means[c]))
            if cos > best_cos:
                best_cos = cos
                best_cls = c
        return best_cls


# ─── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(name, clf_factory, task_mats_tr, task_labels_tr,
                  task_mats_te, y_te):
    """
    task_mats_tr: list of (n_train, K, K) arrays, one per task
    task_labels_tr: list of label arrays, one per task
    task_mats_te: list of (n_test, K, K) arrays, one per task
    """
    print(f"\n--- {name} ---", flush=True)
    clf = clf_factory()
    acc_matrix = [[None]*N_TASKS for _ in range(N_TASKS)]

    for task_id in range(N_TASKS):
        t0 = time.time()
        mats_tr = task_mats_tr[task_id]
        labels_tr = task_labels_tr[task_id]

        # Train each class
        for c in range(N_CLASSES):
            idx = np.where(labels_tr == c)[0]
            clf.train_class(mats_tr[idx], c)
        t_tr = time.time() - t0

        # Test on all seen tasks
        # Diagnose class eigenforms after first task
        if hasattr(clf, 'diagnose') and task_id == 0:
            clf.diagnose()

        for et in range(task_id + 1):
            t1 = time.time()
            correct = sum(1 for i in range(len(y_te))
                         if clf.classify(task_mats_te[et][i]) == y_te[i])
            acc = correct / len(y_te)
            acc_matrix[et][task_id] = acc
            print(f"  Task {et+1} after training task {task_id+1}: {acc*100:.1f}% "
                  f"(train={t_tr:.1f}s, test={time.time()-t1:.1f}s)", flush=True)

    # Metrics
    aa = np.mean([acc_matrix[t][N_TASKS-1] for t in range(N_TASKS)])
    forgetting = acc_matrix[0][0] - acc_matrix[0][N_TASKS-1] if N_TASKS > 1 else 0.0
    print(f"  AA={aa*100:.1f}%, Forgetting={forgetting*100:.1f}%")
    return aa, forgetting


def main():
    t0 = time.time()
    print(f"Step 95 -- P-MNIST spectral substrate benchmark", flush=True)
    print(f"k={K}, spectral Phi, Formula C composition", flush=True)
    print(f"Baseline (Step 76 vector cosine): 46.2% AA", flush=True)
    print()

    # Load MNIST
    print("Loading MNIST...", flush=True)
    X_tr, y_tr, X_te, y_te = load_mnist()
    print(f"  Train: {len(X_tr)}, Test: {len(X_te)}")

    # Build projections
    P1 = make_proj1()
    P2 = make_proj2()
    perms = [make_permutation(SEED + t) for t in range(N_TASKS)]

    # Build task data
    print("Embedding tasks...", flush=True)
    task_mats_tr = []
    task_labels_tr = []
    task_mats_te = []

    for t in range(N_TASKS):
        # Training: stratified sample
        X_sub, y_sub = stratified_sample(X_tr, y_tr, TRAIN_PER_CLS, SEED + t)
        mats_tr = embed_task(X_sub, perms[t], P1, P2)
        task_mats_tr.append(mats_tr)
        task_labels_tr.append(y_sub)
        # Test (full 10k)
        mats_te = embed_task(X_te, perms[t], P1, P2)
        task_mats_te.append(mats_te)
        print(f"  Task {t+1}: {len(mats_tr)} train, {len(mats_te)} test")

    print()

    # Run compositional classifier
    aa_comp, fgt_comp = run_benchmark(
        "Compositional (Formula C + spectral Phi)",
        lambda: CompositionalClassifier(N_CLASSES),
        task_mats_tr, task_labels_tr, task_mats_te, y_te
    )

    # Run grouped multi-prototype (10 groups per class)
    aa_grp, fgt_grp = run_benchmark(
        "Grouped compositional (10 groups/class)",
        lambda: GroupedCompositionalClassifier(N_CLASSES, k_groups=10),
        task_mats_tr, task_labels_tr, task_mats_te, y_te
    )

    # Run vector mean baseline
    aa_vec, fgt_vec = run_benchmark(
        "Vector mean baseline",
        lambda: VectorMeanClassifier(N_CLASSES),
        task_mats_tr, task_labels_tr, task_mats_te, y_te
    )

    elapsed = time.time() - t0
    print(f"\n=== Summary ===")
    print(f"  Compositional single EF:   AA={aa_comp*100:.1f}%, Forgetting={fgt_comp*100:.1f}%")
    print(f"  Grouped (10 groups/class): AA={aa_grp*100:.1f}%, Forgetting={fgt_grp*100:.1f}%")
    print(f"  Vector mean baseline:      AA={aa_vec*100:.1f}%, Forgetting={fgt_vec*100:.1f}%")
    print(f"  Step 76 vector cosine ref: AA=46.2%")
    print(f"  Best vs Step 76: {(max(aa_comp, aa_grp, aa_vec) - 0.462)*100:+.1f}pp")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
