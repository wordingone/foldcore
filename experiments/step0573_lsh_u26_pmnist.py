"""
Step 573 -- U26 validation: Recode self-label accuracy on P-MNIST.

U26 (provisional): self-generated labels compound errors.
  Codebook = 9.8%, Graph = 10.1% on P-MNIST. Near random (10%).
  Those used K=16 codebook cells with ~625 images/cell.

Proper test: train on MNIST train set (10K), evaluate on MNIST test set (10K).
Generalization accuracy = fraction of test images whose Recode cell's
  majority training-label matches the true test label.

Notes on k=16 LSH on P-MNIST:
  ~4742 unique cells for 10K images (~2 images/cell). Training acc = 76.7%
  but this is trivially inflated (nearly unique cells).
  Test accuracy measures actual generalization — expected ~10%.

Kill:  test_acc < 15% -> U26 confirmed for Recode (cells don't generalize).
Signal: test_acc > 25% -> U26 challenged for Recode.

Enc: 784 -> permute -> reshape 28x28 -> avgpool 14x14 = 196 -> center
"""
import numpy as np
import time

# ── encoding ──────────────────────────────────────────────────────────────────

def enc(flat_img, perm):
    """784-dim float32 [0,1] → permute → avgpool 14x14 → center (196 dims)."""
    x = flat_img[perm]
    x14 = x.reshape(28, 28).reshape(14, 2, 14, 2).mean(axis=(1, 3)).flatten()
    return (x14 - x14.mean()).astype(np.float32)


# ── Recode substrate ──────────────────────────────────────────────────────────

N_A = 4
K = 16
DIM = 196
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05


class Recode:
    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, x):
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        did = 0
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 3 or r1[1] < 3:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            self.ns += 1
            did += 1
            if did >= 3:
                break

    def stats(self):
        return len(self.live), self.ns, len(self.G)


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    perm = np.arange(784)
    img = np.random.randint(0, 256, 784, dtype=np.uint8).astype(np.float32) / 255.0
    v = enc(img, perm)
    assert v.shape == (196,), f"enc shape {v.shape}"
    assert abs(float(v.mean())) < 1e-5, "not centered"

    sub = Recode(dim=196, k=16, seed=0)
    n = sub.observe(v)
    sub.act()
    assert n is not None
    print("T0 PASS")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(X_tr, y_tr, X_te, y_te, perm, seed, n_train=10_000, n_test=10_000):
    sub = Recode(dim=196, k=16, seed=seed * 1000)
    rng = np.random.RandomState(42 + seed)
    tr_idx = rng.choice(len(X_tr), n_train, replace=False)
    te_idx = rng.choice(len(X_te), n_test, replace=False)

    t_start = time.time()

    # Train: build cell → class mapping
    nodes_tr = []
    labels_tr = []
    for i in tr_idx:
        x = enc(X_tr[i], perm)
        node = sub.observe(x)
        sub.act()
        nodes_tr.append(node)
        labels_tr.append(int(y_tr[i]))

    # Build cell majority label from training data
    node_dist = {}
    for node, lbl in zip(nodes_tr, labels_tr):
        d = node_dist.setdefault(node, {})
        d[lbl] = d.get(lbl, 0) + 1
    node_majority = {nd: max(d, key=d.get) for nd, d in node_dist.items()}

    # Training accuracy (inflated for k=16 since cells are near-unique)
    tr_correct = sum(1 for nd, lbl in zip(nodes_tr, labels_tr)
                     if node_majority.get(nd) == lbl)
    tr_acc = tr_correct / n_train

    # Test: map test images to training cells, predict label
    te_correct = te_seen = 0
    for i in te_idx:
        x = enc(X_te[i], perm)
        node = sub._node(x)
        if node in node_majority:
            te_seen += 1
            if node_majority[node] == int(y_te[i]):
                te_correct += 1

    te_acc_all = te_correct / n_test          # denominator = all test images
    te_acc_seen = te_correct / max(te_seen, 1) # denominator = seen-cell test images
    te_coverage = te_seen / n_test

    nc, ns, ne = sub.stats()
    elapsed = time.time() - t_start
    print(f"  s{seed}: tr_acc={tr_acc:.1%}  te_acc={te_acc_all:.1%}  "
          f"coverage={te_coverage:.1%}  splits={ns}  cells={nc}  {elapsed:.1f}s")
    return dict(seed=seed, tr_acc=tr_acc, te_acc=te_acc_all,
                te_acc_seen=te_acc_seen, coverage=te_coverage, cells=nc, splits=ns)


def main():
    t0()

    try:
        import torchvision
        tr = torchvision.datasets.MNIST('./data/mnist', train=True, download=True)
        te = torchvision.datasets.MNIST('./data/mnist', train=False, download=True)
        X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_tr = tr.targets.numpy()
        X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_te = te.targets.numpy()
        print(f"MNIST loaded: train={len(X_tr)}, test={len(X_te)}, 10 classes")
    except Exception as e:
        print(f"MNIST load failed: {e}")
        return

    # Task 0 permutation
    perm = np.random.RandomState(0).permutation(784)

    results = []
    t_global = time.time()
    for seed in range(3):
        if time.time() - t_global > 270:
            print("TIME CAP HIT")
            break
        r = run_seed(X_tr, y_tr, X_te, y_te, perm, seed)
        results.append(r)

    avg_tr = float(np.mean([r['tr_acc'] for r in results]))
    avg_te = float(np.mean([r['te_acc'] for r in results]))
    avg_te_seen = float(np.mean([r['te_acc_seen'] for r in results]))
    avg_cov = float(np.mean([r['coverage'] for r in results]))
    avg_cells = float(np.mean([r['cells'] for r in results]))
    avg_splits = float(np.mean([r['splits'] for r in results]))

    print(f"\n{'='*50}")
    print(f"avg: tr_acc={avg_tr:.1%}  te_acc={avg_te:.1%}  te_acc_seen={avg_te_seen:.1%}")
    print(f"     coverage={avg_cov:.1%}  cells={avg_cells:.0f}  splits={avg_splits:.0f}")
    print(f"Random baseline:         10.0%")
    print(f"Codebook (U26 evidence):  9.8%  (K=16 cells, ~625 img/cell)")
    print(f"Graph    (U26 evidence): 10.1%")

    print(f"\nNote: tr_acc={avg_tr:.1%} is inflated (k=16 -> ~{avg_cells:.0f} near-unique cells).")
    print(f"      te_acc={avg_te:.1%} is the real generalization metric.")

    if avg_te < 0.15:
        print(f"\nU26 CONFIRMED for Recode: te_acc={avg_te:.1%} ≈ random. Cells don't generalize.")
    elif avg_te > 0.25:
        print(f"\nU26 CHALLENGED: Recode te_acc={avg_te:.1%}. Cells generalize meaningfully.")
    else:
        print(f"\nWEAK SIGNAL: Recode te_acc={avg_te:.1%}. Marginal over random.")


if __name__ == "__main__":
    main()
