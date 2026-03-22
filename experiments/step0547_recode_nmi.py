"""
Step 547 -- U26 test: Recode argmax classification on CIFAR-100.

U26 (provisional): self-generated labels compound errors (9.8% codebook, 10.1% graph).
Test with Recode: does self-refinement improve class separation beyond LSH?

Step 526: LSH NMI=0.48. Step 543: Recode CIFAR acc=15%.
This experiment measures NMI properly with Recode substrate.

10K CIFAR-100 images. Observe -> act via argmin. Assign each cell its majority class.
NMI between cell assignments and true labels. Self-label accuracy (majority-class voting).

Predictions: NMI ~0.50, acc ~15%. Kill: NMI < 0.40.
"""
import numpy as np
import time


def enc(frame):
    """Avgpool16 + centered. Handles CIFAR (H,W,C) uint8 and LS20 frames."""
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        gray = frame.mean(axis=2).astype(np.float32) / 255.0
        x = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    else:
        a = np.array(frame[0], dtype=np.float32) / 15.0
        x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


N_A = 4
K = 16
DIM = 256
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

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc(frame)
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
        self.dim = len(x)
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

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


def nmi(labels_true, labels_pred):
    """Normalized Mutual Information (sklearn if available, else manual)."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_true, labels_pred))
    except ImportError:
        pass
    # Manual NMI
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = len(labels_true)
    classes_t = np.unique(labels_true)
    classes_p = np.unique(labels_pred)

    def entropy(arr, classes):
        h = 0.0
        for c in classes:
            p = np.sum(arr == c) / n
            if p > 0:
                h -= p * np.log(p)
        return h

    ht = entropy(labels_true, classes_t)
    hp = entropy(labels_pred, classes_p)
    if ht == 0 or hp == 0:
        return 0.0
    mi = 0.0
    for ct in classes_t:
        for cp in classes_p:
            nij = np.sum((labels_true == ct) & (labels_pred == cp))
            if nij > 0:
                mi += (nij / n) * np.log((nij / n) / ((np.sum(labels_true == ct) / n) * (np.sum(labels_pred == cp) / n)))
    return mi / np.sqrt(ht * hp)


def t0():
    # NMI: perfect clustering = 1.0
    lt = [0, 0, 1, 1, 2, 2]
    lp = [0, 0, 1, 1, 2, 2]
    assert abs(nmi(lt, lp) - 1.0) < 0.01, f"perfect NMI should be 1.0"

    # NMI: random < 0.5
    lp2 = [0, 1, 2, 0, 1, 2]
    assert nmi(lt, lp2) < 0.5, f"random NMI should be low"

    # enc() produces 256-dim centered vector for CIFAR
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    x = enc(img)
    assert x.shape == (256,), f"enc shape {x.shape}"
    assert abs(float(x.mean())) < 1e-5, "not centered"

    print("T0 PASS")


def run_seed(images, labels, seed, n=10_000):
    sub = Recode(seed=seed * 1000)
    rng = np.random.RandomState(42 + seed)
    idx = rng.choice(len(images), n, replace=False)

    t_start = time.time()
    nodes = []
    true_labels = []

    for i in idx:
        node = sub.observe(images[i])
        sub.act()
        nodes.append(node)
        true_labels.append(int(labels[i]))

    # Node -> majority class
    node_dist = {}
    for node, lbl in zip(nodes, true_labels):
        if node not in node_dist:
            node_dist[node] = {}
        node_dist[node][lbl] = node_dist[node].get(lbl, 0) + 1
    node_majority = {nd: max(d, key=d.get) for nd, d in node_dist.items()}

    # Self-label accuracy
    correct = sum(1 for node, lbl in zip(nodes, true_labels)
                  if node_majority.get(node) == lbl)
    acc = correct / n

    # NMI: map nodes to integer IDs
    node_to_int = {nd: i for i, nd in enumerate(node_dist.keys())}
    cell_ints = [node_to_int[nd] for nd in nodes]
    score = nmi(true_labels, cell_ints)

    nc, ns, ne = sub.stats()
    elapsed = time.time() - t_start
    print(f"  s{seed}: acc={acc:.1%} nmi={score:.3f} cells={nc} sp={ns} "
          f"nodes={len(node_dist)} {elapsed:.1f}s")
    return dict(seed=seed, acc=acc, nmi=score, cells=nc, splits=ns,
                n_nodes=len(node_dist))


def main():
    t0()

    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(
            './data/cifar100', train=True, download=True)
        images = np.array(ds.data)    # (50000, 32, 32, 3) uint8
        labels = np.array(ds.targets)
        print(f"CIFAR-100 loaded: {len(images)} images, 100 classes")
    except Exception as e:
        print(f"CIFAR-100 load failed: {e}")
        return

    results = []
    for seed in range(3):
        r = run_seed(images, labels, seed)
        results.append(r)

    avg_acc = float(np.mean([r['acc'] for r in results]))
    avg_nmi = float(np.mean([r['nmi'] for r in results]))
    avg_cells = float(np.mean([r['cells'] for r in results]))

    print(f"\n{'='*50}")
    print(f"avg: acc={avg_acc:.1%}  nmi={avg_nmi:.3f}  cells={avg_cells:.0f}")
    print(f"LSH baseline (Step 526): NMI=0.48")
    print(f"Step 543 CIFAR acc: 15% (shared centering chain)")

    if avg_nmi < 0.40:
        print("KILL: NMI < 0.40. Refinement hurts class structure.")
    elif avg_nmi > avg_nmi and avg_nmi > 0.52:
        print(f"BETTER THAN LSH: NMI={avg_nmi:.3f}. Refinement improves separation.")
    else:
        print(f"SIMILAR TO LSH: NMI={avg_nmi:.3f}. Refinement neutral on class structure.")

    print(f"\nU26 implication:")
    if avg_acc < 0.12:
        print(f"  Self-label accuracy {avg_acc:.1%} < 12%. High error rate confirms U26.")
    else:
        print(f"  Self-label accuracy {avg_acc:.1%}. Recode separates classes better than random.")


if __name__ == "__main__":
    main()
