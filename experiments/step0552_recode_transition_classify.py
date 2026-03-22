"""
Step 552 — Transition-equivalent classification on CIFAR-100.

Do nodes with identical transition distributions cluster by class?

Two conditions:
  Sorted: images presented class-by-class (0,0,...,1,1,...,99,99,...)
  Shuffled: random presentation order

For each node with >=2 visits, compute transition fingerprint:
  fp[n] = {successor_node: total_count} (sum over all actions)

Metric: avg within-class cosine similarity vs avg between-class similarity.
Ratio > 1.2x = sorted condition carries class information.

Kill: sorted within-class sim < 1.2x between-class sim.
5-min cap. CIFAR-100 only. No arcagi3.
"""
import numpy as np
import time

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
CIFAR_N = 10_000
MAX_PAIRS = 2000  # cap pairwise comparison


def enc(img):
    """Avgpool16 + centered. img: (32,32,3) uint8."""
    gray = img.mean(axis=2).astype(np.float32) / 255.0
    x = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return x - x.mean()


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

    def observe_cifar(self, img):
        """Observe a CIFAR image directly (bypassing LS20 enc)."""
        x = enc(img)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
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


def get_fingerprint(sub, n):
    """Aggregate successor counts across all actions."""
    fp = {}
    for a in range(N_A):
        for succ, cnt in sub.G.get((n, a), {}).items():
            key = str(succ)
            fp[key] = fp.get(key, 0) + cnt
    return fp


def cosine_sim(fp_a, fp_b):
    """Cosine similarity between two fingerprint dicts."""
    if not fp_a or not fp_b:
        return 0.0
    dot = sum(fp_a.get(k, 0) * fp_b.get(k, 0) for k in fp_a)
    na = float(np.sqrt(sum(v ** 2 for v in fp_a.values())))
    nb = float(np.sqrt(sum(v ** 2 for v in fp_b.values())))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return dot / (na * nb)


def run_condition(images, labels, order, label=""):
    """Run one pass in given order, return analysis."""
    sub = Recode(seed=0)
    node_classes = {}  # node -> {class: count}
    nodes_visited = []

    for i in order:
        img = images[i]
        lbl = int(labels[i])
        n = sub.observe_cifar(img)
        sub.act()
        nodes_visited.append(n)
        if n not in node_classes:
            node_classes[n] = {}
        node_classes[n][lbl] = node_classes[n].get(lbl, 0) + 1

    nc, ns, ne = sub.stats()
    print(f"  {label}: c={nc} sp={ns} nodes_ever={len(node_classes)}", flush=True)

    # Nodes with >=2 visits and fingerprints with >=1 edge
    multi_nodes = {n: cls_d for n, cls_d in node_classes.items()
                   if sum(cls_d.values()) >= 2}
    print(f"  {label}: nodes with >=2 visits: {len(multi_nodes)}", flush=True)

    if len(multi_nodes) < 4:
        return None

    # Assign majority class to each multi-node
    node_label = {n: max(cls_d, key=cls_d.get) for n, cls_d in multi_nodes.items()}
    # Filter: only nodes that have outgoing edges (fingerprint is non-empty)
    node_fp = {}
    for n in multi_nodes:
        fp = get_fingerprint(sub, n)
        if fp:
            node_fp[n] = fp

    print(f"  {label}: nodes with edges (fingerprint): {len(node_fp)}", flush=True)

    if len(node_fp) < 4:
        return None

    # Sample pairwise comparisons
    rng = np.random.RandomState(42)
    fp_nodes = list(node_fp.keys())
    n_nodes = len(fp_nodes)

    within_sims = []
    between_sims = []

    attempts = 0
    max_attempts = MAX_PAIRS * 10
    while (len(within_sims) < MAX_PAIRS or len(between_sims) < MAX_PAIRS) \
            and attempts < max_attempts:
        i, j = rng.choice(n_nodes, 2, replace=False)
        ni, nj = fp_nodes[i], fp_nodes[j]
        li, lj = node_label.get(ni), node_label.get(nj)
        if li is None or lj is None:
            attempts += 1
            continue
        sim = cosine_sim(node_fp[ni], node_fp[nj])
        if li == lj:
            if len(within_sims) < MAX_PAIRS:
                within_sims.append(sim)
        else:
            if len(between_sims) < MAX_PAIRS:
                between_sims.append(sim)
        attempts += 1

    if not within_sims or not between_sims:
        return None

    avg_within = float(np.mean(within_sims))
    avg_between = float(np.mean(between_sims))
    ratio = avg_within / avg_between if avg_between > 1e-8 else float('inf')

    print(f"  {label}: within={avg_within:.4f} between={avg_between:.4f} "
          f"ratio={ratio:.3f} (n_within={len(within_sims)} n_between={len(between_sims)})",
          flush=True)

    return dict(within=avg_within, between=avg_between, ratio=ratio,
                n_multi=len(multi_nodes), n_fp=len(node_fp))


def t0():
    rng = np.random.RandomState(0)

    # enc() shape and centering
    img = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    x = enc(img)
    assert x.shape == (256,), f"enc shape {x.shape}"
    assert abs(float(x.mean())) < 1e-5, "not centered"

    # Recode observe_cifar works
    sub = Recode(seed=0)
    img2 = rng.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    n = sub.observe_cifar(img); sub.act()
    n2 = sub.observe_cifar(img2); sub.act()
    assert n in sub.live or n in sub.ref
    assert len(sub.G) > 0, "G should have edges after 2 obs"

    # cosine_sim: identical fingerprints = 1.0
    fp = {"a": 3, "b": 2}
    assert abs(cosine_sim(fp, fp) - 1.0) < 1e-6

    # cosine_sim: orthogonal = 0.0
    fp_a = {"x": 1}
    fp_b = {"y": 1}
    assert abs(cosine_sim(fp_a, fp_b) - 0.0) < 1e-6

    # fingerprint aggregates over actions
    sub2 = Recode(seed=0)
    sub2.G = {(5, 0): {10: 3}, (5, 1): {20: 2}}
    fp5 = get_fingerprint(sub2, 5)
    assert fp5 == {"10": 3, "20": 2}, f"fp5={fp5}"

    print("T0 PASS")


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

    rng = np.random.RandomState(42)
    base_idx = rng.choice(len(images), CIFAR_N, replace=False)
    base_labels = labels[base_idx]
    base_images = images[base_idx]

    t_start = time.time()

    # Condition 1: Sorted by class
    sort_order = np.argsort(base_labels, stable=True)
    print("\nCondition 1: Sorted (class-by-class):", flush=True)
    r_sorted = run_condition(base_images, base_labels, sort_order, "sorted")

    # Condition 2: Shuffled
    shuf_order = rng.permutation(CIFAR_N)
    print("\nCondition 2: Shuffled (random):", flush=True)
    r_shuffled = run_condition(base_images, base_labels, shuf_order, "shuffled")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.1f}s")

    if r_sorted is None or r_shuffled is None:
        print("Insufficient data for comparison.")
        return

    print(f"\nResults:")
    print(f"  Sorted:   within={r_sorted['within']:.4f}  between={r_sorted['between']:.4f}  "
          f"ratio={r_sorted['ratio']:.3f}")
    print(f"  Shuffled: within={r_shuffled['within']:.4f}  between={r_shuffled['between']:.4f}  "
          f"ratio={r_shuffled['ratio']:.3f}")

    sorted_ratio = r_sorted['ratio']
    if sorted_ratio < 1.2:
        print(f"\nKILL: sorted ratio={sorted_ratio:.3f} < 1.2x. "
              "Transition structure does not carry class info even under sorted presentation.")
    elif sorted_ratio > 1.2 and r_shuffled['ratio'] < 1.2:
        print(f"\nCONFIRMED: sorted={sorted_ratio:.3f} shuffled={r_shuffled['ratio']:.3f}. "
              "Sorted presentation creates class-specific transition structure. "
              "R1-compliant classification possible under correlated presentation.")
    else:
        print(f"\nAMBIGUOUS: sorted={sorted_ratio:.3f} shuffled={r_shuffled['ratio']:.3f}.")


if __name__ == "__main__":
    main()
