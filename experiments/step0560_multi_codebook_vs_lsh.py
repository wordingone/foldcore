"""
Step 560 — What would the codebook do? Analytical comparison (Q13)

Run Recode for 100K steps on LS20 seed=0.
Collect mean_obs per node.
Analysis:
  1. K-means clustering (k=10, 20, 50) — prototype diversity
  2. High-cosine-similarity node pairs (cos > 0.95) with different transitions
     = nodes codebook would merge but Recode keeps separate
  3. Would merging connect abandoned regions?

Kill: 0 merge candidates -> LSH and codebook see same structure.
Find: >100 merge candidates AND merging connects abandoned -> codebook
      provides object-level discrimination LSH lacks.

5-min cap. 100K steps.
"""
import numpy as np
import time
import sys
from sklearn.cluster import KMeans

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class RecodeWithObs:
    """Recode that also accumulates per-node mean obs vectors."""

    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self.dim = dim
        self._last_visit = {}
        # Per-node obs accumulation
        self._obs_sum = {}   # node -> sum of enc vectors
        self._obs_cnt = {}   # node -> count

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
        self._last_visit[n] = self.t
        # Accumulate obs for this node
        s = self._obs_sum.get(n)
        if s is None:
            self._obs_sum[n] = x.astype(np.float64).copy()
            self._obs_cnt[n] = 1
        else:
            s += x.astype(np.float64)
            self._obs_cnt[n] += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            sv, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (sv + self._px.astype(np.float64), c + 1)
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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)

    def mean_obs(self, node):
        s = self._obs_sum.get(node)
        c = self._obs_cnt.get(node, 0)
        if s is None or c == 0:
            return None
        return (s / c).astype(np.float32)

    def transition_profile(self, node):
        """Set of (action, successor) pairs seen from this node."""
        result = set()
        for a in range(N_A):
            d = self.G.get((node, a), {})
            for succ in d:
                result.add((a, succ))
        return result

    def successors(self, node):
        """All successor nodes from this node (any action)."""
        succs = set()
        for a in range(N_A):
            d = self.G.get((node, a), {})
            succs.update(d.keys())
        return succs

    def active_set(self, window=100_000):
        cutoff = self.t - window
        return sum(1 for v in self._last_visit.values() if v >= cutoff)


def cosine_sim(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def t0():
    rng = np.random.RandomState(42)
    sub = RecodeWithObs(seed=0)

    # Generate synthetic frames and run a few steps
    for _ in range(5):
        frame = [rng.randint(0, 16, (64, 64))]
        sub.observe(frame)
        sub.act()

    # Verify mean_obs works
    for n in list(sub.live)[:3]:
        m = sub.mean_obs(n)
        assert m is not None, f"No mean_obs for {n}"
        assert m.shape == (DIM,), f"Wrong shape: {m.shape}"

    # Verify transition_profile works
    for n in list(sub.live)[:3]:
        tp = sub.transition_profile(n)
        assert isinstance(tp, set)

    # Verify cosine_sim
    v1 = np.ones(10, dtype=np.float32)
    v2 = np.ones(10, dtype=np.float32) * 2
    assert abs(cosine_sim(v1, v2) - 1.0) < 1e-5
    v3 = np.array([1.0, 0.0] * 5, dtype=np.float32)
    v4 = np.array([0.0, 1.0] * 5, dtype=np.float32)
    assert abs(cosine_sim(v3, v4)) < 1e-5

    # K-means on small data
    data = rng.randn(50, DIM).astype(np.float32)
    km = KMeans(n_clusters=5, n_init=3, random_state=0).fit(data)
    assert len(set(km.labels_)) <= 5

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    t_start = time.time()
    sub = RecodeWithObs(seed=0)
    obs = env.reset(seed=0)
    level = 0
    go = 0
    l1_step = None

    print("Running 100K steps...", flush=True)
    for step in range(1, 100_001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=0)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            sub.on_reset()
            if cl == 1 and l1_step is None:
                l1_step = step
                print(f"  L1 at step={step}", flush=True)

        if step % 25_000 == 0:
            elapsed = time.time() - t_start
            print(f"  @{step} live={len(sub.live)} go={go} {elapsed:.0f}s", flush=True)

        if time.time() - t_start > 240:
            print(f"  Timeout at step={step}")
            break

    elapsed = time.time() - t_start
    print(f"\nRun done: {elapsed:.0f}s, steps={step}, go={go}, live={len(sub.live)}", flush=True)

    # ---- Analysis ----
    live_nodes = list(sub.live)
    print(f"Live nodes: {len(live_nodes)}", flush=True)

    # Extract mean obs for live nodes
    mean_obs_data = {}
    for n in live_nodes:
        m = sub.mean_obs(n)
        if m is not None:
            mean_obs_data[n] = m

    nodes_with_obs = list(mean_obs_data.keys())
    print(f"Nodes with obs vectors: {len(nodes_with_obs)}", flush=True)

    if len(nodes_with_obs) < 10:
        print("Insufficient nodes for analysis.")
        return

    # Stack into matrix
    node_list = nodes_with_obs
    X = np.stack([mean_obs_data[n] for n in node_list], axis=0)  # (N, 256)
    N = len(node_list)
    print(f"Obs matrix: {X.shape}", flush=True)

    # ---- 1. K-means clustering ----
    print("\n=== 1. K-means clustering ===", flush=True)
    for k in [10, 20, 50]:
        if k >= N:
            print(f"  k={k}: skip (only {N} nodes)")
            continue
        km = KMeans(n_clusters=k, n_init=5, random_state=0, max_iter=100).fit(X)
        labels = km.labels_
        sizes = np.bincount(labels)
        n_large = int(np.sum(sizes > 5))
        n_singleton = int(np.sum(sizes == 1))
        print(f"  k={k}: large(>5)={n_large}/{k}  singleton={n_singleton}  "
              f"max={sizes.max()}  min={sizes.min()}  mean={sizes.mean():.1f}")

    # ---- 2. High-cosine-similarity pairs with different transitions ----
    print("\n=== 2. Merge candidate pairs (cos > 0.95, diff transitions) ===", flush=True)

    # Compute pairwise cosine similarity efficiently via matrix mult
    # Normalize rows
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    Xn = X / norms  # (N, 256) normalized

    # Pairwise cosine: (N, N) matrix — only compute upper triangle
    # For large N, do in blocks
    COS_THRESH = 0.95
    merge_candidates = []   # (i, j) pairs
    BLOCK = 200

    for i0 in range(0, N, BLOCK):
        i1 = min(i0 + BLOCK, N)
        block_cos = Xn[i0:i1] @ Xn.T  # (block, N)
        for bi, i in enumerate(range(i0, i1)):
            for j in range(i + 1, N):
                if block_cos[bi, j] > COS_THRESH:
                    merge_candidates.append((i, j))

    print(f"  Pairs with cos > {COS_THRESH}: {len(merge_candidates)}", flush=True)

    if not merge_candidates:
        print("  KILL: 0 merge candidates. LSH and codebook see same structure.")
        print("  The codebook ban is irrelevant to L2.")
        return

    # For each candidate pair, check transition profile overlap
    pairs_diff_trans = []
    for i, j in merge_candidates:
        ni, nj = node_list[i], node_list[j]
        tp_i = sub.transition_profile(ni)
        tp_j = sub.transition_profile(nj)
        if not tp_i or not tp_j:
            continue
        # Jaccard similarity of transition profiles
        union = tp_i | tp_j
        inter = tp_i & tp_j
        jac = len(inter) / len(union) if union else 1.0
        pairs_diff_trans.append((i, j, jac))

    pairs_diff_trans.sort(key=lambda x: x[2])  # sort by Jaccard (lowest first)
    n_diff = sum(1 for _, _, j in pairs_diff_trans if j < 0.5)
    print(f"  Pairs with Jaccard < 0.5 (different transitions): {n_diff}", flush=True)
    print(f"  Sample pairs (lowest Jaccard):")
    for i, j, jac in pairs_diff_trans[:5]:
        cos = float(Xn[i] @ Xn[j])
        print(f"    nodes ({i},{j}) cos={cos:.4f} jaccard={jac:.4f}")

    # ---- 3. Would merging connect abandoned regions? ----
    print("\n=== 3. Merging -> new graph connections? ===", flush=True)

    # Define "abandoned" = nodes visited but last seen > 50K steps ago
    cutoff = sub.t - 50_000
    abandoned = {n for n, lv in sub._last_visit.items() if lv < cutoff and n in sub.live}
    print(f"  Abandoned nodes (last_visit < t-50K): {len(abandoned)}", flush=True)

    new_connections = 0
    for i, j, jac in pairs_diff_trans:
        if jac >= 0.5:
            continue
        ni, nj = node_list[i], node_list[j]
        succs_i = sub.successors(ni)
        succs_j = sub.successors(nj)
        # If merging ni+nj: ni gains access to succs_j and vice versa
        # New connection if succs_j \ succs_i intersects abandoned (or vice versa)
        new_from_j = succs_j - succs_i
        new_from_i = succs_i - succs_j
        if new_from_j & abandoned or new_from_i & abandoned:
            new_connections += 1

    print(f"  Merge pairs that would connect abandoned regions: {new_connections}", flush=True)

    # ---- Summary ----
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s | live={len(live_nodes)} | go={go} | L1={l1_step}")
    print(f"\nMerge candidates (cos > 0.95): {len(merge_candidates)}")
    print(f"Different transitions (Jaccard < 0.5): {n_diff}")
    print(f"Connecting abandoned via merge: {new_connections}")

    if len(merge_candidates) == 0:
        print("\nKILL: 0 merge candidates. LSH and codebook see same structure.")
        print("Codebook ban is irrelevant to L2.")
    elif n_diff > 100 and new_connections > 0:
        print(f"\nFIND: {n_diff} merge candidates with different transitions, "
              f"{new_connections} connect abandoned.")
        print("Codebook's spatial engine provides object-level discrimination LSH lacks.")
        print("BUT: this doesn't help L2 — L2 needs navigation TOWARD objects, not graph structure.")
    elif len(merge_candidates) > 0 and n_diff == 0:
        print(f"\n{len(merge_candidates)} high-cos pairs BUT all have similar transitions.")
        print("Merging would be safe but provides no new information.")
        print("Codebook advantage: compression only, not discrimination.")
    else:
        print(f"\n{len(merge_candidates)} high-cos pairs, {n_diff} diff-transition, "
              f"{new_connections} new connections.")
        print("Partial evidence: codebook provides some additional structure.")


if __name__ == "__main__":
    main()
