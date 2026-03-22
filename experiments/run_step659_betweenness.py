"""
Step 659 — Spatial composition via graph betweenness centrality.

Standard LSH k=12 + betweenness centrality (BC) of successors modulates counts.

Action selection: argmin_a(count(N,a) / (1 + BC(best_successor(N,a))))
High-BC successors = bridges = MORE attractive (lower effective count).

BC recomputed every 5K steps. ~300 nodes, fast computation.

If reaches L2: spatial composition suffices, temporal patterns unnecessary.
If L1 matches/beats argmin: spatial structure helps.
If L1 degrades: spatial composition fails.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25
N_SEEDS = 10
BC_RECOMPUTE_EVERY = 5000

BASELINE_L1 = [1362, 3270, 48391, 62727, 846, None, None, None, None, None]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def compute_bc_simple(G_dict, live):
    """
    Simple betweenness centrality via BFS from each node.
    Returns dict: node -> BC score (normalized).
    For ~300 nodes this is fast (O(V*E)).
    """
    # Build adjacency (directed): node -> set of successors
    nodes = set()
    succ = {}
    for (n, a), d in G_dict.items():
        if n not in live:
            continue
        nodes.add(n)
        for m in d:
            nodes.add(m)
            succ.setdefault(n, set()).add(m)

    if len(nodes) < 3:
        return {}

    # Try networkx first, fall back to manual BFS
    try:
        import networkx as nx
        dg = nx.DiGraph()
        for n, ms in succ.items():
            for m in ms:
                dg.add_edge(n, m)
        return nx.betweenness_centrality(dg, normalized=True)
    except ImportError:
        pass

    # Manual: approximate BC via random-sample BFS (sample 50 source nodes)
    node_list = list(nodes)
    bc = {n: 0.0 for n in node_list}
    sample = node_list[:min(50, len(node_list))]

    for src in sample:
        # BFS to find shortest paths
        dist = {src: 0}
        paths = {src: 1}
        sigma = {src: 1}
        prev = {n: [] for n in node_list}
        queue = [src]
        order = []

        while queue:
            v = queue.pop(0)
            order.append(v)
            for w in succ.get(v, []):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    queue.append(w)
                if dist.get(w) == dist[v] + 1:
                    sigma[w] = sigma.get(w, 0) + sigma.get(v, 0)
                    prev[w].append(v)

        delta = {n: 0.0 for n in node_list}
        for w in reversed(order):
            for v in prev[w]:
                c = (sigma.get(v, 0) / max(sigma.get(w, 1), 1)) * (1.0 + delta[w])
                delta[v] = delta.get(v, 0) + c
            if w != src:
                bc[w] = bc.get(w, 0) + delta[w]

    n_nodes = len(node_list)
    if n_nodes > 2:
        norm = 1.0 / ((n_nodes - 1) * (n_nodes - 2))
        bc = {n: v * norm for n, v in bc.items()}

    return bc


class BCRecode:
    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.bc = {}  # node -> betweenness centrality

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc_frame(frame)
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

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % BC_RECOMPUTE_EVERY == 0:
            self.bc = compute_bc_simple(self.G, self.live)

        return n

    def act(self):
        best_action = 0
        best_score = float('inf')

        for a in range(N_A):
            d = self.G.get((self._cn, a), {})
            total_count = sum(d.values())
            if total_count == 0:
                effective = 0.0
            else:
                # Use BC of most frequent successor
                best_succ = max(d, key=d.get) if d else None
                bc_val = self.bc.get(best_succ, 0.0) if best_succ else 0.0
                effective = total_count / (1.0 + bc_val)

            if effective < best_score:
                best_score = effective
                best_action = a

        self._pn = self._cn
        self._pa = best_action
        return best_action

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


def run(seed, make):
    env = make()
    sub = BCRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    bsl = BASELINE_L1[seed]
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no baseline"
    else:
        spd = "NO_L1"

    bc_stats = ""
    if sub.bc:
        bc_vals = list(sub.bc.values())
        bc_stats = f"bc_max={max(bc_vals):.3f} bc_mean={np.mean(bc_vals):.3f}"

    print(f"  s{seed}: L1={l1} ({spd}) L2={l2} go={go} unique={len(sub.live)} {bc_stats}",
          flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, unique=len(sub.live))


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"BC-weighted selection: {N_SEEDS} seeds, {PER_SEED_TIME}s cap, BC every {BC_RECOMPUTE_EVERY} steps")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    l2_n = sum(1 for r in results if r['l2'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in results
                if r['l1'] and BASELINE_L1[r['seed']]]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0

    for r in results:
        bsl = BASELINE_L1[r['seed']]
        if r['l1'] and bsl:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        elif r['l1']:
            spd = "no baseline"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) L2={r['l2']}")

    print(f"\nL1={l1_n}/10  L2={l2_n}/10  avg_speedup={avg_ratio:.2f}x")

    if l2_n > 0:
        print("BREAKTHROUGH: L2 reached — spatial composition suffices")
    elif l1_n >= 6 and avg_ratio > 1.0:
        print("SIGNAL: BC-weighted selection faster than argmin")
    elif l1_n < 4:
        print("KILL: spatial BC composition fails navigation")
    else:
        print(f"MARGINAL: L1={l1_n}/10")


if __name__ == "__main__":
    main()
