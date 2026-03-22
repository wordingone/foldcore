"""
Step 550 — LS20 state graph topology dump.

Diagnostic: map the state graph Recode builds navigating LS20.
5-min cap, seed=0, 500K steps. Dump full topology + JSON.

No navigation metric — this is mapping.
Output: stdout stats + experiments/ls20_graph_dump.json
"""
import numpy as np
import time
import sys
import json

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def node_id(n):
    """Serialize node key to JSON-safe string."""
    return str(n)


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
        self._first_visit = {}
        self._last_visit = {}

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
        if n not in self._first_visit:
            self._first_visit[n] = self.t
        self._last_visit[n] = self.t
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


# ---- Analysis ----

def connected_components(all_nodes, G):
    """Weakly connected components via union-find."""
    parent = {n: n for n in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        pa, pb = find(a), find(b)
        if pa != pb:
            parent[pa] = pb

    for (src, a), d in G.items():
        if src in parent:
            for dst in d:
                if dst in parent:
                    union(src, dst)

    comps = {}
    for n in all_nodes:
        root = find(n)
        comps.setdefault(root, []).append(n)
    return list(comps.values())


def analyze(sub, total_steps, out_path):
    G = sub.G
    live = sub.live
    ref = sub.ref
    first_v = sub._first_visit
    last_v = sub._last_visit

    all_nodes = set(first_v.keys())  # every node ever visited
    all_graph_nodes = set()          # nodes appearing as src in G
    for (n, a) in G:
        all_graph_nodes.add(n)
    for (n, a), d in G.items():
        for dst in d:
            all_graph_nodes.add(dst)

    W1 = 100_000        # first window cutoff
    W2 = max(1, total_steps - 100_000)  # last window start

    # 1. Full graph statistics
    total_edges = sum(len(d) for d in G.values())
    total_obs = sum(sum(d.values()) for d in G.values())

    # Out-degree per node per action
    out_deg = {}
    for (n, a), d in G.items():
        out_deg[n] = out_deg.get(n, 0) + len(d)

    # In-degree per node
    in_deg = {}
    for (n, a), d in G.items():
        for dst in d:
            in_deg[dst] = in_deg.get(dst, 0) + 1

    out_degs = list(out_deg.values())
    in_degs = list(in_deg.values())

    print(f"\n{'='*60}")
    print(f"1. FULL GRAPH STATISTICS (steps={total_steps})")
    print(f"   Live nodes:     {len(live)}")
    print(f"   Refined nodes:  {len(ref)}")
    print(f"   All nodes ever: {len(all_nodes)}")
    print(f"   Graph src nodes:{len(all_graph_nodes)}")
    print(f"   (node,action) pairs: {len(G)}")
    print(f"   Total edges (unique successors): {total_edges}")
    print(f"   Total observations: {total_obs}")
    if out_degs:
        print(f"   Out-degree: mean={np.mean(out_degs):.2f} "
              f"max={max(out_degs)} median={np.median(out_degs):.1f}")
    if in_degs:
        print(f"   In-degree:  mean={np.mean(in_degs):.2f} "
              f"max={max(in_degs)} median={np.median(in_degs):.1f}")

    comps = connected_components(all_graph_nodes, G)
    comp_sizes = sorted([len(c) for c in comps], reverse=True)
    print(f"   Connected components: {len(comps)}")
    print(f"   Largest comp: {comp_sizes[0] if comp_sizes else 0} nodes")
    if len(comp_sizes) > 1:
        print(f"   Top-5 comp sizes: {comp_sizes[:5]}")

    # 2. Frontier analysis
    active = {n for n in all_nodes if last_v.get(n, 0) >= W2}
    abandoned = {n for n in all_nodes if first_v.get(n, 0) <= W1
                 and last_v.get(n, 0) < W2}
    late_disc = {n for n in all_nodes if first_v.get(n, 0) >= W2}

    print(f"\n2. FRONTIER ANALYSIS (first={W1}, last_start={W2})")
    print(f"   Active (seen in last 100K):     {len(active)}")
    print(f"   Abandoned (first 100K, not recent): {len(abandoned)}")
    print(f"   Late discoveries (first seen in last 100K): {len(late_disc)}")

    # Determinism of active nodes
    det_scores = []
    for n in active:
        for a in range(N_A):
            d = G.get((n, a), {})
            if sum(d.values()) > 5:
                top_frac = max(d.values()) / sum(d.values()) if d else 0
                det_scores.append(top_frac)
    if det_scores:
        print(f"   Active node determinism: mean={np.mean(det_scores):.3f} "
              f"(1=fully deterministic, 0.25=uniform random)")

    # 3. Transition entropy map
    entropies = []
    for (n, a), d in G.items():
        total = sum(d.values())
        if total >= 10:
            v = np.array(list(d.values()), np.float64)
            p = v / v.sum()
            h = float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))
            entropies.append((h, n, a, total))

    entropies.sort(key=lambda x: x[0], reverse=True)
    print(f"\n3. TRANSITION ENTROPY (min 10 obs, n={len(entropies)} pairs)")
    if entropies:
        print(f"   H range: [{entropies[-1][0]:.3f}, {entropies[0][0]:.3f}]")
        print(f"   H > 1.0 (high entropy): {sum(1 for h,*_ in entropies if h > 1.0)}")
        print(f"   H < 0.1 (low entropy):  {sum(1 for h,*_ in entropies if h < 0.1)}")
        print(f"   Top-5 high entropy (noisy TV candidates):")
        for h, n, a, total in entropies[:5]:
            print(f"     node={node_id(n)} action={a} H={h:.3f} obs={total}")
        print(f"   Top-5 low entropy + low visits (unexploited frontier):")
        low_h_low_v = sorted([(h, n, a, total) for h, n, a, total in entropies
                               if h < 0.3 and total < 100], key=lambda x: (x[3], x[0]))
        for h, n, a, total in low_h_low_v[:5]:
            print(f"     node={node_id(n)} action={a} H={h:.3f} obs={total}")

    # 4. Dead ends
    self_loops = []
    saturated = []
    for n in live:
        # Self-loop: all actions lead back to n
        all_self = True
        all_sat = True
        for a in range(N_A):
            d = G.get((n, a), {})
            total_a = sum(d.values())
            if total_a < 100:
                all_sat = False
            if not d or set(d.keys()) != {n}:
                all_self = False
        if all_self:
            self_loops.append(n)
        if all_sat:
            saturated.append(n)

    print(f"\n4. DEAD ENDS")
    print(f"   Self-loops (all actions -> same node): {len(self_loops)}")
    print(f"   Saturated (all actions >100 obs):     {len(saturated)}")

    # 5. JSON dump
    nodes_out = []
    for n in all_nodes:
        nodes_out.append({
            "id": node_id(n),
            "type": "refined" if n in ref else ("live" if n in live else "visited"),
            "first_visit": first_v.get(n),
            "last_visit": last_v.get(n),
        })

    edges_out = []
    for (src, a), d in G.items():
        for dst, count in d.items():
            edges_out.append({
                "src": node_id(src),
                "action": a,
                "dst": node_id(dst),
                "count": count,
            })

    dump = {
        "meta": {
            "total_steps": total_steps,
            "live_nodes": len(live),
            "refined_nodes": len(ref),
            "total_nodes_ever": len(all_nodes),
            "total_edges": total_edges,
            "total_obs": total_obs,
            "connected_components": len(comps),
            "largest_component": comp_sizes[0] if comp_sizes else 0,
        },
        "nodes": nodes_out,
        "edges": edges_out,
    }

    with open(out_path, 'w') as f:
        json.dump(dump, f, indent=2)
    print(f"\n5. JSON DUMP saved to: {out_path}")
    print(f"   {len(nodes_out)} nodes, {len(edges_out)} edges")


def t0():
    rng = np.random.RandomState(0)

    # enc() shape
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,), f"enc shape {x.shape}"
    assert abs(float(x.mean())) < 1e-5, "not centered"

    # visit tracking
    sub = Recode(seed=0)
    f1 = [rng.randint(0, 16, (64, 64))]
    f2 = [rng.randint(0, 16, (64, 64))]
    n1 = sub.observe(f1); sub.act()
    n2 = sub.observe(f2); sub.act()
    assert sub._first_visit[n1] == 1
    assert sub._first_visit[n2] == 2
    assert sub._last_visit[n2] == 2

    # node_id serialization
    assert isinstance(node_id(12345), str)
    assert isinstance(node_id((12345, 0)), str)

    # JSON round-trip
    test_dump = {"nodes": [{"id": "42", "type": "live"}], "edges": []}
    s = json.dumps(test_dump)
    d = json.loads(s)
    assert d["nodes"][0]["id"] == "42"

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

    sub = Recode(seed=0)
    obs = env.reset(seed=0)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 500_001):
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
            sub.on_reset()
            nc, ns, ne = sub.stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"L1@{step} c={nc} sp={ns} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"L2@{step} c={nc} sp={ns} go={go}", flush=True)
            level = cl

        if step % 100_000 == 0:
            nc, ns, ne = sub.stats()
            el = time.time() - t_start
            print(f"@{step} c={nc} sp={ns} go={go} {el:.0f}s", flush=True)

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    elapsed = time.time() - t_start
    tag = "L2" if l2 else ("L1" if l1 else "---")
    print(f"\nDone: {tag} steps={step} c={nc} sp={ns} go={go} {elapsed:.0f}s")

    out_path = "experiments/ls20_graph_dump.json"
    analyze(sub, step, out_path)


if __name__ == "__main__":
    main()
