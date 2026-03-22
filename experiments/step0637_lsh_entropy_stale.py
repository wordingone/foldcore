"""
Step 637 — Transition entropy per edge (successor-destination stale).

For each edge (N, A), compute Shannon entropy H over the empirical distribution
of successor cells S: H = -sum(p_s * log2(p_s)).

Edges with H < 0.1 (low entropy → deterministic, always same successor) get
STALE_PENALTY=100 added to their count.

Tests whether "stuck" signal is in DELTA (visual change, step 636) or
DESTINATION (where you end up, step 637).

Same protocol: 5 seeds × 60s, LS20, k=16 LSH.
Compare L1 to 620 baseline.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
H_STALE_THRESH = 0.1    # entropy below this → edge stale (deterministic)
MIN_TRANSITIONS = 5     # minimum edge visits before checking entropy
STALE_PENALTY = 100
PER_SEED_TIME = 60

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

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
        self.stale_count = 0
        self.total_act = 0
        self.unique_cells = set()

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _edge_entropy(self, n, a):
        """Shannon entropy of successor distribution for edge (n, a).
        Returns None if fewer than MIN_TRANSITIONS observations.
        """
        d = self.G.get((n, a))
        if not d or sum(d.values()) < MIN_TRANSITIONS:
            return None
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _is_edge_stale(self, n, a):
        h = self._edge_entropy(n, a)
        if h is None:
            return False
        return h < H_STALE_THRESH

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.unique_cells.add(n)
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
        return n

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            if self._is_edge_stale(self._cn, a):
                modified = base_count + STALE_PENALTY
                self.stale_count += 1
            else:
                modified = base_count
            counts.append(modified)
            self.total_act += 1
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

    def stale_pct(self):
        if self.total_act == 0:
            return 0.0
        return 100.0 * self.stale_count / self.total_act

    def stale_edges_count(self):
        return sum(1 for (n, a) in self.G if self._is_edge_stale(n, a))


def t0():
    sub = Recode(seed=0)

    # No stale when no data
    assert not sub._is_edge_stale(1, 0)

    # Deterministic edge (always same successor) → stale
    sub.G[(1, 0)] = {99: 10}  # 10 transitions, all to node 99
    assert sub._is_edge_stale(1, 0), "deterministic edge should be stale"

    # Mixed successors → not stale
    sub.G[(2, 0)] = {10: 5, 11: 5}  # equal split → H=1.0
    assert not sub._is_edge_stale(2, 0), "mixed successors should not be stale"

    # Near-deterministic (99/100 same) → stale (H ≈ 0.08)
    sub.G[(3, 0)] = {10: 99, 11: 1}
    h = sub._edge_entropy(3, 0)
    assert h is not None and h < H_STALE_THRESH, \
        f"99/100 deterministic should be stale, H={h}"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 500_001):
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

    stale_pct = sub.stale_pct()
    stale_edges = sub.stale_edges_count()
    unique = len(sub.unique_cells)
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"stale_pct={stale_pct:.1f}% stale_edges={stale_edges} "
          f"unique={unique}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                stale_pct=stale_pct, stale_edges=stale_edges, unique=unique)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        print("\nDry run only. T0 passed.")
        return

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
    avg_stale_pct = np.mean([r['stale_pct'] for r in R])
    avg_stale_edges = np.mean([r['stale_edges'] for r in R])
    avg_unique = np.mean([r['unique'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"stale_pct={r['stale_pct']:.1f}% stale_edges={r['stale_edges']} "
              f"unique={r['unique']}")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_stale_pct={avg_stale_pct:.1f}%  avg_stale_edges={avg_stale_edges:.1f}  "
          f"avg_unique={avg_unique:.0f}")

    if avg_stale_pct > 1.0 and l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: entropy stale detection active AND faster than baseline.")
    elif avg_stale_pct > 1.0:
        print(f"PARTIAL: stale detection active ({avg_stale_pct:.1f}%), "
              f"L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print("INERT: entropy stale not firing.")


if __name__ == "__main__":
    main()
