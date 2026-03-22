"""
Step 635 — Frontier-gradient action selection.

For each (cell, action) pair, bias toward actions leading to rarely-visited successors.

frontier_score(S) = 1 / (1 + total_visits(S))
avg_frontier_score(N, A) = mean(frontier_score(S)) for S in successors(N, A)
effective_count(N, A) = count(N, A) - FRONTIER_BONUS * avg_frontier_score(N, A)
action = argmin(effective_count)

No delta tracking. No self-observation. Pure local graph structure.

5 seeds × 60s, LS20, k=16 LSH. FRONTIER_BONUS=50.
Compare L1 and unique cells reached to 620 baseline.
"""
import numpy as np
import time
import sys
from collections import defaultdict

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
FRONTIER_BONUS = 50
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
        self.G = {}       # (n, a) -> {successor: count}
        self.C = {}       # (n, a, succ) -> (sum_x, count) for refinement
        self.V = defaultdict(int)   # total_visits per node
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.unique_cells = set()
        self.frontier_ops = 0
        self.total_ops = 0

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
        self.V[n] += 1
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

    def _frontier_score(self, n, a):
        """Average frontier score of successors of (n, a).
        frontier_score(S) = 1 / (1 + total_visits(S)).
        Returns 0 if no successors known.
        """
        d = self.G.get((n, a))
        if not d:
            return 0.0
        scores = [1.0 / (1.0 + self.V[s]) for s in d]
        return float(np.mean(scores))

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            fs = self._frontier_score(self._cn, a)
            effective = base_count - FRONTIER_BONUS * fs
            counts.append(effective)
            if fs > 0:
                self.frontier_ops += 1
            self.total_ops += 1
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


def t0():
    sub = Recode(seed=0)
    rng = np.random.RandomState(42)

    # _hash returns int
    x = rng.randn(DIM).astype(np.float32)
    assert isinstance(sub._hash(x), int)

    # frontier_score = 0 when no successors
    sub._cn = 999
    assert sub._frontier_score(999, 0) == 0.0

    # Manually set up G and V to test frontier scoring
    sub.G[(1, 0)] = {10: 3, 11: 2}
    sub.V[10] = 0   # never visited → high frontier score
    sub.V[11] = 100  # heavily visited → low frontier score
    fs = sub._frontier_score(1, 0)
    # Expected: mean([1/(1+0), 1/(1+100)]) = mean([1.0, ~0.0099]) ≈ 0.505
    assert 0.4 < fs < 0.6, f"frontier_score expected ~0.5, got {fs}"

    # With high visit counts, score should be much lower
    sub.V[10] = 50
    fs2 = sub._frontier_score(1, 0)
    assert fs2 < fs, "More visits should lower frontier score"

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

    unique = len(sub.unique_cells)
    frac = 100.0 * sub.frontier_ops / sub.total_ops if sub.total_ops else 0.0
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} unique_cells={unique} "
          f"frontier_active={frac:.0f}%", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go, unique=unique,
                frontier_pct=frac)


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
    avg_unique = np.mean([r['unique'] for r in R])
    avg_fp = np.mean([r['frontier_pct'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) unique={r['unique']} "
              f"frontier_active={r['frontier_pct']:.0f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  avg_unique={avg_unique:.0f}  "
          f"frontier_active={avg_fp:.0f}%")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: frontier-gradient faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print("MARGINAL: L1 >= 3/5 but no clear speedup.")


if __name__ == "__main__":
    main()
