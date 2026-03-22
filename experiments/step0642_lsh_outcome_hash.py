"""
Step 642 — Outcome-hash edges (passive data collection).

Each edge (N, A) stores:
  count(N, A): visit count (unchanged from baseline)
  outcomes_hash(N, A): rolling XOR hash of successor cells seen

Action selection: unchanged (argmin on counts only). Passive data only.

Metrics:
1. L1 rate vs baseline (should match — selection unchanged)
2. avg_distinct: avg # distinct outcome hashes per cell (across explored actions)
3. n_unique_sigs: # cells with unique 4-tuple outcome signature
4. pct_all_distinct: % cells where all explored actions have different hashes

Purpose: test whether outcome hashing produces differentiated profiles under argmin.
If yes → meta-transfer on outcome similarity (Step 644).

5 seeds × 50K steps max, LS20, k=16 LSH.
Compare L1 to 620 baseline.
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
MAX_STEPS = 50_001
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
        # Outcome hash per edge: XOR of cell hashes
        self.outcomes_hash = defaultdict(int)  # (N, A) -> rolling XOR

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    @staticmethod
    def _cell_hash(n):
        """Deterministic 64-bit hash for a cell ID (for XOR chaining)."""
        return hash(n) & 0xFFFFFFFFFFFFFFFF

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._node(x)
        self.live.add(n)
        self.t += 1

        if self._pn is not None:
            # Update outcomes_hash BEFORE G update (order doesn't matter here, but explicit)
            self.outcomes_hash[(self._pn, self._pa)] ^= self._cell_hash(n)

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

    def outcome_metrics(self):
        """
        Per-cell: collect outcome hashes for each explored action.
        Returns (avg_distinct, n_unique_sigs, pct_all_distinct).
        """
        cells_2plus = []
        for n in self.live:
            explored = [a for a in range(N_A) if self.G.get((n, a)) is not None]
            if len(explored) >= 2:
                oh = tuple(self.outcomes_hash.get((n, a), 0) for a in explored)
                cells_2plus.append((n, oh, len(explored)))

        if not cells_2plus:
            return 0.0, 0, 0.0

        # Avg distinct hash values per cell
        distinctnesses = [len(set(oh)) for _, oh, _ in cells_2plus]
        avg_distinct = float(np.mean(distinctnesses))

        # Unique 4-tuple signatures (full cell signature across all 4 actions)
        full_sigs = []
        for n in self.live:
            if len([a for a in range(N_A) if self.G.get((n, a)) is not None]) >= 2:
                sig = tuple(self.outcomes_hash.get((n, a), 0) for a in range(N_A))
                full_sigs.append(sig)
        n_unique_sigs = len(set(full_sigs))

        # % cells where all explored actions have distinct hashes
        all_distinct = sum(1 for _, oh, cnt in cells_2plus if len(set(oh)) == len(oh))
        pct_all_distinct = 100.0 * all_distinct / len(cells_2plus)

        return avg_distinct, n_unique_sigs, pct_all_distinct


def t0():
    sub = Recode(seed=0)

    # outcomes_hash starts at 0
    assert sub.outcomes_hash[(1, 0)] == 0

    # XOR updates: same cell twice → back to 0
    h = sub._cell_hash(99)
    sub.outcomes_hash[(1, 0)] ^= h
    sub.outcomes_hash[(1, 0)] ^= h
    assert sub.outcomes_hash[(1, 0)] == 0, "double XOR should cancel"

    # Two different cells → nonzero
    sub.outcomes_hash[(1, 0)] ^= sub._cell_hash(99)
    sub.outcomes_hash[(1, 0)] ^= sub._cell_hash(100)
    assert sub.outcomes_hash[(1, 0)] != 0, "different cells should produce nonzero hash"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
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

    avg_distinct, n_unique_sigs, pct_all_distinct = sub.outcome_metrics()
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"avg_distinct={avg_distinct:.2f} unique_sigs={n_unique_sigs} "
          f"pct_all_distinct={pct_all_distinct:.1f}% live={len(sub.live)}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                avg_distinct=avg_distinct, n_unique_sigs=n_unique_sigs,
                pct_all_distinct=pct_all_distinct, live=len(sub.live))


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
    avg_distinct = np.mean([r['avg_distinct'] for r in R])
    avg_sigs = np.mean([r['n_unique_sigs'] for r in R])
    avg_pct = np.mean([r['pct_all_distinct'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"avg_distinct={r['avg_distinct']:.2f} unique_sigs={r['n_unique_sigs']} "
              f"pct_all_distinct={r['pct_all_distinct']:.1f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_distinct={avg_distinct:.2f}  avg_unique_sigs={avg_sigs:.0f}  "
          f"avg_pct_all_distinct={avg_pct:.1f}%")

    if avg_distinct >= 2.0:
        print("SIGNAL: outcome hashes differentiated — proceed to Step 644.")
    elif avg_distinct >= 1.5:
        print("PARTIAL: moderate differentiation. Investigate distribution.")
    else:
        print("FLAT: outcome hashes near-uniform. Profile uniformity persists beyond counts.")


if __name__ == "__main__":
    main()
