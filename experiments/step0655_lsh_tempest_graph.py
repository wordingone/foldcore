"""
Step 655 — Tempest graph: composition substrate.

Standard LSH k=12 + 3-bit recency register per node.
Pattern-sensitive action selection: temporal visit patterns modulate counts.

PATTERN_WEIGHT: frontier (000) preferred, attractor (111) avoided.
Alonso-Sanz principle: adding memory to fixed operations changes complexity class.

Compare to Step 459 baseline: 6/10 seeds at 50K steps.
"""
import numpy as np
import time
import sys
from collections import defaultdict

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 25  # 10 seeds × 25s = 250s < 5 min
N_SEEDS = 10

BASELINE_L1 = [1362, 3270, 48391, 62727, 846, None, None, None, None, None]  # seeds 0-4 known

# Pattern weights: 000=frontier(prefer), 111=attractor(avoid), 101=oscillating(discourage)
PATTERN_WEIGHT = {
    0b000: 0.5,   # frontier — never recently visited → halve count → prefer
    0b001: 0.8,   # visited this step
    0b010: 1.0,   # visited 2 steps ago, not 1
    0b011: 1.2,   # visited 1 and 2 steps ago
    0b100: 0.7,   # visited 3 steps ago, recently free
    0b101: 1.3,   # oscillating (skip 1 step)
    0b110: 1.5,   # visited 1 and 2 steps ago but not 3
    0b111: 2.0,   # attractor — always here → double count → avoid
}


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class TempestRecode:
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
        # 3-bit recency: step_visits[cell] = set of recent step indices (pruned to last 3)
        self.step_visits = defaultdict(set)

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
        # Record visit for pattern computation
        visits = self.step_visits[n]
        visits.add(self.t)
        # Prune: keep only steps in [t-2, t] (3-step window)
        stale = {v for v in visits if v < self.t - 2}
        if stale:
            visits -= stale

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

    def _get_pattern(self, cell):
        """Compute 3-bit recency pattern: bit 0=this step, bit 1=prev, bit 2=2 ago."""
        t = self.t
        visits = self.step_visits.get(cell)
        if not visits:
            return 0b000
        p = 0
        for age in range(3):
            if (t - age) in visits:
                p |= (1 << age)
        return p

    def act(self):
        best_action = 0
        best_score = float('inf')

        for a in range(N_A):
            d = self.G.get((self._cn, a), {})
            if not d:
                # Never tried this action — zero effective count → strongly prefer
                effective_count = 0.0
            else:
                effective_count = sum(
                    cnt * PATTERN_WEIGHT[self._get_pattern(succ)]
                    for succ, cnt in d.items()
                )
            if effective_count < best_score:
                best_score = effective_count
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

    def pattern_dist(self):
        """Count distribution of 3-bit patterns across all known cells."""
        from collections import Counter
        counts = Counter()
        for cell in self.live:
            counts[self._get_pattern(cell)] += 1
        return dict(counts)


def run(seed, make):
    env = make()
    sub = TempestRecode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()
    pat_snapshot = None  # pattern dist at step 10K

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            continue

        sub.observe(obs)

        if step == 10000 and pat_snapshot is None:
            pat_snapshot = sub.pattern_dist()

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

    elapsed = time.time() - t_start
    bsl = BASELINE_L1[seed]
    if l1 and bsl:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    elif l1:
        spd = "no baseline"
    else:
        spd = "NO_L1"

    print(f"  s{seed}: L1={l1} ({spd}) L2={l2} go={go} "
          f"unique={len(sub.live)} t={elapsed:.1f}s", flush=True)

    if pat_snapshot:
        total = sum(pat_snapshot.values())
        if total > 0:
            pat_str = {bin(k): f"{100*v/total:.0f}%" for k, v in sorted(pat_snapshot.items())}
            print(f"    pat@10K: {pat_str}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go, unique=len(sub.live))


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Tempest graph: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")
    print(f"PATTERN_WEIGHT: {PATTERN_WEIGHT}")

    results = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
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
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) unique={r['unique']}")

    print(f"\nL1={l1_n}/{N_SEEDS}  avg_speedup={avg_ratio:.2f}x (vs baseline seeds 0-4)")
    # Step 459 baseline: 6/10 at 50K steps
    print(f"Step 459 baseline for comparison: 6/10 at 50K steps")

    if l1_n >= 6 and avg_ratio > 1.0:
        print("SIGNAL: Tempest graph faster than argmin baseline")
    elif l1_n >= 6:
        print(f"NEUTRAL: L1={l1_n}/10 matched 459 baseline but not faster")
    elif l1_n < 4:
        print("KILL: L1 < 4/10")
    else:
        print(f"MARGINAL: L1={l1_n}/10")


if __name__ == "__main__":
    main()
