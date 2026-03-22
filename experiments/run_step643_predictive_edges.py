"""
Step 643 — Predictive edges (passive data collection).

Each edge (N, A) stores:
  count(N, A): visit count (unchanged from baseline)
  pred(N, A): most-frequent-successor-cell (plurality vote from G)
  surprise_count(N, A): # traversals where successor != pred

Surprise check happens BEFORE G update, so pred is from prior observations.

Action selection: unchanged (argmin on counts only). Passive data only.

Metrics:
1. L1 rate vs baseline (should match — selection unchanged)
2. surprise_rate = surprise_count / count per edge
3. Distribution: % edges at surprise_rate 0%, <10%, 10-50%, >50%
4. mean/std of surprise_rate across edges

Purpose: test whether surprise_rate varies meaningfully across edges (→ useful signal).
If all edges have similar rates, environment is too deterministic for this to differentiate.

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
        # Surprise tracking per edge
        self.surprise_count = defaultdict(int)  # (N, A) -> # surprises

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
            # Check surprise BEFORE updating G (pred uses prior observations)
            d_prev = self.G.get((self._pn, self._pa))
            if d_prev:
                pred = max(d_prev, key=d_prev.get)
                if n != pred:
                    self.surprise_count[(self._pn, self._pa)] += 1

            # Update G as normal
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

    def surprise_metrics(self):
        """
        Compute per-edge surprise_rate = surprise_count / count.
        Returns (mean_sr, std_sr, pct_zero, pct_low, pct_mid, pct_high, n_edges_qualified).
        Only considers edges with count >= 5 (enough data).
        """
        rates = []
        for (n, a), d in self.G.items():
            cnt = sum(d.values())
            if cnt < 5:
                continue
            sc = self.surprise_count.get((n, a), 0)
            rates.append(sc / cnt)

        if not rates:
            return 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0

        rates = np.array(rates)
        mean_sr = float(np.mean(rates))
        std_sr = float(np.std(rates))
        pct_zero = 100.0 * np.mean(rates == 0.0)
        pct_low = 100.0 * np.mean((rates > 0.0) & (rates < 0.1))
        pct_mid = 100.0 * np.mean((rates >= 0.1) & (rates < 0.5))
        pct_high = 100.0 * np.mean(rates >= 0.5)
        return mean_sr, std_sr, pct_zero, pct_low, pct_mid, pct_high, len(rates)


def t0():
    sub = Recode(seed=0)

    # No surprise before any data
    assert sub.surprise_count[(1, 0)] == 0

    # Manually set up G to test surprise detection
    sub.G[(1, 0)] = {99: 10}  # pred = 99
    sub._pn = 1
    sub._pa = 0
    # Fake observe: successor = 99 (matches pred, no surprise)
    d_prev = sub.G.get((sub._pn, sub._pa))
    pred = max(d_prev, key=d_prev.get)
    assert pred == 99
    # Successor 100 != pred → surprise
    if 100 != pred:
        sub.surprise_count[(1, 0)] += 1
    assert sub.surprise_count[(1, 0)] == 1, "surprise should have fired"

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

    mean_sr, std_sr, pct_zero, pct_low, pct_mid, pct_high, n_edges = sub.surprise_metrics()
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"mean_sr={mean_sr:.3f}±{std_sr:.3f} "
          f"zero={pct_zero:.0f}% low={pct_low:.0f}% mid={pct_mid:.0f}% high={pct_high:.0f}% "
          f"n_edges={n_edges}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                mean_sr=mean_sr, std_sr=std_sr,
                pct_zero=pct_zero, pct_low=pct_low,
                pct_mid=pct_mid, pct_high=pct_high, n_edges=n_edges)


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
    avg_mean_sr = np.mean([r['mean_sr'] for r in R])
    avg_std_sr = np.mean([r['std_sr'] for r in R])
    avg_zero = np.mean([r['pct_zero'] for r in R])
    avg_high = np.mean([r['pct_high'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"mean_sr={r['mean_sr']:.3f}±{r['std_sr']:.3f} "
              f"zero={r['pct_zero']:.0f}% high={r['pct_high']:.0f}% n_edges={r['n_edges']}")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_mean_sr={avg_mean_sr:.3f}±{avg_std_sr:.3f}  "
          f"avg_zero={avg_zero:.0f}%  avg_high={avg_high:.0f}%")

    # Signal assessment
    if avg_std_sr > 0.1:
        print("SIGNAL: surprise_rate varies widely across edges — differentiated signal.")
    elif avg_mean_sr < 0.05 and avg_zero > 80:
        print("DETERMINISTIC: most edges have zero surprise — environment near-deterministic.")
    else:
        print(f"PARTIAL: mean_sr={avg_mean_sr:.3f}, std={avg_std_sr:.3f}. "
              f"Moderate variation. Investigate distribution.")


if __name__ == "__main__":
    main()
