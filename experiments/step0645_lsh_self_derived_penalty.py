"""
Step 645 — Self-derived penalty (R3 version of 581d).

Instead of prescribed PENALTY=100, the penalty IS the surprise_count:
  effective_count(N, A) = count(N, A) + surprise_count(N, A)

surprise_count(N, A) = # times action A from cell N led to a cell != pred(N,A).
pred(N, A) = most-frequent-successor-cell (plurality vote of past observations).
Surprise check happens BEFORE G update (uses prior observations).

Death edges accumulate high surprise (reset → variable successor) → naturally avoided.
No designer-prescribed magnitude. Penalty is data-derived.

This is the R3-compliant version of 581d.

5 seeds × 50K steps max, LS20, k=16 LSH.
Compare L1 to 620 baseline [1362, 3270, 48391, 62727, 846]
and 581d (PENALTY=100, ~7.0x faster on best seeds).
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
        self.surprise_count = defaultdict(int)  # (N, A) -> # surprises
        # Stats
        self.penalty_active = 0  # steps where any action had nonzero surprise penalty
        self.total_act = 0

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
        effective = []
        any_penalty = False
        for a in range(N_A):
            count = sum(self.G.get((self._cn, a), {}).values())
            surprise = self.surprise_count.get((self._cn, a), 0)
            if surprise > 0:
                any_penalty = True
            effective.append(count + surprise)

        action = int(np.argmin(effective))
        self.total_act += 1
        if any_penalty:
            self.penalty_active += 1

        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def surprise_metrics(self):
        """Distribution of surprise_rate = surprise_count/count per edge (count>=5)."""
        rates = []
        for (n, a), d in self.G.items():
            cnt = sum(d.values())
            if cnt < 5:
                continue
            sc = self.surprise_count.get((n, a), 0)
            rates.append(sc / cnt)
        if not rates:
            return 0.0, 0.0, 0.0, 0
        rates = np.array(rates)
        return (float(np.mean(rates)), float(np.std(rates)),
                100.0 * np.mean(rates > 0.1), len(rates))

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

    # All actions equal count=5, no surprise → pick first (a=0)
    sub._cn = 2
    sub.live.add(2)
    sub.G[(2, 0)] = {99: 5}
    sub.G[(2, 1)] = {100: 5}
    sub.G[(2, 2)] = {101: 5}
    sub.G[(2, 3)] = {102: 5}
    action = sub.act()
    assert action == 0, "no surprises: pick first"

    # Add surprise to a=0, a=2, a=3 → a=1 wins (effective=5, others=8)
    sub.surprise_count[(2, 0)] = 3
    sub.surprise_count[(2, 2)] = 3
    sub.surprise_count[(2, 3)] = 3
    action = sub.act()
    assert action == 1, "a=1 has no surprise, lower effective count wins"

    # Remove surprise → back to first (a=0)
    sub.surprise_count[(2, 0)] = 0
    sub.surprise_count[(2, 2)] = 0
    sub.surprise_count[(2, 3)] = 0
    action = sub.act()
    assert action == 0, "zero surprise: argmin on count, pick first"

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

    mean_sr, std_sr, pct_high, n_edges = sub.surprise_metrics()
    penalty_rate = 100.0 * sub.penalty_active / sub.total_act if sub.total_act else 0.0
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"mean_sr={mean_sr:.3f}±{std_sr:.3f} pct_high_sr={pct_high:.0f}% "
          f"penalty_active={penalty_rate:.1f}% n_edges={n_edges}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                mean_sr=mean_sr, std_sr=std_sr,
                pct_high_sr=pct_high, penalty_rate=penalty_rate, n_edges=n_edges)


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
    avg_sr = np.mean([r['mean_sr'] for r in R])
    avg_pen = np.mean([r['penalty_rate'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"mean_sr={r['mean_sr']:.3f}±{r['std_sr']:.3f} "
              f"pct_high={r['pct_high_sr']:.0f}% penalty_active={r['penalty_rate']:.1f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_mean_sr={avg_sr:.3f}  avg_penalty_active={avg_pen:.1f}%")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: self-derived penalty faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print(f"MARGINAL: L1={l1n}/5 avg_speedup={avg_ratio:.2f}x avg_sr={avg_sr:.3f}")


if __name__ == "__main__":
    main()
