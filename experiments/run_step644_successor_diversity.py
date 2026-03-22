"""
Step 644 — Successor diversity tie-breaking.

Each edge (N, A) tracks n_unique_successors = len(G[(N,A)]).

Action selection: argmin on count. Ties broken by LOWEST n_unique_successors
(prefer deterministic edges). If count AND diversity tied, pick first.

Hypothesis: deterministic edges are navigation; stochastic edges are death/noise.
Tie-breaking toward determinism = R1-compliant death-avoidance without prescribed penalty.

5 seeds × 50K steps max, LS20, k=16 LSH.
Compare L1 to 620 baseline [1362, 3270, 48391, 62727, 846].
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
        # Stats
        self.tie_count = 0
        self.div_changed = 0  # ties where diversity broke the tie differently
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
        min_count = min(counts)
        tied = [a for a, c in enumerate(counts) if c == min_count]
        self.total_act += 1

        if len(tied) == 1:
            action = tied[0]
        else:
            self.tie_count += 1
            # Break ties by lowest n_unique_successors (most deterministic)
            diversities = [len(self.G.get((self._cn, a), {})) for a in tied]
            min_div = min(diversities)
            action = next(a for a, div in zip(tied, diversities) if div == min_div)
            if action != tied[0]:
                self.div_changed += 1

        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None

    def diversity_metrics(self):
        """Avg and distribution of n_unique_successors per edge."""
        divs = [len(d) for d in self.G.values() if sum(d.values()) >= 3]
        if not divs:
            return 0.0, 0.0, 0
        return float(np.mean(divs)), float(np.std(divs)), len(divs)

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

    # No successors → diversity = 0 → all tied → pick first
    sub._cn = 1
    sub.live.add(1)
    action = sub.act()
    assert action == 0, "all zero diversity: should pick first"

    # With diversity data: prefer lower diversity in ties
    sub._cn = 2
    sub.live.add(2)
    # All 4 actions tied at count=3; a=1 has diversity=1 (lowest) → wins tie-break
    sub.G[(2, 0)] = {10: 1, 11: 1, 12: 1}  # count=3, diversity=3
    sub.G[(2, 1)] = {20: 3}                 # count=3, diversity=1 → wins
    sub.G[(2, 2)] = {30: 1, 31: 1, 32: 1}  # count=3, diversity=3
    sub.G[(2, 3)] = {40: 1, 41: 1, 42: 1}  # count=3, diversity=3
    action = sub.act()
    assert action == 1, "a=1 has lowest diversity: should win tie-break"

    # All same diversity → pick first tied
    sub.G[(2, 1)] = {20: 1, 21: 1, 22: 1}  # diversity=3, same as rest
    action = sub.act()
    assert action == 0, "same diversity: pick first tied"

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

    avg_div, std_div, n_edges = sub.diversity_metrics()
    tie_rate = 100.0 * sub.tie_count / sub.total_act if sub.total_act else 0.0
    changed_rate = 100.0 * sub.div_changed / sub.tie_count if sub.tie_count else 0.0
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"avg_div={avg_div:.2f}±{std_div:.2f} n_edges={n_edges} "
          f"tie={tie_rate:.1f}% changed={changed_rate:.0f}%", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                avg_div=avg_div, std_div=std_div, n_edges=n_edges,
                tie_rate=tie_rate, changed_rate=changed_rate)


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
    avg_div = np.mean([r['avg_div'] for r in R])
    avg_changed = np.mean([r['changed_rate'] for r in R if r['tie_rate'] > 0])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"avg_div={r['avg_div']:.2f} changed={r['changed_rate']:.0f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  "
          f"avg_div={avg_div:.2f}  avg_changed={avg_changed:.0f}%")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: diversity tie-breaking faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print(f"MARGINAL: L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
