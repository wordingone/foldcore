"""
Step 630 — Delta-augmented graph on LS20.

Extend Recode graph with delta_cell tracking. Action selection:
- PREFER: action A at node N whose typical delta is in productive_set
- AVOID:  action A at node N whose typical delta is in stale_set (>80% bg)
- NEUTRAL: otherwise

Productive set: built R1-compliantly — when L1 triggers, tag the
delta_cell of the L1-triggering transition.

Stale set: delta_cells appearing on >80% of all transitions (background noise).
Computed after WARMUP=500 steps, updated every STALE_UPDATE_EVERY steps.

5 seeds × 60s. Signal: L1 faster than 620 baseline.
Kill: L1 same or worse.

620 baseline L1 steps: [1362, 3270, 48391, 62727, 846]
"""
import numpy as np
import time
import sys
from collections import Counter, defaultdict

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
WARMUP = 500
STALE_UPDATE_EVERY = 1000
PER_SEED_TIME = 60
AVOID_PENALTY = 100
PREFER_BONUS = 50

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]  # step 620 baseline


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def enc_delta(curr_frame, prev_frame):
    a = np.array(curr_frame[0], dtype=np.float32) / 15.0
    b = np.array(prev_frame[0], dtype=np.float32) / 15.0
    d = (a - b).reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return d - d.mean()


class Recode:

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._prev_frame = None
        self._last_dc = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        # Delta tracking
        self.delta_by_action = [Counter() for _ in range(N_A)]
        self.delta_by_na = defaultdict(Counter)
        self.total_transitions = 0
        # Action selection
        self.productive_set = set()
        self.stale_set = set()
        self.op_counts = [0, 0, 0]  # [neutral, avoid, prefer]

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

            if self._prev_frame is not None:
                dv = enc_delta(frame, self._prev_frame)
                dc = self._hash(dv)
                self._last_dc = dc
                self.delta_by_action[self._pa][dc] += 1
                self.delta_by_na[(self._pn, self._pa)][dc] += 1
                self.total_transitions += 1

        self._px = x
        self._cn = n
        self._prev_frame = frame

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t >= WARMUP and self.t % STALE_UPDATE_EVERY == 0:
            self._update_stale()
        return n

    def tag_productive(self):
        """Tag current _last_dc as productive (L1-triggering transition)."""
        if self._last_dc is not None:
            self.productive_set.add(self._last_dc)

    def _update_stale(self):
        """Mark delta_cells appearing on >80% of all transitions as stale."""
        if self.total_transitions == 0:
            return
        combined = Counter()
        for a in range(N_A):
            combined.update(self.delta_by_action[a])
        total = sum(combined.values())
        self.stale_set = {dc for dc, cnt in combined.items() if cnt / total > 0.8}

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            na_ctr = self.delta_by_na.get((self._cn, a))
            if na_ctr:
                top_dc = na_ctr.most_common(1)[0][0]
                if top_dc in self.productive_set:
                    modified = max(0, base_count - PREFER_BONUS)
                    self.op_counts[2] += 1
                elif top_dc in self.stale_set:
                    modified = base_count + AVOID_PENALTY
                    self.op_counts[1] += 1
                else:
                    modified = base_count
                    self.op_counts[0] += 1
            else:
                modified = base_count
                self.op_counts[0] += 1
            counts.append(modified)
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_reset(self):
        self._pn = None
        self._prev_frame = None

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

    def op_stats(self):
        total = sum(self.op_counts)
        if total == 0:
            return 0.0, 0.0, 100.0
        return (
            100.0 * self.op_counts[1] / total,
            100.0 * self.op_counts[2] / total,
            100.0 * self.op_counts[0] / total,
        )


def t0():
    rng = np.random.RandomState(42)
    sub = Recode(seed=0)

    # Test tag_productive
    sub._last_dc = 42
    sub.tag_productive()
    assert 42 in sub.productive_set, "productive_set should contain _last_dc"

    # Test _update_stale: cell with >80% frequency becomes stale
    sub2 = Recode(seed=0)
    sub2.delta_by_action[0][100] = 900
    sub2.delta_by_action[0][200] = 100
    sub2.total_transitions = 1000
    sub2._update_stale()
    assert 100 in sub2.stale_set, "90%-frequency cell should be stale"
    assert 200 not in sub2.stale_set, "10%-frequency cell should not be stale"

    # Test act: prefer productive, avoid stale
    sub3 = Recode(seed=0)
    sub3._cn = 99
    sub3.G[(99, 0)] = {100: 5}
    sub3.G[(99, 1)] = {100: 5}
    sub3.G[(99, 2)] = {100: 5}
    sub3.G[(99, 3)] = {100: 5}
    sub3.delta_by_na[(99, 0)][42] = 10   # action 0 → productive delta
    sub3.delta_by_na[(99, 1)][99] = 10   # action 1 → stale delta
    sub3.delta_by_na[(99, 2)][77] = 10   # action 2 → neutral
    sub3.delta_by_na[(99, 3)][77] = 10   # action 3 → neutral
    sub3.productive_set = {42}
    sub3.stale_set = {99}
    action = sub3.act()
    assert action == 0, f"should prefer action 0 (productive delta), chose {action}"

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
                sub.tag_productive()
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

    nc = len(sub.live)
    ns = sub.ns
    ne = len(sub.G)
    av, pr, nt = sub.op_stats()
    prod = len(sub.productive_set)
    stale = len(sub.stale_set)

    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        print(f"  s{seed} L1@{l1} ({spd} vs baseline {bsl}) "
              f"c={nc} sp={ns} e={ne} go={go} "
              f"prod={prod} stale={stale} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%",
              flush=True)
    else:
        print(f"  s{seed} NO_L1 c={nc} sp={ns} e={ne} go={go} "
              f"prod={prod} stale={stale} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%",
              flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go, cells=nc, splits=ns, edges=ne,
                prod=prod, stale=stale, avoid_pct=av, prefer_pct=pr)


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
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: {tag:>3}  l1={r['l1']}  baseline={bsl}  {spd}  "
              f"prod={r['prod']}  stale={r['stale']}  "
              f"ops=A{r['avoid_pct']:.0f}%P{r['prefer_pct']:.0f}%")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
    if l1_valid:
        ratios = [b / l for l, b in l1_valid]
        avg_ratio = np.mean(ratios)
    else:
        avg_ratio = 0.0

    print(f"\nL1={l1n}/5  L2={l2n}/5  avg_speedup={avg_ratio:.2f}x")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: L1 faster than baseline — delta navigation is productive.")
    elif l1n < 3:
        print("KILL: L1 < 3/5 — delta selection interferes.")
    else:
        print(f"MARGINAL: L1={l1n}/5  avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
