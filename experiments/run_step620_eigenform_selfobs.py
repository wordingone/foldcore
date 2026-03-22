"""
Step 620 — Eigenform self-observation on L1.

Base: LSH k=16, centered_enc, argmin (Step 542 Recode).
Addition: self-derived op codes from graph statistics every M=500 steps.

Every 500 steps the substrate observes its own graph:
  1. self_signal[n][a] = edge_count[n][a] / sum(edge_counts[n])
  2. p90, p10 = percentiles across all self_signals
  3. op codes: sig > p90 → AVOID (+100), sig < p10 → PREFER (-50), else NEUTRAL
  4. argmin uses op-modified edge counts between observations

This is INTERNAL self-observation (eigenform F(s)(enc(s))), NOT external targets.
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
SELF_OBS_EVERY = 500
PER_SEED_TIME = 60

OP_NEUTRAL = 0
OP_AVOID = 1
OP_PREFER = 2
AVOID_PENALTY = 100
PREFER_BONUS = 50


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


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
        # Op codes: {(n, a): OP_AVOID/OP_PREFER/OP_NEUTRAL}
        self.ops = {}
        # Cumulative op usage counts: [neutral, avoid, prefer]
        self.op_counts = [0, 0, 0]
        # Snapshots: [(t, avoid%, prefer%, neutral%)]
        self.op_history = []

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
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        self.dim = len(x)
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        if self.t > 0 and self.t % SELF_OBS_EVERY == 0:
            self._self_observe()
        return n

    def act(self):
        counts = []
        for a in range(N_A):
            base_count = sum(self.G.get((self._cn, a), {}).values())
            op = self.ops.get((self._cn, a), OP_NEUTRAL)
            if op == OP_AVOID:
                modified = base_count + AVOID_PENALTY
                self.op_counts[1] += 1
            elif op == OP_PREFER:
                modified = max(0, base_count - PREFER_BONUS)
                self.op_counts[2] += 1
            else:
                modified = base_count
                self.op_counts[0] += 1
            counts.append(modified)
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def _self_observe(self):
        """Recompute op codes from current graph structure."""
        na_signals = {}
        nodes = set(n for (n, a) in self.G)
        for n in nodes:
            total = sum(sum(self.G.get((n, a), {}).values()) for a in range(N_A))
            if total == 0:
                continue
            for a in range(N_A):
                edge_count = sum(self.G.get((n, a), {}).values())
                if edge_count == 0:
                    continue
                na_signals[(n, a)] = edge_count / total

        if len(na_signals) < 10:
            return  # not enough data

        signals_arr = np.array(list(na_signals.values()))
        p90 = np.percentile(signals_arr, 90)
        p10 = np.percentile(signals_arr, 10)

        new_ops = {}
        avoid_n = prefer_n = neutral_n = 0
        for (n, a), sig in na_signals.items():
            if sig > p90:
                new_ops[(n, a)] = OP_AVOID
                avoid_n += 1
            elif sig < p10:
                new_ops[(n, a)] = OP_PREFER
                prefer_n += 1
            else:
                new_ops[(n, a)] = OP_NEUTRAL
                neutral_n += 1

        self.ops = new_ops
        total_na = avoid_n + prefer_n + neutral_n
        if total_na > 0:
            self.op_history.append((
                self.t,
                100.0 * avoid_n / total_na,
                100.0 * prefer_n / total_na,
                100.0 * neutral_n / total_na,
            ))

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
    sub = Recode(dim=8, k=3, seed=0)
    sub.H = rng.randn(3, 8).astype(np.float32)
    sub.dim = 8

    x1 = rng.randn(8).astype(np.float32)
    x2 = x1 + 0.001 * rng.randn(8).astype(np.float32)
    x3 = -x1

    n1 = sub._node(x1)
    n2 = sub._node(x2)
    n3 = sub._node(x3)
    assert n1 == n2, f"local continuity: {n1} != {n2}"
    assert n1 != n3, f"discrimination: {n1} == {n3}"

    # Verify self-observe doesn't crash on sparse graph
    sub2 = Recode(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 8
    sub2.G = {
        (0, 0): {1: 50, 2: 50},
        (0, 1): {1: 100},
    }
    sub2.live = {0, 1, 2}
    sub2.t = 1
    sub2._self_observe()  # < 10 signals, should return early without crash

    # Enough signals to compute percentiles — use clear outliers so p90/p10 separation works
    # node 0: sig[0,0]=0.999 (outlier high), sig[0,1]=0.001 (outlier low)
    # nodes 1-19: sig[i,0]=sig[i,1]=0.5 (uniform middle)
    # → p90 ≈ 0.5, so 0.999 > p90; p10 ≈ 0.5, so 0.001 < p10
    sub3 = Recode(dim=8, k=3, seed=0)
    sub3.H = sub.H.copy()
    sub3.dim = 8
    sub3.G[(0, 0)] = {1: 999}
    sub3.G[(0, 1)] = {1: 1}
    for i in range(1, 20):
        sub3.G[(i, 0)] = {i+1: 50}
        sub3.G[(i, 1)] = {i+1: 50}
    sub3.live = set(range(25))
    sub3.t = 1
    sub3._self_observe()
    assert len(sub3.ops) > 0, "op codes not assigned"
    avoid_count = sum(1 for v in sub3.ops.values() if v == OP_AVOID)
    prefer_count = sum(1 for v in sub3.ops.values() if v == OP_PREFER)
    assert avoid_count > 0, "no AVOID ops assigned"
    assert prefer_count > 0, "no PREFER ops assigned"

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

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            nc, ns, ne = sub.stats()
            av, pr, nt = sub.op_stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
            level = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    nc, ns, ne = sub.stats()
    av, pr, nt = sub.op_stats()
    hist = sub.op_history

    # Self-calibration: did distribution change over time?
    hist_summary = ""
    if len(hist) >= 2:
        first = hist[0]
        last = hist[-1]
        hist_summary = (f" hist_first=A{first[1]:.0f}%P{first[2]:.0f}%"
                        f" hist_last=A{last[1]:.0f}%P{last[2]:.0f}%")

    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go,
                avoid_pct=av, prefer_pct=pr, neutral_pct=nt,
                op_history=hist, hist_summary=hist_summary)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        print("\nDry run only (no ARC environment). T0 passed.")
        return

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        av, pr, nt = r['avoid_pct'], r['prefer_pct'], r['neutral_pct']
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  "
              f"sp={r['splits']:>3}  e={r['edges']:>5}  go={r['go']}  "
              f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%{r['hist_summary']}")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)

    print(f"\nL1={l1n}/5  L2={l2n}/5  max_cells={mc}  max_splits={ms}")

    if l1n >= 4:
        print("SIGNAL: L1 >= 4/5 — eigenform self-obs matches/beats 582 baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5 — self-obs does not maintain L1 baseline.")
    else:
        print(f"MARGINAL: L1={l1n}/5")


if __name__ == "__main__":
    main()
