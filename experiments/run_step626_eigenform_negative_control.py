"""
Step 626 — Eigenform negative control.

Run Step 620 but DISABLE self-observation after step 5000.
Op codes freeze at their last values from step 5000.

Signal: performance DROPS after disabling (proves self-observation is ongoing/load-bearing)
Kill: performance unchanged (self-observation was inert)

Reporting: track L1 step, compare seeds where L1 found before vs after step 5000 disable point.
Also report AVOID% at freeze vs AVOID% at end (should stay constant if frozen).
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
DISABLE_AT_STEP = 5000
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
        self.ops = {}
        self.op_counts = [0, 0, 0]
        self.op_history = []
        self._obs_disabled = False
        self._frozen_dist = None  # op distribution at freeze point

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
        # Disable self-observation after DISABLE_AT_STEP
        if self.t == DISABLE_AT_STEP and not self._obs_disabled:
            self._obs_disabled = True
            av, pr, nt = self._current_dist()
            self._frozen_dist = (av, pr, nt)
        if not self._obs_disabled and self.t > 0 and self.t % SELF_OBS_EVERY == 0:
            self._self_observe()
        return n

    def _current_dist(self):
        total_na = len(self.ops)
        if total_na == 0:
            return 0.0, 0.0, 100.0
        av = 100.0 * sum(1 for v in self.ops.values() if v == OP_AVOID) / total_na
        pr = 100.0 * sum(1 for v in self.ops.values() if v == OP_PREFER) / total_na
        nt = 100.0 * (total_na - sum(1 for v in self.ops.values() if v in (OP_AVOID, OP_PREFER))) / total_na
        return av, pr, nt

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
        if self._obs_disabled:
            return
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
            return

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
            av = 100.0 * avoid_n / total_na
            pr = 100.0 * prefer_n / total_na
            nt = 100.0 * neutral_n / total_na
            self.op_history.append((self.t, av, pr, nt))

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
    assert isinstance(sub._node(x1), int)

    # Test disable: build graph with outliers, observe, then disable
    sub2 = Recode(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 8
    sub2.G[(0, 0)] = {1: 999}
    sub2.G[(0, 1)] = {1: 1}
    for i in range(1, 20):
        sub2.G[(i, 0)] = {i+1: 50}
        sub2.G[(i, 1)] = {i+1: 50}
    sub2.live = set(range(25))
    sub2.t = 1
    sub2._self_observe()
    initial_ops = dict(sub2.ops)
    assert len(initial_ops) > 0

    # Disable and verify ops don't change on subsequent _self_observe calls
    sub2._obs_disabled = True
    sub2.G[(0, 0)] = {1: 1}   # flip the distribution
    sub2.G[(0, 1)] = {1: 999}
    # Would normally cause ops to flip, but we're disabled
    sub2._self_observe()  # should be skipped
    assert sub2.ops == initial_ops, "ops should be frozen after disable"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()
    printed_disable = False

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

        if sub._obs_disabled and not printed_disable:
            av, pr, nt = sub.op_stats()
            fd = sub._frozen_dist or (0.0, 0.0, 100.0)
            print(f"  s{seed} DISABLED@{sub.t} frozen_ops=A{fd[0]:.0f}%P{fd[1]:.0f}%N{fd[2]:.0f}%", flush=True)
            printed_disable = True

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            nc, ns, ne = sub.stats()
            av, pr, nt = sub.op_stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"disabled={sub._obs_disabled} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} e={ne} go={go}", flush=True)
            level = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    nc, ns, ne = sub.stats()
    av, pr, nt = sub.op_stats()
    fd = sub._frozen_dist or (0.0, 0.0, 100.0)

    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go,
                avoid_pct=av, prefer_pct=pr, neutral_pct=nt,
                frozen_avoid=fd[0], frozen_prefer=fd[1])


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
              f"frozen=A{r['frozen_avoid']:.0f}%P{r['frozen_prefer']:.0f}%  "
              f"final_ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])

    print(f"\nL1={l1n}/5  L2={l2n}/5")
    print(f"Note: frozen_ops = op dist at step {DISABLE_AT_STEP}; final_ops = cumulative usage")
    print(f"If final_ops AVOID% ~= frozen_avoid% → ops effectively frozen (stale but applied)")
    print(f"Compare L1 rate against step 620 (5/5) to assess self-obs contribution")

    if l1n >= 4:
        print("L1 MAINTAINED: disabling self-obs did not degrade performance → self-obs may be inert OR ops initialized correctly.")
    elif l1n < 3:
        print("L1 DEGRADED: disabling self-obs hurt performance → ongoing self-obs is load-bearing.")
    else:
        print(f"MARGINAL: L1={l1n}/5 — inconclusive")


if __name__ == "__main__":
    main()
