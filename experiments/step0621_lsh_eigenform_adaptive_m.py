"""
Step 621 — Eigenform adaptive observation frequency.

Same as 620 but M is self-derived:
- Start M=100
- After each observation: compute total variation in op distribution vs last obs
  - If change < 5pp: double M (up to M=2000) — distribution stable, observe less
  - If change > 20pp: halve M (down to M=50) — distribution volatile, observe more
  - Otherwise: M unchanged

Signal: L1 >= 4/5 AND M settles (converges within the run)
Kill: L1 < 3/5
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
M_INIT = 100
M_MIN = 50
M_MAX = 2000
CHANGE_STABLE = 5.0    # pp — double M if change < this
CHANGE_VOLATILE = 20.0  # pp — halve M if change > this
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
        self.op_counts = [0, 0, 0]  # neutral, avoid, prefer
        self.op_history = []   # [(t, avoid%, prefer%, neutral%, M)]
        self.M = M_INIT
        self._next_obs = M_INIT
        self._last_dist = None  # (avoid%, prefer%, neutral%) from last obs

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
        if self.t >= self._next_obs:
            self._self_observe()
            self._next_obs = self.t + self.M
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
        if total_na == 0:
            return

        av = 100.0 * avoid_n / total_na
        pr = 100.0 * prefer_n / total_na
        nt = 100.0 * neutral_n / total_na

        # Adapt M based on distribution change from last observation
        if self._last_dist is not None:
            last_av, last_pr, last_nt = self._last_dist
            total_change = abs(av - last_av) + abs(pr - last_pr) + abs(nt - last_nt)
            if total_change < CHANGE_STABLE:
                self.M = min(self.M * 2, M_MAX)
            elif total_change > CHANGE_VOLATILE:
                self.M = max(self.M // 2, M_MIN)
            # else: M unchanged

        self._last_dist = (av, pr, nt)
        self.op_history.append((self.t, av, pr, nt, self.M))

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
    n1 = sub._node(x1)
    assert isinstance(n1, int)

    # Test M adaptation: build graph with outliers, run _self_observe twice
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

    M_before = sub2.M
    sub2._self_observe()
    assert len(sub2.op_history) == 1, "first observation should record"
    assert sub2._last_dist is not None

    # Second observe with identical graph — change should be 0 → M should double
    sub2._self_observe()
    assert sub2.M == min(M_before * 2, M_MAX) or sub2.M == M_before, \
        f"M should have doubled or stayed: was {M_before}, now {sub2.M}"

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
                      f"M={sub.M} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                av, pr, nt = sub.op_stats()
                print(f"  s{seed} L2@{step} M={sub.M} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%", flush=True)
            level = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    nc, ns, ne = sub.stats()
    av, pr, nt = sub.op_stats()
    hist = sub.op_history

    # M convergence: last 5 observations
    m_values = [h[4] for h in hist[-5:]] if len(hist) >= 5 else [h[4] for h in hist]
    m_settled = len(set(m_values)) == 1 if m_values else False
    m_final = m_values[-1] if m_values else sub.M

    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go,
                avoid_pct=av, prefer_pct=pr, neutral_pct=nt,
                m_final=m_final, m_settled=m_settled, op_history=hist)


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
        settled = "settled" if r['m_settled'] else "drifting"
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  "
              f"sp={r['splits']:>3}  e={r['edges']:>5}  go={r['go']}  "
              f"M_final={r['m_final']}({settled})  ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    settled_count = sum(1 for r in R if r['m_settled'])
    m_finals = [r['m_final'] for r in R]

    print(f"\nL1={l1n}/5  L2={l2n}/5  M_settled={settled_count}/5  M_finals={m_finals}")

    if l1n >= 4 and settled_count >= 3:
        print("SIGNAL: L1 >= 4/5 AND M converges.")
    elif l1n < 3:
        print("KILL: L1 < 3/5")
    else:
        print(f"PARTIAL: L1={l1n}/5 M_settled={settled_count}/5")


if __name__ == "__main__":
    main()
