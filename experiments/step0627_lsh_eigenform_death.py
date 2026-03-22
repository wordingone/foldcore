"""
Step 627 — Eigenform + death signal: combine 620 self-observation with 582 death penalty.

Base: 620 (self-obs every 500 steps) + death signal from step 582.
Death signal: when the agent dies, mark the current node-action pair with DEATH penalty.

From 582: death_count[n][a] -> if action leads to death, DEATH_PENALTY added to argmin.
From 620: edge_ratio self-obs -> AVOID/PREFER/NEUTRAL op codes.

Integration: both signals modify argmin independently.
- Death: immediate, per (n,a) pair
- Self-obs: periodic, distributional

Question: do they complement or interfere?

5 seeds x 60s. Signal: L1 >= 4/5 AND better than 620 (faster L1 or more L2).
Kill: L1 < 3/5.
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
DEATH_PENALTY = 200


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
        # Death signal: {(n, a): count}
        self.death_counts = {}
        self.total_deaths = 0

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

    def on_death(self):
        """Called when agent dies. Penalizes the last (n, a) pair."""
        if self._pn is not None and self._pa is not None:
            key = (self._pn, self._pa)
            self.death_counts[key] = self.death_counts.get(key, 0) + 1
            self.total_deaths += 1

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
            # Add death penalty on top of op-modified count
            dc = self.death_counts.get((self._cn, a), 0)
            modified += dc * DEATH_PENALTY
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
    assert isinstance(sub._node(x1), int)

    # Test death signal: after on_death, acting should penalize last (n,a)
    sub2 = Recode(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 8
    sub2._cn = 99
    sub2._pn = 99
    sub2._pa = 0  # last action was (99, 0)
    sub2.on_death()
    assert sub2.death_counts.get((99, 0), 0) == 1, "death should be recorded for (99, 0)"
    assert sub2.total_deaths == 1

    # Test that death penalty adds to act() counts
    sub2.G[(99, 0)] = {100: 5}
    sub2.G[(99, 1)] = {100: 5}
    sub2.G[(99, 2)] = {100: 5}
    sub2.G[(99, 3)] = {100: 5}
    action = sub2.act()
    assert action != 0, f"should avoid action 0 (has death penalty), chose {action}"

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
            # Check if death (reward < 0 or specific info)
            if isinstance(info, dict) and info.get('death', False):
                sub.on_death()
            elif reward < 0:
                sub.on_death()
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
                      f"deaths={sub.total_deaths} ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%",
                      flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go} "
                      f"deaths={sub.total_deaths}", flush=True)
            level = cl

        if time.time() - t_start > PER_SEED_TIME:
            break

    nc, ns, ne = sub.stats()
    av, pr, nt = sub.op_stats()

    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go,
                avoid_pct=av, prefer_pct=pr, neutral_pct=nt,
                total_deaths=sub.total_deaths,
                death_pairs=len(sub.death_counts))


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
              f"deaths={r['total_deaths']}  dp={r['death_pairs']}  "
              f"ops=A{av:.0f}%P{pr:.0f}%N{nt:.0f}%")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    avg_deaths = np.mean([r['total_deaths'] for r in R])

    print(f"\nL1={l1n}/5  L2={l2n}/5  avg_deaths={avg_deaths:.0f}")

    if l1n >= 4:
        print("SIGNAL: L1 >= 4/5 -- eigenform+death maintains L1.")
    elif l1n < 3:
        print("KILL: L1 < 3/5 -- death signal interferes with eigenform.")
    else:
        print(f"MARGINAL: L1={l1n}/5")


if __name__ == "__main__":
    main()
