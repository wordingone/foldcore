"""
Step 554 — Aggressive hub splitting to escape the attractor.

Step 553: 98.8% of high-entropy edges are reducible. Hub nodes (obs=2700-5260)
confuse 27 distinct successor regions but Recode never splits them fast enough.

Modification from Step 542 baseline:
  REFINE_EVERY: 2000 (was 5000)
  MIN_OBS: 4 (was 8)
  No `did >= 3` cap per round (split all qualifying nodes)

Predictions:
  L1: 3/3, L2: 0/3 (hub splitting may still be insufficient)
  Cells: >2000, Active set: >364 (attractor disrupted)

Kill: L2=0/3 AND active_set <= 364: splitting speed not the issue.
Find: active_set > 500: attractor disrupted even without L2.

5-min cap. LS20. 3 seeds.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000   # was 5000
MIN_OBS = 4            # was 8
H_SPLIT = 0.05


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
        self.dim = dim
        self._last_visit = {}

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
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
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
        # NO did >= 3 cap — split all qualifying nodes
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
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)
            self.ns += 1

    def active_set(self, window=100_000):
        """Nodes visited in last `window` steps."""
        cutoff = self.t - window
        return sum(1 for v in self._last_visit.values() if v >= cutoff)

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def t0():
    rng = np.random.RandomState(42)
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,)
    assert abs(float(x.mean())) < 1e-5

    sub = Recode(seed=0)
    f1 = [rng.randint(0, 16, (64, 64))]
    f2 = [rng.randint(0, 16, (64, 64))]
    sub.observe(f1); sub.act()
    sub.observe(f2); sub.act()
    assert sub._last_visit  # populated
    assert sub.active_set(100_000) == len(sub._last_visit)  # all recent

    # _refine: no cap — should split multiple nodes
    sub2 = Recode(seed=0)
    # Create 3 qualifying (node, action) pairs
    for ni in [100, 200, 300]:
        sub2.live.add(ni)
        sub2.G[(ni, 0)] = {ni + 1: 5, ni + 2: 5}
        sub2.C[(ni, 0, ni + 1)] = (rng.randn(DIM).astype(np.float64) * 5, 5)
        sub2.C[(ni, 0, ni + 2)] = (-rng.randn(DIM).astype(np.float64) * 5, 5)
    sub2._refine()
    assert sub2.ns >= 2, f"Expected multiple splits, got {sub2.ns}"

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
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{step} c={nc} sp={ns} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} go={go}", flush=True)
            level = cl

        if step % 100_000 == 0:
            nc, ns, ne = sub.stats()
            ac = sub.active_set()
            el = time.time() - t_start
            print(f"  s{seed} @{step} c={nc} sp={ns} active={ac} go={go} {el:.0f}s",
                  flush=True)

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    ac = sub.active_set()
    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, go=go,
                steps=step, active=ac)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    R = []
    for seed in range(3):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  sp={r['splits']:>4}  "
              f"active={r['active']:>4}  go={r['go']}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)
    ma = max(r['active'] for r in R)

    print(f"\nL1={l1n}/3  L2={l2n}/3  max_cells={mc}  max_splits={ms}  max_active={ma}")
    print(f"Baseline (Step 542): L1=5/5, max_cells=1267, sp=300, active=364")

    if l2n > 0:
        print(f"FIND: L2={l2n}/3. Aggressive splitting unlocks L2!")
    elif ma > 500:
        print(f"ATTRACTOR DISRUPTED: active={ma} > 500. Agent exploring new regions.")
        if l1n == 3:
            print("L1=3/3. Splitting faster disrupts attractor but L2 gap remains.")
    else:
        print(f"KILL: L2=0/3 AND active_set={ma} <= 500. Split speed not the issue.")


if __name__ == "__main__":
    main()
