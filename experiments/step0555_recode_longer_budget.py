"""
Step 555 — Does living longer help? Aggressive Recode at 2M steps.

Same params as Step 554 (REFINE_EVERY=2000, MIN_OBS=4, no split cap).
Single seed=0. 5-min cap (~1.5-2M steps).

Track at 500K checkpoints: cells, splits, active set, growth rate.

Predictions:
  L2: 0/1 (structural, time won't fix it)
  Cells: >3000, active plateaus ~5000
  Growth rate decelerates as tree deepens

Kill: L2=0 AND active flat from 500K to cap: time doesn't help.
"""
import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000
MIN_OBS = 4
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
        cutoff = self.t - window
        return sum(1 for v in self._last_visit.values() if v >= cutoff)

    def stats(self):
        return len(self.live), self.ns, len(self.G)


def t0():
    rng = np.random.RandomState(0)
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,) and abs(float(x.mean())) < 1e-5
    sub = Recode(seed=0)
    sub.observe(frame); sub.act()
    assert sub._last_visit
    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        env = arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    sub = Recode(seed=0)
    obs = env.reset(seed=0)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()
    checkpoints = {500_000, 1_000_000, 1_500_000, 2_000_000}
    prev_cells = 0

    for step in range(1, 2_000_001):
        if obs is None:
            obs = env.reset(seed=0)
            sub.on_reset()
            continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=0)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            nc, ns, ne = sub.stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"L1@{step} c={nc} sp={ns} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"L2@{step} c={nc} sp={ns} go={go}", flush=True)
            level = cl

        if step % 100_000 == 0:
            nc, ns, ne = sub.stats()
            ac = sub.active_set()
            el = time.time() - t_start
            growth = nc - prev_cells
            prev_cells = nc
            print(f"@{step} c={nc}(+{growth}) sp={ns} active={ac} go={go} {el:.0f}s",
                  flush=True)

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    ac = sub.active_set()
    tag = "L2" if l2 else ("L1" if l1 else "---")
    elapsed = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"Result: {tag}  steps={step}  c={nc}  sp={ns}  active={ac}  go={go}  {elapsed:.0f}s")

    if l2:
        print("FIND: L2 reached. Extended budget unlocks L2!")
    elif ac <= 5500:
        print(f"KILL: L2=0, active={ac} plateaued. Time doesn't help. Structural gap confirmed.")
    else:
        print(f"GROWING: active={ac} still growing. Extended budget may eventually reach L2.")


if __name__ == "__main__":
    main()
