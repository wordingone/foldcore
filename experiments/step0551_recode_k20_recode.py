"""
Step 551 — Recode k=20 at 1M steps. Does finer partition reach L2?

Hypothesis: k=20 gives ~1600 cells at 200K. At 1M, may push toward 2000+ cells.
If L2 is a resolution problem, k=20 + refinement + extended budget might find it.

Predictions:
  L1: 3/3 (k=20 navigates, confirmed Step 531)
  L2: 0/3 (L2 is topology gap, not resolution gap)
  cells: ~1800-2200 at cap

Kill: 0/3 L2 -> resolution is NOT the bottleneck.
Find: any L2 -> k=20 is new baseline, L2 is resolution problem.

5-min cap. 3 seeds. LS20.
"""
import numpy as np
import time
import sys

N_A = 4
K = 20   # increased from 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05


def enc(frame):
    """Avgpool16 + centered."""
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

    def _base(self, x):
        # k=20: pack 20 bits into 3 bytes (top 4 bits of 3rd byte unused)
        bits = (self.H @ x > 0).astype(np.uint8)
        return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)

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


def t0():
    rng = np.random.RandomState(42)

    # enc() shape and centering
    frame = [rng.randint(0, 16, (64, 64))]
    x = enc(frame)
    assert x.shape == (256,), f"enc shape {x.shape}"
    assert abs(float(x.mean())) < 1e-5, "not centered"

    # k=20 hash produces distinguishable nodes
    sub = Recode(seed=0)
    assert sub.H.shape == (K, DIM), f"H shape {sub.H.shape}"
    f1 = [rng.randint(0, 16, (64, 64))]
    f2 = [rng.randint(0, 16, (64, 64))]
    n1 = sub.observe(f1); sub.act()
    n2 = sub.observe(f2); sub.act()
    # Nodes should be non-negative integers
    if isinstance(n1, int):
        assert n1 >= 0, f"node should be non-negative: {n1}"

    # _node follows ref chain
    x_test = enc(f1)
    n_direct = sub._node(x_test)
    assert n_direct == n1 or n_direct in sub.live, "node should be in live"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
    go = 0
    t_start = time.time()

    for step in range(1, 1_000_001):
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
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go}", flush=True)
            level = cl

        if step % 200_000 == 0:
            nc, ns, ne = sub.stats()
            el = time.time() - t_start
            print(f"  s{seed} @{step} c={nc} sp={ns} go={go} {el:.0f}s", flush=True)

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, go=go, steps=step)


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
        print(f"  s{r['seed']}: {tag:>3}  steps={r['steps']:>7}  c={r['cells']:>5}  "
              f"sp={r['splits']:>3}  go={r['go']}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)
    avg_steps = int(np.mean([r['steps'] for r in R]))

    print(f"\nL1={l1n}/3  L2={l2n}/3  max_cells={mc}  max_splits={ms}  avg_steps={avg_steps}")

    if l2n > 0:
        print(f"FIND: L2 reached {l2n}/3. k=20 breaks L2 barrier. Resolution problem confirmed.")
    elif l1n == 3:
        print(f"KILL: 0/3 L2. L1={l1n}/3. Resolution is NOT the L2 bottleneck.")
    else:
        print(f"PARTIAL: L1={l1n}/3 L2={l2n}/3.")


if __name__ == "__main__":
    main()
