import numpy as np
import time
import sys

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
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
        return n

    def act(self):
        counts = []
        for a in range(N_A):
            counts.append(sum(self.G.get((self._cn, a), {}).values()))
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

    sub2 = Recode(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.dim = 8
    sub2.G = {
        (0, 0): {1: 50, 2: 50},
        (0, 1): {1: 100},
    }
    sub2.live = {0, 1, 2}
    sub2.C = {
        (0, 0, 1): (rng.randn(8).astype(np.float64) * 50, 50),
        (0, 0, 2): (-rng.randn(8).astype(np.float64) * 50, 50),
    }

    h = sub2._h(0, 0)
    assert h > 0.9, f"bimodal entropy: {h}"
    assert sub2._h(0, 1) == 0.0

    sub2._cn = 0
    a = sub2.act()
    assert a in range(N_A)

    sub2._refine()
    assert 0 in sub2.ref, "node 0 should be refined"
    assert 0 not in sub2.live, "node 0 should not be live"

    # Verify hyperplane separates the two mean observations
    # (synthetic nodes don't hash to 0 via _base, so test hyperplane directly)
    x0 = (sub2.C[(0, 0, 1)][0] / 50).astype(np.float32)
    x1 = (sub2.C[(0, 0, 2)][0] / 50).astype(np.float32)
    assert int(sub2.ref[0] @ x0 > 0) != int(sub2.ref[0] @ x1 > 0), \
        "hyperplane should separate children"

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
                print(f"  s{seed} L1@{step} c={nc} sp={ns} e={ne} go={go}")
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} sp={ns} e={ne} go={go}")
            level = cl

        if step % 100_000 == 0:
            nc, ns, ne = sub.stats()
            el = time.time() - t_start
            print(f"  s{seed} @{step} c={nc} sp={ns} e={ne} go={go} {el:.0f}s")

        if time.time() - t_start > 300:
            break

    nc, ns, ne = sub.stats()
    return dict(seed=seed, l1=l1, l2=l2, cells=nc, splits=ns, edges=ne, go=go)


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
        print(f"\nseed {seed}:")
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  "
              f"sp={r['splits']:>3}  e={r['edges']:>5}  go={r['go']}")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)

    print(f"\nL1={l1n}/5  L2={l2n}/5  max_cells={mc}  max_splits={ms}")

    baseline_cells = 440
    if ms == 0:
        print(f"ZERO SPLITS: no confused transitions at k={K}. "
              f"self-observation inert. argmin-only baseline.")
    elif mc > baseline_cells:
        print(f"EXPANDED: {mc} > {baseline_cells}. "
              f"refinement created {ms} new distinctions beyond LSH plateau.")
        if l2n > 0:
            print(f"L2 REACHED: self-observation enabled Level 2 on {l2n}/5 seeds.")
        else:
            print(f"NO L2: expanded reachable set but Level 2 remains disconnected.")
    else:
        print(f"NO EXPANSION: {mc} <= {baseline_cells}. "
              f"refinement splits ({ms}) did not expand reachable set.")


if __name__ == "__main__":
    main()
