import numpy as np
from collections import deque
import time
import sys

N_A = 4
K = 12
DIM = 256
REFINE_EVERY = 5000
H_SPLIT = 0.05
MIN_OBS = 8
MAX_DEPTH = 8
ROUTE_HORIZON = 50


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Absorb:

    def __init__(self, dim=DIM, k=K, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(k, dim).astype(np.float32)
        self.dim = dim
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0

    def _raw(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._raw(x)
        for _ in range(MAX_DEPTH):
            if n not in self.ref:
                break
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        x = enc(frame)
        node = self._node(x)
        self.live.add(node)
        self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[node] = d.get(node, 0) + 1
            k = (self._pn, self._pa, node)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px, c + 1)
        self._x = x.astype(np.float64)
        self._cn = node
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return node

    def act(self, a):
        self._pn = self._cn
        self._pa = a
        self._px = self._x

    def on_reset(self):
        self._pn = None

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d:
            return float('inf')
        total = sum(d.values())
        if total < 3:
            return float('inf')
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def select(self, node):
        h = [self._h(node, a) for a in range(N_A)]
        mx = max(h)

        if mx == float('inf'):
            u = [a for a in range(N_A)
                 if sum(self.G.get((node, a), {}).values()) < 3]
            if u:
                return u[self.t % len(u)]

        if mx > 0.01:
            return int(np.argmax(h))

        tgt = self._seek()
        if tgt is not None and tgt != node:
            path = self._bfs(node, tgt)
            if path:
                return path[0]

        c = [sum(self.G.get((node, a), {}).values()) for a in range(N_A)]
        return int(np.argmin(c))

    def _seek(self):
        best_h, best_n = 0.0, None
        for (n, a) in self.G:
            if n not in self.live:
                continue
            e = self._h(n, a)
            if 0 < e < float('inf') and e > best_h:
                best_h, best_n = e, n
        return best_n

    def _bfs(self, s, g):
        L = {}
        for (n, a), d in self.G.items():
            if n not in self.live:
                continue
            L.setdefault(n, {})[a] = max(d, key=d.get)
        V = {s}
        q = deque([(s, [])])
        while q:
            cur, path = q.popleft()
            if len(path) >= ROUTE_HORIZON:
                continue
            for a in range(N_A):
                nxt = L.get(cur, {}).get(a)
                if nxt is None or nxt in V:
                    continue
                if nxt == g:
                    return path + [a]
                V.add(nxt)
                q.append((nxt, path + [a]))
        return None

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
        nh = 0
        mh = 0.0
        for (n, a) in self.G:
            if n not in self.live:
                continue
            e = self._h(n, a)
            if 0 < e < float('inf'):
                nh += 1
                if e > mh:
                    mh = e
        return len(self.live), nh, mh, self.ns


def t0():
    rng = np.random.RandomState(42)
    sub = Absorb(dim=8, k=3, seed=0)
    sub.H = rng.randn(3, 8).astype(np.float32)

    x1 = rng.randn(8).astype(np.float32)
    x2 = x1 + 0.001 * rng.randn(8).astype(np.float32)
    x3 = -x1

    n1 = sub._node(x1)
    n2 = sub._node(x2)
    n3 = sub._node(x3)
    assert n1 == n2, f"local continuity: {n1} != {n2}"
    assert n1 != n3, f"discrimination: {n1} == {n3}"

    sub2 = Absorb(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.G = {
        (0, 0): {1: 50, 2: 50},
        (0, 1): {1: 100},
        (0, 2): {3: 100},
        (0, 3): {4: 100},
    }
    sub2.live = {0, 1, 2, 3, 4}

    h00 = sub2._h(0, 0)
    h01 = sub2._h(0, 1)
    assert h00 > 0.9, f"bimodal entropy should be ~1.0, got {h00}"
    assert h01 == 0.0, f"unimodal entropy should be 0.0, got {h01}"

    assert sub2.select(0) == 0, "should select max-entropy action"

    sub2.live = {1, 2, 3, 4}
    sub2.G[(1, 0)] = {5: 40, 6: 60}
    sub2.live.add(1)
    assert sub2._seek() == 1, "should seek node 1 (has confused edge)"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Absorb(seed=seed * 1000)
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

        node = sub.observe(obs)
        action = sub.select(node)
        sub.act(action)
        obs, reward, done, info = env.step(action)

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            sub.on_reset()
            nc, nh, mh, ns = sub.stats()
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{step} c={nc} h={nh} mh={mh:.4f} sp={ns} go={go}")
            if cl == 2 and l2 is None:
                l2 = step
                print(f"  s{seed} L2@{step} c={nc} h={nh} mh={mh:.4f} sp={ns} go={go}")
            level = cl

        if step % 100_000 == 0:
            nc, nh, mh, ns = sub.stats()
            el = time.time() - t_start
            print(f"  s{seed} @{step} c={nc} h={nh} mh={mh:.4f} sp={ns} go={go} {el:.0f}s")

        if time.time() - t_start > 300:
            break

    nc, nh, mh, ns = sub.stats()
    return dict(seed=seed, l1=l1, l2=l2, cells=nc, confused=nh,
                max_h=mh, splits=ns, go=go)


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    R = []
    for seed in range(3):
        print(f"\nseed {seed}:")
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>8}  c={r['cells']:>4}  "
              f"sp={r['splits']:>2}  h={r['confused']:>3}  "
              f"mh={r['max_h']:.4f}  go={r['go']}")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)
    mh = max(r['max_h'] for r in R)

    print(f"\nL1={l1n}/3  L2={l2n}/3  cells={mc}  splits={ms}  max_h={mh:.4f}")

    if ms == 0:
        print("ZERO SPLITS: mapping clean. absorb = argmin. no R3 activation.")
    elif l2n > 0:
        print(f"L2 REACHED: {l2n}/3. self-observation enabled Level 2.")
    elif mc > 439:
        print(f"NEW CELLS: {mc} > 439 baseline. refinement expanded reachable set.")
    else:
        print(f"NO L2: splits={ms} but cells={mc}. confused edges don't unlock Level 2.")


if __name__ == "__main__":
    main()
