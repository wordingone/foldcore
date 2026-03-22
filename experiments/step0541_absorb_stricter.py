"""
Step 541 -- Absorb with stricter entropy filter (recent-window + higher cutoff).

Step 540: reducible filter (ratio>=0.5, all-time) -> L1=1/3, deaths ~800/100K.
Filter too permissive: early successes permanently enable death-seeking.

Two changes from Step 540:
1. RECENT WINDOW: only last 10 refinements (not all-time).
   Prevents early successes from masking later failures.
2. CUTOFF 0.75: seek only if 75%+ of recent refinements reduced entropy.
   More aggressive gating of entropy-seeking.

Predictions: 2/3 L1. Deaths < 500/100K. Fewer total refinements.
Kill: 0/3 L1 -> filter too aggressive, blocks useful entropy-seeking.
"""
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
SETTLE_STEPS = 1000
RECENT_WINDOW = 10   # only last N refinements
SEEK_CUTOFF = 0.75   # raised from 0.5


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
        self.refined = {}  # node -> (h_before, step_refined), insertion-ordered
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

    def _reduction_ratio(self):
        """Fraction of recently-settled refinements that reduced max child entropy.
        Only considers the last RECENT_WINDOW refinements."""
        recent = list(self.refined.items())[-RECENT_WINDOW:]
        total = 0
        reduced = 0
        for node, (h_before, step_ref) in recent:
            if self.t - step_ref < SETTLE_STEPS:
                continue  # too recent to evaluate
            c0, c1 = (node, 0), (node, 1)
            child_h = [self._h(c0, a) for a in range(N_A)] + \
                      [self._h(c1, a) for a in range(N_A)]
            finite_h = [e for e in child_h if e < float('inf')]
            if not finite_h:
                continue  # children not yet observed -- skip
            h_after = max(finite_h)
            total += 1
            if h_after < h_before:
                reduced += 1
        if total == 0:
            return 1.0  # no settled data -- assume productive
        return reduced / total

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
        ratio = self._reduction_ratio()
        if ratio < SEEK_CUTOFF:
            return None  # recent refinements mostly unhelpful -- argmin fallback

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
            h_before = self._h(n, a)
            self.ref[n] = (diff / nm).astype(np.float32)
            self.refined[n] = (h_before, self.t)
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

    # Test 1: empty -> ratio 1.0
    assert sub._reduction_ratio() == 1.0, "empty refined -> 1.0"

    # Test 2: recent window only uses last RECENT_WINDOW refinements
    # Fill 15 refinements: first 5 good, last 10 bad
    sub2 = Absorb(dim=8, k=3, seed=0)
    sub2.H = sub.H.copy()
    sub2.t = 5000
    # First 5: good refinements (settled, children low entropy)
    # Last 10: bad refinements (settled, children high entropy)
    for i in range(5):
        sub2.refined[i] = (1.0, 0)      # good, settled
        sub2.G[((i, 0), 0)] = {10: 90, 11: 5}   # child (i,0): low entropy
        sub2.G[((i, 1), 0)] = {12: 90, 13: 5}   # child (i,1): low entropy
        sub2.live.update([(i, 0), (i, 1)])
    for i in range(5, 15):
        sub2.refined[i] = (0.8, 0)      # bad, settled
        sub2.G[((i, 0), 0)] = {10: 50, 11: 50}  # child (i,0): high entropy
        sub2.G[((i, 1), 0)] = {12: 50, 13: 50}  # child (i,1): high entropy
        sub2.live.update([(i, 0), (i, 1)])
    # Only last 10 count: all bad (h_after=1.0 >= h_before=0.8 -> not reduced)
    ratio2 = sub2._reduction_ratio()
    assert ratio2 == 0.0, f"recent window should see only bad refinements, got {ratio2}"
    assert sub2._seek() is None, "_seek should return None when ratio<0.75"

    # Test 3: cutoff is 0.75 (not 0.5)
    # 8 good, 2 bad in last 10 -> ratio=0.8 >= 0.75 -> seek
    sub3 = Absorb(dim=8, k=3, seed=0)
    sub3.H = sub.H.copy()
    sub3.t = 5000
    for i in range(8):
        sub3.refined[i] = (1.0, 0)      # good
        sub3.G[((i, 0), 0)] = {10: 90, 11: 5}   # low entropy
        sub3.G[((i, 1), 0)] = {12: 90, 13: 5}   # low entropy
        sub3.live.update([(i, 0), (i, 1)])
    for i in range(8, 10):
        sub3.refined[i] = (0.8, 0)      # bad
        sub3.G[((i, 0), 0)] = {10: 50, 11: 50}  # high entropy
        sub3.G[((i, 1), 0)] = {12: 50, 13: 50}  # high entropy
        sub3.live.update([(i, 0), (i, 1)])
    # Add a more confused live node (entropy ~1.58 > 1.0 of children)
    sub3.G[(99, 0)] = {5: 33, 6: 33, 7: 34}
    sub3.live.add(99)
    ratio3 = sub3._reduction_ratio()
    assert ratio3 == 0.8, f"expected 0.8, got {ratio3}"
    tgt3 = sub3._seek()
    assert tgt3 == 99, f"expected node 99, got {tgt3}"

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
                ratio = sub._reduction_ratio()
                print(f"  s{seed} L1@{step} c={nc} h={nh} mh={mh:.4f} "
                      f"sp={ns} go={go} ratio={ratio:.2f}")
            if cl == 2 and l2 is None:
                l2 = step
                ratio = sub._reduction_ratio()
                print(f"  s{seed} L2@{step} c={nc} h={nh} mh={mh:.4f} "
                      f"sp={ns} go={go} ratio={ratio:.2f}")
            level = cl

        if step % 100_000 == 0:
            nc, nh, mh, ns = sub.stats()
            ratio = sub._reduction_ratio()
            el = time.time() - t_start
            print(f"  s{seed} @{step} c={nc} h={nh} mh={mh:.4f} "
                  f"sp={ns} go={go} ratio={ratio:.2f} {el:.0f}s")

        if time.time() - t_start > 300:
            break

    nc, nh, mh, ns = sub.stats()
    return dict(seed=seed, l1=l1, l2=l2, cells=nc, confused=nh,
                max_h=mh, splits=ns, go=go,
                ratio=sub._reduction_ratio())


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
              f"mh={r['max_h']:.4f}  go={r['go']}  ratio={r['ratio']:.2f}")

    l2n = sum(1 for r in R if r['l2'])
    l1n = sum(1 for r in R if r['l1'])
    mc = max(r['cells'] for r in R)
    ms = max(r['splits'] for r in R)
    mh = max(r['max_h'] for r in R)
    mg = max(r['go'] for r in R)
    avg_ratio = sum(r['ratio'] for r in R) / len(R)

    print(f"\nL1={l1n}/3  L2={l2n}/3  cells={mc}  splits={ms}  "
          f"max_h={mh:.4f}  deaths={mg}  ratio={avg_ratio:.2f}")

    if l1n > 0:
        print(f"SIGNAL: stricter filter enables navigation. "
              f"L1={l1n}/3 ratio={avg_ratio:.2f}")
    elif avg_ratio < SEEK_CUTOFF:
        print(f"CONFIRMED KILL: entropy irreducible (ratio={avg_ratio:.2f}). "
              f"Stricter cutoff blocks all useful entropy-seeking.")
    else:
        print(f"NO L1: ratio={avg_ratio:.2f} but still no navigation.")


if __name__ == "__main__":
    main()
