"""
Step 641 — Meta-graph soft bias.

Same profile similarity as 640, but continuous signal:
  effective_count(N, a) = count(N, a) + alpha * avg_count(neighbors, a)

alpha=0.1 (weak transfer). k=5 neighbors by cosine similarity.

This provides a continuous signal, not just at ties.

5 seeds × 60s, LS20, k=16 LSH. Compare L1 to 620 baseline.
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
K_NEIGHBORS = 5
ALPHA = 0.1
PER_SEED_TIME = 60

BASELINE_L1 = [1362, 3270, 48391, 62727, 846]


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class Recode:

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.transfer_active = 0
        self.total_act = 0

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _profile(self, n):
        return np.array([sum(self.G.get((n, a), {}).values()) for a in range(N_A)],
                        dtype=np.float32)

    def _neighbor_avg(self, n):
        """Average profile of k nearest neighbors.
        Returns None if current cell has no data or too few neighbors.
        """
        target = self._profile(n)
        tn = np.linalg.norm(target)
        if tn < 1e-8:
            return None

        candidates = []
        for cell in self.live:
            if cell == n:
                continue
            v = self._profile(cell)
            vn = np.linalg.norm(v)
            if vn < 1e-8:
                continue
            sim = float(np.dot(target, v) / (tn * vn))
            candidates.append((sim, v))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: x[0])
        top_k = [v for _, v in candidates[:K_NEIGHBORS]]
        return np.mean(top_k, axis=0)

    def observe(self, frame):
        x = enc_frame(frame)
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

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        base = np.array([sum(self.G.get((self._cn, a), {}).values())
                         for a in range(N_A)], dtype=np.float32)
        nb_avg = self._neighbor_avg(self._cn)
        self.total_act += 1

        if nb_avg is not None:
            effective = base + ALPHA * nb_avg
            self.transfer_active += 1
        else:
            effective = base

        action = int(np.argmin(effective))
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


def t0():
    sub = Recode(seed=0)

    # No neighbor avg when no data
    sub._cn = 1
    sub.live.add(1)
    result = sub._neighbor_avg(1)
    assert result is None, "no neighbor avg with empty graph"

    # Profile returns zeros for unknown cell
    p = sub._profile(99)
    assert np.all(p == 0)

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

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
            if cl == 2 and l2 is None:
                l2 = step
            level = cl
            sub.on_reset()

        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()

        if time.time() - t_start > PER_SEED_TIME:
            break

    transfer_rate = 100.0 * sub.transfer_active / sub.total_act if sub.total_act else 0.0
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} "
          f"transfer_active={transfer_rate:.1f}% unique={len(sub.live)}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                transfer_rate=transfer_rate, unique=len(sub.live))


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        print("\nDry run only. T0 passed.")
        return

    R = []
    for seed in range(5):
        print(f"\nseed {seed}:", flush=True)
        R.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1n = sum(1 for r in R if r['l1'])
    l1_valid = [(r['l1'], BASELINE_L1[r['seed']]) for r in R if r['l1']]
    avg_ratio = np.mean([b / l for l, b in l1_valid]) if l1_valid else 0.0
    avg_tr = np.mean([r['transfer_rate'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"transfer_active={r['transfer_rate']:.1f}%")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  avg_transfer={avg_tr:.1f}%")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: meta-graph soft bias faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print(f"MARGINAL: L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
