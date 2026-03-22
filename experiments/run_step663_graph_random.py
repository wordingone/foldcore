"""
Step 663 — Graph without argmin (random walk with graph memory).

Full LSH k=12 graph — accumulates edges as normal.
BUT: action selection is UNIFORM RANDOM.

Isolates: does graph MEMORY help even without graph SELECTION?

Compare to 653's random column (random without graph memory).
If identical: graph memory adds nothing.
If different: graph memory matters (hash refinement, cell growth affects walk).
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
N_A = 4
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 500_001
PER_SEED_TIME = 5
N_SEEDS = 20


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class GraphRandom:
    """Full graph accumulation, random action selection."""

    def __init__(self, k=K, dim=DIM, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.rng = np.random.RandomState(seed * 777 + 42)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim

    def _hash(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._hash(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

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
        action = int(self.rng.randint(N_A))
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


RANDOM_653 = {7: 8959, 10: 9130, 11: 1034, 15: 3453}


def run(seed, make):
    env = make()
    sub = GraphRandom(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = None
    go = 0
    t_start = time.time()

    for step in range(1, MAX_STEPS):
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
            level = cl
            sub.on_reset()
        if done:
            go += 1
            obs = env.reset(seed=seed)
            sub.on_reset()
        if time.time() - t_start > PER_SEED_TIME:
            break

    r653 = RANDOM_653.get(seed)
    match = "same" if l1 == r653 else f"DIFF(was {r653})"
    print(f"  s{seed:2d}: L1={l1} unique={len(sub.live)} {match}", flush=True)
    return dict(seed=seed, l1=l1, unique=len(sub.live))


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except (ImportError, Exception) as e:
        print(f"arcagi3: {e}")
        return

    print(f"Graph+random selection: {N_SEEDS} seeds, {PER_SEED_TIME}s cap")

    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    gr_success = {r['seed'] for r in results if r['l1']}
    rand_success = set(RANDOM_653.keys())
    gr_only = gr_success - rand_success
    rand_only = rand_success - gr_success
    both = gr_success & rand_success

    print(f"graph+random success: {sorted(gr_success)}")
    print(f"pure random success:  {sorted(rand_success)}")
    print(f"gr_only: {sorted(gr_only)}, rand_only: {sorted(rand_only)}, both: {sorted(both)}")

    if gr_only:
        print(f"\nFINDING: Graph memory adds value to random walk ({len(gr_only)} new seeds)")
    elif rand_only:
        print(f"\nFINDING: Graph memory HURTS random walk ({len(rand_only)} lost seeds)")
    else:
        print(f"\nFINDING: Graph memory adds NOTHING — identical to pure random walk")


if __name__ == "__main__":
    main()
