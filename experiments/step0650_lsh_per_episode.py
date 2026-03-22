"""
Step 650 — Per-episode exploration.

Each episode (life) gets its own fresh edge dict for action selection.
Global graph (G) still accumulates all edges for refinement (U3 — no forgetting).

Within each life: argmin on episode-local counts → explore as if starting fresh.
Between lives: local G_ep is cleared. Global G_global tracks everything.

Hypothesis: fresh 43-step resets may reach different territory than
50K-step continuous argmin, which eventually cycles on visited cells.

5 seeds × 60s, LS20, k=12 LSH.
Compare L1 to 620 baseline [1362, 3270, 48391, 62727, 846].
"""
import numpy as np
import time
import sys

N_A = 4
K = 12
DIM = 256
REFINE_EVERY = 5000
MIN_OBS = 8
H_SPLIT = 0.05
MAX_STEPS = 50_001
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
        self.G = {}       # global: (N, A) -> {successor: count} — for refinement
        self.G_ep = {}    # episode-local: (N, A) -> count — for action selection
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self.t = 0
        self.ns = 0
        self.dim = dim
        self.ep_count = 0
        self.cells_per_ep = []
        self._ep_cells = set()

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
        self._ep_cells.add(n)
        self.t += 1

        if self._pn is not None:
            # Update global G (for refinement)
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            # Update episode-local counts
            key = (self._pn, self._pa)
            self.G_ep[key] = self.G_ep.get(key, 0) + 1
            # Update C (for refinement)
            k = (self._pn, self._pa, n)
            s, c = self.C.get(k, (np.zeros(self.dim, np.float64), 0))
            self.C[k] = (s + self._px.astype(np.float64), c + 1)

        self._px = x
        self._cn = n

        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        # Episode-local argmin
        counts = [self.G_ep.get((self._cn, a), 0) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def _end_episode(self):
        self.cells_per_ep.append(len(self._ep_cells))
        self.G_ep = {}
        self._ep_cells = set()
        self.ep_count += 1
        self._pn = None

    def on_reset(self):
        self._end_episode()

    def avg_cells_per_ep(self):
        return float(np.mean(self.cells_per_ep)) if self.cells_per_ep else 0.0

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
    sub._cn = 1
    sub.live.add(1)

    # Fresh episode: all counts 0 → pick action 0
    action = sub.act()
    assert action == 0, f"fresh episode: pick action 0, got {action}"

    # Mark action 0 as tried once
    sub.G_ep[(1, 0)] = 1
    sub._cn = 1
    action = sub.act()
    assert action == 1, f"action 0 tried, pick 1, got {action}"

    # After reset: G_ep cleared → pick 0 again
    sub.on_reset()
    sub._cn = 1
    sub.live.add(1)
    action = sub.act()
    assert action == 0, f"after reset: fresh start, pick 0, got {action}"

    print("T0 PASS")


def run(seed, make):
    env = make()
    sub = Recode(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0
    l1 = l2 = None
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

    avg_ep = sub.avg_cells_per_ep()
    bsl = BASELINE_L1[seed]
    if l1:
        ratio = bsl / l1
        spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
    else:
        spd = "NO_L1"
    print(f"  s{seed}: L1={l1} ({spd}) go={go} unique={len(sub.live)} "
          f"avg_ep_cells={avg_ep:.1f} eps={sub.ep_count}", flush=True)

    return dict(seed=seed, l1=l1, l2=l2, go=go,
                unique=len(sub.live), avg_ep=avg_ep, ep_count=sub.ep_count)


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
    avg_ep = np.mean([r['avg_ep'] for r in R])

    for r in R:
        bsl = BASELINE_L1[r['seed']]
        if r['l1']:
            ratio = bsl / r['l1']
            spd = f"{ratio:.1f}x faster" if ratio > 1.0 else f"{1/ratio:.1f}x slower"
        else:
            spd = "NO_L1"
        print(f"  s{r['seed']}: L1={r['l1']} ({spd}) "
              f"unique={r['unique']} avg_ep_cells={r['avg_ep']:.1f} eps={r['ep_count']}")

    print(f"\nL1={l1n}/5  avg_speedup={avg_ratio:.2f}x  avg_ep_cells={avg_ep:.1f}")

    if l1n >= 4 and avg_ratio > 1.0:
        print("SIGNAL: per-episode exploration faster than baseline.")
    elif l1n < 3:
        print("KILL: L1 < 3/5.")
    else:
        print(f"MARGINAL: L1={l1n}/5 avg_speedup={avg_ratio:.2f}x")


if __name__ == "__main__":
    main()
