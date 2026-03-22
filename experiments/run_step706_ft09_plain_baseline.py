"""
Step 706 — Plain k=12 on FT09, 20 seeds, 120K steps.

674 FT09 at 120K = 17/20 (Step 702).
Does plain k=12 also get 17/20? If yes: 674 adds nothing on FT09.
If lower: 674 helps even with minimal aliasing (1-4 cells).

FT09 action space: 8x8 click grid = 64 actions.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
MAX_STEPS = 120_001
N_SEEDS = 20

GRID_ACTIONS = [(gx * 8 + 4) + (gy * 8 + 4) * 64
                for gy in range(8) for gx in range(8)]
N_A = len(GRID_ACTIONS)  # 64


def enc_frame(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class PlainArgminFT09:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._px = None
        self._cn = None

    def _hash(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._hash(x)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._px = x; self._cn = n
        return n

    def act(self):
        best_a, best_s = 0, float('inf')
        for a in range(N_A):
            s = sum(self.G.get((self._cn, a), {}).values())
            if s < best_s: best_s = s; best_a = a
        self._pn = self._cn; self._pa = best_a; return best_a

    def on_reset(self): self._pn = None


def run(seed, make):
    env = make()
    sub = PlainArgminFT09(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = GRID_ACTIONS[action_idx]
        obs, reward, done, info = env.step(action_int)
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None:
                l1 = step
                print(f"  s{seed} L1@{l1}", flush=True)
            level = cl; sub.on_reset()
        if done:
            obs = env.reset(seed=seed); sub.on_reset()

    elapsed = time.time() - t_start
    status = f"L1={l1}" if l1 else "NO_L1"
    print(f"  s{seed:2d}: {status} t={elapsed:.1f}s", flush=True)
    return dict(seed=seed, l1=l1)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("FT09")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 706: Plain k=12 on FT09, {N_SEEDS} seeds, {MAX_STEPS-1} steps cap")
    results = []
    for seed in range(N_SEEDS):
        results.append(run(seed, mk))

    print(f"\n{'='*60}")
    l1_n = sum(1 for r in results if r['l1'])
    elapsed = time.time() - t_start
    print(f"L1={l1_n}/{N_SEEDS}  total_time={elapsed:.1f}s")
    for r in results:
        status = f"L1={r['l1']}" if r['l1'] else "NO_L1"
        print(f"  s{r['seed']:2d}: {status}")
    print(f"BASELINE: Plain k=12 FT09 at 120K = {l1_n}/{N_SEEDS}")
    print(f"Compare: 674 frame-local FT09 at 120K = 17/20 (Step 702)")
    if l1_n >= 17: print("NOTE: 674 adds nothing on FT09 — aliasing irrelevant")
    else: print(f"NOTE: 674 helps even with minimal aliasing (+{17-l1_n} seeds)")


if __name__ == "__main__":
    main()
