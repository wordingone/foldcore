"""
Step 711 — Plain k=12 on FT09, 20 seeds, 25s cap.

Step 706: Plain k=12 FT09 at 120K = 8/20.
What's the 25s count? Needed for full comparison table.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
CAP_SECONDS = 25
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
        self._pn = self._pa = None
        self._cn = None

    def _hash(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def observe(self, frame):
        x = enc_frame(frame)
        n = self._hash(x)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n
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
    t_start = time.time(); step = 0

    while time.time() - t_start < CAP_SECONDS:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs)
        action_idx = sub.act()
        action_int = GRID_ACTIONS[action_idx]
        obs, reward, done, info = env.step(action_int)
        step += 1
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1 is None: l1 = step
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
    print(f"Step 711: Plain k=12 on FT09, {N_SEEDS} seeds, {CAP_SECONDS}s cap")
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
    print(f"BASELINE: Plain k=12 FT09 at 25s = {l1_n}/{N_SEEDS}")
    print(f"Compare: Plain k=12 FT09 at 120K = 8/20 (Step 706)")
    print(f"Compare: 674 frame-local FT09 at 120K = 17/20 (Step 702)")


if __name__ == "__main__":
    main()
