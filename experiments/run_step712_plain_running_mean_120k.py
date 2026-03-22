"""
Step 712 — Plain k=12 with running-mean centering, LS20, 20 seeds, 120K.

Frame-local plain k=12 at 120K = 16/20 (Step 701).
Running-mean 674 at 120K = 20/20 (Step 708).
Isolation: does running-mean centering alone (no 674 aliasing) reach 20/20?
If yes: centering is the variable, not 674's disambiguation mechanism.
"""
import numpy as np
import time
import sys

K = 12
DIM = 256
MAX_STEPS = 120_001
N_SEEDS = 20
N_A = 4


def enc_raw(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


class PlainRunningMean:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = None
        self._cn = None
        self._mu = np.zeros(DIM, dtype=np.float32)
        self._mu_n = 0

    def _hash(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def observe(self, frame):
        x_raw = enc_raw(frame)
        self._mu_n += 1
        self._mu = self._mu + (x_raw - self._mu) / self._mu_n
        x = x_raw - self._mu
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

    def on_reset(self):
        self._pn = None
        self._mu = np.zeros(DIM, dtype=np.float32)
        self._mu_n = 0


def run(seed, make):
    env = make()
    sub = PlainRunningMean(seed=seed * 1000)
    obs = env.reset(seed=seed)
    level = 0; l1 = None
    t_start = time.time()

    for step in range(1, MAX_STEPS):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); continue
        sub.observe(obs); action = sub.act()
        obs, reward, done, info = env.step(action)
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
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    t_start = time.time()
    print(f"Step 712: Plain k=12 running-mean centering, LS20, {N_SEEDS} seeds, {MAX_STEPS-1} steps")
    print(f"Game: LS20 9607627b. Isolation: centering change alone vs 674 mechanism.")
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
    print(f"Compare: frame-local plain k=12 at 120K = 16/20 (Step 701)")
    print(f"Compare: running-mean 674 at 120K = 20/20 (Step 708)")
    if l1_n >= 20:
        print("FINDING: Running-mean centering alone = 20/20. Centering is the variable, not 674.")
    elif l1_n > 16:
        print(f"FINDING: Running-mean centering alone = {l1_n}/20. Partial centering benefit.")
    else:
        print(f"FINDING: Running-mean centering alone = {l1_n}/20. 674 mechanism adds coverage beyond centering.")


if __name__ == "__main__":
    main()
