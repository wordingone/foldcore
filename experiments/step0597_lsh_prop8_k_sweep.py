"""
Step 597 -- Proposition 8 test: K-dependent random baseline.

Proposition 8: argmin advantage grows with K (larger graph = random walk slower).
  K=8:  ~70 cells  → random should nearly match argmin
  K=16: ~450 cells → argmin should dominate

Three K values, Random vs Argmin, 5 seeds, 10K steps.
Fisher exact (Argmin > Random) at each K.
"""
import numpy as np
import time
import sys

DIM = 256
N_A = 4
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5
K_VALUES = [8, 12, 16]


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class RandomAgent:
    def __init__(self, seed=0, k=12):
        self.rng = np.random.RandomState(seed)
        self.total_deaths = 0

    def observe(self, frame): pass
    def act(self): return int(self.rng.randint(N_A))
    def on_death(self): self.total_deaths += 1
    def on_reset(self): pass


class Argmin:
    def __init__(self, seed=0, k=12):
        self.H = np.random.RandomState(seed).randn(k, DIM).astype(np.float32)
        self.G = {}; self._pn = self._pa = self._cn = None
        self.cells = set(); self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame); n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; return action

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None


def run_seed(mk, seed, SubClass, k):
    env = mk(); sub = SubClass(seed=seed * 100 + 7, k=k)
    obs = env.reset(seed=seed); sub.on_reset()
    l1 = go = step = 0; prev_cl = 0; fresh = True; t0 = time.time()
    l1_first = None

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        sub.observe(obs); action = sub.act()
        obs, _, done, info = env.step(action); step += 1
        if done:
            sub.on_death(); obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh: prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1_first is None: l1_first = step
        prev_cl = cl

    cells = len(getattr(sub, 'cells', set()))
    return dict(l1=l1, l1_first=l1_first, cells=cells)


def run_pair(mk, k):
    rand_wins = am_wins = 0
    print(f"\n  K={k}:", flush=True)
    for seed in range(N_SEEDS):
        r = run_seed(mk, seed, RandomAgent, k)
        a = run_seed(mk, seed, Argmin, k)
        rand_wins += (r['l1'] > 0); am_wins += (a['l1'] > 0)
        print(f"    s{seed}: Rand L1={r['l1']} cells={r['cells']} | "
              f"Argmin L1={a['l1']} cells={a['cells']}", flush=True)

    line = f"  K={k}: Random {rand_wins}/{N_SEEDS}  Argmin {am_wins}/{N_SEEDS}"
    try:
        from scipy.stats import fisher_exact
        tbl = [[am_wins, N_SEEDS - am_wins], [rand_wins, N_SEEDS - rand_wins]]
        _, pval = fisher_exact(tbl, alternative='greater')
        line += f"  Fisher p={pval:.4f}"
    except ImportError: pass
    print(line, flush=True)
    return rand_wins, am_wins


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 597: Proposition 8 test -- K-dependent random baseline", flush=True)
    print(f"  {N_SEEDS} seeds x {MAX_STEPS} steps per condition", flush=True)

    t0 = time.time()
    results = {}
    for k in K_VALUES:
        results[k] = run_pair(mk, k)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 597: Summary", flush=True)
    print(f"  {'K':>4}  {'Random':>8}  {'Argmin':>8}  {'Gap':>6}", flush=True)
    for k in K_VALUES:
        rw, aw = results[k]
        print(f"  K={k:2d}  {rw}/{N_SEEDS}       {aw}/{N_SEEDS}      {aw-rw:+d}", flush=True)

    gaps = [results[k][1] - results[k][0] for k in K_VALUES]
    if gaps[-1] > gaps[0]:
        print(f"\n  PROP 8 SUPPORTED: gap grows with K ({gaps[0]:+d} at K=8 → {gaps[-1]:+d} at K=16).", flush=True)
    else:
        print(f"\n  PROP 8 NOT SUPPORTED: gap does not grow ({gaps[0]:+d} at K=8, {gaps[-1]:+d} at K=16).", flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
