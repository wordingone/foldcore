"""
Step 599 -- Action space size effect.

Proposition 8 extended: smaller action space = smaller graph = random catches up faster.

Four conditions, 5 seeds, 10K:
  A) Random-2: uniform from {0, 1}
  B) Argmin-2: argmin over {0, 1}
  C) Random-4: uniform from {0, 1, 2, 3}  (baseline)
  D) Argmin-4: argmin over {0, 1, 2, 3}   (baseline)

Prediction: gap(Argmin-2 vs Random-2) < gap(Argmin-4 vs Random-4).
"""
import numpy as np
import time
import sys

DIM = 256
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5
K = 12


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class RandomAgent:
    def __init__(self, seed=0, n_actions=4):
        self.rng = np.random.RandomState(seed)
        self.n_actions = n_actions
        self.total_deaths = 0

    def observe(self, frame): pass
    def act(self): return int(self.rng.randint(self.n_actions))
    def on_death(self): self.total_deaths += 1
    def on_reset(self): pass


class Argmin:
    def __init__(self, seed=0, n_actions=4):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}; self._pn = self._pa = self._cn = None
        self.cells = set(); self.total_deaths = 0
        self.n_actions = n_actions

    def observe(self, frame):
        x = enc_vec(frame); n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values())
                  for a in range(self.n_actions)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; return action

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None


def run_seed(mk, seed, SubClass, n_actions):
    env = mk(); sub = SubClass(seed=seed * 100 + 7, n_actions=n_actions)
    obs = env.reset(seed=seed); sub.on_reset()
    l1 = go = step = 0; prev_cl = 0; fresh = True; t0 = time.time()

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
        elif cl >= 1 and prev_cl < 1: l1 += 1
        prev_cl = cl

    return dict(l1=l1, cells=len(getattr(sub, 'cells', set())))


def run_condition(mk, label, SubClass, n_actions):
    wins = 0
    print(f"\n  {label} (N_A={n_actions}):", flush=True)
    for seed in range(N_SEEDS):
        r = run_seed(mk, seed, SubClass, n_actions)
        wins += (r['l1'] > 0)
        print(f"    s{seed}: L1={r['l1']} cells={r['cells']}", flush=True)
    print(f"  {label}: {wins}/{N_SEEDS}", flush=True)
    return wins


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 599: Action space size effect", flush=True)
    print(f"  K={K} | {N_SEEDS} seeds x {MAX_STEPS} steps", flush=True)

    t0 = time.time()
    r2 = run_condition(mk, "Random-2", RandomAgent, 2)
    a2 = run_condition(mk, "Argmin-2", Argmin,      2)
    r4 = run_condition(mk, "Random-4", RandomAgent, 4)
    a4 = run_condition(mk, "Argmin-4", Argmin,      4)

    gap2 = a2 - r2
    gap4 = a4 - r4

    print(f"\n{'='*60}", flush=True)
    print(f"Step 599: Action space size effect", flush=True)
    print(f"  Random-2: {r2}/{N_SEEDS}  Argmin-2: {a2}/{N_SEEDS}  gap={gap2:+d}", flush=True)
    print(f"  Random-4: {r4}/{N_SEEDS}  Argmin-4: {a4}/{N_SEEDS}  gap={gap4:+d}", flush=True)

    if gap2 < gap4:
        print(f"\n  PROP 8 SUPPORTED: smaller action space narrows gap ({gap2:+d} < {gap4:+d}).", flush=True)
    elif gap2 == gap4:
        print(f"\n  NO EFFECT: action space size doesn't change gap.", flush=True)
    else:
        print(f"\n  INVERSION: smaller action space widens gap ({gap2:+d} > {gap4:+d}).", flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
