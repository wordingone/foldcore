"""
Step 592 -- Argmin vs random baseline (sanity check).

We claim argmin is essential for navigation. But we've never tested against pure random.
This is the reviewer's most basic missing comparison.

Three conditions, 5 seeds, 10K steps:
  A) Random action selection -- uniform random from {0,1,2,3}
  B) Argmin (K=12)          -- visit-count minimization baseline
  C) SoftPenalty (K=12)     -- argmin + death avoidance (581d)

Expected: Random << Argmin ~ SoftPenalty. If random gets L1, argmin is not essential.
"""
import numpy as np
import time
import sys

N_A = 4
PENALTY = 100
DIM = 256
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 5


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class RandomAgent:
    """Pure random action selection -- null hypothesis."""

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.total_deaths = 0

    def observe(self, frame):
        pass

    def act(self):
        return int(self.rng.randint(N_A))

    def on_death(self):
        self.total_deaths += 1

    def on_reset(self):
        pass


class Argmin:
    """LSH K=12 argmin -- visit-count minimization, no death penalty."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(12, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()
        self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame)
        n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        self.total_deaths += 1

    def on_reset(self):
        self._pn = None


class SoftPenalty(Argmin):
    """Argmin + permanent soft death penalty (581d)."""

    def __init__(self, seed=0):
        super().__init__(seed=seed)
        self.death_edges = set()

    def act(self):
        counts = np.array(
            [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)],
            dtype=np.float64
        )
        for a in range(N_A):
            if (self._cn, a) in self.death_edges:
                counts[a] += PENALTY
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        return action

    def on_death(self):
        if self._pn is not None:
            self.death_edges.add((self._pn, self._pa))
        self.total_deaths += 1


def run_seed(mk, seed, SubClass):
    env = mk()
    sub = SubClass(seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = go = step = 0
    prev_cl = 0
    fresh = True
    t0 = time.time()

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        sub.observe(obs)
        action = sub.act()
        obs, _, done, info = env.step(action)
        step += 1
        if done:
            sub.on_death()
            obs = env.reset(seed=seed)
            sub.on_reset()
            prev_cl = 0; fresh = True; go += 1
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh:
            prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1 <= 2:
                print(f"    s{seed} L1@{step}", flush=True)
        prev_cl = cl

    cells = getattr(sub, 'cells', None)
    cells_n = len(cells) if cells is not None else 0
    print(f"  s{seed}: L1={l1} go={go} step={step} cells={cells_n} "
          f"deaths={sub.total_deaths} {time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, go=go, steps=step,
                cells=cells_n, deaths=sub.total_deaths)


def run_condition(mk, label, SubClass):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        try:
            results.append(run_seed(mk, seed, SubClass))
        except Exception as e:
            print(f"  s{seed} FAIL: {e}", flush=True)
    wins = sum(1 for r in results if r['l1'] > 0)
    l1 = sum(r['l1'] for r in results)
    print(f"  {label}: {wins}/{N_SEEDS} L1={l1}", flush=True)
    return wins, l1


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 592: Argmin vs random baseline sanity check on LS20", flush=True)
    print(f"  {N_SEEDS} seeds x {MAX_STEPS} steps", flush=True)

    t_total = time.time()
    rand_wins, rand_l1 = run_condition(mk, "Random", RandomAgent)
    am_wins, am_l1 = run_condition(mk, "Argmin (K=12)", Argmin)
    sp_wins, sp_l1 = run_condition(mk, "SoftPenalty (K=12)", SoftPenalty)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 592: Argmin vs random baseline", flush=True)
    print(f"  Random:            {rand_wins}/{N_SEEDS} L1={rand_l1}", flush=True)
    print(f"  Argmin (K=12):     {am_wins}/{N_SEEDS} L1={am_l1}", flush=True)
    print(f"  SoftPenalty(K=12): {sp_wins}/{N_SEEDS} L1={sp_l1}", flush=True)

    if rand_wins == 0 and am_wins > 0:
        print(f"\n  CONFIRMED: Argmin essential. Random=0, Argmin={am_wins}.", flush=True)
        print(f"  Visit-count minimization is the active mechanism.", flush=True)
    elif rand_wins > 0:
        print(f"\n  UNEXPECTED: Random gets {rand_wins}/5. Argmin may not be the key mechanism.", flush=True)
    elif am_wins == 0:
        print(f"\n  CONCERN: Both random and argmin get 0/5 at 10K. Budget too small?", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
