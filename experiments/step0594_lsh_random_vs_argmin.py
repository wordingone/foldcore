"""
Step 594 -- Random vs Argmin, 20 seeds, 50K steps. Reviewer validation.

Step 592 showed Random 2/5 vs Argmin 3/5 at 10K/5 seeds -- gap real but noisy.
This is the definitive comparison: same format as Step 584 (20 seeds, 50K).

Approved 50K format in Step 584; same approval applies here.

Two conditions:
  A) Random -- uniform random from {0,1,2,3}
  B) Argmin -- LSH K=12, visit-count minimization

Fisher exact test: Argmin vs Random (one-sided, Argmin > Random).

Expected: Random ~3/20 (lucky stumbles), Argmin ~13/20.
If random >= 10/20, our core claim about argmin weakens substantially.
"""
import numpy as np
import time
import sys

N_A = 4
DIM = 256
MAX_STEPS = 50_000
TIME_CAP = 300   # 5 min per seed (same as step 584)
N_SEEDS = 20


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
    """LSH K=12 argmin -- visit-count minimization. Same as step 584 baseline."""

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


def run_seed(mk, seed, SubClass):
    env = mk()
    sub = SubClass(seed=seed * 100 + 7)
    obs = env.reset(seed=seed)
    sub.on_reset()

    l1 = l2 = go = step = 0
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
        elif cl >= 2 and prev_cl < 2:
            l2 += 1
        prev_cl = cl

    cells = len(getattr(sub, 'cells', set()))
    print(f"  s{seed}: L1={l1} L2={l2} go={go} step={step} cells={cells} "
          f"deaths={sub.total_deaths} {time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, l2=l2, go=go, steps=step,
                cells=cells, deaths=sub.total_deaths)


def run_condition(mk, label, SubClass):
    print(f"\n--- {label} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        print(f"\nseed {seed}:", flush=True)
        try:
            results.append(run_seed(mk, seed, SubClass))
        except Exception as e:
            print(f"  FAIL: {e}", flush=True)
    wins = sum(1 for r in results if r['l1'] > 0)
    l1 = sum(r['l1'] for r in results)
    print(f"  {label}: {wins}/{N_SEEDS} L1={l1}", flush=True)
    return results


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 594: Random vs Argmin -- {N_SEEDS} seeds x {MAX_STEPS} steps", flush=True)

    t_total = time.time()
    rand_r = run_condition(mk, "Random", RandomAgent)
    am_r = run_condition(mk, "Argmin (K=12)", Argmin)

    def summ(r):
        return sum(x['l1'] for x in r), sum(1 for x in r if x['l1'] > 0)

    rand_l1, rand_s = summ(rand_r)
    am_l1, am_s = summ(am_r)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 594: Random vs Argmin ({N_SEEDS} seeds, {MAX_STEPS} steps)", flush=True)
    print(f"  Random:       {rand_s}/{N_SEEDS} seeds L1={rand_l1}", flush=True)
    print(f"  Argmin K=12:  {am_s}/{N_SEEDS} seeds L1={am_l1}", flush=True)

    if am_s > rand_s:
        print(f"\n  ARGMIN CONFIRMED: {am_s} > {rand_s}. Visit-count minimization is essential.", flush=True)
    elif am_s == rand_s:
        print(f"\n  NO DIFFERENCE: {am_s} == {rand_s}. Argmin claim weakened.", flush=True)
    else:
        print(f"\n  INVERSION: Random {rand_s} > Argmin {am_s}. Serious concern.", flush=True)

    try:
        from scipy.stats import fisher_exact
        table = [[am_s, N_SEEDS - am_s], [rand_s, N_SEEDS - rand_s]]
        odds, pval = fisher_exact(table, alternative='greater')
        print(f"  Fisher (Argmin > Random): odds={odds:.3f} p={pval:.4f}", flush=True)
    except ImportError:
        pass

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
