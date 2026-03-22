"""
Step 590 -- K sensitivity sweep for SoftPenalty (581d) on LS20.

Research question: is K=12 principled or arbitrary?
Tests K=8, 10, 12, 14, 16 with SoftPenalty (permanent soft death penalty).
5 seeds x 10K steps each. Fits 5-min cap per condition.
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
K_VALUES = [8, 10, 12, 14, 16]


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class SoftPenalty:
    def __init__(self, seed=0, k=12):
        self.H = np.random.RandomState(seed).randn(k, DIM).astype(np.float32)
        self.G = {}
        self.death_edges = set()
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

    def on_reset(self):
        self._pn = None


def run_seed(mk, seed, k):
    env = mk()
    sub = SoftPenalty(seed=seed * 100 + 7, k=k)
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
        prev_cl = cl

    print(f"  s{seed}: L1={l1} go={go} step={step} cells={len(sub.cells)} "
          f"deaths={sub.total_deaths} {time.time()-t0:.0f}s", flush=True)
    return dict(seed=seed, l1=l1, go=go, steps=step,
                cells=len(sub.cells), deaths=sub.total_deaths)


def run_k(mk, k):
    print(f"\n--- K={k} ---", flush=True)
    results = []
    for seed in range(N_SEEDS):
        try:
            results.append(run_seed(mk, seed, k))
        except Exception as e:
            print(f"  s{seed} FAIL: {e}", flush=True)
    wins = sum(1 for r in results if r['l1'] > 0)
    l1 = sum(r['l1'] for r in results)
    avg_cells = np.mean([r['cells'] for r in results]) if results else 0
    print(f"  K={k}: {wins}/{N_SEEDS} L1={l1} avg_cells={avg_cells:.0f}", flush=True)
    return wins, l1, avg_cells


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 590: K sensitivity sweep -- SoftPenalty on LS20", flush=True)
    print(f"  K={K_VALUES} | {N_SEEDS} seeds x {MAX_STEPS} steps | PENALTY={PENALTY}", flush=True)

    t_total = time.time()
    results = {}
    for k in K_VALUES:
        wins, l1, avg_cells = run_k(mk, k)
        results[k] = (wins, l1, avg_cells)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 590: K sensitivity sweep ({N_SEEDS} seeds, {MAX_STEPS} steps)", flush=True)
    print(f"  {'K':>4} | {'Seeds':>7} | {'L1':>4} | {'Cells':>6}", flush=True)
    print(f"  {'-'*35}", flush=True)
    for k in K_VALUES:
        wins, l1, avg_cells = results[k]
        print(f"  {k:>4} | {wins:>5}/{N_SEEDS} | {l1:>4} | {avg_cells:>6.0f}", flush=True)

    best_k = max(results, key=lambda k: (results[k][0], results[k][1]))
    print(f"\n  Best K: {best_k} ({results[best_k][0]}/{N_SEEDS} wins)", flush=True)
    k12_wins = results[12][0]
    best_wins = results[best_k][0]
    if k12_wins == best_wins:
        print(f"  CONFIRMED: K=12 is optimal or tied-best.", flush=True)
    else:
        print(f"  K=12 NOT optimal. K={best_k} wins ({best_wins} vs {k12_wins}).", flush=True)

    print(f"\n  Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
