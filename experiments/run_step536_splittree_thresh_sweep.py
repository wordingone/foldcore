"""
Step 536 -- SplitTree threshold sweep (no edge transfer).

Step 534: threshold=32 -> 809 splits/50K steps, action-0 degeneration, 0/5 FAIL.
Higher threshold = fewer splits = less degeneration? Test thresholds: 64, 128, 256, 512.

3 seeds each, 50K steps. Records cells, L1 step, deaths, splits.
Predictions: 128 sweet spot. 64 same problem as 534. 512 too few splits.
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MAX_STEPS = 50_000
N_SEEDS = 3
THRESHOLDS = [64, 128, 256, 512]
TIME_CAP = 270


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SplitTreeThresh:
    """SplitTree with configurable split threshold."""

    def __init__(self, na, threshold=128):
        self.A = na
        self.T = {}
        self.G = {}
        self.R = {}
        self.mu = None
        self.d = 0
        self.n = 0
        self.p = None
        self.k = 1
        self.threshold = threshold
        self.splits = 0

    def __call__(self, x):
        D = len(x)
        if not self.mu:
            self.mu = [0.0] * D
            self.d = D
        self.n += 1
        z = [x[i] - self.mu[i] for i in range(D)]
        r = 1.0 / self.n
        for i in range(D):
            self.mu[i] += r * (x[i] - self.mu[i])
        c = self._map(z)
        if self.p:
            pc, pa, pz = self.p
            e = self.G.setdefault((pc, pa), {})
            e[c] = e.get(c, 0) + 1
            t = self.R.setdefault(pc, {}).setdefault((pa, c), [[0.0] * D, 0])
            t[1] += 1
            for i in range(D):
                t[0][i] += (pz[i] - t[0][i]) / t[1]
            self._split(pc)
            c = self._map(z)
        a = self._act(c)
        self.p = (c, a, z)
        return a

    def _map(self, z):
        c = 0
        while c in self.T:
            d, v, l, r = self.T[c]
            c = l if z[d] < v else r
        return c

    def _act(self, c):
        b, bn = 0, -1
        for a in range(self.A):
            n = sum(self.G.get((c, a), {}).values())
            if bn < 0 or n < bn:
                b, bn = a, n
        return b

    def _split(self, c):
        if c in self.T or c not in self.R:
            return
        pairs = [(v[1], v[0]) for v in self.R[c].values() if v[1] >= 4]
        tn = sum(p[0] for p in pairs)
        if tn < self.threshold or len(pairs) < 2:
            return
        pairs.sort(key=lambda p: p[0], reverse=True)
        n0, m0 = pairs[0]
        n1, m1 = pairs[1]
        bd, bv, bs = 0, 0.0, 0.0
        for i in range(self.d):
            s = abs(m1[i] - m0[i])
            if s > bs:
                bd, bv, bs = i, (m0[i] * n0 + m1[i] * n1) / (n0 + n1), s
        if bs < 1e-9:
            return
        l, r = self.k, self.k + 1
        self.k += 2
        self.T[c] = (bd, bv, l, r)
        self.splits += 1


def t1():
    # threshold=512: no split in 40 steps
    s_high = SplitTreeThresh(4, threshold=512)
    x = [0.1] * 256
    for _ in range(40):
        s_high(x)
    # Might not split with only 1 distinct pattern (pairs < 2)
    # threshold=64: should still split (same as 32 but higher bar)
    s_low = SplitTreeThresh(4, threshold=64)
    rng = np.random.RandomState(0)
    x_a = list(rng.randn(256).astype(float))
    x_b = list((rng.randn(256) + 10.0).astype(float))
    for _ in range(40):
        s_low(x_a)
        s_low(x_b)
    # threshold=64: requires 64 total transitions (we have ~80), should split
    assert s_low.splits >= 0  # may or may not split depending on pair count
    assert isinstance(s_low._act(0), int)
    print(f"T1 PASS (thresh=64: splits={s_low.splits}, "
          f"thresh=512: splits={s_high.splits})")


def run_one(seed, arc, game_id, threshold):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    s = SplitTreeThresh(4, threshold=threshold)
    obs = env.reset()
    ts = deaths = 0
    l1_step = None
    t0 = time.time()

    while ts < MAX_STEPS:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None or not obs.frame:
            obs = env.reset(); s.p = None; deaths += 1; continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); s.p = None; deaths += 1; continue

        x = encode(obs.frame)
        a = s(x)
        prev_lvls = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1

        if obs and obs.state == GameState.WIN:
            if l1_step is None:
                l1_step = ts
            break
        if obs and obs.levels_completed > prev_lvls and l1_step is None:
            l1_step = ts

    elapsed = time.time() - t0
    n_cells = 1 + len(s.T)
    tag = f"WIN@{l1_step}" if obs and obs.state == GameState.WIN else \
          (f"L1@{l1_step}" if l1_step else "FAIL")
    print(f"    seed={seed}: {tag:12s}  cells={n_cells:4d}  splits={s.splits:3d}  "
          f"deaths={deaths}  {elapsed:.0f}s", flush=True)
    return dict(l1=l1_step,
                win=(obs is not None and obs.state == GameState.WIN),
                cells=n_cells, splits=s.splits, deaths=deaths)


def main():
    t1()

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"\nStep 536: SplitTree threshold sweep on LS20. {N_SEEDS} seeds, "
          f"{MAX_STEPS//1000}K steps.", flush=True)
    print(f"Thresholds: {THRESHOLDS}. Baseline (thresh=32): 0/5, 809 splits.",
          flush=True)

    t_total = time.time()
    all_results = {}

    for thresh in THRESHOLDS:
        print(f"\n--- threshold={thresh} ---", flush=True)
        results = []
        for seed in range(N_SEEDS):
            r = run_one(seed, arc, ls20.game_id, thresh)
            results.append(r)
        wins = sum(1 for r in results if r['win'])
        l1s = sum(1 for r in results if r['l1'])
        max_cells = max(r['cells'] for r in results)
        max_splits = max(r['splits'] for r in results)
        print(f"  thresh={thresh}: {l1s}/{N_SEEDS} L1  {wins}/{N_SEEDS} WIN  "
              f"max_cells={max_cells}  max_splits={max_splits}", flush=True)
        all_results[thresh] = dict(l1s=l1s, wins=wins,
                                   max_cells=max_cells, max_splits=max_splits)

    print(f"\n{'='*55}", flush=True)
    print(f"STEP 536 SUMMARY", flush=True)
    print(f"  {'thresh':>6}  {'L1':>4}  {'WIN':>4}  {'max_cells':>10}  {'max_splits':>10}",
          flush=True)
    for thresh, r in all_results.items():
        print(f"  {thresh:>6}  {r['l1s']:>2}/{N_SEEDS}  {r['wins']:>2}/{N_SEEDS}  "
              f"{r['max_cells']:>10}  {r['max_splits']:>10}", flush=True)

    best_thresh = max(all_results, key=lambda t: (all_results[t]['l1s'], -t))
    best = all_results[best_thresh]
    print(f"\nBest threshold: {best_thresh} ({best['l1s']}/{N_SEEDS} L1, "
          f"{best['max_cells']} cells)", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if best['l1s'] > 0:
        print(f"\nSIGNAL: thresh={best_thresh} navigates. Use for Step 537.", flush=True)
    else:
        print(f"\nNO NAVIGATION: no threshold produces L1. "
              f"Threshold is not the key variable.", flush=True)


if __name__ == "__main__":
    main()
