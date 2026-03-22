"""
Step 537 -- SplitTree: edge transfer + threshold=64 (best from Step 536).

Step 535: edge transfer alone, threshold=32 -> 0/5, 1693 splits.
Step 536: no threshold works (64-512 all 0/3). Best was threshold=64 (488 cells).
This combines both fixes.

Root cause hypothesis: SplitTree is fully deterministic (no random element).
All seeds produce identical trajectories. argmin policy + deterministic splits
= fixed trajectory loop. Threshold and edge transfer don't break determinism.

3 seeds, 50K steps. Prediction: 0/3. Both fixes fail independently -> combined fails.
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MAX_STEPS = 50_000
N_SEEDS = 3
THRESHOLD = 64
TIME_CAP = 270


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SplitTreeCombined:
    """SplitTree: edge transfer + configurable threshold."""

    def __init__(self, na, threshold=THRESHOLD):
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

        # Edge transfer
        for (pa, c_next), (mean_pz, count) in list(self.R.get(c, {}).items()):
            child = l if mean_pz[bd] < bv else r
            cg = self.G.setdefault((child, pa), {})
            cg[c_next] = cg.get(c_next, 0) + count
            cr = self.R.setdefault(child, {})
            if (pa, c_next) in cr:
                old_mean, old_count = cr[(pa, c_next)]
                total = old_count + count
                merged = [(old_mean[i] * old_count + mean_pz[i] * count) / total
                          for i in range(self.d)]
                cr[(pa, c_next)] = [merged, total]
            else:
                cr[(pa, c_next)] = [mean_pz[:], count]
        if c in self.R:
            del self.R[c]
        for pa in range(self.A):
            self.G.pop((c, pa), None)


def t1():
    s = SplitTreeCombined(4, threshold=THRESHOLD)
    x = [0.1] * 256
    a = s(x)
    assert 0 <= a < 4
    assert s.threshold == THRESHOLD
    print(f"T1 PASS (threshold={THRESHOLD})")


def run_seed(seed, arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    s = SplitTreeCombined(4, threshold=THRESHOLD)
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
    print(f"  seed={seed}: {tag:12s}  cells={n_cells:4d}  splits={s.splits:3d}  "
          f"deaths={deaths}  {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1_step,
                win=(obs is not None and obs.state == GameState.WIN),
                cells=n_cells, splits=s.splits, deaths=deaths)


def main():
    t1()

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"\nStep 537: SplitTree combined (edge_transfer + threshold={THRESHOLD}).",
          flush=True)
    print(f"535: edge_transfer threshold=32 -> 0/5. "
          f"536: threshold={THRESHOLD} alone -> 0/3.", flush=True)

    t_total = time.time()
    results = []
    for seed in range(N_SEEDS):
        results.append(run_seed(seed, arc, ls20.game_id))

    wins = sum(1 for r in results if r['win'])
    l1s = sum(1 for r in results if r['l1'])
    max_cells = max(r['cells'] for r in results)
    max_splits = max(r['splits'] for r in results)

    print(f"\n{'='*55}", flush=True)
    print(f"STEP 537 SUMMARY (edge_transfer + threshold={THRESHOLD})", flush=True)
    print(f"  Full WIN:   {wins}/{N_SEEDS}", flush=True)
    print(f"  L1:         {l1s}/{N_SEEDS}", flush=True)
    print(f"  max_cells:  {max_cells}", flush=True)
    print(f"  max_splits: {max_splits}", flush=True)
    print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if l1s > 0:
        print(f"\nSIGNAL: Combined fix navigates. Run Step 538 (chain).", flush=True)
    else:
        print(f"\nCONFIRMED: SplitTree fails regardless of threshold or edge transfer.", flush=True)
        print(f"Root cause: fully deterministic (no random element). "
              f"All seeds identical trajectories. Need randomness or reward signal.", flush=True)


if __name__ == "__main__":
    main()
