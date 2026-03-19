"""
Step 535 -- SplitTree with edge transfer on split.

Step 534: 810 cells, 809 splits, 0/5 FAIL. New child cells after splits
have empty G -> _act returns action 0 always -> no exploration.

Fix: when cell c splits into children (l, r), re-assign G and R entries
from c to the appropriate child based on which child the stored mean pz
maps to (pz[bd] < bv -> l, else r). Parent G/R entries deleted after transfer.

Split threshold: 32 (unchanged from Step 534).

Predictions: 3/5 at 50K. Edge transfer prevents empty-cell degeneration.
Kill: 0/5 -> edge transfer doesn't help (other issue causing FAIL).
"""
import time
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MAX_STEPS = 50_000
N_SEEDS = 5
TIME_CAP = 270


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class SplitTreeEdge:
    """SplitTree with edge transfer on split."""

    def __init__(self, na):
        self.A = na
        self.T = {}
        self.G = {}
        self.R = {}
        self.mu = None
        self.d = 0
        self.n = 0
        self.p = None
        self.k = 1
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
        if tn < 32 or len(pairs) < 2:
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

        # Edge transfer: re-assign R[c] and G[(c,*)] entries to children
        for (pa, c_next), (mean_pz, count) in list(self.R.get(c, {}).items()):
            child = l if mean_pz[bd] < bv else r
            # Transfer G
            cg = self.G.setdefault((child, pa), {})
            cg[c_next] = cg.get(c_next, 0) + count
            # Transfer R (merge if existing entry)
            cr = self.R.setdefault(child, {})
            if (pa, c_next) in cr:
                old_mean, old_count = cr[(pa, c_next)]
                total = old_count + count
                merged = [(old_mean[i] * old_count + mean_pz[i] * count) / total
                          for i in range(self.d)]
                cr[(pa, c_next)] = [merged, total]
            else:
                cr[(pa, c_next)] = [mean_pz[:], count]

        # Delete parent G and R entries
        if c in self.R:
            del self.R[c]
        for pa in range(self.A):
            self.G.pop((c, pa), None)


def t1():
    # Basic: returns valid action
    s = SplitTreeEdge(4)
    x = [0.1] * 256
    a = s(x)
    assert 0 <= a < 4

    # After split, children should have non-empty G (edges transferred)
    s2 = SplitTreeEdge(2)
    rng = np.random.RandomState(0)
    # Two distinct patterns, offset to ensure separability
    x_a = list(rng.randn(256).astype(float))
    x_b = list((rng.randn(256) + 10.0).astype(float))
    for _ in range(40):
        s2(x_a)
        s2(x_b)
    assert len(s2.T) >= 1, "Expected at least 1 split"

    # Check that children have edges (not empty)
    if len(s2.T) >= 1:
        # Find a leaf cell that has edges
        leaf_with_edges = [c for c in range(s2.k)
                           if c not in s2.T
                           and any(s2.G.get((c, a)) for a in range(2))]
        assert len(leaf_with_edges) > 0, \
            f"No leaf with edges after split. G keys: {list(s2.G.keys())[:10]}"

    print(f"T1 PASS (splits={s2.splits}, cells={1+len(s2.T)}, "
          f"leaves_with_edges={len(leaf_with_edges)})")


def run_seed(seed, arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    s = SplitTreeEdge(4)
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
    n_edges = len(s.G)
    tag = f"WIN@{l1_step}" if obs and obs.state == GameState.WIN else \
          (f"L1@{l1_step}" if l1_step else "FAIL")
    print(f"  seed={seed}: {tag:12s}  cells={n_cells:4d}  edges={n_edges:5d}  "
          f"splits={s.splits:3d}  deaths={deaths}  {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1_step,
                win=(obs is not None and obs.state == GameState.WIN),
                cells=n_cells, edges=n_edges, splits=s.splits, deaths=deaths)


def main():
    t1()

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"\nStep 535: SplitTree+EdgeTransfer on LS20. {N_SEEDS} seeds, "
          f"{MAX_STEPS//1000}K steps.", flush=True)
    print(f"Fix: parent G/R entries redistributed to children on split.", flush=True)
    print(f"Baseline (Step 534): 0/5, 810 cells, 809 splits, action-0 degeneration.",
          flush=True)

    t_total = time.time()
    results = []
    for seed in range(N_SEEDS):
        results.append(run_seed(seed, arc, ls20.game_id))

    wins = sum(1 for r in results if r['win'])
    l1s = sum(1 for r in results if r['l1'])
    max_cells = max(r['cells'] for r in results)
    max_splits = max(r['splits'] for r in results)

    print(f"\n{'='*55}", flush=True)
    print(f"STEP 535 SUMMARY (edge transfer, threshold=32)", flush=True)
    print(f"  Full WIN:   {wins}/{N_SEEDS}", flush=True)
    print(f"  L1:         {l1s}/{N_SEEDS}", flush=True)
    print(f"  max_cells:  {max_cells}", flush=True)
    print(f"  max_splits: {max_splits}", flush=True)
    print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if l1s > 0:
        print(f"\nSIGNAL: Edge transfer enables navigation. {l1s}/{N_SEEDS} L1.",
              flush=True)
    elif max_splits < 100:
        print(f"\nFEW SPLITS: {max_splits} splits. Tree not growing — threshold too high?",
              flush=True)
    else:
        print(f"\nFAIL: {max_splits} splits but no navigation. "
              f"Edge transfer insufficient fix.", flush=True)


if __name__ == "__main__":
    main()
