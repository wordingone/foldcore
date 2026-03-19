"""
Step 534 -- Substrate S on LS20. Jun's 11th family: self-partitioning decision tree.

S starts with 1 leaf (root=0). Splits when 2+ (action->next_cell) transition patterns
diverge: finds best feature dimension separating the top-2 (by count) edge means.
Split criteria: total >= 32 transitions, 2+ pairs with >= 4 each, separation > 1e-9.

10/10 WIN on 8x8 toy grid @ avg 2006 steps.
Passes codebook ban: no cosine, no k-means, no LVQ, no spatial engine.

Wiring:
- encode(frame): avgpool16 + centered -> 256D numpy array
- S(x): returns action index 0-3
- On death/reset: s.p = None (keep tree/edges, reset episode boundary)
- n_actions = 4

Records: L1 step, tree cells (1+len(s.T)), edges |G|, splits fired, deaths.

Predictions: 3/5 at 50K. Tree must grow to ~300+ LS20 states.
Kill: 0/5 with <10 cells -> splits don't fire on LS20 observations.
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


# --- Substrate S (Jun's implementation, unmodified) ---

class S:

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


# --- Tests ---

def t1():
    # Test 1: S initializes and returns valid action
    s = S(4)
    x = [0.1] * 256
    a = s(x)
    assert 0 <= a < 4, f"action out of range: {a}"
    assert s.n == 1
    assert s.mu is not None
    assert len(s.mu) == 256

    # Test 2: Reset via s.p = None keeps tree/edges
    s.p = None
    a2 = s(x)
    assert 0 <= a2 < 4
    assert s.n == 2  # step counter keeps incrementing

    # Test 3: encode returns 256D zero-mean array
    dummy_frame = [np.zeros((64, 64))]
    enc = encode(dummy_frame)
    assert enc.shape == (256,)
    assert abs(enc.mean()) < 1e-5

    # Test 4: encode works with non-trivial frame
    rng = np.random.RandomState(42)
    frame = [rng.randint(0, 16, (64, 64))]
    enc2 = encode(frame)
    assert enc2.shape == (256,)

    # Test 5: split fires when sufficient transitions accumulated
    s2 = S(2)
    rng2 = np.random.RandomState(0)
    # Feed two distinct patterns many times
    x_a = list(rng2.randn(256).astype(float))
    x_b = list((rng2.randn(256) + 5.0).astype(float))  # offset to ensure separability
    for _ in range(40):
        s2(x_a)
        s2(x_b)
    # After enough transitions, expect at least 1 split
    assert len(s2.T) >= 1, f"Expected split to fire, T={s2.T}"

    print(f"T1 PASS (splits fired: {len(s2.T)}, cells: {1 + len(s2.T)})")


def run_seed(seed, arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    s = S(4)
    obs = env.reset()
    ts = deaths = 0
    l1_step = None
    splits_prev = 0
    splits_total = 0
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

        # Count new splits
        new_splits = len(s.T) - splits_prev
        if new_splits > 0:
            splits_total += new_splits
            splits_prev = len(s.T)

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
          f"splits={splits_total:3d}  deaths={deaths}  {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1_step,
                win=(obs is not None and obs.state == GameState.WIN),
                cells=n_cells, edges=n_edges, splits=splits_total, deaths=deaths)


def main():
    t1()

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    print(f"\nStep 534: Substrate S on LS20. {N_SEEDS} seeds, {MAX_STEPS//1000}K steps.",
          flush=True)
    print(f"Prediction: 3/5 WIN. Kill: 0/5 with <10 cells.", flush=True)

    t_total = time.time()
    results = []
    for seed in range(N_SEEDS):
        results.append(run_seed(seed, arc, ls20.game_id))

    wins = sum(1 for r in results if r['win'])
    l1s = sum(1 for r in results if r['l1'])
    max_cells = max(r['cells'] for r in results)
    max_splits = max(r['splits'] for r in results)
    total_edges = max(r['edges'] for r in results)

    print(f"\n{'='*55}", flush=True)
    print(f"STEP 534 SUMMARY", flush=True)
    print(f"  Full WIN:  {wins}/{N_SEEDS}", flush=True)
    print(f"  L1:        {l1s}/{N_SEEDS}", flush=True)
    print(f"  max_cells: {max_cells}", flush=True)
    print(f"  max_edges: {total_edges}", flush=True)
    print(f"  max_splits:{max_splits}", flush=True)
    print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if l1s > 0:
        print(f"\nSIGNAL: L1 achieved. Tree is navigating LS20.", flush=True)
    elif max_cells >= 10:
        print(f"\nSPLITS FIRE but no navigation. Tree partitions space "
              f"but action policy insufficient.", flush=True)
    else:
        print(f"\nKILL: {max_cells} cells. Splits don't fire on LS20 observations.",
              flush=True)


if __name__ == "__main__":
    main()
