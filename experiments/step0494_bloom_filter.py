#!/usr/bin/env python3
"""
Step 494 — Bloom Filter action selection (Family #9). NO graph, no edges, no nodes.
Per-action bloom filters: score[a] = bits set at hash positions. Argmin = least familiar.
Two variants:
  A) Magnitude hash: position = int(|projection * 1000|) % m  [NOT LSH, loses locality]
  B) LSH sign hash: position using sign bits of k hyperplanes   [LSH, preserves locality]
Prediction: A=0/10 (no local continuity), B possibly navigates (local continuity via LSH).
10 seeds, 50K steps, m=4096, k=12.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
M = 4096
MAX_STEPS = 50_000
TIME_CAP = 35  # per seed (~22s expected)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class BloomGraphMagnitude:
    """Variant A: magnitude-based hash (loses locality)."""
    def __init__(self, k=K, m=M, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.k = k
        self.m = m
        self.n_actions = n_actions
        self.filters = np.zeros((n_actions, m), dtype=np.uint8)
        self.cells_seen = set()

    def _hash(self, x):
        proj = self.H @ x
        return [int(abs(proj[i] * 1000)) % self.m for i in range(self.k)]

    def step(self, x):
        positions = self._hash(x)
        cell_id = sum(1 << i for i, p in enumerate(positions) if p > self.m // 2)
        self.cells_seen.add(cell_id % (2 ** self.k))
        scores = [sum(self.filters[a][p] for p in positions) for a in range(self.n_actions)]
        action = int(np.argmin(scores))
        for p in positions:
            self.filters[action][p] = 1
        return action


class BloomGraphLSH:
    """Variant B: LSH sign-based hash (preserves local continuity)."""
    def __init__(self, k=K, m=M, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.k = k
        self.m = m
        self.n_actions = n_actions
        self.filters = np.zeros((n_actions, m), dtype=np.uint8)
        self.cells_seen = set()
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)

    def _hash(self, x):
        proj = self.H @ x
        # Each bit = sign of projection. Use pairs of consecutive bits as position.
        signs = (proj > 0).astype(np.int32)
        positions = []
        for i in range(self.k):
            # Use sign bit combo to generate a position in [0, m)
            pos = (signs[i] * (i * 317 + 1031)) % self.m
            positions.append(pos)
        return positions

    def step(self, x):
        proj = self.H @ x
        bits = (proj > 0).astype(np.int64)
        cell_id = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell_id)
        positions = self._hash(x)
        scores = [sum(self.filters[a][p] for p in positions) for a in range(self.n_actions)]
        action = int(np.argmin(scores))
        for p in positions:
            self.filters[action][p] = 1
        return action


def run_variant(arc, game_id, GraphClass, n_seeds, max_steps=MAX_STEPS):
    from arcengine import GameState
    results = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = arc.make(game_id)
        na = len(env.action_space)
        g = GraphClass(k=K, m=M, n_actions=na, seed=seed)
        obs = env.reset()
        ts = go = lvls = 0
        level_step = None
        t0 = time.time()
        while ts < max_steps:
            if obs is None: obs = env.reset(); continue
            if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
            x = centered_enc(avgpool16(obs.frame))
            action_idx = g.step(x)
            action = env.action_space[action_idx % na]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs_before = obs.levels_completed
            obs = env.step(action, data=data)
            ts += 1
            if obs is None: break
            if obs.levels_completed > obs_before:
                lvls = obs.levels_completed
                if level_step is None: level_step = ts
            if time.time() - t0 > TIME_CAP: break
        saturation = g.filters.sum(axis=1) / M
        results.append({
            'seed': seed, 'levels': lvls, 'level_step': level_step,
            'cells': len(g.cells_seen), 'sat_mean': saturation.mean(),
            'elapsed': time.time() - t0
        })
        status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
        print(f"    seed={seed:2d}  {status:12s}  cells={len(g.cells_seen):5d}  "
              f"sat={saturation.mean():.3f}  {results[-1]['elapsed']:.0f}s", flush=True)
    return results


def main():
    import arc_agi
    n_seeds = 10
    print(f"Step 494: Bloom filter family. m={M}, k={K}, {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()

    print(f"\nVariant A: Magnitude hash (NOT locality-preserving)", flush=True)
    results_A = run_variant(arc, ls20.game_id, BloomGraphMagnitude, n_seeds)
    wins_A = sum(1 for r in results_A if r['levels'] > 0)
    avg_sat_A = sum(r['sat_mean'] for r in results_A) / n_seeds
    print(f"  -> {wins_A}/{n_seeds}  avg_sat={avg_sat_A:.3f}", flush=True)

    print(f"\nVariant B: LSH sign hash (locality-preserving)", flush=True)
    results_B = run_variant(arc, ls20.game_id, BloomGraphLSH, n_seeds)
    wins_B = sum(1 for r in results_B if r['levels'] > 0)
    avg_sat_B = sum(r['sat_mean'] for r in results_B) / n_seeds
    print(f"  -> {wins_B}/{n_seeds}  avg_sat={avg_sat_B:.3f}", flush=True)

    print(f"\nSUMMARY:", flush=True)
    print(f"  Magnitude hash (A): {wins_A}/{n_seeds}", flush=True)
    print(f"  LSH sign hash (B):  {wins_B}/{n_seeds}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if wins_B > wins_A and wins_B > 0:
        print(f"  U20 CONFIRMED: B ({wins_B}/10) > A ({wins_A}/10). Local continuity is load-bearing.", flush=True)
        print(f"  Bloom+LSH navigates without a graph. Family 9 has signal.", flush=True)
    elif wins_A == 0 and wins_B == 0:
        print(f"  0/10 BOTH VARIANTS. Graph mechanism is required, not just action selection.", flush=True)
        print(f"  Bloom filters cannot navigate even with local continuity (B).", flush=True)
    elif wins_A == wins_B:
        print(f"  A=B={wins_A}/10. Local continuity doesn't matter for bloom filters.", flush=True)
    else:
        print(f"  A={wins_A}/10  B={wins_B}/10. Unexpected pattern.", flush=True)
    print(f"\nTotal elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
