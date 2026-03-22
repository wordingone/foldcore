#!/usr/bin/env python3
"""
Step 503 — FT09 69-action k-means graph. Spec.
Hypothesis: codebook won FT09 via 69-action systematic coverage, not codebook learning.
Step 501 had 6 actions (ACTION6 random-click). Graph couldn't distinguish click positions.
Here: 69 DISTINCT actions — each click position is a separate action in the graph.
Action mapping:
  0-63: ACTION6 click at (gx*8+4, gy*8+4) where gy,gx=divmod(id,8)
  64-68: ACTION1-5 (simple, non-complex)
K-means n=300, warmup=1K, argmin over 69 actions per node.
3 seeds, 50K steps. Prediction: win within ~200 steps if 1/3.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 50_000
TIME_CAP = 60
WARMUP = 1000
N_CLUSTERS = 300
N_ACTIONS = 69  # 64 click positions + 5 simple


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def action_id_to_env(action_id, env_action_space):
    """Map 69-action id to (env_action, data)."""
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        cx, cy = gx * 8 + 4, gy * 8 + 4
        return env_action_space[5], {"x": cx, "y": cy}  # ACTION6
    else:
        simple_idx = action_id - 64  # 0-4 -> ACTION1-5
        return env_action_space[simple_idx], {}


class KMeansGraph69:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=N_ACTIONS, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []

    def _fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._buf, dtype=np.float32)
        n = min(self.n_clusters, len(set(x.tobytes() for x in X)), len(X))
        n = max(n, 2)
        km = MiniBatchKMeans(n_clusters=n, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._buf = []

    def step(self, x):
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return int(np.random.randint(self.n_actions))

        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1

        counts = [sum(self.edges.get((cell, a), {}).values())
                  for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    action_space = env.action_space
    g = KMeansGraph69(n_clusters=N_CLUSTERS, n_actions=N_ACTIONS, warmup=WARMUP)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        x = centered_enc(avgpool16(obs.frame))
        action_id = g.step(x)
        action, data = action_id_to_env(action_id, action_space)

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    elapsed = time.time() - t0
    print(f"  seed={seed}  {status:12s}  cells={len(g.cells_seen):3d}/{n_c}"
          f"  go={go}  {elapsed:.0f}s", flush=True)
    return {'levels': lvls, 'level_step': level_step,
            'cells': len(g.cells_seen), 'go': go}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 503: FT09 69-action k-means graph. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"69 actions: 64 click positions (8x8 grid) + 5 simple actions.", flush=True)
    print(f"Prediction: if action coverage is the codebook mechanism, should win ~1/3.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP — FT09 not found"); return

    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ft09.game_id, seed=seed)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    print(f"\n{'='*50}", flush=True)
    print(f"STEP 503 SUMMARY: {wins}/{n_seeds}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins > 0:
        win_steps = sorted(r['level_step'] for r in results if r['level_step'])
        print(f"  ACTION COVERAGE IS THE MECHANISM: {wins}/{n_seeds} wins!", flush=True)
        print(f"  Level steps: {win_steps}", flush=True)
        print(f"  K-means + 69-action argmin replicates codebook win. Learning not required.", flush=True)
    else:
        avg_cells = sum(r['cells'] for r in results) / n_seeds
        print(f"  0/{n_seeds}. avg_cells={avg_cells:.0f}", flush=True)
        print(f"  Action coverage alone insufficient. Codebook attract/spawn was load-bearing.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
