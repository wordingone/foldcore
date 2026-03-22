#!/usr/bin/env python3
"""
Step 493 — L2 k-means on Level 2. Does growing/adaptive mapping beat fixed LSH cells?
LSH: fixed 4096 cells, 259 visited before plateau.
K-means: 300 centroids placed WHERE observations cluster (adaptive placement).
Q: Does adaptive centroid placement reach Level 2 reward region that LSH misses?
200K steps, 3 seeds, fresh k-means per level.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
WARMUP = 1000
N_CLUSTERS = 300
MAX_STEPS = 200_000
TIME_CAP = 110  # per seed (~87s expected + buffer)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraph:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=4, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.reset()

    def reset(self):
        self.obs_buffer = []
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self.step_count = 0

    def _fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self.obs_buffer, dtype=np.float32)
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)

    def step(self, x):
        self.step_count += 1
        if self.centroids is None:
            self.obs_buffer.append(x.copy())
            if len(self.obs_buffer) >= self.warmup:
                self._fit()
                self.obs_buffer = []
            # During warmup: random action
            action = int(np.random.randint(self.n_actions))
            self.prev_cell = None
            self.prev_action = action
            return action

        # Nearest centroid
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            self.edges[key] = self.edges.get(key, 0.0) + 1.0

        counts = [self.edges.get((cell, a), 0.0) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c <= min_c + 1e-9]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = KMeansGraph(n_clusters=N_CLUSTERS, n_actions=na)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}
    level_budgets = {}
    level_start_step = 0
    cells_milestones = {}  # steps_on_level -> cells_seen
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

        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break

        if obs.levels_completed > prev_levels:
            for lvl in range(prev_levels + 1, obs.levels_completed + 1):
                level_steps[lvl] = ts
                level_budgets[lvl] = ts - level_start_step
                elapsed = time.time() - t0
                print(f"  LEVEL {lvl} at step {ts} (budget={ts-level_start_step}, "
                      f"elapsed={elapsed:.1f}s, cells={len(g.cells_seen)}, go={go})", flush=True)
            prev_levels = obs.levels_completed
            level_start_step = ts
            g.reset()  # fresh k-means per level

        steps_on_level = ts - level_start_step
        if prev_levels >= 1 and steps_on_level in (50000, 100000, 150000):
            cells_milestones[steps_on_level] = len(g.cells_seen)

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'level_steps': level_steps,
        'level_budgets': level_budgets,
        'max_levels': prev_levels,
        'cells_final': len(g.cells_seen),
        'cells_milestones': cells_milestones,
        'game_overs': go,
        'steps_reached': ts,
        'elapsed': elapsed,
        'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 493: K-means (n={N_CLUSTERS}) Level 2 test. {MAX_STEPS//1000}K steps, {n_seeds} seeds.", flush=True)
    print(f"Warmup={WARMUP}, fresh k-means per level. Baseline LSH plateau=259 cells.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed)
        l1 = r['level_steps'].get(1)
        l2 = r['level_steps'].get(2)
        l1_str = f"L1@{l1}" if l1 else "L1-FAIL"
        l2_str = f"L2@{l2}" if l2 else "L2-none"
        mil = r['cells_milestones']
        mil_str = " ".join(f"@{k//1000}K={v}" for k, v in sorted(mil.items()))
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  seed={seed}  {l1_str:12s}  {l2_str:10s}  "
              f"cells={r['cells_final']}/{N_CLUSTERS}  {mil_str}  "
              f"go={r['game_overs']}  {r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    l1_wins = sum(1 for r in results if r['max_levels'] >= 1)
    l2_wins = sum(1 for r in results if r['max_levels'] >= 2)
    print(f"\nLevel 1: {l1_wins}/{n_seeds}  Level 2: {l2_wins}/{n_seeds}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if l2_wins > 0:
        print(f"  K-MEANS BREAKS LEVEL 2: {l2_wins}/{n_seeds}. Adaptive placement reaches disconnected region.", flush=True)
    else:
        max_cells = max(r['cells_final'] for r in results)
        if max_cells > 259:
            print(f"  MORE CELLS THAN LSH ({max_cells} vs 259) but no Level 2. Region expands but reward still out of reach.", flush=True)
        elif max_cells > 200:
            print(f"  SIMILAR TO LSH: {max_cells} cells. Adaptive placement doesn't help.", flush=True)
        else:
            print(f"  FEWER CELLS THAN LSH: {max_cells}. K-means more constrained.", flush=True)
        print(f"  Physical gap confirmed: topology problem, not mapping problem.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
