#!/usr/bin/env python3
"""
Step 502 — VC33 grid-click sweep. Spec.
Step 500: VC33 = 1 action (ACTION6 complex), 50-state cycle, dies every 50 steps.
Test: use 8x8 grid of ACTION6 click positions.
Strategy: each cycle (~50 steps), use ONE click position. Rotate through all 64.
Also: random grid click baseline (pure random each step).
Q: Does any (cycle_state, click_position) produce reward?
30K steps, 3 seeds.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 30_000
TIME_CAP = 40
WARMUP = 500
N_CLUSTERS = 50  # 50 known VC33 states

# 8x8 grid of click positions
GRID_POSITIONS = [(x, y) for y in range(4, 64, 8) for x in range(4, 64, 8)]  # 64 positions


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansVC33:
    def __init__(self, n_clusters=N_CLUSTERS, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.warmup = warmup
        self.centroids = None
        self.cells_seen = set()
        self._buf = []
        self.step_count = 0
        self.click_to_cells = {}   # click_pos -> set of cells seen after that click

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

    def observe(self, x):
        self.step_count += 1
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return None
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)
        return cell


def run_seed_random(arc, game_id, seed, na):
    """Random click from grid each step."""
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = KMeansVC33(n_clusters=N_CLUSTERS, warmup=WARMUP)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    prev_click = None
    t0 = time.time()
    click_outcome = {}  # click_pos -> {'wins': 0, 'total': 0}

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        x = centered_enc(avgpool16(obs.frame))
        cell = g.observe(x)

        action = env.action_space[0]
        click_pos = GRID_POSITIONS[int(np.random.randint(len(GRID_POSITIONS)))]
        data = {"x": click_pos[0], "y": click_pos[1]}
        co = click_outcome.setdefault(click_pos, {'wins': 0, 'total': 0})
        co['total'] += 1
        if cell is not None:
            g.click_to_cells.setdefault(click_pos, set()).add(cell)

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
            co['wins'] += 1
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"    seed={seed}  {status}  cells={len(g.cells_seen)}/{n_c}  go={go}  {time.time()-t0:.0f}s", flush=True)
    return {'levels': lvls, 'level_step': level_step, 'cells': len(g.cells_seen),
            'go': go, 'click_outcome': click_outcome}


def run_seed_cycling(arc, game_id, seed, na):
    """Cycle through click positions: one position per full death-cycle."""
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = KMeansVC33(n_clusters=N_CLUSTERS, warmup=WARMUP)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    click_idx = 0  # which grid position we're testing this cycle
    cycle_outcomes = {}  # click_pos -> {'wins': 0, 'cycles': 0}
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            co = cycle_outcomes.setdefault(GRID_POSITIONS[click_idx % len(GRID_POSITIONS)], {'wins': 0, 'cycles': 0})
            co['cycles'] += 1
            click_idx += 1  # next position for next cycle
            obs = env.reset()
            continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        x = centered_enc(avgpool16(obs.frame))
        cell = g.observe(x)

        action = env.action_space[0]
        pos = GRID_POSITIONS[click_idx % len(GRID_POSITIONS)]
        data = {"x": pos[0], "y": pos[1]}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
            co = cycle_outcomes.setdefault(pos, {'wins': 0, 'cycles': 0})
            co['wins'] += 1
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"    seed={seed}  {status}  cells={len(g.cells_seen)}/{n_c}  go={go}  cycles_per_pos={go//max(len(GRID_POSITIONS),1):.1f}  {time.time()-t0:.0f}s", flush=True)
    # Show any winning positions
    wins_by_pos = {p: v for p, v in cycle_outcomes.items() if v['wins'] > 0}
    if wins_by_pos:
        print(f"    WIN clicks: {wins_by_pos}", flush=True)
    return {'levels': lvls, 'level_step': level_step, 'cells': len(g.cells_seen),
            'go': go, 'cycle_outcomes': cycle_outcomes}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 502: VC33 grid-click sweep. 8x8 grid, {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"Two strategies: random grid (each step) + cycling (1 position per death-cycle).", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next((g for g in games if 'vc33' in g.game_id.lower()), None)
    if not vc33:
        print("SKIP — VC33 not found"); return

    t0 = time.time()
    env0 = arc.make(vc33.game_id)
    na = len(env0.action_space)
    print(f"VC33 actions: {na}. Grid: {len(GRID_POSITIONS)} positions.", flush=True)

    print(f"\n--- Strategy A: Random grid click each step ---", flush=True)
    results_rand = []
    for seed in range(n_seeds):
        r = run_seed_random(arc, vc33.game_id, seed=seed, na=na)
        results_rand.append(r)

    print(f"\n--- Strategy B: Cycle through positions (1 per death-cycle) ---", flush=True)
    results_cyc = []
    for seed in range(n_seeds):
        r = run_seed_cycling(arc, vc33.game_id, seed=seed, na=na)
        results_cyc.append(r)

    wins_rand = sum(1 for r in results_rand if r['levels'] > 0)
    wins_cyc = sum(1 for r in results_cyc if r['levels'] > 0)

    print(f"\n{'='*50}", flush=True)
    print(f"STEP 502 SUMMARY:", flush=True)
    print(f"  Random grid: {wins_rand}/{n_seeds}", flush=True)
    print(f"  Cycling:     {wins_cyc}/{n_seeds}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins_rand > 0 or wins_cyc > 0:
        print(f"  VC33 NAVIGATES! Click position is the key.", flush=True)
        print(f"  Random: {wins_rand}/{n_seeds}  Cycling: {wins_cyc}/{n_seeds}", flush=True)
    else:
        print(f"  0/6 total. VC33 doesn't respond to grid click positions.", flush=True)
        print(f"  Click position may need higher precision or different targeting strategy.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
