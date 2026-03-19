#!/usr/bin/env python3
"""
Step 501 — FT09 grid-click sweep. Leo's spec (mail 1994).
Step 499: ACTION6 is complex, argmax click = self-loop. 32 states reachable = no-click subgraph.
Test: use RANDOM click from 8x8 grid (64 positions) for ACTION6 instead of argmax.
Q: Do new cells appear beyond 32? Does reward appear?
K-means km32, 50K steps, 3 seeds. Report new cells, reward, click→transition map.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 50_000
TIME_CAP = 45
WARMUP = 500
N_CLUSTERS = 32

# 8x8 grid of click positions (evenly spaced across 64x64 frame)
GRID_POSITIONS = [(x, y) for y in range(4, 64, 8) for x in range(4, 64, 8)]  # 64 positions


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraphGridClick:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=6, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []
        # Click tracking
        self.click_transitions = {}   # (cell, click_pos) -> set of next_cells
        self.click_counts = {}        # click_pos -> count
        self.reward_clicks = []       # (cell, click_pos) when reward occurs

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
            return 0, None  # (action_idx, click_pos)

        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen.add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1

        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_cell = cell
        self.prev_action = action
        return action, cell

    def record_click_transition(self, from_cell, click_pos, to_cell):
        key = (from_cell, click_pos)
        s = self.click_transitions.setdefault(key, set())
        s.add(to_cell)
        self.click_counts[click_pos] = self.click_counts.get(click_pos, 0) + 1


def run_seed(arc, game_id, seed, na):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    g = KMeansGraphGridClick(n_clusters=N_CLUSTERS, n_actions=na, warmup=WARMUP)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    total_reward = 0.0
    reward_events = 0
    prev_cell_for_click = None
    last_click_pos = None
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            obs = env.reset()
            prev_cell_for_click = None
            last_click_pos = None
            continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        x = centered_enc(avgpool16(obs.frame))
        action_idx, current_cell = g.step(x)

        # Record click transition from previous step
        if prev_cell_for_click is not None and last_click_pos is not None and current_cell is not None:
            g.record_click_transition(prev_cell_for_click, last_click_pos, current_cell)

        action = env.action_space[action_idx % na]
        data = {}
        click_pos = None
        if action.is_complex():
            # Use RANDOM click from 8x8 grid
            click_pos = GRID_POSITIONS[int(np.random.randint(len(GRID_POSITIONS)))]
            data = {"x": click_pos[0], "y": click_pos[1]}
        last_click_pos = click_pos if action.is_complex() else None
        prev_cell_for_click = current_cell if action.is_complex() else None

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break

        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
            if click_pos is not None and current_cell is not None:
                g.reward_clicks.append((current_cell, click_pos, ts))

        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"

    # Analyze click→transition diversity
    unique_action6_cells = set()  # cells produced by ACTION6
    non_self_transitions = 0
    for (from_c, click_p), to_cells in g.click_transitions.items():
        unique_action6_cells.update(to_cells)
        for tc in to_cells:
            if tc != from_c:
                non_self_transitions += 1

    print(f"\n  seed={seed}  {status}", flush=True)
    print(f"    cells total: {len(g.cells_seen)}/{n_c}  (32 was baseline with argmax)", flush=True)
    print(f"    new cells from grid-clicks: {len(unique_action6_cells)}", flush=True)
    print(f"    non-self transitions via ACTION6: {non_self_transitions}", flush=True)
    print(f"    go={go}  reward_events={reward_events}  {time.time()-t0:.0f}s", flush=True)
    if g.reward_clicks:
        print(f"    REWARD at: {g.reward_clicks}", flush=True)

    # Sample: which click positions led to NEW states
    click_to_new = {}
    for (from_c, click_p), to_cells in g.click_transitions.items():
        for tc in to_cells:
            if tc != from_c:
                click_to_new[click_p] = click_to_new.get(click_p, 0) + 1
    if click_to_new:
        top_clicks = sorted(click_to_new.items(), key=lambda kv: -kv[1])[:5]
        print(f"    Most productive click positions: {top_clicks}", flush=True)

    return {'levels': lvls, 'cells': len(g.cells_seen), 'n_c': n_c,
            'new_cells': len(unique_action6_cells), 'non_self': non_self_transitions,
            'go': go, 'reward_events': reward_events}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 501: FT09 grid-click sweep. 8x8 grid, {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"ACTION6 uses random 8x8 grid position instead of argmax.", flush=True)
    print(f"Baseline: 32/32 cells, 0 reward with argmax. Grid should unlock new states.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP — FT09 not found"); return

    t0 = time.time()
    env0 = arc.make(ft09.game_id)
    na = len(env0.action_space)
    print(f"FT09 actions: {na}, grid positions: {len(GRID_POSITIONS)}", flush=True)

    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ft09.game_id, seed=seed, na=na)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    avg_cells = sum(r['cells'] for r in results) / n_seeds
    avg_new = sum(r['new_cells'] for r in results) / n_seeds

    print(f"\n{'='*50}", flush=True)
    print(f"STEP 501 SUMMARY: {wins}/{n_seeds}", flush=True)
    print(f"  avg cells: {avg_cells:.0f}  avg new_from_clicks: {avg_new:.0f}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins > 0:
        print(f"  FT09 NAVIGATES: {wins}/{n_seeds}! Grid-click found win.", flush=True)
    elif avg_cells > 32:
        print(f"  NEW STATES FOUND: {avg_cells:.0f} cells (was 32). Click space opens new territory.", flush=True)
        print(f"  Navigation not yet found — need targeted click strategy, not random grid.", flush=True)
    elif avg_cells == 32:
        print(f"  NO NEW STATES: Grid clicks produce same 32 cells. Click position doesn't change state.", flush=True)
        print(f"  I3 hypothesis may be wrong for state expansion. Win trigger is elsewhere.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
