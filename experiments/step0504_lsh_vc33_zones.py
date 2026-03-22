#!/usr/bin/env python3
"""
Step 504 — VC33 zone discovery + expanded actions. Spec.
Original win: 3-zone encoding + 3 actions won at step 283.
Phase 1: Click each 8x8 grid position from initial state, hash result frame.
Group positions by frame hash → discover zones.
Phase 2: K-means + N_zones-action graph, 30K steps, 3 seeds.
Report Phase 1 zone count BEFORE Phase 2.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 30_000
TIME_CAP = 40
WARMUP = 500
N_CLUSTERS = 50

GRID_POSITIONS = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]  # 64 positions


def frame_hash(frame):
    arr = np.array(frame[0], dtype=np.uint8)
    return arr.tobytes().__hash__()


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def discover_zones(arc, game_id):
    """Phase 1: click each grid position from initial state, group by frame hash."""
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    action6 = action_space[0]  # only action in VC33

    hash_to_positions = {}
    t0 = time.time()

    for i, (cx, cy) in enumerate(GRID_POSITIONS):
        obs = env.reset()
        if obs is None or obs.state == GameState.GAME_OVER:
            continue
        # Take one click at this position
        obs = env.step(action6, data={"x": cx, "y": cy})
        if obs is None or not obs.frame or len(obs.frame) == 0:
            continue
        h = frame_hash(obs.frame)
        hash_to_positions.setdefault(h, []).append(i)

    zones = list(hash_to_positions.values())
    zone_reps = [GRID_POSITIONS[group[0]] for group in zones]  # representative per zone

    print(f"\nPhase 1 — Zone Discovery ({time.time()-t0:.1f}s):", flush=True)
    print(f"  Grid positions tested: {len(GRID_POSITIONS)}", flush=True)
    print(f"  Unique frame hashes: {len(zones)}", flush=True)
    for i, (h, positions) in enumerate(hash_to_positions.items()):
        rep = GRID_POSITIONS[positions[0]]
        print(f"  Zone {i}: {len(positions)} positions, rep={rep}", flush=True)

    return zone_reps, zones


class KMeansGraphZones:
    def __init__(self, n_clusters=N_CLUSTERS, zone_reps=None, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.n_actions = len(zone_reps)
        self.zone_reps = zone_reps
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
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, zone_reps):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = env.action_space[0]
    g = KMeansGraphZones(n_clusters=N_CLUSTERS, zone_reps=zone_reps, warmup=WARMUP)
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
        action_idx = g.step(x)
        cx, cy = g.zone_reps[action_idx]

        obs_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > TIME_CAP: break

    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}  {status:12s}  cells={len(g.cells_seen):3d}/{n_c}  go={go}  {time.time()-t0:.0f}s", flush=True)
    return {'levels': lvls, 'level_step': level_step, 'cells': len(g.cells_seen), 'go': go}


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 504: VC33 zone discovery + graph. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next((g for g in games if 'vc33' in g.game_id.lower()), None)
    if not vc33:
        print("SKIP — VC33 not found"); return

    t0 = time.time()

    # Phase 1
    zone_reps, zones = discover_zones(arc, vc33.game_id)
    n_zones = len(zone_reps)
    print(f"\n  -> {n_zones} zones found. Proceeding to Phase 2.", flush=True)

    if n_zones < 2:
        print(f"  Only {n_zones} zone(s). 8x8 grid too coarse. Try finer grid.", flush=True)
        return

    # Phase 2
    print(f"\nPhase 2 — Navigation with {n_zones} zone actions:", flush=True)
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, vc33.game_id, seed=seed, zone_reps=zone_reps)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    print(f"\n{'='*50}", flush=True)
    print(f"STEP 504 SUMMARY: {wins}/{n_seeds}", flush=True)
    print(f"  Zones: {n_zones}  Zone reps: {zone_reps}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins > 0:
        win_steps = sorted(r['level_step'] for r in results if r['level_step'])
        print(f"  VC33 NAVIGATES: {wins}/{n_seeds}! Zone discovery works.", flush=True)
        print(f"  Level steps: {win_steps}", flush=True)
    else:
        print(f"  0/{n_seeds}. Zone discovery found {n_zones} zones but navigation fails.", flush=True)
        if n_zones < 3:
            print(f"  Need finer grid to find original 3 zones.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
