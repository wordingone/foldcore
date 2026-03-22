#!/usr/bin/env python3
"""
Step 522 -- K-means graph cross-game transfer.
Centroids fitted on LS20 warmup, FROZEN for all phases.

Warmup: 1000 LS20 steps (random) -> fit k-means n=300
Phase 1: LS20, 50K steps, 3 seeds
Phase 2: FT09, 69-action expansion, 50K steps, 3 seeds
Phase 3: VC33, 3-zone expansion, 30K steps, 3 seeds

Key question: does LS20 centroid geometry transfer to FT09/VC33?
Kill: FT09 0/3 -> cross-game transfer is codebook-specific.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

N_CLUSTERS = 300
WARMUP_STEPS = 1000
N_ARC_ACTIONS_LS20 = 4
N_FT09_ACTIONS = 69
VC33_GRID = [(gx*4+2, gy*4+2) for gy in range(16) for gx in range(16)]


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class FrozenKMeansGraph:
    """K-means graph with centroids frozen after warmup. Separate edges per game."""

    def __init__(self, n_clusters=N_CLUSTERS):
        self.n_clusters = n_clusters
        self.centroids = None
        self._buf = []
        self.edges = {}      # mode -> {(cell, action): {next_cell: count}}
        self.prev_cell = None
        self.prev_action = None
        self._mode = None

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def fit(self, observations):
        from sklearn.cluster import KMeans
        X = np.array(observations, dtype=np.float32)
        km = KMeans(n_clusters=min(self.n_clusters, len(X)), random_state=42, n_init=3, max_iter=100)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        print(f"  k-means fit: {len(self.centroids)} centroids on {len(X)} obs", flush=True)

    def step(self, x, n_actions):
        if self.centroids is None:
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        edges = self.edges.setdefault(self._mode, {})
        if self.prev_cell is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(edges.get((cell, a), {}).values()) for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action

    def cells_used(self):
        edges = self.edges.get(self._mode, {})
        used = set()
        for (c, a) in edges:
            used.add(c)
        return len(used)


def warmup_ls20(arc, game_id, n_steps=WARMUP_STEPS):
    """Collect observations from LS20 with random actions."""
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    buf = []
    t0 = time.time()
    for _ in range(n_steps):
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: obs = env.reset(); continue
        if obs.state == GameState.WIN: obs = env.reset(); continue
        if not obs.frame: obs = env.reset(); continue
        buf.append(encode_arc(obs.frame))
        a = np.random.randint(len(action_space))
        obs = env.step(action_space[a])
        if obs is None: obs = env.reset()
    print(f"  Warmup: {len(buf)} obs in {time.time()-t0:.0f}s", flush=True)
    return buf


def ft09_action(action_id, action_space):
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        return action_space[5], {"x": gx*8+4, "y": gy*8+4}
    return action_space[action_id-64], {}


def discover_vc33_zones(arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action6 = env.action_space[0]
    hash_to_positions = {}
    for i, (cx, cy) in enumerate(VC33_GRID):
        obs = env.reset()
        if obs is None or obs.state == GameState.GAME_OVER: continue
        obs = env.step(action6, data={"x": cx, "y": cy})
        if obs is None or not obs.frame: continue
        h = np.array(obs.frame[0], dtype=np.uint8).tobytes().__hash__()
        hash_to_positions.setdefault(h, []).append(i)
    zones = list(hash_to_positions.values())
    zone_reps = [VC33_GRID[g[0]] for g in zones]
    print(f"  VC33 zones: {len(zones)} ({[len(z) for z in zones]})", flush=True)
    return zone_reps


def run_game_ls20(g, arc, game_id, max_steps=50000, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS_LS20)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    return lvls, level_step, ts, go, time.time()-t0


def run_game_ft09(g, arc, game_id, max_steps=50000, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('ft09')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_FT09_ACTIONS)
        action, data = ft09_action(a, action_space)
        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    return lvls, level_step, ts, go, time.time()-t0


def run_game_vc33(g, arc, game_id, zone_reps, max_steps=30000, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('vc33')
    n_zones = len(zone_reps)
    env = arc.make(game_id)
    action6 = env.action_space[0]
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, n_zones)
        cx, cy = zone_reps[a]
        obs_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    return lvls, level_step, ts, go, time.time()-t0


def run_phase(name, run_fn, g, seeds, **kwargs):
    results = []
    for s in seeds:
        r = run_fn(g, **kwargs, seed=s)
        lvls, level_step, ts, go, elapsed = r
        status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
        print(f"  seed={s}: {status}  steps={ts}  go={go}  cells={g.cells_used()}/{len(g.centroids)}  {elapsed:.0f}s", flush=True)
        results.append(lvls > 0)
        g.edges[g._mode] = {}  # reset edges between seeds but keep centroids
    wins = sum(results)
    print(f"  {name}: {wins}/{len(seeds)} WIN", flush=True)
    return wins


def main():
    t_total = time.time()
    print("Step 522: K-means cross-game transfer (LS20 -> FT09 -> VC33)", flush=True)
    print(f"n_clusters={N_CLUSTERS}  warmup_steps={WARMUP_STEPS}", flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    print("\nVC33 zone discovery...", flush=True)
    zone_reps = discover_vc33_zones(arc, vc33.game_id)

    print("\nWarmup: collecting LS20 observations...", flush=True)
    np.random.seed(0)
    obs_buf = warmup_ls20(arc, ls20.game_id, WARMUP_STEPS)

    print("\nFitting k-means on warmup observations...", flush=True)
    g = FrozenKMeansGraph(n_clusters=N_CLUSTERS)
    g.fit(obs_buf)

    seeds = [0, 1, 2]

    print(f"\n--- Phase 1: LS20 (50K steps, {len(seeds)} seeds) ---", flush=True)
    ls20_wins = run_phase("LS20", run_game_ls20, g, seeds,
                          arc=arc, game_id=ls20.game_id, max_steps=50000)

    print(f"\n--- Phase 2: FT09 (50K steps, {len(seeds)} seeds) ---", flush=True)
    ft09_wins = run_phase("FT09", run_game_ft09, g, seeds,
                          arc=arc, game_id=ft09.game_id, max_steps=50000)

    print(f"\n--- Phase 3: VC33 (30K steps, {len(seeds)} seeds) ---", flush=True)
    vc33_wins = run_phase("VC33", run_game_vc33, g, seeds,
                          arc=arc, game_id=vc33.game_id, zone_reps=zone_reps, max_steps=30000)

    print(f"\n{'='*60}", flush=True)
    print("STEP 522 SUMMARY", flush=True)
    print(f"  Centroids: {N_CLUSTERS} (frozen from LS20 warmup)", flush=True)
    print(f"  LS20: {ls20_wins}/{len(seeds)}  (predicted 3/3)", flush=True)
    print(f"  FT09: {ft09_wins}/{len(seeds)}  (predicted 2/3)", flush=True)
    print(f"  VC33: {vc33_wins}/{len(seeds)}  (predicted 1/3)", flush=True)

    print(f"\nVERDICT:", flush=True)
    if ft09_wins >= 1:
        print(f"  FT09 NAVIGATES with LS20 centroids ({ft09_wins}/{len(seeds)}).", flush=True)
        print(f"  Cross-game transfer is about shared visual geometry, not mechanism.", flush=True)
        print(f"  Universal finding: LS20/FT09 share 64x64 game frame statistics.", flush=True)
    else:
        print(f"  FT09 FAILS with frozen LS20 centroids (0/{len(seeds)}).", flush=True)
        print(f"  KILL: cross-game transfer requires codebook attract update.", flush=True)
        print(f"  The codebook online centroid adjustment was load-bearing.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
