#!/usr/bin/env python3
"""
Step 498 — FT09 budget test with pen10k.
Step 497: pen10k reduces deaths 96% (785→28 per 30K). Agent survives.
Navigation still fails at 30K. Is it budget? Or topology?
Test: pen10k + 200K steps, 5 seeds. Report if/when navigation appears.
Milestone cell counts at 50K/100K/150K/200K.
~23s/seed at 30K → ~153s/seed at 200K. 5 seeds = ~765s. TIME_CAP=200s per seed.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 200_000
TIME_CAP = 200  # per seed
WARMUP = 500
N_CLUSTERS = 32
PENALTY = 10_000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraphDeathAware:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=6, warmup=WARMUP, death_penalty=PENALTY):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.death_penalty = death_penalty
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

    def on_death(self):
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[-1] = d.get(-1, 0) + self.death_penalty
        self.prev_cell = None
        self.prev_action = None


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = KMeansGraphDeathAware(n_clusters=N_CLUSTERS, n_actions=na,
                               warmup=WARMUP, death_penalty=PENALTY)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    milestones = {}
    t0 = time.time()
    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            g.on_death()
            obs = env.reset()
            continue
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
        if ts in (50_000, 100_000, 150_000, 200_000):
            milestones[ts] = (len(g.cells_seen), go)
        if time.time() - t0 > TIME_CAP: break
    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    elapsed = time.time() - t0
    mil_str = " ".join(f"@{k//1000}K=({c}cells,{gv}go)" for k, (c, gv) in sorted(milestones.items()))
    print(f"  seed={seed}  {status:12s}  cells={len(g.cells_seen):2d}/{n_c}"
          f"  go={go}  {elapsed:.0f}s  {mil_str}", flush=True)
    return {'seed': seed, 'levels': lvls, 'level_step': level_step,
            'cells': len(g.cells_seen), 'go': go, 'milestones': milestones, 'elapsed': elapsed}


def main():
    import arc_agi
    n_seeds = 5
    print(f"Step 498: FT09 budget test. pen10k, {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"pen10k: 28 deaths/30K (97% avoidance). Does navigation appear at 200K?", flush=True)
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
    print(f"STEP 498 SUMMARY: {wins}/{n_seeds}", flush=True)
    all_go = [r['go'] for r in results]
    print(f"  go range: {min(all_go)}-{max(all_go)}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins > 0:
        win_steps = [r['level_step'] for r in results if r['level_step']]
        print(f"  FT09 NAVIGATES at 200K: {wins}/{n_seeds}. Budget problem, not topology.", flush=True)
        print(f"  Level steps: {sorted(win_steps)}", flush=True)
    else:
        # Check if cells plateau
        final_cells = [r['cells'] for r in results]
        print(f"  0/{n_seeds} at 200K. cells={sorted(final_cells)}", flush=True)
        if all(c >= 30 for c in final_cells):
            print(f"  TOPOLOGY PROBLEM: All 32 states covered but no win. Win region unreachable from safe actions.", flush=True)
        else:
            print(f"  COVERAGE INCOMPLETE: Not all states reached. Death penalty may block navigation path.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
