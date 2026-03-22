#!/usr/bin/env python3
"""
Step 481 — Prediction-error action selection on LSH k=12.
action_score[a] = 1 - max_dest_count[C][a] / total_count[C][a]
Pick highest score (least confident prediction = most surprising).
Baseline: argmin = 6/10 at 50K (Step 459).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class PredErrorGraph:
    def __init__(self, k=K, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}   # (cell, action) -> {dest: count}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, x):
        cell = self._hash(x)
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1

        # Compute prediction error for each action
        scores = []
        for a in range(self.n_actions):
            key = (cell, a)
            if key not in self.edges or not self.edges[key]:
                scores.append(1.0)  # unseen = max uncertainty
            else:
                dests = self.edges[key]
                total = sum(dests.values())
                max_count = max(dests.values())
                # 1 - concentration = prediction error
                scores.append(1.0 - max_count / total)

        max_score = max(scores)
        candidates = [a for a, s in enumerate(scores) if abs(s - max_score) < 1e-9]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = PredErrorGraph(k=K, n_actions=na, seed=seed)
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
        if time.time() - t0 > 280: break
    return {'seed': seed, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen) / (2**K),
            'elapsed': time.time() - t0}


def main():
    import arc_agi
    n_seeds = 10
    print(f"Step 481: Prediction-error action selection. LSH k={K}, {n_seeds} seeds x 50K.", flush=True)
    print(f"score[a] = 1 - max_dest/total. Pick highest (least confident).", flush=True)
    print(f"Baseline (argmin Step 459): 6/10 at 50K.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:22s}  cells={r['unique_cells']:4d}/{2**K}"
              f"  occ={r['occupancy']:.4f}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    avg_occ = sum(r['occupancy'] for r in results) / n_seeds
    print(f"\n{len(wins)}/{n_seeds}  avg_occ={avg_occ:.4f}", flush=True)
    print(f"level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print("\nVERDICT:", flush=True)
    if len(wins) >= 8:
        print(f"  PRED-ERROR BREAKS CEILING: {len(wins)}/{n_seeds}. Model-based > random exploration.", flush=True)
    elif len(wins) >= 6:
        print(f"  PRED-ERROR COMPARABLE: {len(wins)}/{n_seeds}. Same as baseline.", flush=True)
    elif len(wins) >= 4:
        print(f"  PRED-ERROR MARGINAL: {len(wins)}/{n_seeds}. Close to baseline.", flush=True)
    else:
        print(f"  PRED-ERROR WEAKER: {len(wins)}/{n_seeds} < baseline 6/10. Model overhead hurts.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
