#!/usr/bin/env python3
"""
Step 450 — Binarize-only graph (no CA evolution).
Same as 449 but hash(binary_input) directly, skip CA steps.
Tests if CA adds anything beyond binarization.
10K steps, 10 seeds (same as 449b).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)


class BinarizeGraph:
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0

    def step(self, obs):
        state = (obs > np.median(obs)).astype(np.uint8)
        cell_id = hash(state.tobytes())
        self.cells.add(cell_id)
        self.step_count += 1
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][cell_id] = self.edges[key].get(cell_id, 0) + 1
        visit_counts = [sum(self.edges.get((cell_id, a), {}).values()) for a in range(self.n_actions)]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell_id
        self.prev_action = action
        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def run_seed(arc, game_id, seed, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = BinarizeGraph(n_actions=4)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; action_counts = [0]*na; level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)
        action_counts[action_idx % na] += 1
        action = env.action_space[action_idx % na]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > 280: break
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells) / max(g.step_count, 1)
    return {'seed': seed, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells), 'ratio': ratio, 'dom': dom}


def main():
    import arc_agi
    print("Step 450: Binarize-only graph (no CA). 10K steps, 10 seeds.", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time(); results = []
    for seed in range(10):
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:20s}  unique={r['unique_cells']:4d}"
              f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    print(f"\nStep 450: {len(wins)}/10  steps={sorted([r['level_step'] for r in wins])}"
          f"  avg_cells={sum(r['unique_cells'] for r in results)/10:.0f}"
          f"  elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
