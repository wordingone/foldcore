#!/usr/bin/env python3
"""
Step 449b — CA-Graph reliability: 10 seeds × 10K steps.
Same config as 449: Rule 110, T=10, 256 cells.
449 showed 1/3 (seed 2, Level 1 at step 4081). Characterize reliability.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

RULE110 = np.array([(110 >> i) & 1 for i in range(8)], dtype=np.uint8)


def ca_step(state):
    left = np.roll(state, 1)
    right = np.roll(state, -1)
    return RULE110[(left * 4 + state * 2 + right)]


def obs_to_cell(pooled, t_steps=10):
    state = (pooled > np.median(pooled)).astype(np.uint8)
    for _ in range(t_steps):
        state = ca_step(state)
    return hash(state.tobytes())


class CAGraph:
    def __init__(self, n_actions=4, ca_steps=10):
        self.n_actions = n_actions
        self.ca_steps = ca_steps
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0

    def step(self, obs):
        cell_id = obs_to_cell(obs, self.ca_steps)
        self.cells.add(cell_id)
        self.step_count += 1

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][cell_id] = self.edges[key].get(cell_id, 0) + 1

        visit_counts = [
            sum(self.edges.get((cell_id, a), {}).values())
            for a in range(self.n_actions)
        ]
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
    g = CAGraph(n_actions=4, ca_steps=10)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)

        action_counts[action_idx % na] += 1
        action = env.action_space[action_idx % na]
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts

        if time.time() - t0 > 280:
            break

    elapsed = time.time() - t0
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells) / max(g.step_count, 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells), 'ratio': ratio,
        'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print("Step 449b: CA-Graph reliability — 10 seeds × 10K steps", flush=True)
    print("Rule 110, T=10. 449 showed 1/3 (Level 1 at 4081).", flush=True)
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    for seed in range(10):
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:26s}  unique={r['unique_cells']:4d}"
              f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s",
              flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 449b FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    level_steps = sorted([r['level_step'] for r in wins])
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/10", flush=True)
    print(f"Steps-to-level: {level_steps}", flush=True)
    if level_steps:
        print(f"Min: {min(level_steps)}  Max: {max(level_steps)}  Median: {level_steps[len(level_steps)//2]}",
              flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}  Avg ratio: {avg_ratio:.3f}", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)
    print(f"Codebook 434b:   6/10 at 50K  median~19K", flush=True)
    print(f"Cosine graph:    3/10 at 50K  clustered at 25.7K", flush=True)
    print(f"CA-graph 449b:   {len(wins)}/10 at 10K", flush=True)


if __name__ == '__main__':
    main()
