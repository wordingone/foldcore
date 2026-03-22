#!/usr/bin/env python3
"""
Step 446b — Grid Graph at 30K steps, 3 seeds.
APPROVED by Avir. Sharpest test: does edge structure navigate without cosine?

If navigates ~25-30K: EDGE STRUCTURE IS THE MECHANISM. Cosine was incidental.
If 0/3 with healthy dynamics: cosine geometry load-bearing for graph navigation.
If 0/3 with dead dynamics: grid resolution problem, try LSH next.

Same GridGraph as 446: proj_dim=8, bins_per_dim=4, warmup=500.
No cosine, no attract, no prototypes, no F.normalize.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


class GridGraph:
    def __init__(self, proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256):
        self.proj_dim = proj_dim
        self.bins_per_dim = bins_per_dim
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        rng = np.random.RandomState(42)
        self.P = rng.randn(proj_dim, obs_dim).astype(np.float32)
        self.P /= np.linalg.norm(self.P, axis=1, keepdims=True)

        self.bin_edges = None
        self.warmup_buf = []
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None

    def project(self, obs):
        return self.P @ obs

    def compute_bin_edges(self):
        buf = np.stack(self.warmup_buf)
        edges = []
        for d in range(self.proj_dim):
            pcts = [100.0 * (i + 1) / self.bins_per_dim for i in range(self.bins_per_dim - 1)]
            edges.append(np.percentile(buf[:, d], pcts))
        self.bin_edges = np.array(edges)

    def quantize(self, proj):
        cell = []
        for d in range(self.proj_dim):
            idx = int(np.searchsorted(self.bin_edges[d], proj[d]))
            cell.append(idx)
        return tuple(cell)

    def step(self, obs):
        proj = self.project(obs)
        if self.bin_edges is None:
            self.warmup_buf.append(proj)
            if len(self.warmup_buf) >= 500:
                self.compute_bin_edges()
            return int(np.random.randint(self.n_actions))

        cell_id = self.quantize(proj)
        self.cells.add(cell_id)

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


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = GridGraph(proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique_cells = set()
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

        if g.bin_edges is not None:
            unique_cells.add(g.quantize(g.project(pooled)))

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
            print(f"  LEVEL {lvls} at step {ts}  unique_cells={len(unique_cells)}"
                  f"  cells={len(g.cells)}  edges={len(g.edges)}", flush=True)

        if time.time() - t0 > 280:
            break

    elapsed = time.time() - t0
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(unique_cells), 'cells': len(g.cells),
        'edges': len(g.edges), 'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print("Step 446b: Grid Graph 30K steps, 3 seeds (no cosine, no attract)", flush=True)
    print("Hypothesis: edge structure navigates without cosine geometry.", flush=True)
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    for seed in [0, 1, 2]:
        print(f"--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=30000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:26s}  unique_cells={r['unique_cells']:4d}"
              f"  cells={r['cells']:4d}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 446b FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}  (graph 442b: 4461-5516 unique)", flush=True)
    print(f"Avg dom: {avg_dom:.0f}%  (graph 442b: 25%)", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    if wins:
        print("EDGE STRUCTURE IS THE MECHANISM. Cosine was incidental.", flush=True)
        print("Graph family validated independently of codebook DNA.", flush=True)
    elif avg_dom <= 50 and avg_cells >= 50:
        print("0/3 with healthy dynamics.", flush=True)
        print("Cosine geometry may be load-bearing for graph navigation.", flush=True)
        print("OR: grid resolution mismatch — try LSH (Step 447).", flush=True)
    else:
        print("0/3 with dead dynamics. Grid resolution problem. Try LSH.", flush=True)

    print(flush=True)
    print(f"Graph 442b ref: 1/3 at 30K (Level 1 at 25738), dom=25%, unique=4461-5516",
          flush=True)


if __name__ == '__main__':
    main()
