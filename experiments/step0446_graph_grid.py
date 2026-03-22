#!/usr/bin/env python3
"""
Step 446 — Grid Graph Substrate on LS20.
No cosine, no attract, no prototypes. Fixed spatial cells + edge navigation.

Architecture:
- Random projection: 256D -> 8D (fixed frozen matrix)
- Quantization: 4 bins per dim (percentile-based from 500 warmup obs)
- Cell ID = tuple of 8 bin indices (up to 4^8 = 65536 possible cells)
- Edges: transition counts (cell, action) -> {next_cell: count}
- Action: least-visited edge from current cell

Ban checklist:
1. Cosine/attract? NO — quantization + bin lookup
2. LVQ? NO — fixed grid, no prototypes
3. Codebook + X? NO — partition vs prototype, fundamentally different
4. Shared spatial engine? NO — no match->update->grow cycle

Parameters: proj_dim=8, bins_per_dim=4, warmup=500, n_actions=4
Benchmark: LS20, 10K steps, 3 seeds.
Kill: dom >50% OR unique_cells <50 at 10K.
"""

import time, logging
import numpy as np
import torch

logging.getLogger().setLevel(logging.WARNING)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class GridGraph:
    def __init__(self, proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256):
        self.proj_dim = proj_dim
        self.bins_per_dim = bins_per_dim
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Fixed random projection (frozen) — NO cosine, no normalize
        rng = np.random.RandomState(42)
        self.P = rng.randn(proj_dim, obs_dim).astype(np.float32)
        self.P /= np.linalg.norm(self.P, axis=1, keepdims=True)

        # Bin edges computed from warmup
        self.bin_edges = None  # shape: (proj_dim, bins_per_dim - 1)
        self.warmup_buf = []

        # Graph: edges only, no node positions
        self.edges = {}   # (cell_id, action) -> {next_cell: count}
        self.cells = set()

        self.prev_cell = None
        self.prev_action = None

    def project(self, obs):
        """obs: flat float32 (obs_dim,). Returns (proj_dim,)."""
        return self.P @ obs

    def compute_bin_edges(self):
        buf = np.stack(self.warmup_buf)  # (n, proj_dim)
        edges = []
        for d in range(self.proj_dim):
            pcts = [100.0 * (i + 1) / self.bins_per_dim for i in range(self.bins_per_dim - 1)]
            edges.append(np.percentile(buf[:, d], pcts))
        self.bin_edges = np.array(edges)  # (proj_dim, bins_per_dim-1)

    def quantize(self, proj):
        """proj: (proj_dim,). Returns tuple of bin indices."""
        cell = []
        for d in range(self.proj_dim):
            idx = int(np.searchsorted(self.bin_edges[d], proj[d]))
            cell.append(idx)
        return tuple(cell)

    def step(self, obs):
        """obs: flat float32. Returns action index."""
        proj = self.project(obs)

        # Warmup: collect observations, use random actions
        if self.bin_edges is None:
            self.warmup_buf.append(proj)
            if len(self.warmup_buf) >= 500:
                self.compute_bin_edges()
            return int(np.random.randint(self.n_actions))

        cell_id = self.quantize(proj)
        self.cells.add(cell_id)

        # Record transition from previous cell
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][cell_id] = self.edges[key].get(cell_id, 0) + 1

        # Least-visited action from current cell
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


def run_structural_test():
    """R1-R6 structural check. Target: <30 seconds."""
    print("Structural test...", flush=True)
    t0 = time.time()

    g = GridGraph(proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256)
    rng = np.random.RandomState(0)

    # Feed warmup + navigation steps
    actions = []
    for i in range(600):
        obs = rng.randn(256).astype(np.float32)
        action = g.step(obs)

    assert g.bin_edges is not None, "R1: bin_edges not computed after 600 steps"
    assert g.bin_edges.shape == (8, 3), f"R1: bin_edges shape {g.bin_edges.shape}"

    for i in range(1000):
        obs = rng.randn(256).astype(np.float32)
        action = g.step(obs)
        actions.append(action)

    # R2: Action variety
    unique_a = len(set(actions))
    assert unique_a == 4, f"R2: Only {unique_a}/4 actions used"

    # R3: No action dominance
    from collections import Counter
    counts = Counter(actions)
    dom = max(counts.values()) / len(actions)
    assert dom < 0.5, f"R3: dom={dom:.0%} > 50%"

    # R4: Cells created
    assert len(g.cells) > 0, "R4: No cells created"

    # R5: Edges created
    assert len(g.edges) > 0, "R5: No edges created"

    # R6: No codebook DNA
    assert not hasattr(g, 'nodes'), "R6: .nodes attribute found (codebook DNA)"

    elapsed = time.time() - t0
    print(f"  PASS: cells={len(g.cells)}  edges={len(g.edges)}  dom={dom:.0%}  {elapsed:.1f}s",
          flush=True)
    return True


def run_seed(arc, game_id, seed, max_steps=10000):
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

        # Track unique cells (only after warmup)
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
    print("Step 446: Grid Graph Substrate on LS20 (no cosine, no attract)", flush=True)
    print(f"Device: {DEVICE}  proj_dim=8  bins_per_dim=4  warmup=500  10K steps  3 seeds",
          flush=True)
    print("Kill: dom >50% OR unique_cells <50 at 10K", flush=True)
    print(flush=True)

    if not run_structural_test():
        print("STRUCTURAL FAIL — stopping.", flush=True)
        return
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found")
        return

    t_total = time.time()
    results = []
    for seed in [0, 1, 2]:
        print(f"--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:26s}  unique_cells={r['unique_cells']:4d}"
              f"  cells={r['cells']:4d}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        if r['dom'] > 50 or r['unique_cells'] < 50:
            print(f"  KILL CRITERION HIT: dom={r['dom']:.0f}%  unique_cells={r['unique_cells']}",
                  flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 446 FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {[r['level_step'] for r in wins]}", flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}", flush=True)
    print(f"Avg dom: {avg_dom:.0f}%", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    if wins:
        print("NAVIGATES: Edge structure sufficient without cosine nodes!", flush=True)
    elif avg_cells < 50 or avg_dom > 50:
        print("ARCHITECTURE FAILURE: dynamics dead (grid too coarse/fine or action collapse).",
              flush=True)
    else:
        print("NO NAVIGATION at 10K. Dynamics healthy — request 30K if worth it.", flush=True)

    print(flush=True)
    print(f"Graph 442b ref: 1/3 at 30K (Level 1 at 25738), dom=25%, unique=4461-5516",
          flush=True)


if __name__ == '__main__':
    main()
