#!/usr/bin/env python3
"""
Step 448 — Reservoir-Graph Hybrid on LS20.
Hypothesis: reservoir temporal state prevents edge accumulation.
Every step produces different h (history-dependent), sign(h) → unique cell.
If unique_cells/steps > 0.8 → temporal inconsistency confirmed.

Architecture:
- Reservoir: W (256x256, sparse 10%), h = tanh(W@h + U@x), rho(W)=0.9
- Quantization: sign(h) -> binary vector -> hash to cell ID
- Graph: edges (cell, action) -> {next_cell: count}, action = least-visited

Ban checklist:
1. Cosine/attract? NO — recurrent dynamics + binary hash
2. LVQ? NO — ESN with graph readout
3. Codebook + X? NO — reservoir fundamentally different
4. Shared spatial engine? NO — no match->update->grow

Parameters: d=256, sparse=10%, rho=0.9, input=16x16 avgpool (256D)
LS20, 10K steps, 3 seeds.
Kill: unique_cells/steps > 0.8 -> temporal inconsistency.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


class ReservoirGraph:
    def __init__(self, d=256, sparsity=0.1, rho=0.9, input_dim=256, n_actions=4,
                 seed=42):
        rng = np.random.RandomState(seed)

        # Sparse reservoir W
        W = rng.randn(d, d).astype(np.float32)
        mask = (rng.rand(d, d) < sparsity).astype(np.float32)
        W = W * mask
        # Scale to target spectral radius
        eigvals = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigvals))
        if sr > 1e-8:
            W = W * (rho / sr)
        self.W = W

        # Input weights (small scale)
        self.U = (rng.randn(d, input_dim).astype(np.float32) * 0.1)

        # Reservoir state
        self.h = np.zeros(d, dtype=np.float32)

        # Graph
        self.edges = {}
        self.cells = set()
        self.prev_cell = None
        self.prev_action = None
        self.n_actions = n_actions
        self.step_count = 0

    def state_to_cell(self, h):
        """sign(h) -> binary -> bytes -> hash. Unique per distinct sign pattern."""
        bits = (h > 0).view(np.uint8)
        return hash(bits.tobytes())

    def step(self, x):
        self.h = np.tanh(self.W @ self.h + self.U @ x)
        self.step_count += 1

        cell_id = self.state_to_cell(self.h)
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


def run_structural_test():
    """Structural check < 30s."""
    print("Structural test...", flush=True)
    t0 = time.time()

    g = ReservoirGraph(d=256, sparsity=0.1, rho=0.9, input_dim=256, n_actions=4)
    rng = np.random.RandomState(0)

    actions = []
    for i in range(1000):
        x = rng.randn(256).astype(np.float32)
        action = g.step(x)
        actions.append(action)

    # Check spectral radius was applied
    eigvals = np.linalg.eigvals(g.W)
    actual_sr = np.max(np.abs(eigvals))
    assert abs(actual_sr - 0.9) < 0.01, f"R1: spectral radius {actual_sr:.3f} != 0.9"

    # R2: all actions used
    from collections import Counter
    counts = Counter(actions)
    assert len(counts) == 4, f"R2: only {len(counts)}/4 actions used"

    dom = max(counts.values()) / len(actions)
    assert dom < 0.5, f"R3: dom={dom:.0%}"
    assert len(g.cells) > 0, "R4: no cells"
    assert len(g.edges) > 0, "R5: no edges"
    assert not hasattr(g, 'nodes'), "R6: codebook DNA"

    ratio = len(g.cells) / g.step_count
    elapsed = time.time() - t0
    print(f"  PASS: cells={len(g.cells)}  edges={len(g.edges)}  dom={dom:.0%}"
          f"  unique/steps={ratio:.2f}  sr={actual_sr:.3f}  {elapsed:.1f}s", flush=True)
    return True


def run_seed(arc, game_id, seed, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = ReservoirGraph(d=256, sparsity=0.1, rho=0.9, input_dim=256, n_actions=4,
                       seed=seed + 100)
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
        'unique_cells': len(g.cells), 'edges': len(g.edges),
        'steps': g.step_count, 'ratio': ratio,
        'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print("Step 448: Reservoir-Graph Hybrid on LS20", flush=True)
    print("d=256, sparse=10%, rho=0.9. sign(h)->cell. 10K steps, 3 seeds.", flush=True)
    print("Key diagnostic: unique_cells/steps ratio.", flush=True)
    print("Kill: ratio > 0.8 -> temporal inconsistency confirmed.", flush=True)
    print(flush=True)

    if not run_structural_test():
        print("STRUCTURAL FAIL — stopping.", flush=True)
        return
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
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:26s}  unique_cells={r['unique_cells']:5d}"
              f"  ratio={r['ratio']:.3f}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)

        if r['ratio'] > 0.8:
            print(f"  KILL: unique/steps={r['ratio']:.2f} > 0.8 — temporal inconsistency",
                  flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 448 FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    avg_ratio = sum(r['ratio'] for r in results) / len(results)
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {[r['level_step'] for r in wins]}", flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}", flush=True)
    print(f"Avg unique/steps ratio: {avg_ratio:.3f}", flush=True)
    print(f"Avg dom: {avg_dom:.0f}%", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    if avg_ratio > 0.8:
        print(f"TEMPORAL INCONSISTENCY CONFIRMED: ratio={avg_ratio:.2f}", flush=True)
        print("Reservoir state is history-dependent — no cell revisitation.", flush=True)
        print("Reservoir and graph are incompatible partners.", flush=True)
    elif avg_ratio < 0.5:
        print(f"SURPRISING: reservoir creates coarser categories than expected (ratio={avg_ratio:.2f})",
              flush=True)
        print("Edge accumulation may be possible. Check navigation results.", flush=True)
    else:
        print(f"PARTIAL: ratio={avg_ratio:.2f} — some revisitation but limited.", flush=True)

    print(flush=True)
    print("Reference: random grid 446b ratio~1.0 (no warmup, 1 cell/obs)",
          flush=True)
    print("           cosine graph 442b: nodes~1984 from 30K obs (ratio~0.07)", flush=True)


if __name__ == '__main__':
    main()
