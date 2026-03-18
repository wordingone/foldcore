#!/usr/bin/env python3
"""
Step 453 — LSH Graph (fixed random hyperplanes).
Tests property 2 (local continuity) without property 3 (adaptive density).
P(same cell | sim=0.95) = (1 - arccos(0.95)/pi)^k ≈ 0.34 at k=10.

Ban check (all 4 pass):
1. No cosine matching — random hyperplane dot products, not prototype comparison
2. Not LVQ — no prototypes at all
3. Not codebook+X — hash table
4. No spatial engine — fixed random transform, no match/update/grow

k values: 8 (256 cells), 10 (1024 cells), 11 (2048 cells).
3 seeds each. LS20, 10K steps.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    """Center each observation around its own mean."""
    return x - x.mean()


class LSHGraph:
    def __init__(self, k, seed=0):
        rng = np.random.RandomState(seed + 9999)  # fixed, independent of run seed
        self.H = rng.randn(k, 256).astype(np.float32)
        self.k = k
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = 4
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)
        cell = self._hash(x)
        self.cells_seen.add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[cell] = d.get(cell, 0) + 1

        visit_counts = [
            sum(self.edges.get((cell, a), {}).values())
            for a in range(self.n_actions)
        ]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, k, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    g = LSHGraph(k=k, seed=seed)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

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
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts

        if time.time() - t0 > 280: break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells_seen) / max(g.step_count, 1)
    # Edge density: avg outgoing edges per visited cell
    outgoing = sum(len(g.edges.get((c, a), {})) for c in g.cells_seen for a in range(4))
    edge_density = outgoing / max(len(g.cells_seen), 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'edge_density': edge_density,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    print("Step 453: LSH Graph — fixed random hyperplanes. 3 k values x 3 seeds x 10K steps.", flush=True)
    print("Testing property 2 (local continuity) without property 3 (adaptive density).", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    k_values = [8, 10, 11]
    all_results = {}

    for k in k_values:
        max_cells = 2 ** k
        print(f"\n--- k={k} (max {max_cells} cells) ---", flush=True)
        results = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ls20.game_id, seed=seed, k=k, max_steps=10000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{max_cells}"
                  f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%"
                  f"  edge_dens={r['edge_density']:.2f}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)
        wins = [r for r in results if r['levels'] > 0]
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        avg_cells = sum(r['unique_cells'] for r in results) / len(results)
        avg_edge = sum(r['edge_density'] for r in results) / len(results)
        print(f"  -> {len(wins)}/3  ratio={avg_ratio:.3f}  avg_cells={avg_cells:.0f}"
              f"  avg_edge_dens={avg_edge:.2f}", flush=True)
        all_results[k] = {
            'wins': len(wins), 'avg_ratio': avg_ratio,
            'avg_cells': avg_cells, 'avg_edge_density': avg_edge,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }

    print(f"\n{'='*60}", flush=True)
    print("STEP 453 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'k':<5} {'MaxCells':<10} {'Wins':<6} {'Ratio':<8} {'Cells':<8} {'EdgeDens':<10} {'Steps'}", flush=True)
    for k in k_values:
        rr = all_results[k]
        print(f"  k={k:<2}  {2**k:<10}  {rr['wins']}/3   {rr['avg_ratio']:.3f}   "
              f"{rr['avg_cells']:<8.0f} {rr['avg_edge_density']:<10.2f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Random walk:    1/10 at 10K (step 1329)", flush=True)
    print(f"  Grid graph:     0/3  at 30K (ratio=0.062)", flush=True)
    print(f"  kd-tree (452):  0/3  at 30K (ratio=0.024)", flush=True)
    print(f"  Cosine graph:   3/10 at 30K (step ~25K, ratio=0.07)", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)

    print(f"\nVERDICT:", flush=True)
    for k in k_values:
        rr = all_results[k]
        if 0.02 <= rr['avg_ratio'] <= 0.15:
            dyn = "HEALTHY dynamics"
        elif rr['avg_ratio'] < 0.02:
            dyn = f"TOO COARSE (ratio={rr['avg_ratio']:.3f})"
        else:
            dyn = f"TOO SENSITIVE (ratio={rr['avg_ratio']:.3f})"
        nav = f"NAVIGATES ({rr['wins']}/3)" if rr['wins'] > 0 else "no navigation at 10K"
        print(f"  k={k}: {dyn}, {nav}", flush=True)


if __name__ == '__main__':
    main()
