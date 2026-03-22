#!/usr/bin/env python3
"""
Step 456 — Multi-Resolution LSH Graph (time-based cell growth).
Base: Step 453 LSH graph (k=10, centered_enc, graph + edge-count action).
ONE change: every 5K steps, add one hyperplane. Split cells. Inherit edges.

Growth schedule: k=10 at 0, k=11 at 5K, k=12 at 10K, k=13 at 15K...

Ban check (all 4 pass):
1. No cosine matching — random hyperplanes, not prototype comparison
2. Not LVQ — no prototypes
3. Not codebook+X — hash table
4. No spatial engine — fixed random transform + time trigger, no match/update
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    return x - x.mean()


class MultiResLSHGraph:
    def __init__(self, k_init=10, grow_every=5000, seed=0):
        self.rng = np.random.RandomState(seed + 9999)
        self.k = k_init
        self.H = self.rng.randn(k_init, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k_init)], dtype=np.int64)
        self.grow_every = grow_every
        self.n_actions = 4
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()
        self.growth_log = []  # (step, new_k, occupied_cells, total_edges)

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def _grow(self, x):
        """Add one hyperplane. Split all cells. Inherit edges to both children."""
        old_k = self.k
        new_h = self.rng.randn(1, 256).astype(np.float32)
        self.H = np.vstack([self.H, new_h])
        self.k = old_k + 1
        self.powers = np.array([1 << i for i in range(self.k)], dtype=np.int64)

        # Split edges: old cell C → children C (bit=0) and C|(1<<old_k) (bit=1)
        # Copy all edge counts from parent to both children
        new_edges = {}
        for (from_cell, a), targets in self.edges.items():
            child0 = from_cell
            child1 = from_cell | (1 << old_k)
            new_edges[(child0, a)] = dict(targets)
            new_edges[(child1, a)] = dict(targets)
        self.edges = new_edges

        # Expand cells_seen to include both splits of each old cell
        expanded = set()
        for c in self.cells_seen:
            expanded.add(c)
            expanded.add(c | (1 << old_k))
        self.cells_seen = expanded

        # Update prev_cell to new (longer) hash of current observation
        if self.prev_cell is not None and x is not None:
            self.prev_cell = self._hash(x)

        total_edges = sum(sum(t.values()) for t in self.edges.values())
        self.growth_log.append((self.step_count, self.k, len(self.cells_seen), total_edges))

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)

        # Grow before hashing if at boundary
        if self.step_count > 0 and self.step_count % self.grow_every == 0:
            self._grow(x)

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


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = MultiResLSHGraph(k_init=10, grow_every=5000, seed=seed)
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
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'elapsed': time.time() - t0,
        'growth_log': g.growth_log, 'final_k': g.k,
    }


def main():
    import arc_agi
    print("Step 456: Multi-Resolution LSH Graph — time-based growth every 5K steps.", flush=True)
    print("k=10 start -> k=11 at 5K -> k=12 at 10K -> ... 30K steps, 3 seeds.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []

    for seed in [0, 1, 2]:
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=30000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"\n  seed={seed}  {status:22s}  cells={r['unique_cells']:5d}"
              f"  ratio={r['ratio']:.4f}  dom={r['dom']:.0f}%"
              f"  k_final={r['final_k']}  {r['elapsed']:.0f}s", flush=True)
        if r['growth_log']:
            print(f"  Growth events:", flush=True)
            for step, k, cells, edges in r['growth_log']:
                print(f"    step={step:5d}  k={k}  occupied={cells}  edges={edges}", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    wins = [r for r in results if r['levels'] > 0]
    level_steps = sorted([r['level_step'] for r in wins])
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_ratio = sum(r['ratio'] for r in results) / len(results)

    print(f"Step 456: {len(wins)}/3 at 30K  steps={level_steps}", flush=True)
    print(f"avg_cells={avg_cells:.0f}  avg_ratio={avg_ratio:.4f}", flush=True)
    print(f"\nBaselines (30K, 3 seeds):", flush=True)
    print(f"  Fixed LSH k=10 (453):  3/10 at 30K  ratio~0.003-0.012", flush=True)
    print(f"  Cosine graph (445):    3/10 at 50K  ratio~0.07", flush=True)
    print(f"  Grid (446b):           0/3  at 30K", flush=True)

    # Kill criterion verdict
    if len(wins) == 0:
        cells_grew = any(len(r['growth_log']) > 0 and r['growth_log'][-1][2] > r['growth_log'][0][2] * 1.1
                         for r in results if r['growth_log'])
        if cells_grew:
            print(f"\nVERDICT: 0/3 + cells grew → inherited convergence. Growth signal matters.", flush=True)
        else:
            print(f"\nVERDICT: 0/3 + cells didn't grow → mechanism broken.", flush=True)
    else:
        print(f"\nVERDICT: NAVIGATES. Time-based growth compatible with navigation.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
