#!/usr/bin/env python3
"""
Step 488 — Fresh H per level. LSH k=12 argmin.
At level_completed: reset edges AND re-generate random H matrix.
Tests: is Level 2 structurally harder, or does Level 1's projection fail on Level 2?
300K steps, 3 seeds.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 300000
TIME_CAP = 150  # per seed


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class ArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0, level=0):
        self.k = k
        self.n_actions = n_actions
        self._reinit(seed=seed, level=level)

    def _reinit(self, seed, level):
        # Unique seed per (seed, level) combination
        rng = np.random.RandomState(seed * 1000 + level + 9999)
        self.H = rng.randn(self.k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(self.k)], dtype=np.int64)
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def reset_for_level(self, seed, level):
        """Fresh H + fresh edges for new level."""
        self._reinit(seed=seed, level=level)

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
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


def run_seed(arc, game_id, seed, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    current_level = 0
    g = ArgminGraph(k=K, n_actions=na, seed=seed, level=current_level)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}
    level_budgets = {}
    level_cells = {}
    level_start_step = 0
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

        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break

        if obs.levels_completed > prev_levels:
            for lvl in range(prev_levels + 1, obs.levels_completed + 1):
                level_steps[lvl] = ts
                level_budgets[lvl] = ts - level_start_step
                level_cells[lvl] = len(g.cells_seen)
            prev_levels = obs.levels_completed
            current_level = prev_levels
            level_start_step = ts
            g.reset_for_level(seed=seed, level=current_level)  # fresh H + fresh edges

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'level_steps': level_steps,
        'level_budgets': level_budgets,
        'level_cells': level_cells,
        'max_levels': prev_levels,
        'game_overs': go,
        'steps_reached': ts,
        'elapsed': elapsed,
        'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 488: Fresh H+edges per level. LSH k={K}. {MAX_STEPS//1000}K steps, {n_seeds} seeds.", flush=True)
    print(f"Level transition: re-init H (new projection) + wipe edges.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed)
        parts = []
        for lvl in range(1, r['max_levels'] + 2):
            if lvl in r['level_steps']:
                parts.append(f"L{lvl}@{r['level_steps'][lvl]}({r['level_budgets'][lvl]}steps,{r['level_cells'].get(lvl,0)}cells)")
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  seed={seed}  max_lvl={r['max_levels']}  "
              f"{' '.join(parts) if parts else 'FAIL'}  go={r['game_overs']}  "
              f"{r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    l1 = sum(1 for r in results if r['max_levels'] >= 1)
    l2 = sum(1 for r in results if r['max_levels'] >= 2)
    l3 = sum(1 for r in results if r['max_levels'] >= 3)
    print(f"\nLevel 1: {l1}/{n_seeds}  Level 2: {l2}/{n_seeds}  Level 3: {l3}/{n_seeds}", flush=True)
    if l2 > 0:
        l2_budgets = [r['level_budgets'][2] for r in results if 2 in r['level_budgets']]
        print(f"Level 2 budgets (steps for L2 alone): {l2_budgets}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if l2 >= 2:
        print(f"  FRESH PROJ WORKS: {l2}/{n_seeds} reach Level 2.", flush=True)
        print(f"  Level 1 projection failed L2, not L2 being structurally harder.", flush=True)
    elif l2 >= 1:
        print(f"  PARTIAL: {l2}/{n_seeds} reach Level 2 with fresh projection.", flush=True)
    else:
        print(f"  STRUCTURALLY HARDER: 0/{n_seeds} Level 2 even with fresh random projection.", flush=True)
        print(f"  Level 2 game dynamics differ from Level 1. Not a projection problem.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
