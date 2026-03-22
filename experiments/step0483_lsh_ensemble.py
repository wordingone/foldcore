#!/usr/bin/env python3
"""
Step 483 — Ensemble: argmin OR global-novelty wins. 10 seeds x 30K each.
Win = EITHER mechanism navigates. Tests if 6/10 wall is mechanism or game ceiling.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 30000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(x): return x - x.mean()


class ArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.prev_cell = None; self.prev_action = None

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell; self.prev_action = action
        return action


class GlobalNoveltyGraph:
    def __init__(self, k=K, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.cell_visits = {}; self.max_visits = 1
        self.prev_cell = None; self.prev_action = None

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cell_visits[cell] = self.cell_visits.get(cell, 0) + 1
        if self.cell_visits[cell] > self.max_visits: self.max_visits = self.cell_visits[cell]
        if self.prev_cell is not None and self.prev_action is not None:
            penalty = 1.0 + self.cell_visits[cell] / self.max_visits
            key = (self.prev_cell, self.prev_action)
            self.edges[key] = self.edges.get(key, 0.0) + penalty
        counts = [self.edges.get((cell, a), 0.0) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c <= min_c + 1e-9]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell; self.prev_action = action
        return action


def run_mechanism(arc, game_id, seed, GraphClass, max_steps=MAX_STEPS):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = GraphClass(k=K, n_actions=na, seed=seed)
    obs = env.reset()
    ts = lvls = 0; level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: obs = env.reset(); continue
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
        if time.time() - t0 > 140: break
    return lvls > 0, level_step


def main():
    import arc_agi
    n_seeds = 10
    print(f"Step 483: Ensemble argmin+global_novelty. {n_seeds} seeds x {MAX_STEPS//1000}K each.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results_A = []; results_B = []
    for seed in range(n_seeds):
        win_A, step_A = run_mechanism(arc, ls20.game_id, seed, ArgminGraph)
        win_B, step_B = run_mechanism(arc, ls20.game_id, seed, GlobalNoveltyGraph)
        win_ens = win_A or win_B
        a_str = f"A@{step_A}" if win_A else "A-"
        b_str = f"B@{step_B}" if win_B else "B-"
        print(f"  seed={seed:2d}  {a_str:12s}  {b_str:12s}  ensemble={'WIN' if win_ens else 'FAIL'}", flush=True)
        results_A.append(win_A); results_B.append(win_B)
    wins_A = sum(results_A); wins_B = sum(results_B)
    wins_ens = sum(a or b for a, b in zip(results_A, results_B))
    print(f"\nArgmin: {wins_A}/{n_seeds}  GlobalNovelty: {wins_B}/{n_seeds}  Ensemble: {wins_ens}/{n_seeds}", flush=True)
    print("\nVERDICT:", flush=True)
    if wins_ens >= 8:
        print(f"  ENSEMBLE BREAKS WALL: {wins_ens}/{n_seeds}. Mechanisms complement each other.", flush=True)
        print(f"  6/10 is mechanism-specific, not game ceiling.", flush=True)
    elif wins_ens >= 7:
        print(f"  PARTIAL COMPLEMENT: {wins_ens}/{n_seeds}. Some overlap but mechanisms share hard seeds.", flush=True)
    else:
        print(f"  HARD SEEDS DOMINATE: {wins_ens}/{n_seeds}. Failing seeds are hard for both. 6/10 is game ceiling.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
