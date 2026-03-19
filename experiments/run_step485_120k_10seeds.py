#!/usr/bin/env python3
"""
Step 485 — LSH k=12 argmin at 120K steps, 10 seeds. Capstone reliability test.
Step 484 showed hard seeds (1,2,5,6) navigate at 35K-115K. 120K should be 10/10.
~2300 steps/sec -> early exits make total ~3 min.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 120000
TIME_CAP = 130  # seconds per seed (generous safety cap)


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
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

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
    g = ArgminGraph(k=K, n_actions=na, seed=seed)
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
        if time.time() - t0 > TIME_CAP: break
    elapsed = time.time() - t0
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'steps_reached': ts, 'game_overs': go,
        'unique_cells': len(g.cells_seen),
        'elapsed': elapsed, 'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    n_seeds = 10
    print(f"Step 485: LSH k={K} argmin, {MAX_STEPS//1000}K steps, {n_seeds} seeds. Capstone.", flush=True)
    print(f"Expected: 10/10 based on Step 484 (slowest hard seed = 115K).", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"WIN  @ {r['level_step']:>6d}" if r['levels'] > 0 else "FAIL"
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  seed={seed:2d}  {status}  steps={r['steps_reached']:>6d}  "
              f"cells={r['unique_cells']:4d}  {r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    print(f"\n{len(wins)}/{n_seeds}", flush=True)
    if wins:
        print(f"level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if len(wins) == 10:
        print(f"  10/10 CONFIRMED. LSH k=12 argmin is fully reliable at 120K.", flush=True)
        print(f"  Zero-codebook substrate navigates LS20 with 100% reliability.", flush=True)
    elif len(wins) >= 8:
        print(f"  {len(wins)}/10 — near-perfect. Budget not quite sufficient for all seeds.", flush=True)
    else:
        print(f"  {len(wins)}/10 — Step 484 hard seeds not representative. Ceiling persists.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
