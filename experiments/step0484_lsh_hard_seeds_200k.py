#!/usr/bin/env python3
"""
Step 484 — Hard seeds (1,2,5,6) at 200K steps. LSH k=12 argmin baseline.
Hypothesis: hard seeds are slow but not impossible.
Kill: 0/4 at 200K -> genuine ceiling, not budget limit.
NOTE: Per-seed cap 400s. Total runtime ~27 min worst case.
Explicitly requested this extended budget test (mail 1959).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 200000
TIME_CAP = 400  # seconds per seed


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
        'occupancy': len(g.cells_seen) / (2 ** K),
        'elapsed': elapsed, 'timed_out': elapsed >= TIME_CAP - 1
    }


def main():
    import arc_agi
    hard_seeds = [1, 2, 5, 6]
    print(f"Step 484: Hard seeds {hard_seeds} at {MAX_STEPS//1000}K steps. LSH k={K} argmin.", flush=True)
    print(f"Per-seed cap: {TIME_CAP}s. Hypothesis: slow but reachable.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in hard_seeds:
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        timeout_flag = " [TIMEOUT]" if r['timed_out'] else ""
        print(f"  seed={seed}  {status:22s}  steps={r['steps_reached']:>6d}  "
              f"cells={r['unique_cells']:4d}/{2**K}  occ={r['occupancy']:.4f}  "
              f"go={r['game_overs']}  {r['elapsed']:.0f}s{timeout_flag}", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    total_steps = sum(r['steps_reached'] for r in results)
    print(f"\n{len(wins)}/{len(hard_seeds)} hard seeds navigated", flush=True)
    if wins:
        print(f"level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"Total steps across all seeds: {total_steps:,}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if len(wins) >= 3:
        print(f"  SLOW NOT IMPOSSIBLE: {len(wins)}/{len(hard_seeds)}. Hard seeds just need more budget.", flush=True)
        print(f"  6/10 ceiling is a step-budget artifact.", flush=True)
    elif len(wins) >= 1:
        print(f"  PARTIAL ESCAPE: {len(wins)}/{len(hard_seeds)}. Some hard seeds solvable with patience.", flush=True)
        print(f"  Mixed ceiling — some seeds trapped, some slow.", flush=True)
    else:
        print(f"  HARD CEILING CONFIRMED: 0/{len(hard_seeds)} at {max(r['steps_reached'] for r in results):,} steps.", flush=True)
        print(f"  Seeds 1,2,5,6 cannot navigate LS20 with argmin. 6/10 is a real ceiling.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
