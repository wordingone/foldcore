#!/usr/bin/env python3
"""
Step 489 — Level 2 brute force. 1M steps, seed 0. Fresh H+edges per level.
Closes multi-level question: is Level 2 slow (budget) or structurally impossible?
Seed 0 finds L1 at ~471 steps -> ~999.5K steps for Level 2.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 1_000_000
TIME_CAP = 450  # 1M at 2300 steps/sec ~ 435s


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class ArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0, level=0):
        self.k = k
        self.n_actions = n_actions
        self._reinit(seed, level)

    def _reinit(self, seed, level):
        rng = np.random.RandomState(seed * 1000 + level + 9999)
        self.H = rng.randn(self.k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(self.k)], dtype=np.int64)
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def reset_for_level(self, seed, level):
        self._reinit(seed, level)

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


def main():
    import arc_agi
    from arcengine import GameState
    seed = 0
    print(f"Step 489: Level 2 brute force. 1M steps, seed={seed}. Fresh H+edges per level.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    np.random.seed(seed)
    env = arc.make(ls20.game_id)
    na = len(env.action_space)
    current_level = 0
    g = ArgminGraph(k=K, n_actions=na, seed=seed, level=current_level)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}
    level_budgets = {}
    level_start_step = 0
    t0 = time.time()
    last_report = 0

    while ts < MAX_STEPS:
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
                elapsed = time.time() - t0
                print(f"  LEVEL {lvl} at step {ts} (budget={ts-level_start_step}, elapsed={elapsed:.1f}s, "
                      f"cells={len(g.cells_seen)}, go={go})", flush=True)
            prev_levels = obs.levels_completed
            current_level = prev_levels
            level_start_step = ts
            g.reset_for_level(seed=seed, level=current_level)

        # Progress report every 100K steps (for Level 2 monitoring)
        if ts - last_report >= 100000:
            elapsed = time.time() - t0
            steps_on_level = ts - level_start_step
            print(f"  [progress] step={ts:>7d}  level={prev_levels}  "
                  f"on_level_step={steps_on_level:>7d}  cells={len(g.cells_seen):4d}  "
                  f"go={go}  {elapsed:.0f}s", flush=True)
            last_report = ts

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    print(f"\nFINAL: max_level={prev_levels}  steps={ts}  elapsed={elapsed:.0f}s", flush=True)
    print(f"\nVERDICT:", flush=True)
    if prev_levels >= 2:
        print(f"  LEVEL 2 FOUND at step {level_steps[2]} (budget={level_budgets[2]} for L2).", flush=True)
        print(f"  L2 is a patience problem, not structural. Budget needed: {level_budgets[2]}", flush=True)
    else:
        steps_on_l2 = ts - level_start_step
        print(f"  LEVEL 2 NOT FOUND after {steps_on_l2} L2-specific steps.", flush=True)
        print(f"  Level 2 is structurally different. Argmin cannot navigate it.", flush=True)


if __name__ == '__main__':
    main()
