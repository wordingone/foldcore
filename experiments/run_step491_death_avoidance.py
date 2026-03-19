#!/usr/bin/env python3
"""
Step 491 — Death-state avoidance on Level 2. LSH k=12, fresh H per level.
On game_over: edge_count[prev_cell][prev_action] += DEATH_PENALTY (1000).
Argmin avoids death-causing actions. Hypothesis: expands reachable region beyond 240.
500K steps, seed 0.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
MAX_STEPS = 500_000
TIME_CAP = 230
DEATH_PENALTY = 1000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class DeathAvoidArgminGraph:
    def __init__(self, k=K, n_actions=4, seed=0, level=0):
        self.k = k
        self.n_actions = n_actions
        self._reinit(seed, level)

    def _reinit(self, seed, level):
        rng = np.random.RandomState(seed * 1000 + level + 9999)
        self.H = rng.randn(self.k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(self.k)], dtype=np.int64)
        self.edges = {}   # (cell, action) -> float (visit count)
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self.death_count = 0

    def reset_for_level(self, seed, level):
        self._reinit(seed, level)

    def penalize_death(self):
        """Called when game_over occurs: heavily penalize the last action taken."""
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            self.edges[key] = self.edges.get(key, 0.0) + DEATH_PENALTY
            self.death_count += 1
        # Reset prev — don't carry into next episode
        self.prev_cell = None
        self.prev_action = None

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            self.edges[key] = self.edges.get(key, 0.0) + 1.0
        counts = [self.edges.get((cell, a), 0.0) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c <= min_c + 1e-9]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def main():
    import arc_agi
    from arcengine import GameState
    seed = 0
    print(f"Step 491: Death avoidance. LSH k={K}, seed={seed}, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"Death penalty = {DEATH_PENALTY}. Fresh H+edges per level.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    np.random.seed(seed)
    env = arc.make(ls20.game_id)
    na = len(env.action_space)
    current_level = 0
    g = DeathAvoidArgminGraph(k=K, n_actions=na, seed=seed, level=current_level)
    obs = env.reset()
    ts = go = 0
    prev_levels = 0
    level_steps = {}
    level_budgets = {}
    level_start_step = 0
    cells_milestones = {}
    t0 = time.time()

    while ts < MAX_STEPS:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            g.penalize_death()
            obs = env.reset()
            continue
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
                print(f"  LEVEL {lvl} at step {ts} (budget={ts-level_start_step}, "
                      f"elapsed={elapsed:.1f}s, cells={len(g.cells_seen)}, deaths={go})", flush=True)
            prev_levels = obs.levels_completed
            current_level = prev_levels
            level_start_step = ts
            g.reset_for_level(seed=seed, level=current_level)

        # Track cells at L2 milestones
        steps_on_level = ts - level_start_step
        if prev_levels >= 1 and steps_on_level in (100000, 200000, 300000, 400000):
            cells_milestones[steps_on_level] = len(g.cells_seen)

        if time.time() - t0 > TIME_CAP: break

    elapsed = time.time() - t0
    l2_steps = ts - level_start_step if prev_levels >= 1 else 0
    cells_l2 = len(g.cells_seen)
    print(f"\nFINAL: max_level={prev_levels}  L2_steps={l2_steps}  cells={cells_l2}  "
          f"go={go}  elapsed={elapsed:.0f}s", flush=True)
    mil_str = "  ".join(f"@{k//1000}K={v}" for k, v in sorted(cells_milestones.items()))
    if mil_str:
        print(f"L2 cells milestones: {mil_str}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if prev_levels >= 2:
        print(f"  DEATH AVOIDANCE BREAKS TRAP: Level 2 found at step {level_steps[2]}!", flush=True)
        print(f"  L2 budget: {level_budgets[2]} steps.", flush=True)
    elif cells_l2 > 300:
        print(f"  REGION EXPANDED: {cells_l2} cells (was ~240). Death avoidance helps.", flush=True)
        print(f"  Level 2 not found but reachable region is larger. More budget needed.", flush=True)
    else:
        print(f"  TRAP UNCHANGED: {cells_l2} cells (was ~240). Deaths aren't the barrier.", flush=True)
        print(f"  The gap is physical — no action sequence reaches L2 reward.", flush=True)


if __name__ == '__main__':
    main()
