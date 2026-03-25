"""
Step 1037 -- R3 Meta-Learning Goal Inference (Debate: R3+D2 Synthesis)

R3 HYPOTHESIS: If the substrate modifies WHICH goal inference method it uses
based on interaction outcomes (level transitions = reward), then it achieves
L2+ on games where fixed-method substrates fail.
Falsified if: meta-learning adds zero vs best fixed method.

Three goal inference methods (all game-agnostic):
  A: freq_mode -- per-pixel most-frequent color (simple toggle targets)
  B: spatial_majority -- spatially-smoothed freq mode (cross-zone majority)
  C: novelty -- running mean (deviation = exploration signal)

Meta-weights w=[w_A, w_B, w_C] self-modify via level transitions (+reward)
and stagnation (-reward). This is l_1 R3 (data-modified operations).

D2 MEASUREMENT: does method B dominate on VC33? method C on LS20?
If different methods dominate on different games AND L2+ > 0 -> R3 matters.

All 3 games, randomized order, 5 seeds, 10K steps/game.
KILL: meta-learning adds zero vs best fixed method.
SUCCESS: L2+ on any game + different method dominance per game.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import uniform_filter

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# --- Config (ONE config, all games) ---
WARMUP_STEPS = 3000
META_LR = 0.1
STAGNATION_WINDOW = 50   # clicks with no level change -> punish
ALPHA_CHANGE = 0.99
ALPHA_NOVELTY = 0.999
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
MAX_STEPS = 10_000
TIME_CAP = 120

# Action space
GAME_ACTIONS = list(GameAction)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
N_KB_ACTIONS = 7  # ACTION1-ACTION7


class R3MetaGoalSubstrate:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)

        # WHERE
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None

        # Goal inference methods
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)  # Method A+B
        self.novelty_map = None  # Method C

        # Meta-weights (R3: these self-modify)
        self.meta_w = np.array([1.0/3, 1.0/3, 1.0/3], dtype=np.float32)
        self.active_method = 0
        self.stagnation_counter = 0

        # Recency suppression
        self.suppress = np.zeros((64, 64), dtype=np.int32)

        # Influence maps for keyboard actions
        self.kb_influence = np.zeros((N_KB_ACTIONS, 64, 64), dtype=np.float32)
        self.prev_action_type = None  # 'click' or int(0-6)
        self.prev_kb_idx = None

        self.step_count = 0
        self._click_xy = (32, 32)

    def _goal_A(self, obs):
        """freq_mode: per-pixel most frequent color."""
        return np.argmax(self.freq_hist, axis=2).astype(np.float32)

    def _goal_B(self, obs):
        """spatial_majority: spatially-smoothed freq mode."""
        # Smooth each color channel's frequency, then argmax
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        for c in range(16):
            smoothed[:, :, c] = uniform_filter(
                self.freq_hist[:, :, c].astype(np.float32), size=16)
        return np.argmax(smoothed, axis=2).astype(np.float32)

    def _goal_C(self, obs):
        """novelty: running mean target."""
        if self.novelty_map is None:
            return obs.copy()
        return self.novelty_map.copy()

    def step(self, frame):
        """Process one frame. Returns (action_type, click_xy_or_kb_idx)."""
        arr = np.array(frame[0], dtype=np.float32)
        obs_int = np.array(frame[0], dtype=np.int32)
        self.step_count += 1

        # Update suppress
        self.suppress = np.maximum(0, self.suppress - 1)

        # First step
        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            # Update freq histogram
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            self._click_xy = (32, 32)
            return 'click'

        # 1. diff + WHERE
        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        # Update keyboard influence
        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        # 2. Update goal method inputs
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r, c, obs_int] += 1
        self.novelty_map = ALPHA_NOVELTY * self.novelty_map + (1 - ALPHA_NOVELTY) * arr

        # 3. Compute three goal maps
        goals = [self._goal_A(arr), self._goal_B(arr), self._goal_C(arr)]

        # 4. Weighted goal (R3: meta-weights determine effective operation)
        w = self.meta_w / self.meta_w.sum()  # normalize
        combined_goal = w[0] * goals[0] + w[1] * goals[1] + w[2] * goals[2]

        # 5. Mismatch
        mismatch = np.abs(arr - combined_goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        # 6. Active method = dominant weight
        self.active_method = int(np.argmax(w))

        # 7. Action selection
        self.prev_obs = arr.copy()

        # During warmup: random
        if self.step_count < WARMUP_STEPS:
            if self.rng.random() < 0.9:  # 90% click, 10% keyboard
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                return ('kb', kb)

        # Post-warmup: score click vs keyboard
        click_score = float(np.max(smoothed))

        kb_scores = np.zeros(N_KB_ACTIONS)
        for k in range(N_KB_ACTIONS):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)

        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        # Epsilon exploration
        if self.rng.random() < 0.1:
            if self.rng.random() < 0.5:
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                return ('kb', kb)

        if click_score >= best_kb_score:
            # Click at argmax
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            self._click_xy = (int(x), int(y))
            # Apply suppress
            y0 = max(0, y - SUPPRESS_RADIUS)
            y1 = min(64, y + SUPPRESS_RADIUS + 1)
            x0 = max(0, x - SUPPRESS_RADIUS)
            x1 = min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
            return 'click'
        else:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = best_kb
            return ('kb', best_kb)

    def on_level_transition(self):
        """Reward active method on level transition (R5-compliant R3 trigger)."""
        self.meta_w[self.active_method] += META_LR
        self.stagnation_counter = 0
        # Reset freq histogram for new level layout
        self.freq_hist[:] = 0
        self.suppress[:] = 0

    def on_stagnation(self):
        """Punish active method on stagnation."""
        self.meta_w[self.active_method] = max(0.01, self.meta_w[self.active_method] - META_LR)


def run_game(arc, game_id, seed, substrate):
    np.random.seed(seed)
    env = arc.make(game_id)

    obs = env.reset()
    t0 = time.time()
    max_levels = 0
    level_steps = {}
    total_steps = 0
    stagnation = 0
    meta_log = []

    for step_i in range(MAX_STEPS):
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        result = substrate.step(obs.frame)
        total_steps += 1

        levels_before = obs.levels_completed

        if result == 'click':
            cx, cy = substrate._click_xy
            obs = env.step(GAME_ACTIONS[6], data={"x": cx, "y": cy})
        else:
            _, kb_idx = result
            obs = env.step(GAME_ACTIONS[kb_idx + 1], data={})

        if obs and obs.levels_completed > levels_before:
            if obs.levels_completed > max_levels:
                max_levels = obs.levels_completed
                level_steps[max_levels] = step_i + 1
            substrate.on_level_transition()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= STAGNATION_WINDOW:
                substrate.on_stagnation()
                stagnation = 0

    elapsed = time.time() - t0
    w = substrate.meta_w / substrate.meta_w.sum()
    return {
        'max_levels': max_levels,
        'level_steps': level_steps,
        'steps': total_steps,
        'elapsed': round(elapsed, 1),
        'meta_w': [round(float(x), 3) for x in w],
        'active': int(np.argmax(w)),
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    games = {}
    for e in envs:
        gname = e.game_id.split('-')[0]
        if gname not in games:
            games[gname] = e.game_id
    game_names = sorted(games.keys())

    METHOD_NAMES = ['freq_mode', 'spatial_maj', 'novelty']

    print("=== Step 1037: R3 Meta-Learning Goal Inference ===")
    print("R3: meta-weights over 3 goal methods self-modify from level transitions")
    print(f"Methods: {METHOD_NAMES}")
    print(f"Config: warmup={WARMUP_STEPS}, meta_lr={META_LR}, stag_window={STAGNATION_WINDOW}")
    print(f"Games: {', '.join(f'{g} ({games[g]})' for g in game_names)}")
    print()

    all_results = {g: [] for g in game_names}

    for seed in range(5):
        rng = np.random.RandomState(seed)
        order = list(game_names)
        rng.shuffle(order)
        print(f"--- Seed {seed} (order: {' > '.join(order)}) ---")

        substrate = R3MetaGoalSubstrate(seed)

        for gname in order:
            r = run_game(arc, games[gname], seed, substrate)
            w_str = ' '.join(f'{METHOD_NAMES[i]}={r["meta_w"][i]:.2f}'
                           for i in range(3))
            lvl_str = ""
            if r['level_steps']:
                lvl_str = "  " + "  ".join(
                    f"L{l}@{s}" for l, s in sorted(r['level_steps'].items()))
            print(f"  {gname}: L={r['max_levels']}  steps={r['steps']}  "
                  f"dominant={METHOD_NAMES[r['active']]}  [{w_str}]  "
                  f"{r['elapsed']}s{lvl_str}")
            all_results[gname].append(r)

    # Summary
    print("\n=== Summary ===")
    any_l2 = False
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        counts = {}
        for lvl in range(1, 6):
            c = sum(1 for r in results if r['max_levels'] >= lvl)
            if c > 0:
                counts[lvl] = c
        max_l = max(r['max_levels'] for r in results)
        parts = [f"L{l}: {c}/5" for l, c in sorted(counts.items())]

        # Method dominance
        dom_counts = [0, 0, 0]
        for r in results:
            dom_counts[r['active']] += 1
        dom_str = ' '.join(f'{METHOD_NAMES[i]}={dom_counts[i]}'
                          for i in range(3) if dom_counts[i] > 0)
        print(f"  {gname}: {', '.join(parts) if parts else 'L0 only'}  "
              f"(max={max_l})  dominant: {dom_str}")
        if max_l >= 2:
            any_l2 = True

    # R3 diagnostic: do different methods dominate on different games?
    print("\n=== R3 Diagnostic: Method Dominance Per Game ===")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        avg_w = np.mean([r['meta_w'] for r in results], axis=0)
        print(f"  {gname}: avg_w = [{', '.join(f'{METHOD_NAMES[i]}={avg_w[i]:.3f}' for i in range(3))}]")

    if any_l2:
        print("\n  SIGNAL: R3 meta-learning produces L2+!")
    else:
        print("\n  FAIL: No L2+ from meta-learning.")

    print("\nStep 1037 DONE")


if __name__ == "__main__":
    main()
