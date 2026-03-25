"""
Step 1037b -- R3 Counterfactual Meta-Learning Goal Inference

R3 HYPOTHESIS: If the substrate modifies WHICH goal inference method it uses
based on per-click counterfactual evaluation (did click change the zone? did
method predict this zone needed clicking?), then meta-weights differentiate
per game and L2+ emerges from self-modification.
Falsified if: meta-weights still uniform after warmup.

Fix for 1037: counterfactual evaluation on EVERY click (3000 signals during
warmup) instead of level transitions only (1-2 signals per game).

Three goal inference methods (game-agnostic):
  A: freq_mode -- per-pixel most-frequent color
  B: spatial_majority -- spatially-smoothed freq mode (k=16)
  C: novelty -- running mean

Counterfactual test per click at (x,y):
  For each method M: did M predict high mismatch at (x,y)?
    If yes AND click changed zone -> reward M (correctly identified wrong zone)
    If yes AND click didn't change -> punish M (false positive)

All 3 games, randomized order, 5 seeds, 10K steps/game.
KILL: meta-weights uniform after warmup (counterfactual signal too noisy).
SUCCESS: weights differentiate per game AND L1 >= 5/5 on click games.
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

# --- Config ---
WARMUP_STEPS = 3000
CF_LR = 0.02           # counterfactual learning rate (smaller, many updates)
CF_MISMATCH_THRESH = 0.5  # method "would have clicked" threshold
CF_CHANGE_THRESH = 0.1    # pixel change = "zone responded"
ALPHA_CHANGE = 0.99
ALPHA_NOVELTY = 0.999
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
ZONE_RADIUS = 3
MAX_STEPS = 10_000
TIME_CAP = 120

GAME_ACTIONS = list(GameAction)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
N_KB_ACTIONS = 7


class R3CounterfactualSubstrate:
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.meta_w = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB_ACTIONS, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._click_xy = (32, 32)
        self._last_click_xy = None
        self._frame_before = None

    def _goal_A(self, obs):
        return np.argmax(self.freq_hist, axis=2).astype(np.float32)

    def _goal_B(self, obs):
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        for c in range(16):
            smoothed[:, :, c] = uniform_filter(
                self.freq_hist[:, :, c].astype(np.float32), size=16)
        return np.argmax(smoothed, axis=2).astype(np.float32)

    def _goal_C(self, obs):
        if self.novelty_map is None:
            return obs.copy()
        return self.novelty_map.copy()

    def _local_change(self, frame_before, frame_after, cx, cy):
        a = np.array(frame_before[0], dtype=np.float32)
        b = np.array(frame_after[0], dtype=np.float32)
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)
        return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())

    def counterfactual_update(self, frame_before, frame_after, cx, cy):
        """Test all 3 methods against this click's outcome."""
        zone_changed = self._local_change(frame_before, frame_after, cx, cy) > CF_CHANGE_THRESH
        obs = np.array(frame_before[0], dtype=np.float32)

        goals = [self._goal_A(obs), self._goal_B(obs), self._goal_C(obs)]
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        for i, goal in enumerate(goals):
            local_mismatch = float(np.abs(obs[y0:y1, x0:x1] - goal[y0:y1, x0:x1]).mean())
            if local_mismatch > CF_MISMATCH_THRESH:
                # Method predicted this zone needs clicking
                if zone_changed:
                    self.meta_w[i] += CF_LR  # correct: zone was interactive
                else:
                    self.meta_w[i] = max(0.01, self.meta_w[i] - CF_LR)  # false positive
        # Clamp
        self.meta_w = np.clip(self.meta_w, 0.01, 10.0)

    def step(self, frame):
        arr = np.array(frame[0], dtype=np.float32)
        obs_int = np.array(frame[0], dtype=np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        # Counterfactual update from previous click
        if self._frame_before is not None and self._last_click_xy is not None:
            cx, cy = self._last_click_xy
            self.counterfactual_update(self._frame_before, frame, cx, cy)

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            self._click_xy = (32, 32)
            self._frame_before = frame
            self._last_click_xy = (32, 32)
            return 'click'

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        r_idx, c_idx = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r_idx, c_idx, obs_int] += 1
        self.novelty_map = ALPHA_NOVELTY * self.novelty_map + (1 - ALPHA_NOVELTY) * arr

        # Weighted goal (R3: meta-weights determine effective operation)
        w = self.meta_w / self.meta_w.sum()
        goals = [self._goal_A(arr), self._goal_B(arr), self._goal_C(arr)]
        combined_goal = w[0] * goals[0] + w[1] * goals[1] + w[2] * goals[2]

        mismatch = np.abs(arr - combined_goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # Warmup: random
        if self.step_count < WARMUP_STEPS:
            if self.rng.random() < 0.9:
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self._frame_before = frame
                self._last_click_xy = (cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                self._frame_before = None
                self._last_click_xy = None
                return ('kb', kb)

        # Post-warmup: score-based
        click_score = float(np.max(smoothed))
        kb_scores = np.zeros(N_KB_ACTIONS)
        for k in range(N_KB_ACTIONS):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        if self.rng.random() < 0.1:
            if self.rng.random() < 0.5:
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self._frame_before = frame
                self._last_click_xy = (cx, cy)
                self.prev_action_type = 'click'
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                self._frame_before = None
                self._last_click_xy = None
                return ('kb', kb)

        if click_score >= best_kb_score:
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            self._click_xy = (int(x), int(y))
            y0 = max(0, y - SUPPRESS_RADIUS)
            y1 = min(64, y + SUPPRESS_RADIUS + 1)
            x0 = max(0, x - SUPPRESS_RADIUS)
            x1 = min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self._frame_before = frame
            self._last_click_xy = (int(x), int(y))
            self.prev_action_type = 'click'
            return 'click'
        else:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = best_kb
            self._frame_before = None
            self._last_click_xy = None
            return ('kb', best_kb)

    def on_level_transition(self):
        self.freq_hist[:] = 0
        self.suppress[:] = 0


def run_game(arc, game_id, seed, substrate):
    np.random.seed(seed)
    env = arc.make(game_id)
    obs = env.reset()
    t0 = time.time()
    max_levels = 0
    level_steps = {}
    total_steps = 0

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

    print("=== Step 1037b: R3 Counterfactual Meta-Learning ===")
    print("R3: per-click counterfactual evaluation updates meta-weights")
    print(f"Config: warmup={WARMUP_STEPS}, cf_lr={CF_LR}, cf_thresh={CF_MISMATCH_THRESH}")
    print(f"Games: {', '.join(f'{g} ({games[g]})' for g in game_names)}")
    print()

    all_results = {g: [] for g in game_names}

    for seed in range(5):
        rng = np.random.RandomState(seed)
        order = list(game_names)
        rng.shuffle(order)
        print(f"--- Seed {seed} (order: {' > '.join(order)}) ---")
        substrate = R3CounterfactualSubstrate(seed)

        for gname in order:
            r = run_game(arc, games[gname], seed, substrate)
            w_str = ' '.join(f'{METHOD_NAMES[i]}={r["meta_w"][i]:.2f}' for i in range(3))
            lvl_str = ""
            if r['level_steps']:
                lvl_str = "  " + "  ".join(
                    f"L{l}@{s}" for l, s in sorted(r['level_steps'].items()))
            print(f"  {gname}: L={r['max_levels']}  steps={r['steps']}  "
                  f"dom={METHOD_NAMES[r['active']]}  [{w_str}]  "
                  f"{r['elapsed']}s{lvl_str}")
            all_results[gname].append(r)

    print("\n=== Summary ===")
    any_l2 = False
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        counts = {}
        for lvl in range(1, 6):
            c = sum(1 for r in results if r['max_levels'] >= lvl)
            if c > 0: counts[lvl] = c
        max_l = max(r['max_levels'] for r in results)
        parts = [f"L{l}: {c}/5" for l, c in sorted(counts.items())]
        dom_counts = [0, 0, 0]
        for r in results: dom_counts[r['active']] += 1
        dom_str = ' '.join(f'{METHOD_NAMES[i]}={dom_counts[i]}' for i in range(3) if dom_counts[i] > 0)
        print(f"  {gname}: {', '.join(parts) if parts else 'L0 only'}  (max={max_l})  dom: {dom_str}")
        if max_l >= 2: any_l2 = True

    print("\n=== R3 Diagnostic ===")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        avg_w = np.mean([r['meta_w'] for r in results], axis=0)
        print(f"  {gname}: [{', '.join(f'{METHOD_NAMES[i]}={avg_w[i]:.3f}' for i in range(3))}]")

    if any_l2:
        print("\n  SIGNAL: R3 counterfactual meta-learning produces L2+!")
    else:
        print("\n  FAIL: No L2+.")
    # Check R3 differentiation
    all_avg = {}
    for gname in sorted(all_results.keys()):
        all_avg[gname] = np.mean([r['meta_w'] for r in all_results[gname]], axis=0)
    max_diff = max(np.max(np.abs(all_avg[g1] - all_avg[g2]))
                   for g1 in all_avg for g2 in all_avg if g1 != g2)
    if max_diff > 0.1:
        print(f"  R3 ACTIVE: methods differentiate across games (max_diff={max_diff:.3f})")
    else:
        print(f"  R3 WEAK: methods similar across games (max_diff={max_diff:.3f})")

    print("\nStep 1037b DONE")


if __name__ == "__main__":
    main()
