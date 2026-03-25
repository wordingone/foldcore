"""
Step 1039 -- Prosecution: Attention-Gated l_pi R3

PROSECUTION HYPOTHESIS: Attention-gated encoding is genuine l_pi R3.
The attention map modifies WHAT the frequency histogram accumulates
(weighted accumulation), not just how it processes fixed inputs.

Architecture (ONE system, ONE config, NO bootloader):
  raw_freq[64,64,16]    - pixel freq from raw obs (baseline, l_0)
  gated_freq[64,64,16]  - pixel freq weighted by attention (l_pi encoding)
  attention[64,64]      - encoding gate in [0.01, 1.0], starts uniform 0.5
  change_map[64,64]     - EMA of pixel change magnitude

Processing each step:
  1. raw_goal = mode(raw_freq)              # baseline goal
  2. gated_freq[y,x,obs] += attention[y,x]  # weighted accumulation (l_pi)
  3. gated_goal = mode(gated_freq)           # l_pi goal
  4. mismatch = attention * |obs - gated_goal| * change_map
  5. action = click/kb based on mismatch scoring

R3 mechanism (encoding self-modification via counterfactual advantage):
  After each click, for each CHANGED pixel p:
    raw_error = |raw_goal[p] - post_obs[p]|
    gated_error = |gated_goal[p] - post_obs[p]|
    advantage = raw_error - gated_error  (positive = gated predicted better)
    attention[p] += eta * advantage

Why l_pi: attention changes what gated_freq accumulates. Different attention
= different frequency data = different goal. The ENCODING self-modifies.

Kill: attention stays uniform (max-min < 0.05) after 500 clicks
Success: attention differentiates per game AND L1+ improves over 1037b

Note: Leo spec'd 2000 steps/game. Using 10K for fair comparison with
1037b/1038. Warmup 3000 (same as all previous experiments).

All 3 preview games, randomized order, 5 seeds.
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
GAME_FILTER = ['ft09', 'ls20', 'vc33']  # preview games only
WARMUP_STEPS = 3000
MAX_STEPS = 10_000
TIME_CAP = 120
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# --- Attention R3 Config ---
ATT_INIT = 0.5
ATT_LR = 0.02    # learning rate for attention updates
ATT_MIN = 0.01
ATT_MAX = 1.0

GAME_ACTIONS = list(GameAction)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
N_KB_ACTIONS = 7


class AttentionGatedSubstrate:
    """l_pi R3: attention-gated encoding with counterfactual advantage."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        # Dual frequency histograms
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        # Attention map (l_pi: this modifies the encoding)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)
        # Statistics
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB_ACTIONS, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._click_xy = (32, 32)
        self._last_click_xy = None
        self._frame_before = None
        # Cache goals for R3 update
        self._raw_goal = None
        self._gated_goal = None
        # Diagnostics
        self.r3_updates = 0
        self.att_updates_total = 0

    def _r3_attention_update(self, frame_before, frame_after):
        """Counterfactual advantage: update attention at changed pixels."""
        pre = np.array(frame_before[0], dtype=np.float32)
        post = np.array(frame_after[0], dtype=np.float32)

        # Find changed pixels
        diff = np.abs(post - pre)
        changed = diff > CF_CHANGE_THRESH

        n_changed = int(np.sum(changed))
        if n_changed == 0:
            return

        # Use cached goals from the step that produced the click
        if self._raw_goal is None or self._gated_goal is None:
            return

        # Advantage at changed pixels: positive = gated predicted better
        raw_error = np.abs(self._raw_goal - post)
        gated_error = np.abs(self._gated_goal - post)
        advantage = raw_error - gated_error

        # Update attention only at changed pixels
        self.attention[changed] += ATT_LR * advantage[changed]
        self.attention = np.clip(self.attention, ATT_MIN, ATT_MAX)

        self.r3_updates += 1
        self.att_updates_total += n_changed

    def step(self, frame):
        arr = np.array(frame[0], dtype=np.float32)
        obs_int = np.array(frame[0], dtype=np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        # R3 update from previous click
        if self._frame_before is not None and self._last_click_xy is not None:
            self._r3_attention_update(self._frame_before, frame)

        # Update frequency histograms
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0
        # l_pi: weighted accumulation based on attention
        self.gated_freq[r, c, obs_int] += self.attention

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._click_xy = (32, 32)
            self._frame_before = frame
            self._last_click_xy = (32, 32)
            # Compute initial goals
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
            return 'click'

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        # Compute goals
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)

        # Mismatch using gated goal and attention weighting
        mismatch = self.attention * np.abs(arr - self._gated_goal) * self.change_map
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
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.suppress[:] = 0
        # Attention PERSISTS across levels (l_pi encoding carries forward)


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
    att = substrate.attention
    return {
        'max_levels': max_levels,
        'level_steps': level_steps,
        'steps': total_steps,
        'elapsed': round(elapsed, 1),
        'att_mean': round(float(np.mean(att)), 4),
        'att_std': round(float(np.std(att)), 4),
        'att_min': round(float(np.min(att)), 4),
        'att_max': round(float(np.max(att)), 4),
        'att_range': round(float(np.max(att) - np.min(att)), 4),
        'r3_updates': substrate.r3_updates,
        'att_pixel_updates': substrate.att_updates_total,
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    games = {}
    for e in envs:
        gname = e.game_id.split('-')[0]
        if gname in GAME_FILTER and gname not in games:
            games[gname] = e.game_id
    game_names = sorted(games.keys())

    if set(game_names) != set(GAME_FILTER):
        missing = set(GAME_FILTER) - set(game_names)
        print(f"WARNING: missing games: {missing}")

    print("=== Step 1039: Prosecution -- Attention-Gated l_pi R3 ===")
    print("R3: attention modifies encoding (weighted freq accumulation)")
    print(f"Config: warmup={WARMUP_STEPS}, att_lr={ATT_LR}, att_init={ATT_INIT}")
    print(f"Games: {', '.join(f'{g} ({games[g]})' for g in game_names)}")
    n_envs = len(arc.get_environments())
    print(f"Total environments available: {n_envs} (filtered to {len(game_names)})")
    print()

    all_results = {g: [] for g in game_names}

    for seed in range(5):
        rng = np.random.RandomState(seed)
        order = list(game_names)
        rng.shuffle(order)
        print(f"--- Seed {seed} (order: {' > '.join(order)}) ---")
        substrate = AttentionGatedSubstrate(seed)

        for gname in order:
            r = run_game(arc, games[gname], seed, substrate)
            lvl_str = ""
            if r['level_steps']:
                lvl_str = "  " + "  ".join(
                    f"L{l}@{s}" for l, s in sorted(r['level_steps'].items()))
            print(f"  {gname}: L={r['max_levels']}  steps={r['steps']}  "
                  f"att=[{r['att_mean']:.3f}+/-{r['att_std']:.3f} "
                  f"r={r['att_range']:.3f}]  "
                  f"r3n={r['r3_updates']}  "
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
        print(f"  {gname}: {', '.join(parts) if parts else 'L0 only'}  (max={max_l})")
        if max_l >= 2: any_l2 = True

    print("\n=== R3 Diagnostic (l_pi: attention map) ===")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        avg_mean = np.mean([r['att_mean'] for r in results])
        avg_std = np.mean([r['att_std'] for r in results])
        avg_range = np.mean([r['att_range'] for r in results])
        avg_r3n = np.mean([r['r3_updates'] for r in results])
        avg_pxu = np.mean([r['att_pixel_updates'] for r in results])
        print(f"  {gname}: att_mean={avg_mean:.4f}  att_std={avg_std:.4f}  "
              f"att_range={avg_range:.4f}  r3_updates={avg_r3n:.0f}  "
              f"pixel_updates={avg_pxu:.0f}")

    # Check if attention is uniform (kill criterion)
    all_ranges = {g: np.mean([r['att_range'] for r in all_results[g]])
                  for g in game_names}

    if any_l2:
        print("\n  SIGNAL: Attention-gated l_pi produces L2+!")
    else:
        print("\n  FAIL: No L2+.")

    if all(r < 0.05 for r in all_ranges.values()):
        print("  R3 INERT (l_pi): attention stayed uniform across ALL games. KILL.")
    elif any(r >= 0.05 for r in all_ranges.values()):
        active_games = [g for g, r in all_ranges.items() if r >= 0.05]
        print(f"  R3 ACTIVE (l_pi): attention differentiated in: "
              f"{', '.join(active_games)}")
    else:
        print("  R3 WEAK: attention barely moved")

    # Cross-game differentiation
    att_means = {g: np.mean([r['att_mean'] for r in all_results[g]])
                 for g in game_names}
    max_mean_diff = max(abs(att_means[g1] - att_means[g2])
                        for g1 in game_names for g2 in game_names if g1 != g2)
    print(f"  Cross-game att_mean max_diff: {max_mean_diff:.4f}")

    print("\n=== Comparison ===")
    print("  1037b (prosecution l_1): ft09 1/5 L1, vc33 0/5, ls20 0/5")
    print("  1038  (defense l_1):     ft09 1/5 L1, vc33 0/5, ls20 0/5")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        l1 = sum(1 for r in results if r['max_levels'] >= 1)
        l2 = sum(1 for r in results if r['max_levels'] >= 2)
        print(f"  1039  (prosecution l_pi): {gname} {l1}/5 L1, {l2}/5 L2")

    print("\nStep 1039 DONE")


if __name__ == "__main__":
    main()
