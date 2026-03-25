"""
sub1044_defense_v3.py — Defense v3: systematic scan + adaptive exploit + multi-signal goal (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1044 --substrate experiments/sub1044_defense_v3.py

FAMILY: parametric-goal
R3 HYPOTHESIS: ℓ₁ R3 with systematic warmup scan + adaptive exploration-exploitation.
  Three improvements over v2:
  1. SYSTEMATIC SCAN during warmup: deterministic grid coverage guarantees every
     region is explored at least once. Random warmup misses interactive zones on
     games with small/sparse targets.
  2. ADAPTIVE EXPLORATION: after warmup, if no change detected in N steps,
     temporarily increase exploration rate. SPSA-tunable stagnation threshold.
  3. MULTI-SIGNAL GOAL (from v2): blend freq_mode + initial_obs + change_target.

  If systematic scan finds interactive zones that random warmup misses, then L1 rate
  improves on games where v2 got 0% (LF52, SC25).
  Falsified if: L1 rate same as v2 (interactive zones are not the bottleneck).

Includes transition-reset (shared with prosecution).

KILL: chain_score < v2 (no improvement)
SUCCESS: L1 > 0 on games where v2 got 0%
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
WARMUP_STEPS = 1500       # shorter warmup — systematic scan is more efficient
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
ZONE_RADIUS = 3
CF_CHANGE_THRESH = 0.1

# Stagnation detection
STAGNATION_WINDOW = 200   # check for stagnation every N steps
STAGNATION_THRESH = 0.01  # mean change_map delta below this = stagnating
EXPLORE_BOOST = 0.5       # exploration rate during stagnation

# R3 SPSA parameters — 4 parameters
SIGMA_INIT = 1.0
ALPHA_INIT = 0.999
W_INIT_INIT = 0.0
W_CHANGE_INIT = 0.0

SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_DELTA_W = 0.1
SPSA_LR = 0.02

# Action encoding
N_KB = 7

# Systematic scan grid: 16x16 = 256 positions covering the 64x64 space
SCAN_GRID = [(gx * 4 + 2, gy * 4 + 2) for gy in range(16) for gx in range(16)]
# Coarser grid for exploration after warmup
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x


def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)


class DefenseV3Substrate:
    """
    ℓ₁ R3 v3: systematic scan + adaptive exploration + multi-signal goal.

    Warmup: deterministic scan through 256 grid positions (4px spacing).
    Post-warmup: mismatch-guided targeting with stagnation detection.
    If stagnating (no change detected), temporarily boost exploration.
    Goal: blend of freq_mode + initial_obs + change_target, SPSA-tuned.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._initial_obs = None
        self._change_target = None
        self._change_count = np.zeros((64, 64), dtype=np.float32)
        # Systematic scan state
        self._scan_idx = 0
        self._kb_scan_idx = 0
        # Stagnation detection
        self._prev_change_sum = 0.0
        self._stagnating = False
        # R3 parameters (persist across games)
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.w_init = W_INIT_INIT
            self.w_change = W_CHANGE_INIT
            self.r3_updates = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._initial_obs = None
        self._change_target = None
        self._change_count = np.zeros((64, 64), dtype=np.float32)
        self._scan_idx = 0
        self._kb_scan_idx = 0
        self._prev_change_sum = 0.0
        self._stagnating = False

    def _goal(self, obs, sigma=None, alpha=None, w_init=None, w_change=None):
        """Multi-signal parametric goal."""
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha
        if w_init is None:
            w_init = self.w_init
        if w_change is None:
            w_change = self.w_change

        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c_idx in range(16):
            smoothed[:, :, c_idx] = gaussian_filter(
                self.freq_hist[:, :, c_idx].astype(np.float32), sigma=s)
        freq_goal = np.argmax(smoothed, axis=2).astype(np.float32)

        if self.novelty_map is not None:
            freq_goal = alpha * freq_goal + (1.0 - alpha) * self.novelty_map

        if self._initial_obs is None:
            return freq_goal

        change_goal = self._change_target if self._change_target is not None else self._initial_obs

        w_f = max(0.0, 1.0 - abs(w_init) - abs(w_change))
        w_i = max(0.0, min(1.0, abs(w_init)))
        w_c = max(0.0, min(1.0, abs(w_change)))
        total = w_f + w_i + w_c
        if total > 0:
            w_f /= total
            w_i /= total
            w_c /= total
        else:
            w_f = 1.0

        return w_f * freq_goal + w_i * self._initial_obs + w_c * change_goal

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        """SPSA gradient on 4 parameters from click outcome."""
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        # Update change_target
        diff = np.abs(obs_after - obs_before)
        changed_mask = diff > CF_CHANGE_THRESH
        if np.any(changed_mask):
            if self._change_target is None:
                self._change_target = obs_after.copy()
            else:
                self._change_target[changed_mask] = (
                    0.8 * self._change_target[changed_mask] +
                    0.2 * obs_after[changed_mask])
            self._change_count[changed_mask] += 1.0

        # SPSA for sigma
        s_plus = np.clip(self.sigma + SPSA_DELTA_SIGMA, 0.5, 16.0)
        s_minus = np.clip(self.sigma - SPSA_DELTA_SIGMA, 0.5, 16.0)
        if s_plus != s_minus:
            goal_sp = self._goal(obs_before, sigma=s_plus)
            goal_sm = self._goal(obs_before, sigma=s_minus)
            mm_sp = float(np.abs(obs_before[y0:y1, x0:x1] - goal_sp[y0:y1, x0:x1]).mean())
            mm_sm = float(np.abs(obs_before[y0:y1, x0:x1] - goal_sm[y0:y1, x0:x1]).mean())
            score_sp = mm_sp if zone_changed else -mm_sp
            score_sm = mm_sm if zone_changed else -mm_sm
            grad = (score_sp - score_sm) / (s_plus - s_minus)
            self.sigma = float(np.clip(self.sigma + SPSA_LR * grad, 0.5, 16.0))

        # SPSA for alpha
        a_plus = np.clip(self.alpha + SPSA_DELTA_ALPHA, 0.9, 0.999)
        a_minus = np.clip(self.alpha - SPSA_DELTA_ALPHA, 0.9, 0.999)
        if a_plus != a_minus:
            goal_ap = self._goal(obs_before, alpha=a_plus)
            goal_am = self._goal(obs_before, alpha=a_minus)
            mm_ap = float(np.abs(obs_before[y0:y1, x0:x1] - goal_ap[y0:y1, x0:x1]).mean())
            mm_am = float(np.abs(obs_before[y0:y1, x0:x1] - goal_am[y0:y1, x0:x1]).mean())
            score_ap = mm_ap if zone_changed else -mm_ap
            score_am = mm_am if zone_changed else -mm_am
            grad = (score_ap - score_am) / (a_plus - a_minus)
            self.alpha = float(np.clip(self.alpha + SPSA_LR * grad, 0.9, 0.999))

        # SPSA for w_init
        wi_plus = self.w_init + SPSA_DELTA_W
        wi_minus = self.w_init - SPSA_DELTA_W
        goal_wp = self._goal(obs_before, w_init=wi_plus)
        goal_wm = self._goal(obs_before, w_init=wi_minus)
        mm_wp = float(np.abs(obs_before[y0:y1, x0:x1] - goal_wp[y0:y1, x0:x1]).mean())
        mm_wm = float(np.abs(obs_before[y0:y1, x0:x1] - goal_wm[y0:y1, x0:x1]).mean())
        score_wp = mm_wp if zone_changed else -mm_wp
        score_wm = mm_wm if zone_changed else -mm_wm
        grad_wi = (score_wp - score_wm) / (2 * SPSA_DELTA_W)
        self.w_init = float(np.clip(self.w_init + SPSA_LR * grad_wi, -1.0, 1.0))

        # SPSA for w_change
        wc_plus = self.w_change + SPSA_DELTA_W
        wc_minus = self.w_change - SPSA_DELTA_W
        goal_cp = self._goal(obs_before, w_change=wc_plus)
        goal_cm = self._goal(obs_before, w_change=wc_minus)
        mm_cp = float(np.abs(obs_before[y0:y1, x0:x1] - goal_cp[y0:y1, x0:x1]).mean())
        mm_cm = float(np.abs(obs_before[y0:y1, x0:x1] - goal_cm[y0:y1, x0:x1]).mean())
        score_cp = mm_cp if zone_changed else -mm_cp
        score_cm = mm_cm if zone_changed else -mm_cm
        grad_wc = (score_cp - score_cm) / (2 * SPSA_DELTA_W)
        self.w_change = float(np.clip(self.w_change + SPSA_LR * grad_wc, -1.0, 1.0))

        self.r3_updates += 1

    def _warmup_action(self):
        """Systematic scan during warmup: cycle through grid + keyboard actions."""
        if self._supports_click:
            # Alternate: 3 systematic clicks, then 1 keyboard
            if self.step_count % 4 < 3:
                cx, cy = SCAN_GRID[self._scan_idx % len(SCAN_GRID)]
                self._scan_idx += 1
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            else:
                kb = self._kb_scan_idx % N_KB
                self._kb_scan_idx += 1
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
        else:
            # Keyboard-only: cycle through all actions
            kb = self._kb_scan_idx % N_KB
            self._kb_scan_idx += 1
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = kb
        return action

    def _check_stagnation(self):
        """Check if the substrate is stagnating (no new changes detected)."""
        if self.step_count % STAGNATION_WINDOW != 0:
            return
        current_sum = float(self.change_map.sum())
        delta = abs(current_sum - self._prev_change_sum)
        self._stagnating = delta < STAGNATION_THRESH * 64 * 64
        self._prev_change_sum = current_sum

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        obs_int = obs.astype(np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        if self._initial_obs is None:
            self._initial_obs = arr.copy()

        # R3 SPSA update from previous click
        if self._prev_obs_arr is not None and self._prev_action is not None:
            click_xy = _decode_click(self._prev_action)
            if click_xy is not None:
                cx, cy = click_xy
                self._r3_spsa_update(self._prev_obs_arr, arr, cx, cy)

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            action = self._warmup_action()
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        r_idx, c_idx = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r_idx, c_idx, obs_int] += 1
        alpha_nov = 0.999
        self.novelty_map = alpha_nov * self.novelty_map + (1 - alpha_nov) * arr

        # Multi-signal parametric goal
        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # Warmup: systematic scan
        if self.step_count < WARMUP_STEPS:
            action = self._warmup_action()
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # Stagnation check
        self._check_stagnation()

        # Post-warmup: score-based with adaptive exploration
        click_score = float(np.max(smoothed)) if self._supports_click else -1.0
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        # Adaptive epsilon: boost exploration during stagnation
        epsilon = EXPLORE_BOOST if self._stagnating else 0.1

        if self._rng.random() < epsilon:
            if self._supports_click and self._rng.random() < 0.5:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        if self._supports_click and click_score >= best_kb_score:
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            action = _click_action(int(x), int(y))
            y0 = max(0, y - SUPPRESS_RADIUS)
            y1 = min(64, y + SUPPRESS_RADIUS + 1)
            x0 = max(0, x - SUPPRESS_RADIUS)
            x1 = min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self.prev_action_type = 'click'
        else:
            action = best_kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = best_kb

        self._prev_obs_arr = arr.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        """Reset exploration state for new level."""
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.freq_hist[:] = 0
        self.novelty_map = None
        self.prev_obs = None
        self._initial_obs = None
        self._change_target = None
        self._change_count = np.zeros((64, 64), dtype=np.float32)
        self._scan_idx = 0
        self._kb_scan_idx = 0
        self._prev_change_sum = 0.0
        self._stagnating = False
        # Reset step_count so warmup re-runs on new level
        self.step_count = 0
        # Keep R3 parameters — learned priors carry forward
        # Keep kb_influence — keyboard effects shared


CONFIG = {
    "warmup": WARMUP_STEPS,
    "sigma_init": SIGMA_INIT,
    "alpha_init": ALPHA_INIT,
    "w_init_init": W_INIT_INIT,
    "w_change_init": W_CHANGE_INIT,
    "spsa_lr": SPSA_LR,
    "scan_grid_size": "16x16 = 256 positions",
    "stagnation_window": STAGNATION_WINDOW,
    "explore_boost": EXPLORE_BOOST,
    "v3_features": "systematic scan + adaptive exploration + warmup re-run on level transition",
}

SUBSTRATE_CLASS = DefenseV3Substrate
