"""
sub1040_defense_prism.py — Defense R3 substrate (SPSA parametric goal) for PRISM.

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1040 --substrate experiments/sub1040_defense_prism.py

FAMILY: parametric-goal
R3 HYPOTHESIS: ℓ₁ R3 (SPSA on sigma/alpha) produces game-specific parameter
  differentiation across diverse unknown games using one config.
  If SPSA self-modification, then sigma/alpha converge to different values per game
  because different games have different spatial/temporal structure.
  Falsified if: parameters stay at init OR converge to same values across games.

KILL: chain_score < baseline OR parameters don't move
SUCCESS: chain_score >= baseline AND parameters differentiate per game

Port of Step 1038 (direct API) to PRISM template. Action encoding:
  action 0-6:  keyboard ACTION1-ACTION7
  action 7+:   click at pixel (x, y) where click_idx = action - 7,
               x = click_idx % 64, y = click_idx // 64

BUDGET: 10K steps/game, 5 seeds (debate), 3 games (FT09, LS20, VC33)
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
WARMUP_STEPS = 3000
ALPHA_CHANGE = 0.99      # EMA decay for change_map
KERNEL = 5               # uniform_filter kernel for mismatch smoothing
SUPPRESS_RADIUS = 3      # click suppression radius
SUPPRESS_DURATION = 8    # click suppression steps
ZONE_RADIUS = 3          # R3 evaluation zone
CF_CHANGE_THRESH = 0.1   # counterfactual change threshold

# R3 SPSA parameters
SIGMA_INIT = 1.0         # gaussian smoothing scale
ALPHA_INIT = 0.999       # temporal blending weight
SPSA_DELTA_SIGMA = 1.0   # perturbation magnitude for sigma
SPSA_DELTA_ALPHA = 0.01  # perturbation magnitude for alpha
SPSA_LR = 0.02           # SPSA learning rate

# Action encoding
N_KB = 7                 # keyboard actions (ACTION1-ACTION7)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    """Encode click at pixel (x, y) as PRISM action integer."""
    return N_KB + y * 64 + x


def _decode_click(action):
    """Decode PRISM action integer to (x, y) pixel coords. Returns None if keyboard."""
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)


# ─── Substrate ───
class DefenseR3Substrate:
    """
    ℓ₁ R3: parametric goal with SPSA self-modification of sigma (spatial) and alpha (temporal).

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        """Initialize/reset all per-game state."""
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None   # stored for R3 evaluation
        self._prev_action = None    # stored for R3 evaluation
        # R3 parameters (persist across games — self-modification accumulates)
        # Only reset on first call
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.r3_updates = 0

    def set_game(self, n_actions: int):
        """Called on game switch. Reset per-game state, keep R3 parameters."""
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        # Reset per-game accumulators
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

    def _goal(self, obs, sigma=None, alpha=None):
        """Parametric goal: gaussian-smoothed freq mode + temporal blending."""
        if sigma is None:
            sigma = self.sigma
        if alpha is None:
            alpha = self.alpha

        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c in range(16):
            smoothed[:, :, c] = gaussian_filter(
                self.freq_hist[:, :, c].astype(np.float32), sigma=s)
        spatial_goal = np.argmax(smoothed, axis=2).astype(np.float32)

        if self.novelty_map is not None:
            return alpha * spatial_goal + (1.0 - alpha) * self.novelty_map
        return spatial_goal

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        """SPSA gradient on sigma and alpha from click outcome."""
        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        # SPSA for sigma
        s_plus = np.clip(self.sigma + SPSA_DELTA_SIGMA, 0.5, 16.0)
        s_minus = np.clip(self.sigma - SPSA_DELTA_SIGMA, 0.5, 16.0)
        if s_plus != s_minus:
            goal_sp = self._goal(obs_before, sigma=s_plus)
            goal_sm = self._goal(obs_before, sigma=s_minus)
            mm_sp = float(np.abs(obs_before[y0:y1, x0:x1] - goal_sp[y0:y1, x0:x1]).mean())
            mm_sm = float(np.abs(obs_before[y0:y1, x0:x1] - goal_sm[y0:y1, x0:x1]).mean())
            if zone_changed:
                score_sp, score_sm = mm_sp, mm_sm
            else:
                score_sp, score_sm = -mm_sp, -mm_sm
            grad_s = (score_sp - score_sm) / (s_plus - s_minus)
            self.sigma = float(np.clip(self.sigma + SPSA_LR * grad_s, 0.5, 16.0))

        # SPSA for alpha
        a_plus = np.clip(self.alpha + SPSA_DELTA_ALPHA, 0.9, 0.999)
        a_minus = np.clip(self.alpha - SPSA_DELTA_ALPHA, 0.9, 0.999)
        if a_plus != a_minus:
            goal_ap = self._goal(obs_before, alpha=a_plus)
            goal_am = self._goal(obs_before, alpha=a_minus)
            mm_ap = float(np.abs(obs_before[y0:y1, x0:x1] - goal_ap[y0:y1, x0:x1]).mean())
            mm_am = float(np.abs(obs_before[y0:y1, x0:x1] - goal_am[y0:y1, x0:x1]).mean())
            if zone_changed:
                score_ap, score_am = mm_ap, mm_am
            else:
                score_ap, score_am = -mm_ap, -mm_am
            grad_a = (score_ap - score_am) / (a_plus - a_minus)
            self.alpha = float(np.clip(self.alpha + SPSA_LR * grad_a, 0.9, 0.999))

        self.r3_updates += 1

    def process(self, obs: np.ndarray) -> int:
        """Process one observation, return PRISM-encoded action."""
        obs = np.asarray(obs, dtype=np.float32)
        # Handle different obs formats:
        # ARC games: (1, 64, 64) or (64, 64) — 2D grid, values 0-15
        # CIFAR: (32, 32, 3) — RGB image, values 0-1
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]  # (1, 64, 64) → (64, 64)
        if obs.shape != (64, 64):
            # Non-ARC observation (CIFAR, etc.) — random action
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        obs_int = obs.astype(np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

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
            if self._supports_click:
                action = _click_action(32, 32)
            else:
                action = 0
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

        # Parametric goal
        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # Warmup: random
        if self.step_count < WARMUP_STEPS:
            if self._supports_click and self._rng.random() < 0.9:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # Post-warmup: score-based
        click_score = float(np.max(smoothed)) if self._supports_click else -1.0
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        # Epsilon-greedy exploration
        if self._rng.random() < 0.1:
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
        """Called on level-up or game reset."""
        pass


CONFIG = {
    "warmup": WARMUP_STEPS,
    "sigma_init": SIGMA_INIT,
    "alpha_init": ALPHA_INIT,
    "spsa_lr": SPSA_LR,
    "spsa_delta_sigma": SPSA_DELTA_SIGMA,
    "spsa_delta_alpha": SPSA_DELTA_ALPHA,
}

SUBSTRATE_CLASS = DefenseR3Substrate
