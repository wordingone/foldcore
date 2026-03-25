"""
Step 1036b — Action-Influence Substrate v2 (Debate v3: Prosecution, iteration 2)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1036b --substrate experiments/step1036b_action_influence_freq.py

FAMILY: action-influence (v2 — freq histogram target)
R3 HYPOTHESIS: Same as 1036. Additionally: frequency histogram (running mode) is
  sufficient for goal state inference in discrete-color games, replacing majority-vote
  from the scripted pipeline.
  Falsified if: freq_map target doesn't distinguish correct/incorrect zone states
  (both have similar frequencies due to random clicking).

FIXES from 1036 (per Eli's defense objections):
  1. target = running MODE (frequency histogram) instead of running MEAN
     - freq_map[y,x,c] counts how often pixel (y,x) had color c
     - target[y,x] = argmax(freq_map[y,x,:]) = most common color
     - mismatch = (obs != target) — discrete, not continuous
  2. freq_map resets on level transition (fresh statistics for new layout)
  3. Recency suppression: after clicking (cx,cy), suppress 5x5 region for N steps

KILL: same as 1036 (L1 < 1 on all games)
SUCCESS: VC33 L2 5/5 + FT09 L1 5/5 (matches scripted pipeline)
BUDGET: 10K steps/game, 5 seeds, all PRISM phases

HARNESS: same as 1036 — needs ACTION6 coordinate support via _click_xy attribute.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

# ─── Hyperparameters ───
ALPHA_CHANGE = 0.98       # change_map decay (slightly faster than 1036)
ALPHA_INFLUENCE = 0.1     # influence map learning rate
WARMUP_STEPS = 150        # random exploration before scoring
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 1500
CLICK_THRESHOLD = 0.005   # min signal to trigger click action
SMOOTH_KERNEL = 5
N_COLORS = 16             # arc_agi color palette (0-15)
SUPPRESS_RADIUS = 3       # recency suppression radius (pixels)
SUPPRESS_DURATION = 8     # steps to suppress after clicking a region


def _spatial_smooth(arr, k=SMOOTH_KERNEL):
    """Fast spatial averaging via cumsum (box filter)."""
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, pad, mode='edge')
    cs = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    h, w = arr.shape
    out = cs[k:h+k, k:w+k] - cs[:h, k:w+k] - cs[k:h+k, :w] + cs[:h, :w]
    return out / (k * k)


class ActionInfluenceSubstrateV2:
    """
    Atomic substrate v2: frequency histogram replaces running mean for target.

    Key change from v1: target is running MODE (most frequent color per pixel),
    not running MEAN. For discrete-color games (0-15 palette), mode correctly
    identifies "normal" state while mean gives meaningless intermediate values.

    State:
      change_map (64x64): per-pixel running change frequency
      freq_map (64x64x16): per-pixel color frequency histogram
      influence (7x64x64): per-action influence map
      suppress_map (64x64): recency suppression countdown
      prev_obs (64x64 int): previous frame
      prev_action (int): last action taken

    Action selection:
      target = argmax(freq_map, axis=2)  [most common color per pixel]
      mismatch = (obs != target).astype(float)  [binary: wrong or right]
      signal = change_map * mismatch  [interactive AND wrong]
      For each action a: score[a] = sum(influence[a] * signal * (1 - suppress_active))
      If best = ACTION6: click at argmax(signal), else return direction
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 7
        self._step = 0
        self._init_maps()

    def _init_maps(self):
        """Initialize all maps."""
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.freq_map = np.zeros((64, 64, N_COLORS), dtype=np.int32)
        self.influence = np.zeros((7, 64, 64), dtype=np.float32)
        self.suppress_map = np.zeros((64, 64), dtype=np.int32)  # countdown
        self.prev_obs = None
        self.prev_action = None
        self._click_xy = (32, 32)
        self._step = 0

    def set_game(self, n_actions: int):
        """Called on game switch. Full reset."""
        self._n_actions = min(n_actions, 7)
        self._init_maps()

    def _update_freq(self, obs):
        """Update frequency histogram for each pixel."""
        obs_int = np.clip(obs.astype(np.int32), 0, N_COLORS - 1)
        rows = np.arange(64)[:, None]
        cols = np.arange(64)[None, :]
        self.freq_map[rows, cols, obs_int] += 1

    def _get_target(self):
        """Target = most common color per pixel."""
        return np.argmax(self.freq_map, axis=2).astype(np.float32)

    def process(self, obs: np.ndarray) -> int:
        """Process one observation, return action index."""
        obs = np.asarray(obs, dtype=np.float32)

        # Normalize observation to 64x64
        if obs.ndim == 3:
            obs = obs[0] if obs.shape[0] < obs.shape[-1] else obs[:, :, 0]
        if obs.ndim != 2 or obs.shape != (64, 64):
            obs = obs.ravel()[:4096].reshape(64, 64) if obs.size >= 4096 else np.zeros((64, 64))

        # ─── Update statistics ───
        self._update_freq(obs)

        if self.prev_obs is not None:
            diff = np.abs(obs - self.prev_obs)
            self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

            if self.prev_action is not None and self.prev_action < self._n_actions:
                a = self.prev_action
                self.influence[a] = (
                    (1 - ALPHA_INFLUENCE) * self.influence[a]
                    + ALPHA_INFLUENCE * diff
                )

        # Decrement suppression counters
        self.suppress_map = np.maximum(0, self.suppress_map - 1)

        # ─── Compute signal ───
        target = self._get_target()
        mismatch = (obs != target).astype(np.float32)

        # Suppress recently-acted regions
        active_mask = (self.suppress_map == 0).astype(np.float32)

        signal = self.change_map * mismatch * active_mask
        smooth_signal = _spatial_smooth(signal)

        self.prev_obs = obs.copy()
        self._step += 1

        # ─── Epsilon exploration ───
        epsilon = max(EPSILON_END,
                      EPSILON_START - (EPSILON_START - EPSILON_END) * self._step / EPSILON_DECAY)

        if self._step < WARMUP_STEPS or self._rng.random() < epsilon:
            action = self._rng.randint(0, self._n_actions)
            if action == 5:
                cx, cy = self._rng.randint(0, 64), self._rng.randint(0, 64)
                self._click_xy = (int(cx), int(cy))
                self._apply_suppression(int(cx), int(cy))
            self.prev_action = action
            return action

        # ─── Score actions ───
        scores = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            scores[a] = np.sum(self.influence[a] * smooth_signal)

        best_action = int(np.argmax(scores))

        # For click action: target the highest-signal unsuppressed pixel
        if best_action == 5 or (scores.max() < CLICK_THRESHOLD and smooth_signal.max() > 0):
            flat_idx = np.argmax(smooth_signal)
            cy, cx = np.unravel_index(flat_idx, (64, 64))
            self._click_xy = (int(cx), int(cy))
            self._apply_suppression(int(cx), int(cy))
            best_action = 5

        self.prev_action = best_action
        return best_action

    def _apply_suppression(self, cx, cy):
        """Suppress region around (cx, cy) for SUPPRESS_DURATION steps."""
        r = SUPPRESS_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)
        self.suppress_map[y0:y1, x0:x1] = SUPPRESS_DURATION

    def on_level_transition(self):
        """Reset frequency histogram and suppression on level transition."""
        self.freq_map = np.zeros((64, 64, N_COLORS), dtype=np.int32)
        self.suppress_map = np.zeros((64, 64), dtype=np.int32)
        # Keep change_map and influence — these represent learned knowledge
        # about which pixels are interactive and which actions do what


CONFIG = {
    "alpha_change": ALPHA_CHANGE,
    "alpha_influence": ALPHA_INFLUENCE,
    "warmup_steps": WARMUP_STEPS,
    "epsilon_start": EPSILON_START,
    "epsilon_end": EPSILON_END,
    "epsilon_decay": EPSILON_DECAY,
    "click_threshold": CLICK_THRESHOLD,
    "smooth_kernel": SMOOTH_KERNEL,
    "n_colors": N_COLORS,
    "suppress_radius": SUPPRESS_RADIUS,
    "suppress_duration": SUPPRESS_DURATION,
    "family": "action-influence-v2",
    "debate": "prosecution",
    "fixes": "freq_target, level_reset, recency_suppression",
}

SUBSTRATE_CLASS = ActionInfluenceSubstrateV2
