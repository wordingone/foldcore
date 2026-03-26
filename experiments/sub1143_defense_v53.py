"""
sub1143_defense_v53.py — Temporal difference encoding + reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1143 --substrate experiments/sub1143_defense_v53.py

FAMILY: Temporal difference encoding (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: avgpool4 encodes ABSOLUTE brightness. If a game responds to
actions with SMALL pixel changes (1-5 pixel value shift), the absolute encoding
dominates and the change is invisible. Temporal differencing (enc_t - enc_{t-1})
zeros out the static background and amplifies the signal from action effects.

If the substrate can detect changes via temporal differencing that avgpool4 misses
→ the 0% wall was sensitivity, not perception. The substrate modifies its
action selection based on which actions produce the LARGEST temporal differences
(R3: self-modifying action policy based on discovered responsiveness).

Architecture:
- raw_enc = avgpool4 (256D) — same as v30
- diff_enc = |raw_enc_t - raw_enc_{t-1}| — temporal difference (256D)
- combined_enc = [raw_enc, diff_enc] — 512D (raw for goal, diff for action selection)
- Goal: distance-to-initial on RAW encoding (same as v30)
- Action selection: reactive switching, but PROGRESS is measured by temporal
  difference magnitude (did this action cause visible change?)
- If diff magnitude > threshold → action is effective, stick with it
- If diff magnitude ≈ 0 → action has no visible effect, switch
- Same saliency click regions (16 blocks from raw encoding)
- ZERO learned parameters (defense: ℓ₁)

CONTROLLED COMPARISON: vs v30 (same encoding, different progress signal).
v30: progress = distance-to-initial DECREASING
v53: progress = temporal difference INCREASING (action caused visible change)

KILL: ARC ≤ v30 (0.3319) AND no improvement on 0% games.
SUCCESS: Temporal differencing detects action effects that distance-to-initial misses.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
DIFF_THRESH = 0.1  # minimum temporal diff to count as "action had effect"


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class TemporalDiffSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_diff_mag = 0.0

        # Click regions
        self._click_regions = []
        self._click_actions = []
        self._n_active = N_KB
        self._regions_set = False

        # Reactive switching — driven by temporal difference
        self._current_action_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _discover_regions(self, enc):
        """Find top-16 salient blocks for click targeting."""
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        self._click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in self._click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _action_idx_to_env_action(self, idx):
        """Convert internal index to PRISM action."""
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(enc)
            self._current_action_idx = self._rng.randint(min(self._n_active, N_KB))
            return self._action_idx_to_env_action(self._current_action_idx)

        # Temporal difference: magnitude of frame-to-frame change
        diff = np.abs(enc - self._prev_enc)
        diff_mag = np.sum(diff)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action_idx = self.step_count % self._n_active
            self._current_action_idx = action_idx
            self._prev_enc = enc.copy()
            self._prev_diff_mag = diff_mag
            return self._action_idx_to_env_action(action_idx)

        # Reactive switching: progress = action caused visible change
        # "progress" = temporal difference is above threshold (action had effect)
        progress = diff_mag > DIFF_THRESH
        no_change = diff_mag < 1e-8

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3
                if self._actions_tried_this_round >= self._n_active:
                    self._current_action_idx = self._rng.randint(self._n_active)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action_idx = (self._current_action_idx + 1) % self._n_active

        self._prev_enc = enc.copy()
        self._prev_diff_mag = diff_mag
        return self._action_idx_to_env_action(self._current_action_idx)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_diff_mag = 0.0
        self._current_action_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._regions_set = False


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "diff_thresh": DIFF_THRESH,
    "family": "temporal difference encoding",
    "tag": "defense v53 (ℓ₁ temporal diff: progress = |enc_t - enc_{t-1}| > thresh, action caused visible change)",
}

SUBSTRATE_CLASS = TemporalDiffSubstrate
