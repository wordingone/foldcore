"""
sub1216_defense_v90.py — Multi-resolution v80 (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1216 --substrate experiments/sub1216_defense_v90.py

FAMILY: Multi-resolution change detection. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v80's 0% wall might be an encoding artifact. avgpool4 (4x4
blocks → 256D) averages away sub-block changes. Some games might change at
pixel-level (64x64) or at coarse level (8x8) but not at 4x4.

v76 tested raw 4096D → 3.0/5 (= random). But v76 used a DIFFERENT mechanism,
not v80's recipe. This tests: does v80's recipe with MULTI-RESOLUTION change
detection break the ceiling?

Change detection uses MAX across three resolutions:
  - avgpool2: 32x32 = 1024D (fine — catches small changes)
  - avgpool4: 16x16 = 256D (standard — v80's resolution)
  - avgpool8: 8x8 = 64D (coarse — catches large-scale changes)

For each action during exploration, delta = max(delta_fine, delta_std, delta_coarse).
This ensures we catch changes at ANY spatial scale. The exploit phase uses
v80's exact recipe (cycling + epsilon) with multi-resolution rankings.

Process() still returns actions based on the 256D encoding (v80 standard),
but CHANGE DETECTION uses all three resolutions.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.3/5 (encoding not the bottleneck).
SUCCESS: avg L1 > 3.3/5 (multi-resolution catches what avgpool4 misses).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
EPSILON = 0.2
CHANGE_THRESH = 0.1
EXPLORE_STEPS = 100


def _obs_to_enc_multi(obs):
    """Multi-resolution encoding: fine (1024D), standard (256D), coarse (64D)."""
    # Fine: avgpool2 → 32x32 = 1024D
    fine = obs.reshape(32, 2, 32, 2).mean(axis=(1, 3)).ravel().astype(np.float32)
    # Standard: avgpool4 → 16x16 = 256D
    std = obs.reshape(16, 4, 16, 4).mean(axis=(1, 3)).ravel().astype(np.float32)
    # Coarse: avgpool8 → 8x8 = 64D
    coarse = obs.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel().astype(np.float32)
    return fine, std, coarse


class MultiResolutionV80Substrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_fine = None
        self._prev_std = None
        self._prev_coarse = None
        self._prev_action = 0
        self._exploring = True

        # Per-action change statistics (using max across resolutions)
        self._action_change_sum = {}
        self._action_change_count = {}

        # Exploit state
        self._ranked_actions = []
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _transition_to_exploit(self):
        self._exploring = False
        self.r3_updates += 1
        self.att_updates_total += 1

        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))

        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for _, a in action_avgs if _ > CHANGE_THRESH]

        if not self._ranked_actions:
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

        self._current_idx = self._current_idx % len(self._ranked_actions)

    def _multi_res_delta(self, fine, std, coarse):
        """Max change across three resolutions."""
        if self._prev_fine is None:
            return 0.0

        # Normalize each resolution's delta by its dimensionality
        d_fine = float(np.sum(np.abs(fine - self._prev_fine))) / 1024.0
        d_std = float(np.sum(np.abs(std - self._prev_std))) / 256.0
        d_coarse = float(np.sum(np.abs(coarse - self._prev_coarse))) / 64.0

        # Use max — catches change at ANY scale
        return max(d_fine, d_std, d_coarse)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        fine, std, coarse = _obs_to_enc_multi(obs)

        if self._prev_fine is None:
            self._prev_fine = fine.copy()
            self._prev_std = std.copy()
            self._prev_coarse = coarse.copy()
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Multi-resolution change detection
        delta = self._multi_res_delta(fine, std, coarse)

        # Record stats for previous action
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_fine = fine.copy()
                self._prev_std = std.copy()
                self._prev_coarse = coarse.copy()
                self._prev_action = action
                return action

        # Periodic rebuild
        if self.step_count % 500 == 0:
            self._transition_to_exploit()

        # === EXPLOIT: v80-style cycling ===
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_fine = fine.copy()
            self._prev_std = std.copy()
            self._prev_coarse = coarse.copy()
            self._prev_action = action
            return action

        self._current_change_sum += delta
        self._current_hold_count += 1

        current_avg = self._current_change_sum / max(self._current_hold_count, 1)
        if current_avg < CHANGE_THRESH and self._current_hold_count > 5:
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._current_change_sum = 0.0
            self._current_hold_count = 0
            self._patience = 0
        elif self._current_hold_count > 20:
            self._patience += 1
            if self._patience > 3:
                self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
                self._current_change_sum = 0.0
                self._current_hold_count = 0
                self._patience = 0

        action = self._ranked_actions[self._current_idx]
        self._prev_fine = fine.copy()
        self._prev_std = std.copy()
        self._prev_coarse = coarse.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_fine = None
        self._prev_std = None
        self._prev_coarse = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0


CONFIG = {
    "n_dims": 256,
    "block_size": 4,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "explore_steps": EXPLORE_STEPS,
    "resolutions": "fine(1024D) + standard(256D) + coarse(64D)",
    "family": "multi-resolution change detection",
    "tag": "defense v90 (ℓ₁ multi-resolution v80: change detection uses max(avgpool2, avgpool4, avgpool8). Catches changes at ANY spatial scale. Tests if 0% wall is encoding artifact — games changing at sub-block or coarse level that avgpool4 misses.)",
}

SUBSTRATE_CLASS = MultiResolutionV80Substrate
