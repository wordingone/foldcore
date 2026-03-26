"""
sub1124_defense_v41.py — Raw pixel + bidirectional brute force (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1124 --substrate experiments/sub1124_defense_v41.py

FAMILY: Reactive action switching (raw pixel + bidirectional)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v40 (raw pixels) solved GAME_2 fully (ARC=0.4082) — avgpool4
hides pixel-level signal. v39 (bidirectional) showed v31's 3/5 pattern can
sometimes be matched without learning. Combining both: raw pixel distance
with bidirectional brute force (try toward AND away directions per action).

TWO CHANGES FROM v30:
1. Distance on raw 4096D pixels, not avgpool4 256D (from v40)
2. Try both toward/away directions per action (from v39)

Zero learned parameters. Pure ℓ₁.

KILL: ARC < v40 (0.0816) — bidirectional adds noise to raw pixels.
SUCCESS: ARC > v40 — stacking both insights helps.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20


class RawPixelBidirectionalSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._obs_0 = None
        self._prev_obs = None
        self._prev_dist = None
        self._current_action = 0
        self._current_direction = 1  # +1 = toward, -1 = away
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _dist_to_initial(self, obs):
        if self._obs_0 is None:
            return 0.0
        return float(np.sum(np.abs(obs - self._obs_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1

        if self._obs_0 is None:
            self._obs_0 = obs.copy()
            self._prev_obs = obs.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(obs)

        # Explore phase
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_obs = obs.copy()
            self._prev_dist = dist
            return action

        # Bidirectional progress check
        if self._current_direction > 0:
            progress = (self._prev_dist - dist) > 1e-4
        else:
            progress = (dist - self._prev_dist) > 1e-4

        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0
            if self._steps_on_action >= self._patience or no_change:
                self._steps_on_action = 0
                self._patience = 3

                if self._current_direction > 0:
                    # Tried toward — try away with same action
                    self._current_direction = -1
                else:
                    # Tried both — next action
                    self._current_direction = 1
                    self._actions_tried_this_round += 1

                    if self._actions_tried_this_round >= self._n_actions:
                        self._current_action = self._rng.randint(self._n_actions)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_action = (self._current_action + 1) % self._n_actions

        self._prev_obs = obs.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._obs_0 = None
        self._prev_obs = None
        self._prev_dist = None
        self._current_action = 0
        self._current_direction = 1
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_kb": N_KB,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v41 (ℓ₁ raw pixel 4096D + bidirectional brute force)",
}

SUBSTRATE_CLASS = RawPixelBidirectionalSubstrate
