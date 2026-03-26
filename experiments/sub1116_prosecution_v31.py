"""
sub1116_prosecution_v31.py — Bidirectional progress discovery (ℓ_π goal function)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1116 --substrate experiments/sub1116_prosecution_v31.py

FAMILY: Learned goal function (NEW prosecution axis). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: The substrate discovers whether "toward initial" or "away from
initial" is progress for each action. The goal function self-modifies from
interaction. ℓ_π at the objective level, not just encoding.

On toward-initial games: discovers positive direction → behaves like v30.
On away-from-initial games: discovers negative direction → INVERTS the metric.

KILL: ARC ≤ v30 defense (0.33).
SUCCESS: ANY previously-0% game shows L1 > 0.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
DIRECTION_DECAY = 0.95
DIRECTION_THRESH = 0.3  # confidence threshold for direction discovery
CHANGE_MIN = 0.01  # minimum change to count as "action did something"


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class BidirectionalProgressSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        # Per-action direction tracking (ℓ_π)
        self._action_direction = np.zeros(N_KB, dtype=np.float32)

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)
        change_mag = float(np.sum(np.abs(enc - self._prev_enc)))

        # Update direction tracking for previous action
        if self._prev_enc is not None and change_mag > CHANGE_MIN:
            delta_dist = self._prev_dist - dist  # positive = moved toward initial
            direction_signal = 1.0 if delta_dist > 0 else -1.0
            self._action_direction[self._current_action] = (
                DIRECTION_DECAY * self._action_direction[self._current_action] +
                (1 - DIRECTION_DECAY) * direction_signal
            )
            self.r3_updates += 1
            self.att_updates_total += 1

        # Explore phase
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Bidirectional progress check
        ad = self._action_direction[self._current_action]
        if abs(ad) > DIRECTION_THRESH:
            # Direction discovered for this action
            if ad > 0:
                # Toward-initial is progress
                progress = (self._prev_dist - dist) > 1e-4
            else:
                # Away-from-initial is progress
                progress = (dist - self._prev_dist) > 1e-4
        else:
            # Direction unknown — any change is progress
            progress = change_mag > CHANGE_MIN

        no_change = change_mag < 1e-6

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

                if self._actions_tried_this_round >= self._n_actions:
                    self._current_action = self._rng.randint(self._n_actions)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % self._n_actions

        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep action_direction across levels (ℓ_π cross-level transfer)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "direction_decay": DIRECTION_DECAY,
    "direction_thresh": DIRECTION_THRESH,
    "change_min": CHANGE_MIN,
    "family": "learned goal function",
    "tag": "prosecution v31 (ℓ_π bidirectional progress — learned per-action direction)",
}

SUBSTRATE_CLASS = BidirectionalProgressSubstrate
