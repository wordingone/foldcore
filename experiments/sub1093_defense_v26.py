"""
sub1093_defense_v26.py — Difference-frame reactive switching

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1093 --substrate experiments/sub1093_defense_v26.py

FAMILY: Difference-frame reactive (defense-only, new encoding basis)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Encoding the DIFFERENCE between consecutive observations
(not raw observations) improves reactive switching by isolating the signal
(what changed) from the noise (static background). If fixed difference-
encoding works, prosecution's learned alpha (which also concentrates on
changing dims) is unnecessary — the signal IS the difference.

ALL PREVIOUS DEFENSE EXPERIMENTS used the same encoding (avgpool8 64D raw).
This is the first to change the ENCODING BASIS while keeping reactive logic.

ARCHITECTURE:
- avgpool8 (64D) raw encoding — same pooling as v21
- Difference frame: diff_enc = enc_current - enc_previous (what changed)
- Distance-to-initial DIFFERENCE: compare current diff_enc to initial diff_enc
  - Initial diff is all zeros (first frame, nothing changed yet)
  - Progress = diff_enc magnitude INCREASING (actions cause more change)
  - OR diff_enc magnitude DECREASING back toward zero (changes stabilizing)
- Reactive switching: same as v21 but measuring change-magnitude gradient
- Zero learned parameters

WHY DIFFERENT FROM PROSECUTION:
- No alpha, no W_pred, no forward model, no attention
- Fixed encoding (difference frame), not learned

WHY DIFFERENT FROM v21-v25:
- v21: raw encoding, distance-to-initial
- v26: difference encoding, change-magnitude gradient
- Different information: v21 asks "how far from start?" v26 asks "how much is changing?"

KILL: 0/3 ARC games.
SUCCESS: any ARC > 0.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64
N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
CHANGE_THRESH = 1e-4


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class DifferenceFrameReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None          # initial raw encoding
        self._prev_enc = None       # previous raw encoding
        self._prev_change_mag = 0.0 # previous change magnitude
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._max_change_mag = 0.0  # best change magnitude seen

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

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
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        # Difference frame: what changed since last step?
        diff = enc - self._prev_enc
        change_mag = float(np.sum(np.abs(diff)))

        # Also track distance to initial (dual criterion like v24)
        dist_to_initial = float(np.sum(np.abs(enc - self._enc_0)))

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            self._prev_enc = enc.copy()
            self._prev_change_mag = change_mag
            self._max_change_mag = max(self._max_change_mag, change_mag)
            return self.step_count % self._n_actions

        # Dual progress detection:
        # 1. Change gradient: this action causes MORE change than before
        change_increasing = (change_mag - self._prev_change_mag) > CHANGE_THRESH
        # 2. New territory: change magnitude exceeds previous maximum
        new_territory = change_mag > self._max_change_mag + CHANGE_THRESH

        progress = change_increasing or new_territory
        no_change = change_mag < CHANGE_THRESH  # action did nothing

        self._max_change_mag = max(self._max_change_mag, change_mag)
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
        self._prev_change_mag = change_mag
        return self._current_action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_change_mag = 0.0
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._max_change_mag = 0.0


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "difference-frame reactive",
    "tag": "defense v26 (ℓ₁ difference-frame reactive, change-magnitude gradient)",
}

SUBSTRATE_CLASS = DifferenceFrameReactiveSubstrate
