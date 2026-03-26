"""
sub1222_defense_v92.py — Immediate reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1222 --substrate experiments/sub1222_defense_v92.py

FAMILY: Immediate reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v80-v91 all use per-action STATISTICS (cumulative or EMA).
800b theorem says global stats have SNR ≤ 1/|N_a|. What if we abandon
statistics entirely and use ONLY the immediate observation?

v92 = purely reactive: repeat the current action if it caused change above
threshold. Switch to random if it didn't. No exploration phase, no ranking,
no cycling. Just: "did the last action work? yes → repeat. no → try random."

This is the SIMPLEST possible reactive substrate:
  - 1 parameter: change threshold (0.5)
  - 0 statistics, 0 memory, 0 ranking
  - Decision based purely on previous step's delta

If v92 ≥ v80: statistics are overhead, immediate reaction is sufficient.
If v92 < v80 but > random: statistics add marginal value.
If v92 ≈ random: immediate observation is insufficient without accumulation.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (= random, immediate reactive insufficient).
SUCCESS: avg L1 > 3.3/5 (beats v80, statistics were overhead).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

SWITCH_THRESH = 0.5  # if delta < this, switch action
EPSILON = 0.1  # small random chance even when current action works


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class ImmediateReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._consecutive_low = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Measure change from previous action
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Pure reactive logic:
        # If last action caused significant change → repeat it (with small epsilon)
        # If not → switch to random
        if delta >= SWITCH_THRESH:
            self._consecutive_low = 0
            if self._rng.random() < EPSILON:
                action = int(self._rng.randint(0, self._n_actions_env))
            else:
                action = self._prev_action  # repeat what worked
        else:
            self._consecutive_low += 1
            # Random action
            action = int(self._rng.randint(0, self._n_actions_env))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._consecutive_low = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "switch_thresh": SWITCH_THRESH,
    "epsilon": EPSILON,
    "family": "immediate reactive",
    "tag": "defense v92 (ℓ₁ immediate reactive: zero statistics. Repeat action if it caused change, random otherwise. Tests if per-action statistics add value or just overhead. Simplest possible reactive substrate.)",
}

SUBSTRATE_CLASS = ImmediateReactiveSubstrate
