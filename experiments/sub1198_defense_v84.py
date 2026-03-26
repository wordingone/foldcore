"""
sub1198_defense_v84.py — Null hypothesis: pure cycling (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1198 --substrate experiments/sub1198_defense_v84.py

FAMILY: Null hypothesis / diagnostic. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: This tests whether OBSERVATION PROCESSING helps at all.

22 experiments (v73-v83) of reactive ℓ₁ ALL average ~3.0/5 = random.
v80 (change-rate max) at 3.3/5 is the only mechanism marginally above.
Every refinement (fusion, coherence, pairs) degrades from v80 or random.

FUNDAMENTAL QUESTION: Is observation processing relevant for L1 on
unknown games? Or is pure action coverage sufficient?

v84 = PURE ACTION CYCLING. No observation processing. No change
detection. No ranking. No statistics. No exploration phase. Just
cycle through keyboard actions 0-6 in order, one step each, repeat.

Predictions:
- If v84 ≈ 3.0/5 (random): observation processing adds NOTHING to L1.
  All v73-v83 complexity was wasted. Defense should argue parsimony.
- If v84 > 3.0/5: systematic cycling > random. Deterministic coverage
  helps. Build on cycling, not reactive.
- If v84 < 3.0/5: randomness itself helps (breaks correlations,
  covers more state space). Observation processing may be wasted but
  stochastic exploration isn't.

This is the simplest possible ℓ₁ substrate. ZERO parameters.
ZERO observation processing. ZERO state. Pure deterministic cycling.

KILL: N/A — this is a diagnostic, not a mechanism to improve.
SUCCESS: Clear separation from random (either direction) informs
the entire defense strategy going forward.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7


class NullCyclingSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._step = 0
        self.r3_updates = 0
        self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._step = 0

    def process(self, obs: np.ndarray) -> int:
        # Pure cycling: 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, ...
        # No observation processing whatsoever.
        n_kb = min(self._n_actions_env, N_KB)
        action = self._step % n_kb
        self._step += 1
        return action

    def on_level_transition(self):
        # Reset cycling on level transition
        self._step = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "family": "null hypothesis",
    "tag": "defense v84 (ℓ₁ null: pure cycling 0-6, NO observation processing. Tests if obs processing helps at all. 22 experiments of reactive ℓ₁ ≈ random. Is observation processing irrelevant for L1?)",
}

SUBSTRATE_CLASS = NullCyclingSubstrate
