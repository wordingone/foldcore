"""
sub{NNNN}_{name}.py — [One-line description]

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step {NNNN} --substrate experiments/sub{NNNN}_{name}.py

FAMILY: [family name]
R3 HYPOTHESIS: [What R3 prediction does this test? What would falsify it?]
  If [mechanism X], then [observable Y] because [R3 reason Z].
  Falsified if: [specific outcome].

KILL: chain_score < [threshold] (e.g. < baseline 994 chain_score)
SUCCESS: chain_score >= baseline AND no game degraded (chain_kill = PASS)
BUDGET: 10K steps/game, 10 seeds, all 5 phases (Split-CIFAR × 2, LS20, FT09, VC33)

Jun directive 2026-03-24:
- Substrate defines the class. Harness (run_experiment.py) is the constant.
- Game order randomized per seed — cannot be bypassed
- Results saved to chain_results/runs/ — enforced by harness
- chain_kill verdict: PASS / KILL / FAIL vs baseline_994.json
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
# All tunable constants here. ONE config for all games — no per-game branches.


# ─── Substrate ───
class MySubstrate:
    """
    [Describe mechanism here]

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int        # called every step
      on_level_transition()                  # called on game reset / level up
      set_game(n_actions: int)               # called when game switches
    NOTE: reset(seed) is optional — ChainRunner shim calls set_game() as fallback.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        # Initialize substrate state here
        self._n_actions = 4

    def set_game(self, n_actions: int):
        """Called on game switch. Reset per-game state, keep cross-game learned weights."""
        self._n_actions = n_actions
        # Reset per-game accumulators (delta_per_action, h, etc.)
        # DO NOT reset global learned matrices (W_pred, alpha, W_h, etc.)

    def process(self, obs: np.ndarray) -> int:
        """Process one observation, return action index."""
        obs = np.asarray(obs, dtype=np.float32)
        # Encode
        # Select action
        return int(self._rng.randint(0, self._n_actions))

    def on_level_transition(self):
        """Called on episode done or level completion."""
        pass


# Exposed as CONFIG dict — picked up by run_experiment.py for save_results()
CONFIG = {
    # "param_name": value,
}

# Explicit substrate declaration — required for run_experiment.py auto-discovery
SUBSTRATE_CLASS = MySubstrate
