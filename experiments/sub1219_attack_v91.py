"""
sub1219_attack_v91.py — Minimal ℓ_π baseline (ATTACK experiment)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1219 --substrate experiments/sub1219_attack_v91.py

FAMILY: Minimal learned parameter. Tagged: ATTACK (tests prosecution claim).
R3 HYPOTHESIS: Prosecution claims ℓ_π > ℓ₁. Defense has proven v80 = ℓ₁
ceiling at 3.3/5 (49 experiments, 5 independent proofs). This ATTACK experiment
tests: does the SIMPLEST possible ℓ_π substrate beat v80?

v91 = v80 recipe + ONE learned parameter: per-action EXPONENTIAL MOVING
AVERAGE of change rate (instead of v80's cumulative mean). EMA adapts to
the CURRENT game state — recent observations weighted more than old ones.

This is the MINIMAL step from ℓ₁ to ℓ_π:
  - ℓ₁ (v80): cumulative mean change rate per action (global, time-invariant)
  - ℓ_π (v91): EMA change rate per action (local, time-variant, α=0.1 learned)

If v91 ≤ v80: the minimal ℓ_π step adds nothing. Prosecution needs
SUBSTANTIALLY more complexity to beat ℓ₁. Defense wins on parsimony.
If v91 > v80: even minimal learning helps. Prosecution has a point.

NOT shared architecture with prosecution (they use attention buffers,
forward models). This is a minimal baseline for the ℓ₁→ℓ_π transition.

KILL: avg L1 ≤ 3.3/5 (minimal ℓ_π = no improvement).
SUCCESS: avg L1 > 3.3/5 (learning helps even minimally).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EPSILON = 0.2
CHANGE_THRESH = 0.1
EXPLORE_STEPS = 100
EMA_ALPHA = 0.1  # the ONE learned parameter: recency weighting


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class MinimalLpiSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._exploring = True

        # Per-action EMA change rate (the ℓ_π part)
        self._action_ema = {}
        # Also keep cumulative stats for explore phase ranking
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

        # Rank by EMA change rate (adapts to current state)
        action_avgs = []
        for a in self._action_ema:
            action_avgs.append((self._action_ema[a], a))

        # Fallback to cumulative stats if EMA is empty
        if not action_avgs:
            for a, total in self._action_change_sum.items():
                count = self._action_change_count.get(a, 1)
                action_avgs.append((total / count, a))

        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for _, a in action_avgs if _ > CHANGE_THRESH]

        if not self._ranked_actions:
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

        self._current_idx = self._current_idx % len(self._ranked_actions)

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

        # Measure change
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Update BOTH cumulative and EMA stats
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # EMA update (the ℓ_π part)
        if a in self._action_ema:
            self._action_ema[a] = (1 - EMA_ALPHA) * self._action_ema[a] + EMA_ALPHA * delta
        else:
            self._action_ema[a] = delta

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # Periodic rebuild using EMA (adapts rankings to current state)
        if self.step_count % 500 == 0:
            self._transition_to_exploit()

        # === EXPLOIT: v80-style cycling with EMA-based rankings ===
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
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
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0
        # Keep EMA stats across levels (this IS the learning)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "explore_steps": EXPLORE_STEPS,
    "ema_alpha": EMA_ALPHA,
    "family": "minimal ℓ_π baseline",
    "tag": "ATTACK v91 (minimal ℓ_π: v80 recipe but EMA change rate instead of cumulative mean. ONE learned parameter (α=0.1). Tests if the simplest ℓ₁→ℓ_π step helps. If not, prosecution needs substantially more complexity.)",
}

SUBSTRATE_CLASS = MinimalLpiSubstrate
