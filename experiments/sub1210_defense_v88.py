"""
sub1210_defense_v88.py — Adaptive exploration v80 (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1210 --substrate experiments/sub1210_defense_v88.py

FAMILY: Adaptive exploration. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v80 (ℓ₁ ceiling at 3.3/5) uses fixed 100-step exploration
regardless of action space size. Games report 7-68 actions. With 7 actions,
100 steps gives ~14 samples/action — adequate. With 68 actions, 100 steps
gives ~1.5 samples/action — insufficient for reliable ranking.

v88 = v80 with adaptive exploration length:
- n_actions ≤ 10: 100 steps (standard v80)
- n_actions > 10: min(500, n_actions * 7) steps
  (enough for ~7 samples per action)

Also explores the FULL action space (including clicks), not just keyboard.
This tests whether v80's ceiling is due to under-sampling in large action
space games.

20% epsilon-greedy. Change-rate ranking. Cycling with switching.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.3/5 (no improvement over v80).
SUCCESS: avg L1 > 3.3/5 (adaptive exploration helps).
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
MIN_SAMPLES_PER_ACTION = 7


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class AdaptiveExplorationSubstrate:
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

        # Adaptive exploration length
        self._explore_steps = min(500, max(100, self._n_actions_env * MIN_SAMPLES_PER_ACTION))

        # Per-action change statistics (ALL actions, not just keyboard)
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

        # Rank ALL actions by change rate (highest first)
        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))

        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for _, a in action_avgs if _ > CHANGE_THRESH]

        if not self._ranked_actions:
            # Fallback: keyboard actions
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

        # Clamp index after rebuild (ranked_actions may have shrunk)
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

        # Record stats for previous action
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= self._explore_steps:
                self._transition_to_exploit()
            else:
                # Explore FULL action space (including clicks)
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # Periodically rebuild ranking
        if self.step_count % 500 == 0:
            self._transition_to_exploit()

        # === EXPLOIT: v80-style cycling ===

        # Epsilon-greedy from FULL action space
        if self._rng.random() < EPSILON:
            a = self._prev_action
            self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
            self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Track current action's change rate
        self._current_change_sum += delta
        self._current_hold_count += 1

        # Switch when change drops or patience exhausted
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
        # Keep ranked actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "min_samples_per_action": MIN_SAMPLES_PER_ACTION,
    "family": "adaptive exploration",
    "tag": "defense v88 (ℓ₁ adaptive v80: exploration length scales with action space size. 7 actions=100 steps, 68 actions=476 steps. Ranks ALL actions including clicks. Tests if v80's ceiling is due to under-sampling in large action spaces.)",
}

SUBSTRATE_CLASS = AdaptiveExplorationSubstrate
