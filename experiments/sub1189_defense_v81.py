"""
sub1189_defense_v81.py — Dual-signal reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1189 --substrate experiments/sub1189_defense_v81.py

FAMILY: Dual-signal reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v30 (minimize distance) gets 2/5 with high ARC. v80
(maximize change) gets 3.3/5 with moderate ARC. They find DIFFERENT
games — v30 finds distance-convergent games, v80 finds change-responsive
games.

v81 FUSES both signals:
- Action RANKING from v80: prioritize actions by change rate (which
  actions cause the most effect?)
- Action SWITCHING from v30: switch when distance-to-initial stops
  decreasing (are we making progress toward a goal?)

The fusion: try the highest-change-rate action first. If distance-to-initial
decreases, HOLD (v30 says progress). If not, try the NEXT highest-change-rate
action. This covers both game types:
- Distance-convergent games: v30 logic holds the right action
- Change-responsive games: v80 ranking finds the right action

Phase 1 (100 steps): random exploration to discover action change rates.
Phase 2: ranked reactive cycling (change-rate order + distance switching).
20% epsilon-greedy throughout.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (no improvement over random/v80).
SUCCESS: avg L1 > 3.3/5 (beats v80) OR same L1 with better ARC.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 100
EPSILON = 0.2
CHANGE_THRESH = 0.1


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class DualSignalReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._prev_action = 0
        self._exploring = True

        # Per-action change statistics
        self._action_change_sum = {}
        self._action_change_count = {}

        # Ranked action set (by change rate, highest first)
        self._ranked_actions = []
        self._current_idx = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _rebuild_ranking(self):
        """Rank actions by average change rate (highest first)."""
        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))
        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for avg, a in action_avgs if avg > CHANGE_THRESH]

        # Fallback: keyboard actions
        if not self._ranked_actions:
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

    def _transition_to_exploit(self):
        self._exploring = False
        self._rebuild_ranking()
        self.r3_updates += 1
        self.att_updates_total += 1

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
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Measure change
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Record change stats for previous action
        a = self._prev_action
        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # Periodically rebuild ranking (new responsive actions from epsilon)
        if self.step_count % 500 == 0:
            self._rebuild_ranking()

        # === EXPLOIT: change-rate ranking + distance switching ===
        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Epsilon-greedy exploration
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            self._prev_action = action
            return action

        # v30 switching logic on change-rate-ranked actions
        if dist >= self._prev_dist:
            # No progress toward goal — try next highest-change action
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._patience = 0
        else:
            # Progress — hold current action
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)

        action = self._ranked_actions[self._current_idx]
        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_idx = 0
        self._patience = 0
        # Keep action rankings across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "family": "dual-signal reactive",
    "tag": "defense v81 (ℓ₁ dual-signal: v80 change-rate ranking + v30 distance switching. Actions ranked by how much change they cause, switched when distance progress stalls. Fuses both strategies.)",
}

SUBSTRATE_CLASS = DualSignalReactiveSubstrate
