"""
sub1213_defense_v89.py — Multi-mode reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1213 --substrate experiments/sub1213_defense_v89.py

FAMILY: Multi-mode reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Step 1079 diagnostic found TWO distinct 0% failure modes:
  Mode 1 (near-inert): pixel_var=0.001. Detection problem.
  Mode 2 (responsive-unsolved): pixel_var=0.37, 47/74 sign changes. Oscillation.
v80 treats ALL games identically. What if we detect the game type during
exploration and apply a TYPE-SPECIFIC strategy?

Explore phase (200 steps): measure per-action pixel variance AND sign-change
frequency. Classify game into one of three types:

Type A (RESPONSIVE): high avg change (>0.5), low sign-flip rate (<0.4).
  → v80 recipe: rank by change rate, cycle through ranks, 20% epsilon.

Type B (OSCILLATING): high avg change (>0.5), high sign-flip rate (>0.4).
  → Alternating pair: find the two actions with most anti-correlated changes,
    alternate between them every 2 steps. 20% epsilon.

Type C (NEAR-INERT): low avg change (≤0.5).
  → Extended systematic scan: hold each action for 10 steps (detect slow
    responses), explore FULL action space including clicks. 50% epsilon
    (more exploration needed for inert games).

This is NOT the same as v85 (binary CPG vs v80). This uses THREE modes
with empirically-motivated strategies and sign-change-based detection.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.3/5 (no improvement over v80).
SUCCESS: avg L1 > 3.3/5 (multi-mode helps).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 200
CHANGE_THRESH = 0.1
EPSILON_NORMAL = 0.2
EPSILON_INERT = 0.5
SIGN_FLIP_THRESH = 0.4
INERT_CHANGE_THRESH = 0.5
HOLD_DURATION_INERT = 10


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class MultiModeReactiveSubstrate:
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

        # Exploration diagnostics
        self._action_change_sum = {}
        self._action_change_count = {}
        self._action_prev_delta = {}  # for sign-flip tracking
        self._action_sign_flips = {}
        self._action_sign_total = {}

        # Strategy state
        self._game_type = 'A'  # default responsive
        self._ranked_actions = []
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0

        # Type B: alternating pair
        self._alt_pair = (0, 1)
        self._alt_toggle = 0

        # Type C: systematic scan
        self._scan_action = 0
        self._scan_hold = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _classify_game(self):
        """Classify game type from exploration statistics."""
        # Compute average change and sign-flip rate across all observed actions
        total_change = 0.0
        total_count = 0
        total_flips = 0
        total_sign_obs = 0

        for a in self._action_change_sum:
            count = self._action_change_count.get(a, 1)
            total_change += self._action_change_sum[a]
            total_count += count
            total_flips += self._action_sign_flips.get(a, 0)
            total_sign_obs += self._action_sign_total.get(a, 0)

        avg_change = total_change / max(total_count, 1)
        sign_flip_rate = total_flips / max(total_sign_obs, 1)

        if avg_change <= INERT_CHANGE_THRESH:
            return 'C'  # near-inert
        elif sign_flip_rate > SIGN_FLIP_THRESH:
            return 'B'  # oscillating
        else:
            return 'A'  # responsive

    def _build_v80_strategy(self):
        """v80-style: rank by change rate, prepare cycling."""
        action_avgs = []
        for a, total in self._action_change_sum.items():
            count = self._action_change_count.get(a, 1)
            action_avgs.append((total / count, a))

        action_avgs.sort(reverse=True)
        self._ranked_actions = [a for _, a in action_avgs if _ > CHANGE_THRESH]

        if not self._ranked_actions:
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

        self._current_idx = 0

    def _build_alternating_strategy(self):
        """Find the two actions with most different change patterns."""
        actions = list(self._action_change_sum.keys())
        if len(actions) < 2:
            self._alt_pair = (0, 1)
            return

        # Find pair with most different average change magnitudes
        best_pair = (actions[0], actions[1] if len(actions) > 1 else actions[0])
        best_diff = -1.0

        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                a1, a2 = actions[i], actions[j]
                c1 = self._action_change_sum[a1] / max(self._action_change_count.get(a1, 1), 1)
                c2 = self._action_change_sum[a2] / max(self._action_change_count.get(a2, 1), 1)
                diff = abs(c1 - c2)
                if diff > best_diff:
                    best_diff = diff
                    best_pair = (a1, a2)

        self._alt_pair = best_pair
        self._alt_toggle = 0

    def _transition_to_exploit(self):
        self._exploring = False
        self.r3_updates += 1
        self.att_updates_total += 1

        self._game_type = self._classify_game()

        if self._game_type == 'A':
            self._build_v80_strategy()
        elif self._game_type == 'B':
            self._build_alternating_strategy()
            # Also build v80 as fallback
            self._build_v80_strategy()
        # Type C: systematic scan, no setup needed

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

        # Track sign flips (delta increasing vs decreasing)
        if a in self._action_prev_delta:
            prev_d = self._action_prev_delta[a]
            if (delta > prev_d + 0.01) != (prev_d > delta + 0.01):
                # Direction changed
                self._action_sign_flips[a] = self._action_sign_flips.get(a, 0) + 1
            self._action_sign_total[a] = self._action_sign_total.get(a, 0) + 1
        self._action_prev_delta[a] = delta

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # Periodically reclassify
        if self.step_count % 1000 == 0:
            self._transition_to_exploit()

        # === TYPE A: v80 recipe ===
        if self._game_type == 'A':
            action = self._type_a_action(delta)

        # === TYPE B: alternating pair ===
        elif self._game_type == 'B':
            action = self._type_b_action(delta)

        # === TYPE C: systematic scan ===
        else:
            action = self._type_c_action(delta)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def _type_a_action(self, delta):
        """v80-style cycling through ranked actions."""
        if self._rng.random() < EPSILON_NORMAL:
            return int(self._rng.randint(0, self._n_actions_env))

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

        return self._ranked_actions[self._current_idx]

    def _type_b_action(self, delta):
        """Alternating pair for oscillating games."""
        if self._rng.random() < EPSILON_NORMAL:
            return int(self._rng.randint(0, self._n_actions_env))

        # Alternate every 2 steps
        self._alt_toggle += 1
        if self._alt_toggle % 2 == 0:
            return self._alt_pair[0]
        else:
            return self._alt_pair[1]

    def _type_c_action(self, delta):
        """Systematic scan for near-inert games."""
        if self._rng.random() < EPSILON_INERT:
            return int(self._rng.randint(0, self._n_actions_env))

        # Hold each action for HOLD_DURATION_INERT steps
        self._scan_hold += 1
        if self._scan_hold >= HOLD_DURATION_INERT:
            self._scan_hold = 0
            self._scan_action = (self._scan_action + 1) % self._n_actions_env

        return self._scan_action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0
        self._alt_toggle = 0
        self._scan_hold = 0
        self._scan_action = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon_normal": EPSILON_NORMAL,
    "epsilon_inert": EPSILON_INERT,
    "sign_flip_thresh": SIGN_FLIP_THRESH,
    "inert_change_thresh": INERT_CHANGE_THRESH,
    "hold_duration_inert": HOLD_DURATION_INERT,
    "family": "multi-mode reactive",
    "tag": "defense v89 (ℓ₁ multi-mode: classifies games during exploration into Type A (responsive→v80 cycling), Type B (oscillating→alternating pair), Type C (near-inert→systematic scan with extended hold). Step 1079 found two 0% modes. Tests if mode-specific strategies break v80's ceiling.)",
}

SUBSTRATE_CLASS = MultiModeReactiveSubstrate
