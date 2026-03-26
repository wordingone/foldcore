"""
sub1183_defense_v79.py — Decaying epsilon multi-timescale (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1183 --substrate experiments/sub1183_defense_v79.py

FAMILY: Decaying epsilon reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v78 showed multi-timescale + epsilon-greedy can achieve
high ARC (0.2559) on responsive games. But fixed epsilon wastes steps on
exploration after responsive actions are already found.

v79 = v78 + DECAYING epsilon:
- Start: epsilon=0.5 (50% random, maximum coverage)
- Decay: epsilon *= 0.999 each step
- Floor: epsilon_min=0.02 (always keep 2% exploration)
- At step 200: epsilon ≈ 0.41 (still exploring)
- At step 1000: epsilon ≈ 0.18 (transitioning to exploit)
- At step 5000: epsilon ≈ 0.003 (nearly pure exploit)

This naturally transitions from random's coverage to v30's exploitation.
Early steps discover responsive actions; late steps exploit them efficiently.

Multi-timescale detection (1/20/100 steps) from v78 for responsive action
identification.

ZERO learned parameters (defense: ℓ₁). Decay schedule is fixed, not learned.

KILL: ARC ≤ v78 (0.0512 best draw) AND L1 ≤ random (3/5).
SUCCESS: L1 ≥ 3/5 with ARC > v78.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

# Epsilon decay
EPSILON_START = 0.5
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.02

# Timescale windows
SHORT_WINDOW = 1
MEDIUM_WINDOW = 20
LONG_WINDOW = 100

DRIFT_THRESH = 0.1


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class DecayingEpsilonReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._prev_action = 0
        self._epsilon = EPSILON_START

        # Observation history
        self._enc_history = []
        self._max_history = LONG_WINDOW + 1

        # Responsive action tracking
        self._responsive_actions = {}
        self._action_set = []

        # Reactive state
        self._current_idx = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()
        n_kb = min(n_actions, N_KB)
        self._action_set = list(range(n_kb))

    def _multi_timescale_score(self, enc):
        scores = []
        if len(self._enc_history) >= SHORT_WINDOW:
            scores.append(float(np.sum(np.abs(enc - self._enc_history[-SHORT_WINDOW]))))
        if len(self._enc_history) >= MEDIUM_WINDOW:
            scores.append(float(np.sum(np.abs(enc - self._enc_history[-MEDIUM_WINDOW]))))
        if len(self._enc_history) >= LONG_WINDOW:
            scores.append(float(np.sum(np.abs(enc - self._enc_history[-LONG_WINDOW]))))
        return max(scores) if scores else 0.0

    def _add_responsive(self, action, score):
        if action in self._responsive_actions:
            self._responsive_actions[action] = max(
                self._responsive_actions[action], score
            )
        else:
            self._responsive_actions[action] = score
            self.r3_updates += 1
            self.att_updates_total += 1
        sorted_resp = sorted(
            self._responsive_actions.items(),
            key=lambda x: -x[1]
        )[:30]
        self._action_set = [a for a, _ in sorted_resp]
        if len(self._action_set) < N_KB:
            n_kb = min(self._n_actions_env, N_KB)
            for a in range(n_kb):
                if a not in self._action_set:
                    self._action_set.append(a)

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
            self._enc_history.append(enc.copy())
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Multi-timescale change detection
        mt_score = self._multi_timescale_score(enc)
        if mt_score > DRIFT_THRESH:
            self._add_responsive(self._prev_action, mt_score)

        # Store history
        self._enc_history.append(enc.copy())
        if len(self._enc_history) > self._max_history:
            self._enc_history.pop(0)

        dist = float(np.sum(np.abs(enc - self._enc_0)))

        # Decay epsilon
        self._epsilon = max(EPSILON_MIN, self._epsilon * EPSILON_DECAY)

        # Epsilon-greedy with decaying exploration
        if self._rng.random() < self._epsilon:
            action = int(self._rng.randint(0, self._n_actions_env))
        else:
            # Reactive cycling over action set
            if dist >= self._prev_dist:
                self._current_idx = (self._current_idx + 1) % len(self._action_set)
                self._patience = 0
            else:
                self._patience += 1
                if self._patience > 10:
                    self._patience = 0
                    self._current_idx = (self._current_idx + 1) % len(self._action_set)

            action = self._action_set[self._current_idx % len(self._action_set)]

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
        self._enc_history = []
        # Reset epsilon for new level — need to re-explore
        self._epsilon = EPSILON_START
        # Keep responsive actions across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon_start": EPSILON_START,
    "epsilon_decay": EPSILON_DECAY,
    "epsilon_min": EPSILON_MIN,
    "short_window": SHORT_WINDOW,
    "medium_window": MEDIUM_WINDOW,
    "long_window": LONG_WINDOW,
    "drift_thresh": DRIFT_THRESH,
    "family": "decaying epsilon reactive",
    "tag": "defense v79 (ℓ₁ decaying epsilon: starts 50% random → decays to 2%. Multi-timescale detection. Natural transition from random's coverage to v30's exploitation.)",
}

SUBSTRATE_CLASS = DecayingEpsilonReactiveSubstrate
