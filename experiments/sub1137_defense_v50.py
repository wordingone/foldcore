"""
sub1137_defense_v50.py — Cycle-breaking reactive switching (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1137 --substrate experiments/sub1137_defense_v50.py

FAMILY: Reactive action switching (NEW: temporal cycle detection)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Step 1082 diagnostic showed OSCILLATION is the Mode 2 bottleneck:
47/74 sign changes under best-action repetition. The game bounces back and forth.
Previous anti-oscillation attempt (v23, ring buffer + threshold) FAILED because
it interfered with working games.

v50 takes a different approach: detect cycles by comparing current encoding to
encodings from K steps ago (K=5,10,20). If any match within tolerance, the game
is CYCLING — force a different action than the one that caused the cycle.

Key difference from v23: v23 counted oscillation frequency and forced switches
when frequency exceeded a threshold. v50 detects EXACT return to prior state
and only intervenes then. This is more conservative — won't interfere with
games that are making progress even if oscillating locally.

Architecture:
- enc = avgpool4 (256D)
- Ring buffer of last 30 encodings
- Cycle check: if |enc_t - enc_{t-K}| < CYCLE_THRESH for K in {5, 10, 20}
  → game is cycling → exclude current action, pick next
- Otherwise: standard reactive switching with ℓ₁ goal

KILL: ARC ≤ v30 (0.3319).
SUCCESS: breaks oscillation on games where v30 gets stuck in cycles.
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
BUFFER_SIZE = 30
CYCLE_THRESH = 0.5  # L1 distance below which = "same state"
CYCLE_LAGS = [5, 10, 20]  # check these lags for cycles


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class CycleBreakReactiveSubstrate:
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

        # Cycle detection buffer
        self._enc_buffer = []
        self._action_buffer = []
        self._cycle_breaks = 0

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

    def _detect_cycle(self, enc):
        """Check if current enc matches any enc from K steps ago."""
        n = len(self._enc_buffer)
        for lag in CYCLE_LAGS:
            if lag <= n:
                old_enc = self._enc_buffer[n - lag]
                dist = float(np.sum(np.abs(enc - old_enc)))
                if dist < CYCLE_THRESH:
                    return True, lag
        return False, 0

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
            self._enc_buffer.append(enc.copy())
            self._action_buffer.append(self._current_action)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Store in buffer (keep last BUFFER_SIZE)
        self._enc_buffer.append(enc.copy())
        self._action_buffer.append(self._current_action)
        if len(self._enc_buffer) > BUFFER_SIZE:
            self._enc_buffer.pop(0)
            self._action_buffer.pop(0)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            self._action_buffer[-1] = action
            return action

        # Cycle detection — conservative intervention
        is_cycling, lag = self._detect_cycle(enc)
        if is_cycling:
            # State returned to where it was `lag` steps ago
            # Exclude the action that was active then
            n = len(self._action_buffer)
            if lag <= n:
                bad_action = self._action_buffer[n - lag]
                # Pick a different action
                candidates = [a for a in range(self._n_actions) if a != bad_action]
                if candidates:
                    self._current_action = candidates[self._rng.randint(len(candidates))]
                else:
                    self._current_action = self._rng.randint(self._n_actions)
                self._cycle_breaks += 1
                self._steps_on_action = 0
                self._patience = 3
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                self._action_buffer[-1] = self._current_action
                return self._current_action

        # Standard reactive switching (v30 logic)
        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6

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
        self._action_buffer[-1] = self._current_action
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
        self._enc_buffer.clear()
        self._action_buffer.clear()


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "buffer_size": BUFFER_SIZE,
    "cycle_thresh": CYCLE_THRESH,
    "cycle_lags": CYCLE_LAGS,
    "family": "reactive action switching",
    "tag": "defense v50 (cycle-breaking reactive: detect state return at lag 5/10/20, exclude cycling action)",
}

SUBSTRATE_CLASS = CycleBreakReactiveSubstrate
