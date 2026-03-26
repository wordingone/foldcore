"""
sub1088_defense_v23.py — Anti-oscillation reactive switching

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1088 --substrate experiments/sub1088_defense_v23.py

FAMILY: Reactive action switching with oscillation detection (defense-only)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Oscillation detection without learning breaks more games than
pure reactive switching. Step 1082 proved oscillation IS the bottleneck:
47/74 sign changes, game state drifts under fixed strategy. v21 can't detect
oscillation — it keeps the "progressive" action even when bouncing. If
fixed-threshold oscillation detection works, learned encoding (prosecution's
alpha) is unnecessary for breaking oscillation.

BUILDS ON Step 1084 (defense v21, ARC=0.2973):
- v21 solved 1/3 ARC games (100%, 10/10 seeds)
- v21 limitation: no oscillation detection. If action A reduces dist_to_initial
  but game oscillates A→B→A→B, v21 keeps picking A indefinitely.
- v22 (multi-scale) was CHAIN KILL — complexity hurts

ARCHITECTURE (genuinely different from prosecution):
- avgpool8 (64D) encoding — same scale as v21
- Distance-to-initial criterion — same as v21
- ADDED: oscillation detection via ring buffer of last K=30 encodings
  - After each step: compute min L1 distance to recent buffer
  - If min_dist < OSC_THRESH → oscillation detected
  - On oscillation: force switch to next action, mark current as oscillatory
  - Oscillatory actions deprioritized (skipped in round-robin)
  - If ALL actions oscillatory → random + clear flags
- Zero learned parameters

WHY DIFFERENT FROM PROSECUTION:
- No alpha (no learned attention weights)
- No W_pred (no prediction matrix)
- No attention mechanism (no softmax retrieval)
- No trajectory buffer with action-delta association
- Fixed-threshold oscillation detection, not learned similarity

WHY DIFFERENT FROM v22 (KILLED):
- Single scale (avgpool8), not multi-scale
- No action memory with mean-progress scoring
- Oscillation detection instead of scale selection

KILL: worse than v21 (0/3 games solved OR ARC < v21's 0.2973 on same game).
SUCCESS: solve v21's game + any additional game.
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
OSC_WINDOW = 30     # check last 30 states for oscillation
OSC_THRESH = 0.5    # L1 distance threshold for "same state" (64D, values 0-1)
OSC_SKIP_LIMIT = 3  # max consecutive oscillation-forced switches before reset


def _obs_to_enc(obs):
    """avgpool8: 64x64 → 8x8 = 64D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class AntiOscillationReactiveSubstrate:
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

        # Oscillation detection
        self._recent_encs = np.zeros((OSC_WINDOW, N_DIMS), dtype=np.float32)
        self._recent_count = 0
        self._recent_idx = 0  # ring buffer write pointer
        self._osc_actions = set()  # actions flagged as oscillatory
        self._osc_switches = 0  # consecutive oscillation-forced switches

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

    def _add_to_recent(self, enc):
        """Add encoding to ring buffer."""
        self._recent_encs[self._recent_idx] = enc
        self._recent_idx = (self._recent_idx + 1) % OSC_WINDOW
        self._recent_count = min(self._recent_count + 1, OSC_WINDOW)

    def _is_oscillating(self, enc):
        """Check if current state matches any recent state (within threshold)."""
        if self._recent_count < 5:
            return False
        n = self._recent_count
        dists = np.sum(np.abs(self._recent_encs[:n] - enc), axis=1)
        return float(np.min(dists)) < OSC_THRESH

    def _next_non_osc_action(self):
        """Get next action in round-robin, skipping oscillatory ones."""
        for _ in range(self._n_actions):
            self._current_action = (self._current_action + 1) % self._n_actions
            if self._current_action not in self._osc_actions:
                return self._current_action
        # All actions oscillatory — clear flags and pick random
        self._osc_actions.clear()
        self._osc_switches = 0
        return self._rng.randint(self._n_actions)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # Store initial encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._prev_dist = 0.0
            self._add_to_recent(enc)
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            self._add_to_recent(enc)
            action = self.step_count % self._n_actions
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Check for oscillation BEFORE normal reactive logic
        oscillating = self._is_oscillating(enc)
        self._add_to_recent(enc)

        if oscillating:
            # Current action is causing oscillation — flag it and switch
            self._osc_actions.add(self._current_action)
            self._osc_switches += 1

            if self._osc_switches >= OSC_SKIP_LIMIT:
                # Too many oscillation switches — random reset
                self._current_action = self._rng.randint(self._n_actions)
                self._osc_actions.clear()
                self._osc_switches = 0
            else:
                self._current_action = self._next_non_osc_action()

            self._steps_on_action = 0
            self._patience = 3
            self._consecutive_progress = 0
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._current_action

        # Not oscillating — use v21's reactive logic
        self._osc_switches = 0  # reset oscillation counter on non-oscillating step

        progress = (self._prev_dist - dist) > 1e-4
        no_change = abs(self._prev_dist - dist) < 1e-6

        self._steps_on_action += 1

        if progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
            # Clear oscillation flag for this action — it's making progress
            self._osc_actions.discard(self._current_action)
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
                    self._current_action = self._next_non_osc_action()

        self._prev_enc = enc.copy()
        self._prev_dist = dist
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
        self._recent_encs = np.zeros((OSC_WINDOW, N_DIMS), dtype=np.float32)
        self._recent_count = 0
        self._recent_idx = 0
        self._osc_actions = set()
        self._osc_switches = 0


CONFIG = {
    "n_dims": N_DIMS,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "osc_window": OSC_WINDOW,
    "osc_thresh": OSC_THRESH,
    "family": "reactive action switching (anti-oscillation)",
    "tag": "defense v23 (ℓ₁ anti-oscillation reactive, zero learned params)",
}

SUBSTRATE_CLASS = AntiOscillationReactiveSubstrate
