"""
sub1164_defense_v70.py — Transition-triggered action memory (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1164 --substrate experiments/sub1164_defense_v70.py

FAMILY: Transition-triggered memory. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Every defense substrate ignores the most valuable signal
available: on_level_transition(). This is the ONLY explicit success feedback
the substrate receives. When a level is solved, the actions that caused it
are load-bearing.

No defense substrate has EVER used the transition signal to guide future
levels. v30 resets everything on transition — the successful action sequence
is discarded.

This substrate records the last MEMORY_SIZE actions before each level
transition, then REPLAYS them at the start of the next level. If replay
triggers another transition: the pattern works across levels. If not:
fall back to v30 reactive after exhausting the replay.

CONTROLLED COMPARISON vs v30:
- SAME: reactive distance-to-initial when not replaying
- DIFFERENT: v30 discards success history. v70 replays it.

ZERO learned parameters (defense: ℓ₁). Action memory is a list, not weights.
No per-observation conditioning — conditions on transition EVENT, not
specific past observations.

KILL: ARC ≤ v30.
SUCCESS: Cross-level replay improves L2+ rate on responsive games.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
MEMORY_SIZE = 50        # remember last N actions before transition
REPLAY_MAX = 100        # max steps to spend replaying before fallback


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class TransitionMemorySubstrate:
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

        self._n_active = N_KB

        # Action history (ring buffer for recording)
        self._action_history = deque(maxlen=MEMORY_SIZE)

        # Transition memory (successful action sequences)
        self._replay_sequence = []   # action sequence to replay
        self._replay_idx = 0
        self._replay_count = 0       # how many times we've replayed
        self._replaying = False

        # v30-style reactive state
        self._current_action = 0
        self._patience = 0

        # Click regions
        self._click_actions = []
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        # Reset most state but NOT the replay sequence
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._action_history = deque(maxlen=MEMORY_SIZE)
        self._replay_idx = 0
        self._replay_count = 0
        self._replaying = False
        self._current_action = 0
        self._patience = 0
        self._click_actions = []
        self._regions_set = False
        # Clear replay sequence for new game
        self._replay_sequence = []
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(n_actions, N_KB)

    def _discover_regions(self, enc):
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

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
            self._discover_regions(enc)
            # Start with replay if we have a saved sequence
            if self._replay_sequence:
                self._replaying = True
                self._replay_idx = 0
            return 0

        dist = np.sum(np.abs(enc - self._enc_0))

        # === REPLAY PHASE: repeat successful action sequence ===
        if self._replaying and self._replay_idx < len(self._replay_sequence):
            action = self._replay_sequence[self._replay_idx]
            self._replay_idx += 1
            self._action_history.append(action)

            # Check if replay is exhausted
            if self._replay_idx >= len(self._replay_sequence):
                # Cycle replay up to REPLAY_MAX steps
                self._replay_count += 1
                if self._replay_count * len(self._replay_sequence) < REPLAY_MAX:
                    self._replay_idx = 0  # replay again
                else:
                    self._replaying = False  # done replaying, fall back

            self._prev_dist = dist
            self._prev_enc = enc.copy()
            # Return raw action (already in env space from recording)
            if action < self._n_actions_env:
                return action
            return self._rng.randint(min(self._n_actions_env, N_KB))

        # === REACTIVE PHASE: v30-style ===
        if dist >= self._prev_dist:
            self._current_action = (self._current_action + 1) % self._n_active
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action = (self._current_action + 1) % self._n_active

        env_action = self._idx_to_env_action(self._current_action)
        self._action_history.append(env_action)

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        return env_action

    def on_level_transition(self):
        # CRITICAL: save the action sequence that led to this transition
        if self._action_history:
            self._replay_sequence = list(self._action_history)
            self.r3_updates += 1
            self.att_updates_total += 1

        # Reset observation state for new level
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action = 0
        self._patience = 0
        # Will start replaying on next process() call (when enc_0 is set)
        self._replaying = False  # set to True in process() when enc_0 is initialized
        self._replay_idx = 0
        self._replay_count = 0
        self._action_history = deque(maxlen=MEMORY_SIZE)


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "memory_size": MEMORY_SIZE,
    "replay_max": REPLAY_MAX,
    "family": "transition-triggered memory",
    "tag": "defense v70 (ℓ₁ transition memory: record last 50 actions before level solve, replay on next level. Tests cross-level action transfer via success signal.)",
}

SUBSTRATE_CLASS = TransitionMemorySubstrate
