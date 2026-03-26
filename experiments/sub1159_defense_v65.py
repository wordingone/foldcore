"""
sub1159_defense_v65.py — Bidirectional reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1159 --substrate experiments/sub1159_defense_v65.py

FAMILY: Bidirectional reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v30 assumes games start near the goal (minimize distance
to initial = progress). But some games might start FAR from the goal
(need to maximize distance from initial). v30 gets 2-3/5 because it
optimizes the right direction for some games but wrong for others.

This substrate tests BOTH directions simultaneously:
- Track distance to initial for each action
- Track whether distance is INCREASING or DECREASING over time
- If distance is increasing over many steps → game wants to MOVE AWAY
  (distance-from-initial is progress). Switch to argmax.
- If distance is decreasing → game wants to RETURN
  (distance-to-initial is progress). Stay with argmin.

CONTROLLED COMPARISON vs v30:
- SAME: encoding, reactive switching, distance metric
- DIFFERENT: v30 always minimizes. v65 detects direction and adapts.

Architecture:
- Phase 1 (steps 0-100): cycle all actions, track per-action effect on distance
- Phase 2 (steps 100+): for each action, track if dist increased or decreased
  - direction_score[a] = (times a decreased dist) - (times a increased dist)
  - If direction_score > 0: this action HELPS return → use when goal = return
  - If direction_score < 0: this action MOVES AWAY → use when goal = explore
  - Auto-detect: if L1 not achieved after 2000 steps with argmin, switch to argmax

ZERO learned parameters (defense: ℓ₁). Fixed detection + switching protocol.

KILL: ARC ≤ v30.
SUCCESS: Bidirectional > unidirectional (some games need argmax).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
SCAN_END = 100
DIRECTION_SWITCH_THRESHOLD = 2000  # switch direction after this many steps


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class BidirectionalReactiveSubstrate:
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

        # Per-action direction tracking
        self._approach_count = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)  # times action decreased dist
        self._retreat_count = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)  # times action increased dist
        self._prev_action = 0

        # Direction mode: 'approach' (minimize dist) or 'retreat' (maximize dist)
        self._direction = 'approach'
        self._direction_switched = False

        # Reactive state
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
        self._init_state()

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
            return 0

        dist = np.sum(np.abs(enc - self._enc_0))

        # Track direction for previous action
        if dist < self._prev_dist:
            self._approach_count[self._prev_action] += 1
        elif dist > self._prev_dist:
            self._retreat_count[self._prev_action] += 1

        # Phase 1: scan all actions
        if self.step_count <= SCAN_END:
            action = (self.step_count - 1) % self._n_active
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            self._prev_action = action
            return self._idx_to_env_action(action)

        # Auto-switch: if approach hasn't worked after threshold, try retreat
        if not self._direction_switched and self.step_count > DIRECTION_SWITCH_THRESHOLD:
            # Check if any progress has been made
            if dist > 1.0:  # still far from initial after 2000 steps
                self._direction = 'retreat'
                self._direction_switched = True
                self.r3_updates += 1
                self.att_updates_total += 1

        # Reactive phase: pick best action for current direction
        if self._direction == 'approach':
            # Improvement = distance decreased
            improved = (dist < self._prev_dist)
        else:
            # Improvement = distance increased
            improved = (dist > self._prev_dist)

        if not improved:
            # Switch action
            self._current_action = (self._current_action + 1) % self._n_active
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                # Even if improving, periodically try other actions
                self._patience = 0
                self._current_action = (self._current_action + 1) % self._n_active

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action = self._current_action
        return self._idx_to_env_action(self._current_action)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action = 0
        self._patience = 0
        # Reset direction for new level (might need different direction)
        self._direction = 'approach'
        self._direction_switched = False
        # Keep per-action direction stats across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "scan_end": SCAN_END,
    "direction_switch_threshold": DIRECTION_SWITCH_THRESHOLD,
    "family": "bidirectional reactive",
    "tag": "defense v65 (ℓ₁ bidirectional: approach first, auto-switch to retreat after 2000 steps if no progress. Tests both distance directions.)",
}

SUBSTRATE_CLASS = BidirectionalReactiveSubstrate
