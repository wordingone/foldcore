"""
sub1134_defense_v48.py — Lagged probe + full block scan (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1134 --substrate experiments/sub1134_defense_v48.py

FAMILY: Full-coverage click with lagged detection (NEW from explore)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Previous click probes checked for change 2-3 steps after
clicking. If games have DELAYED responses (change appears 5-10 steps after
click), our probes miss them entirely. v48 does a full block scan (all 256
blocks, 1 click each) and checks for change with a LAG: compare encoding
at step t+10 vs step t (not t+1 vs t).

Architecture:
- Phase 1 (steps 1-50): keyboard explore
- Phase 2 (steps 51-561): full block scan. Click each of 256 blocks once.
  Every 10 steps, compare current encoding to encoding 10 steps ago.
  If change > thresh, mark the block clicked 10 steps ago as responsive.
- Phase 3 (step 562+): reactive switching on discovered actions.

Budget: 50 keyboard + 256 click scan + ~255 lag buffer = 561 explore steps.
Remaining: ~9439 steps for exploitation.

KILL: ARC ≤ 0.
SUCCESS: Lagged detection finds responsive blocks that immediate detection missed.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
KB_EXPLORE = 50
LAG = 10  # check for change 10 steps after click
MAX_PATIENCE = 20
CHANGE_THRESH = 0.3  # lower threshold for lagged detection


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _block_to_click_action(block_idx):
    """Block center → click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class LaggedProbeSubstrate:
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
        self._prev_dist = None

        # Phase management
        self._phase = "keyboard"
        self._kb_responsive = []

        # Full block scan with lagged detection
        self._scan_order = []  # shuffled block indices
        self._scan_idx = 0
        self._enc_history = deque(maxlen=LAG + 1)  # rolling window
        self._action_history = deque(maxlen=LAG + 1)  # what we clicked
        self._responsive_blocks = []
        self._drain_remaining = 0  # steps left to drain lag buffer

        # Reactive switching
        self._active_actions = []
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()
        # Build shuffled scan order
        self._scan_order = list(range(N_BLOCKS * N_BLOCKS))
        self._rng.shuffle(self._scan_order)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

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
            self._prev_dist = 0.0
            self._current_idx = self._rng.randint(min(self._n_actions_env, N_KB))
            return self._current_idx

        dist = self._dist_to_initial(enc)

        # === Phase 1: Keyboard explore ===
        if self._phase == "keyboard":
            if self._prev_enc is not None:
                change = float(np.sum(np.abs(enc - self._prev_enc)))
                prev_a = (self.step_count - 1) % min(self._n_actions_env, N_KB)
                if change > CHANGE_THRESH and prev_a not in self._kb_responsive:
                    self._kb_responsive.append(prev_a)

            if self.step_count <= KB_EXPLORE:
                action = self.step_count % min(self._n_actions_env, N_KB)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            if len(self._kb_responsive) >= 2:
                self._phase = "reactive"
                self._active_actions = self._kb_responsive[:]
                self._current_idx = 0
            elif self._has_clicks:
                self._phase = "block_scan"
                self._scan_idx = 0
                self._enc_history.clear()
                self._action_history.clear()
                self._enc_history.append(enc.copy())
                self._action_history.append(-1)
            else:
                self._phase = "reactive"
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._current_idx = 0

        # === Phase 2: Full block scan with lagged detection ===
        if self._phase == "block_scan":
            # Check lagged response: compare current enc to enc LAG steps ago
            if len(self._enc_history) >= LAG:
                old_enc = self._enc_history[0]
                old_action = self._action_history[0]
                lagged_change = float(np.sum(np.abs(enc - old_enc)))
                if lagged_change > CHANGE_THRESH and old_action >= 0:
                    if old_action not in self._responsive_blocks:
                        self._responsive_blocks.append(old_action)

            # Record history
            self._enc_history.append(enc.copy())

            if self._scan_idx >= len(self._scan_order):
                # Scan complete — drain remaining lag buffer
                self._action_history.append(-1)
                if self._drain_remaining == 0:
                    self._drain_remaining = LAG  # start draining
                self._drain_remaining -= 1
                if self._drain_remaining > 0:
                    self._prev_enc = enc.copy()
                    self._prev_dist = dist
                    return self._rng.randint(min(self._n_actions_env, N_KB))
                # Build action set from responsive blocks
                self._active_actions = self._kb_responsive[:]
                for block_idx in self._responsive_blocks:
                    self._active_actions.append(_block_to_click_action(block_idx))
                if not self._active_actions:
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._phase = "reactive"
                self._current_idx = 0
            else:
                block = self._scan_order[self._scan_idx]
                self._action_history.append(block)
                self._scan_idx += 1
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _block_to_click_action(block)

        # === Phase 3: Reactive switching ===
        if self._phase == "reactive":
            n_active = len(self._active_actions)
            if n_active == 0:
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                n_active = len(self._active_actions)

            progress = (self._prev_dist - dist) > 1e-4 if self._prev_dist is not None else False
            no_change = abs(self._prev_dist - dist) < 1e-6 if self._prev_dist is not None else True

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
                    if self._actions_tried_this_round >= n_active:
                        self._current_idx = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_idx = (self._current_idx + 1) % n_active

            action = self._active_actions[self._current_idx % n_active]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return action

        # Fallback
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_idx = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_kb": N_KB,
    "kb_explore": KB_EXPLORE,
    "lag": LAG,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "full-coverage lagged click detection",
    "tag": "defense v48 (ℓ₁ full 256-block scan + lagged change detection, lag=10 steps)",
}

SUBSTRATE_CLASS = LaggedProbeSubstrate
