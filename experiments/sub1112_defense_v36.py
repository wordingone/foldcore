"""
sub1112_defense_v36.py — Adaptive click detection (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1112 --substrate experiments/sub1112_defense_v36.py

FAMILY: Reactive action switching (v30 logic, adaptive action space)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v35 wastes budget exploring 263 actions uniformly. Keyboard-only
games don't need click exploration. Adaptive detection: try keyboard first (50
steps). If keyboard produces change → stay keyboard (v30 behavior). If keyboard
is INERT and game supports clicks → probe 16 random block centers. Reactive
switching only among RESPONSIVE actions.

ONE CHANGE FROM v35: adaptive click detection instead of uniform 263-action explore.
Phase 1 (steps 1-50): keyboard explore (7 actions, identical to v30).
Phase 2 (steps 51-82): if keyboard was inert AND has_clicks, probe 16 random
  block centers (2 steps each = 32 steps). Track which produce change.
Phase 3 (step 83+): reactive switching among responsive actions only.
If keyboard worked → pure v30 from step 51 onward (zero overhead).

KILL: same 0% pattern as v35 (click detection not the bottleneck).
SUCCESS: ARC > v35 (0.0030) on click draws, AND ARC ≥ v30 (0.33) on keyboard draws.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
EXPLORE_STEPS_KB = 50
N_CLICK_PROBES = 16
CLICK_PROBE_REPS = 2  # steps per probe
MAX_PATIENCE = 20


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
    """Block center pixel → click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class AdaptiveClickSubstrate:
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
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0

        # Keyboard responsiveness tracking
        self._kb_total_change = 0.0
        self._kb_responsive = False

        # Click probing
        self._click_probes = []  # block indices to probe
        self._click_probe_idx = 0
        self._click_probe_step = 0
        self._responsive_clicks = []  # block indices that produced change
        self._click_probe_enc = None  # enc before each probe

        # Final action set
        self._active_actions = list(range(N_KB))  # start keyboard-only
        self._mode = "keyboard_explore"  # keyboard_explore → click_probe → reactive

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _to_env_action(self, internal_action):
        """Map active action index to environment action."""
        if internal_action < N_KB:
            return internal_action
        return _block_to_click_action(internal_action - N_KB)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _setup_click_probes(self):
        """Select N_CLICK_PROBES random block centers to test."""
        all_blocks = list(range(N_BLOCKS * N_BLOCKS))
        self._rng.shuffle(all_blocks)
        self._click_probes = all_blocks[:N_CLICK_PROBES]
        self._click_probe_idx = 0
        self._click_probe_step = 0
        self._responsive_clicks = []

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
            self._current_action = self._rng.randint(N_KB)
            return self._current_action

        dist = self._dist_to_initial(enc)

        # === Phase 1: Keyboard explore (steps 1-50) ===
        if self._mode == "keyboard_explore":
            # Track total change from keyboard actions
            change = float(np.sum(np.abs(enc - self._prev_enc)))
            self._kb_total_change += change

            if self.step_count <= EXPLORE_STEPS_KB:
                action = self.step_count % N_KB
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Keyboard explore done — decide next mode
            if self._kb_total_change > 1.0:
                # Keyboard is responsive → pure v30 reactive
                self._kb_responsive = True
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._mode = "reactive"
            elif self._has_clicks:
                # Keyboard inert, game has clicks → probe click targets
                self._mode = "click_probe"
                self._setup_click_probes()
                self._click_probe_enc = enc.copy()
            else:
                # Keyboard inert, no clicks → reactive with keyboard anyway
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._mode = "reactive"

        # === Phase 2: Click probing (if keyboard was inert) ===
        if self._mode == "click_probe":
            # Check if previous probe produced change
            if self._click_probe_step > 0 and self._click_probe_enc is not None:
                probe_change = float(np.sum(np.abs(enc - self._click_probe_enc)))
                if probe_change > 0.5:
                    block = self._click_probes[self._click_probe_idx]
                    self._responsive_clicks.append(block)

            self._click_probe_step += 1

            # Move to next probe after CLICK_PROBE_REPS steps
            if self._click_probe_step >= CLICK_PROBE_REPS:
                self._click_probe_idx += 1
                self._click_probe_step = 0
                self._click_probe_enc = enc.copy()

            # Still probing?
            if self._click_probe_idx < len(self._click_probes):
                block = self._click_probes[self._click_probe_idx]
                action = _block_to_click_action(block)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Probing done — build active action set
            if self._responsive_clicks:
                # Use responsive clicks + keyboard
                self._active_actions = list(range(N_KB))
                for block in self._responsive_clicks:
                    self._active_actions.append(N_KB + block)
            else:
                # No clicks responded either — fall back to keyboard
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
            self._mode = "reactive"
            self._current_action = 0
            self._steps_on_action = 0

        # === Phase 3: Reactive switching (v30 logic) ===
        if self._mode == "reactive":
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
                        self._current_action = self._rng.randint(n_active)
                        self._actions_tried_this_round = 0
                    else:
                        self._current_action = (self._current_action + 1) % n_active

            action_idx = self._active_actions[self._current_action % n_active]
            self._prev_enc = enc.copy()
            self._prev_dist = dist
            return self._to_env_action(action_idx)

        # Fallback (shouldn't reach here)
        self._prev_enc = enc.copy()
        self._prev_dist = dist
        return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = None
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Keep mode and active_actions — game type doesn't change between levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_kb": N_KB,
    "explore_steps_kb": EXPLORE_STEPS_KB,
    "n_click_probes": N_CLICK_PROBES,
    "click_probe_reps": CLICK_PROBE_REPS,
    "max_patience": MAX_PATIENCE,
    "family": "reactive action switching",
    "tag": "defense v36 (ℓ₁ adaptive click detection — keyboard-first, probe only if inert)",
}

SUBSTRATE_CLASS = AdaptiveClickSubstrate
