"""
sub1126_defense_v43.py — Hierarchical click search (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1126 --substrate experiments/sub1126_defense_v43.py

FAMILY: Hierarchical click exploration (NEW defense family for click games)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: v35 proved clicks CAN produce signal (ARC=0.0150) but
263-action uniform explore wastes budget. Hierarchical search: coarse grid
first (4×4 = 16 macro-blocks), find responsive quadrant, refine to 4×4
sub-blocks within it. Total explore: 16 coarse + 16 fine = 32 click probes
instead of 256, then reactive switching on responsive targets.

Architecture:
- Phase 1 (steps 1-50): keyboard explore (same as v30)
- If keyboard works → pure v30 behavior
- Phase 2 (steps 51-98): if keyboard inert AND has_clicks → probe 16
  macro-blocks (4×4 grid of 16×16 pixel regions). 3 steps per probe.
- Phase 3 (steps 99-146): refine into best macro-block's 16 sub-blocks.
  3 steps per probe.
- Phase 4 (step 147+): reactive switching among responsive click targets
  + any responsive keyboard actions.

Budget: 50 (keyboard) + 48 (coarse) + 48 (fine) = 146 explore steps.
Remaining: 9854 steps for reactive exploitation. v35 used 263 for explore.

KILL: ARC ≤ v35 (0.0030) on click games.
SUCCESS: ARC > v35 — hierarchical finds click targets faster.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
KB_EXPLORE = 50
MACRO_SIZE = 4  # 4×4 macro-blocks = 16 regions of 16×16 pixels
PROBE_REPS = 3  # steps per click probe
MAX_PATIENCE = 20
CHANGE_THRESH = 0.5


def _obs_to_enc(obs):
    """avgpool4: 64x64 → 16x16 = 256D."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return enc


def _pixel_to_click_action(px, py):
    """Pixel coordinate → click action index."""
    return N_KB + px + py * 64


class HierarchicalClickSubstrate:
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
        self._kb_total_change = 0.0
        self._kb_responsive = []

        # Hierarchical click probing
        self._macro_probes = []  # (center_px, center_py) for each macro-block
        self._macro_idx = 0
        self._macro_step = 0
        self._macro_pre_enc = None
        self._macro_response = {}  # macro_idx → change magnitude

        self._fine_probes = []  # (px, py) for fine-grid within best macro
        self._fine_idx = 0
        self._fine_step = 0
        self._fine_pre_enc = None
        self._responsive_clicks = []  # (px, py) that produced change

        # Reactive switching
        self._active_actions = []  # env action indices
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
        # Build macro-block probes: center of each 16×16 region
        self._macro_probes = []
        macro_step = 64 // MACRO_SIZE  # 16 pixels per macro-block
        for my in range(MACRO_SIZE):
            for mx in range(MACRO_SIZE):
                cx = mx * macro_step + macro_step // 2
                cy = my * macro_step + macro_step // 2
                self._macro_probes.append((cx, cy))
        self._rng.shuffle(self._macro_probes)

    def _dist_to_initial(self, enc):
        if self._enc_0 is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0)))

    def _build_fine_probes(self, best_macro_idx):
        """Build 16 fine-grid probes within the best macro-block."""
        cx, cy = self._macro_probes[best_macro_idx]
        macro_step = 64 // MACRO_SIZE  # 16
        fine_step = macro_step // 4  # 4 pixels per fine block
        base_x = cx - macro_step // 2
        base_y = cy - macro_step // 2
        self._fine_probes = []
        for fy in range(4):
            for fx in range(4):
                px = base_x + fx * fine_step + fine_step // 2
                py = base_y + fy * fine_step + fine_step // 2
                px = max(0, min(63, px))
                py = max(0, min(63, py))
                self._fine_probes.append((px, py))
        self._rng.shuffle(self._fine_probes)

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
                self._kb_total_change += change
                prev_a = (self.step_count - 1) % min(self._n_actions_env, N_KB)
                if change > CHANGE_THRESH and prev_a not in self._kb_responsive:
                    self._kb_responsive.append(prev_a)

            if self.step_count <= KB_EXPLORE:
                action = self.step_count % min(self._n_actions_env, N_KB)
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return action

            # Keyboard done — decide next
            if len(self._kb_responsive) >= 2:
                self._phase = "reactive"
                self._active_actions = self._kb_responsive[:]
                self._current_idx = 0
            elif self._has_clicks:
                self._phase = "macro_probe"
                self._macro_idx = 0
                self._macro_step = 0
                self._macro_pre_enc = enc.copy()
            else:
                self._phase = "reactive"
                self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._current_idx = 0

        # === Phase 2: Macro-block click probing ===
        if self._phase == "macro_probe":
            # Check previous probe result
            if self._macro_step > 0 and self._macro_pre_enc is not None:
                if self._macro_step == PROBE_REPS:
                    probe_change = float(np.sum(np.abs(enc - self._macro_pre_enc)))
                    self._macro_response[self._macro_idx] = probe_change
                    self._macro_idx += 1
                    self._macro_step = 0
                    self._macro_pre_enc = enc.copy()

            if self._macro_idx >= len(self._macro_probes):
                # Macro probing done — find best
                if self._macro_response:
                    best_idx = max(self._macro_response, key=self._macro_response.get)
                    best_change = self._macro_response[best_idx]
                    if best_change > CHANGE_THRESH:
                        self._build_fine_probes(best_idx)
                        self._phase = "fine_probe"
                        self._fine_idx = 0
                        self._fine_step = 0
                        self._fine_pre_enc = enc.copy()
                        # Also add the macro center as a responsive click
                        cx, cy = self._macro_probes[best_idx]
                        self._responsive_clicks.append((cx, cy))
                    else:
                        # No macro-block responded — fall back
                        self._phase = "reactive"
                        self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                        self._current_idx = 0
                else:
                    self._phase = "reactive"
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                    self._current_idx = 0
            else:
                self._macro_step += 1
                cx, cy = self._macro_probes[self._macro_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _pixel_to_click_action(cx, cy)

        # === Phase 3: Fine-grid click probing ===
        if self._phase == "fine_probe":
            if self._fine_step > 0 and self._fine_pre_enc is not None:
                if self._fine_step == PROBE_REPS:
                    probe_change = float(np.sum(np.abs(enc - self._fine_pre_enc)))
                    if probe_change > CHANGE_THRESH:
                        px, py = self._fine_probes[self._fine_idx]
                        self._responsive_clicks.append((px, py))
                    self._fine_idx += 1
                    self._fine_step = 0
                    self._fine_pre_enc = enc.copy()

            if self._fine_idx >= len(self._fine_probes):
                # Fine probing done — build action set
                self._active_actions = self._kb_responsive[:]
                for px, py in self._responsive_clicks:
                    self._active_actions.append(_pixel_to_click_action(px, py))
                if not self._active_actions:
                    self._active_actions = list(range(min(self._n_actions_env, N_KB)))
                self._phase = "reactive"
                self._current_idx = 0
            else:
                self._fine_step += 1
                px, py = self._fine_probes[self._fine_idx]
                self._prev_enc = enc.copy()
                self._prev_dist = dist
                return _pixel_to_click_action(px, py)

        # === Phase 4: Reactive switching ===
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
        # Keep phase and active_actions — same game type


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_kb": N_KB,
    "kb_explore": KB_EXPLORE,
    "macro_size": MACRO_SIZE,
    "probe_reps": PROBE_REPS,
    "max_patience": MAX_PATIENCE,
    "change_thresh": CHANGE_THRESH,
    "family": "hierarchical click exploration",
    "tag": "defense v43 (ℓ₁ hierarchical click: coarse 4×4 → fine 4×4 = 32 probes vs v35's 256)",
}

SUBSTRATE_CLASS = HierarchicalClickSubstrate
