"""
sub1163_defense_v69.py — Coarse-to-fine click search + reactive keyboard (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1163 --substrate experiments/sub1163_defense_v69.py

FAMILY: Spatial search reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All prior click substrates (v59, v63, v64) target clicks
based on VISUAL SALIENCY — regions that look different from the mean.
But the responsive click target might NOT be visually salient. A game
with a plain-colored button won't show saliency at the button location.

This substrate uses COARSE-TO-FINE spatial search:
1. Start with a 4×4 grid (16 points covering the full 64×64 space)
2. Try each point for HOLD steps, measure change
3. If a region shows response: subdivide into 4 sub-regions, repeat
4. After finding responsive regions: exploit them + v30 keyboard reactive

For keyboard-only games (n_actions ≤ 7): pure v30 reactive (proven best).

DIFFERENT from all prior click targeting:
- v63/v64: saliency-based (assumes responsive region is visually distinct)
- v59: empowerment over fixed salient regions
- v69: SYSTEMATIC search (no visual assumption, finds ANY responsive pixel)

ZERO learned parameters (defense: ℓ₁). Fixed spatial search protocol.

KILL: ARC ≤ v30.
SUCCESS: Spatial search finds click targets that saliency misses.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
HOLD_STEPS = 5        # steps to hold each click during search
REFINE_DEPTH = 3      # max subdivision depth (4→8→16 grid)
RESPONSE_THRESH = 0.5  # minimum change to count as responsive


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class CoarseToFineClickSubstrate:
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

        # Click search state
        self._search_phase = True
        self._search_queue = []       # [(cx, cy, half_size), ...] regions to test
        self._current_search = None   # (cx, cy, half_size) being tested
        self._search_hold = 0         # steps held on current search point
        self._search_enc_start = None # enc when hold started
        self._responsive_regions = [] # [(cx, cy, change), ...] found responsive
        self._search_depth = 0

        # Exploit state
        self._exploit_clicks = []     # env actions for responsive regions
        self._exploit_idx = 0
        self._exploit_patience = 0

        # Keyboard reactive state (v30)
        self._kb_action = 0
        self._kb_patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

        if self._has_clicks:
            # Initialize coarse grid: 4×4 = 16 points
            self._search_queue = []
            step_size = 16  # 64/4
            for gy in range(4):
                for gx in range(4):
                    cx = gx * step_size + step_size // 2
                    cy = gy * step_size + step_size // 2
                    self._search_queue.append((cx, cy, step_size // 2))
        else:
            self._search_phase = False

    def _click_action(self, x, y):
        x = max(0, min(63, x))
        y = max(0, min(63, y))
        return N_KB + y * 64 + x

    def _refine_region(self, cx, cy, half_size):
        """Subdivide a responsive region into 4 sub-regions."""
        new_half = max(half_size // 2, 2)
        sub_regions = []
        for dy in [-1, 1]:
            for dx in [-1, 1]:
                nx = cx + dx * new_half
                ny = cy + dy * new_half
                nx = max(new_half, min(63 - new_half, nx))
                ny = max(new_half, min(63 - new_half, ny))
                sub_regions.append((nx, ny, new_half))
        return sub_regions

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
            return 0

        dist = np.sum(np.abs(enc - self._enc_0))

        # === KEYBOARD-ONLY GAMES: pure v30 reactive ===
        if not self._has_clicks:
            n_kb = min(self._n_actions_env, N_KB)
            if dist >= self._prev_dist:
                self._kb_action = (self._kb_action + 1) % n_kb
                self._kb_patience = 0
            else:
                self._kb_patience += 1
                if self._kb_patience > 10:
                    self._kb_patience = 0
                    self._kb_action = (self._kb_action + 1) % n_kb
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            return self._kb_action

        # === CLICK GAMES: coarse-to-fine search then exploit ===

        # Search phase
        if self._search_phase:
            # Start new search point
            if self._current_search is None:
                if not self._search_queue:
                    # No more regions to search — check if we should refine
                    if self._responsive_regions and self._search_depth < REFINE_DEPTH:
                        # Refine responsive regions
                        self._search_depth += 1
                        for cx, cy, change in self._responsive_regions:
                            half = 64 // (4 * (2 ** self._search_depth))
                            half = max(half, 2)
                            self._search_queue.extend(self._refine_region(cx, cy, half))
                        self._responsive_regions = []
                    else:
                        # Done searching — transition to exploit
                        self._search_phase = False
                        self.r3_updates += 1
                        self.att_updates_total += 1
                        if self._responsive_regions:
                            # Sort by change magnitude
                            self._responsive_regions.sort(key=lambda x: -x[2])
                            self._exploit_clicks = [
                                self._click_action(cx, cy)
                                for cx, cy, _ in self._responsive_regions[:8]
                            ]
                        if not self._exploit_clicks:
                            # No responsive regions found — fall back to keyboard
                            self._exploit_clicks = []
                        # Fall through to exploit
                if self._search_queue and self._search_phase:
                    self._current_search = self._search_queue.pop(0)
                    self._search_hold = 0
                    self._search_enc_start = enc.copy()

            if self._search_phase and self._current_search is not None:
                cx, cy, half_size = self._current_search
                self._search_hold += 1

                if self._search_hold >= HOLD_STEPS:
                    # Evaluate response
                    change = float(np.sum(np.abs(enc - self._search_enc_start)))
                    if change > RESPONSE_THRESH:
                        self._responsive_regions.append((cx, cy, change))
                    self._current_search = None

                self._prev_dist = dist
                self._prev_enc = enc.copy()
                return self._click_action(cx, cy)

        # Exploit phase: alternate keyboard reactive + responsive clicks
        is_even = (self.step_count % 2 == 0)

        if is_even or not self._exploit_clicks:
            # Keyboard reactive (v30)
            n_kb = min(self._n_actions_env, N_KB)
            if dist >= self._prev_dist:
                self._kb_action = (self._kb_action + 1) % n_kb
                self._kb_patience = 0
            else:
                self._kb_patience += 1
                if self._kb_patience > 10:
                    self._kb_patience = 0
                    self._kb_action = (self._kb_action + 1) % n_kb
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            return self._kb_action
        else:
            # Exploit responsive click regions
            action = self._exploit_clicks[self._exploit_idx]
            self._exploit_patience += 1
            if self._exploit_patience >= 8:
                self._exploit_patience = 0
                self._exploit_idx = (self._exploit_idx + 1) % len(self._exploit_clicks)
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._kb_action = 0
        self._kb_patience = 0
        self._exploit_idx = 0
        self._exploit_patience = 0
        # Keep responsive regions and exploit clicks across levels
        # Don't re-search — responsive regions likely persist


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "hold_steps": HOLD_STEPS,
    "refine_depth": REFINE_DEPTH,
    "response_thresh": RESPONSE_THRESH,
    "family": "spatial search reactive",
    "tag": "defense v69 (ℓ₁ coarse-to-fine: systematic 4×4→8×8→16×16 click search, no saliency assumption. Tests whether click targets are missed by visual saliency.)",
}

SUBSTRATE_CLASS = CoarseToFineClickSubstrate
