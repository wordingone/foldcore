"""
sub1145_defense_v55.py — Ultra-patient action persistence (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1145 --substrate experiments/sub1145_defense_v55.py

FAMILY: Ultra-patient persistence (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: RETHINK origin. ALL 44 prior substrates switch actions within
3-20 steps. Some games might require HOLDING an action for 50-200 steps
before any state change occurs (e.g., animation plays out, timer counts down,
character walks across screen). If the substrate switches too fast, it never
sees the delayed response.

This substrate holds EACH action for HOLD_DURATION steps before switching.
During the hold, it monitors raw pixel change. If ANY change is detected
(even at step 150 of holding), it marks the action as "delayed-responsive"
and enters exploitation mode on that action.

Architecture:
- Phase 1 (steps 0-HOLD*7): hold each keyboard action for HOLD_DURATION steps
  - During hold: compute |obs_t - obs_{t-1}| per step
  - Record total_change_per_action[a] = sum of all per-step changes during hold
  - Also record MAX single-step change (for delayed-spike detection)
- Phase 2 (steps HOLD*7 to HOLD*23): same for top-16 click regions
  - Only for click games (n_actions > 7)
- Phase 3 (remaining steps): exploit
  - Sort actions by total_change (delayed-responsive actions first)
  - Hold responsive actions for HOLD_DURATION, cycle through them
  - If NO responsive actions found → accept failure (Mode 1 game)

HOLD_DURATION = 200: ~5x longer than any prior substrate's max patience.
Budget: 7 keyboard × 200 = 1400 steps. 16 clicks × 200 = 3200. Total scan = 4600.
Exploitation: 5400 steps remaining.

KILL: ARC ≤ v30 (0.3319).
SUCCESS: Ultra-patience discovers delayed responses invisible to fast switching.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
HOLD_DURATION = 200
MAX_PATIENCE = 200  # match hold duration for exploitation


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    """Block center -> click action index."""
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class UltraPatientSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._enc_0 = None

        # Action scanning
        self._scan_actions = []  # actions to scan (keyboard + clicks)
        self._env_actions = []   # corresponding environment action indices
        self._scan_phase = True
        self._current_scan_idx = 0
        self._hold_step = 0  # steps held on current action

        # Per-action responsiveness
        self._total_change = {}  # action_idx → total ℓ₁ change during hold
        self._max_change = {}    # action_idx → max single-step change
        self._any_change = {}    # action_idx → bool (any change > threshold)

        # Exploitation
        self._responsive_actions = []
        self._exploit_idx = 0
        self._exploit_hold_step = 0

        # Click regions (discovered from first obs)
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
        """Find top-16 salient blocks for click targeting."""
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]

        # Build scan list: keyboard first, then clicks
        self._scan_actions = list(range(min(self._n_actions_env, N_KB)))
        self._env_actions = list(range(min(self._n_actions_env, N_KB)))
        if self._has_clicks:
            for ca in self._click_actions:
                self._scan_actions.append(len(self._env_actions))
                self._env_actions.append(ca)

        self._regions_set = True

    def _build_exploit_set(self):
        """After scanning: rank actions by responsiveness."""
        self._scan_phase = False
        CHANGE_THRESH = 0.5  # minimum total change to count as responsive

        responsive = []
        for i, env_action in enumerate(self._env_actions):
            total = self._total_change.get(i, 0.0)
            if total > CHANGE_THRESH:
                responsive.append((total, i, env_action))

        # Sort by total change, highest first
        responsive.sort(reverse=True)
        self._responsive_actions = [(idx, env_a) for _, idx, env_a in responsive]

        self.r3_updates += 1
        self.att_updates_total += 1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        # First observation
        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(enc)
            # Start holding first action
            self._current_scan_idx = 0
            self._hold_step = 0
            return self._env_actions[0] if self._env_actions else 0

        # Record change from previous step
        change = np.sum(np.abs(enc - self._prev_enc))

        # SCANNING PHASE: hold each action for HOLD_DURATION
        if self._scan_phase:
            idx = self._current_scan_idx
            if idx < len(self._env_actions):
                # Accumulate change for current action
                self._total_change[idx] = self._total_change.get(idx, 0.0) + change
                self._max_change[idx] = max(self._max_change.get(idx, 0.0), change)
                if change > 0.1:
                    self._any_change[idx] = True

                self._hold_step += 1

                if self._hold_step >= HOLD_DURATION:
                    # Move to next action
                    self._current_scan_idx += 1
                    self._hold_step = 0

                    if self._current_scan_idx >= len(self._env_actions):
                        # All actions scanned → build exploit set
                        self._build_exploit_set()
                    else:
                        self._prev_enc = enc.copy()
                        return self._env_actions[self._current_scan_idx]

                self._prev_enc = enc.copy()
                return self._env_actions[idx]
            else:
                self._build_exploit_set()

        # EXPLOITATION PHASE: cycle through responsive actions with long holds
        if self._responsive_actions:
            scan_idx, env_action = self._responsive_actions[self._exploit_idx]
            self._exploit_hold_step += 1

            # Check if making progress (distance-to-initial decreasing)
            dist = np.sum(np.abs(enc - self._enc_0))

            if self._exploit_hold_step >= HOLD_DURATION:
                # Switch to next responsive action
                self._exploit_idx = (self._exploit_idx + 1) % len(self._responsive_actions)
                self._exploit_hold_step = 0

            self._prev_enc = enc.copy()
            return env_action

        # No responsive actions found — keyboard-only reactive switching (fallback)
        n_active = min(self._n_actions_env, N_KB)
        action_idx = (self.step_count // 20) % n_active  # slow cycling
        self._prev_enc = enc.copy()
        return action_idx

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        # Keep scan results across levels — responsive actions persist
        self._scan_phase = False
        self._exploit_idx = 0
        self._exploit_hold_step = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "hold_duration": HOLD_DURATION,
    "n_click_regions": N_CLICK_REGIONS,
    "family": "ultra-patient persistence",
    "tag": "defense v55 (ℓ₁ ultra-patient: hold each action 200 steps, detect delayed responses)",
}

SUBSTRATE_CLASS = UltraPatientSubstrate
