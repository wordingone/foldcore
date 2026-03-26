"""
sub1162_defense_v68.py — Adaptive MI reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1162 --substrate experiments/sub1162_defense_v68.py

FAMILY: Adaptive MI reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v30 (rapid reactive) is FAST on easy games but blind on hard
games. v67 (MI-detected reactive) is SMART but wastes 300 steps on MI warmup
even when the game responds immediately to reactive cycling.

This substrate ADAPTS: start with v30's rapid reactive. If no L1 progress
after FALLBACK_THRESHOLD steps, switch to MI estimation (sustained holds)
then MI-ranked reactive cycling.

CONTROLLED COMPARISON:
- vs v30: SAME initial behavior (rapid reactive). DIFFERENT: MI fallback.
- vs v67: SAME MI detection on hard games. DIFFERENT: no warmup on easy games.

Expected: match v30 speed on easy games, beat v30 on hard games via MI.
If v30 solves games in <200 steps, v68 matches. If game needs MI, v68
detects it and switches.

ZERO learned parameters (defense: ℓ₁). Fixed adaptive protocol.

KILL: ARC ≤ v30.
SUCCESS: Adaptive > either pure approach.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
FALLBACK_THRESHOLD = 200   # switch to MI after this many steps with no L1
MI_SUSTAIN = 15             # hold each action for 15 steps during MI estimation
MI_EST_STEPS = 300          # total MI estimation steps
MI_EMA = 0.95
MI_EPSILON = 1e-8
TOP_K = 5


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class AdaptiveMIReactiveSubstrate:
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
        self._best_dist = float('inf')

        self._n_active = N_KB

        # Phase tracking
        self._phase = 'reactive'  # 'reactive' → 'mi_est' → 'mi_exploit'
        self._reactive_improved = False

        # v30-style reactive state
        self._current_action = 0
        self._patience = 0

        # MI estimation state
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_est_start = 0

        # MI exploit state
        self._best_actions = []
        self._mi_action_pos = 0

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

    def _init_mi_arrays(self):
        self._mi_mu = np.zeros((self._n_active, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((self._n_active, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(self._n_active, dtype=np.float32)

    def _update_mi_stats(self, action_idx, delta):
        if self._mi_mu is None or action_idx >= len(self._mi_mu):
            return
        alpha = 1.0 - MI_EMA
        self._mi_count[action_idx] += 1
        self._mi_mu[action_idx] = MI_EMA * self._mi_mu[action_idx] + alpha * delta
        residual = delta - self._mi_mu[action_idx]
        self._mi_var[action_idx] = MI_EMA * self._mi_var[action_idx] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta ** 2)

    def _compute_mi_and_rank(self):
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            self._best_actions = list(range(min(self._n_active, N_KB)))
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

        scores = []
        for a in range(self._n_active):
            score = float(np.sum(self._mi_values * np.abs(self._mi_mu[a])))
            scores.append((score, a))
        scores.sort(reverse=True)
        self._best_actions = [a for s, a in scores[:TOP_K] if s > 0.001]
        if not self._best_actions:
            self._best_actions = list(range(min(self._n_active, N_KB)))
        self._mi_action_pos = 0
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

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(enc)
            return 0

        dist = np.sum(np.abs(enc - self._enc_0))
        delta = enc - self._prev_enc

        # Track best distance seen
        if dist < self._best_dist:
            self._best_dist = dist
            self._reactive_improved = True

        # === PHASE: REACTIVE (v30-style, fast) ===
        if self._phase == 'reactive':
            # Check fallback trigger
            if self.step_count > FALLBACK_THRESHOLD and not self._reactive_improved:
                # No improvement after threshold — switch to MI estimation
                self._phase = 'mi_est'
                self._mi_est_start = self.step_count
                self._init_mi_arrays()
                # Fall through to mi_est handling below
            else:
                # Standard v30 reactive
                if dist >= self._prev_dist:
                    self._current_action = (self._current_action + 1) % self._n_active
                    self._patience = 0
                else:
                    self._patience += 1
                    if self._patience > 10:
                        self._patience = 0
                        self._current_action = (self._current_action + 1) % self._n_active

                self._prev_dist = dist
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(self._current_action)

        # === PHASE: MI ESTIMATION (sustained holds) ===
        if self._phase == 'mi_est':
            est_step = self.step_count - self._mi_est_start
            action_idx = (est_step // MI_SUSTAIN) % self._n_active
            self._update_mi_stats(action_idx, delta)

            if est_step >= MI_EST_STEPS:
                self._compute_mi_and_rank()
                self._phase = 'mi_exploit'

            self._prev_dist = dist
            self._prev_enc = enc.copy()
            return self._idx_to_env_action(action_idx)

        # === PHASE: MI EXPLOIT (reactive over MI-ranked actions) ===
        current_action = self._best_actions[self._mi_action_pos]

        if dist >= self._prev_dist:
            self._mi_action_pos = (self._mi_action_pos + 1) % len(self._best_actions)
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._mi_action_pos = (self._mi_action_pos + 1) % len(self._best_actions)

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        return self._idx_to_env_action(self._best_actions[self._mi_action_pos])

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._best_dist = float('inf')
        self._current_action = 0
        self._patience = 0
        self._reactive_improved = False
        # Reset to reactive phase for new level — fast start
        self._phase = 'reactive'
        # Keep MI arrays if computed (cross-level transfer)
        if self._mi_mu is not None:
            self._mi_mu[:] = 0
            self._mi_var[:] = 1e-4
            self._mi_count[:] = 0
            self._mi_var_total[:] = 0
            self._mi_values[:] = 0
        self._mi_action_pos = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "fallback_threshold": FALLBACK_THRESHOLD,
    "mi_sustain": MI_SUSTAIN,
    "mi_est_steps": MI_EST_STEPS,
    "mi_ema": MI_EMA,
    "top_k": TOP_K,
    "family": "adaptive MI reactive",
    "tag": "defense v68 (ℓ₁ adaptive: start v30-fast, fallback to MI estimation if no progress after 200 steps. Best of both: speed on easy games, MI on hard games.)",
}

SUBSTRATE_CLASS = AdaptiveMIReactiveSubstrate
