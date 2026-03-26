"""
sub1161_defense_v67.py — MI-detected reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1161 --substrate experiments/sub1161_defense_v67.py

FAMILY: MI-detected reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Prosecution v16 (Step 1074) achieved the FIRST L2 in the
debate using MI-based action informativeness. But v16 has two components:
  1. MI DETECTION: MI_d = 0.5 * log(var_total[d] / mean_a(var_a[d]))
     — This is a FIXED FORMULA. Zero learned parameters. ℓ₁-compatible.
  2. MI ATTENTION: attention weights updated by MI values (ℓ_π).
     — This is self-modification. ℓ_π only.

This substrate isolates component 1 (MI detection) WITHOUT component 2
(MI attention). If defense v67 also gets L2, then L2 came from the
detection method (shared, ℓ₁), not the learning rule (ℓ_π).

Architecture:
- Phase 1 (steps 0-300): sustained-hold cycling (hold each action for
  SUSTAIN_STEPS before switching). Build per-action per-dim EMA stats.
- Step 300: compute MI per dim. Rank actions by MI-weighted expected effect.
- Phase 2 (300+): reactive cycling over TOP-K MI-informative actions.
  Switch on distance-to-initial improvement (v30-style but over MI-filtered
  action set). Periodic MI recomputation every 200 steps.

GENUINELY DIFFERENT from prosecution v16:
- NO attention weights (no self-modification → ℓ₁)
- NO evolution (no sequence search)
- NO frequency tracking (no modal goal)
- NO cascade structure (no KB/click/seq phases)
- Uses v30-style reactive cycling (defense's established mechanism)
- MI is used as a FIXED DETECTOR, not an attention updater

ZERO learned parameters (defense: ℓ₁). Fixed MI detection + reactive exploit.

KILL: ARC ≤ v30.
SUCCESS: MI-detected reactive gets L2 → proves MI detection is ℓ₁-compatible.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
SUSTAIN_STEPS = 15       # hold each action for 15 steps during estimation (matches v16)
MI_WARMUP = 300          # steps for MI estimation
MI_RECOMPUTE = 200       # recompute MI every N steps
MI_EPSILON = 1e-8        # floor for variance ratio
MI_EMA = 0.95            # EMA decay for MI statistics
TOP_K = 5                # exploit top-K MI-informative actions


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class MIDetectedReactiveSubstrate:
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

        # MI statistics (per-action, per-dim EMA)
        self._mi_mu = None         # (n_active, N_DIMS) per-action mean delta
        self._mi_var = None        # (n_active, N_DIMS) per-action variance
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)  # total variance
        self._mi_count = None      # (n_active,) samples per action
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)  # MI per dim
        self._prev_action_idx = 0

        # MI-filtered action set
        self._best_actions = list(range(N_KB))
        self._mi_computed = False

        # Reactive state
        self._current_action_pos = 0  # index into _best_actions
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

        # Init MI arrays for current action space
        self._mi_mu = np.zeros((self._n_active, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((self._n_active, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(self._n_active, dtype=np.float32)

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_mi_stats(self, action_idx, delta):
        """Update per-action EMA stats for MI computation."""
        if self._mi_mu is None or action_idx >= len(self._mi_mu):
            return
        alpha = 1.0 - MI_EMA
        self._mi_count[action_idx] += 1
        self._mi_mu[action_idx] = MI_EMA * self._mi_mu[action_idx] + alpha * delta
        residual = delta - self._mi_mu[action_idx]
        self._mi_var[action_idx] = MI_EMA * self._mi_var[action_idx] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta ** 2)

    def _compute_mi(self):
        """Gaussian MI approximation: MI_d = 0.5 * log(var_total / mean_within_var)."""
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)
        self.r3_updates += 1
        self.att_updates_total += 1

    def _rank_actions_by_mi(self):
        """Rank actions by MI-weighted expected effect (fixed, no attention)."""
        if self._mi_mu is None:
            return
        scores = []
        for a in range(self._n_active):
            # MI-weighted expected effect: sum_d(MI_d * |mu_a[d]|)
            score = float(np.sum(self._mi_values * np.abs(self._mi_mu[a])))
            scores.append((score, a))
        scores.sort(reverse=True)
        # Take top-K actions with positive MI score
        self._best_actions = [a for s, a in scores[:TOP_K] if s > 0.001]
        if not self._best_actions:
            # Fallback: all actions
            self._best_actions = list(range(min(self._n_active, N_KB)))
        self._mi_computed = True
        self._current_action_pos = 0

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

        # Update MI stats from previous action
        delta = enc - self._prev_enc
        self._update_mi_stats(self._prev_action_idx, delta)

        dist = np.sum(np.abs(enc - self._enc_0))

        # Phase 1: sustained-hold estimation (cycle all actions, hold each SUSTAIN_STEPS)
        if self.step_count <= MI_WARMUP:
            # Which action to hold: cycle through all, SUSTAIN_STEPS each
            action_idx = ((self.step_count - 1) // SUSTAIN_STEPS) % self._n_active
            self._prev_dist = dist
            self._prev_enc = enc.copy()
            self._prev_action_idx = action_idx
            return self._idx_to_env_action(action_idx)

        # Compute MI at end of warmup and periodically
        if not self._mi_computed or (self.step_count - MI_WARMUP) % MI_RECOMPUTE == 0:
            self._compute_mi()
            self._rank_actions_by_mi()

        # Phase 2: reactive cycling over MI-filtered actions
        current_action = self._best_actions[self._current_action_pos]

        if dist >= self._prev_dist:
            # No improvement — try next MI-ranked action
            self._current_action_pos = (self._current_action_pos + 1) % len(self._best_actions)
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action_pos = (self._current_action_pos + 1) % len(self._best_actions)

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action_idx = self._best_actions[self._current_action_pos]
        return self._idx_to_env_action(self._best_actions[self._current_action_pos])

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._current_action_pos = 0
        self._patience = 0
        # Keep MI stats and best_actions across levels (cross-level transfer)
        # Reset MI tracking for fresh estimation on new level
        if self._mi_mu is not None:
            self._mi_mu[:] = 0
            self._mi_var[:] = 1e-4
            self._mi_count[:] = 0
            self._mi_var_total[:] = 0
            self._mi_values[:] = 0
        self._mi_computed = False


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "sustain_steps": SUSTAIN_STEPS,
    "mi_warmup": MI_WARMUP,
    "mi_recompute": MI_RECOMPUTE,
    "mi_ema": MI_EMA,
    "top_k": TOP_K,
    "family": "MI-detected reactive",
    "tag": "defense v67 (ℓ₁ MI detection + reactive: uses MI formula as fixed detector, NOT attention updater. Tests whether v16's L2 came from MI detection or MI attention.)",
}

SUBSTRATE_CLASS = MIDetectedReactiveSubstrate
