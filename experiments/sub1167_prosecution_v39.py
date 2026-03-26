"""
sub1167_prosecution_v39.py — MI + attention update for L2 (prosecution: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1167 --substrate experiments/sub1167_prosecution_v39.py

FAMILY: MI attention (prosecution). Tagged: prosecution (ℓ_π).
R3 HYPOTHESIS: MI detection + alpha attention updates maintain useful
encoding structure across level transitions, enabling L2. The controlled
comparison: v67 (MI, ℓ₁, no L2) vs v39 (MI + alpha, ℓ_π, L2 target).
If v39 gets L2 and v67 doesn't → ℓ_π attention updates ARE the active
ingredient for L2.

Architecture (v16 base + avgpool4 for resolution):
- enc = avgpool4 (256D)
- MI detection: MI_d = 0.5 * log(var_total_d / mean_within_action_var_d)
- Alpha: softmax over |MI * prediction_error| per dim (ℓ_π)
- Alpha UPDATE: every step, alpha += lr * |pred_error| * MI_weight
- W_pred: 256×256 outer product update
- Action selection: reactive switching in alpha-weighted space
- On level transition: keep alpha + W_pred (cross-level retention → L2)

WHY THIS TARGETS L2:
- v67 (ℓ₁ MI, no alpha update): L1 perfect, L2 = 0
- v16 (ℓ_π MI + attention): L2 achieved (once)
- Alpha remembers which dims were informative → faster re-acquisition

Kill: no L2 across 3 draws.
Success: ANY L2 (max_lvl >= 2).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8

# MI parameters
MI_EMA = 0.95
MI_EPSILON = 1e-8
MI_WARMUP = 200        # steps before MI is computed
MI_RECOMPUTE = 100     # recompute MI every N steps
SUSTAIN_STEPS = 10     # hold each action during warmup

# Attention (ℓ_π) parameters
ATT_LR = 0.02          # attention update learning rate
ATT_MIN = 0.01
ATT_MAX = 1.0

# Prediction parameters
PRED_LR = 0.001        # W_pred learning rate


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class ProsecutionV39Substrate:
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

        self._n_active = N_KB

        # MI statistics (per-action, per-dim EMA)
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._prev_action_idx = 0

        # Alpha attention (ℓ_π — self-modifying)
        self._alpha = np.full(N_DIMS, 0.5, dtype=np.float32)

        # W_pred prediction matrix (256×256)
        self._W_pred = np.zeros((N_DIMS, N_DIMS), dtype=np.float32)
        self._pred_error = np.zeros(N_DIMS, dtype=np.float32)

        # Action selection (reactive in alpha-weighted space)
        self._current_action = 0
        self._patience = 0
        self._prev_weighted_dist = float('inf')

        # MI computed flag
        self._mi_ready = False

        # Click regions
        self._click_actions = []

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        # Reset per-game state but KEEP alpha and W_pred across games
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._prev_action_idx = 0
        self._current_action = 0
        self._patience = 0
        self._prev_weighted_dist = float('inf')
        self._mi_ready = False
        self._click_actions = []
        self._pred_error = np.zeros(N_DIMS, dtype=np.float32)

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

    def _compute_mi(self):
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)
        self._mi_ready = True

    def _update_alpha(self, pred_error):
        """ℓ_π: update attention from MI × prediction error."""
        if not self._mi_ready:
            return
        # Attention signal = |MI * pred_error|
        signal = np.abs(self._mi_values * pred_error)
        # Softmax-like normalization
        signal_max = signal.max()
        if signal_max < 1e-10:
            return
        signal_norm = signal / signal_max
        # Alpha update: shift attention toward informative+unpredicted dims
        self._alpha = np.clip(
            self._alpha + ATT_LR * signal_norm,
            ATT_MIN, ATT_MAX
        )
        # Normalize alpha to sum to N_DIMS (preserve total attention budget)
        alpha_sum = self._alpha.sum()
        if alpha_sum > 0:
            self._alpha *= N_DIMS / alpha_sum
        self.r3_updates += 1
        self.att_updates_total += 1

    def _update_prediction(self, prev_enc, enc):
        """Update W_pred via outer product rule."""
        pred = self._W_pred @ prev_enc
        self._pred_error = enc - pred
        # Outer product update: W += lr * error ⊗ input
        update = PRED_LR * np.outer(self._pred_error, prev_enc)
        self._W_pred += update
        # Clip to prevent explosion
        norm = np.linalg.norm(self._W_pred)
        if norm > 100.0:
            self._W_pred *= 100.0 / norm

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
            self._init_mi_arrays()
            return 0

        delta = enc - self._prev_enc

        # Update MI stats
        self._update_mi_stats(self._prev_action_idx, delta)

        # Update prediction model
        self._update_prediction(self._prev_enc, enc)

        # Update alpha attention (ℓ_π)
        self._update_alpha(self._pred_error)

        # Compute MI periodically
        if self.step_count % MI_RECOMPUTE == 0 or self.step_count == MI_WARMUP:
            self._compute_mi()

        # Warmup: sustained-hold cycling for MI estimation
        if self.step_count <= MI_WARMUP:
            action_idx = ((self.step_count - 1) // SUSTAIN_STEPS) % self._n_active
            self._prev_enc = enc.copy()
            self._prev_action_idx = action_idx
            return self._idx_to_env_action(action_idx)

        # Exploit: reactive switching in alpha-weighted space
        weighted_dist = float(np.sum(self._alpha * np.abs(enc - self._enc_0)))

        if weighted_dist >= self._prev_weighted_dist:
            self._current_action = (self._current_action + 1) % self._n_active
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action = (self._current_action + 1) % self._n_active

        self._prev_weighted_dist = weighted_dist
        self._prev_enc = enc.copy()
        self._prev_action_idx = self._current_action
        return self._idx_to_env_action(self._current_action)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_weighted_dist = float('inf')
        self._current_action = 0
        self._patience = 0
        self._pred_error = np.zeros(N_DIMS, dtype=np.float32)
        # KEEP alpha and W_pred across levels (cross-level retention → L2)
        # Reset MI for fresh estimation on new level
        if self._mi_mu is not None:
            self._mi_mu[:] = 0
            self._mi_var[:] = 1e-4
            self._mi_count[:] = 0
        self._mi_var_total[:] = 0
        self._mi_values[:] = 0
        self._mi_ready = False


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "mi_warmup": MI_WARMUP,
    "mi_recompute": MI_RECOMPUTE,
    "sustain_steps": SUSTAIN_STEPS,
    "mi_ema": MI_EMA,
    "att_lr": ATT_LR,
    "pred_lr": PRED_LR,
    "family": "MI attention (prosecution)",
    "tag": "prosecution v39 (ℓ_π MI + alpha attention + W_pred: targets L2 via cross-level attention retention. Controlled comparison vs defense v67.)",
}

SUBSTRATE_CLASS = ProsecutionV39Substrate
