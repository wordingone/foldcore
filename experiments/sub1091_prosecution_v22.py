"""
sub1091_prosecution_v22.py — Directional attention-trajectory

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1091 --substrate experiments/sub1091_prosecution_v22.py

FAMILY: Attention-trajectory (prosecution-only architecture)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: Storing progress DIRECTION (toward/away from initial) instead
of raw change magnitude improves action efficiency. Alpha-weighted encoding
determines both "similar state" (attention keys) and "meaningful progress"
(alpha focuses on informative dims). ℓ_π self-modifies what "progress" means.

ONE CHANGE FROM v20 (Step 1085):
- Old V: delta = ||enc_w_{t+1} - enc_w_t|| (magnitude, always positive)
- New V: progress = dist_to_initial(prev) - dist_to_initial(curr) (signed)
  - Positive = moved toward initial (progress for some games)
  - Negative = moved away (progress for other games)
  - Score uses |progress| (direction-agnostic)
  - Buffer drops entries with smallest |progress|

Everything else IDENTICAL to v20: buffer 2000, W_pred outer product,
alpha from prediction error, 20% epsilon, scaled dot-product attention.

KILL: ARC < v20 (0.0037) → direction hurts.
SUCCESS: ARC > 0.01 (3x improvement).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
BUFFER_MAX = 2000
EPSILON = 0.20
PRED_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
ATTN_TEMP = np.sqrt(256.0)


def _obs_to_enc(obs):
    """avgpool16 + center: 64x64 → 16x16 = 256D, zero-centered."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


class DirectionalAttentionSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None          # raw initial encoding (unweighted)
        self._prev_enc = None       # raw previous encoding
        self._prev_enc_w = None     # alpha-weighted previous encoding
        self._prev_action = None

        # Trajectory buffer
        self._buf_enc = np.zeros((BUFFER_MAX, N_DIMS), dtype=np.float32)
        self._buf_action = np.zeros(BUFFER_MAX, dtype=np.int32)
        self._buf_progress = np.zeros(BUFFER_MAX, dtype=np.float32)  # SIGNED progress
        self._buf_size = 0

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS
        # W_pred
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _alpha_weight(self, enc):
        return enc * self._alpha

    def _dist_to_initial_weighted(self, enc_w):
        """L1 distance to initial in alpha-weighted space."""
        if self._enc_0 is None:
            return 0.0
        enc_0_w = self._alpha_weight(self._enc_0)
        return float(np.sum(np.abs(enc_w - enc_0_w)))

    def _update_alpha(self, pred_error):
        error_mag = np.abs(pred_error)
        emax = error_mag.max()
        if emax < 1e-8:
            return
        logits = ALPHA_CONC * error_mag / emax
        logits -= logits.max()
        exp_l = np.exp(logits)
        target = exp_l / (exp_l.sum() + 1e-8)
        self._alpha = (1 - ALPHA_LR) * self._alpha + ALPHA_LR * target
        self.r3_updates += 1
        self.att_updates_total += 1

    def _update_w_pred(self, enc, prev_enc):
        pred = self._w_pred @ prev_enc
        error = enc - pred
        norm_sq = np.dot(prev_enc, prev_enc) + 1e-8
        self._w_pred += PRED_LR * np.outer(error, prev_enc) / norm_sq
        return error

    def _add_to_buffer(self, enc_w, action, progress):
        """Add to buffer. Drop least informative (smallest |progress|) if full."""
        abs_progress = abs(progress)
        if self._buf_size < BUFFER_MAX:
            idx = self._buf_size
            self._buf_size += 1
        else:
            idx = int(np.argmin(np.abs(self._buf_progress[:self._buf_size])))
            if abs_progress <= abs(self._buf_progress[idx]):
                return
        self._buf_enc[idx] = enc_w
        self._buf_action[idx] = action
        self._buf_progress[idx] = progress  # SIGNED

    def _select_action(self, enc_w):
        """Attention-based action selection. Score = |progress| weighted by state similarity."""
        if self._buf_size < self._n_actions * 3:
            return self._rng.randint(self._n_actions)

        if self._rng.random() < EPSILON:
            return self._rng.randint(self._n_actions)

        q = enc_w
        scores = np.zeros(self._n_actions, dtype=np.float32)
        valid_actions = 0

        for a in range(self._n_actions):
            mask = self._buf_action[:self._buf_size] == a
            n_entries = int(mask.sum())
            if n_entries < 2:
                scores[a] = 0.0
                continue
            valid_actions += 1

            K_a = self._buf_enc[:self._buf_size][mask]
            V_a = self._buf_progress[:self._buf_size][mask]  # signed progress

            # Scaled dot-product attention
            attn_logits = K_a @ q / ATTN_TEMP
            attn_logits -= attn_logits.max()
            attn_weights = np.exp(attn_logits)
            attn_weights /= attn_weights.sum() + 1e-8

            # Score = attention-weighted mean of |progress| (direction-agnostic)
            scores[a] = float(attn_weights @ np.abs(V_a))

        if valid_actions < 2:
            return self._rng.randint(self._n_actions)

        return int(np.argmax(scores))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        enc_w = self._alpha_weight(enc)

        # Store initial raw encoding
        if self._enc_0 is None:
            self._enc_0 = enc.copy()

        if self._prev_enc is not None and self._prev_action is not None:
            prev_enc_w = self._alpha_weight(self._prev_enc)

            # Signed progress: positive = moved toward initial in alpha space
            prev_dist = self._dist_to_initial_weighted(prev_enc_w)
            curr_dist = self._dist_to_initial_weighted(enc_w)
            progress = prev_dist - curr_dist  # positive = closer to initial

            self._add_to_buffer(prev_enc_w, self._prev_action, progress)

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        action = self._select_action(enc_w)

        self._prev_enc = enc.copy()
        self._prev_enc_w = enc_w.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_enc_w = None
        self._prev_action = None


CONFIG = {
    "buffer_max": BUFFER_MAX,
    "epsilon": EPSILON,
    "pred_lr": PRED_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "n_dims": N_DIMS,
    "family": "attention-trajectory (directional)",
    "tag": "prosecution v22 (ℓ_π directional progress, signed V, attention retrieval)",
}

SUBSTRATE_CLASS = DirectionalAttentionSubstrate
