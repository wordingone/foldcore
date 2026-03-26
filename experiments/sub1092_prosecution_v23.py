"""
sub1092_prosecution_v23.py — Forward model action selection

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1092 --substrate experiments/sub1092_prosecution_v23.py

FAMILY: Forward-model action selection (NEW — not attention-trajectory)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: W_fwd forward model selects actions by predicting which action
causes the most change in alpha-weighted space. Alpha determines what "change"
means → change alpha → change action selection. Simplest possible ℓ_π: one
forward model, one argmax, no history. PB30 says simplicity wins.

ARCHITECTURE:
- enc = avgpool16 + centered (256D)
- W_pred: 256×256 outer product (prediction error → alpha, same as v20)
- Alpha: softmax concentration from W_pred error (ℓ_π)
- W_fwd: 256×263 action-conditioned forward model
  - Input: [enc_weighted(256D), one_hot_action(7D)] = 263D
  - Output: predicted_next_enc_weighted (256D)
  - Update: normalized outer product from actual vs predicted
- Action selection:
  - For each action a: predict next state, measure predicted change
  - score[a] = ||predicted_next - enc_weighted||
  - action = argmax(score) with 20% epsilon

WHY NEW FAMILY:
- No buffer (v20-v22 had 2000-entry trajectory buffer)
- No attention retrieval (v20-v22 used softmax attention)
- No reactive comparison (defense used dist-to-initial)
- Pure forward prediction + argmax

KILL: 0/3 ARC games AND ARC=0.
SUCCESS: any ARC > 0 (prosecution signal from new family).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
INPUT_DIMS = N_DIMS + N_KB    # 263
EPSILON = 0.20
PRED_LR = 0.001
FWD_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0


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


class ForwardModelSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = None

        # Alpha (ℓ_π)
        self._alpha = np.ones(N_DIMS, dtype=np.float32) / N_DIMS

        # W_pred: predict next enc from current (for alpha updates)
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # W_fwd: action-conditioned forward model (256 × 263)
        # Initialize: identity for enc dims, zeros for action dims
        self._w_fwd = np.zeros((N_DIMS, INPUT_DIMS), dtype=np.float32)
        self._w_fwd[:N_DIMS, :N_DIMS] = np.eye(N_DIMS) * 0.99

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _alpha_weight(self, enc):
        return enc * self._alpha

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

    def _make_input(self, enc_w, action):
        """Create forward model input: [enc_weighted, one_hot_action]."""
        x = np.zeros(INPUT_DIMS, dtype=np.float32)
        x[:N_DIMS] = enc_w
        x[N_DIMS + action] = 1.0
        return x

    def _update_w_fwd(self, enc_w_actual, prev_enc_w, prev_action):
        """Update action-conditioned forward model."""
        x = self._make_input(prev_enc_w, prev_action)
        pred = self._w_fwd @ x
        error = enc_w_actual - pred
        norm_sq = np.dot(x, x) + 1e-8
        self._w_fwd += FWD_LR * np.outer(error, x) / norm_sq

    def _select_action(self, enc_w):
        """Select action by predicting which causes most change."""
        if self.step_count < 10:
            return self._rng.randint(self._n_actions)

        if self._rng.random() < EPSILON:
            return self._rng.randint(self._n_actions)

        scores = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            x = self._make_input(enc_w, a)
            predicted_next = self._w_fwd @ x
            predicted_change = float(np.sqrt(np.sum((predicted_next - enc_w) ** 2)))
            scores[a] = predicted_change

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

        if self._prev_enc is not None and self._prev_action is not None:
            prev_enc_w = self._alpha_weight(self._prev_enc)

            # Update forward model
            self._update_w_fwd(enc_w, prev_enc_w, self._prev_action)

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

        action = self._select_action(enc_w)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Keep forward model and alpha across levels
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "n_dims": N_DIMS,
    "input_dims": INPUT_DIMS,
    "epsilon": EPSILON,
    "pred_lr": PRED_LR,
    "fwd_lr": FWD_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "family": "forward-model action selection",
    "tag": "prosecution v23 (ℓ_π forward model + argmax, no buffer, no attention)",
}

SUBSTRATE_CLASS = ForwardModelSubstrate
