"""
sub1106_prosecution_v27.py — Level-transition forward model

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1106 --substrate experiments/sub1106_prosecution_v27.py

FAMILY: Forward-model action selection (prosecution branch)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: W_fwd can detect level transitions via prediction error spike
AND selectively retain cross-level knowledge. Alpha determines which game
mechanics persist across levels. If action effects are preserved across levels,
W_fwd retains → instant correct action in new level. If effects change, W_fwd
partially resets → re-learns from scratch.

ONE CHANGE FROM v24: level-transition detection + selective W_fwd retention.
- Detect: prediction error spike > 5× running EMA → transition
- Retain: save pre-transition action-effect directions, compare after 20 steps
- If direction cosine < 0.3 → action's effect changed → partial W_fwd reset
- If direction cosine >= 0.3 → effect preserved → retain W_fwd

KILL: no level transitions detected OR retention doesn't help (same as v24).
SUCCESS: ANY L2 solve (max_lvl >= 2). Even 1/10 seeds.
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
WARMUP = 100
PRED_ERROR_DECAY = 0.99
TRANSITION_MULT = 5.0
DIRECTION_THRESH = 0.3
POST_TRANSITION_STEPS = 20


def _obs_to_enc(obs):
    """avgpool4 + center: 64x64 → 16x16 = 256D, zero-centered."""
    enc = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            enc[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    enc -= enc.mean()
    return enc


class LevelTransitionForwardModelSubstrate:
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

        # W_pred: predict next enc from current
        self._w_pred = np.eye(N_DIMS, dtype=np.float32) * 0.99

        # W_fwd: action-conditioned forward model (256 × 263)
        self._w_fwd = np.zeros((N_DIMS, INPUT_DIMS), dtype=np.float32)
        self._w_fwd[:N_DIMS, :N_DIMS] = np.eye(N_DIMS) * 0.99

        # Level-transition detection (v27 addition)
        self._pred_error_ema = 0.0
        self._transition_detected = False
        self._post_countdown = 0
        self._pre_directions = {}

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
        x = np.zeros(INPUT_DIMS, dtype=np.float32)
        x[:N_DIMS] = enc_w
        x[N_DIMS + action] = 1.0
        return x

    def _update_w_fwd(self, enc_w_actual, prev_enc_w, prev_action):
        x = self._make_input(prev_enc_w, prev_action)
        pred = self._w_fwd @ x
        error = enc_w_actual - pred
        norm_sq = np.dot(x, x) + 1e-8
        self._w_fwd += FWD_LR * np.outer(error, x) / norm_sq
        return float(np.sum(error ** 2))

    def _save_pre_directions(self, enc_w):
        """Save action-effect directions before level transition."""
        self._pre_directions = {}
        for a in range(self._n_actions):
            x = self._make_input(enc_w, a)
            pred = self._w_fwd @ x
            direction = pred - enc_w
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                self._pre_directions[a] = direction / norm

    def _check_direction_retention(self, enc_w):
        """Compare post-transition directions to pre-transition. Selective reset."""
        reset_any = False
        for a in self._pre_directions:
            x = self._make_input(enc_w, a)
            pred = self._w_fwd @ x
            new_direction = pred - enc_w
            norm = np.linalg.norm(new_direction)
            if norm > 1e-6:
                new_direction = new_direction / norm
                cosine = float(np.dot(self._pre_directions[a], new_direction))
                if cosine < DIRECTION_THRESH:
                    # Direction changed for this action — reset its column
                    self._w_fwd[:, N_DIMS + a] = 0.0
                    reset_any = True

        if reset_any:
            # Partial dampen enc columns (single application)
            self._w_fwd[:, :N_DIMS] *= 0.5

        self._pre_directions = {}
        self._transition_detected = False

    def _select_action(self, enc_w):
        if self.step_count < WARMUP:
            return self._rng.randint(self._n_actions)

        if self._rng.random() < EPSILON:
            return self._rng.randint(self._n_actions)

        scores = np.zeros(self._n_actions, dtype=np.float32)
        for a in range(self._n_actions):
            x = self._make_input(enc_w, a)
            predicted_next = self._w_fwd @ x
            scores[a] = float(np.sqrt(np.sum((predicted_next - enc_w) ** 2)))

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

            # Update forward model and get prediction error
            fwd_sq_error = self._update_w_fwd(enc_w, prev_enc_w, self._prev_action)

            # Update W_pred and alpha
            pred_error = self._update_w_pred(enc, self._prev_enc)
            self._update_alpha(pred_error)

            # Level-transition detection
            if self.step_count > WARMUP and not self._transition_detected:
                self._pred_error_ema = (
                    PRED_ERROR_DECAY * self._pred_error_ema +
                    (1 - PRED_ERROR_DECAY) * fwd_sq_error
                )
                if (self._pred_error_ema > 0 and
                        fwd_sq_error > TRANSITION_MULT * self._pred_error_ema):
                    # Transition detected — save directions
                    self._transition_detected = True
                    self._save_pre_directions(prev_enc_w)
                    self._post_countdown = POST_TRANSITION_STEPS

        # Post-transition countdown
        if self._post_countdown > 0:
            self._post_countdown -= 1
            if self._post_countdown == 0 and self._transition_detected:
                self._check_direction_retention(enc_w)

        action = self._select_action(enc_w)

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        # Keep forward model and alpha across levels (prosecution ℓ_π)
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
    "warmup": WARMUP,
    "pred_error_decay": PRED_ERROR_DECAY,
    "transition_mult": TRANSITION_MULT,
    "direction_thresh": DIRECTION_THRESH,
    "post_transition_steps": POST_TRANSITION_STEPS,
    "family": "forward-model action selection",
    "tag": "prosecution v27 (ℓ_π level-transition detection + selective W_fwd retention + avgpool4 256D)",
}

SUBSTRATE_CLASS = LevelTransitionForwardModelSubstrate
