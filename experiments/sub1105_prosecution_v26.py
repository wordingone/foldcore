"""
sub1105_prosecution_v26.py — Uncertainty-directed forward model

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1105 --substrate experiments/sub1105_prosecution_v26.py

FAMILY: Forward-model action selection (prosecution branch)
Tagged: prosecution (ℓ_π)
R3 HYPOTHESIS: Uncertainty-directed exploration discovers informative actions
on games where novelty-seeking fails. Alpha determines which prediction
uncertainties matter. v24 picks argmax(predicted_change) — repeats high-change
actions, oscillates. v26 picks argmax(prediction_variance) — explores the
LEAST KNOWN actions first, covering the action space more efficiently.

ONE CHANGE FROM v24: action selection criterion.
- v24: argmax(||predicted_next - current||) → novelty-seeking
- v26: argmax(pred_var) when uncertainty high → uncertainty-directed
       argmax(predicted_change) when uncertainty low → exploitation (same as v24)

EXPLOIT_THRESH = median(pred_var) after warmup (auto-calibrated).

KILL: 0/3 ARC AND no change in 0% game behavior (same actions as v24).
SUCCESS: ANY previously-0% game L1 > 0 OR action diversity increases on 0%.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
INPUT_DIMS = N_DIMS + N_KB    # 263
EPSILON_EXPLOIT = 0.20
EPSILON_EXPLORE = 0.10
PRED_LR = 0.001
FWD_LR = 0.001
ALPHA_LR = 0.01
ALPHA_CONC = 50.0
WARMUP = 100
PRED_VAR_DECAY = 0.95


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


class UncertaintyForwardModelSubstrate:
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
        self._w_fwd = np.zeros((N_DIMS, INPUT_DIMS), dtype=np.float32)
        self._w_fwd[:N_DIMS, :N_DIMS] = np.eye(N_DIMS) * 0.99

        # Uncertainty tracking (v26 addition)
        self._pred_var = {}  # action -> EMA of squared prediction error
        self._exploit_thresh = None  # set once after warmup

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

    def _select_action(self, enc_w):
        if self.step_count < WARMUP:
            return self._rng.randint(self._n_actions)

        # Set exploit threshold once after warmup
        if self._exploit_thresh is None:
            if len(self._pred_var) > 0:
                self._exploit_thresh = float(
                    np.median(list(self._pred_var.values())))
            else:
                self._exploit_thresh = 1.0

        # Get pred_var for all actions (default 1.0 = high uncertainty for untried)
        vars_all = np.array([
            self._pred_var.get(a, 1.0) for a in range(self._n_actions)
        ])

        if vars_all.max() > self._exploit_thresh:
            # High uncertainty exists — explore most uncertain action
            if self._rng.random() < EPSILON_EXPLORE:
                return self._rng.randint(self._n_actions)
            return int(np.argmax(vars_all))
        else:
            # All actions well-predicted — exploit (same as v24)
            if self._rng.random() < EPSILON_EXPLOIT:
                return self._rng.randint(self._n_actions)
            scores = np.zeros(self._n_actions, dtype=np.float32)
            for a in range(self._n_actions):
                x = self._make_input(enc_w, a)
                predicted_next = self._w_fwd @ x
                scores[a] = float(np.sqrt(
                    np.sum((predicted_next - enc_w) ** 2)))
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

            # Compute forward prediction error BEFORE update (for pred_var)
            x = self._make_input(prev_enc_w, self._prev_action)
            predicted = self._w_fwd @ x
            sq_error = float(np.sum((enc_w - predicted) ** 2))

            # Update pred_var for the action taken
            old_var = self._pred_var.get(self._prev_action, 1.0)
            self._pred_var[self._prev_action] = (
                PRED_VAR_DECAY * old_var +
                (1 - PRED_VAR_DECAY) * sq_error
            )

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
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "n_dims": N_DIMS,
    "input_dims": INPUT_DIMS,
    "epsilon_exploit": EPSILON_EXPLOIT,
    "epsilon_explore": EPSILON_EXPLORE,
    "pred_lr": PRED_LR,
    "fwd_lr": FWD_LR,
    "alpha_lr": ALPHA_LR,
    "alpha_conc": ALPHA_CONC,
    "warmup": WARMUP,
    "pred_var_decay": PRED_VAR_DECAY,
    "family": "forward-model action selection",
    "tag": "prosecution v26 (ℓ_π uncertainty-directed forward model + avgpool4 256D)",
}

SUBSTRATE_CLASS = UncertaintyForwardModelSubstrate
