"""
Step 1383 — Selective SSM (Mamba-style): input-dependent parameters bypass action-blind attractor.
Leo mail 3948, 2026-03-30.

1379-1381: Linear diagonal SSM is action-blind — B, C are CONSTANT matrices. The state update
x' = A*x + B@u has constant B, making action contribution state-independent. No gradient-based
patch can fix this because the action-blind solution is a stable attractor.

Fix: Selective SSM — make B, C, delta functions of the input u. When action changes, B_t changes,
writing DIFFERENT information to state. When action changes, C_t changes, reading DIFFERENT
features from state. delta_t changes too → different step size. This IS Mamba's selective scan.

Architecture (per layer):
  u_t = W_in @ concat(proj_obs, act_emb)      # shared input projection

  # Selection: all depend on u_t
  delta_t = softplus(W_delta @ u_t + b_delta)  # step size (N,)
  B_vec_t = W_B @ u_t                          # B direction (N,) [input-dependent write]
  C_vec_t = W_C @ u_t                          # C direction (N,) [input-dependent read]

  # Discretize
  A_bar_t = exp(-delta_t * |A_diag|)           # (N,) decay
  B_bar_t = delta_t * B_vec_t                   # (N,) effective write vector

  # State update
  h_{t+1} = A_bar_t * h_t + B_bar_t            # selective state update

  # Output
  y_t = W_out @ (C_vec_t * h_{t+1})            # (D,) — input-dependent readout

RTRL: local gradient for W_B, W_C, W_delta, W_out + S-trace for A_param.
W_in is updated per-step (local gradient from layer 1's backprop to input).

Mandatory first diagnostic: 3 draws of SELECTIVE + SELECTIVE_MASKED.
Compute action-blind ratio = pred_loss(SELECTIVE_MASKED) / pred_loss(SELECTIVE).
Gate: ratio > 1.05 required before full 30-draw experiment.

Conditions (30 draws, paired):
  SELECTIVE: Selective SSM + frozen projection try2. COUNT try1.
  LINEAR:    Standard linear SSM (= 1379 baseline). Frozen projection try2.
Both: PRNG fix, h reset, weights carry. Same W_fixed per (draw, game).

Seeds: 14080-14109 (fresh).
"""

import os
import sys
import json
import time
import math
import numpy as np

sys.path.insert(0, 'B:/M/the-search/experiments/compositions')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')
sys.path.insert(0, 'B:/M/the-search')

from prism_masked import (
    select_games, seal_mapping, label_filename, det_weights,
    compute_progress_speedup, compute_rhae_try2,
    mask_result_row, ARC_OPTIMAL_STEPS_PROXY,
    masked_game_list,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEP        = 1383
N_DRAWS     = 30
DRAW_SEEDS  = [14080 + i for i in range(N_DRAWS)]
N_DIAG_INIT = 3      # draws for mandatory first diagnostic
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM dimensions (same as 1378-1382)
PROJ_DIM      = 64
ACT_EMBED_DIM = 16
D_IN          = PROJ_DIM + ACT_EMBED_DIM   # 80: input before W_in
D             = 128                         # model dim (after W_in)
N_STATE       = 32
N_LAYERS      = 2
SSM_LR        = 1e-3
H_DIM         = N_LAYERS * N_STATE          # 64 — h_concat for frozen projection

# Frozen projection (same as 1378-1382)
TEMPERATURE   = 3.0

# Thresholds
ACTION_BLIND_THRESHOLD = 0.05   # ratio > 1.05 → uses action info

CONDITIONS     = ['SELECTIVE', 'LINEAR']
DIAG_CONDITION = 'SELECTIVE_MASKED'

MAX_N_ACTIONS = 4103
I3_STEP       = 200
I4_EARLY_MAX  = 100
I4_LATE_MIN   = 1900
TIER1_STEPS   = 200


# ---------------------------------------------------------------------------
# Game helpers
# ---------------------------------------------------------------------------

def make_game(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.make(gn)
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_optimal_steps(game_name, seed):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)
    return ARC_OPTIMAL_STEPS_PROXY


# ---------------------------------------------------------------------------
# Obs encoding
# ---------------------------------------------------------------------------

def _is_arc_obs(arr):
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _encode_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        frame = np.round(arr).astype(np.int32).squeeze(0)
        frame = np.clip(frame, 0, 15)
        one_hot = (frame[None, :, :] == np.arange(16, dtype=np.int32)[:, None, None]).astype(np.float32)
        return one_hot.flatten()
    return arr.flatten()


# ---------------------------------------------------------------------------
# Deterministic init helpers
# ---------------------------------------------------------------------------

def _det_init(m, n, scale=0.01):
    k = max(m, n)
    if k <= 256:
        W = det_weights(m, n)
        return (W * scale).astype(np.float32)
    else:
        i = np.arange(m, dtype=np.float64).reshape(-1, 1)
        j = np.arange(n, dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.7 + j * 2.3 + 1.0)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        return (W / (norms + 1e-8) * scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def binomial_p_one_sided(wins, n):
    if n <= 0:
        return 1.0
    total = sum(math.comb(n, k) for k in range(wins, n + 1))
    return round(float(total) * (0.5 ** n), 6)


def entropy_from_actions(action_list, n_actions):
    if not action_list:
        return None
    counts = np.bincount(action_list, minlength=n_actions).astype(np.float32)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def visit_cv(counts):
    mean_c = float(counts.mean())
    if mean_c <= 0:
        return None
    return float(counts.std() / mean_c)


# ---------------------------------------------------------------------------
# Selective SSM Layer
# ---------------------------------------------------------------------------

class SelectiveSSMLayer:
    """
    Mamba-style selective SSM: B_t, C_t, delta_t all depend on input x.

    Forward:
      delta_t = softplus(W_delta @ x + b_delta)   (N,)
      B_vec_t = W_B @ x                             (N,)
      C_vec_t = W_C @ x                             (N,)
      A_bar_t = exp(-delta_t * |A_diag|)            (N,)
      B_bar_t = delta_t * B_vec_t                   (N,)
      h       = A_bar_t * h_prev + B_bar_t          (N,)
      y       = W_out @ (C_vec_t * h)               (d,)

    RTRL: local 1-step gradient for W_B, W_C, W_delta, W_out + S-trace for A_param.
    """

    def __init__(self, d, n_state, lr):
        self.d = d
        self.n = n_state
        self.lr = lr

        # Selection parameters
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)
        self.W_B     = _det_init(n_state, d, scale=0.1)   # input-dependent B direction
        self.W_C     = _det_init(n_state, d, scale=0.1)   # input-dependent C direction
        self.W_out   = _det_init(d, n_state, scale=0.1)   # state-to-output projection
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5

        self._init_W_B = self.W_B.copy()

        # State
        self.h = np.zeros(n_state, dtype=np.float32)
        self.S_A = np.zeros(n_state, dtype=np.float32)   # S-trace for A_param

        # Cached for RTRL
        self._x = None
        self._h_prev = None
        self._A_bar = None
        self._delta = None
        self._pre_delta = None
        self._B_vec = None
        self._C_vec = None

    def forward(self, x):
        """x: (d,) → y: (d,)"""
        self._x = x
        self._h_prev = self.h.copy()

        pre_delta = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        self._pre_delta = pre_delta
        delta = np.log1p(np.exp(pre_delta))   # softplus
        self._delta = delta

        B_vec = self.W_B @ x   # (N,)
        C_vec = self.W_C @ x   # (N,)
        self._B_vec = B_vec
        self._C_vec = C_vec

        A_bar = np.exp(-delta * self.A_param)
        self._A_bar = A_bar

        B_bar = delta * B_vec   # (N,)
        self.h = A_bar * self._h_prev + B_bar
        self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)

        y = self.W_out @ (C_vec * self.h)   # (d,)
        return y

    def rtrl_update(self, e_y):
        """
        e_y: gradient from downstream, (d,).
        Returns e_x: gradient to input x, (d,).
        Updates W_B, W_C, W_delta, b_delta, W_out, A_param.
        """
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)

        e_y = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)

        # --- y = W_out @ (C_vec * h) ---
        # Gradient to (C_vec * h): W_out.T @ e_y
        d_C_h = self.W_out.T @ e_y            # (N,)
        d_C_h = np.nan_to_num(np.clip(d_C_h, -1e4, 1e4), nan=0.0)

        # Gradient to h through C_vec * h
        e_h = self._C_vec * d_C_h             # (N,) — gradient to h

        # Gradient to C_vec through C_vec * h
        e_C_vec = self.h * d_C_h              # (N,)
        e_C_vec = np.nan_to_num(np.clip(e_C_vec, -1e4, 1e4), nan=0.0)

        # Update W_out
        self.W_out -= self.lr * np.outer(e_y, self._C_vec * self.h)

        # Update W_C: C_vec = W_C @ x → ∂L/∂W_C = outer(e_C_vec, x)
        self.W_C -= self.lr * np.outer(e_C_vec, self._x)
        e_x_from_C = self.W_C.T @ e_C_vec    # (d,)

        # --- h = A_bar * h_prev + delta * B_vec ---
        e_h = np.nan_to_num(np.clip(e_h, -1e4, 1e4), nan=0.0)

        # S-trace for A_param (multi-step credit)
        self.S_A = self._A_bar * self.S_A + (-self._delta * self._A_bar * self._h_prev)
        self.A_param -= self.lr * (e_h * self.S_A)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)

        # Gradient to delta: from A_bar path AND B_bar path
        # A_bar = exp(-delta * A_param): ∂A_bar/∂delta = -A_param * A_bar
        # B_bar = delta * B_vec:         ∂B_bar/∂delta = B_vec
        e_delta = e_h * (self._h_prev * (-self.A_param * self._A_bar) + self._B_vec)
        e_delta = np.nan_to_num(np.clip(e_delta, -1e4, 1e4), nan=0.0)
        sigmoid_dp = 1.0 / (1.0 + np.exp(-self._pre_delta))
        e_pre_delta = e_delta * sigmoid_dp
        e_pre_delta = np.nan_to_num(np.clip(e_pre_delta, -1e4, 1e4), nan=0.0)

        # Update W_delta, b_delta
        self.W_delta -= self.lr * np.outer(e_pre_delta, self._x)
        self.b_delta -= self.lr * e_pre_delta
        e_x_from_delta = self.W_delta.T @ e_pre_delta   # (d,)

        # Gradient to B_vec: B_bar = delta * B_vec → ∂L/∂B_vec = delta * e_h
        e_B_vec = self._delta * e_h   # (N,)
        e_B_vec = np.nan_to_num(np.clip(e_B_vec, -1e4, 1e4), nan=0.0)

        # Update W_B: B_vec = W_B @ x
        self.W_B -= self.lr * np.outer(e_B_vec, self._x)
        e_x_from_B = self.W_B.T @ e_B_vec   # (d,)

        # Total gradient to x
        e_x = e_x_from_C + e_x_from_delta + e_x_from_B
        return e_x

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.W_B - self._init_W_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S_A[:] = 0.0
        self._x = self._h_prev = self._A_bar = self._delta = self._pre_delta = None
        self._B_vec = self._C_vec = None


# ---------------------------------------------------------------------------
# Linear SSM Layer (identical to 1379-1382, for LINEAR control)
# ---------------------------------------------------------------------------

class LinearSSMLayer:
    def __init__(self, d, n_state, lr):
        self.d = d
        self.n = n_state
        self.lr = lr

        self.B       = _det_init(n_state, d, scale=0.1)
        self.C       = _det_init(d, n_state, scale=0.1)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)

        self._init_B = self.B.copy()

        self.h = np.zeros(n_state, dtype=np.float32)
        self.S = np.zeros(n_state, dtype=np.float32)
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None

    def forward(self, x):
        self._x = x
        self._h_prev = self.h.copy()
        delta_pre = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        delta = np.log1p(np.exp(delta_pre))
        self._delta_pre = delta_pre
        self._delta = delta
        A_diag = np.exp(-delta * self.A_param)
        self._A_diag = A_diag
        self.h = A_diag * self._h_prev + self.B @ x
        self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)
        return self.C @ self.h

    def rtrl_update(self, e_y):
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)
        e_y = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)
        e_h = self.C.T @ e_y
        e_h = np.nan_to_num(np.clip(e_h, -1e4, 1e4), nan=0.0)
        e_x = self.B.T @ e_h
        self.C -= self.lr * np.outer(e_y, self.h)
        self.B -= self.lr * np.outer(e_h, self._x)
        self.S = self._A_diag * self.S + (-self._delta * self._A_diag * self._h_prev)
        self.A_param -= self.lr * (e_h * self.S)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)
        sigmoid_dp = 1.0 / (1.0 + np.exp(-self._delta_pre))
        dL_d_delta = e_h * (-self.A_param * self._A_diag * self._h_prev)
        dL_d_logit = dL_d_delta * sigmoid_dp
        self.W_delta -= self.lr * np.outer(dL_d_logit, self._x)
        self.b_delta -= self.lr * dL_d_logit
        return e_x

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.B - self._init_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None


# ---------------------------------------------------------------------------
# SSM Substrate
# ---------------------------------------------------------------------------

class SSMSubstrate:
    """
    mode: 'selective'         — SelectiveSSMLayer × N_LAYERS. Frozen projection try2.
          'selective_masked'  — same but action token zeroed in SSM input.
          'linear'            — LinearSSMLayer × N_LAYERS (= 1379 standard). Frozen projection try2.

    W_fixed: (n_actions, H_DIM) frozen projection.
    """

    def __init__(self, n_actions, mode='linear', W_fixed=None):
        self.n_actions = n_actions
        self.mode      = mode
        self._W_fixed  = W_fixed

        # Input projection (shared, fixed or updated)
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)
        self._zero_act  = np.zeros(ACT_EMBED_DIM, dtype=np.float32)
        self._W_in      = _det_init(D, D_IN, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)
        self._last_x_in = None   # cached for W_in update

        # Layers
        if mode in ('selective', 'selective_masked'):
            self._layers = [SelectiveSSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]
        else:
            self._layers = [LinearSSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]

        # Prediction head
        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        self._step         = 0
        self._prev_action  = 0
        self._prev_y       = None
        self._in_try2      = False
        self._visit_count  = np.zeros(n_actions, dtype=np.int32)
        self._obs_proj     = None

        self._try2_visit_at_i3   = None
        self._try2_actions_early = []
        self._try2_actions_late  = []
        self._try2_h_by_level    = {}
        self._current_level      = 0
        self._try2_max_level     = 0
        self._r3_diff            = None

        self._pred_losses      = []
        self._pred_losses_try2 = []

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim,  dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    def _ssm_forward(self, proj_obs):
        if self.mode == 'selective_masked':
            act_emb = self._zero_act
        else:
            act_emb = self._act_embed[self._prev_action]

        x_in = np.concatenate([proj_obs, act_emb])   # (D_IN,)
        self._last_x_in = x_in
        u = self._W_in @ x_in + self._b_in           # (D,)
        y = u
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y, action):
        """Update W_pred, prediction head, then RTRL through layers + W_in."""
        # Prediction head
        pred_obs   = self._W_pred @ y + self._b_pred
        pred_obs   = np.nan_to_num(pred_obs, nan=0.0, posinf=0.0, neginf=0.0)
        error_obs  = pred_obs - proj_obs_next
        error_safe = np.nan_to_num(np.clip(error_obs, -1e4, 1e4), nan=0.0)
        y_safe     = np.nan_to_num(np.clip(y, -1e4, 1e4), nan=0.0)

        fwd_loss = float(np.mean(error_safe ** 2))

        # Update W_pred, b_pred with safe gradient
        self._W_pred -= SSM_LR * np.outer(error_safe, y_safe)
        self._b_pred -= SSM_LR * error_safe

        # Backpropagate through layers (reverse order)
        e_y = np.nan_to_num(np.clip(self._W_pred.T @ error_safe, -1e4, 1e4), nan=0.0)
        for layer in reversed(self._layers):
            e_y = layer.rtrl_update(e_y)

        # Update W_in using gradient from first layer
        if self._last_x_in is not None:
            e_y_win = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)
            self._W_in -= SSM_LR * np.outer(e_y_win, self._last_x_in)
            self._b_in -= SSM_LR * e_y_win

        return fwd_loss

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat
        y = self._ssm_forward(proj_obs)
        self._prev_y = y

        if self._in_try2 and self._step % 200 == 0:
            h_concat = np.concatenate([l.h.copy() for l in self._layers])
            lvl = self._current_level
            if lvl not in self._try2_h_by_level:
                self._try2_h_by_level[lvl] = []
            if len(self._try2_h_by_level[lvl]) < 20:
                self._try2_h_by_level[lvl].append(h_concat)

        # Action selection: COUNT try1, frozen projection try2
        if self._in_try2:
            if self._W_fixed is not None:
                h_concat = np.concatenate([l.h for l in self._layers])
                h_concat = np.nan_to_num(h_concat, nan=0.0)
                logits = self._W_fixed @ h_concat / TEMPERATURE
                logits -= logits.max()
                probs = np.exp(logits); probs /= probs.sum()
                probs = np.nan_to_num(probs, nan=1.0 / self.n_actions)
                probs = np.clip(probs, 0.0, 1.0); probs /= probs.sum()
                action = int(np.random.choice(self.n_actions, p=probs))
            else:
                action = int(np.random.randint(self.n_actions))
        else:
            min_count  = self._visit_count.min()
            candidates = np.where(self._visit_count == min_count)[0]
            action     = int(np.random.choice(candidates))

        self._visit_count[action] += 1

        if self._in_try2 and self._step == I3_STEP:
            self._try2_visit_at_i3 = self._visit_count.copy()

        if self._in_try2:
            if self._step <= I4_EARLY_MAX:
                self._try2_actions_early.append(action)
            elif self._step >= I4_LATE_MIN:
                self._try2_actions_late.append(action)

        self._step += 1
        self._prev_action = action
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._prev_y is None or self._obs_proj is None:
            return
        obs_next_flat = _encode_obs(np.asarray(obs_next, dtype=np.float32))
        proj_obs_next = self._obs_proj @ obs_next_flat
        loss = self._rtrl_step(proj_obs_next, self._prev_y, action)
        self._pred_losses.append(loss)
        if self._in_try2:
            self._pred_losses_try2.append(loss)

    def on_level_transition(self, new_level=None):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None
        if new_level is not None:
            self._current_level = new_level
            if self._in_try2:
                self._try2_max_level = max(self._try2_max_level, new_level)

    def prepare_for_try2(self):
        diffs = [l.r3_weight_diff() for l in self._layers]
        self._r3_diff = round(float(np.mean(diffs)), 6)
        for layer in self._layers:
            layer.reset_state()
        self._prev_y         = None
        self._step           = 0
        self._prev_action    = 0
        self._in_try2        = True
        self._visit_count[:] = 0
        self._current_level  = 0

    def compute_stage_metrics(self):
        i3_cv = None
        if self._try2_visit_at_i3 is not None:
            cv = visit_cv(self._try2_visit_at_i3[:self.n_actions])
            i3_cv = round(cv, 4) if cv is not None else None

        h_early = entropy_from_actions(self._try2_actions_early, self.n_actions)
        h_late  = entropy_from_actions(self._try2_actions_late,  self.n_actions)
        i4_reduction = None
        if h_early and h_late and h_early > 0:
            i4_reduction = round((h_early - h_late) / h_early, 4)

        i1_within = i1_between = i1_pass = None
        levels_ok = [lvl for lvl, vecs in self._try2_h_by_level.items() if len(vecs) >= 2]
        if len(levels_ok) >= 2:
            within_dists, between_dists = [], []
            for lvl in levels_ok:
                vecs = np.array(self._try2_h_by_level[lvl])
                for a in range(len(vecs)):
                    for b in range(a + 1, len(vecs)):
                        within_dists.append(float(np.linalg.norm(vecs[a] - vecs[b])))
            all_vecs = {lvl: np.array(self._try2_h_by_level[lvl]) for lvl in levels_ok}
            for ii in range(len(levels_ok)):
                for jj in range(ii + 1, len(levels_ok)):
                    for va in all_vecs[levels_ok[ii]]:
                        for vb in all_vecs[levels_ok[jj]]:
                            between_dists.append(float(np.linalg.norm(va - vb)))
            if within_dists and between_dists:
                i1_within  = round(float(np.mean(within_dists)), 4)
                i1_between = round(float(np.mean(between_dists)), 4)
                i1_pass    = bool(i1_within < i1_between)

        return {
            'i3_cv':          i3_cv,
            'i4_h_early':     round(h_early, 4) if h_early else None,
            'i4_h_late':      round(h_late,  4) if h_late  else None,
            'i4_reduction':   i4_reduction,
            'i1_within':      i1_within,
            'i1_between':     i1_between,
            'i1_pass':        i1_pass,
            'i5_max_level':   self._try2_max_level,
            'r3_weight_diff': self._r3_diff,
        }

    def get_stats(self):
        pred_mean = round(float(np.mean(self._pred_losses)),      6) if self._pred_losses      else None
        pred_try2 = round(float(np.mean(self._pred_losses_try2)), 6) if self._pred_losses_try2 else None
        return {
            'pred_loss_mean': pred_mean,
            'pred_loss_try2': pred_try2,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs = env.reset(seed=seed)
    steps = 0
    level = 0
    steps_to_first_progress = None
    t_start = time.time()
    fresh_episode = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action  = substrate.process(obs_arr) % n_actions
        obs_next, reward, done, info = env.step(action)
        steps += 1
        if obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)
        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
            level = cl
            substrate.on_level_transition(new_level=cl)
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
        else:
            obs = obs_next

    return steps_to_first_progress, round(time.time() - t_start, 2)


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, cond_name, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}
    mode = cond_name.lower()   # 'selective', 'selective_masked', 'linear'

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        # Frozen projection: same seed formula as 1378-1382
        w_rng   = np.random.RandomState(draw_seed * 100 + game_idx)
        W_fixed = w_rng.randn(n_actions, H_DIM).astype(np.float32)

        substrate = SSMSubstrate(n_actions=n_actions, mode=mode, W_fixed=W_fixed)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)   # PRNG fix
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stats = substrate.get_stats()
        stage = substrate.compute_stage_metrics()

        speedup = compute_progress_speedup(p1, p2)
        opt     = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq  = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff_sq = round(min(1.0, opt / p2) ** 2, 6)

        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        row = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'condition': cond_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'pred_loss_mean':   stats['pred_loss_mean'],
            'pred_loss_try2':   stats['pred_loss_try2'],
            'i3_cv':            stage['i3_cv'],
            'i4_h_early':       stage['i4_h_early'],
            'i4_h_late':        stage['i4_h_late'],
            'i4_reduction':     stage['i4_reduction'],
            'i1_within':        stage['i1_within'],
            'i1_between':       stage['i1_between'],
            'i1_pass':          stage['i1_pass'],
            'i5_max_level':     stage['i5_max_level'],
            'r3_weight_diff':   stage['r3_weight_diff'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    pred_losses = [r['pred_loss_mean'] for r in draw_results if r.get('pred_loss_mean') is not None]
    draw_pred   = round(float(np.mean(pred_losses)), 6) if pred_losses else None
    pred_try2   = [r['pred_loss_try2'] for r in draw_results if r.get('pred_loss_try2') is not None]
    draw_pred_t2 = round(float(np.mean(pred_try2)), 6) if pred_try2 else None

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  pred={draw_pred}  pred_t2={draw_pred_t2}")
    return round(rhae, 7), draw_pred, draw_pred_t2, draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, _ = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    w_rng_t1   = np.random.RandomState(DRAW_SEEDS[0] * 100)
    W_fixed_t1 = w_rng_t1.randn(na_t1, H_DIM).astype(np.float32)

    sub_t1 = SSMSubstrate(n_actions=na_t1, mode='selective', W_fixed=W_fixed_t1)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    n_eps_main  = N_DRAWS * 3 * 2 * len(CONDITIONS)
    n_eps_diag  = N_DIAG_INIT * 3 * 2
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * (n_eps_main + n_eps_diag)
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * (n_eps_main + n_eps_diag))
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Mandatory first diagnostic: 3 draws of SELECTIVE vs SELECTIVE_MASKED ─
    print(f"\n=== MANDATORY DIAGNOSTIC: {N_DIAG_INIT} draws SELECTIVE vs SELECTIVE_MASKED ===")
    diag_sel_rhae  = []; diag_sel_pred  = []; diag_sel_pred_t2  = []
    diag_mask_rhae = []; diag_mask_pred = []; diag_mask_pred_t2 = []

    t_diag = time.time()
    for di in range(N_DIAG_INIT):
        ds = DRAW_SEEDS[di]
        rhae_s, pred_s, pred_t2_s, _ = run_draw(di, ds, 'SELECTIVE',        max_steps)
        rhae_m, pred_m, pred_t2_m, _ = run_draw(di, ds, 'SELECTIVE_MASKED', max_steps)
        diag_sel_rhae.append(rhae_s);  diag_sel_pred.append(pred_s);  diag_sel_pred_t2.append(pred_t2_s)
        diag_mask_rhae.append(rhae_m); diag_mask_pred.append(pred_m); diag_mask_pred_t2.append(pred_t2_m)
    print(f"  Diagnostic done in {time.time()-t_diag:.0f}s")

    # Compute action-blind ratio
    sel_preds_valid  = [p for p in diag_sel_pred_t2  if p is not None and not math.isnan(p)]
    mask_preds_valid = [p for p in diag_mask_pred_t2 if p is not None and not math.isnan(p)]

    if sel_preds_valid and mask_preds_valid and np.mean(sel_preds_valid) > 0:
        action_blind_ratio = round(float(np.mean(mask_preds_valid)) / float(np.mean(sel_preds_valid)), 4)
    else:
        action_blind_ratio = None

    print(f"\n  SELECTIVE   pred_try2: {diag_sel_pred_t2}")
    print(f"  MASKED      pred_try2: {diag_mask_pred_t2}")
    print(f"  Action-blind ratio (MASKED/SELECTIVE): {action_blind_ratio}")

    # Gate check
    if action_blind_ratio is None or action_blind_ratio < 1.0 + ACTION_BLIND_THRESHOLD:
        verdict = 'DIAGNOSTIC_FAIL'
        print(f"\n*** DIAGNOSTIC FAIL: ratio={action_blind_ratio} < {1.0 + ACTION_BLIND_THRESHOLD:.2f} ***")
        print("Selection mechanism is not forcing action conditioning. Debug before full run.")
        summary = {
            'step':               STEP,
            'verdict':            verdict,
            'action_blind_ratio': action_blind_ratio,
            'diag_draws':         N_DIAG_INIT,
            'diag_selective_pred_t2':  diag_sel_pred_t2,
            'diag_masked_pred_t2':     diag_mask_pred_t2,
            'note': 'Full experiment aborted — selection mechanism not action-conditional.',
        }
        summary_path = os.path.join(RESULTS_DIR, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nDiagnostic summary saved to {summary_path}")
        return

    print(f"\n  *** DIAGNOSTIC PASS: ratio={action_blind_ratio:.4f} > {1.0 + ACTION_BLIND_THRESHOLD:.2f} ***")
    print(f"  Proceeding to full {N_DRAWS}-draw experiment.")

    # ── Full experiment: SELECTIVE and LINEAR, 30 draws ───────────────────
    print(f"\n=== STEP {STEP}: SELECTIVE vs LINEAR, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond    = {c: [] for c in CONDITIONS}
    pred_by_cond    = {c: [] for c in CONDITIONS}
    pred_t2_by_cond = {c: [] for c in CONDITIONS}

    # Reuse first N_DIAG_INIT draws from diagnostic for SELECTIVE
    for di in range(N_DIAG_INIT):
        rhae_by_cond['SELECTIVE'].append(round(diag_sel_rhae[di], 7))
        pred_by_cond['SELECTIVE'].append(diag_sel_pred[di])
        pred_t2_by_cond['SELECTIVE'].append(diag_sel_pred_t2[di])
        print(f"  [SELECTIVE] Draw {di} (from diagnostic) RHAE={diag_sel_rhae[di]:.6e}")

    # Run remaining SELECTIVE draws
    print(f"\n--- Condition: SELECTIVE (draws {N_DIAG_INIT}-{N_DRAWS-1}) ---")
    t_cond = time.time()
    for di in range(N_DIAG_INIT, N_DRAWS):
        ds = DRAW_SEEDS[di]
        rhae, pred, pred_t2, _ = run_draw(di, ds, 'SELECTIVE', max_steps)
        rhae_by_cond['SELECTIVE'].append(round(rhae, 7))
        pred_by_cond['SELECTIVE'].append(pred)
        pred_t2_by_cond['SELECTIVE'].append(pred_t2)
    print(f"  SELECTIVE done in {time.time()-t_cond:.0f}s")

    # Run LINEAR (all 30 draws)
    print(f"\n--- Condition: LINEAR ---")
    t_cond = time.time()
    for di, ds in enumerate(DRAW_SEEDS):
        rhae, pred, pred_t2, _ = run_draw(di, ds, 'LINEAR', max_steps)
        rhae_by_cond['LINEAR'].append(round(rhae, 7))
        pred_by_cond['LINEAR'].append(pred)
        pred_t2_by_cond['LINEAR'].append(pred_t2)
    print(f"  LINEAR done in {time.time()-t_cond:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'step':               STEP,
        'n_draws':            N_DRAWS,
        'draw_seeds':         DRAW_SEEDS,
        'max_steps_used':     max_steps,
        'ms_per_step':        round(ms_per_step, 2),
        'mlp_tp_baseline':    MLP_TP_BASELINE,
        'action_blind_ratio': action_blind_ratio,
        'conditions':         {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list  = rhae_by_cond[cond_name]
        pred_list  = [p for p in pred_by_cond[cond_name] if p is not None]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        pred_mean  = round(float(np.mean(pred_list)), 6) if pred_list else None

        print(f"\n{cond_name}: chain_mean={chain_mean:.6e}  nz={nz}/{N_DRAWS}  pred_loss={pred_mean}")

        summary['conditions'][cond_name] = {
            'chain_mean': chain_mean,
            'nz':         nz,
            'rhae_list':  rhae_list,
            'pred_mean':  pred_mean,
        }

    # Paired sign test: SELECTIVE vs LINEAR
    sel_rhae = rhae_by_cond['SELECTIVE']
    lin_rhae = rhae_by_cond['LINEAR']
    wins   = sum(1 for s, l in zip(sel_rhae, lin_rhae) if s > l)
    losses = sum(1 for s, l in zip(sel_rhae, lin_rhae) if s < l)
    ties   = N_DRAWS - wins - losses
    p_val  = binomial_p_one_sided(wins, wins + losses)

    print(f"\nPaired sign test (SELECTIVE vs LINEAR):")
    print(f"  Wins={wins}  Losses={losses}  Ties={ties}  p={p_val:.4f}")
    print(f"\nAction-blind ratio (MASKED/SELECTIVE, {N_DIAG_INIT} diagnostic draws): {action_blind_ratio}")

    # Verdict
    if wins > losses and p_val <= 0.10:
        verdict = 'SIGNAL'
    elif wins <= losses:
        verdict = 'KILL'
    else:
        verdict = 'INCONCLUSIVE'

    print(f"\nVERDICT: {verdict}")
    print(f"MLP_TP_BASELINE:    {MLP_TP_BASELINE:.6e}")
    print(f"SELECTIVE chain_mean: {summary['conditions']['SELECTIVE']['chain_mean']:.6e}")
    print(f"LINEAR    chain_mean: {summary['conditions']['LINEAR']['chain_mean']:.6e}")

    summary['paired_wins']   = wins
    summary['paired_losses'] = losses
    summary['paired_ties']   = ties
    summary['p_value']       = p_val
    summary['verdict']       = verdict

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    main()
