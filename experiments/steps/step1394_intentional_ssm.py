"""
Step 1394 — Intentional SSM: state-conditioned gates.
Leo mail 3995 (spec) + 3997 (gate fixes), 2026-03-30. All 11 gates passed.

ANIMA separation theorem: reactive SSMs (Δ,B,C = f(x_t)) can't route based on
accumulated state — their gates are blind to history. FIX: make gates INTENTIONAL
by conditioning on previous output y_{t-1} = f(h_{t-1}).

ONE architectural change from Steps 1379-1391:
  REACTIVE (1379+): gate_input = u_t
  INTENTIONAL (1394): gate_input = concat(y_prev, u_t)  [2*D dims]

W_delta, W_B, W_C all double their input dimension. Everything else identical.

Why this breaks action-blindness: y_prev = W_out @ (C @ h_{t-1}) encodes accumulated
state. If h has learned action-relevant structure, delta_t = f(y_prev, x_t) can be
different depending on prior actions — routing is now history-dependent.

Diagnostic: INTENTIONAL vs INTENTIONAL-MASKED (y_prev zeroed = reactive fallback).
  Pass: action-blind ratio (pred_loss_MASKED / pred_loss_INT) > 1.05
  Fail: intentional gates don't help prediction → same attractor as 1379-1391.

Full: INTENTIONAL vs DISCONNECTED vs CONTROL_C, 30 draws.
  DISCONNECTED: random try1 actions, act_emb zeroed, SSM trains on obs only.
  CONTROL_C: argmin throughout, no SSM, no learning.
  Kill: INTENTIONAL ≤ DISCONNECTED (paired sign test, p > 0.10) → KILL.

Seeds: 14470-14499. 30 draws. Max 2000 steps.
"""

import os
import sys
import json
import time
import math
import numpy as np

sys.path.insert(0, 'B:/M/the-search/experiments/compositions')
sys.path.insert(0, 'B:/M/the-search/experiments/steps')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/experiments/environments')
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

STEP        = 1394
N_DRAWS     = 30
DRAW_SEEDS  = [14470 + i for i in range(N_DRAWS)]
N_DIAG_INIT = 3
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280
TIER1_STEPS = 200

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM dimensions — D=32, N=16 (smaller than 1391 to offset doubled gate width)
PROJ_DIM        = 64
SPATIAL_ACT_DIM = 9      # [type_onehot(7), x_norm, y_norm]
D_IN            = PROJ_DIM + SPATIAL_ACT_DIM   # 73
D               = 32
N_STATE         = 16
N_LAYERS        = 2
SSM_LR          = 1e-3
GATE_DIM        = 2 * D   # 64 — gate_input = [y_prev; u_t]
H_DIM           = N_LAYERS * N_STATE   # 32

TEMPERATURE           = 3.0
ACTION_BLIND_THRESHOLD = 0.05   # ratio > 1.05 → intentional gates break action-blind attractor

CONDITIONS  = ['INTENTIONAL', 'DISCONNECTED', 'CONTROL_C']
DIAG_COND   = 'INTENTIONAL-MASKED'

MAX_N_ACTIONS = 4103
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
# Obs / action encoding
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


def _encode_action_spatial(action_id, masked=False):
    """9-dim spatial encoding. Zeroed if masked=True."""
    enc = np.zeros(SPATIAL_ACT_DIM, dtype=np.float32)
    if masked:
        return enc
    if action_id < 7:
        enc[action_id] = 1.0
    else:
        click_idx = action_id - 7
        x = click_idx % 64
        y = click_idx // 64
        enc[7] = x / 63.0
        enc[8] = y / 63.0
    return enc


# ---------------------------------------------------------------------------
# Deterministic init
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


# ---------------------------------------------------------------------------
# Intentional SSM Layer
# State-conditioned gates: gate_input = concat(y_prev, u_t)
# ---------------------------------------------------------------------------

class IntentionalSSMLayer:
    """
    Diagonal SSM with intentional (state-conditioned) gates.
    W_delta, W_B, W_C take gate_input = [y_prev; u_t] of dimension GATE_DIM=2*D.
    y_prev: previous step's output. Initially zero.
    """

    def __init__(self, d, n_state, lr, mask_y_prev=False):
        self.d       = d
        self.n       = n_state
        self.lr      = lr
        self.gate_dim = 2 * d

        # mask_y_prev=True → INTENTIONAL-MASKED (y_prev zeroed = reactive fallback)
        self._mask_y_prev = mask_y_prev

        # Gate matrices: input is gate_input (2*D) instead of u_t (D)
        self.W_delta = np.zeros((n_state, self.gate_dim), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)
        self.W_B     = _det_init(n_state, self.gate_dim, scale=0.1)
        self.W_C     = _det_init(n_state, self.gate_dim, scale=0.1)
        self.W_out   = _det_init(d, n_state, scale=0.1)   # [D, N_STATE] — unchanged
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5

        self._init_W_B = self.W_B.copy()

        # State
        self.h   = np.zeros(n_state, dtype=np.float32)
        self.S_A = np.zeros(n_state, dtype=np.float32)   # RTRL trace for A_param

        # Per-step storage
        self._x         = None    # input u_t
        self._h_prev    = None
        self._A_bar     = None
        self._delta     = None
        self._pre_delta = None
        self._B_vec     = None
        self._C_vec     = None
        self._gate_input = None   # gate_input = [y_prev; u_t]
        self._ch_prev   = None    # C_vec * h from previous step (for y_prev gradient)

        # y_prev: previous output (zero-initialized each episode)
        self._y_prev    = np.zeros(d, dtype=np.float32)

    def forward(self, x):
        """Forward pass. x = u_t (input to this layer, dimension D)."""
        self._x      = x
        self._h_prev = self.h.copy()

        # Store C_vec * h from PREVIOUS step for y_prev gradient
        # (before overwriting _C_vec and h)
        if self._C_vec is not None:
            self._ch_prev = self._C_vec * self.h
        else:
            self._ch_prev = np.zeros(self.n, dtype=np.float32)

        # Build gate_input: [y_prev; u_t] — intentional conditioning on previous output
        y_prev_gate = np.zeros(self.d, dtype=np.float32) if self._mask_y_prev else self._y_prev
        gate_input = np.concatenate([y_prev_gate, x])
        self._gate_input = gate_input

        pre_delta = np.clip(self.W_delta @ gate_input + self.b_delta, -20.0, 20.0)
        self._pre_delta = pre_delta
        delta = np.log1p(np.exp(pre_delta))
        self._delta = delta

        B_vec = self.W_B @ gate_input
        C_vec = self.W_C @ gate_input
        self._B_vec = B_vec
        self._C_vec = C_vec

        A_bar = np.exp(-delta * self.A_param)
        self._A_bar = A_bar

        self.h = A_bar * self._h_prev + delta * B_vec
        self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)

        y = self.W_out @ (C_vec * self.h)
        # Update y_prev for next step
        self._y_prev = y.copy()
        return y

    def rtrl_update(self, e_y):
        """RTRL update. e_y: error signal from prediction head (dimension D)."""
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)
        e_y = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)

        # Backprop through W_out: e_y → e_h (via C_vec * h)
        d_C_h    = self.W_out.T @ e_y
        d_C_h    = np.nan_to_num(np.clip(d_C_h, -1e4, 1e4), nan=0.0)
        e_h      = self._C_vec * d_C_h
        e_C_vec  = self.h * d_C_h
        e_C_vec  = np.nan_to_num(np.clip(e_C_vec, -1e4, 1e4), nan=0.0)

        self.W_out -= self.lr * np.outer(e_y, self._C_vec * self.h)
        self.W_C   -= self.lr * np.outer(e_C_vec, self._gate_input)
        e_gate_from_C = self.W_C.T @ e_C_vec

        e_h = np.nan_to_num(np.clip(e_h, -1e4, 1e4), nan=0.0)

        # RTRL trace for A_param
        self.S_A = self._A_bar * self.S_A + (-self._delta * self._A_bar * self._h_prev)
        self.A_param -= self.lr * (e_h * self.S_A)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)

        # Backprop through delta → W_delta
        e_delta      = e_h * (self._h_prev * (-self.A_param * self._A_bar) + self._B_vec)
        e_delta      = np.nan_to_num(np.clip(e_delta, -1e4, 1e4), nan=0.0)
        sigmoid_dp   = 1.0 / (1.0 + np.exp(-self._pre_delta))
        e_pre_delta  = e_delta * sigmoid_dp
        e_pre_delta  = np.nan_to_num(np.clip(e_pre_delta, -1e4, 1e4), nan=0.0)

        self.W_delta -= self.lr * np.outer(e_pre_delta, self._gate_input)
        self.b_delta -= self.lr * e_pre_delta
        e_gate_from_delta = self.W_delta.T @ e_pre_delta

        # Backprop through B_vec → W_B
        e_B_vec  = self._delta * e_h
        e_B_vec  = np.nan_to_num(np.clip(e_B_vec, -1e4, 1e4), nan=0.0)
        self.W_B -= self.lr * np.outer(e_B_vec, self._gate_input)
        e_gate_from_B = self.W_B.T @ e_B_vec

        # Total gradient w.r.t. gate_input
        e_gate = e_gate_from_C + e_gate_from_delta + e_gate_from_B
        e_gate = np.nan_to_num(np.clip(e_gate, -1e4, 1e4), nan=0.0)

        # Extra term from y_prev path (intentional gate: one-step RTRL)
        # y_prev = W_out @ (C_prev_vec * h_prev), so e_y_prev feeds into W_out
        e_y_prev = e_gate[:self.d]   # first D elements of gate_input gradient
        if self._ch_prev is not None and np.any(e_y_prev != 0):
            self.W_out -= self.lr * np.outer(e_y_prev, self._ch_prev)

        # Return gradient w.r.t. x = u_t (last D elements of gate_input gradient)
        return e_gate[self.d:]

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.W_B - self._init_W_B, 'fro'))

    def reset_state(self):
        """Reset h and y_prev for try2 or level transition. Keep learned weights."""
        self.h[:] = 0.0
        self.S_A[:] = 0.0
        self._y_prev[:] = 0.0
        self._x = self._h_prev = self._A_bar = self._delta = self._pre_delta = None
        self._B_vec = self._C_vec = self._gate_input = self._ch_prev = None


# ---------------------------------------------------------------------------
# SSM Substrate (INTENTIONAL or DISCONNECTED)
# ---------------------------------------------------------------------------

class SSMSubstrate:
    """
    mode: 'intentional'        — full intentional gates, spatial act_emb
          'intentional-masked' — y_prev zeroed (reactive fallback, diagnostic)
          'disconnected'       — random try1 actions, act_emb zeroed, intentional gates
    """

    def __init__(self, n_actions, mode='intentional', W_fixed=None):
        self.n_actions  = n_actions
        self.mode       = mode
        self._W_fixed   = W_fixed

        self._mask_y_prev  = (mode == 'intentional-masked')
        self._mask_action  = (mode in ('intentional-masked', 'disconnected'))
        self._random_try1  = (mode == 'disconnected')

        self._W_in      = _det_init(D, D_IN, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)
        self._last_u    = None

        self._layers    = [IntentionalSSMLayer(D, N_STATE, SSM_LR, mask_y_prev=self._mask_y_prev)
                           for _ in range(N_LAYERS)]

        self._W_pred    = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred    = np.zeros(PROJ_DIM, dtype=np.float32)

        self._step          = 0
        self._prev_action   = 0
        self._prev_y        = None
        self._prev_proj_obs = None
        self._in_try2       = False
        self._visit_count   = np.zeros(n_actions, dtype=np.int32)
        self._obs_proj      = None

        self._try2_max_level = 0
        self._current_level  = 0
        self._r3_diff        = None

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
        act_emb = _encode_action_spatial(self._prev_action, masked=self._mask_action)
        x_in = np.concatenate([proj_obs, act_emb])
        u = self._W_in @ x_in + self._b_in
        self._last_u = u
        y = u
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y):
        if self._prev_proj_obs is None:
            return None
        target     = proj_obs_next
        pred_obs   = self._W_pred @ y + self._b_pred
        pred_obs   = np.nan_to_num(pred_obs, nan=0.0, posinf=0.0, neginf=0.0)
        error_obs  = pred_obs - target
        error_safe = np.nan_to_num(np.clip(error_obs, -1e4, 1e4), nan=0.0)
        y_safe     = np.nan_to_num(np.clip(y,         -1e4, 1e4), nan=0.0)

        fwd_loss = float(np.mean(error_safe ** 2))

        self._W_pred -= SSM_LR * np.outer(error_safe, y_safe)
        self._b_pred -= SSM_LR * error_safe

        e_y = np.nan_to_num(np.clip(self._W_pred.T @ error_safe, -1e4, 1e4), nan=0.0)
        for layer in reversed(self._layers):
            e_y = layer.rtrl_update(e_y)

        if self._last_u is not None:
            e_u = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)
            self._W_in -= SSM_LR * np.outer(e_u, np.concatenate([
                self._prev_proj_obs,
                _encode_action_spatial(self._prev_action, masked=self._mask_action)
            ]))
            self._b_in -= SSM_LR * e_u

        return fwd_loss

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat
        y = self._ssm_forward(proj_obs)
        self._prev_y        = y
        self._prev_proj_obs = proj_obs.copy()

        if self._in_try2 and self._W_fixed is not None:
            h_concat = np.concatenate([l.h for l in self._layers])
            h_concat = np.nan_to_num(h_concat, nan=0.0)
            logits   = self._W_fixed @ h_concat / TEMPERATURE
            logits  -= logits.max()
            probs    = np.exp(logits)
            probs   /= probs.sum()
            probs    = np.nan_to_num(probs, nan=1.0 / self.n_actions)
            probs    = np.clip(probs, 0.0, 1.0)
            probs   /= probs.sum()
            action   = int(np.random.choice(self.n_actions, p=probs))
        elif self._in_try2:
            action = int(np.random.randint(self.n_actions))
        elif self._random_try1:
            action = int(np.random.randint(self.n_actions))
        else:
            # Argmin exploration (non-disconnected conditions)
            min_count  = self._visit_count.min()
            candidates = np.where(self._visit_count == min_count)[0]
            action     = int(np.random.choice(candidates))

        self._visit_count[action] += 1
        self._step       += 1
        self._prev_action = action
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._prev_y is None or self._obs_proj is None:
            return
        obs_next_flat = _encode_obs(np.asarray(obs_next, dtype=np.float32))
        proj_obs_next = self._obs_proj @ obs_next_flat
        loss = self._rtrl_step(proj_obs_next, self._prev_y)
        if loss is not None:
            self._pred_losses.append(loss)
            if self._in_try2:
                self._pred_losses_try2.append(loss)

    def on_level_transition(self, new_level=None):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y        = None
        self._prev_proj_obs = None
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
        self._prev_proj_obs  = None
        self._step           = 0
        self._prev_action    = 0
        self._in_try2        = True
        self._visit_count[:] = 0
        self._current_level  = 0

    def compute_stage_metrics(self):
        vc   = self._visit_count.astype(float)
        i3cv = float(np.std(vc) / (np.mean(vc) + 1e-8)) if vc.sum() > 0 else 0.0
        n    = self.n_actions
        if vc.sum() > 0:
            p   = vc / vc.sum()
            kl  = float(np.sum(p * np.log(p * n + 1e-12)))  # KL(p || uniform)
        else:
            kl = 0.0
        return {
            'r3_weight_diff': self._r3_diff,
            'i3_cv':          i3cv,
            'action_kl':      round(kl, 6),
            'i5_max_level':   self._try2_max_level,
        }

    def get_stats(self):
        p1m = round(float(np.mean(self._pred_losses)),      6) if self._pred_losses      else None
        p2m = round(float(np.mean(self._pred_losses_try2)), 6) if self._pred_losses_try2 else None
        return {'pred_loss_mean': p1m, 'pred_loss_try2': p2m}


# ---------------------------------------------------------------------------
# Control C substrate (argmin, no SSM)
# ---------------------------------------------------------------------------

class ControlCSubstrate:
    """Pure argmin exploration, no SSM, no learning. Baseline."""

    def __init__(self, n_actions):
        self.n_actions    = n_actions
        self._visit_count = np.zeros(n_actions, dtype=np.int32)
        self._step        = 0
        self._try2_max_level = 0

    def process(self, obs_arr):
        min_count  = self._visit_count.min()
        candidates = np.where(self._visit_count == min_count)[0]
        action     = int(np.random.choice(candidates))
        self._visit_count[action] += 1
        self._step += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self, new_level=None):
        if new_level is not None and new_level > self._try2_max_level:
            self._try2_max_level = max(self._try2_max_level, new_level)

    def prepare_for_try2(self):
        self._visit_count[:] = 0
        self._step = 0

    def compute_stage_metrics(self):
        vc   = self._visit_count.astype(float)
        i3cv = float(np.std(vc) / (np.mean(vc) + 1e-8)) if vc.sum() > 0 else 0.0
        return {
            'r3_weight_diff': 0.0,
            'i3_cv':          i3cv,
            'action_kl':      0.0,
            'i5_max_level':   self._try2_max_level,
        }

    def get_stats(self):
        return {'pred_loss_mean': None, 'pred_loss_try2': None}


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

def _make_substrate(cond_name, n_actions, W_fixed):
    mode = cond_name.lower()
    if mode == 'control_c':
        return ControlCSubstrate(n_actions=n_actions)
    return SSMSubstrate(n_actions=n_actions, mode=mode, W_fixed=W_fixed)


def run_draw(draw_idx, draw_seed, cond_name, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        w_rng   = np.random.RandomState(draw_seed * 100 + game_idx)
        W_fixed = w_rng.randn(n_actions, H_DIM).astype(np.float32)

        substrate = _make_substrate(cond_name, n_actions, W_fixed)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)  # PRNG fix
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
            'pred_loss_mean': stats['pred_loss_mean'],
            'pred_loss_try2': stats['pred_loss_try2'],
            'r3_weight_diff': stage['r3_weight_diff'],
            'i3_cv':          stage['i3_cv'],
            'action_kl':      stage['action_kl'],
            'i5_max_level':   stage['i5_max_level'],
        }
        masked_row = mask_result_row(row, game_labels)
        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')
        draw_results.append(masked_row)

    rhae     = compute_rhae_try2(try2_progress, optimal_steps_d)
    pred_t2  = [r['pred_loss_try2'] for r in draw_results if r.get('pred_loss_try2') is not None]
    pred_t2m = round(float(np.mean(pred_t2)), 6) if pred_t2 else None
    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  pred_t2={pred_t2m}")
    return round(rhae, 7), pred_t2m, draw_results


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
    sub_t1 = SSMSubstrate(n_actions=na_t1, mode='intentional', W_fixed=W_fixed_t1)

    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    # Estimate: diag (3 × 3 × 2 × 2) + full (30 × 3 × len(CONDITIONS) × 2)
    n_eps_diag = N_DIAG_INIT * 3 * 2 * 2
    n_eps_main = (N_DRAWS - N_DIAG_INIT) * 3 * 2 * len(CONDITIONS) + N_DIAG_INIT * 3 * 2 * (len(CONDITIONS) - 1)
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * (n_eps_diag + n_eps_main)
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * (n_eps_diag + n_eps_main))
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Mandatory diagnostic: 3 draws INTENTIONAL vs INTENTIONAL-MASKED ──
    print(f"\n=== MANDATORY DIAGNOSTIC: {N_DIAG_INIT} draws INTENTIONAL vs INTENTIONAL-MASKED ===")
    diag_int_rhae = []
    diag_int_pred = []
    diag_msk_pred = []

    t_diag = time.time()
    for di in range(N_DIAG_INIT):
        ds = DRAW_SEEDS[di]
        r_int, p_int, _ = run_draw(di, ds, 'INTENTIONAL',        max_steps)
        _,     p_msk, _ = run_draw(di, ds, 'INTENTIONAL-MASKED', max_steps)
        diag_int_rhae.append(r_int)
        diag_int_pred.append(p_int)
        diag_msk_pred.append(p_msk)
    print(f"  Diagnostic done in {time.time()-t_diag:.0f}s")

    int_valid = [p for p in diag_int_pred if p is not None and not math.isnan(p)]
    msk_valid = [p for p in diag_msk_pred if p is not None and not math.isnan(p)]

    if int_valid and msk_valid and np.mean(int_valid) > 0:
        action_blind_ratio = round(float(np.mean(msk_valid)) / float(np.mean(int_valid)), 4)
    else:
        action_blind_ratio = None

    print(f"\n  INTENTIONAL pred_t2: {diag_int_pred}")
    print(f"  MASKED      pred_t2: {diag_msk_pred}")
    print(f"  Action-blind ratio (MASKED/INT): {action_blind_ratio}")

    threshold = 1.0 + ACTION_BLIND_THRESHOLD
    if action_blind_ratio is None or action_blind_ratio < threshold:
        print(f"\n*** DIAGNOSTIC FAIL: ratio={action_blind_ratio} < {threshold:.2f} ***")
        print("Intentional gates did not break the action-blind attractor.")
        summary = {
            'step':               STEP,
            'verdict':            'DIAGNOSTIC_FAIL',
            'action_blind_ratio': action_blind_ratio,
            'diag_int_pred_t2':   diag_int_pred,
            'diag_msk_pred_t2':   diag_msk_pred,
        }
        with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        return

    print(f"\n  *** DIAGNOSTIC PASS: ratio={action_blind_ratio} > {threshold:.2f} ***")
    print(f"  Intentional gates break action-blind attractor. Proceeding to full experiment.")

    # ── Full experiment: 3 conditions, 30 draws ───────────────────────────
    print(f"\n=== STEP {STEP}: 3 conditions, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond = {c: [] for c in CONDITIONS}

    # Reuse INTENTIONAL diagnostic draws
    for di in range(N_DIAG_INIT):
        rhae_by_cond['INTENTIONAL'].append(diag_int_rhae[di])
        for cond in ('DISCONNECTED', 'CONTROL_C'):
            r, _, _ = run_draw(di, DRAW_SEEDS[di], cond, max_steps)
            rhae_by_cond[cond].append(r)

    for di in range(N_DIAG_INIT, N_DRAWS):
        for cond in CONDITIONS:
            r, _, _ = run_draw(di, DRAW_SEEDS[di], cond, max_steps)
            rhae_by_cond[cond].append(r)

    # Statistics
    int_rhae  = np.array(rhae_by_cond['INTENTIONAL'])
    disc_rhae = np.array(rhae_by_cond['DISCONNECTED'])
    ctrl_rhae = np.array(rhae_by_cond['CONTROL_C'])

    # Primary comparison: INTENTIONAL vs DISCONNECTED
    wins_id   = int(np.sum(int_rhae > disc_rhae))
    losses_id = int(np.sum(int_rhae < disc_rhae))
    ties_id   = int(np.sum(int_rhae == disc_rhae))
    p_id      = binomial_p_one_sided(wins_id, wins_id + losses_id)

    # Secondary comparison: INTENTIONAL vs CONTROL_C
    wins_ic   = int(np.sum(int_rhae > ctrl_rhae))
    losses_ic = int(np.sum(int_rhae < ctrl_rhae))
    p_ic      = binomial_p_one_sided(wins_ic, wins_ic + losses_ic)

    int_mean  = float(np.mean(int_rhae))
    disc_mean = float(np.mean(disc_rhae))
    ctrl_mean = float(np.mean(ctrl_rhae))
    int_nz    = int(np.sum(int_rhae > 0))
    disc_nz   = int(np.sum(disc_rhae > 0))
    ctrl_nz   = int(np.sum(ctrl_rhae > 0))

    if p_id <= 0.10 and int_mean > disc_mean:
        verdict = 'SIGNAL'
    elif int_mean > disc_mean and int_mean > ctrl_mean:
        verdict = 'INT_BETTER_NOT_SIG'
    elif int_mean > disc_mean:
        verdict = 'INT_BEATS_DISCONN_NOT_SIG'
    else:
        verdict = 'KILL'

    # Note if INTENTIONAL ≤ CONTROL_C (SSM helps vs random but not vs argmin)
    note = ''
    if int_mean > disc_mean and int_mean <= ctrl_mean:
        note = 'SSM_BEATS_RANDOM_NOT_ARGMIN'

    print(f"\n=== STEP {STEP} RESULTS ===")
    print(f"  INTENTIONAL  chain_mean={int_mean:.3e}  nz={int_nz}/{N_DRAWS}")
    print(f"  DISCONNECTED chain_mean={disc_mean:.3e}  nz={disc_nz}/{N_DRAWS}")
    print(f"  CONTROL_C    chain_mean={ctrl_mean:.3e}  nz={ctrl_nz}/{N_DRAWS}")
    print(f"  INT vs DISC: {wins_id}W-{losses_id}L-{ties_id}T  p={p_id:.4f}")
    print(f"  INT vs CTRL: {wins_ic}W-{losses_ic}L p={p_ic:.4f}")
    print(f"  Action-blind ratio (diag) = {action_blind_ratio}")
    print(f"  Verdict: {verdict}  {note}")

    summary = {
        'step':     STEP,
        'n_draws':  N_DRAWS,
        'draw_seeds': DRAW_SEEDS,
        'diag_action_blind_ratio': action_blind_ratio,
        'conditions': {
            'INTENTIONAL':  {'chain_mean': round(int_mean,  8), 'nz': int_nz},
            'DISCONNECTED': {'chain_mean': round(disc_mean, 8), 'nz': disc_nz},
            'CONTROL_C':    {'chain_mean': round(ctrl_mean, 8), 'nz': ctrl_nz},
        },
        'int_vs_disc': {
            'paired_wins': wins_id, 'paired_losses': losses_id, 'paired_ties': ties_id,
            'p_value': round(p_id, 6),
        },
        'int_vs_ctrl': {
            'paired_wins': wins_ic, 'paired_losses': losses_ic,
            'p_value': round(p_ic, 6),
        },
        'verdict': verdict,
        'note':    note,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/summary.json")


if __name__ == '__main__':
    main()
