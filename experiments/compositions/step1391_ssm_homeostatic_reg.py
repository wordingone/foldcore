"""
Step 1391 — SSM + embedding-space prediction + homeostatic regularization.
Leo mail 3981, 2026-03-30. All 11 gates passed.

Root cause of SSM action-blindness (1379-1384): nothing prevented h from collapsing
to the obs-autocorrelation fixed point. The prediction objective is achievable without
actions → gradient drives h toward observation statistics → action info extinguished.

Fix: homeostatic regularizer (Turrigiano 2008, BCM 1982, SIGReg/LeCun 2026).
Every 100 steps, normalize h distribution to zero mean/unit variance per dimension.
If any h dimension collapses (action-related dims → 0), regularizer rescales it back.
The gradient can't extinguish action info because regularization restores it.

Architecture:
1. SelectiveSSM (from 1383): 2 layers, D=128, N=32.
2. Predict in EMBEDDING SPACE (JEPA-style): target = proj_obs_next (64-dim), not raw pixels.
3. Homeostatic regularizer (REG only): every 100 steps, h ← (h - mean(h)) / (std(h) + ε).
4. Spatial action encoding (from 1385): [type(7), x_norm, y_norm] = 9 dims.

Mandatory diagnostic (3 draws):
  REG vs REG-MASKED (action zeroed). ratio = pred_loss(MASKED) / pred_loss(REG).
  ratio > 1.05 → homeostatic reg breaks the action-blind attractor. Proceed.
  ratio < 1.05 → regularization doesn't help → DIAGNOSTIC_FAIL. Abort.

Conditions (30 draws, paired):
  REG:   SSM + embedding pred + homeostatic regularizer + spatial encoding.
  NOREG: SSM + embedding pred + NO regularizer + spatial encoding (1383/1385 baseline).

Seeds: 14380-14409.
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

STEP        = 1391
N_DRAWS     = 30
DRAW_SEEDS  = [14380 + i for i in range(N_DRAWS)]
N_DIAG_INIT = 3
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM dimensions
PROJ_DIM        = 64
SPATIAL_ACT_DIM = 9     # [type_onehot(7), x_norm, y_norm]
D_IN            = PROJ_DIM + SPATIAL_ACT_DIM   # 73
D               = 128
N_STATE         = 32
N_LAYERS        = 2
SSM_LR          = 1e-3
H_DIM           = N_LAYERS * N_STATE   # 64 — h_concat for frozen projection

TEMPERATURE          = 3.0
ACTION_BLIND_THRESHOLD = 0.05   # ratio > 1.05 → action-conditional
HOMEOSTATIC_INTERVAL   = 100    # normalize h every N steps

CONDITIONS    = ['REG', 'NOREG']
DIAG_COND     = 'REG-MASKED'

MAX_N_ACTIONS = 4103
I3_STEP       = 200
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
# Obs encoding + spatial action encoding
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


# ---------------------------------------------------------------------------
# Selective SSM Layer (identical to 1383-1385, plus homeostatic buffer)
# ---------------------------------------------------------------------------

class SelectiveSSMLayer:
    def __init__(self, d, n_state, lr):
        self.d = d
        self.n = n_state
        self.lr = lr

        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)
        self.W_B     = _det_init(n_state, d, scale=0.1)
        self.W_C     = _det_init(n_state, d, scale=0.1)
        self.W_out   = _det_init(d, n_state, scale=0.1)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5

        self._init_W_B = self.W_B.copy()

        self.h   = np.zeros(n_state, dtype=np.float32)
        self.S_A = np.zeros(n_state, dtype=np.float32)

        self._x = None
        self._h_prev = self._A_bar = self._delta = self._pre_delta = None
        self._B_vec = self._C_vec = None

        # Homeostatic buffer
        self._h_log = []

    def forward(self, x):
        self._x = x
        self._h_prev = self.h.copy()

        pre_delta = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        self._pre_delta = pre_delta
        delta = np.log1p(np.exp(pre_delta))
        self._delta = delta

        B_vec = self.W_B @ x
        C_vec = self.W_C @ x
        self._B_vec = B_vec
        self._C_vec = C_vec

        A_bar = np.exp(-delta * self.A_param)
        self._A_bar = A_bar

        self.h = A_bar * self._h_prev + delta * B_vec
        self.h = np.nan_to_num(self.h, nan=0.0, posinf=0.0, neginf=0.0)
        return self.W_out @ (C_vec * self.h)

    def rtrl_update(self, e_y):
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)
        e_y = np.nan_to_num(np.clip(e_y, -1e4, 1e4), nan=0.0)

        d_C_h = self.W_out.T @ e_y
        d_C_h = np.nan_to_num(np.clip(d_C_h, -1e4, 1e4), nan=0.0)

        e_h     = self._C_vec * d_C_h
        e_C_vec = self.h * d_C_h
        e_C_vec = np.nan_to_num(np.clip(e_C_vec, -1e4, 1e4), nan=0.0)

        self.W_out -= self.lr * np.outer(e_y, self._C_vec * self.h)
        self.W_C   -= self.lr * np.outer(e_C_vec, self._x)
        e_x_from_C  = self.W_C.T @ e_C_vec

        e_h = np.nan_to_num(np.clip(e_h, -1e4, 1e4), nan=0.0)
        self.S_A = self._A_bar * self.S_A + (-self._delta * self._A_bar * self._h_prev)
        self.A_param -= self.lr * (e_h * self.S_A)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)

        e_delta = e_h * (self._h_prev * (-self.A_param * self._A_bar) + self._B_vec)
        e_delta = np.nan_to_num(np.clip(e_delta, -1e4, 1e4), nan=0.0)
        sigmoid_dp  = 1.0 / (1.0 + np.exp(-self._pre_delta))
        e_pre_delta = e_delta * sigmoid_dp
        e_pre_delta = np.nan_to_num(np.clip(e_pre_delta, -1e4, 1e4), nan=0.0)

        self.W_delta -= self.lr * np.outer(e_pre_delta, self._x)
        self.b_delta -= self.lr * e_pre_delta
        e_x_from_delta = self.W_delta.T @ e_pre_delta

        e_B_vec = self._delta * e_h
        e_B_vec = np.nan_to_num(np.clip(e_B_vec, -1e4, 1e4), nan=0.0)
        self.W_B -= self.lr * np.outer(e_B_vec, self._x)
        e_x_from_B = self.W_B.T @ e_B_vec

        return e_x_from_C + e_x_from_delta + e_x_from_B

    def add_h_to_log(self):
        """Buffer h for homeostatic normalization."""
        self._h_log.append(self.h.copy())

    def apply_homeostatic(self):
        """Normalize h to zero mean / unit variance using buffered history."""
        if len(self._h_log) < 2:
            self._h_log = []
            return
        h_arr  = np.stack(self._h_log)      # (N, n_state)
        mean_h = h_arr.mean(axis=0)
        std_h  = h_arr.std(axis=0)
        self.h = (self.h - mean_h) / (std_h + 1e-8)
        self.h = np.nan_to_num(self.h, nan=0.0)
        self._h_log = []

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.W_B - self._init_W_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S_A[:] = 0.0
        self._x = self._h_prev = self._A_bar = self._delta = self._pre_delta = None
        self._B_vec = self._C_vec = None
        self._h_log = []


# ---------------------------------------------------------------------------
# SSM Substrate
# ---------------------------------------------------------------------------

class SSMSubstrate:
    """
    mode: 'reg'        — spatial encoding + embedding pred + homeostatic regularizer
          'reg-masked' — same but action zeroed (diagnostic)
          'noreg'      — spatial encoding + embedding pred + NO regularizer

    W_fixed: (n_actions, H_DIM) frozen projection for try2 action selection.
    """

    def __init__(self, n_actions, mode='reg', W_fixed=None):
        self.n_actions        = n_actions
        self.mode             = mode
        self._W_fixed         = W_fixed
        self._use_homeostatic = (mode in ('reg', 'reg-masked'))
        self._mask_action     = (mode == 'reg-masked')

        self._W_in      = _det_init(D, D_IN, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)
        self._last_x_in = None

        self._layers = [SelectiveSSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]

        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        self._step           = 0
        self._prev_action    = 0
        self._prev_y         = None
        self._prev_proj_obs  = None
        self._in_try2        = False
        self._visit_count    = np.zeros(n_actions, dtype=np.int32)
        self._obs_proj       = None

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
        self._last_x_in = x_in
        u = self._W_in @ x_in + self._b_in
        y = u
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y):
        """Embedding-space prediction: target = proj_obs_next (absolute, not diff)."""
        if self._prev_proj_obs is None:
            return None

        # Target: absolute next embedding (JEPA-style)
        target = proj_obs_next

        pred_obs   = self._W_pred @ y + self._b_pred
        pred_obs   = np.nan_to_num(pred_obs, nan=0.0, posinf=0.0, neginf=0.0)
        error_obs  = pred_obs - target
        error_safe = np.nan_to_num(np.clip(error_obs, -1e4, 1e4), nan=0.0)
        y_safe     = np.nan_to_num(np.clip(y, -1e4, 1e4), nan=0.0)

        fwd_loss = float(np.mean(error_safe ** 2))

        self._W_pred -= SSM_LR * np.outer(error_safe, y_safe)
        self._b_pred -= SSM_LR * error_safe

        e_y = np.nan_to_num(np.clip(self._W_pred.T @ error_safe, -1e4, 1e4), nan=0.0)
        for layer in reversed(self._layers):
            e_y = layer.rtrl_update(e_y)

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
        self._prev_y        = y
        self._prev_proj_obs = proj_obs.copy()

        if self._in_try2 and self._W_fixed is not None:
            h_concat = np.concatenate([l.h for l in self._layers])
            h_concat = np.nan_to_num(h_concat, nan=0.0)
            logits = self._W_fixed @ h_concat / TEMPERATURE
            logits -= logits.max()
            probs   = np.exp(logits)
            probs  /= probs.sum()
            probs   = np.nan_to_num(probs, nan=1.0 / self.n_actions)
            probs   = np.clip(probs, 0.0, 1.0)
            probs  /= probs.sum()
            action  = int(np.random.choice(self.n_actions, p=probs))
        elif self._in_try2:
            action = int(np.random.randint(self.n_actions))
        else:
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

        # Homeostatic regularization: log h, apply every HOMEOSTATIC_INTERVAL steps
        if self._use_homeostatic:
            for layer in self._layers:
                layer.add_h_to_log()
            if self._step > 0 and (self._step % HOMEOSTATIC_INTERVAL) == 0:
                for layer in self._layers:
                    layer.apply_homeostatic()

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
        return {
            'r3_weight_diff': self._r3_diff,
            'i5_max_level':   self._try2_max_level,
        }

    def get_stats(self):
        pred_mean = round(float(np.mean(self._pred_losses)),      6) if self._pred_losses      else None
        pred_try2 = round(float(np.mean(self._pred_losses_try2)), 6) if self._pred_losses_try2 else None
        return {'pred_loss_mean': pred_mean, 'pred_loss_try2': pred_try2}


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
    mode = cond_name.lower()   # 'reg', 'reg-masked', 'noreg'

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

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
            'i5_max_level':     stage['i5_max_level'],
            'r3_weight_diff':   stage['r3_weight_diff'],
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

    sub_t1 = SSMSubstrate(n_actions=na_t1, mode='reg', W_fixed=W_fixed_t1)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    n_eps_main  = N_DRAWS * 3 * 2 * len(CONDITIONS)
    n_eps_diag  = N_DIAG_INIT * 3 * 2 * 2  # REG + MASKED
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

    # ── Mandatory diagnostic: 3 draws REG vs REG-MASKED ──────────────────
    print(f"\n=== MANDATORY DIAGNOSTIC: {N_DIAG_INIT} draws REG vs REG-MASKED ===")
    diag_reg_rhae  = []; diag_reg_pred  = []
    diag_msk_rhae  = []; diag_msk_pred  = []

    t_diag = time.time()
    for di in range(N_DIAG_INIT):
        ds = DRAW_SEEDS[di]
        r_reg, p_reg, _ = run_draw(di, ds, 'REG',        max_steps)
        r_msk, p_msk, _ = run_draw(di, ds, 'REG-MASKED', max_steps)
        diag_reg_rhae.append(r_reg);  diag_reg_pred.append(p_reg)
        diag_msk_rhae.append(r_msk);  diag_msk_pred.append(p_msk)
    print(f"  Diagnostic done in {time.time()-t_diag:.0f}s")

    reg_valid = [p for p in diag_reg_pred if p is not None and not math.isnan(p)]
    msk_valid = [p for p in diag_msk_pred if p is not None and not math.isnan(p)]

    if reg_valid and msk_valid and np.mean(reg_valid) > 0:
        action_blind_ratio = round(float(np.mean(msk_valid)) / float(np.mean(reg_valid)), 4)
    else:
        action_blind_ratio = None

    print(f"\n  REG    pred_t2: {diag_reg_pred}")
    print(f"  MASKED pred_t2: {diag_msk_pred}")
    print(f"  Action-blind ratio (MASKED/REG): {action_blind_ratio}")

    threshold = 1.0 + ACTION_BLIND_THRESHOLD
    if action_blind_ratio is None or action_blind_ratio < threshold:
        print(f"\n*** DIAGNOSTIC FAIL: ratio={action_blind_ratio} < {threshold:.2f} ***")
        print("Homeostatic regularization did NOT break the action-blind attractor.")
        summary = {
            'step':               STEP,
            'verdict':            'DIAGNOSTIC_FAIL',
            'action_blind_ratio': action_blind_ratio,
            'diag_draws':         N_DIAG_INIT,
            'diag_reg_pred_t2':   diag_reg_pred,
            'diag_msk_pred_t2':   diag_msk_pred,
            'note': 'Full experiment aborted — homeostatic regularization also action-blind.',
        }
        fn = os.path.join(RESULTS_DIR, 'summary.json')
        with open(fn, 'w') as f:
            json.dump(summary, f, indent=2)
        return

    print(f"\n  *** DIAGNOSTIC PASS: ratio={action_blind_ratio:.4f} > {threshold:.2f} ***")
    print(f"  Homeostatic reg breaks the action-blind attractor!")
    print(f"  Proceeding to full {N_DRAWS}-draw experiment.")

    # ── Full experiment: REG vs NOREG, 30 draws ───────────────────────────
    print(f"\n=== STEP {STEP}: REG vs NOREG, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond = {c: [] for c in CONDITIONS}
    pred_by_cond = {c: [] for c in CONDITIONS}

    # Reuse REG diagnostic draws
    for di in range(N_DIAG_INIT):
        rhae_by_cond['REG'].append(round(diag_reg_rhae[di], 7))
        pred_by_cond['REG'].append(diag_reg_pred[di])
        print(f"  [REG] Draw {di} (diagnostic) RHAE={diag_reg_rhae[di]:.6e}")

    print(f"\n--- Condition: REG (draws {N_DIAG_INIT}-{N_DRAWS-1}) ---")
    t_cond = time.time()
    for di in range(N_DIAG_INIT, N_DRAWS):
        rhae, pred, _ = run_draw(di, DRAW_SEEDS[di], 'REG', max_steps)
        rhae_by_cond['REG'].append(round(rhae, 7))
        pred_by_cond['REG'].append(pred)
    print(f"  REG done in {time.time()-t_cond:.0f}s")

    print(f"\n--- Condition: NOREG ---")
    t_cond = time.time()
    for di, ds in enumerate(DRAW_SEEDS):
        rhae, pred, _ = run_draw(di, ds, 'NOREG', max_steps)
        rhae_by_cond['NOREG'].append(round(rhae, 7))
        pred_by_cond['NOREG'].append(pred)
    print(f"  NOREG done in {time.time()-t_cond:.0f}s")

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
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        pred_vals  = [p for p in pred_by_cond[cond_name] if p is not None]
        pred_mean  = round(float(np.mean(pred_vals)), 6) if pred_vals else None
        print(f"\n{cond_name}:  chain_mean={chain_mean:.6e}  nz={nz}/{N_DRAWS}  pred_t2={pred_mean}")
        summary['conditions'][cond_name] = {
            'chain_mean': chain_mean,
            'nz':         nz,
            'rhae_list':  rhae_list,
            'pred_t2_mean': pred_mean,
        }

    # Paired sign test: REG vs NOREG
    reg_rhae  = rhae_by_cond['REG']
    nreg_rhae = rhae_by_cond['NOREG']
    wins   = sum(1 for a, b in zip(reg_rhae, nreg_rhae) if a > b)
    losses = sum(1 for a, b in zip(reg_rhae, nreg_rhae) if a < b)
    ties   = N_DRAWS - wins - losses
    p_val  = binomial_p_one_sided(wins, wins + losses)

    print(f"\nPaired sign test (REG vs NOREG):")
    print(f"  Wins={wins}  Losses={losses}  Ties={ties}  p={p_val:.6f}")

    if wins > losses and p_val <= 0.10:
        verdict = 'SIGNAL'
    elif wins < losses:
        verdict = 'REG_WORSE'
    else:
        verdict = 'KILL'

    summary['paired_wins']   = wins
    summary['paired_losses'] = losses
    summary['paired_ties']   = ties
    summary['p_value']       = p_val
    summary['verdict']       = verdict

    fn = os.path.join(RESULTS_DIR, 'summary.json')
    with open(fn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {fn}")

    print(f"\nVERDICT: {verdict}")
    for cond_name in CONDITIONS:
        c = summary['conditions'][cond_name]
        print(f"  {cond_name}  chain_mean: {c['chain_mean']:.6e}")


if __name__ == '__main__':
    main()
