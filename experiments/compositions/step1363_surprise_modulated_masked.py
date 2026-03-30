"""
Step 1363 — Neuromodulatory bridge: prediction surprise modulates W_act learning rate.
Leo mail 3888, 2026-03-30. Three-factor plasticity.

Architecture:
- Same 2-layer diagonal SSM + RTRL as step 1360 (obs prediction only).
- W_act NOT in RTRL graph (RTRL trace too weak for W_act credit assignment).
- Instead: prediction SURPRISE acts as a global neuromodulator.

Surprise-modulated REINFORCE:
    surprise_t = |pred_loss_t - pred_loss_ema_t|
    pred_loss_ema_t = 0.99 * ema_{t-1} + 0.01 * pred_loss_t
    W_act += ACT_LR * surprise_t * outer(log_grad_t, y_t)
    where log_grad[i] = δ(i==a_t) - π_t[i]  (REINFORCE direction)

Three-factor plasticity:
  - Presynaptic: y_t (SSM output)
  - Postsynaptic: a_t (chosen action)
  - Neuromodulator: surprise_t (unexpected prediction = high dopamine)

Key difference from step 1307 (REINFORCE with pred error as reward):
  1307 used signed prediction error as reward → direction depends on sign.
  This uses ABSOLUTE surprise as magnitude → always reinforces taken direction.

Protocol:
- 10 draws, seeds 13600-13609
- 2K steps per episode (or fewer if Tier 1 timing exceeds budget)
- Warmup=500 steps before using action head or W_act updates
- Kill: chain_mean RHAE > 3.26e-5 (1360 SSM baseline) → SIGNAL
- Baseline comparison: step 1360 (same seeds, same SSM, W_act fixed)
"""

import os
import sys
import json
import time
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

STEP        = 1363
N_DRAWS     = 10
DRAW_SEEDS  = [13600 + i for i in range(N_DRAWS)]
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')

MLP_TP_BASELINE  = 4.59e-5  # Step 1349
SSM_BASE_1360    = 3.26e-5  # Step 1360 — primary comparison for this step

# SSM config (same as 1360)
PROJ_DIM      = 64    # fixed random projection: obs -> this
ACT_EMBED_DIM = 16    # action embedding dim
D             = 128   # SSM model dimension
N_STATE       = 32    # SSM state dimension per layer
N_LAYERS      = 2     # number of SSM layers
WARMUP        = 500   # steps before using action head or W_act updates
SSM_LR        = 1e-3  # RTRL learning rate (obs prediction)

# Neuromodulation config
ACT_LR              = 1e-3   # W_act update learning rate
SURPRISE_EMA_ALPHA  = 0.01   # EMA decay: pred_loss_ema = 0.01*new + 0.99*old

MAX_N_ACTIONS   = 4103
ENT_CHECKPOINTS = [100, 500, 1000, 1999]
BUDGET_SECONDS  = 290   # hard cap (5 min - margin)
TIER1_STEPS     = 200   # steps for timing check

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
    """Deterministic weight init. Uses QR for small matrices, sin-pattern for large."""
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
# SSM Layer
# ---------------------------------------------------------------------------

class SSMLayer:
    """Single Mamba-style diagonal SSM layer with RTRL updates."""

    def __init__(self, d, n_state, lr):
        self.d = d
        self.n = n_state
        self.lr = lr

        self.B       = _det_init(n_state, d, scale=0.1)
        self.C       = _det_init(d, n_state, scale=0.1)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)
        self.b_delta = np.zeros(n_state, dtype=np.float32)

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
        return self.C @ self.h

    def rtrl_update(self, e_y):
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)
        e_h = self.C.T @ e_y
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

    def reset_state(self):
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None


# ---------------------------------------------------------------------------
# SSM Substrate
# ---------------------------------------------------------------------------

class MambaSSMSubstrate:
    """SSM + RTRL (obs prediction) + surprise-modulated REINFORCE (W_act)."""

    def __init__(self, n_actions, max_steps=TRY1_STEPS):
        self.n_actions = n_actions
        self.max_steps = max_steps
        self._in_dim = PROJ_DIM + ACT_EMBED_DIM

        self._obs_proj = None

        # Action embedding: fixed. Discrete lookup (no differentiable path needed).
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)

        self._W_in = _det_init(D, self._in_dim, scale=0.1)
        self._b_in = np.zeros(D, dtype=np.float32)

        self._layers = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]

        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        # Action head: trained via surprise-modulated REINFORCE (NOT RTRL).
        self._W_act = _det_init(n_actions, D, scale=0.1)
        self._b_act = np.zeros(n_actions, dtype=np.float32)

        # Running state
        self._step        = 0
        self._prev_action = 0
        self._prev_y      = None

        # Neuromodulation state
        self._pred_loss_ema = None   # exponential moving average of pred_loss

        # Diagnostics
        self._try1_ent        = {}
        self._try2_ent        = {}
        self._try1_state_norm = {}
        self._try2_state_norm = {}
        self._in_try2         = False
        self._pred_losses     = []
        self._surprise_vals   = []

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim, dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    def _ssm_forward(self, proj_obs):
        act_emb = self._act_embed[self._prev_action]
        x_in = np.concatenate([proj_obs, act_emb])
        x = self._W_in @ x_in + self._b_in
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y):
        """Obs prediction RTRL only. Returns pred_loss."""
        pred = self._W_pred @ y + self._b_pred
        error = pred - proj_obs_next
        e = self._W_pred.T @ error
        self._W_pred -= SSM_LR * np.outer(error, y)
        self._b_pred -= SSM_LR * error
        for layer in reversed(self._layers):
            e = layer.rtrl_update(e)
        return float(np.mean(error ** 2))

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat
        y = self._ssm_forward(proj_obs)
        self._prev_y = y

        if self._step < WARMUP:
            action = int(np.random.randint(self.n_actions))
        else:
            logits = self._W_act @ y + self._b_act
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            action = int(np.random.choice(self.n_actions, p=probs))

        if self._step in ENT_CHECKPOINTS:
            logits = self._W_act @ y + self._b_act
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            ent = float(-np.sum(probs * np.log(probs + 1e-10)))
            snorm = [float(np.linalg.norm(l.h)) for l in self._layers]
            if self._in_try2:
                self._try2_ent[self._step] = round(ent, 4)
                self._try2_state_norm[self._step] = [round(s, 4) for s in snorm]
            else:
                self._try1_ent[self._step] = round(ent, 4)
                self._try1_state_norm[self._step] = [round(s, 4) for s in snorm]

        self._step += 1
        self._prev_action = action
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._prev_y is None or self._obs_proj is None:
            return
        obs_next_flat = _encode_obs(np.asarray(obs_next, dtype=np.float32))
        proj_obs_next = self._obs_proj @ obs_next_flat

        pred_loss = self._rtrl_step(proj_obs_next, self._prev_y)
        self._pred_losses.append(pred_loss)

        # Compute prediction surprise
        if self._pred_loss_ema is None:
            self._pred_loss_ema = pred_loss
        surprise = abs(pred_loss - self._pred_loss_ema)
        self._pred_loss_ema = (SURPRISE_EMA_ALPHA * pred_loss
                               + (1.0 - SURPRISE_EMA_ALPHA) * self._pred_loss_ema)
        self._surprise_vals.append(surprise)

        # Surprise-modulated REINFORCE update for W_act (only after warmup)
        if self._step > WARMUP and surprise > 0.0:
            y = self._prev_y
            logits = self._W_act @ y + self._b_act
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            # REINFORCE direction: ∂log π(a_t)/∂logits_i = δ(i==a_t) - π_i
            log_grad = -probs.copy()
            log_grad[int(action)] += 1.0
            self._W_act += ACT_LR * surprise * np.outer(log_grad, y)
            self._b_act += ACT_LR * surprise * log_grad

    def on_level_transition(self):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None

    def reset_for_try2(self):
        """Reset recurrent state. Keep learned weights + EMA (R4 compliance)."""
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None
        self._step = 0
        self._prev_action = 0
        self._in_try2 = True
        # Keep: _W_act, _W_pred, _pred_loss_ema (calibrated baseline carries to try2)

    def get_stats(self):
        pred_loss_mean  = round(float(np.mean(self._pred_losses)), 6)  if self._pred_losses  else None
        pred_loss_final = round(float(np.mean(self._pred_losses[-50:])), 6) if len(self._pred_losses) >= 50 else pred_loss_mean
        surprise_mean   = round(float(np.mean(self._surprise_vals)), 6) if self._surprise_vals else None
        return {
            'try1_act_ent':    self._try1_ent,
            'try2_act_ent':    self._try2_ent,
            'try1_state_norm': self._try1_state_norm,
            'try2_state_norm': self._try2_state_norm,
            'pred_loss_mean':  pred_loss_mean,
            'pred_loss_final': pred_loss_final,
            'surprise_mean':   surprise_mean,
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
            substrate.on_level_transition()
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
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
            substrate.on_level_transition()
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
        else:
            obs = obs_next

    elapsed = time.time() - t_start
    return steps_to_first_progress, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    print(f"  Draw {draw_idx} (seed={draw_seed}): {masked_game_list(game_labels)}")
    draw_results = []
    try2_progress = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = MambaSSMSubstrate(n_actions=n_actions, max_steps=max_steps)

        t0 = time.time()
        p1, t1 = run_episode(env, substrate, n_actions, seed=0, max_steps=max_steps)
        substrate.reset_for_try2()
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stats = substrate.get_stats()
        speedup = compute_progress_speedup(p1, p2)
        opt = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff = min(1.0, opt / p2)
            eff_sq = round(eff ** 2, 6)

        try2_progress[label] = p2
        optimal_steps_d[label] = opt

        row = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'try1_act_ent_100':  stats['try1_act_ent'].get(100),
            'try1_act_ent_500':  stats['try1_act_ent'].get(500),
            'try1_act_ent_1000': stats['try1_act_ent'].get(1000),
            'try1_act_ent_1999': stats['try1_act_ent'].get(1999),
            'try2_act_ent_100':  stats['try2_act_ent'].get(100),
            'try2_act_ent_500':  stats['try2_act_ent'].get(500),
            'try2_act_ent_1000': stats['try2_act_ent'].get(1000),
            'try2_act_ent_1999': stats['try2_act_ent'].get(1999),
            'try1_state_norm_100':  stats['try1_state_norm'].get(100),
            'try1_state_norm_1999': stats['try1_state_norm'].get(1999),
            'try2_state_norm_100':  stats['try2_state_norm'].get(100),
            'try2_state_norm_1999': stats['try2_state_norm'].get(1999),
            'pred_loss_mean':  stats['pred_loss_mean'],
            'pred_loss_final': stats['pred_loss_final'],
            'surprise_mean':   stats['surprise_mean'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)
        elapsed_total = round(time.time() - t0, 1)
        print(f"    {label}: p1={p1} p2={p2} eff_sq={eff_sq:.6f} "
              f"pred_loss={stats['pred_loss_final']} surp={stats['surprise_mean']} ({elapsed_total}s)")

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    print(f"  Draw {draw_idx} RHAE={rhae:.6e}")
    return round(rhae, 7), draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, labels_t1 = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    sub_t1 = MambaSSMSubstrate(n_actions=na_t1)
    # Pre-warmup: exclude lazy-init (obs projection init) from timing
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    n_episodes = N_DRAWS * 3 * 2  # draws x games x tries
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * n_episodes
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated full run at {TRY1_STEPS} steps: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > BUDGET_SECONDS:
        max_steps = max(200, int(BUDGET_SECONDS / (ms_per_step / 1000 * n_episodes)))
        print(f"  Budget exceeded — capping at {max_steps} steps per episode")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — proceeding with {max_steps} steps")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"\n=== STEP {STEP}: SSM + surprise-modulated REINFORCE ===")
    print(f"N_DRAWS={N_DRAWS}, max_steps={max_steps}, WARMUP={WARMUP}")
    print(f"SSM: D={D}, N_STATE={N_STATE}, N_LAYERS={N_LAYERS}, PROJ_DIM={PROJ_DIM}")
    print(f"ACT_LR={ACT_LR}, SURPRISE_EMA_ALPHA={SURPRISE_EMA_ALPHA}")
    print(f"Baseline comparison: 1360 (SSM fixed W_act) = {SSM_BASE_1360:.2e}")
    print(f"MLP+TP baseline (1349): {MLP_TP_BASELINE:.2e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rhae_list   = []
    all_results = []

    for di, ds in enumerate(DRAW_SEEDS):
        rhae, draw_res = run_draw(di, ds, max_steps)
        rhae_list.append(round(rhae, 7))
        all_results.extend(draw_res)

    chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
    nz = sum(1 for r in rhae_list if r > 0)

    summary = {
        'step':              STEP,
        'n_draws':           N_DRAWS,
        'draw_seeds':        DRAW_SEEDS,
        'rhae_per_draw':     rhae_list,
        'chain_mean_rhae':   chain_mean,
        'nonzero_draws':     nz,
        'ssm_base_1360':     SSM_BASE_1360,
        'mlp_tp_baseline':   MLP_TP_BASELINE,
        'max_steps_used':    max_steps,
        'ms_per_step':       round(ms_per_step, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"SSM+surprise chain_mean RHAE = {chain_mean:.3e}  ({nz}/{N_DRAWS} nz)")
    print(f"SSM baseline (1360):           {SSM_BASE_1360:.3e}")
    print(f"MLP+TP baseline (1349):        {MLP_TP_BASELINE:.3e}")
    verdict = "SIGNAL" if chain_mean > SSM_BASE_1360 else "KILL"
    print(f"Verdict: {verdict}")
    print(f"(Kill criterion: chain_mean > {SSM_BASE_1360:.2e})")

    return chain_mean


if __name__ == '__main__':
    main()
