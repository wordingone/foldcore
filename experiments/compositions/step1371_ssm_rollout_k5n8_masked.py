"""
Step 1371 — Stronger rollout K=5 N=8, paired 30 draws.
Leo mail 3911, 2026-03-30.

Same seeds as 1370 (13700-13729) for direct comparison: K=3 N=8 (1370) vs K=5 N=8 (1371).
K=5 N=8: 40 rollout steps per action (~1.67x 1370's 24). Est ~0.8ms/step.

Kill criteria:
  - K=5 wins > K=3 wins (4) on same seeds → more resolution helps. SIGNAL.
  - K=5 losses < K=3 losses (2) → fewer errors. SIGNAL.
  - p < 0.10 → approaching significance.
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

STEP        = 1371
N_DRAWS     = 30
DRAW_SEEDS  = [13700 + i for i in range(N_DRAWS)]  # Same as 1370 for direct comparison
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5  # Step 1349 mean RHAE

# SSM config
PROJ_DIM      = 64
ACT_EMBED_DIM = 16
D             = 128
N_STATE       = 32
N_LAYERS      = 2
WARMUP        = 200
SSM_LR        = 1e-3

# Rollout config — increased from K=3 N=8
K_ROLLOUT    = 5   # was 3
N_CANDIDATES = 8   # same as 1370

CONDITIONS = ['ROLLOUT-ARGMAX', 'SSM-DISCONNECTED']

MAX_N_ACTIONS   = 4103
ENT_CHECKPOINTS = [100, 500, 1000, 1999]
TIER1_STEPS     = 200


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

    def dream_forward(self, x):
        delta_pre = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        delta = np.log1p(np.exp(delta_pre))
        A_diag = np.exp(-delta * self.A_param)
        self.h = A_diag * self.h + self.B @ x
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

    def __init__(self, n_actions, rollout_mode='argmax', max_steps=TRY1_STEPS):
        self.n_actions = n_actions
        self.rollout_mode = rollout_mode
        self._in_dim = PROJ_DIM + ACT_EMBED_DIM

        self._obs_proj = None
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)

        self._W_in = _det_init(D, self._in_dim, scale=0.1)
        self._b_in = np.zeros(D, dtype=np.float32)

        self._layers = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]

        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        # Fixed random action head (entropy/disconnected mode)
        self._W_act = _det_init(n_actions, D, scale=0.1)
        self._b_act = np.zeros(n_actions, dtype=np.float32)

        self._step        = 0
        self._prev_action = 0
        self._prev_y      = None

        self._try1_ent        = {}
        self._try2_ent        = {}
        self._try1_state_norm = {}
        self._try2_state_norm = {}
        self._in_try2         = False
        self._pred_losses     = []

        self._try1_rollout_score_mean = {}
        self._try2_rollout_score_mean = {}

        self._try1_autocorr_count = 0
        self._try1_autocorr_steps = 0
        self._try2_autocorr_count = 0
        self._try2_autocorr_steps = 0

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
        pred = self._W_pred @ y + self._b_pred
        error = pred - proj_obs_next
        e = self._W_pred.T @ error
        self._W_pred -= SSM_LR * np.outer(error, y)
        self._b_pred -= SSM_LR * error
        for layer in reversed(self._layers):
            e = layer.rtrl_update(e)
        return float(np.mean(error ** 2))

    def _rollout_score(self, proj_obs, a_cand):
        saved_h = [layer.h.copy() for layer in self._layers]

        act_emb = self._act_embed[a_cand]
        x_in = np.concatenate([proj_obs, act_emb])
        x = self._W_in @ x_in + self._b_in
        y = x
        for layer in self._layers:
            y = layer.dream_forward(y)
        pred = self._W_pred @ y + self._b_pred

        score = 0.0
        for _ in range(K_ROLLOUT - 1):
            r_act = int(np.random.randint(self.n_actions))
            act_emb_r = self._act_embed[r_act]
            x_in_d = np.concatenate([pred, act_emb_r])
            x_d = self._W_in @ x_in_d + self._b_in
            y_d = x_d
            for layer in self._layers:
                y_d = layer.dream_forward(y_d)
            pred_new = self._W_pred @ y_d + self._b_pred
            score += float(np.linalg.norm(pred_new - pred))
            pred = pred_new

        for i, layer in enumerate(self._layers):
            layer.h[:] = saved_h[i]

        return score

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat
        y = self._ssm_forward(proj_obs)
        self._prev_y = y

        if self._step < WARMUP:
            action = int(np.random.randint(self.n_actions))
        elif self.rollout_mode == 'entropy':
            logits = self._W_act @ y + self._b_act
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            action = int(np.random.choice(self.n_actions, p=probs))
        else:
            candidates = np.random.randint(0, self.n_actions, size=N_CANDIDATES)
            scores = np.array([self._rollout_score(proj_obs, int(c)) for c in candidates],
                              dtype=np.float32)
            if self.rollout_mode == 'argmin':
                action = int(candidates[np.argmin(scores)])
            else:
                action = int(candidates[np.argmax(scores)])

            if self._in_try2:
                self._try2_autocorr_steps += 1
                if action == self._prev_action:
                    self._try2_autocorr_count += 1
            else:
                self._try1_autocorr_steps += 1
                if action == self._prev_action:
                    self._try1_autocorr_count += 1

        if self._step in ENT_CHECKPOINTS and self._step >= WARMUP:
            test_cands = np.random.randint(0, self.n_actions, size=8)
            test_scores = [self._rollout_score(proj_obs, int(c)) for c in test_cands]
            score_mean = float(np.mean(test_scores))
            score_std = float(np.std(test_scores))
            ent_proxy = round(score_std / (score_mean + 1e-10), 4)
            if self._in_try2:
                self._try2_ent[self._step] = ent_proxy
                self._try2_rollout_score_mean[self._step] = round(score_mean, 6)
            else:
                self._try1_ent[self._step] = ent_proxy
                self._try1_rollout_score_mean[self._step] = round(score_mean, 6)

        self._step += 1
        self._prev_action = action
        return action

    def update_after_step(self, obs_next, action, reward):
        if self._prev_y is None or self._obs_proj is None:
            return
        obs_next_flat = _encode_obs(np.asarray(obs_next, dtype=np.float32))
        proj_obs_next = self._obs_proj @ obs_next_flat
        loss = self._rtrl_step(proj_obs_next, self._prev_y)
        self._pred_losses.append(loss)

    def on_level_transition(self):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None

    def reset_for_try2(self):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None
        self._step = 0
        self._prev_action = 0
        self._in_try2 = True
        self._try2_autocorr_count = 0
        self._try2_autocorr_steps = 0

    def get_stats(self):
        pred_loss_mean = round(float(np.mean(self._pred_losses)), 6) if self._pred_losses else None
        pred_loss_final = round(float(np.mean(self._pred_losses[-50:])), 6) if len(self._pred_losses) >= 50 else pred_loss_mean
        t1_ac = round(self._try1_autocorr_count / self._try1_autocorr_steps, 4) if self._try1_autocorr_steps > 0 else None
        t2_ac = round(self._try2_autocorr_count / self._try2_autocorr_steps, 4) if self._try2_autocorr_steps > 0 else None
        return {
            'try1_score_cv':       self._try1_ent,
            'try2_score_cv':       self._try2_ent,
            'try1_score_mean':     self._try1_rollout_score_mean,
            'try2_score_mean':     self._try2_rollout_score_mean,
            'try1_state_norm':     self._try1_state_norm,
            'try2_state_norm':     self._try2_state_norm,
            'pred_loss_mean':      pred_loss_mean,
            'pred_loss_final':     pred_loss_final,
            'try1_autocorr':       t1_ac,
            'try2_autocorr':       t2_ac,
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

def run_draw(draw_idx, draw_seed, cond_name, rollout_mode, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
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

        substrate = MambaSSMSubstrate(n_actions=n_actions, rollout_mode=rollout_mode,
                                      max_steps=max_steps)

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
            'condition': cond_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'try1_score_cv_100':  stats['try1_score_cv'].get(100),
            'try1_score_cv_1999': stats['try1_score_cv'].get(1999),
            'try2_score_cv_100':  stats['try2_score_cv'].get(100),
            'try2_score_cv_1999': stats['try2_score_cv'].get(1999),
            'try1_score_mean_100':  stats['try1_score_mean'].get(100),
            'try1_score_mean_1999': stats['try1_score_mean'].get(1999),
            'try2_score_mean_100':  stats['try2_score_mean'].get(100),
            'try2_score_mean_1999': stats['try2_score_mean'].get(1999),
            'pred_loss_mean':  stats['pred_loss_mean'],
            'pred_loss_final': stats['pred_loss_final'],
            'try1_autocorr':   stats['try1_autocorr'],
            'try2_autocorr':   stats['try2_autocorr'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)
        elapsed_total = round(time.time() - t0, 1)
        print(f"    [{cond_name}] {label}: p1={p1} p2={p2} eff_sq={eff_sq:.6f} pred_loss={stats['pred_loss_final']} ({elapsed_total}s)")

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}")
    return round(rhae, 7), draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check (ROLLOUT K=5 N=8) ===")
    games_t1, labels_t1 = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    sub_t1 = MambaSSMSubstrate(n_actions=na_t1, rollout_mode='argmax')
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)
    sub_t1._step = WARMUP
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step_rollout = tier1_elapsed / TIER1_STEPS * 1000
    ms_per_step_disc = ms_per_step_rollout * 0.15  # DISC ~15% of rollout

    n_eps_per_cond = N_DRAWS * 3 * 2
    est_rollout_s = ms_per_step_rollout / 1000 * TRY1_STEPS * n_eps_per_cond
    est_disc_s = ms_per_step_disc / 1000 * TRY1_STEPS * n_eps_per_cond
    est_total_s = est_rollout_s + est_disc_s

    print(f"  {TIER1_STEPS} steps (K=5 N=8 ROLLOUT): {tier1_elapsed:.2f}s  ({ms_per_step_rollout:.2f}ms/step)")
    print(f"  Estimated: ROLLOUT={est_rollout_s:.0f}s  DISC~{est_disc_s:.0f}s  TOTAL={est_total_s:.0f}s ({est_total_s/60:.1f} min)")
    print(f"  1370 comparison: K=3 N=8 was 0.60ms/step. K=5 ratio={ms_per_step_rollout/0.60:.2f}x")

    if est_total_s > MAX_SECONDS:
        max_steps = max(WARMUP + 50, int((MAX_SECONDS * 0.85) / (ms_per_step_rollout / 1000 * n_eps_per_cond)))
        print(f"  Budget exceeded — capping at {max_steps} steps per episode")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — proceeding with {max_steps} steps")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"\n=== STEP {STEP}: K=5 N=8 rollout vs disconnected, 30 draws ===")
    print(f"Same seeds as 1370 (13700-13729) for direct K=3 vs K=5 comparison.")
    print(f"K_ROLLOUT={K_ROLLOUT}, N_CANDIDATES={N_CANDIDATES}, WARMUP={WARMUP}")
    print(f"SSM: D={D}, N_STATE={N_STATE}, N_LAYERS={N_LAYERS}")
    print(f"1370 reference: ROLLOUT K=3 N=8 wins=4, losses=2 on same seeds.")
    print(f"Target: wins > 4 OR losses < 2.")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rhae_by_cond = {c: [] for c in CONDITIONS}
    all_results  = []

    for cond_name in CONDITIONS:
        if 'DISCONNECTED' in cond_name:
            rollout_mode = 'entropy'
        elif 'ARGMIN' in cond_name:
            rollout_mode = 'argmin'
        else:
            rollout_mode = 'argmax'
        print(f"\n--- Condition: {cond_name} ---")
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, draw_res = run_draw(di, ds, cond_name, rollout_mode, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
            all_results.extend(draw_res)

    # Summary
    summary = {
        'step': STEP,
        'n_draws': N_DRAWS,
        'draw_seeds': DRAW_SEEDS,
        'max_steps_used': max_steps,
        'ms_per_step_rollout': round(ms_per_step_rollout, 2),
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'k_rollout': K_ROLLOUT,
        'n_candidates': N_CANDIDATES,
        'conditions': {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list = rhae_by_cond[cond_name]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz = sum(1 for r in rhae_list if r > 0)
        verdict = "SIGNAL" if chain_mean > MLP_TP_BASELINE else "KILL"
        print(f"  {cond_name}: chain_mean={chain_mean:.3e}  ({nz}/{N_DRAWS} nz)  [{verdict}]")
        summary['conditions'][cond_name] = {
            'chain_mean_rhae': chain_mean,
            'nonzero_draws': nz,
            'rhae_per_draw': rhae_list,
        }

    # Paired comparison
    if 'ROLLOUT-ARGMAX' in rhae_by_cond and 'SSM-DISCONNECTED' in rhae_by_cond:
        rollout = rhae_by_cond['ROLLOUT-ARGMAX']
        disc = rhae_by_cond['SSM-DISCONNECTED']
        rollout_wins = sum(1 for r, d in zip(rollout, disc) if r > d)
        disc_wins = sum(1 for r, d in zip(rollout, disc) if d > r)
        ties = sum(1 for r, d in zip(rollout, disc) if r == d)
        print(f"\n  Paired: ROLLOUT wins {rollout_wins}/{N_DRAWS}, DISCONNECTED wins {disc_wins}/{N_DRAWS}, ties {ties}/{N_DRAWS}")
        print(f"  1370 reference (K=3 N=8 same seeds): ROLLOUT wins 4, DISC wins 2, ties 24")

        if disc_wins == 0 and rollout_wins >= 6:
            verdict_str = 'CONFIRMED_MONOTONIC'
            print(f"  *** CONFIRMED MONOTONIC — {rollout_wins} wins, 0 losses ***")
        elif disc_wins > rollout_wins:
            verdict_str = 'WORSE_THAN_K3'
            print(f"  *** K=5 WORSE than K=3 — disc_wins={disc_wins} > rollout_wins={rollout_wins} ***")
        elif rollout_wins > 4 or disc_wins < 2:
            verdict_str = 'IMPROVED_OVER_K3'
            print(f"  *** IMPROVED over K=3: wins {rollout_wins} (was 4), losses {disc_wins} (was 2) ***")
        else:
            verdict_str = 'NO_CHANGE'
            print(f"  No improvement over K=3 N=8")

        summary['paired_rollout_wins'] = rollout_wins
        summary['paired_disc_wins'] = disc_wins
        summary['paired_ties'] = ties
        summary['verdict_vs_k3'] = verdict_str
        # Per-draw comparison with 1370
        summary['ref_1370_k3_n8'] = {
            'paired_rollout_wins': 4,
            'paired_disc_wins': 2,
            'paired_ties': 24,
        }

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {RESULTS_DIR}/summary.json")
    return rhae_by_cond


if __name__ == '__main__':
    main()
