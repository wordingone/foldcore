"""
Step 1379 — Action-blind diagnostic: does masking the action token change prediction accuracy?
Leo mail 3939, 2026-03-30.

1378 showed frozen projection hurts (h has structure at H_norm=0.72 but random coupling fails).
Question: does the SSM actually USE the action token for prediction?

Two conditions (paired, 30 draws):
  FULL:   SSM processes (obs, act) normally. Standard RTRL.
  MASKED: SSM processes (obs, ZERO) — action token replaced with zero vector.

Both: COUNT try1, disconnected try2, h reset, PRNG fix.
Same actions chosen (COUNT try1, random try2) for both conditions — only the SSM input differs.

Primary metric: pred_loss_mean (does masking actions hurt prediction?).
Secondary: RHAE (should be identical — same actions for both due to PRNG fix + disconnected try2).

Kill criterion:
  |FULL_pred / MASKED_pred - 1.0| < 0.05  →  ACTION_BLIND (actions don't help prediction)
  FULL_pred << MASKED_pred               →  ACTION_CONDITIONAL (actions help)

Seeds: 13960-13989 (fresh).
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

STEP        = 1379
N_DRAWS     = 30
DRAW_SEEDS  = [13960 + i for i in range(N_DRAWS)]
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

# SSM config (SSM-DIAG-RTRL, same as 1360-1378)
PROJ_DIM      = 64
ACT_EMBED_DIM = 16
D             = 128
N_STATE       = 32
N_LAYERS      = 2
SSM_LR        = 1e-3

CONDITIONS = ['FULL', 'MASKED']

MAX_N_ACTIONS      = 4103
H_NORM_CHECKPOINTS = [100, 500, 1000, 1999]
I3_STEP            = 200
I4_EARLY_MAX       = 100
I4_LATE_MIN        = 1900
TIER1_STEPS        = 200

# Action-blind verdict thresholds
ACTION_BLIND_THRESHOLD = 0.05   # |ratio - 1.0| < 5% → action-blind


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
# SSM Layer
# ---------------------------------------------------------------------------

class SSMLayer:
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

    def r3_weight_diff(self):
        return float(np.linalg.norm(self.B - self._init_B, 'fro'))

    def reset_state(self):
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None


# ---------------------------------------------------------------------------
# SSM Substrate
# ---------------------------------------------------------------------------

class MambaSSMSubstrate:
    """
    try1_mode: 'count'. try2: disconnected (random) for both conditions.
    mask_action_token: if True, replace action embedding with zeros before SSM input.
    This tests whether the SSM uses action information for prediction.
    """

    def __init__(self, n_actions, mask_action_token=False):
        self.n_actions         = n_actions
        self.mask_action_token = mask_action_token
        self._in_dim           = PROJ_DIM + ACT_EMBED_DIM

        self._obs_proj  = None
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)
        self._zero_act  = np.zeros(ACT_EMBED_DIM, dtype=np.float32)  # for MASKED
        self._W_in      = _det_init(D, self._in_dim, scale=0.1)
        self._b_in      = np.zeros(D, dtype=np.float32)
        self._layers    = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(N_LAYERS)]
        self._W_pred    = _det_init(PROJ_DIM, D, scale=0.1)
        self._b_pred    = np.zeros(PROJ_DIM, dtype=np.float32)

        self._step         = 0
        self._prev_action  = 0
        self._prev_y       = None
        self._in_try2      = False
        self._visit_count  = np.zeros(n_actions, dtype=np.int32)

        self._try1_h_norm_at        = {}
        self._try2_visit_at_i3      = None
        self._try2_actions_early    = []
        self._try2_actions_late     = []
        self._try2_h_by_level       = {}
        self._current_level         = 0
        self._try2_max_level        = 0
        self._r3_diff               = None

        self._pred_losses_try1 = []   # prediction MSE during try1
        self._pred_losses_try2 = []   # prediction MSE during try2

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim, dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    def _ssm_forward(self, proj_obs):
        if self.mask_action_token:
            act_emb = self._zero_act   # MASKED: zero out action token
        else:
            act_emb = self._act_embed[self._prev_action]  # FULL: real action
        x_in = np.concatenate([proj_obs, act_emb])
        x = self._W_in @ x_in + self._b_in
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    def _rtrl_step(self, proj_obs_next, y):
        pred  = self._W_pred @ y + self._b_pred
        error = pred - proj_obs_next
        e     = self._W_pred.T @ error
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

        if not self._in_try2 and self._step in H_NORM_CHECKPOINTS:
            h_norm = float(np.mean([np.linalg.norm(l.h) for l in self._layers]))
            self._try1_h_norm_at[self._step] = round(h_norm, 4)

        if self._in_try2 and self._step % 200 == 0:
            h_concat = np.concatenate([l.h.copy() for l in self._layers])
            lvl = self._current_level
            if lvl not in self._try2_h_by_level:
                self._try2_h_by_level[lvl] = []
            if len(self._try2_h_by_level[lvl]) < 20:
                self._try2_h_by_level[lvl].append(h_concat)

        # Action selection: COUNT for try1, random for try2 (disconnected)
        if self._in_try2:
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
        loss = self._rtrl_step(proj_obs_next, self._prev_y)
        if self._in_try2:
            self._pred_losses_try2.append(loss)
        else:
            self._pred_losses_try1.append(loss)

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
        t1_mean = round(float(np.mean(self._pred_losses_try1)), 6) if self._pred_losses_try1 else None
        t2_mean = round(float(np.mean(self._pred_losses_try2)), 6) if self._pred_losses_try2 else None
        all_losses = self._pred_losses_try1 + self._pred_losses_try2
        overall    = round(float(np.mean(all_losses)), 6) if all_losses else None
        return {
            'try1_h_norm_at':    self._try1_h_norm_at,
            'pred_loss_try1':    t1_mean,
            'pred_loss_try2':    t2_mean,
            'pred_loss_overall': overall,
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

def run_draw(draw_idx, draw_seed, cond_name, mask_action_token, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = MambaSSMSubstrate(n_actions=n_actions, mask_action_token=mask_action_token)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)  # PRNG fix: same try2 RNG for both conditions
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
            'try1_h_norm_1999':  stats['try1_h_norm_at'].get(1999),
            'pred_loss_try1':    stats['pred_loss_try1'],
            'pred_loss_try2':    stats['pred_loss_try2'],
            'pred_loss_overall': stats['pred_loss_overall'],
            'i3_cv':             stage['i3_cv'],
            'i4_reduction':      stage['i4_reduction'],
            'i1_pass':           stage['i1_pass'],
            'i5_max_level':      stage['i5_max_level'],
            'r3_weight_diff':    stage['r3_weight_diff'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    # Per-draw pred_loss aggregate (mean across games)
    pred_losses_overall = [r['pred_loss_overall'] for r in draw_results if r.get('pred_loss_overall') is not None]
    draw_pred_loss = round(float(np.mean(pred_losses_overall)), 6) if pred_losses_overall else None

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  pred_loss={draw_pred_loss}")
    return round(rhae, 7), draw_pred_loss, draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, _ = select_games(seed=DRAW_SEEDS[0])
    env_t1 = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    sub_t1 = MambaSSMSubstrate(n_actions=na_t1, mask_action_token=False)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    n_eps_total = N_DRAWS * 3 * 2   # draws × games × (try1 + try2)
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * n_eps_total * 2
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.9) / (ms_per_step / 1000 * n_eps_total * 2)
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"\n=== STEP {STEP}: FULL vs MASKED action token, {N_DRAWS} draws ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")
    print(f"Kill criterion: |FULL/MASKED ratio - 1.0| < {ACTION_BLIND_THRESHOLD} → ACTION_BLIND")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rhae_by_cond     = {c: [] for c in CONDITIONS}
    pred_loss_by_cond = {c: [] for c in CONDITIONS}

    for cond_name in CONDITIONS:
        mask = (cond_name == 'MASKED')
        print(f"\n--- Condition: {cond_name} (mask_action_token={mask}) ---")
        t_cond_start = time.time()
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, pred_loss, _ = run_draw(di, ds, cond_name, mask, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
            pred_loss_by_cond[cond_name].append(pred_loss)
        print(f"  Condition done in {time.time()-t_cond_start:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'step':            STEP,
        'n_draws':         N_DRAWS,
        'draw_seeds':      DRAW_SEEDS,
        'max_steps_used':  max_steps,
        'ms_per_step':     round(ms_per_step, 2),
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'conditions':      {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list  = rhae_by_cond[cond_name]
        pred_list  = [p for p in pred_loss_by_cond[cond_name] if p is not None]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        pred_mean  = round(float(np.mean(pred_list)), 6) if pred_list else None
        verdict    = "SIGNAL" if chain_mean > MLP_TP_BASELINE else "KILL"
        print(f"  {cond_name}: chain_mean={chain_mean:.3e} ({nz}/{N_DRAWS} nz) [{verdict}]"
              f"  pred_loss={pred_mean}")
        summary['conditions'][cond_name] = {
            'chain_mean_rhae':    chain_mean,
            'nonzero_draws':      nz,
            'pred_loss_mean':     pred_mean,
            'rhae_per_draw':      rhae_list,      # internal — NOT mailed to Leo
            'pred_loss_per_draw': pred_loss_by_cond[cond_name],
        }

    # ── Action-blind verdict ──────────────────────────────────────────────
    full_pred   = summary['conditions']['FULL']['pred_loss_mean']
    masked_pred = summary['conditions']['MASKED']['pred_loss_mean']

    if full_pred is not None and masked_pred is not None and masked_pred > 0:
        ratio = full_pred / masked_pred
        pct_diff = abs(ratio - 1.0)
        if pct_diff < ACTION_BLIND_THRESHOLD:
            action_verdict = 'ACTION_BLIND'
        elif ratio < 1.0 - ACTION_BLIND_THRESHOLD:
            action_verdict = 'ACTION_CONDITIONAL'
        else:
            action_verdict = 'ACTION_HURTS'  # masked < full (unexpected)
    else:
        ratio = None
        pct_diff = None
        action_verdict = 'UNKNOWN'

    print(f"\n=== ACTION-BLIND DIAGNOSTIC ===")
    print(f"  FULL pred_loss:   {full_pred}")
    print(f"  MASKED pred_loss: {masked_pred}")
    print(f"  Ratio (FULL/MASKED): {ratio:.4f}" if ratio else "  Ratio: N/A")
    print(f"  Pct diff from 1.0: {pct_diff*100:.1f}%" if pct_diff else "")
    if action_verdict == 'ACTION_BLIND':
        print(f"  *** ACTION_BLIND: actions don't contribute to prediction (<{ACTION_BLIND_THRESHOLD*100:.0f}% difference). ***")
        print(f"  SSM ignores action token. No readout can produce useful action selection.")
    elif action_verdict == 'ACTION_CONDITIONAL':
        print(f"  *** ACTION_CONDITIONAL: actions help prediction. h encodes action info. ***")
        print(f"  SSM uses actions but for prediction accuracy, not game-solving.")
    else:
        print(f"  {action_verdict}: unexpected result (masked predicts better).")

    summary['action_blind_diagnostic'] = {
        'full_pred_loss':   full_pred,
        'masked_pred_loss': masked_pred,
        'ratio':            round(ratio, 6) if ratio else None,
        'pct_diff':         round(pct_diff, 6) if pct_diff else None,
        'verdict':          action_verdict,
    }

    # ── RHAE paired comparison ────────────────────────────────────────────
    full_list   = rhae_by_cond['FULL']
    masked_list = rhae_by_cond['MASKED']
    full_wins   = sum(1 for a, b in zip(full_list, masked_list) if a > b)
    mask_wins   = sum(1 for a, b in zip(full_list, masked_list) if b > a)
    ties        = sum(1 for a, b in zip(full_list, masked_list) if a == b)
    n_dec       = full_wins + mask_wins
    p_val       = binomial_p_one_sided(full_wins, n_dec)

    print(f"\n=== RHAE PAIRED (expected identical) ===")
    print(f"  FULL wins {full_wins}/{N_DRAWS}, MASKED wins {mask_wins}/{N_DRAWS}, "
          f"ties {ties}/{N_DRAWS}  p={p_val:.4f}")
    if ties < N_DRAWS - 2:
        print(f"  WARNING: RHAE not identical — unexpected divergence.")
    else:
        print(f"  RHAE as expected: near-identical (PRNG-fixed disconnected try2).")

    summary['rhae_paired'] = {
        'full_wins': full_wins, 'masked_wins': mask_wins, 'ties': ties,
        'p_value':   p_val,
    }

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {RESULTS_DIR}/summary.json")
    return rhae_by_cond


if __name__ == '__main__':
    main()
