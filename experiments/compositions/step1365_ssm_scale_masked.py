"""
Step 1365 — SSM disconnected 2K: replicate 1364 + 4-layer scale test.
Leo mail 3896, 2026-03-30.

1364 result: RHAE=1.34e-4, 2.92× MLP+TP. Replicate and test if deeper SSM helps.

Two conditions:
- SSM-2L: Same as 1364 exactly (2 layers, D=128, N=32). Fresh seeds.
- SSM-4L: 4 layers, D=128, N=32. Everything else identical.

Both disconnected: W_act fixed random, RTRL trains obs prediction only.

Protocol:
- 10 draws, fresh seeds 13640-13649
- 2K steps per episode
- Kill: SSM-2L chain_mean < 5e-5 → 1364 was draw variance
- Signal: SSM-4L > SSM-2L → depth helps
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

STEP        = 1365
N_DRAWS     = 10
DRAW_SEEDS  = [13640 + i for i in range(N_DRAWS)]
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')

MLP_TP_BASELINE  = 4.59e-5  # Step 1349
SSM_2L_BASELINE  = 1.341e-4  # Step 1364 (2-layer disconnected 2K)

# SSM config
PROJ_DIM      = 64    # fixed random projection: obs -> this
ACT_EMBED_DIM = 16    # action embedding dim
D             = 128   # SSM model dimension
N_STATE       = 32    # SSM state dimension per layer
WARMUP        = 500   # random actions before using action head
SSM_LR        = 1e-3  # RTRL learning rate

# Two conditions
CONDITIONS = {
    'SSM-2L': {'n_layers': 2},
    'SSM-4L': {'n_layers': 4},
}

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
        # Vectorized one-hot: (16, 64, 64) -> flatten to 65536
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
        # Vectorized sin pattern — deterministic, no QR overhead
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

        # Learnable parameters
        self.B       = _det_init(n_state, d, scale=0.1)    # (N, D)
        self.C       = _det_init(d, n_state, scale=0.1)    # (D, N)
        self.A_param = np.ones(n_state, dtype=np.float32) * 0.5  # (N,) -> A_diag~0.7
        self.W_delta = np.zeros((n_state, d), dtype=np.float32)  # (N, D)
        self.b_delta = np.zeros(n_state, dtype=np.float32)       # (N,)

        # Recurrent state + RTRL sensitivity trace
        self.h = np.zeros(n_state, dtype=np.float32)
        self.S = np.zeros(n_state, dtype=np.float32)  # d(h_t)/d(A_param)

        # Stored activations for RTRL (set during forward)
        self._x         = None
        self._h_prev    = None
        self._A_diag    = None
        self._delta     = None
        self._delta_pre = None

    def forward(self, x):
        """Forward pass. x: (D,) -> y: (D,)."""
        self._x = x
        self._h_prev = self.h.copy()

        delta_pre = np.clip(self.W_delta @ x + self.b_delta, -20.0, 20.0)
        delta = np.log1p(np.exp(delta_pre))  # softplus, (N,)
        self._delta_pre = delta_pre
        self._delta = delta

        A_diag = np.exp(-delta * self.A_param)  # (N,) in (0, 1]
        self._A_diag = A_diag

        self.h = A_diag * self._h_prev + self.B @ x  # (N,)
        return self.C @ self.h                        # (D,)

    def rtrl_update(self, e_y):
        """RTRL update. e_y: error in D-space. Returns e_x: error propagated to input D-space."""
        if self._x is None:
            return np.zeros(self.d, dtype=np.float32)

        e_h = self.C.T @ e_y    # (N,) error in state space

        # Propagate to input using OLD B (before B update)
        e_x = self.B.T @ e_h   # (D,)

        # Update C: dL/dC = outer(e_y, h_t)
        self.C -= self.lr * np.outer(e_y, self.h)

        # Update B: dL/dB = outer(e_h, x)
        self.B -= self.lr * np.outer(e_h, self._x)

        # Sensitivity trace: S_t = A_diag * S_{t-1} + (-delta * A_diag * h_{t-1})
        self.S = self._A_diag * self.S + (-self._delta * self._A_diag * self._h_prev)

        # Update A_param: dL/dA = e_h * S_t
        self.A_param -= self.lr * (e_h * self.S)
        self.A_param = np.clip(self.A_param, 0.01, 10.0)

        # Update W_delta via chain rule through softplus and A_diag
        # dL/d(delta) = e_h * (-A_param * A_diag * h_prev)
        # dL/d(delta_pre) = dL/d(delta) * sigmoid(delta_pre)
        sigmoid_dp = 1.0 / (1.0 + np.exp(-self._delta_pre))
        dL_d_delta = e_h * (-self.A_param * self._A_diag * self._h_prev)
        dL_d_logit = dL_d_delta * sigmoid_dp
        self.W_delta -= self.lr * np.outer(dL_d_logit, self._x)
        self.b_delta -= self.lr * dL_d_logit

        return e_x

    def reset_state(self):
        """Reset recurrent state. Keep learned weights."""
        self.h[:] = 0.0
        self.S[:] = 0.0
        self._x = self._h_prev = self._A_diag = self._delta = self._delta_pre = None


# ---------------------------------------------------------------------------
# SSM Substrate
# ---------------------------------------------------------------------------

class MambaSSMSubstrate:
    """Mamba-style SSM with online RTRL. n_layers configurable."""

    def __init__(self, n_actions, n_layers=2, max_steps=TRY1_STEPS):
        self.n_actions = n_actions
        self.max_steps = max_steps
        self._in_dim = PROJ_DIM + ACT_EMBED_DIM

        # Fixed random obs projection (lazy: initialized on first obs)
        self._obs_proj = None   # (PROJ_DIM, obs_dim) float32

        # Action embedding: fixed (not trained). Deterministic init.
        self._act_embed = _det_init(n_actions, ACT_EMBED_DIM, scale=0.1)  # (n_actions, 16)

        # Input merge: concat(PROJ_DIM, ACT_EMBED_DIM) -> D
        self._W_in = _det_init(D, self._in_dim, scale=0.1)  # (D, in_dim)
        self._b_in = np.zeros(D, dtype=np.float32)

        # SSM layers (n_layers configurable)
        self._layers = [SSMLayer(D, N_STATE, SSM_LR) for _ in range(n_layers)]

        # Prediction head: D -> PROJ_DIM (learnable via RTRL)
        self._W_pred = _det_init(PROJ_DIM, D, scale=0.1)   # (PROJ_DIM, D)
        self._b_pred = np.zeros(PROJ_DIM, dtype=np.float32)

        # Action head: D -> n_actions (FIXED, not trained)
        self._W_act = _det_init(n_actions, D, scale=0.1)   # (n_actions, D)
        self._b_act = np.zeros(n_actions, dtype=np.float32)

        # Running state
        self._step        = 0
        self._prev_action = 0
        self._prev_y      = None

        # Diagnostics
        self._try1_ent        = {}
        self._try2_ent        = {}
        self._try1_state_norm = {}
        self._try2_state_norm = {}
        self._in_try2         = False
        self._pred_losses     = []

    # ---- Obs projection ----

    def _init_obs_proj(self, obs_flat):
        obs_dim = obs_flat.shape[0]
        # Deterministic sin-pattern projection (no QR — obs_dim can be 65536)
        i = np.arange(PROJ_DIM, dtype=np.float64).reshape(-1, 1)
        j = np.arange(obs_dim, dtype=np.float64).reshape(1, -1)
        W = np.sin(i * 1.234 + j * 0.00731 + 0.5)
        norms = np.linalg.norm(W, axis=1, keepdims=True)
        self._obs_proj = (W / (norms + 1e-8)).astype(np.float32)

    # ---- Forward pass ----

    def _ssm_forward(self, proj_obs):
        """Full SSM forward: proj_obs (PROJ_DIM,) -> y (D,)."""
        act_emb = self._act_embed[self._prev_action]       # (ACT_EMBED_DIM,)
        x_in = np.concatenate([proj_obs, act_emb])         # (in_dim,)
        x = self._W_in @ x_in + self._b_in               # (D,)
        y = x
        for layer in self._layers:
            y = layer.forward(y)
        return y

    # ---- RTRL update ----

    def _rtrl_step(self, proj_obs_next, y):
        """Compute prediction error and run RTRL backward."""
        pred = self._W_pred @ y + self._b_pred             # (PROJ_DIM,)
        error = pred - proj_obs_next                       # (PROJ_DIM,)

        # Propagate error through prediction head using OLD W_pred
        e = self._W_pred.T @ error                        # (D,)

        # Update prediction head
        self._W_pred -= SSM_LR * np.outer(error, y)
        self._b_pred -= SSM_LR * error

        # RTRL backward through SSM layers (reverse order)
        for layer in reversed(self._layers):
            e = layer.rtrl_update(e)

        return float(np.mean(error ** 2))

    # ---- Substrate interface ----

    def process(self, obs_arr):
        obs_flat = _encode_obs(obs_arr)
        if self._obs_proj is None:
            self._init_obs_proj(obs_flat)

        proj_obs = self._obs_proj @ obs_flat              # (PROJ_DIM,)
        y = self._ssm_forward(proj_obs)
        self._prev_y = y

        # Action selection
        if self._step < WARMUP:
            action = int(np.random.randint(self.n_actions))
        else:
            logits = self._W_act @ y + self._b_act
            logits = logits - logits.max()
            probs = np.exp(logits)
            probs /= probs.sum()
            action = int(np.random.choice(self.n_actions, p=probs))

        # Diagnostics at checkpoints
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
        loss = self._rtrl_step(proj_obs_next, self._prev_y)
        self._pred_losses.append(loss)

    def on_level_transition(self):
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None

    def reset_for_try2(self):
        """Reset recurrent state. Keep learned weights (R4 compliance)."""
        for layer in self._layers:
            layer.reset_state()
        self._prev_y = None
        self._step = 0
        self._prev_action = 0
        self._in_try2 = True

    def get_stats(self):
        pred_loss_mean = round(float(np.mean(self._pred_losses)), 6) if self._pred_losses else None
        pred_loss_final = round(float(np.mean(self._pred_losses[-50:])), 6) if len(self._pred_losses) >= 50 else pred_loss_mean
        return {
            'try1_act_ent':        self._try1_ent,
            'try2_act_ent':        self._try2_ent,
            'try1_state_norm':     self._try1_state_norm,
            'try2_state_norm':     self._try2_state_norm,
            'pred_loss_mean':      pred_loss_mean,
            'pred_loss_final':     pred_loss_final,
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

def run_draw(draw_idx, draw_seed, cond_name, n_layers, max_steps):
    """Run one draw for one condition."""
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results = []
    try2_progress = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = MambaSSMSubstrate(n_actions=n_actions, n_layers=n_layers,
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
            'draw': draw_idx, 'condition': cond_name, 'label': label, 'game': game_name,
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
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)
        elapsed_total = round(time.time() - t0, 1)
        print(f"    [{cond_name}] {label}: p1={p1} p2={p2} eff_sq={eff_sq:.6f} "
              f"pred_loss={stats['pred_loss_final']} ({elapsed_total}s)")

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}")
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

    # Use SSM-2L for timing (SSM-4L is slightly slower; estimate from 2L)
    sub_t1 = MambaSSMSubstrate(n_actions=na_t1, n_layers=2)
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # pre-warmup
    t_tier1 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t_tier1

    ms_per_step = tier1_elapsed / TIER1_STEPS * 1000
    n_conditions = len(CONDITIONS)
    n_episodes = N_DRAWS * 3 * 2 * n_conditions  # draws x games x tries x conditions
    est_total_s = ms_per_step / 1000 * TRY1_STEPS * n_episodes
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated full run ({n_conditions} conditions): {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > BUDGET_SECONDS:
        max_steps = max(200, int(BUDGET_SECONDS / (ms_per_step / 1000 * n_episodes)))
        print(f"  Budget exceeded — capping at {max_steps} steps per episode")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — proceeding with {max_steps} steps")

    # ── Full run ──────────────────────────────────────────────────────────
    print(f"\n=== STEP {STEP}: SSM disconnected 2K — replication + scale ===")
    print(f"N_DRAWS={N_DRAWS}, max_steps={max_steps}, WARMUP={WARMUP}")
    print(f"SSM: D={D}, N_STATE={N_STATE}, PROJ_DIM={PROJ_DIM}, conditions={list(CONDITIONS.keys())}")
    print(f"1364 baseline (2L 2K): {SSM_2L_BASELINE:.2e}  |  MLP+TP: {MLP_TP_BASELINE:.2e}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Results per condition
    rhae_by_cond   = {c: [] for c in CONDITIONS}
    all_results    = []

    for cond_name, cond_cfg in CONDITIONS.items():
        n_layers = cond_cfg['n_layers']
        print(f"\n--- Condition: {cond_name} (N_LAYERS={n_layers}) ---")
        for di, ds in enumerate(DRAW_SEEDS):
            print(f"  Draw {di} (seed={ds}): {masked_game_list(select_games(seed=ds)[1])}")
            rhae, draw_res = run_draw(di, ds, cond_name, n_layers, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
            all_results.extend(draw_res)

    # Per-condition chain means
    chain_means = {}
    nz_counts   = {}
    for cond_name, rhae_list in rhae_by_cond.items():
        chain_means[cond_name] = round(sum(rhae_list) / len(rhae_list), 7)
        nz_counts[cond_name]   = sum(1 for r in rhae_list if r > 0)

    summary = {
        'step':           STEP,
        'n_draws':        N_DRAWS,
        'draw_seeds':     DRAW_SEEDS,
        'conditions':     list(CONDITIONS.keys()),
        'rhae_by_cond':   rhae_by_cond,
        'chain_mean_rhae': chain_means,
        'nonzero_draws':  nz_counts,
        'ssm_2l_baseline': SSM_2L_BASELINE,
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'max_steps_used': max_steps,
        'ms_per_step':    round(ms_per_step, 2),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        cm = chain_means[cond_name]
        nz = nz_counts[cond_name]
        marker = " ← SIGNAL" if cm > SSM_2L_BASELINE else (" ← replicated" if cm > MLP_TP_BASELINE else "")
        print(f"  {cond_name}: chain_mean={cm:.3e}  ({nz}/{N_DRAWS} nz){marker}")
    print(f"  1364 baseline (SSM-2L 2K):  {SSM_2L_BASELINE:.3e}")
    print(f"  MLP+TP baseline (1349):     {MLP_TP_BASELINE:.3e}")

    return chain_means


if __name__ == '__main__':
    main()
