"""
Step 1347 — Deliberation + informed action selection (K=10 + action-conditional model).
Leo mail 3829, 2026-03-29.

1346: K=10 random RHAE=0, K=1 random RHAE=0. No deliberation signal with random actions.
Key: 1346 cr=1.0 — 200 action transitions → sparse dataset, no useful compression.

This step tests: does the better-trained model USE deliberation to choose better actions?

K=10 deliberation, but at each action step: sample 32 candidates, pick argmax(||pred_h3_next - h3||).
Model has 9 fresh TP training steps since last action → recalibrated features for selection.

Key difference from 1340-1343: model recalibrates BETWEEN every action (not just at start).

Conditions: only K10-MODEL (new run). Compare against:
  K10-ENT from 1346: RHAE=0 (5 draws)
  K1-ENT from 1344:  RHAE=0 (5 draws)

Protocol: same 5 seeds (13440-13444). 30 episodes × ~15s ≈ 7.5 min.

Constitutional audit:
  R0: deterministic init ✓ | R1: self-supervised prediction ✓
  R2: TP local targets ✓. Action-conditional selection reads from W (h3). ✓
  K only affects action frequency, not update mechanism.
"""
import sys, os, time, json, logging, hashlib
from collections import deque

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import (
    select_games, seal_mapping, label_filename,
    masked_game_list,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2, write_experiment_results,
    ARC_OPTIMAL_STEPS_PROXY,
)

STEP        = 1347
TRY1_STEPS  = 2000     # computation steps per try
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 340

N_DRAWS     = 5
DRAW_SEEDS  = [13440, 13441, 13442, 13443, 13444]  # same as 1344/1345/1346

K           = 10       # act every K computation steps
K_CANDIDATES = 32      # candidates for model-based selection

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1347')

_DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN      = 512
_LR          = 1e-4
_TP_LR       = 0.01
_BUFFER_MAX  = 200_000
_BATCH_SIZE  = 64
_ACTION_ENC_DIM = 16
LOSS_CHECKPOINTS = [500, 1000, 2000]


# ---------------------------------------------------------------------------
# Obs encoding
# ---------------------------------------------------------------------------

def _is_arc_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _obs_to_one_hot(obs_arr):
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _encode_obs_mlp(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        return _obs_to_one_hot(arr).astype(np.float32).flatten()
    return arr.flatten()


def _encode_arc_action(action):
    """Rich ARC action encoding: (type_1hot[7], x_norm, y_norm, zeros[7]) = 16-dim."""
    enc = np.zeros(16, dtype=np.float32)
    enc[int(action) % 7] = 1.0           # type one-hot (7 types)
    enc[7]  = (int(action) // 7 % 64) / 63.0   # x_norm
    enc[8]  = (int(action) // (7 * 64) % 64) / 63.0  # y_norm
    # enc[9:16] = zeros (pad)
    return enc


# ---------------------------------------------------------------------------
# MLP model with 2-layer action-conditional prediction head (from 1343)
# ---------------------------------------------------------------------------

class MlpModelAC(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden)
        self.fc1         = nn.Linear(hidden, hidden)
        self.fc2         = nn.Linear(hidden, hidden)
        self.fc3         = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        # 2-layer action-conditional pred head (same as 1343)
        self.pred_head1  = nn.Linear(hidden + _ACTION_ENC_DIM, hidden)
        self.pred_head2  = nn.Linear(hidden, hidden)
        self.dropout     = nn.Dropout(0.2)

    def forward(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        return self.action_head(h3), h3

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0

    def pred_next(self, h3, action_enc):
        """Predict next h3 given current h3 and action encoding."""
        pred_in = torch.cat([h3, action_enc], dim=-1)
        return self.pred_head2(F.relu(self.pred_head1(pred_in)))


# ---------------------------------------------------------------------------
# Deliberation + model-based substrate (K=10 + argmax novelty)
# ---------------------------------------------------------------------------

class DelibModelSubstrate:
    """MLP + TP + K=10 deliberation + action-conditional argmax selection.

    At each action step (every K=10 comp steps): sample K_CANDIDATES actions,
    compute predicted novelty = ||pred_h3_next - h3|| for each, pick argmax.
    Between action steps: run internal TP training on buffer.

    MBPP: char_embed(n_actions, 16). ARC: rich positional encoding.
    """

    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._is_mbpp    = n_actions <= 200
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._embed      = None   # for MBPP char embedding
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._opt_embed  = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._prev_h3       = None   # cached h3 for novelty computation
        self._step          = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpModelAC(input_dim, self.n_actions).to(_DEVICE)
        if self._is_mbpp:
            self._embed = nn.Embedding(self.n_actions, _ACTION_ENC_DIM).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(
            list(self._model.pred_head1.parameters()) +
            list(self._model.pred_head2.parameters()), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                       for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }
        if self._is_mbpp and self._embed is not None:
            self._opt_embed = torch.optim.Adam(self._embed.parameters(), lr=_LR)

    def _encode_action(self, action_int):
        """Encode single action to 16-dim numpy array."""
        if self._is_mbpp:
            idx = torch.tensor([int(action_int) % self.n_actions], device=_DEVICE)
            with torch.no_grad():
                return self._embed(idx).squeeze(0).cpu().numpy()
        else:
            return _encode_arc_action(action_int)

    def _encode_actions_batch(self, actions):
        """Encode batch of action ints to (N, 16) tensor on device."""
        if self._is_mbpp:
            idx = torch.tensor([int(a) % self.n_actions for a in actions], device=_DEVICE)
            with torch.no_grad():
                return self._embed(idx)
        else:
            encs = np.stack([_encode_arc_action(a) for a in actions])
            return torch.from_numpy(encs).to(_DEVICE)

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._prev_h3 = None

    def process(self, obs_arr):
        """Action step: encode obs, select best of K_CANDIDATES by predicted novelty."""
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            _, h3 = self._model(tensor)
        self._prev_h3 = h3  # cache for internal training

        # Model-based selection: argmax ||pred_h3_next - h3||
        candidates = self._rng.randint(0, self.n_actions, K_CANDIDATES)
        with torch.no_grad():
            action_encs = self._encode_actions_batch(candidates)  # (K, 16)
            h3_expanded = h3.expand(K_CANDIDATES, -1)              # (K, hidden)
            pred_nexts  = self._model.pred_next(h3_expanded, action_encs)  # (K, hidden)
            novelty     = torch.norm(pred_nexts - h3_expanded, dim=1)      # (K,)
            best_idx    = int(torch.argmax(novelty).item())
        return int(candidates[best_idx]) % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next     = _encode_obs_mlp(obs_next)
        action_enc   = self._encode_action(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state':      self._prev_enc.copy(),
                                 'action_enc': action_enc.copy(),
                                 'action_int': int(action),
                                 'next_state': enc_next.copy()})

    def internal_train(self):
        """Internal processing: TP training on buffer. Advances _step."""
        self._step += 1
        if len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                self._check_checkpoints()

    def _check_checkpoints(self):
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_h3  = None

    def _tp_train_step(self):
        n   = len(self._buffer); buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states      = torch.from_numpy(np.stack([b['state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32)
                                                 for b in batch])).to(_DEVICE)
        action_encs = torch.from_numpy(np.stack([b['action_enc']
                                                 for b in batch])).to(_DEVICE)

        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_n = self._model.pred_next(h3, action_encs)
            pred_err  = pred_n - target_h3_next
            pred_loss = float((pred_err ** 2).mean())
            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        with torch.enable_grad():
            pred_n2  = self._model.pred_next(h3.detach(), action_encs.detach())
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                hr = self._g[gk](h_in.detach())
                lg = F.mse_loss(hr, h_out.detach())
                self._opt_g[gk].zero_grad(); lg.backward(); self._opt_g[gk].step()

        for fk, layer, x_in, target in [
            ('f3',     self._model.fc3,        h2,     target_h3),
            ('f2',     self._model.fc2,        h1,     target_h2),
            ('f1',     self._model.fc1,        h0,     target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                hf = F.relu(layer(x_in.detach()))
                lf = F.mse_loss(hf, target.detach())
                self._opt_f[fk].zero_grad(); lf.backward(); self._opt_f[fk].step()

        return pred_loss

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None


# ---------------------------------------------------------------------------
# Game factory
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
# Episode runner (K-aware, same as 1346)
# ---------------------------------------------------------------------------

def run_episode_k(env, substrate, n_actions, seed, max_comp_steps, K):
    obs            = env.reset(seed=seed)
    comp_step      = 0
    action_step    = 0
    level          = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True

    while comp_step < max_comp_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)

        if comp_step % K == 0:
            action   = substrate.process(obs_arr) % n_actions
            obs_next, reward, done, info = env.step(action)
            action_step += 1
            comp_step   += 1

            if obs_next is not None:
                substrate.update_after_step(obs_next, action, reward)

            if fresh_episode:
                fresh_episode = False
                obs = obs_next
                continue

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                progress_count += 1
                if steps_to_first_progress is None:
                    steps_to_first_progress = action_step
                level = cl
                substrate.on_level_transition()

            if done:
                obs = env.reset(seed=seed)
                substrate.on_level_transition()
                fresh_episode = True
                level = 0
            else:
                obs = obs_next
        else:
            substrate.internal_train()
            comp_step += 1

    elapsed = time.time() - t_start
    return steps_to_first_progress, progress_count, round(elapsed, 2)


# ---------------------------------------------------------------------------
# Single draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    print(f"  Draw {draw_idx} (seed={draw_seed}): {masked_game_list(game_labels)}")
    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}

    for game_name, label in zip(games, game_labels.values()):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = 4103

        substrate = DelibModelSubstrate(n_actions=n_actions)

        p1, _, t1 = run_episode_k(env, substrate, n_actions, seed=0,
                                  max_comp_steps=TRY1_STEPS, K=K)
        substrate.reset_for_try2()
        p2, _, t2 = run_episode_k(env, substrate, n_actions, seed=TRY2_SEED,
                                  max_comp_steps=TRY2_STEPS, K=K)

        speedup = compute_progress_speedup(p1, p2)
        opt     = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq  = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff    = min(1.0, opt / p2)
            eff_sq = round(eff ** 2, 6)

        cr = substrate.get_compression_ratio()
        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        result = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
            't1': t1, 't2': t2,
        }
        draw_results.append(result)

        print(f"    {label}: speedup={format_speedup(speedup)}  eff²={eff_sq}  cr={cr}  ({t1+t2:.1f}s)")

        out_path = os.path.join(draw_dir, label_filename(label, STEP))
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

    rhae     = compute_rhae_try2(try2_progress, optimal_steps_d)
    chain_sp = sum(speedup_for_chain(r['speedup']) for r in draw_results) / len(draw_results)
    print(f"    → RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")
    return draw_results, rhae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — K=10 deliberation + model-based selection. {N_DRAWS} draws.")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × 3 games × 2 tries = {N_DRAWS*3*2} episodes")
    print(f"K={K} comp steps/action (200 actions). K_CAND={K_CANDIDATES}. Seeds: {DRAW_SEEDS}")
    print()

    # Tier 1: timing on first draw's first ARC game
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 comp steps, K={K})...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = DelibModelSubstrate(n_actions=na_t)
        t0    = time.time()
        obs_t = env_t.reset(seed=0)
        for cs in range(100):
            if obs_t is None:
                obs_t = env_t.reset(seed=0)
                continue
            if cs % K == 0:
                a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
                obs_t, _, done_t, _ = env_t.step(a)
                if done_t or obs_t is None:
                    obs_t = env_t.reset(seed=0)
            else:
                sub_t.internal_train()
        elapsed_100 = time.time() - t0
        est_ep      = elapsed_100 / 100 * 2000
        est_total   = est_ep * N_DRAWS * 3 * 2
        print(f"  100 comp steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_total:.0f}s")
        if est_total > 900:
            print(f"  WARNING: est {est_total:.0f}s > 15-min budget.")
        print()

    all_results   = []
    rhae_per_draw = []

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        all_results.extend(draw_results)
        rhae_per_draw.append(draw_rhae)
        print()

    chain_mean    = sum(rhae_per_draw) / len(rhae_per_draw)
    nonzero_draws = sum(1 for r in rhae_per_draw if r is not None and r > 0)
    nonzero_games = sum(1 for r in all_results if r['eff_sq'] > 0)
    all_eff_sq    = [r['eff_sq'] for r in all_results]

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (K=10 deliberation + model-based selection)")
    print()
    print(f"  RHAE per draw:  {[f'{r:.6f}' for r in rhae_per_draw]}")
    print(f"  Chain mean:     {chain_mean:.6f}")
    print(f"  Non-zero draws: {nonzero_draws}/{N_DRAWS}")
    print(f"  Non-zero games: {nonzero_games}/15")
    print(f"  eff² max={max(all_eff_sq):.6f}  mean={sum(all_eff_sq)/len(all_eff_sq):.6f}")
    print()
    print("  Games with progress:")
    for r in all_results:
        if r['eff_sq'] > 0:
            print(f"    Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}  cr={r['cr']}")
    print()

    # Comparison
    print("  Baselines (same seeds):")
    print("    K10-ENT (1346): RHAE=[0,0,0,0,0] chain=0. Non-zero: 0/5 draws.")
    print("    K1-ENT  (1344): RHAE=[0,0,0,0,0] chain=0. Non-zero: 0/5 draws.")
    print()

    print("ASSESSMENT:")
    if nonzero_draws >= 2:
        print(f"  >>> DELIBERATION + SELECTION SIGNAL: {nonzero_draws}/5 draws non-zero.")
        print(f"  >>> Chain mean={chain_mean:.6f}. Model-based selection beats random in K=10.")
        print(f"  >>> Recalibration between actions enables better choices. FORWARD.")
    elif nonzero_draws == 1:
        print(f"  >>> POSSIBLE SIGNAL: 1/5 draws non-zero (chain mean={chain_mean:.6f}).")
        print(f"  >>> Ambiguous — could be luck. Need more draws or larger K_CAND.")
    else:
        print(f"  >>> NO SIGNAL: K10-MODEL RHAE=0 same as K10-ENT and K1-ENT.")
        print(f"  >>> 200 model-selected actions still can't reach progress.")
        print(f"  >>> 1346 contingency met → proceed to 1348 (childhood pre-training).")
    print("=" * 80)

    summary = {
        'step':          STEP,
        'K':             K,
        'K_candidates':  K_CANDIDATES,
        'n_draws':       N_DRAWS,
        'draw_seeds':    DRAW_SEEDS,
        'rhae_per_draw': rhae_per_draw,
        'chain_mean':    chain_mean,
        'nonzero_draws': nonzero_draws,
        'nonzero_games': nonzero_games,
        'all_eff_sq':    all_eff_sq,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump({'step': STEP, 'results': all_results}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
