"""
Step 1352 — HIER replication on 10 draws + type_head entropy tracking.
Leo mail 3853, 2026-03-29.

Replication of 1351 on 10 fresh draws. Adds:
- type_entropy at steps [100, 500, 1000, 2000]: does type_head(h3) concentrate?
- try1 vs try2 click_frac separately: does action distribution shift between tries?
- 10 draws (seeds 13500-13509) vs 1349 baseline (seeds 13490-13499, chain_mean=4.6e-5)

Kill criteria:
- ≤ 2/10 non-zero draws → 1351 was lucky, KILL.
- ≥ 3/10 non-zero draws → replicates, SIGNAL. Establish HIER as new baseline.
- type_entropy drops during episode → LANDMARK. First evidence that TP shifts type distribution.
"""
import sys, os, time, json, logging, hashlib, math
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

STEP        = 1352
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

N_DRAWS     = 10
DRAW_SEEDS  = [13500 + i for i in range(N_DRAWS)]

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1352')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]

# ARC action space structure
ARC_N_ACTIONS   = 4103
N_TYPES         = 8      # 0-6: keyboard, 7: click
N_KB_ACTIONS    = 7      # keyboard actions (indices 0-6)
N_CLICK_POS     = 4096   # click positions (indices 7-4102)

# Entropy checkpoints
ENT_CHECKPOINTS = [100, 500, 1000, 2000]
MAX_TYPE_ENTROPY = round(math.log(N_TYPES), 4)   # 2.0794


# ---------------------------------------------------------------------------
# Obs encoding (same as 1344-1351)
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


def _action_type_vec(action, n_types=6):
    vec = np.zeros(6, dtype=np.float32)
    vec[int(action) % n_types] = 1.0
    return vec


def _entropy(probs):
    """Shannon entropy of a probability array."""
    eps = 1e-10
    return float(-np.sum(probs * np.log(probs + eps)))


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class MlpModel(nn.Module):
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, hidden)
        self.fc1         = nn.Linear(hidden, hidden)
        self.fc2         = nn.Linear(hidden, hidden)
        self.fc3         = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.pred_head   = nn.Linear(hidden + 6, hidden)
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


# ---------------------------------------------------------------------------
# Hierarchical substrate
# ---------------------------------------------------------------------------

class HierarchicalSubstrate:
    """
    MLP+TP with hierarchical action selection for ARC (n_actions=4103).
    Adds type_entropy tracking at ENT_CHECKPOINTS.
    """

    def __init__(self, n_actions):
        self.n_actions    = n_actions
        self._is_hier     = (n_actions == ARC_N_ACTIONS)
        self._rng         = np.random.RandomState(42)
        self._model       = None
        self._type_head   = None
        self._pos_head    = None
        self._g           = None
        self._opt_pred    = None
        self._opt_g       = None
        self._opt_f       = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._train_counter = 0
        self._step          = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

        # Action selection stats
        self._type_counts       = np.zeros(N_TYPES, dtype=np.int64)
        self._click_count       = 0

        # Type entropy at checkpoints
        self._type_entropy_at   = {ck: None for ck in ENT_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpModel(input_dim, self.n_actions).to(_DEVICE)
        if self._is_hier:
            self._type_head = nn.Linear(_HIDDEN, N_TYPES).to(_DEVICE)
            self._pos_head  = nn.Linear(_HIDDEN, N_CLICK_POS).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR) for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }

    def get_try_stats(self):
        """Returns (click_frac, type_entropy_dict) for the current try."""
        if not self._is_hier or self._step == 0:
            return None, {ck: None for ck in ENT_CHECKPOINTS}
        total = self._type_counts.sum()
        cf = round(float(self._type_counts[N_TYPES - 1]) / total, 4) if total > 0 else 0.0
        return cf, dict(self._type_entropy_at)

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}
        self._type_counts   = np.zeros(N_TYPES, dtype=np.int64)
        self._click_count   = 0
        self._type_entropy_at = {ck: None for ck in ENT_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)

        with torch.no_grad():
            _, h3 = self._model(tensor)

            if self._is_hier:
                # Step 1: select type
                type_logits = self._type_head(h3).squeeze(0).cpu()
                type_probs  = torch.softmax(type_logits, dim=-1).numpy()
                action_type = int(np.random.choice(N_TYPES, p=type_probs))
                self._type_counts[action_type] += 1

                # Track type entropy at checkpoints
                if self._step in self._type_entropy_at and self._type_entropy_at[self._step] is None:
                    self._type_entropy_at[self._step] = round(_entropy(type_probs), 4)

                if action_type == N_TYPES - 1:  # click
                    pos_logits = self._pos_head(h3).squeeze(0).cpu()
                    pos_probs  = torch.softmax(pos_logits, dim=-1).numpy()
                    pos        = int(np.random.choice(N_CLICK_POS, p=pos_probs))
                    action     = N_KB_ACTIONS + pos
                    self._click_count += 1
                else:
                    action = action_type
            else:
                logits, _ = self._model(tensor)
                probs = torch.softmax(logits.squeeze(0).cpu(), dim=-1).numpy()
                action = int(np.random.choice(self.n_actions, p=probs))

        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state': self._prev_enc.copy(),
                                 'action_type_vec': at_vec,
                                 'next_state': enc_next.copy()})
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None

    def _tp_train_step(self):
        n = len(self._buffer); buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]
        states = torch.from_numpy(np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)
        with torch.no_grad():
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())
            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)
        with torch.enable_grad():
            pred_n2 = self._model.pred_head(torch.cat([h3.detach(), action_vecs.detach()], dim=1))
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()
        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                hr = self._g[gk](h_in.detach())
                lg = F.mse_loss(hr, h_out.detach())
                self._opt_g[gk].zero_grad(); lg.backward(); self._opt_g[gk].step()
        for fk, layer, x_in, target in [
            ('f3', self._model.fc3, h2, target_h3),
            ('f2', self._model.fc2, h1, target_h2),
            ('f1', self._model.fc1, h0, target_h1),
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
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs            = env.reset(seed=seed)
    steps          = 0
    level          = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
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
# Single draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed):
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
            n_actions = 4103

        substrate = HierarchicalSubstrate(n_actions=n_actions)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0, max_steps=TRY1_STEPS)
        try1_click_frac, try1_entropy = substrate.get_try_stats()

        substrate.reset_for_try2()
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=TRY2_STEPS)
        try2_click_frac, try2_entropy = substrate.get_try_stats()

        speedup = compute_progress_speedup(p1, p2)
        opt = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff = min(1.0, opt / p2)
            eff_sq = round(eff ** 2, 6)

        cr = substrate.get_compression_ratio()
        hier_str = f"click={try2_click_frac}" if substrate._is_hier else "flat"

        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        result = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
            'is_hier': substrate._is_hier,
            'try1_click_frac': try1_click_frac,
            'try2_click_frac': try2_click_frac,
            'try1_type_entropy_at_100':  try1_entropy.get(100),
            'try1_type_entropy_at_500':  try1_entropy.get(500),
            'try1_type_entropy_at_1000': try1_entropy.get(1000),
            'try1_type_entropy_at_2000': try1_entropy.get(2000),
            'try2_type_entropy_at_100':  try2_entropy.get(100),
            'try2_type_entropy_at_500':  try2_entropy.get(500),
            'try2_type_entropy_at_1000': try2_entropy.get(1000),
            'try2_type_entropy_at_2000': try2_entropy.get(2000),
            'max_type_entropy': MAX_TYPE_ENTROPY,
            't1': t1, 't2': t2,
        }
        draw_results.append(result)

        print(f"    {label}: speedup={format_speedup(speedup)}  eff²={eff_sq}  cr={cr}  {hier_str}  ({t1+t2:.1f}s)")
        if substrate._is_hier:
            te_traj = [try2_entropy.get(ck) for ck in ENT_CHECKPOINTS]
            print(f"      type_entropy(try2) at {ENT_CHECKPOINTS}: {te_traj}  (max={MAX_TYPE_ENTROPY})")
            click_shift = None
            if try1_click_frac is not None and try2_click_frac is not None:
                click_shift = round(try2_click_frac - try1_click_frac, 4)
            print(f"      click_frac: try1={try1_click_frac} try2={try2_click_frac} shift={click_shift}")

        out_path = os.path.join(draw_dir, label_filename(label, STEP))
        with open(out_path, 'a') as f:
            f.write(json.dumps(result, default=str) + '\n')

    rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
    chain_sp = sum(speedup_for_chain(r['speedup']) for r in draw_results) / len(draw_results)
    print(f"    → RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")
    print()
    return draw_results, rhae


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — HIER replication on 10 draws + type_head entropy tracking.")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × 3 games × 2 tries = {N_DRAWS*3*2} episodes")
    print(f"HIER: type_head(h3) + position_head(h3). ARC=hierarchical, MBPP/KB=flat.")
    print(f"Max type entropy: {MAX_TYPE_ENTROPY} (log {N_TYPES})")
    print(f"Draw seeds: {DRAW_SEEDS}")
    print()

    # Tier 1: timing
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, draw 0)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = HierarchicalSubstrate(n_actions=na_t)
        t0 = time.time()
        obs_t = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_ep  = elapsed_100 / 100 * 2000
        est_tot = est_ep * N_DRAWS * 3 * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_tot:.0f}s")
        print()

    all_results   = []
    rhae_per_draw = []

    for draw_idx in range(N_DRAWS):
        draw_results, draw_rhae = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        all_results.extend(draw_results)
        rhae_per_draw.append(draw_rhae)

    # -------------------------------------------------------------------------
    # Aggregate report
    # -------------------------------------------------------------------------
    chain_mean    = sum(rhae_per_draw) / len(rhae_per_draw)
    nonzero_draws = sum(1 for r in rhae_per_draw if r is not None and r > 0)
    nonzero_games = sum(1 for r in all_results if r['eff_sq'] > 0)
    total_games   = len(all_results)
    all_eff_sq    = [r['eff_sq'] for r in all_results]

    # Game B (click) progress
    gameb_progress = sum(1 for r in all_results if r['label'] == 'Game B' and r['eff_sq'] > 0)

    # Type entropy trajectory: mean entropy at each checkpoint across all ARC try2 episodes
    hier_results = [r for r in all_results if r['is_hier']]
    ent_traj_mean = {}
    for ck in ENT_CHECKPOINTS:
        vals = [r[f'try2_type_entropy_at_{ck}'] for r in hier_results if r[f'try2_type_entropy_at_{ck}'] is not None]
        ent_traj_mean[ck] = round(sum(vals) / len(vals), 4) if vals else None

    # Try1 vs try2 click_frac shift
    cf_shifts = []
    for r in hier_results:
        if r['try1_click_frac'] is not None and r['try2_click_frac'] is not None:
            cf_shifts.append(r['try2_click_frac'] - r['try1_click_frac'])
    mean_cf_shift = round(sum(cf_shifts) / len(cf_shifts), 4) if cf_shifts else None

    # Mean click fractions
    try2_cfs = [r['try2_click_frac'] for r in hier_results if r['try2_click_frac'] is not None]
    mean_click_frac = round(sum(try2_cfs) / len(try2_cfs), 4) if try2_cfs else None

    print("=" * 80)
    print(f"STEP {STEP} — RESULT (HIER replication, 10 draws)")
    print()
    print(f"  RHAE per draw:  {[f'{r:.6f}' for r in rhae_per_draw]}")
    print(f"  Chain mean:     {chain_mean:.7f}")
    print(f"  Non-zero draws: {nonzero_draws}/{N_DRAWS}")
    print(f"  Non-zero games: {nonzero_games}/{total_games}")
    print(f"  eff² max={max(all_eff_sq):.6f}  mean={sum(all_eff_sq)/len(all_eff_sq):.7f}")
    print(f"  Game B (click) progress: {gameb_progress}/{N_DRAWS*2} episodes")
    print()
    print(f"  Type entropy (try2, mean across ARC episodes):")
    print(f"    At {ENT_CHECKPOINTS}: {[ent_traj_mean[ck] for ck in ENT_CHECKPOINTS]}")
    print(f"    Max possible: {MAX_TYPE_ENTROPY}")
    ent_vals = [ent_traj_mean[ck] for ck in ENT_CHECKPOINTS if ent_traj_mean[ck] is not None]
    if len(ent_vals) >= 2:
        ent_drop = round(ent_vals[0] - ent_vals[-1], 4)
        print(f"    Drop H100→H2000: {ent_drop} {'(LEARNING SIGNAL)' if ent_drop > 0.05 else '(flat)'}")
    print()
    print(f"  Click fraction: try2 mean={mean_click_frac}  try1→try2 shift mean={mean_cf_shift}")
    print()
    print("  Games with progress:")
    for r in all_results:
        if r['eff_sq'] > 0:
            hier_str = f"click={r['try2_click_frac']}" if r['is_hier'] else "flat"
            print(f"    Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}  {hier_str}")
    print()

    # Comparison to 1349 baseline (10 draws, seeds 13490-13499)
    flat_rhae_1349 = [0.0, 0.0, 0.000243, 0.0, 0.000009, 0.000207, 0.0, 0.0, 0.0, 0.0]
    flat_mean_1349 = sum(flat_rhae_1349) / len(flat_rhae_1349)
    flat_nz_1349   = sum(1 for r in flat_rhae_1349 if r > 0)
    print(f"  FLAT baseline (1349, seeds 13490-13499):")
    print(f"    Chain mean: {flat_mean_1349:.7f}  Non-zero: {flat_nz_1349}/{N_DRAWS}")
    print()

    print("ASSESSMENT:")
    if nonzero_draws <= 2:
        print(f"  >>> KILL: {nonzero_draws}/{N_DRAWS} non-zero ≤ 2. 1351 was lucky.")
    elif nonzero_draws >= 3:
        ratio = chain_mean / flat_mean_1349 if flat_mean_1349 > 0 else float('inf')
        print(f"  >>> SIGNAL: {nonzero_draws}/{N_DRAWS} non-zero ≥ 3. HIER replicates.")
        print(f"  >>> HIER chain_mean {chain_mean:.7f} vs FLAT {flat_mean_1349:.7f} = {ratio:.1f}× improvement.")
    if gameb_progress > 0:
        print(f"  >>> LANDMARK: Game B (click) progress in {gameb_progress} episodes.")
    ent_drop_val = (ent_vals[0] - ent_vals[-1]) if len(ent_vals) >= 2 else 0
    if ent_drop_val > 0.05:
        print(f"  >>> LANDMARK: type_entropy drops {ent_drop_val:.4f} — TP shifts type distribution.")
    else:
        print(f"  >>> type_entropy flat (drop={ent_drop_val:.4f}) — no learned type selection.")
    print("=" * 80)

    summary = {
        'step': STEP,
        'n_draws': N_DRAWS,
        'draw_seeds': DRAW_SEEDS,
        'rhae_per_draw': rhae_per_draw,
        'chain_mean_rhae': chain_mean,
        'nonzero_draws': nonzero_draws,
        'nonzero_games': nonzero_games,
        'total_games': total_games,
        'gameb_progress': gameb_progress,
        'mean_click_frac': mean_click_frac,
        'mean_cf_shift': mean_cf_shift,
        'type_entropy_traj_mean': {str(k): v for k, v in ent_traj_mean.items()},
        'max_type_entropy': MAX_TYPE_ENTROPY,
        'flat_baseline_1349_mean': flat_mean_1349,
        'flat_baseline_1349_nz': flat_nz_1349,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        json.dump({'step': STEP, 'results': all_results}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
