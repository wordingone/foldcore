"""
Step 1350 — Reflexive bridge: encoder output drives action vs pure uniform random.
Leo mail 3839/3841, 2026-03-29.

Architecture question clarified (mail 3842): 1344 already uses action_head(h3) for
selection. The meaningful comparison is:
  REFLEX: softmax(action_head(h3)) — encoder in action loop, entropy tracked
  ENT:    pure uniform random (np.random.randint) — encoder NOT in action loop

Both conditions use identical TP training. Only action selection differs.

Key diagnostic: track action entropy at steps [100, 500, 1000, 2000].
  - For REFLEX: entropy of softmax(action_head(h3)) at checkpoint steps
  - For ENT: entropy = log(n_actions) always (uniform), also computed from logits for comparison
  If REFLEX entropy decreases over time → h3's structure propagates to actions.
  If flat → the random linear map washes out h3's signal.

Protocol: 5 draws × 3 games × 2 conditions × 2 tries = 60 episodes
Seeds: 13490-13494 (subset of 1349, which showed 3/10 non-zero draws)
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

STEP        = 1350
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

N_DRAWS     = 5
DRAW_SEEDS  = [13490 + i for i in range(N_DRAWS)]

CONDITIONS  = [('reflex', True), ('ent', False)]  # (name, use_action_head)

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1350')

_DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN          = 512
_LR              = 1e-4
_TP_LR           = 0.01
_BUFFER_MAX      = 200_000
_TRAIN_FREQ      = 5
_BATCH_SIZE      = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]
ENT_CHECKPOINTS  = [100, 500, 1000, 2000]


# ---------------------------------------------------------------------------
# Obs encoding (same as 1344-1349)
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


def _entropy(probs_np):
    """Shannon entropy in nats. probs_np: 1-D numpy array."""
    p = probs_np + 1e-12
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# MLP model (same as 1344)
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
# Substrate with condition switch
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    """
    MLP + TP substrate.
    use_action_head=True  → REFLEX: softmax(action_head(h3)) for action selection
    use_action_head=False → ENT:    pure uniform random (np.random.randint)
    Both conditions track action entropy from the model's logits at checkpoints.
    """

    def __init__(self, n_actions, use_action_head=True, rng_seed=42):
        self.n_actions       = n_actions
        self.use_action_head = use_action_head
        self._rng            = np.random.RandomState(rng_seed)
        self._model          = None
        self._g              = None
        self._opt_pred       = None
        self._opt_g          = None
        self._opt_f          = None

        self._buffer         = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes  = set()
        self._prev_enc       = None
        self._train_counter  = 0
        self._step           = 0
        self._recent_losses  = deque(maxlen=50)
        self._pred_loss_at   = {ck: None for ck in LOSS_CHECKPOINTS}

        # Entropy tracking: accumulated per checkpoint window
        self._ent_accum   = {ck: [] for ck in ENT_CHECKPOINTS}
        self._ent_at      = {ck: None for ck in ENT_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpModel(input_dim, self.n_actions).to(_DEVICE)
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

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._ent_accum    = {ck: [] for ck in ENT_CHECKPOINTS}
        self._ent_at       = {ck: None for ck in ENT_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)
        if self._model is None:
            self._init_model(enc.shape[0])
        self._prev_enc = enc
        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
        probs_np = torch.softmax(logits.squeeze(0).cpu(), dim=-1).numpy()

        # Track entropy at checkpoints (accumulate steps before checkpoint)
        H = _entropy(probs_np)
        for ck in ENT_CHECKPOINTS:
            if self._ent_at[ck] is None:
                self._ent_accum[ck].append(H)
                if self._step >= ck:
                    self._ent_at[ck] = round(float(np.mean(self._ent_accum[ck])), 4)

        # Action selection
        if self.use_action_head:
            return int(torch.multinomial(torch.from_numpy(probs_np), 1).item()) % self.n_actions
        else:
            return int(self._rng.randint(0, self.n_actions))

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

    def get_entropy_at(self, ck):
        return self._ent_at.get(ck)

    def max_theoretical_entropy(self):
        return round(math.log(self.n_actions), 4)


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
    progress_count = 0
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
            progress_count += 1
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
    cond_results = {}

    for cond_name, use_ah in CONDITIONS:
        draw_results = []
        try2_progress = {}
        optimal_steps_d = {}

        for game_name, label in zip(games, game_labels.values()):
            env = make_game(game_name)
            try:
                n_actions = int(env.n_actions)
            except AttributeError:
                n_actions = 4103

            substrate = MlpTpSubstrate(n_actions=n_actions, use_action_head=use_ah)

            p1, _, t1 = run_episode(env, substrate, n_actions, seed=0, max_steps=TRY1_STEPS)
            substrate.reset_for_try2()
            p2, _, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=TRY2_STEPS)

            speedup = compute_progress_speedup(p1, p2)
            opt = get_optimal_steps(game_name, TRY2_SEED)
            eff_sq = 0.0
            if p2 is not None and opt is not None and opt > 0:
                eff = min(1.0, opt / p2)
                eff_sq = round(eff ** 2, 6)

            cr  = substrate.get_compression_ratio()
            max_ent = substrate.max_theoretical_entropy()
            ent_traj = {ck: substrate.get_entropy_at(ck) for ck in ENT_CHECKPOINTS}

            try2_progress[label]  = p2
            optimal_steps_d[label] = opt

            result = {
                'draw': draw_idx, 'label': label, 'game': game_name,
                'condition': cond_name,
                'p1': p1, 'p2': p2, 'speedup': speedup,
                'eff_sq': eff_sq, 'optimal_steps': opt, 'cr': cr,
                'max_entropy': max_ent,
                'entropy_at_100':  ent_traj[100],
                'entropy_at_500':  ent_traj[500],
                'entropy_at_1000': ent_traj[1000],
                'entropy_at_2000': ent_traj[2000],
                't1': t1, 't2': t2,
            }
            draw_results.append(result)

            ent_str = f"H=[{ent_traj[100]},{ent_traj[500]},{ent_traj[1000]},{ent_traj[2000]}] max={max_ent:.2f}"
            print(f"    [{cond_name}] {label}: eff²={eff_sq}  cr={cr}  {ent_str}  ({t1+t2:.1f}s)")

            out_path = os.path.join(draw_dir, label_filename(f"{label}_{cond_name}", STEP))
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae = compute_rhae_try2(try2_progress, optimal_steps_d)
        chain_sp = sum(speedup_for_chain(r['speedup']) for r in draw_results) / len(draw_results)
        print(f"    [{cond_name}] → RHAE(try2) = {rhae:.6f}  chain_speedup={chain_sp:.4f}")
        cond_results[cond_name] = {'results': draw_results, 'rhae': rhae}

    print()
    return cond_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — REFLEX vs ENT: encoder-driven vs pure-random actions.")
    print(f"Device: {_DEVICE}")
    print(f"Protocol: {N_DRAWS} draws × 3 games × 2 conditions × 2 tries = {N_DRAWS*3*2*2} episodes")
    print(f"Conditions: {[c[0] for c in CONDITIONS]}. Draw seeds: {DRAW_SEEDS}")
    print()

    # Tier 1 timing
    games0, labels0 = select_games(seed=DRAW_SEEDS[0])
    first_arc = next((g for g in games0 if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = labels0[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = MlpTpSubstrate(n_actions=na_t)
        t0 = time.time()
        obs_t = env_t.reset(seed=0)
        for _ in range(100):
            a = sub_t.process(np.asarray(obs_t, dtype=np.float32)) % na_t
            obs_t, _, done_t, _ = env_t.step(a)
            if done_t or obs_t is None:
                obs_t = env_t.reset(seed=0)
        elapsed_100 = time.time() - t0
        est_ep  = elapsed_100 / 100 * 2000
        est_tot = est_ep * N_DRAWS * 3 * 4  # 2 conds × 2 tries
        print(f"  100 steps: {elapsed_100:.1f}s → est per episode: {est_ep:.0f}s → est total: {est_tot:.0f}s")
        print()

    all_results = {c[0]: [] for c in CONDITIONS}
    rhae_per_draw = {c[0]: [] for c in CONDITIONS}

    for draw_idx in range(N_DRAWS):
        cond_results = run_draw(draw_idx, DRAW_SEEDS[draw_idx])
        for cond_name in [c[0] for c in CONDITIONS]:
            all_results[cond_name].extend(cond_results[cond_name]['results'])
            rhae_per_draw[cond_name].append(cond_results[cond_name]['rhae'])

    # -------------------------------------------------------------------------
    # Aggregate report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(f"STEP {STEP} — RESULT (REFLEX vs ENT: encoder-driven vs pure-random actions)")
    print()

    for cond_name in [c[0] for c in CONDITIONS]:
        rhaes  = rhae_per_draw[cond_name]
        cmean  = sum(rhaes) / len(rhaes)
        cvar   = sum((r - cmean)**2 for r in rhaes) / len(rhaes)
        cstd   = cvar ** 0.5
        nzdraw = sum(1 for r in rhaes if r > 0)
        nzgame = sum(1 for r in all_results[cond_name] if r['eff_sq'] > 0)
        print(f"  [{cond_name}]")
        print(f"    RHAE per draw: {[f'{r:.6f}' for r in rhaes]}")
        print(f"    Chain mean:    {cmean:.6f}  std: {cstd:.6f}")
        print(f"    Non-zero:      draws={nzdraw}/{N_DRAWS}  games={nzgame}/{len(all_results[cond_name])}")
        print()

    # Entropy comparison
    print("  Entropy trajectories (mean over draw×game, try2):")
    print(f"  {'Condition':<10} {'n_actions':<12} {'H@100':<10} {'H@500':<10} {'H@1000':<10} {'H@2000':<10} {'H_max':<10}")
    for cond_name in [c[0] for c in CONDITIONS]:
        results = all_results[cond_name]
        for na_label in ['ARC (Game A/B)', 'MBPP']:
            if na_label == 'MBPP':
                sub = [r for r in results if r['label'] == 'MBPP']
            else:
                sub = [r for r in results if r['label'] != 'MBPP']
            if not sub: continue
            def mean_ent(ck):
                vals = [r[f'entropy_at_{ck}'] for r in sub if r[f'entropy_at_{ck}'] is not None]
                return round(sum(vals)/len(vals), 3) if vals else None
            h_max = round(sum(r['max_entropy'] for r in sub) / len(sub), 3)
            print(f"  {cond_name:<10} {na_label:<12} {mean_ent(100):<10} {mean_ent(500):<10} {mean_ent(1000):<10} {mean_ent(2000):<10} {h_max:<10}")
    print()

    print("  Games with progress:")
    for cond_name in [c[0] for c in CONDITIONS]:
        for r in all_results[cond_name]:
            if r['eff_sq'] > 0:
                print(f"    [{cond_name}] Draw {r['draw']} {r['label']}: eff²={r['eff_sq']}  p2={r['p2']}")
    print()

    # Assessment
    reflex_rhaes = rhae_per_draw['reflex']
    ent_rhaes    = rhae_per_draw['ent']
    reflex_mean  = sum(reflex_rhaes) / len(reflex_rhaes)
    ent_mean     = sum(ent_rhaes)    / len(ent_rhaes)

    reflex_h2000 = [r['entropy_at_2000'] for r in all_results['reflex'] if r['entropy_at_2000'] is not None]
    ent_h2000    = [r['entropy_at_2000'] for r in all_results['ent']    if r['entropy_at_2000'] is not None]

    reflex_h100  = [r['entropy_at_100']  for r in all_results['reflex'] if r['entropy_at_100']  is not None]
    ent_h100     = [r['entropy_at_100']  for r in all_results['ent']    if r['entropy_at_100']  is not None]

    reflex_h_drop = (sum(reflex_h100)/len(reflex_h100) - sum(reflex_h2000)/len(reflex_h2000)) if reflex_h100 and reflex_h2000 else None
    ent_h_drop    = (sum(ent_h100)/len(ent_h100)       - sum(ent_h2000)/len(ent_h2000))       if ent_h100    and ent_h2000    else None

    print("ASSESSMENT:")
    if reflex_mean > ent_mean * 1.5:
        print(f"  >>> REFLEX WINS: RHAE {reflex_mean:.6f} > ENT {ent_mean:.6f}. Encoder-driven actions help.")
    elif ent_mean > reflex_mean * 1.5:
        print(f"  >>> ENT WINS: RHAE {ent_mean:.6f} > REFLEX {reflex_mean:.6f}. Pure random actions better.")
    else:
        print(f"  >>> SAME RHAE: REFLEX={reflex_mean:.6f} ≈ ENT={ent_mean:.6f}. Action selection doesn't change RHAE.")

    if reflex_h_drop is not None and ent_h_drop is not None:
        if reflex_h_drop > ent_h_drop + 0.05:
            print(f"  >>> REFLEX ENTROPY DROPS: H drop={reflex_h_drop:.3f} vs ENT={ent_h_drop:.3f}. Encoder shifts action distribution.")
        elif abs(reflex_h_drop - ent_h_drop) < 0.05:
            print(f"  >>> ENTROPY SAME: REFLEX drop={reflex_h_drop:.3f} ≈ ENT={ent_h_drop:.3f}. Random linear map washes h3 signal.")
        else:
            print(f"  >>> ENT ENTROPY DROPS MORE: ENT={ent_h_drop:.3f} vs REFLEX={reflex_h_drop:.3f}.")
    print("=" * 80)

    # Save
    summary = {
        'step': STEP,
        'n_draws': N_DRAWS,
        'draw_seeds': DRAW_SEEDS,
        'rhae_per_draw': {c: rhae_per_draw[c] for c in [cn for cn, _ in CONDITIONS]},
        'chain_mean': {c: sum(rhae_per_draw[c])/len(rhae_per_draw[c]) for c in [cn for cn, _ in CONDITIONS]},
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(RESULTS_DIR, 'diagnostics.json'), 'w') as f:
        all_r = [r for c in [cn for cn, _ in CONDITIONS] for r in all_results[c]]
        json.dump({'step': STEP, 'results': all_r}, f, indent=2, default=str)


if __name__ == '__main__':
    main()
