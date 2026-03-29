"""
Step 1330 — Target Propagation full 10-game validation (unkillable)
Leo mail 3768, 2026-03-29.

Validates step 1329's SIGNAL (TP cr=0.082) across all 10 ARC games + MBPP.
No kill criteria — this is validation of the strongest R2-compliant signal.

Architecture: same TP as step 1329.
  - Learned g_i inverses in avg-pooled feature space
  - Local Adam per layer (conv_i + g_i + pred_head)
  - 1K steps try1, 1K steps try2

DFA baseline established in step 1326 (cr=0.66). No DFA control needed.
CNN+Adam locked as reference: cr=0.003, speedup=10.5× (R2-violating).

Constitutional audit: same as step 1329 (R2 PASS — local backward only).
"""
import sys, os, time, json, logging, hashlib

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import masked_run_log, format_speedup, write_experiment_results

STEP        = 1330
MAX_STEPS   = 1_000
MAX_SECONDS = 120
SEED_A      = 0
SEED_B      = 1

# All 10 ARC games + MBPP (explicit pool, no selection bias)
ARC_GAMES   = ['cd82', 'cn04', 'ft09', 'lp85', 'ls20', 'sb26', 'sp80', 'tr87', 'tu93', 'vc33']
GAMES       = ['mbpp'] + ARC_GAMES
GAME_LABELS = {'mbpp': 'MBPP'}
for _i, _g in enumerate(ARC_GAMES, 1):
    GAME_LABELS[_g] = f'Game {_i:02d}'

CONDITIONS  = ['tp']
LABELS      = {'tp': 'TP'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1330')
PDIR        = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'all_actions'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'all_actions'),
    'lp85':  ('lp85_fullchain.json',  'all_actions'),
}
ACTION_OFFSET = {'ls20': -1, 'vc33': 7}

_DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BUFFER_MAX   = 200_000
_TRAIN_FREQ   = 5
_BATCH_SIZE   = 64
_LR           = 0.0001
_TP_LR        = 0.01
_SG_OUT_DIM   = 4101
N_KEYBOARD    = 7

LOSS_CHECKPOINTS = [250, 500, 1000]


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,   kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool = nn.MaxPool2d(4, 4)
        self.action_fc   = nn.Linear(256 * 16 * 16, 512)
        self.action_head = nn.Linear(512, 5)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64,  kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32,   kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1,    kernel_size=1)
        self.dropout     = nn.Dropout(0.2)
        self.pred_head   = nn.Linear(256 + 6, 256)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        avg4 = h4.mean([2, 3])
        af = self.action_pool(h4)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        al = self.action_head(af)
        cf = F.relu(self.coord_conv1(h4))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        cf = self.coord_conv4(cf)
        cl = cf.view(cf.size(0), -1)
        return torch.cat([al, cl], dim=1), avg4

    def forward_all_layers(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        return h4.mean([2,3]), h3.mean([2,3]), h2.mean([2,3]), h1.mean([2,3]), h1, h2, h3, h4


def _obs_to_one_hot(obs_arr):
    frame = np.round(obs_arr).astype(np.int32).squeeze(0)
    frame = np.clip(frame, 0, 15)
    one_hot = np.zeros((16, 64, 64), dtype=np.bool_)
    for c in range(16):
        one_hot[c] = (frame == c)
    return one_hot


def _is_arc_obs(obs_arr):
    arr = np.asarray(obs_arr, dtype=np.float32)
    return arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] == 64 and arr.shape[2] == 64


def _sg_to_action_type_vec(sg_idx):
    vec = np.zeros(6, dtype=np.float32)
    vec[sg_idx if sg_idx < 5 else 5] = 1.0
    return vec


# ---------------------------------------------------------------------------
# TP substrate (same as step 1329)
# ---------------------------------------------------------------------------

class TpCnnSubstrate:
    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(42)
        self._use_random = not (n_actions > 128)
        self._model = SgSelfSupModel().to(_DEVICE)
        self._g = nn.ModuleDict({
            'g4': nn.Linear(256, 128).to(_DEVICE),
            'g3': nn.Linear(128, 64).to(_DEVICE),
            'g2': nn.Linear(64, 32).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g    = {k: torch.optim.Adam(v.parameters(), lr=_LR) for k, v in self._g.items()}
        self._opt_f    = {
            'f4': torch.optim.Adam(self._model.conv4.parameters(), lr=_LR),
            'f3': torch.optim.Adam(self._model.conv3.parameters(), lr=_LR),
            'f2': torch.optim.Adam(self._model.conv2.parameters(), lr=_LR),
            'f1': torch.optim.Adam(self._model.conv1.parameters(), lr=_LR),
        }
        self._buffer = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._train_counter = 0
        self._step = 0
        self._episode_start_state_dict = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def reset_loss_tracking(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        if self._use_random or not _is_arc_obs(obs_arr):
            self._prev_one_hot = None
            return int(self._rng.randint(self.n_actions))
        one_hot = _obs_to_one_hot(obs_arr)
        self._prev_one_hot = one_hot
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
            logits = logits.squeeze(0).cpu()
        ap = torch.sigmoid(logits[:5])
        cp = torch.sigmoid(logits[5:]) / 4096.0
        combined = torch.cat([ap, cp])
        total = combined.sum().item()
        combined = combined / (total + 1e-12) if total > 1e-12 else torch.ones(_SG_OUT_DIM) / _SG_OUT_DIM
        sg_idx = int(torch.multinomial(combined, 1).item())
        self._prev_sg_idx = sg_idx
        action = sg_idx if sg_idx < 5 else N_KEYBOARD + (sg_idx - 5)
        return action if action < self.n_actions else sg_idx % 5

    def update_after_step(self, obs_next, action, reward):
        if self._use_random or self._prev_one_hot is None:
            return
        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return
        one_hot_next = _obs_to_one_hot(obs_next_arr)
        sg_idx = self._prev_sg_idx if self._prev_sg_idx is not None else (action % 5)
        at_vec = _sg_to_action_type_vec(sg_idx)
        h = hashlib.md5(self._prev_one_hot.tobytes() + np.array([sg_idx], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state': self._prev_one_hot.copy(), 'action_type_vec': at_vec,
                                  'next_state': one_hot_next.copy()})
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_one_hot = None
        self._prev_sg_idx = None

    def _tp_train_step(self):
        n = len(self._buffer)
        buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]
        states = torch.from_numpy(np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            avg4, avg3, avg2, avg1, h1, h2, h3, h4 = self._model.forward_all_layers(states)
            _, target_avg4 = self._model(next_states)
            pred_in   = torch.cat([avg4, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_avg4
            pred_loss = float((pred_err ** 2).mean())
            # Top-down targets
            t4 = avg4 - _TP_LR * pred_err
            t3 = self._g['g4'](t4)
            t2 = self._g['g3'](t3)
            t1 = self._g['g2'](t2)

        # pred_head update
        with torch.enable_grad():
            p_in2 = torch.cat([avg4.detach(), action_vecs.detach()], dim=1)
            pn2   = self._model.pred_head(p_in2)
            lp = F.mse_loss(pn2, target_avg4.detach())
            self._opt_pred.zero_grad(); lp.backward(); self._opt_pred.step()

        # g_i reconstruction updates
        for g_key, hi, hi_minus in [('g4', avg4, avg3), ('g3', avg3, avg2), ('g2', avg2, avg1)]:
            with torch.enable_grad():
                lr = F.mse_loss(self._g[g_key](hi.detach()), hi_minus.detach())
                self._opt_g[g_key].zero_grad(); lr.backward(); self._opt_g[g_key].step()

        # conv_i local updates
        for f_key, conv, x_in, target in [
            ('f4', self._model.conv4, h3, t4),
            ('f3', self._model.conv3, h2, t3),
            ('f2', self._model.conv2, h1, t2),
            ('f1', self._model.conv1, states, t1),
        ]:
            with torch.enable_grad():
                hf = F.relu(conv(x_in.detach()))
                lf = F.mse_loss(hf.mean([2, 3]), target.detach())
                self._opt_f[f_key].zero_grad(); lf.backward(); self._opt_f[f_key].step()

        return pred_loss

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._episode_start_state_dict:
                drift += (param.data.cpu() - self._episode_start_state_dict[name]).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(250)
        l_late  = self._pred_loss_at.get(1000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)


# ---------------------------------------------------------------------------
# Game factory and solvers
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


def load_prescription(game_name):
    if game_name.lower() not in SOLVER_PRESCRIPTIONS:
        return None
    fname, field = SOLVER_PRESCRIPTIONS[game_name.lower()]
    path = os.path.join(PDIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(field)


def compute_solver_level_steps(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=1)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            env.reset(seed=1)
            fresh_episode = True
    return level_first_step


def compute_arc_score(level_first_step, solver_level_steps):
    if not level_first_step or not solver_level_steps:
        return 0.0
    scores = []
    for lvl, s_step in solver_level_steps.items():
        a_step = level_first_step.get(lvl)
        if a_step is not None and s_step > 0:
            scores.append((s_step / a_step) ** 2)
    return round(float(np.mean(scores)), 6) if scores else 0.0


def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c  = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions:
            late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p  = (late_c  + 1e-8) / (late_c.sum()  + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
    action_counts = np.zeros(n_actions, np.float32)
    i3_counts_at_200 = None
    steps = 0
    level = 0
    max_level = 0
    level_first_step = {}
    t_start = time.time()
    fresh_episode = True

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue
        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

        obs_arr = np.asarray(obs, dtype=np.float32)
        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)
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
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
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
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    action_kl = compute_action_kl(action_log, n_actions)
    return {
        'steps_taken':      steps,
        'elapsed_seconds':  round(elapsed, 2),
        'max_level':        max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score':        round(arc_score, 6),
        'RHAE':             round(arc_score, 6),
        'I3_cv':            i3_cv,
        'action_kl':        action_kl,
        'wdrift':           round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':   substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = TpCnnSubstrate(n_actions=n_actions)
    r1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)
    substrate.reset_loss_tracking()
    r2 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_B)

    l1_1 = r1['level_first_step'].get(1)
    l1_2 = r2['level_first_step'].get(1)
    if l1_1 is not None and l1_2 is not None and l1_2 > 0:
        speedup = round(l1_1 / l1_2, 4)
    elif l1_1 is None and l1_2 is not None:
        speedup = float('inf')
    elif l1_1 is not None and l1_2 is None:
        speedup = 0.0
    else:
        speedup = None

    return {
        'game':                    game_name,
        'condition':               'tp',
        'try1':                    r1,
        'try2':                    r2,
        'second_exposure_speedup': speedup,
        'compression_ratio':       r1['compression_ratio'],
        'n_actions':               n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Target Propagation validation (all 10 ARC + MBPP)")
    print(f"Device: {_DEVICE}")
    print(f"Architecture: TP with learned g_i inverses + local Adam (same as 1329)")
    print(f"Reference: CNN+Adam cr=0.003 (R2-violating). TP 1329: cr=0.082 (R2-compliant).")
    print(f"11 games × 2 tries ({MAX_STEPS} steps each). Unkillable — validation run.")
    print()

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    all_results = []
    speedup_vals = []

    for game in GAMES:
        label = GAME_LABELS[game]
        t0 = time.time()
        result = run_game(game, solver_steps_cache[game])
        all_results.append(result)
        speedup_vals.append(result['second_exposure_speedup'])
        elapsed = time.time() - t0
        cr = result['compression_ratio']
        sp = result['second_exposure_speedup']
        sp_str = format_speedup(sp) if sp is not None else 'N/A'
        cr_str = f"cr={cr:.4f}" if cr is not None else "cr=N/A"
        print(f"  {label}: speedup={sp_str}  {cr_str}  ({elapsed:.1f}s)")

    # Chain aggregate
    def chain_agg(vals):
        finite = [v for v in vals if v is not None and v != float('inf') and v > 0]
        if finite:
            return round(float(np.mean(finite)), 4)
        if any(v == float('inf') for v in vals if v is not None):
            return float('inf')
        return None

    final_speedup = {'tp': chain_agg(speedup_vals)}
    write_experiment_results(RESULTS_DIR, STEP, final_speedup, all_results, CONDITIONS)

    # Save diagnostics
    diag_path = os.path.join(RESULTS_DIR, 'diagnostics.json')
    import json as _json
    with open(diag_path, 'w') as f:
        _json.dump({'step': STEP, 'results': all_results}, f, default=str)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT (Target Propagation validation)\n")

    cr_vals = [r['compression_ratio'] for r in all_results if r['compression_ratio'] is not None]
    cr_mean = float(np.mean(cr_vals)) if cr_vals else None
    sp_str  = format_speedup(final_speedup['tp'])
    cr_str  = f"{cr_mean:.4f}" if cr_mean is not None else "N/A"
    print(f"  TP: speedup={sp_str}  cr_mean={cr_str} (over {len(cr_vals)}/{len(GAMES)} games)")
    print()

    # L1 summary
    l1_games = [r for r in all_results if r['level_first_step'].get(1) or
                r['try1']['level_first_step'].get(1) or r['try2']['level_first_step'].get(1)]
    print(f"  Games with L1: {len(l1_games)}/{len(GAMES)}")
    for r in all_results:
        l1_1 = r['try1']['level_first_step'].get(1)
        l1_2 = r['try2']['level_first_step'].get(1)
        sp = r['second_exposure_speedup']
        cr = r['compression_ratio']
        if l1_1 or l1_2 or (cr is not None and cr < 0.5):
            lbl = GAME_LABELS[r['game']]
            print(f"    {lbl}: try1_L1={l1_1}  try2_L1={l1_2}  speedup={sp}  cr={cr}")
    print()

    # Compression summary vs references
    print(f"  Compression comparison:")
    print(f"    LPL baseline:      cr=0.93  (7% compression)")
    print(f"    DFA (step 1326):   cr=0.66  (34% compression)")
    print(f"    TP step 1329:      cr=0.082 (92% compression, 1 game)")
    if cr_mean is not None:
        print(f"    TP step 1330:      cr={cr_mean:.3f}  ({(1-cr_mean)*100:.0f}% compression, {len(cr_vals)} games)")
    print(f"    CNN+Adam ref:      cr=0.003 (99.7% compression, R2-violating)")
    print(sep)


if __name__ == '__main__':
    main()
