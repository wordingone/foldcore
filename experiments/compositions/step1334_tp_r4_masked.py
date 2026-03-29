"""
Step 1334 — Internal R4: substrate detects its own overfitting (catalog #37)
Leo mail 3785, 2026-03-29.

Substrate holds out first 200 steps of observations as a validation set.
Every 100 steps, computes validation prediction loss on this fixed set.
If val_loss > prev_val_loss × 1.1 → overfitting detected → halve all LRs.

This IS "criteria it generates" (R4): the evaluation signal is self-computed
from the substrate's own past observations, not from external reward.

Conditions:
  R4-TP: TP + internal overfitting detection + LR reduction
  TP:    TP without detection (1329/1333 baseline)

Kill criteria:
  R4-TP speedup ≤ TP speedup → detection doesn't help → KILL
  overfitting_count = 0 → detection never triggers → calibration issue

Level masking (Jun directive, 2026-03-29): steps_to_first_progress (opaque).
"""
import sys, os, time, json, logging, hashlib

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from prism_masked import (select_games, seal_mapping, label_filename,
                           masked_game_list, masked_run_log,
                           compute_progress_speedup, format_speedup,
                           write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1334)

STEP        = 1334
MAX_STEPS   = 2_000
MAX_SECONDS = 120
SEED_A      = 0
SEED_B      = 1

CONDITIONS  = ['r4_tp', 'tp']
LABELS      = {'r4_tp': 'R4-TP', 'tp': 'TP'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1334')
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

_DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BUFFER_MAX      = 200_000
_TRAIN_FREQ      = 5
_BATCH_SIZE      = 64
_LR              = 0.0001
_TP_LR           = 0.01
_SG_OUT_DIM      = 4101
N_KEYBOARD       = 7

LOSS_CHECKPOINTS  = [500, 1000, 2000]
_WARMUP_STEPS     = 200   # random-action warmup: populates validation buffer
_VAL_CHECK_FREQ   = 100   # validate every N steps after warmup
_OVERFIT_THRESH   = 1.1   # val_loss > prev × 1.1 → overfitting
_LR_REDUCTION     = 0.5   # halve all LRs on overfitting detection


# ---------------------------------------------------------------------------
# CNN model (same as steps 1323–1333)
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,   kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool   = nn.MaxPool2d(4, 4)
        self.action_fc     = nn.Linear(256 * 16 * 16, 512)
        self.action_head   = nn.Linear(512, 5)
        self.coord_conv1   = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2   = nn.Conv2d(128, 64,  kernel_size=3, padding=1)
        self.coord_conv3   = nn.Conv2d(64, 32,   kernel_size=1)
        self.coord_conv4   = nn.Conv2d(32, 1,    kernel_size=1)
        self.dropout       = nn.Dropout(0.2)
        self.pred_head     = nn.Linear(256 + 6, 256)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        avg_features = h4.mean([2, 3])
        af = self.action_pool(h4)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)
        cf = F.relu(self.coord_conv1(h4))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        coord_logits = cf.view(cf.size(0), -1)
        return torch.cat([action_logits, coord_logits], dim=1), avg_features

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
# TP substrate with optional internal R4 (overfitting detection)
# ---------------------------------------------------------------------------

class TpR4Substrate:
    """TP substrate with optional internal overfitting detection.

    r4_enabled=True  → R4-TP condition:
        Warmup (first _WARMUP_STEPS steps): random actions, populate validation buffer.
        Post-warmup: TP action selection + every _VAL_CHECK_FREQ steps compute
        validation loss. If val_loss > prev × _OVERFIT_THRESH: halve all LRs.

    r4_enabled=False → TP baseline (same as 1329/1333).

    Validation buffer: (state, action_type_vec, next_state) from warmup steps.
    These are transitions the substrate collected during random exploration —
    its own past observations. Validation loss = prediction error on these.
    Persists across try1→try2 (fixed reference set).
    """

    def __init__(self, n_actions, r4_enabled=False):
        self.n_actions    = n_actions
        self._r4_enabled  = r4_enabled
        self._rng         = np.random.RandomState(42)
        self._use_random  = not (n_actions > 128)

        self._model = SgSelfSupModel().to(_DEVICE)

        self._g = nn.ModuleDict({
            'g4': nn.Linear(256, 128).to(_DEVICE),
            'g3': nn.Linear(128, 64).to(_DEVICE),
            'g2': nn.Linear(64, 32).to(_DEVICE),
            'g1': nn.Linear(32, 16).to(_DEVICE),
        })

        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g    = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                         for k, v in self._g.items()}
        self._opt_f    = {
            'f1': torch.optim.Adam(self._model.conv1.parameters(), lr=_LR),
            'f2': torch.optim.Adam(self._model.conv2.parameters(), lr=_LR),
            'f3': torch.optim.Adam(self._model.conv3.parameters(), lr=_LR),
            'f4': torch.optim.Adam(self._model.conv4.parameters(), lr=_LR),
        }

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_one_hot  = None
        self._prev_sg_idx   = None
        self._train_counter = 0
        self._step          = 0

        self._episode_start_state_dict = {k: v.clone().cpu()
                                          for k, v in self._model.state_dict().items()}
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

        # R4 state
        self._val_buffer       = []   # (state, action_type_vec, next_state) — fixed after warmup
        self._warmup_done      = False
        self._prev_val_loss    = None
        self._overfitting_count = 0
        self._val_log          = []   # list of (step, val_loss) for diagnostics

    def reset_for_try2(self):
        """Reset loss tracking and R4 counters. Model weights, buffer, and val_buffer persist."""
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at   = {ck: None for ck in LOSS_CHECKPOINTS}
        self._overfitting_count = 0
        self._prev_val_loss  = None
        self._val_log        = []
        # val_buffer intentionally preserved: fixed validation set from try1 warmup

    def get_overfitting_count(self):
        return self._overfitting_count

    def get_val_log(self):
        return list(self._val_log)

    def _all_optimizers(self):
        opts = [self._opt_pred]
        opts.extend(self._opt_g.values())
        opts.extend(self._opt_f.values())
        return opts

    def _reduce_all_lr(self):
        for opt in self._all_optimizers():
            for pg in opt.param_groups:
                pg['lr'] *= _LR_REDUCTION

    def _compute_val_loss(self):
        if not self._val_buffer:
            return None
        n = len(self._val_buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [self._val_buffer[i] for i in idx]
        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)
        with torch.no_grad():
            avg4, _, _, _, _, _, _, _ = self._model.forward_all_layers(states)
            _, target_avg4 = self._model(next_states)
            pred_in   = torch.cat([avg4, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            val_loss  = float(F.mse_loss(pred_next, target_avg4).item())
        return val_loss

    def _maybe_check_overfitting(self):
        if not self._r4_enabled or not self._warmup_done or not self._val_buffer:
            return
        if self._step % _VAL_CHECK_FREQ != 0:
            return
        val_loss = self._compute_val_loss()
        if val_loss is None:
            return
        self._val_log.append((self._step, round(val_loss, 6)))
        if self._prev_val_loss is not None and val_loss > self._prev_val_loss * _OVERFIT_THRESH:
            self._reduce_all_lr()
            self._overfitting_count += 1
        self._prev_val_loss = val_loss

    def _is_in_warmup(self):
        return self._r4_enabled and not self._warmup_done and not self._use_random

    def process(self, obs_arr):
        self._step += 1
        obs_arr = np.asarray(obs_arr, dtype=np.float32)

        if self._use_random or not _is_arc_obs(obs_arr):
            self._prev_one_hot = None
            return int(self._rng.randint(self.n_actions))

        one_hot = _obs_to_one_hot(obs_arr)
        self._prev_one_hot = one_hot

        # R4-TP warmup: random actions until _WARMUP_STEPS
        if self._is_in_warmup():
            if self._step >= _WARMUP_STEPS:
                self._warmup_done = True
            return int(self._rng.randint(self.n_actions))

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
        return self._sg_to_prism(sg_idx)

    def update_after_step(self, obs_next, action, reward):
        if self._use_random or self._prev_one_hot is None:
            return
        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return

        one_hot_next = _obs_to_one_hot(obs_next_arr)
        sg_idx = self._prev_sg_idx if self._prev_sg_idx is not None else (action % 5)
        at_vec = _sg_to_action_type_vec(sg_idx)
        transition = {'state':           self._prev_one_hot.copy(),
                      'action_type_vec': at_vec,
                      'next_state':      one_hot_next.copy()}

        # During R4-TP warmup: store to validation buffer (first _WARMUP_STEPS transitions)
        if self._r4_enabled and not self._warmup_done:
            self._val_buffer.append(transition)

        h = hashlib.md5(self._prev_one_hot.tobytes() +
                        np.array([sg_idx], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append(transition)

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

        # R4: check for overfitting after each training step
        self._maybe_check_overfitting()

    def on_level_transition(self):
        self._prev_one_hot = None
        self._prev_sg_idx  = None

    def _tp_train_step(self):
        n   = len(self._buffer)
        buf = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf[i] for i in idx]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

        with torch.no_grad():
            avg4, avg3, avg2, avg1, h1, h2, h3, h4 = self._model.forward_all_layers(states)
            _, target_avg4 = self._model(next_states)
            pred_in   = torch.cat([avg4, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_avg4
            pred_loss = float((pred_err ** 2).mean())

            target_avg4_tp = avg4 - _TP_LR * pred_err
            target_avg3    = self._g['g4'](target_avg4_tp)
            target_avg2    = self._g['g3'](target_avg3)
            target_avg1    = self._g['g2'](target_avg2)

        with torch.enable_grad():
            pred_in2 = torch.cat([avg4.detach(), action_vecs.detach()], dim=1)
            pred_n2  = self._model.pred_head(pred_in2)
            loss_pred = F.mse_loss(pred_n2, target_avg4.detach())
            self._opt_pred.zero_grad()
            loss_pred.backward()
            self._opt_pred.step()

        for g_key, h_in, h_out_target in [('g4', avg4, avg3), ('g3', avg3, avg2), ('g2', avg2, avg1)]:
            with torch.enable_grad():
                h_recon    = self._g[g_key](h_in.detach())
                loss_recon = F.mse_loss(h_recon, h_out_target.detach())
                self._opt_g[g_key].zero_grad()
                loss_recon.backward()
                self._opt_g[g_key].step()

        for f_key, conv, x_in, target in [
            ('f4', self._model.conv4, h3, target_avg4_tp),
            ('f3', self._model.conv3, h2, target_avg3),
            ('f2', self._model.conv2, h1, target_avg2),
            ('f1', self._model.conv1, states, target_avg1),
        ]:
            with torch.enable_grad():
                h_fresh     = F.relu(conv(x_in.detach()))
                h_fresh_avg = h_fresh.mean([2, 3])
                loss_local  = F.mse_loss(h_fresh_avg, target.detach())
                self._opt_f[f_key].zero_grad()
                loss_local.backward()
                self._opt_f[f_key].step()

        return pred_loss

    def _sg_to_prism(self, sg_idx):
        if sg_idx < 5:
            return sg_idx
        action = N_KEYBOARD + (sg_idx - 5)
        return action if action < self.n_actions else sg_idx % 5

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._episode_start_state_dict:
                drift += (param.data.cpu() - self._episode_start_state_dict[name]).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
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
# Episode runner (level-masked)
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed):
    obs           = env.reset(seed=seed)
    action_log    = []
    action_counts = np.zeros(n_actions, np.float32)
    i3_counts_at_200 = None
    steps         = 0
    level         = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start       = time.time()
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
        action  = substrate.process(obs_arr) % n_actions
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

    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    return steps_to_first_progress, progress_count, {
        'steps_taken':              steps,
        'elapsed_seconds':          round(elapsed, 2),
        'steps_to_first_progress':  steps_to_first_progress,
        'progress_count':           progress_count,
        'RHAE':                     0.0,
        'I3_cv':                    i3_cv,
        'action_kl':                compute_action_kl(action_log, n_actions),
        'wdrift':                   round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':           substrate.get_pred_loss_trajectory(),
        'compression_ratio':        substrate.get_compression_ratio(),
        'overfitting_count':        substrate.get_overfitting_count(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, condition):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    r4 = (condition == 'r4_tp')
    substrate = TpR4Substrate(n_actions=n_actions, r4_enabled=r4)

    p1, _, result_try1 = run_episode(env, substrate, n_actions, seed=SEED_A)
    substrate.reset_for_try2()   # weights + buffer + val_buffer persist; counters reset

    p2, _, result_try2 = run_episode(env, substrate, n_actions, seed=SEED_B)

    speedup = compute_progress_speedup(p1, p2)
    total_overfit = result_try1['overfitting_count'] + result_try2['overfitting_count']

    return {
        'game':                    game_name,
        'condition':               condition,
        'try1':                    result_try1,
        'try2':                    result_try2,
        'second_exposure_speedup': speedup,
        'overfitting_count_total': total_overfit,
        'val_log':                 substrate.get_val_log(),
        'compression_ratio':       result_try1['compression_ratio'],
        'n_actions':               n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Internal R4: substrate detects its own overfitting (catalog #37)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"R4-TP: warmup={_WARMUP_STEPS} steps (random) → val check every {_VAL_CHECK_FREQ} steps")
    print(f"       overfitting threshold={_OVERFIT_THRESH}x → halve all LRs")
    print(f"TP: same as 1329/1333 baseline (no detection)")
    print(f"Kill: R4-TP speedup ≤ TP speedup  OR  overfitting_count=0 → KILL")
    print(f"Signal: R4-TP speedup > TP speedup → SIGNAL")
    print(f"6 game×condition pairs × 2 tries = 12 episodes ({MAX_STEPS} steps each).")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Tier 1: timing check
    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        label_t = GAME_LABELS.get(first_arc, first_arc)
        print(f"Tier 1: timing check on {label_t} (100 steps, R4-TP)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = TpR4Substrate(n_actions=na_t, r4_enabled=True)
        obs_t = env_t.reset(seed=0)
        t0 = time.time()
        for _ in range(100):
            if obs_t is None:
                obs_t = env_t.reset(seed=0)
                continue
            a = sub_t.process(np.asarray(obs_t, np.float32)) % na_t
            obs_next_t, _, done_t, _ = env_t.step(a)
            if obs_next_t is not None:
                sub_t.update_after_step(obs_next_t, a, 0)
            obs_t = obs_next_t if not done_t else env_t.reset(seed=0)
        t_100 = time.time() - t0
        est_2k    = t_100 * 20
        est_total = est_2k * 2 * 2 * 2  # 2 ARC × 2 conditions × 2 tries
        print(f"  100 steps: {t_100:.1f}s → est 2K: {est_2k:.0f}s → est total: {est_total:.0f}s")
        if est_total > 280:
            print(f"  WARNING: estimated {est_total:.0f}s > 5 min cap. Proceeding.")
        print()
        del sub_t, env_t

    all_results          = []
    speedup_by_condition = {c: [] for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")

        for condition in CONDITIONS:
            t0     = time.time()
            result = run_game(game, condition)
            all_results.append(result)
            speedup_by_condition[condition].append(result['second_exposure_speedup'])
            elapsed = time.time() - t0

            sp    = result['second_exposure_speedup']
            oc    = result['overfitting_count_total']
            cr    = result['compression_ratio']
            print(masked_run_log(f"{label}/{LABELS[condition]}", elapsed))
            print(f"  speedup={format_speedup(sp)}  overfitting_count={oc}  cr={cr}")

        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        game_results = [r for r in all_results if GAME_LABELS.get(r['game']) == label]
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r, default=str) + '\n')

    def chain_agg(vals):
        finite = [v for v in vals if v is not None and v != float('inf') and v > 0]
        if finite:
            return round(float(np.mean(finite)), 4)
        if any(v == float('inf') for v in vals if v is not None):
            return float('inf')
        return None

    final_speedup = {c: chain_agg(speedup_by_condition[c]) for c in CONDITIONS}
    write_experiment_results(RESULTS_DIR, STEP, final_speedup, all_results, CONDITIONS,
                             game_labels=GAME_LABELS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT (Internal R4 overfitting detection)\n")

    for cond in CONDITIONS:
        sp      = final_speedup[cond]
        cr_vals = [r['compression_ratio'] for r in all_results
                   if r['condition'] == cond and r['compression_ratio'] is not None]
        oc_vals = [r['overfitting_count_total'] for r in all_results if r['condition'] == cond]
        cr_str  = f"cr_mean={np.mean(cr_vals):.4f}" if cr_vals else "cr=N/A"
        oc_str  = f"overfit={sum(oc_vals)}"
        print(f"  {LABELS[cond]}: speedup={format_speedup(sp)}  {cr_str}  {oc_str}")
    print()

    print("Per-game detail:")
    for r in all_results:
        label = GAME_LABELS.get(r['game'], '?')
        sp    = r['second_exposure_speedup']
        oc    = r['overfitting_count_total']
        cr    = r['compression_ratio']
        print(f"  {label}/{LABELS[r['condition']]}: speedup={format_speedup(sp)} "
              f"cr={cr} overfitting_count={oc}")
    print()

    # Kill / Signal
    r4_sp = final_speedup.get('r4_tp')
    tp_sp = final_speedup.get('tp')
    r4_oc = sum(r['overfitting_count_total'] for r in all_results if r['condition'] == 'r4_tp')

    print("KILL/SIGNAL ASSESSMENT:")
    if r4_oc == 0:
        print(f"  >>> KILL: overfitting_count=0. Detection never triggered. "
              f"Threshold={_OVERFIT_THRESH}x too high OR game produces no overfitting.")
    elif r4_sp is None and tp_sp is None:
        print(f"  >>> INCONCLUSIVE: neither condition produced speedup. "
              f"overfitting_count={r4_oc} (detection IS triggering).")
    elif r4_sp is not None and tp_sp is not None and r4_sp <= tp_sp:
        print(f"  >>> KILL: R4-TP speedup={format_speedup(r4_sp)} ≤ TP speedup={format_speedup(tp_sp)}. "
              f"overfitting detection doesn't improve generalization.")
    elif r4_sp is not None and (tp_sp is None or r4_sp > tp_sp):
        print(f"  >>> SIGNAL: R4-TP speedup={format_speedup(r4_sp)} > TP speedup={format_speedup(tp_sp)}. "
              f"overfitting detection improves generalization.")
    else:
        print(f"  >>> INCONCLUSIVE: R4-TP={format_speedup(r4_sp)} TP={format_speedup(tp_sp)} "
              f"overfit_count={r4_oc}.")
    print(sep)


if __name__ == '__main__':
    os.chdir('B:/M/the-search/experiments/compositions')
    main()
