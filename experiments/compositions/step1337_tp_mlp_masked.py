"""
Step 1337 — MLP-TP: Drop CNN, modality-agnostic encoder.
Leo mail 3794, 2026-03-29.

CNN is a frozen frame: 30+ experiments, zero MBPP progress, no transfer.
Hypothesis: MLP removes spatial inductive bias that blocks text modality.
If MLP can process MBPP (256-dim text obs), RHAE gets a non-zero term.

Conditions:
  MLP-TP:  MLP encoder (3×512 FC, lazy init from obs_dim). Handles ARC + MBPP.
  CNN-TP:  Same CNN + TP as steps 1334-1336 (control baseline).

Primary metric: RHAE(try2) = mean(efficiency²) across all 3 games.
- MBPP: optimal_steps = len(correct_solution) via mbpp_game.compute_solver_steps
- ARC:  optimal_steps = None (no solver endpoint → efficiency = 0 until baseline builds)

Kill:     MLP RHAE = 0 AND MBPP compression still 0 → KILL MLP direction
Continue: MLP RHAE = 0 but MBPP cr < 0.9 → encoding works, need more steps
Signal:   MLP RHAE > 0 on ANY game
Landmark: MLP MBPP efficiency > 0 → FIRST TEXT PROGRESS EVER

One draw. Level-masked (steps_to_first_progress). Seed-free substrate init.
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
    masked_game_list, masked_run_log,
    compute_progress_speedup, format_speedup,
    speedup_for_chain, compute_rhae_try2, write_experiment_results,
)

GAMES, GAME_LABELS = select_games(seed=1337)

STEP        = 1337
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280   # 4.7 min per game (5-min cap)

CONDITIONS  = ['mlp_tp', 'cnn_tp']
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1337')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]

N_KEYBOARD  = 7  # action types for SG decomposition in CNN-TP


# ---------------------------------------------------------------------------
# Obs encoding (shared)
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
    """Encode any obs to 1D float32 for MLP input.
    ARC: one-hot (65536). Non-ARC (MBPP): raw flatten (256).
    """
    arr = np.asarray(obs_arr, dtype=np.float32)
    if _is_arc_obs(arr):
        return _obs_to_one_hot(arr).astype(np.float32).flatten()  # 65536
    return arr.flatten()  # MBPP: 256


def _action_type_vec(action, n_types=6):
    """Coarse action-type vector (6-dim one-hot of action % 6)."""
    vec = np.zeros(6, dtype=np.float32)
    vec[int(action) % n_types] = 1.0
    return vec


# ---------------------------------------------------------------------------
# MLP model (lazy init: created on first obs)
# ---------------------------------------------------------------------------

class MlpSelfSupModel(nn.Module):
    """4-layer MLP: input_proj → fc1 → fc2 → fc3 (all ReLU).
    Same TP structure as CNN-4: 4 forward layers, g3/g2/g1 inverse mappings.
    input_proj handles modality-specific input dimension.
    """
    def __init__(self, input_dim, n_actions, hidden=_HIDDEN):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden)
        self.fc1        = nn.Linear(hidden, hidden)
        self.fc2        = nn.Linear(hidden, hidden)
        self.fc3        = nn.Linear(hidden, hidden)
        self.action_head = nn.Linear(hidden, n_actions)
        self.pred_head   = nn.Linear(hidden + 6, hidden)
        self.dropout     = nn.Dropout(0.2)

    def forward(self, x):  # x: (B, input_dim)
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        logits = self.action_head(h3)
        return logits, h3

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0  # deepest → shallowest


# ---------------------------------------------------------------------------
# MLP-TP substrate (modality-agnostic, lazy init)
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    """MLP + Target Propagation. Handles ARC (65536-dim) and MBPP (256-dim).

    Model lazily initialized on first process() call based on obs shape.
    No modality-gating: MLP trains on ALL observations.
    """
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._model      = None   # created on first process()
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None
        self._input_dim  = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._prev_action   = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = None
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._input_dim = input_dim
        self._model = MlpSelfSupModel(input_dim, self.n_actions).to(_DEVICE)
        self._g = nn.ModuleDict({
            'g3': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g2': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
            'g1': nn.Linear(_HIDDEN, _HIDDEN).to(_DEVICE),
        })
        self._opt_pred = torch.optim.Adam(self._model.pred_head.parameters(), lr=_LR)
        self._opt_g = {k: torch.optim.Adam(v.parameters(), lr=_LR)
                       for k, v in self._g.items()}
        self._opt_f = {
            'f_proj': torch.optim.Adam(self._model.input_proj.parameters(), lr=_LR),
            'f1':     torch.optim.Adam(self._model.fc1.parameters(), lr=_LR),
            'f2':     torch.optim.Adam(self._model.fc2.parameters(), lr=_LR),
            'f3':     torch.optim.Adam(self._model.fc3.parameters(), lr=_LR),
        }
        self._ep_start_sd = {k: v.clone().cpu()
                             for k, v in self._model.state_dict().items()}

    def reset_for_try2(self):
        """Keep weights + buffer. Reset step counter and loss tracking."""
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        enc = _encode_obs_mlp(obs_arr)

        if self._model is None:
            self._init_model(enc.shape[0])

        self._prev_enc = enc

        tensor = torch.from_numpy(enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
            logits = logits.squeeze(0).cpu()

        probs = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        self._prev_action = action
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
            self._buffer.append({
                'state':           self._prev_enc.copy(),
                'action_type_vec': at_vec,
                'next_state':      enc_next.copy(),
            })
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._tp_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc    = None
        self._prev_action = None

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
            h3, h2, h1, h0 = self._model.forward_all_layers(states)
            _, target_h3_next = self._model(next_states)
            pred_in   = torch.cat([h3, action_vecs], dim=1)
            pred_next = self._model.pred_head(pred_in)
            pred_err  = pred_next - target_h3_next
            pred_loss = float((pred_err ** 2).mean())

            # TP: correct h3 target, propagate back through g functions
            target_h3 = h3 - _TP_LR * pred_err
            target_h2 = self._g['g3'](target_h3)
            target_h1 = self._g['g2'](target_h2)
            target_h0 = self._g['g1'](target_h1)

        # Update pred_head
        with torch.enable_grad():
            pred_in2 = torch.cat([h3.detach(), action_vecs.detach()], dim=1)
            pred_n2  = self._model.pred_head(pred_in2)
            loss_pred = F.mse_loss(pred_n2, target_h3_next.detach())
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()

        # Update g functions (approximate inverses)
        for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
            with torch.enable_grad():
                h_recon = self._g[gk](h_in.detach())
                loss_g  = F.mse_loss(h_recon, h_out.detach())
                self._opt_g[gk].zero_grad(); loss_g.backward(); self._opt_g[gk].step()

        # Local layer updates toward TP targets
        for fk, layer, x_in, target in [
            ('f3',     self._model.fc3,        h2,     target_h3),
            ('f2',     self._model.fc2,        h1,     target_h2),
            ('f1',     self._model.fc1,        h0,     target_h1),
            ('f_proj', self._model.input_proj, states, target_h0),
        ]:
            with torch.enable_grad():
                h_fresh = F.relu(layer(x_in.detach()))
                loss_local = F.mse_loss(h_fresh, target.detach())
                self._opt_f[fk].zero_grad(); loss_local.backward(); self._opt_f[fk].step()

        return pred_loss

    def compute_weight_drift(self):
        if self._model is None or self._ep_start_sd is None:
            return 0.0
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._ep_start_sd:
                drift += (param.data.cpu() - self._ep_start_sd[name]).norm().item()
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
# CNN model + TP substrate (control, unchanged from steps 1334-1336)
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,   kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128,  kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool   = nn.MaxPool2d(4, 4)
        self.action_fc     = nn.Linear(256 * 16 * 16, _HIDDEN)
        self.action_head   = nn.Linear(_HIDDEN, 5)
        self.coord_conv1   = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2   = nn.Conv2d(128, 64,  kernel_size=3, padding=1)
        self.coord_conv3   = nn.Conv2d(64,  32,  kernel_size=1)
        self.coord_conv4   = nn.Conv2d(32,  1,   kernel_size=1)
        self.dropout       = nn.Dropout(0.2)
        self.pred_head     = nn.Linear(256 + 6, 256)

    def forward(self, x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        avg_features = h4.mean([2, 3])
        af = self.action_pool(h4); af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af)); af = self.dropout(af)
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


def _sg_to_action_type_vec(sg_idx):
    vec = np.zeros(6, dtype=np.float32)
    vec[sg_idx if sg_idx < 5 else 5] = 1.0
    return vec


class TpSubstrate:
    """CNN-TP control (same as steps 1334-1336)."""
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._use_random = not (n_actions > 128)  # random for MBPP
        self._model      = SgSelfSupModel().to(_DEVICE)
        self._g = nn.ModuleDict({
            'g4': nn.Linear(256, 128).to(_DEVICE),
            'g3': nn.Linear(128, 64).to(_DEVICE),
            'g2': nn.Linear(64,  32).to(_DEVICE),
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
        self._ep_start_sd   = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def reset_for_try2(self):
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
        combined = combined / (total + 1e-12) if total > 1e-12 else torch.ones(4101) / 4101
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
        h = hashlib.md5(self._prev_one_hot.tobytes() +
                        np.array([sg_idx], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state':           self._prev_one_hot.copy(),
                                  'action_type_vec': at_vec,
                                  'next_state':      one_hot_next.copy()})
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
            self._opt_pred.zero_grad(); loss_pred.backward(); self._opt_pred.step()
        for g_key, h_in, h_out_target in [('g4', avg4, avg3), ('g3', avg3, avg2), ('g2', avg2, avg1)]:
            with torch.enable_grad():
                h_recon    = self._g[g_key](h_in.detach())
                loss_recon = F.mse_loss(h_recon, h_out_target.detach())
                self._opt_g[g_key].zero_grad(); loss_recon.backward(); self._opt_g[g_key].step()
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
                self._opt_f[f_key].zero_grad(); loss_local.backward(); self._opt_f[f_key].step()
        return pred_loss

    def _sg_to_prism(self, sg_idx):
        if sg_idx < 5:
            return sg_idx
        action = N_KEYBOARD + (sg_idx - 5)
        return action if action < self.n_actions else sg_idx % 5

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._ep_start_sd:
                drift += (param.data.cpu() - self._ep_start_sd[name]).norm().item()
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
    """Return optimal_steps for RHAE computation.
    MBPP: len(correct_solution) via compute_solver_steps.
    ARC: None (no solver endpoint — efficiency = 0).
    """
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)  # level 1 optimal steps
    return None  # ARC: unknown


# ---------------------------------------------------------------------------
# Episode runner (level-masked, same pattern as steps 1334-1336)
# ---------------------------------------------------------------------------

def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c  = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions: early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions: late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p  = (late_c  + 1e-8) / (late_c.sum()  + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


def run_episode(env, substrate, n_actions, seed, max_steps):
    obs            = env.reset(seed=seed)
    action_log     = []
    action_counts  = np.zeros(n_actions, np.float32)
    steps          = 0
    level          = 0
    progress_count = 0
    steps_to_first_progress = None
    t_start        = time.time()
    fresh_episode  = True
    i3_counts_at_200 = None

    while steps < max_steps:
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
        'steps_taken':             steps,
        'elapsed_seconds':         round(elapsed, 2),
        'steps_to_first_progress': steps_to_first_progress,
        'progress_count':          progress_count,
        'I3_cv':                   i3_cv,
        'action_kl':               compute_action_kl(action_log, n_actions),
        'wdrift':                  round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':          substrate.get_pred_loss_trajectory(),
        'compression_ratio':       substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, label, condition):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    if condition == 'mlp_tp':
        substrate = MlpTpSubstrate(n_actions=n_actions)
    else:
        substrate = TpSubstrate(n_actions=n_actions)

    p1, _, result_try1 = run_episode(env, substrate, n_actions,
                                     seed=0, max_steps=TRY1_STEPS)
    substrate.reset_for_try2()

    p2, _, result_try2 = run_episode(env, substrate, n_actions,
                                     seed=TRY2_SEED, max_steps=TRY2_STEPS)

    speedup = compute_progress_speedup(p1, p2)
    opt = get_optimal_steps(game_name, TRY2_SEED)
    efficiency_sq = 0.0
    if p2 is not None and opt is not None and opt > 0:
        eff = min(1.0, opt / p2)
        efficiency_sq = round(eff ** 2, 6)

    return {
        'label':                   label,
        'condition':               condition,
        'n_actions':               n_actions,
        'try1':                    result_try1,
        'try2':                    result_try2,
        'second_exposure_speedup': speedup,
        'efficiency_sq_try2':      efficiency_sq,
        'optimal_steps':           opt,
        'compression_ratio':       result_try1.get('compression_ratio'),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — MLP-TP vs CNN-TP: modality-agnostic encoder")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Try1: {TRY1_STEPS} steps (seed 0), Try2: {TRY2_STEPS} steps (seed {TRY2_SEED})")
    print(f"Primary metric: RHAE(try2) = mean(efficiency²) across all games")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Tier 1: timing check on first ARC game (MLP condition)
    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = GAME_LABELS[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, MLP-TP)...")
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
        est_2k = elapsed_100 / 100 * 2000
        est_total = est_2k * len(GAMES) * len(CONDITIONS) * 2
        print(f"  100 steps: {elapsed_100:.1f}s → est 2K: {est_2k:.0f}s → est total: {est_total:.0f}s")
        if est_total > 300:
            print(f"  WARNING: estimated {est_total:.0f}s > 300s cap. Consider reducing.")
        print()

    all_results = []
    rhae_by_condition  = {c: None for c in CONDITIONS}
    speedup_by_condition = {c: [] for c in CONDITIONS}
    eff_sq_by_game_cond = {c: {} for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"=== Condition: {condition.upper()} ===")
        try2_progress = {}
        optimal_steps = {}

        for game_name, label in zip(GAMES, GAME_LABELS.values()):
            t_game = time.time()
            result = run_game(game_name, label, condition)
            elapsed = time.time() - t_game

            all_results.append(result)
            p2 = result['try2']['steps_to_first_progress']
            speedup = result['second_exposure_speedup']
            cr  = result['compression_ratio']
            eff = result['efficiency_sq_try2']

            try2_progress[label] = p2
            optimal_steps[label] = result['optimal_steps']
            eff_sq_by_game_cond[condition][label] = eff

            print(f"  {label}: speedup={format_speedup(speedup)}  cr={cr}  eff²={eff}  ({elapsed:.1f}s)")
            speedup_by_condition[condition].append(speedup_for_chain(speedup))

            # Save per-game JSONL
            out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae = compute_rhae_try2(try2_progress, optimal_steps)
        rhae_by_condition[condition] = rhae
        chain_speedup = sum(speedup_by_condition[condition]) / len(speedup_by_condition[condition])
        print(f"  → RHAE(try2) = {rhae:.4f}  (chain_speedup={chain_speedup:.4f})")
        print()

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(f"STEP {STEP} — RESULT (MLP-TP vs CNN-TP: modality-agnostic encoder)")
    print()
    print(f"  MLP-TP  RHAE(try2) = {rhae_by_condition['mlp_tp']:.4f}")
    print(f"  CNN-TP  RHAE(try2) = {rhae_by_condition['cnn_tp']:.4f}")
    print()
    print("Per-game efficiency²:")
    for label in GAME_LABELS.values():
        mlp_eff = eff_sq_by_game_cond['mlp_tp'].get(label, 0.0)
        cnn_eff = eff_sq_by_game_cond['cnn_tp'].get(label, 0.0)
        print(f"  {label}: MLP={mlp_eff:.6f}  CNN={cnn_eff:.6f}")
    print()

    # Compression diagnostic (MBPP specifically)
    print("Compression ratio (pred_loss_traj — proxy for encoding):")
    for result in all_results:
        if result['condition'] == 'mlp_tp':
            lbl = result['label']
            cr = result['compression_ratio']
            traj = result['try1'].get('pred_loss_traj', {})
            print(f"  {lbl}/MLP: cr={cr}  traj={traj}")

    print()
    # Kill/Signal assessment
    mlp_rhae = rhae_by_condition['mlp_tp']
    cnn_rhae = rhae_by_condition['cnn_tp']
    mlp_mbpp_cr  = next((r['compression_ratio'] for r in all_results
                         if r['condition'] == 'mlp_tp' and r['label'] == 'MBPP'), None)
    mlp_mbpp_eff = eff_sq_by_game_cond['mlp_tp'].get('MBPP', 0.0)

    print("KILL/SIGNAL ASSESSMENT:")
    if mlp_mbpp_eff > 0:
        print(f"  >>> LANDMARK: MLP MBPP efficiency² = {mlp_mbpp_eff:.6f} > 0 — FIRST TEXT PROGRESS.")
    elif mlp_rhae > 0:
        print(f"  >>> SIGNAL: MLP RHAE(try2) = {mlp_rhae:.4f} > 0 on some ARC game.")
    elif mlp_mbpp_cr is not None and mlp_mbpp_cr < 0.9:
        print(f"  >>> CONTINUE: MLP MBPP cr={mlp_mbpp_cr} < 0.9 — encoding works, need more steps.")
    else:
        print(f"  >>> KILL: MLP RHAE=0 AND MBPP cr={mlp_mbpp_cr} (no encoding signal). CNN was not the blocker.")
    print("=" * 80)

    # Write results files
    chain_speedups = {c: round(sum(speedup_by_condition[c]) / len(speedup_by_condition[c]), 4)
                      for c in CONDITIONS}
    write_experiment_results(
        RESULTS_DIR, STEP,
        rhae_by_condition=rhae_by_condition,
        all_results=all_results,
        conditions=CONDITIONS,
        game_labels=GAME_LABELS,
        speedup_by_condition=chain_speedups,
    )


if __name__ == '__main__':
    main()
