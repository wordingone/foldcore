"""
Step 1326 — CNN + Direct Feedback Alignment (DFA): R2-compliant credit assignment.
Leo mail 3757, 2026-03-29.

## Direction change

LPL exhausted (16 experiments). CNN+Adam achieves 10.5× speedup but violates R2 (backward
pass). DFA replaces Adam with forward-only credit assignment: fixed random feedback
matrices project top-level prediction error directly to each conv layer. No backward pass.

## Why DFA is R2-compliant

Adam: forward pass (processes input) → BACKWARD pass (separate mechanism drives change).
DFA: forward pass 1 (processes input) → project top error via fixed B matrices → local
     update. Update signal comes from forward-direction computation only. One mechanism.

Reference: Leo mail 3757. DFA alternative to FTP (arxiv 2506.11030) — simpler,
same R2 property tested.

## Architecture

Same 4-layer CNN as step 1323 (proven: Adam achieves cr=0.0028, speedup=10.5×).
Replace Adam with DFA:

    Forward pass (same):
        obs → conv1 → conv2 → conv3 → conv4 → avg_features → pred_head

    DFA update (replaces Adam backward pass):
        top_error = pred_next - target_features               # (batch, 256)
        pred_head update: standard MSE gradient (top layer — no feedback needed)
        conv layer i update via B_i (fixed random matrix):
            delta_i = top_error @ B_i.T                       # (batch, C_out_i)
            dW_conv_i ≈ outer(delta_i.mean, input_i.mean) / kernel_size
            conv_i.weight -= LR * dW_conv_i

    Feedback matrices B_i (frozen after init, deterministic seed):
        B_conv1: (32, 256), B_conv2: (64, 256), B_conv3: (128, 256), B_conv4: (256, 256)

    Action selection: entropy-driven (same as 1323). Action/coord heads FROZEN.
    (Only prediction pathway updated — tests compression, not action optimization.)

## Constitutional audit
| R0 | Deterministic init. B matrices initialized from fixed numpy seed (42). |
| R1 | PASS — prediction error is self-generated |
| R2 | PASS — no backward pass. DFA = forward computation only |
| R3 | All 4 conv layers + pred_head modified by local DFA rule |
| R4 | second_exposure_speedup |
| R5 | PASS |
| R6 | Remove B matrices → no learning (RAND condition) |

## Predictions (from Leo)
1. DFA compression between LPL (cr≈0.93) and Adam (cr≈0.003) — stronger than local Hebbian
2. If cr > 0.5: forward-only not enough; gap is about optimization quality
3. If cr < 0.3: DFA achieves meaningful compression; shared representation may produce speedup

## Protocol
- MBPP + 2 masked ARC, seed-free, 1 run per game
- 2K steps try1, 2K steps try2 (weights persist — try2 starts where try1 ended)
- Conditions: DFA vs RAND
- One metric: second_exposure_speedup
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
                           format_speedup, write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1326)

STEP = 1326
MAX_STEPS   = 2_000
MAX_SECONDS = 120     # per episode

CONDITIONS = ['dfa', 'rand']
LABELS     = {'dfa': 'DFA', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1326')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

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

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ    = 5
_BATCH_SIZE    = 64
_DFA_LR        = 0.001    # DFA uses approximate gradients — needs larger LR than Adam
_ACTION_ENTROPY_COEF = 0.0001
_SG_OUTPUT_DIM = 4101
N_KEYBOARD     = 7

LOSS_CHECKPOINTS = [500, 1000, 2000]
SEED_A = 0
SEED_B = 1


# ---------------------------------------------------------------------------
# CNN model (same architecture as step 1323)
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.action_pool = nn.MaxPool2d(4, 4)
        action_flattened_size = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)
        self.pred_head = nn.Linear(256 + 6, 256)

    def forward_with_activations(self, x):
        """Forward pass that returns intermediate activations (for DFA).

        Returns:
            combined: action/coord logits (for action selection)
            avg_features: (batch, 256) pooled features
            layer_inputs: list of [x0, x1, x2, x3] (inputs to each conv layer)
        """
        x0 = x
        x1 = F.relu(self.conv1(x0))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        avg_features = x4.mean(dim=[2, 3])

        af = self.action_pool(x4)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)

        cf = F.relu(self.coord_conv1(x4))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        cf = self.coord_conv4(cf)
        coord_logits = cf.view(cf.size(0), -1)

        combined = torch.cat([action_logits, coord_logits], dim=1)
        return combined, avg_features, [x0, x1, x2, x3]

    def forward(self, x):
        combined, avg_features, _ = self.forward_with_activations(x)
        return combined, avg_features

    def predict_next(self, avg_features, action_type_vec):
        inp = torch.cat([avg_features, action_type_vec], dim=1)
        return self.pred_head(inp)


def _sg_to_action_type_vec(sg_idx):
    vec = np.zeros(6, dtype=np.float32)
    if sg_idx < 5:
        vec[sg_idx] = 1.0
    else:
        vec[5] = 1.0
    return vec


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


# ---------------------------------------------------------------------------
# DFA substrate
# ---------------------------------------------------------------------------

class DfaCnnSubstrate:
    """CNN with Direct Feedback Alignment (DFA) replacing Adam.

    R2-COMPLIANT: no backward pass. Top prediction error projected to
    each conv layer via fixed random feedback matrices B_i.
    Action/coord heads are frozen — only prediction pathway learns.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(42)
        self._use_random = not (n_actions > 128)

        self._model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        # NO optimizer — DFA uses manual updates

        # Fixed random feedback matrices (deterministic seed=42 for R0 compliance)
        _b_rng = np.random.RandomState(42)
        scale = 0.01
        self._B = {
            'conv1': torch.from_numpy(_b_rng.randn(32, 256).astype(np.float32) * scale).to(_DEVICE),
            'conv2': torch.from_numpy(_b_rng.randn(64, 256).astype(np.float32) * scale).to(_DEVICE),
            'conv3': torch.from_numpy(_b_rng.randn(128, 256).astype(np.float32) * scale).to(_DEVICE),
            'conv4': torch.from_numpy(_b_rng.randn(256, 256).astype(np.float32) * scale).to(_DEVICE),
        }  # FROZEN throughout training

        self._buffer = deque(maxlen=_BUFFER_MAXLEN)
        self._buffer_hashes = set()
        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._train_counter = 0
        self._step = 0

        self._episode_start_state_dict = {k: v.clone().cpu() for k, v in
                                           self._model.state_dict().items()}
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

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
            logits, avg_features, _ = self._model.forward_with_activations(tensor)
            logits = logits.squeeze(0).cpu()

        # Entropy-driven selection (same as step 1323)
        action_logits = logits[:5]
        coord_logits = logits[5:]
        action_probs = torch.sigmoid(action_logits)
        coord_probs_scaled = torch.sigmoid(coord_logits) / 4096.0
        combined = torch.cat([action_probs, coord_probs_scaled])
        total = combined.sum().item()
        if total <= 1e-12:
            combined = torch.ones(_SG_OUTPUT_DIM) / _SG_OUTPUT_DIM
        else:
            combined = combined / total

        sg_idx = int(torch.multinomial(combined, 1).item())
        self._prev_sg_idx = sg_idx
        action = self._sg_to_prism(sg_idx)
        return action

    def update_after_step(self, obs_next, action, reward_env):
        if self._use_random or self._prev_one_hot is None:
            return
        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return

        one_hot_next = _obs_to_one_hot(obs_next_arr)
        sg_idx = self._prev_sg_idx if self._prev_sg_idx is not None else (action % 5)
        action_type_vec = _sg_to_action_type_vec(sg_idx)

        self._add_to_buffer(self._prev_one_hot, sg_idx, action_type_vec, one_hot_next)

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = self._dfa_train_step()
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_one_hot = None
        self._prev_sg_idx = None

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            if name in self._episode_start_state_dict:
                init_val = self._episode_start_state_dict[name]
                drift += (param.data.cpu() - init_val.cpu()).norm().item()
        return float(drift)

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def _sg_to_prism(self, sg_idx):
        if sg_idx < 5:
            return sg_idx
        click_idx = sg_idx - 5
        prism_action = N_KEYBOARD + click_idx
        if prism_action < self.n_actions:
            return prism_action
        return sg_idx % 5

    def _add_to_buffer(self, one_hot, sg_idx, action_type_vec, one_hot_next):
        h = hashlib.md5(one_hot.tobytes() + np.array([sg_idx], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state': one_hot.copy(),
                                  'action_type_vec': action_type_vec.copy(),
                                  'next_state': one_hot_next.copy()})

    def _dfa_train_step(self):
        """DFA training: no backward pass. Forward-only credit assignment."""
        n = len(self._buffer)
        buf_list = list(self._buffer)
        idx = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf_list[i] for i in idx]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])
        ).to(_DEVICE)

        with torch.no_grad():
            # Forward pass 1: current states — store layer inputs for DFA
            _, avg_features, layer_inputs = self._model.forward_with_activations(states)

            # Forward pass 2: next states — get target features
            _, target_features, _ = self._model.forward_with_activations(next_states)

            # Prediction
            pred_in = torch.cat([avg_features, action_vecs], dim=1)  # (batch, 262)
            pred_next = self._model.pred_head(pred_in)               # (batch, 256)

            # Top-level prediction error (= ∂MSE/∂pred_next, no division by 2 for scale)
            top_error = pred_next - target_features                   # (batch, 256)
            pred_loss = float((top_error ** 2).mean().item())

            # --- DFA updates (no .backward()) ---

            # 1. pred_head update (standard MSE gradient — top layer, direct)
            dW_pred = (top_error.T @ pred_in) / len(states)   # (256, 262)
            db_pred = top_error.mean(dim=0)                    # (256,)
            self._model.pred_head.weight.data -= _DFA_LR * dW_pred
            self._model.pred_head.bias.data -= _DFA_LR * db_pred

            # 2. Conv layer updates via DFA feedback matrices
            conv_layers = [self._model.conv1, self._model.conv2,
                           self._model.conv3, self._model.conv4]
            b_keys = ['conv1', 'conv2', 'conv3', 'conv4']

            for conv_layer, b_key, x_in in zip(conv_layers, b_keys, layer_inputs):
                B = self._B[b_key]                             # (C_out, 256)

                # DFA feedback: project top error to this layer's output space
                # delta: (batch, C_out) via top_error (batch,256) @ B.T (256,C_out)
                delta = top_error @ B.T                        # (batch, C_out)
                delta_mean = delta.mean(dim=0)                 # (C_out,)

                # Input spatial mean: (C_in,)
                x_in_mean = x_in.mean(dim=[0, 2, 3])          # (C_in,)

                # Approximate conv weight gradient:
                # ΔW[c_out, c_in, kH, kW] ≈ delta_mean[c_out] * x_in_mean[c_in] / (kH*kW)
                kH = conv_layer.weight.shape[2]
                kW = conv_layer.weight.shape[3]
                dW_conv = torch.outer(delta_mean, x_in_mean)   # (C_out, C_in)
                # Broadcast over kernel spatial dims
                dW_conv = (dW_conv.unsqueeze(-1).unsqueeze(-1)
                           .expand_as(conv_layer.weight.data)) / (kH * kW)
                conv_layer.weight.data -= _DFA_LR * dW_conv

        return pred_loss


# ---------------------------------------------------------------------------
# RAND substrate (no weight updates)
# ---------------------------------------------------------------------------

class RandCnnSubstrate:
    """Random CNN — same architecture, no weight updates. Control for DFA."""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(42)
        self._use_random = not (n_actions > 128)
        self._model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._step = 0
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def reset_loss_tracking(self):
        self._step = 0
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_arr):
        self._step += 1
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        if self._use_random or not _is_arc_obs(obs_arr):
            return int(self._rng.randint(self.n_actions))

        one_hot = _obs_to_one_hot(obs_arr)
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
            logits = logits.squeeze(0).cpu()
        ap = torch.sigmoid(logits[:5])
        cp = torch.sigmoid(logits[5:]) / 4096.0
        combined = torch.cat([ap, cp])
        total = combined.sum().item()
        if total <= 1e-12:
            combined = torch.ones(_SG_OUTPUT_DIM) / _SG_OUTPUT_DIM
        else:
            combined = combined / total
        sg_idx = int(torch.multinomial(combined, 1).item())
        click_idx = sg_idx - 5
        if sg_idx >= 5:
            action = N_KEYBOARD + click_idx
            return action if action < self.n_actions else sg_idx % 5
        return sg_idx

    def update_after_step(self, obs_next, action, reward_env):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return 0.0

    def get_compression_ratio(self):
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
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj': substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, condition, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    if condition == 'dfa':
        substrate = DfaCnnSubstrate(n_actions=n_actions)
    else:
        substrate = RandCnnSubstrate(n_actions=n_actions)

    result_try1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)
    substrate.reset_loss_tracking()
    result_try2 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_B)

    l1_try1 = result_try1['level_first_step'].get(1)
    l1_try2 = result_try2['level_first_step'].get(1)

    if l1_try1 is not None and l1_try2 is not None and l1_try2 > 0:
        speedup = round(l1_try1 / l1_try2, 4)
    elif l1_try1 is None and l1_try2 is not None:
        speedup = float('inf')
    elif l1_try1 is not None and l1_try2 is None:
        speedup = 0.0
    else:
        speedup = None

    return {
        'game': game_name,
        'condition': condition,
        'try1': result_try1,
        'try2': result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio': result_try1['compression_ratio'],
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — CNN + DFA (Direct Feedback Alignment, R2-compliant)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"DFA: 4 conv layers, fixed random B matrices, no backward pass, forward-only credit")
    print(f"R2 PASS — no backward mechanism. Prediction pathway updated only.")
    print(f"Primary metric: second_exposure_speedup. Conditions: DFA vs RAND")
    print(f"6 runs (3 games × 2 conditions), try1+try2 per run ({MAX_STEPS} steps each).")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    all_results = []
    speedup_by_condition = {c: [] for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for condition in CONDITIONS:
            t0 = time.time()
            result = run_game(game, condition, solver_steps)
            all_results.append(result)
            speedup_by_condition[condition].append(result['second_exposure_speedup'])
            elapsed = time.time() - t0
            print(masked_run_log(f"{label}/{LABELS[condition]}", elapsed))

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
    write_experiment_results(RESULTS_DIR, STEP, final_speedup, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT (CNN+DFA vs RAND)\n")

    for cond in CONDITIONS:
        sp = final_speedup[cond]
        cr_vals = [r['compression_ratio'] for r in all_results
                   if r['condition'] == cond and r['compression_ratio'] is not None]
        cr_str = f"cr_mean={np.mean(cr_vals):.4f}" if cr_vals else "cr=N/A"
        print(f"  {LABELS[cond]}: speedup={format_speedup(sp)}  {cr_str}")
    print()

    print("Compression analysis:")
    for r in all_results:
        if r['condition'] == 'dfa':
            label = GAME_LABELS.get(r['game'], '?')
            cr = r['compression_ratio']
            traj = r['try1']['pred_loss_traj']
            print(f"  {label}: cr={cr}  traj={traj}")
    print()

    print("KILL ASSESSMENT:")
    dfa_sp = final_speedup['dfa']
    rand_sp = final_speedup['rand']

    dfa_cr_vals = [r['compression_ratio'] for r in all_results
                   if r['condition'] == 'dfa' and r['compression_ratio'] is not None]
    dfa_cr_mean = float(np.mean(dfa_cr_vals)) if dfa_cr_vals else None

    if dfa_sp is None and (dfa_cr_mean is None or dfa_cr_mean >= 0.93):
        print(f"  >>> KILL: DFA speedup=N/A AND cr≥0.93 (no improvement over LPL)")
        print(f"  >>> Forward-only credit assignment insufficient.")
    elif dfa_cr_mean is not None and dfa_cr_mean < 0.3:
        print(f"  >>> SIGNAL: DFA cr={dfa_cr_mean:.4f} < 0.3 — meaningful compression!")
        print(f"  >>> speedup={format_speedup(dfa_sp)} — check for experience transfer.")
        print(f"  >>> NO KILL — DFA achieves compression. Investigate shared representation.")
    elif dfa_cr_mean is not None and dfa_cr_mean < 0.93:
        print(f"  >>> PARTIAL: DFA cr={dfa_cr_mean:.4f} — compression between LPL(0.93) and Adam(0.003)")
        print(f"  >>> Forward-only credit assignment works, but gap remains vs Adam.")
    else:
        print(f"  >>> KILL: DFA cr={dfa_cr_mean} ≥ 0.93 — no improvement over LPL")
    print(sep)
    print(f"\nResults: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
