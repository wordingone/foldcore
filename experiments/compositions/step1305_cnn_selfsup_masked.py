"""
Step 1305 — CNN Self-Supervised Forward Prediction (Measurement Fixed)
Leo mail 3681, 2026-03-29.

Three infrastructure fixes from Step 1304 artifact diagnosis:
  1. _episode_start_state_dict stored at creation, NEVER reset on level transitions → true wdrift
  2. Drop R3 for CNN (Jacobian perturbation near-zero through deep net — broken metric)
  3. Prediction loss trajectory at steps 1K/3K/5K/7K/9K + compression_ratio = loss@9K/loss@1K

Kill criteria (updated):
  1. wdrift < 0.1 AND action_KL < 0.01 → CNN not learning → KILL
  2. RHAE < 1e-6 chain mean → RHAE-dead → KILL

Protocol: random seed 1305, masked PRISM (Game A/B/C), 3 draws × 2 conditions = 18 runs
"""
import sys, os, time, json, random
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

import numpy as np
from collections import deque
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from substrates.step0674 import _enc_frame

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Random game selection from solved pool (seed 1305) ---
SOLVED_POOL = ['ft09', 'ls20', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
random.seed(1305)
FILTER_GAMES = sorted(random.sample(SOLVED_POOL, 3))
GAMES = FILTER_GAMES

# Masked labels: stdout never shows real game IDs
GAME_LABELS = {game: label for game, label in zip(GAMES, ['Game A', 'Game B', 'Game C'])}

N_DRAWS = 3
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # for ARGMIN-PE

# SG-SELFSUP hyperparameters
_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
_LR = 0.0001
_ACTION_ENTROPY_COEF = 0.0001
_COORD_ENTROPY_COEF = 0.00001
_SG_OUTPUT_DIM = 4101  # 5 discrete + 4096 click

N_KEYBOARD = 7   # PRISM keyboard offset for click actions

# Prediction loss trajectory checkpoints
LOSS_CHECKPOINTS = [1000, 3000, 5000, 7000, 9000]

# R3 measurement (linear substrate only)
R3_STEP = 5000
R3_N_OBS = 30
R3_N_DIRS = 10
R3_EPSILON = 0.01

# ARGMIN-PE (1282) hyperparameters
ETA_H_FLOW = 0.05
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.1
SELECTION_ALPHA = 0.1
DECAY = 0.001

CONDITIONS = ['sg_selfsup', 'argmin_pe']
LABELS = {'sg_selfsup': 'SG-SELFSUP', 'argmin_pe': 'PEEMA'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1305')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

# Prescriptions for all 10 solved pool games
SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'sequence'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'sequence'),
    'lp85':  ('lp85_fullchain.json',  'full_sequence'),
}


# ---------------------------------------------------------------------------
# CNN with forward prediction head
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    """SG CNN backbone + forward prediction head."""

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

        # Forward prediction head: avg_features (256) + action_type_vec (6) → 256
        self.pred_head = nn.Linear(256 + 6, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))           # (batch, 256, 64, 64)

        avg_features = conv_features.mean(dim=[2, 3])   # (batch, 256)

        af = self.action_pool(conv_features)
        af = af.view(af.size(0), -1)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)             # (batch, 5)

        cf = F.relu(self.coord_conv1(conv_features))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        cf = self.coord_conv4(cf)
        coord_logits = cf.view(cf.size(0), -1)           # (batch, 4096)

        combined = torch.cat([action_logits, coord_logits], dim=1)  # (batch, 4101)
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


# ---------------------------------------------------------------------------
# Self-supervised substrate
# ---------------------------------------------------------------------------

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


class SgSelfSupSubstrate:
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self._use_random = (n_actions == 128)

        self._model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_LR)
        self._buffer = deque(maxlen=_BUFFER_MAXLEN)
        self._buffer_hashes = set()

        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._train_counter = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

        # FIX 1: Store at creation time, NEVER reset on level transitions
        self._episode_start_state_dict = {k: v.clone().cpu() for k, v in
                                           self._model.state_dict().items()}

        # Prediction loss trajectory: track recent losses, record at checkpoints
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._ck_set = set(LOSS_CHECKPOINTS)

    def process(self, obs_arr):
        self.step += 1
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

        sg_idx = self._sample_sg_action(logits)
        self._prev_sg_idx = sg_idx
        action = self._sg_to_prism(sg_idx)
        self._action_counts[action % self.n_actions] += 1
        return action

    def update_after_step(self, obs_next, action, reward_env):
        if self._use_random or self._prev_one_hot is None:
            return
        obs_next_arr = np.asarray(obs_next, dtype=np.float32)
        if not _is_arc_obs(obs_next_arr):
            return

        one_hot_next = _obs_to_one_hot(obs_next_arr)
        sg_idx = self._prev_sg_idx if self._prev_sg_idx is not None else self._prism_to_sg(action)
        action_type_vec = _sg_to_action_type_vec(sg_idx)

        self._add_to_buffer(self._prev_one_hot, sg_idx, action_type_vec, one_hot_next)

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            self._train_step()

    def on_level_transition(self):
        # FIX 1: Do NOT reset model or _episode_start_state_dict
        # Only reset per-episode transient state
        self._prev_one_hot = None
        self._prev_sg_idx = None

    def get_state(self):
        return {
            'step': self.step,
            'buffer_size': len(self._buffer),
            'state_dict_snapshot': {k: v.clone().cpu() for k, v in
                                     self._model.state_dict().items()},
        }

    def get_internal_repr_readonly(self, obs_raw, snapshot_state_dict, init_state_dict=None):
        obs_arr = np.asarray(obs_raw, dtype=np.float32).ravel()
        if obs_arr.size >= 64 * 64:
            obs_3d = obs_arr[:64*64].reshape(1, 64, 64)
        else:
            return np.zeros(256, np.float32)

        one_hot = _obs_to_one_hot(obs_3d)
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)

        model_snap = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        model_snap.load_state_dict(snapshot_state_dict)
        model_snap.eval()
        with torch.no_grad():
            _, avg_features = model_snap(tensor)
        return avg_features.squeeze(0).cpu().numpy()

    def get_collapse_fraction(self):
        if self.step == 0:
            return 0.0
        return float(self._action_counts.max() / self.step)

    def compute_weight_drift(self):
        """Total parameter norm change from episode-start weights (never reset)."""
        drift = 0.0
        for name, param in self._model.named_parameters():
            init_val = self._episode_start_state_dict[name]
            drift += (param.data.cpu() - init_val.cpu()).norm().item()
        return float(drift)

    def get_pred_loss_trajectory(self):
        """Returns {step: mean_loss} for each checkpoint. None if not reached."""
        return dict(self._pred_loss_at)

    def _sample_sg_action(self, logits):
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
        return int(torch.multinomial(combined, 1).item())

    def _sg_to_prism(self, sg_idx):
        if sg_idx < 5:
            return sg_idx
        click_idx = sg_idx - 5
        prism_action = N_KEYBOARD + click_idx
        if prism_action < self.n_actions:
            return prism_action
        return sg_idx % 5

    def _prism_to_sg(self, prism_action):
        if prism_action < 5:
            return prism_action
        elif prism_action >= N_KEYBOARD:
            return 5 + (prism_action - N_KEYBOARD)
        return prism_action % 5

    def _add_to_buffer(self, one_hot, sg_idx, action_type_vec, one_hot_next):
        state_bytes = one_hot.tobytes()
        action_bytes = np.array([sg_idx], dtype=np.int32).tobytes()
        h = hashlib.md5(state_bytes + action_bytes).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state': one_hot.copy(),
                'action_type_vec': action_type_vec.copy(),
                'next_state': one_hot_next.copy(),
            })

    def _train_step(self):
        n = len(self._buffer)
        buf_list = list(self._buffer)
        indices = self._rng.choice(n, min(_BATCH_SIZE, n), replace=False)
        batch = [buf_list[i] for i in indices]

        states = torch.from_numpy(
            np.stack([b['state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        next_states = torch.from_numpy(
            np.stack([b['next_state'].astype(np.float32) for b in batch])
        ).to(_DEVICE)
        action_vecs = torch.from_numpy(
            np.stack([b['action_type_vec'] for b in batch])
        ).to(_DEVICE)

        logits, avg_features_t = self._model(states)
        with torch.no_grad():
            _, avg_features_next = self._model(next_states)

        predicted_next = self._model.predict_next(avg_features_t, action_vecs)
        pred_loss = F.mse_loss(predicted_next, avg_features_next)

        al = logits[:, :5]
        cl = logits[:, 5:]
        ap = torch.sigmoid(al)
        cp = torch.sigmoid(cl)
        action_entropy = -(ap * (ap + 1e-8).log()).sum(dim=1).mean()
        coord_entropy = -(cp * (cp + 1e-8).log()).sum(dim=1).mean()

        total_loss = pred_loss - _ACTION_ENTROPY_COEF * action_entropy - _COORD_ENTROPY_COEF * coord_entropy

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        # FIX 3: Track prediction loss at checkpoints
        loss_val = float(pred_loss.item())
        self._recent_losses.append(loss_val)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(list(self._recent_losses))), 6)


# ---------------------------------------------------------------------------
# ARGMIN-PE reference
# ---------------------------------------------------------------------------

class ArgminPeSubstrate:
    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)
        scale = 1.0 / np.sqrt(float(ENC_DIM + H_DIM))
        W_action_init = rng_w.randn(n_actions, ENC_DIM + H_DIM).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01
        self._visit_counts = np.zeros(n_actions, np.float32)
        self.pe_ema = np.zeros(n_actions, np.float32)
        self._prev_enc = None
        self._prev_ext = None
        self.step = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action):
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }

    @property
    def W_action_init_val(self):
        return self.W_action_init

    def process(self, obs_raw):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])
        self._prev_ext = ext.copy()
        score = self._visit_counts - SELECTION_ALPHA * self.pe_ema
        action = int(np.argmin(score))
        self._visit_counts[action] += 1
        self._prev_enc = enc.copy()
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        if self._prev_enc is None:
            return
        enc_after = _enc_frame(np.asarray(next_obs_raw, dtype=np.float32)) - self.running_mean
        pred_enc = self.W_pred @ self._prev_enc
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc)
        pe = float(np.linalg.norm(enc_after - pred_enc))
        self.pe_ema[action] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[action] + PE_EMA_ALPHA * pe
        flow = float(np.linalg.norm(enc_after - self._prev_enc))
        if self._prev_ext is not None:
            self.W_action[action] += ETA_H_FLOW * flow * self._prev_ext
            self.W_action *= (1.0 - DECAY)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_ext = None

    def get_collapse_fraction(self):
        return 0.0

    def compute_weight_drift(self):
        """W_action drift from init (main learned parameter)."""
        return float(np.linalg.norm(self.W_action - self.W_action_init))


# ---------------------------------------------------------------------------
# R3 computation (linear substrate only — FIX 2: dropped for CNN)
# ---------------------------------------------------------------------------

def compute_r3_linear(substrate, obs_sample, snapshot):
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot.get('running_mean')
    frozen_h = snapshot.get('h')
    frozen_W = snapshot.get('W_action')
    if frozen_rm is None or frozen_h is None or frozen_W is None:
        return 0.0, False
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W = substrate.W_action_init.copy()
    obs_subset = obs_sample[-R3_N_OBS:]
    rng = np.random.RandomState(42)
    diffs = []
    for obs_arr in obs_subset:
        obs_flat = obs_arr.ravel()
        dirs = rng.randn(R3_N_DIRS, len(obs_flat)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        base_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W)
        base_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W)
        for d in dirs:
            pert = obs_flat + R3_EPSILON * d
            pe = substrate.get_internal_repr_readonly(pert, frozen_rm, frozen_h, frozen_W)
            pf = substrate.get_internal_repr_readonly(pert, fresh_rm, fresh_h, fresh_W)
            jac_exp = (pe - base_exp) / R3_EPSILON
            jac_fresh = (pf - base_fresh) / R3_EPSILON
            diffs.append(np.linalg.norm(jac_exp - jac_fresh))
    r3 = float(np.mean(diffs)) if diffs else 0.0
    return round(r3, 4), r3 >= 0.05


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_game(game_name):
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
    try:
        with open(os.path.join(PDIR, fname)) as f:
            d = json.load(f)
        return d.get(field)
    except Exception:
        return None


def compute_solver_level_steps(game_name, seed=1):
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=seed)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    ACTION_OFFSET = {'ls20': -1, 'vc33': 7}
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
            env.reset(seed=seed)
            fresh_episode = True
    return level_first_step


def compute_arc_score(level_first_steps, solver_level_steps):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_action_kl(action_log, n_actions):
    """KL divergence from early (steps 0-500) to late (last 500 steps)."""
    if len(action_log) < 500:
        return None
    early = action_log[:500]
    late = action_log[-500:]
    early_c = np.zeros(n_actions, np.float32)
    late_c = np.zeros(n_actions, np.float32)
    for a in early:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in late:
        if 0 <= a < n_actions:
            late_c[a] += 1
    if early_c.sum() == 0 or late_c.sum() == 0:
        return None
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p = (late_c + 1e-8) / (late_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(late_p * np.log(late_p / early_p))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, condition, n_actions, solver_level_steps, seed,
                take_r3_snapshot=False):
    obs = env.reset(seed=seed)
    action_log = []
    obs_store = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

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

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()
        if take_r3_snapshot and steps == R3_STEP and condition == 'argmin_pe':
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

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

    # I3cv
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    rhae = arc_score
    action_kl = compute_action_kl(action_log, n_actions)
    collapse_frac = substrate.get_collapse_fraction() if hasattr(substrate, 'get_collapse_fraction') else 0.0

    wdrift = substrate.compute_weight_drift() if hasattr(substrate, 'compute_weight_drift') else None

    # FIX 2: R3 only for linear substrate
    r3_val, r3_pass = None, False
    if take_r3_snapshot and r3_snapshot is not None and condition == 'argmin_pe':
        r3_val, r3_pass = compute_r3_linear(substrate, r3_obs_sample, r3_snapshot)

    # FIX 3: Prediction loss trajectory (SG-SELFSUP only)
    pred_loss_traj = None
    compression_ratio = None
    if condition == 'sg_selfsup' and hasattr(substrate, 'get_pred_loss_trajectory'):
        pred_loss_traj = substrate.get_pred_loss_trajectory()
        l1k = pred_loss_traj.get(1000)
        l9k = pred_loss_traj.get(9000)
        if l1k is not None and l9k is not None and l1k > 1e-8:
            compression_ratio = round(l9k / l1k, 4)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'collapse_fraction': round(collapse_frac, 4),
        'wdrift': round(wdrift, 4) if wdrift is not None else None,
        'R3': r3_val,
        'R3_pass': r3_pass,
        'pred_loss_traj': pred_loss_traj,
        'compression_ratio': compression_ratio,
    }


def make_substrate(condition, n_actions, seed):
    if condition == 'sg_selfsup':
        return SgSelfSupSubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'argmin_pe':
        return ArgminPeSubstrate(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def run_draw(condition, game_name, draw_idx, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_a = draw_idx * 2
    seed_b = draw_idx * 2 + 1
    take_r3 = True

    substrate = make_substrate(condition, n_actions, seed_a)

    result_a = run_episode(env, substrate, condition, n_actions, solver_level_steps,
                           seed=seed_a, take_r3_snapshot=take_r3)
    result_b = run_episode(env, substrate, condition, n_actions, solver_level_steps,
                           seed=seed_b, take_r3_snapshot=False)

    speedup = None
    l1_a = result_a['level_first_step'].get(1)
    l1_b = result_b['level_first_step'].get(1)
    if l1_a is not None and l1_b is not None:
        speedup = round(l1_a / l1_b, 3)
    elif l1_a is None and l1_b is not None:
        speedup = float('inf')

    return {
        'game': game_name,
        'draw': draw_idx,
        'condition': condition,
        'episode_A': result_a,
        'episode_B': result_b,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1305 — CNN Self-Supervised Forward Prediction (Measurement Fixed)")
    print(f"Device: {_DEVICE}")
    print(f"Games: {list(GAME_LABELS.values())} (real IDs masked in stdout)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Draws: {N_DRAWS} per game per condition")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
    print(f"Fixes: wdrift from episode-start (never reset), R3 dropped for CNN, loss trajectory added")
    print()

    # Write game mapping to JSONL (reproducibility, not for analysis)
    mapping_path = os.path.join(RESULTS_DIR, 'game_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({'games': GAMES, 'labels': GAME_LABELS}, f)

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception as e:
            solver_steps_cache[game] = {}
    print()

    all_results = []

    # Per-condition chain-level accumulators
    chain = {c: {
        'r3_vals': [], 'rhae_vals': [], 'wdrift_vals': [],
        'action_kl_vals': [], 'i3cv_vals': [], 'runtimes': [],
        'compression_ratios': [],
        'pred_loss_by_ck': {ck: [] for ck in LOSS_CHECKPOINTS},
    } for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        solver_level_steps = solver_steps_cache.get(game, {})

        print(f"=== {label} ===")
        for draw in range(N_DRAWS):
            t0 = time.time()
            draw_results = {}
            for condition in CONDITIONS:
                try:
                    result = run_draw(condition, game, draw, solver_level_steps)
                except Exception as e:
                    import traceback
                    print(f"  {LABELS[condition]} draw {draw} ERROR: {e}")
                    traceback.print_exc()
                    continue

                all_results.append(result)
                draw_results[condition] = result
                ea = result['episode_A']

                chain[condition]['rhae_vals'].append(ea['RHAE'])
                if ea['I3_cv'] is not None:
                    chain[condition]['i3cv_vals'].append(ea['I3_cv'])
                if ea['R3'] is not None:
                    chain[condition]['r3_vals'].append(ea['R3'])
                if ea['wdrift'] is not None:
                    chain[condition]['wdrift_vals'].append(ea['wdrift'])
                if ea['action_kl'] is not None:
                    chain[condition]['action_kl_vals'].append(ea['action_kl'])
                chain[condition]['runtimes'].append(
                    ea['elapsed_seconds'] + result['episode_B']['elapsed_seconds'])
                if ea.get('compression_ratio') is not None:
                    chain[condition]['compression_ratios'].append(ea['compression_ratio'])
                if ea.get('pred_loss_traj'):
                    for ck in LOSS_CHECKPOINTS:
                        val = ea['pred_loss_traj'].get(ck)
                        if val is not None:
                            chain[condition]['pred_loss_by_ck'][ck].append(val)

            elapsed = time.time() - t0
            ss = draw_results.get('sg_selfsup', {}).get('episode_A', {})
            pe = draw_results.get('argmin_pe', {}).get('episode_A', {})
            cr = ss.get('compression_ratio')
            print(f"  {label} draw {draw}: "
                  f"SS[wdrift={ss.get('wdrift')},kl={ss.get('action_kl')},RHAE={ss.get('RHAE',0):.2e},cr={cr}] "
                  f"PE[R3={pe.get('R3')},wdrift={pe.get('wdrift')},RHAE={pe.get('RHAE',0):.2e}]  ({elapsed:.1f}s)")

        # Save raw per-game JSONL (real IDs in file — for reproducibility only)
        out_path = os.path.join(RESULTS_DIR, f"{game}_1305.jsonl")
        game_results = [r for r in all_results if r['game'] == game]
        with open(out_path, 'w') as f:
            for r in game_results:
                f.write(json.dumps(r) + '\n')
        print()

    # ---------------------------------------------------------------------------
    # Chain-level summary
    # ---------------------------------------------------------------------------
    print("\n" + "="*100)
    print("STEP 1305 — CHAIN-LEVEL SUMMARY (aggregate across all games)")
    print()

    chain_summary = {}
    for condition in CONDITIONS:
        d = chain[condition]
        mean_r3 = round(float(np.mean(d['r3_vals'])), 4) if d['r3_vals'] else None
        mean_rhae = round(float(np.mean(d['rhae_vals'])), 6) if d['rhae_vals'] else 0.0
        mean_wdrift = round(float(np.mean(d['wdrift_vals'])), 4) if d['wdrift_vals'] else None
        mean_action_kl = round(float(np.mean(d['action_kl_vals'])), 4) if d['action_kl_vals'] else None
        mean_i3cv = round(float(np.mean(d['i3cv_vals'])), 4) if d['i3cv_vals'] else None
        mean_cr = round(float(np.mean(d['compression_ratios'])), 4) if d['compression_ratios'] else None

        # Prediction loss trajectory chain aggregates
        loss_traj_summary = {}
        for ck in LOSS_CHECKPOINTS:
            vals = d['pred_loss_by_ck'][ck]
            loss_traj_summary[ck] = round(float(np.mean(vals)), 6) if vals else None

        chain_summary[condition] = {
            'mean_R3': mean_r3,
            'mean_RHAE': mean_rhae,
            'mean_wdrift': mean_wdrift,
            'mean_action_KL': mean_action_kl,
            'mean_I3cv': mean_i3cv,
            'mean_compression_ratio': mean_cr,
            'pred_loss_trajectory': loss_traj_summary,
        }

        print(f"  {LABELS[condition]}:")
        print(f"    mean_RHAE={mean_rhae:.2e}  mean_wdrift={mean_wdrift}  mean_action_KL={mean_action_kl}  mean_I3cv={mean_i3cv}")
        if condition == 'sg_selfsup':
            print(f"    pred_loss_trajectory: {loss_traj_summary}")
            print(f"    mean_compression_ratio (loss@9K/loss@1K): {mean_cr}")
        else:
            print(f"    mean_R3={mean_r3}")

    print()

    # ---------------------------------------------------------------------------
    # Kill assessment
    # ---------------------------------------------------------------------------
    print("KILL ASSESSMENT (chain aggregates only):")
    ss = chain_summary.get('sg_selfsup', {})

    ss_wdrift = ss.get('mean_wdrift')
    ss_rhae = ss.get('mean_RHAE', 0.0)
    ss_kl = ss.get('mean_action_KL')
    ss_cr = ss.get('mean_compression_ratio')

    killed = False

    if ss_wdrift is not None and ss_kl is not None:
        if ss_wdrift < 0.1 and ss_kl < 0.01:
            print(f"  >>> KILL: wdrift={ss_wdrift} < 0.1 AND action_KL={ss_kl} < 0.01 → CNN not learning")
            killed = True
        elif ss_wdrift >= 0.1 and ss_kl >= 0.01:
            print(f"  >>> SIGNAL: wdrift={ss_wdrift} ≥ 0.1 AND action_KL={ss_kl} ≥ 0.01 → CNN showing learning")
        elif ss_wdrift >= 0.1:
            print(f"  >>> PARTIAL: wdrift={ss_wdrift} ≥ 0.1 (weight change) but action_KL={ss_kl} < 0.01 (no behavioral change)")
        else:
            print(f"  >>> PARTIAL: action_KL={ss_kl} ≥ 0.01 (behavioral change) but wdrift={ss_wdrift} < 0.1")

    if ss_rhae is not None and ss_rhae < 1e-6:
        print(f"  >>> KILL: RHAE={ss_rhae:.2e} < 1e-6 → RHAE-dead")
        killed = True

    if ss_cr is not None:
        if ss_cr < 0.5:
            print(f"  >>> compression_ratio={ss_cr} < 0.5 → loss dropped by >50% (prediction improving)")
        elif ss_cr > 1.5:
            print(f"  >>> compression_ratio={ss_cr} > 1.5 → loss increased (prediction diverging)")
        else:
            print(f"  >>> compression_ratio={ss_cr} ≈ 1.0 → no systematic loss trend")

    if not killed:
        print("  >>> NO KILL triggered")

    print("="*100)

    # Save chain summary
    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'chain_summary': chain_summary,
            'n_draws': N_DRAWS,
            'games_label_only': list(GAME_LABELS.values()),
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
