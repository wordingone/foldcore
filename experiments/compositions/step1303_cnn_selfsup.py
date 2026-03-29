"""
Step 1303 — CNN Self-Supervised Forward Prediction (15-min filter)
Leo mail 3671, 2026-03-28.

First non-Hebbian experiment. Does credit assignment (backprop) + depth (CNN) +
self-supervised loss produce genuine learning that Hebbian on a linear map cannot?

Architecture: StochasticGoose CNN backbone + forward prediction head.
Replace frame-change reward with:
  loss = MSE(predict(avg_features_t, action_type), avg_features_{t+1})
where avg_features = global avg pool of conv4 output (256-dim).

Key property: backprop trains the ENTIRE backbone via prediction loss.
Action head piggybacks on backbone representations. No external reward.

R2 compliance: conv backbone (W) encodes observations AND feeds action head
that selects actions. Same computation path.
R1 compliance: forward prediction error is self-computed from observations,
not an evaluative signal from the environment.

Prediction head (NEW component):
  Input: avg_features (256) + action_type_vec (6: 5 keyboard one-hot + is_click)
  Output: predicted next avg_features (256)
  Size: Linear(262, 256) = 67K params. Trainable online.

Action selection: unchanged from SG (sample from softmax of action_head logits).
Entropy regularization: preserved from SG.

Conditions (2):
  SG-SELFSUP: CNN backbone + self-supervised prediction loss (no frame-change reward)
  ARGMIN-PE: Step 1282 reference (Hebbian linear map)

Protocol: 15-min filter. 3 games × 3 draws × 2 conditions = 18 runs.

Kill criteria:
  - SG-SELFSUP R3 < 0.01 on all 3 games → self-supervised loss insufficient
  - SG-SELFSUP I3cv matches random (~sqrt(n_actions)) → no action structure

Decision tree:
  R3 > 0.05 + non-random I3cv → CNN + self-supervised works → full PRISM
  R3 > 0.05 but random actions → encoder learns but action head drifts
  R3 < 0.01 → budget/loss design issue → re-spec
"""
import sys, os, time, json
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

# --- Config ---
FILTER_GAMES = ['ls20', 'ft09', 'sp80']
MBPP_GAMES = []
GAMES = FILTER_GAMES

N_DRAWS = 3
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # for ARGMIN-PE

# SG-SELFSUP hyperparameters (same as SG except no frame-change reward)
_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
_LR = 0.0001
_ACTION_ENTROPY_COEF = 0.0001
_COORD_ENTROPY_COEF = 0.00001
_SG_OUTPUT_DIM = 4101  # 5 discrete + 4096 click

N_KEYBOARD = 7   # PRISM keyboard offset for click actions

# R3 measurement
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

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1303')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'sequence'),
}


# ---------------------------------------------------------------------------
# CNN with forward prediction head
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    """SG CNN backbone + forward prediction head.

    forward() returns (combined_logits, avg_features):
      - combined_logits: (batch, 4101) — same as original ActionModel
      - avg_features: (batch, 256) — global avg pool of conv4, used for prediction
    """

    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5

        # Shared convolutional backbone (exact SG)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        # Action head (exact SG)
        self.action_pool = nn.MaxPool2d(4, 4)
        action_flattened_size = 256 * 16 * 16
        self.action_fc = nn.Linear(action_flattened_size, 512)
        self.action_head = nn.Linear(512, self.num_action_types)

        # Coordinate head (exact SG)
        self.coord_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.coord_conv3 = nn.Conv2d(64, 32, kernel_size=1)
        self.coord_conv4 = nn.Conv2d(32, 1, kernel_size=1)

        self.dropout = nn.Dropout(0.2)

        # Forward prediction head (NEW):
        # avg_features (256) + action_type_vec (6: 5 keyboard + is_click) → 256
        _PRED_INPUT = 256 + 6
        self.pred_head = nn.Linear(_PRED_INPUT, 256)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        conv_features = F.relu(self.conv4(x))           # (batch, 256, 64, 64)

        # Global avg pool for prediction target
        avg_features = conv_features.mean(dim=[2, 3])   # (batch, 256)

        # Action head
        af = self.action_pool(conv_features)            # (batch, 256, 16, 16)
        af = af.view(af.size(0), -1)                    # (batch, 65536)
        af = F.relu(self.action_fc(af))
        af = self.dropout(af)
        action_logits = self.action_head(af)            # (batch, 5)

        # Coordinate head
        cf = F.relu(self.coord_conv1(conv_features))
        cf = F.relu(self.coord_conv2(cf))
        cf = F.relu(self.coord_conv3(cf))
        cf = self.coord_conv4(cf)
        coord_logits = cf.view(cf.size(0), -1)          # (batch, 4096)

        combined = torch.cat([action_logits, coord_logits], dim=1)  # (batch, 4101)
        return combined, avg_features

    def predict_next(self, avg_features, action_type_vec):
        """Predict next avg_features given current avg_features + action type."""
        inp = torch.cat([avg_features, action_type_vec], dim=1)
        return self.pred_head(inp)


def _sg_to_action_type_vec(sg_idx):
    """Map SG action index → 6-dim action type vector (5 keyboard one-hot + is_click)."""
    vec = np.zeros(6, dtype=np.float32)
    if sg_idx < 5:
        vec[sg_idx] = 1.0
    else:
        vec[5] = 1.0  # is_click flag
    return vec


# ---------------------------------------------------------------------------
# Self-supervised substrate
# ---------------------------------------------------------------------------

def _obs_to_one_hot(obs_arr):
    """Convert (1, 64, 64) float32 → (16, 64, 64) bool array."""
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
    """CNN + self-supervised forward prediction loss.

    Replaces frame-change reward with:
      MSE(predict(avg_features_t, action_type), avg_features_{t+1})

    Action selection: same as SG (sample from softmax of combined logits).
    Buffer: stores (state, action_idx, action_type_vec, next_state). No reward.
    Training: minimizes prediction MSE + entropy regularization.
    """

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

        # For R3: store initial weights snapshot
        self._init_state_dict = {k: v.clone().cpu() for k, v in
                                  self._model.state_dict().items()}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

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
        self._model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=_LR)
        self._buffer.clear()
        self._buffer_hashes.clear()
        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._init_state_dict = {k: v.clone().cpu() for k, v in
                                  self._model.state_dict().items()}

    def get_state(self):
        return {'step': self.step, 'buffer_size': len(self._buffer),
                'state_dict_snapshot': {k: v.clone().cpu() for k, v in
                                         self._model.state_dict().items()}}

    def get_internal_repr_readonly(self, obs_raw, snapshot_state_dict, init_state_dict=None):
        """Return avg_features (256-dim) using snapshot weights."""
        obs_arr = np.asarray(obs_raw, dtype=np.float32).ravel()
        # Reconstruct (1, 64, 64) from flattened
        if obs_arr.size >= 64 * 64:
            obs_3d = obs_arr[:64*64].reshape(1, 64, 64)
        else:
            return np.zeros(256, np.float32)

        one_hot = _obs_to_one_hot(obs_3d)
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)

        # Load snapshot weights temporarily
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

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

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

        # Forward pass: get avg_features for current and next states
        logits, avg_features_t = self._model(states)          # (B, 4101), (B, 256)
        with torch.no_grad():
            _, avg_features_next = self._model(next_states)   # (B, 256) — target

        # Forward prediction loss
        predicted_next = self._model.predict_next(avg_features_t, action_vecs)
        pred_loss = F.mse_loss(predicted_next, avg_features_next)

        # Entropy regularization (preserved from SG)
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


# ---------------------------------------------------------------------------
# ARGMIN-PE reference
# ---------------------------------------------------------------------------

class ArgminPeSubstrate:
    """Step 1282 pe_ema composition (reference)."""

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


# ---------------------------------------------------------------------------
# R3 computation (adapted for CNN vs linear substrates)
# ---------------------------------------------------------------------------

def compute_r3_cnn(substrate, obs_sample, snapshot_state_dict):
    """R3 for CNN substrate: Jacobian diff of experienced vs fresh state_dict."""
    if not obs_sample or snapshot_state_dict is None:
        return None, False

    init_sd = substrate._init_state_dict
    rng = np.random.RandomState(42)
    diffs = []
    obs_subset = obs_sample[-R3_N_OBS:]

    for obs_arr in obs_subset:
        obs_flat = obs_arr.ravel()
        if len(obs_flat) < 64 * 64:
            continue
        dirs = rng.randn(R3_N_DIRS, 64 * 64).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

        obs_3d = obs_flat[:64*64].reshape(1, 64, 64).astype(np.float32)

        base_exp = substrate.get_internal_repr_readonly(obs_flat, snapshot_state_dict)
        base_fresh = substrate.get_internal_repr_readonly(obs_flat, init_sd)

        for d in dirs:
            pert = obs_flat.copy()
            pert[:64*64] += R3_EPSILON * d
            pe = substrate.get_internal_repr_readonly(pert, snapshot_state_dict)
            pf = substrate.get_internal_repr_readonly(pert, init_sd)
            jac_exp = (pe - base_exp) / R3_EPSILON
            jac_fresh = (pf - base_fresh) / R3_EPSILON
            diffs.append(np.linalg.norm(jac_exp - jac_fresh))

    r3 = float(np.mean(diffs)) if diffs else 0.0
    return round(r3, 4), r3 >= 0.05


def compute_r3_linear(substrate, obs_sample, snapshot):
    """R3 for linear substrate (ARGMIN-PE)."""
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


def compute_rhae(level_first_steps, solver_level_steps, total_actions):
    return compute_arc_score(level_first_steps, solver_level_steps)


def compute_post_transition_kl(action_log, l1_step, n_actions, window=100):
    if l1_step is None or l1_step < window:
        return None
    pre = action_log[max(0, l1_step - window):l1_step]
    post = action_log[l1_step:min(len(action_log), l1_step + window)]
    if len(pre) < 10 or len(post) < 10:
        return None
    pre_c = np.zeros(n_actions, np.float32)
    post_c = np.zeros(n_actions, np.float32)
    for a in pre:
        if 0 <= a < n_actions:
            pre_c[a] += 1
    for a in post:
        if 0 <= a < n_actions:
            post_c[a] += 1
    if pre_c.sum() == 0 or post_c.sum() == 0:
        return None
    pre_p = (pre_c + 1e-8) / (pre_c.sum() + 1e-8 * n_actions)
    post_p = (post_c + 1e-8) / (post_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(post_p * np.log(post_p / pre_p))), 4)


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
        if take_r3_snapshot and steps == R3_STEP:
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

    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    rhae = compute_rhae(level_first_step, solver_level_steps, steps)
    post_kl = compute_post_transition_kl(action_log, l1_step, n_actions)
    collapse_frac = substrate.get_collapse_fraction() if hasattr(substrate, 'get_collapse_fraction') else 0.0

    # R3
    r3_val, r3_pass = None, False
    if take_r3_snapshot and r3_snapshot is not None:
        if condition == 'sg_selfsup':
            sd = r3_snapshot.get('state_dict_snapshot')
            r3_val, r3_pass = compute_r3_cnn(substrate, r3_obs_sample, sd)
        else:
            r3_val, r3_pass = compute_r3_linear(substrate, r3_obs_sample, r3_snapshot)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 6),
        'I3_cv': i3_cv,
        'post_transition_kl': post_kl,
        'collapse_fraction': round(collapse_frac, 4),
        'R3': r3_val,
        'R3_pass': r3_pass,
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
    take_r3 = True  # measure R3 on all conditions (key comparison)

    substrate = make_substrate(condition, n_actions, seed_a)

    result_a = run_episode(env, substrate, condition, n_actions, solver_level_steps,
                           seed=seed_a, take_r3_snapshot=take_r3)
    result_b = run_episode(env, substrate, condition, n_actions, solver_level_steps,
                           seed=seed_b, take_r3_snapshot=False)

    speedup = None
    if result_a['l1_step'] is not None and result_b['l1_step'] is not None:
        speedup = round(result_a['l1_step'] / result_b['l1_step'], 3)
    elif result_a['l1_step'] is None and result_b['l1_step'] is not None:
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
    print(f"Step 1303 — CNN Self-Supervised Forward Prediction (15-min filter)")
    print(f"Device: {_DEVICE}")
    print(f"Games: {GAMES}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Draws: {N_DRAWS} per game per condition")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
    print()

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
            print(f"  {game}: {solver_steps_cache[game]}")
        except Exception as e:
            print(f"  {game}: ERROR {e}")
            solver_steps_cache[game] = {}
    print()

    all_results = []
    summary_rows = []

    for game in GAMES:
        solver_level_steps = solver_steps_cache.get(game, {})
        per_cond = {c: {'results': [], 'l1_a': 0, 'arc_a': [],
                        'r3_vals': [], 'i3cv_a': [],
                        'collapse_a': [], 'runtimes': []}
                   for c in CONDITIONS}

        print(f"=== {game.upper()} ===")
        for draw in range(N_DRAWS):
            t0 = time.time()
            for condition in CONDITIONS:
                try:
                    result = run_draw(condition, game, draw, solver_level_steps)
                except Exception as e:
                    import traceback
                    print(f"  {condition} draw {draw} ERROR: {e}")
                    traceback.print_exc()
                    continue

                all_results.append(result)
                d = per_cond[condition]
                d['results'].append(result)
                ea = result['episode_A']
                if ea['L1_solved']:
                    d['l1_a'] += 1
                d['arc_a'].append(ea['arc_score'])
                if ea['I3_cv'] is not None:
                    d['i3cv_a'].append(ea['I3_cv'])
                if ea['R3'] is not None:
                    d['r3_vals'].append(ea['R3'])
                d['collapse_a'].append(ea['collapse_fraction'])
                d['runtimes'].append(ea['elapsed_seconds'] + result['episode_B']['elapsed_seconds'])

            elapsed = time.time() - t0
            ss_ea = per_cond['sg_selfsup']['results'][-1]['episode_A'] if per_cond['sg_selfsup']['results'] else {}
            pe_ea = per_cond['argmin_pe']['results'][-1]['episode_A'] if per_cond['argmin_pe']['results'] else {}
            print(f"  draw {draw}: SS[L1={ss_ea.get('L1_solved',False)},R3={ss_ea.get('R3')},cf={ss_ea.get('collapse_fraction',0):.3f},I3cv={ss_ea.get('I3_cv')}] "
                  f"PE[L1={pe_ea.get('L1_solved',False)},R3={pe_ea.get('R3')}]  ({elapsed:.1f}s)")

        # Game summary
        for condition in CONDITIONS:
            d = per_cond[condition]
            n = len(d['results'])
            if n == 0:
                continue
            row = {
                'game': game, 'condition': condition, 'n': n,
                'L1_rate': f"{d['l1_a']}/{n}",
                'arc_mean': round(float(np.mean(d['arc_a'])) if d['arc_a'] else 0.0, 4),
                'I3cv_mean': round(float(np.mean(d['i3cv_a'])), 4) if d['i3cv_a'] else None,
                'R3_mean': round(float(np.mean(d['r3_vals'])), 4) if d['r3_vals'] else None,
                'collapse_mean': round(float(np.mean(d['collapse_a'])), 4) if d['collapse_a'] else 0.0,
                'runtime_mean': round(float(np.mean(d['runtimes'])), 1) if d['runtimes'] else 0.0,
            }
            summary_rows.append(row)

        ss_r3 = float(np.mean(per_cond['sg_selfsup']['r3_vals'])) if per_cond['sg_selfsup']['r3_vals'] else 0.0
        pe_r3 = float(np.mean(per_cond['argmin_pe']['r3_vals'])) if per_cond['argmin_pe']['r3_vals'] else 0.0
        ss_cf = float(np.mean(per_cond['sg_selfsup']['collapse_a'])) if per_cond['sg_selfsup']['collapse_a'] else 0.0
        print(f"  GAME SUMMARY: SS[R3={ss_r3:.4f},cf={ss_cf:.3f}] PE[R3={pe_r3:.4f}]")
        print()

        out_path = os.path.join(RESULTS_DIR, f"{game}_1303.jsonl")
        game_results = [r for r in all_results if r['game'] == game]
        with open(out_path, 'w') as f:
            for r in game_results:
                f.write(json.dumps(r) + '\n')

    summary_path = os.path.join(RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({'summary': summary_rows, 'n_draws': N_DRAWS}, f, indent=2)

    # Kill assessment
    ss_all_r3 = [r['R3_mean'] for r in summary_rows
                 if r['condition'] == 'sg_selfsup' and r['R3_mean'] is not None]

    print("\n" + "="*100)
    print("STEP 1303 — KILL ASSESSMENT")
    if ss_all_r3:
        print(f"  SG-SELFSUP R3 range: {min(ss_all_r3):.4f} – {max(ss_all_r3):.4f}")
        print(f"  Kill threshold: R3 < 0.01 on ALL games")
        if all(r < 0.01 for r in ss_all_r3):
            print("  >>> KILL: self-supervised loss insufficient")
        elif any(r > 0.05 for r in ss_all_r3):
            print("  >>> PASS: R3 > 0.05 on at least one game → proceed to full PRISM")
        else:
            print("  >>> BORDERLINE: R3 between 0.01–0.05")
    print()

    print(f"{'Game':<10} {'Cond':<14} {'L1':>6} {'ARC':>8} {'R3':>8} {'CF':>8} {'I3cv':>8}")
    print("-"*100)
    for row in summary_rows:
        print(f"{row['game']:<10} {LABELS[row['condition']]:<14} "
              f"{row['L1_rate']:>6} {row['arc_mean']:>8.4f} "
              f"{str(round(row['R3_mean'],4)) if row['R3_mean'] is not None else 'N/A':>8} "
              f"{row['collapse_mean']:>8.4f} "
              f"{str(round(row['I3cv_mean'],4)) if row['I3cv_mean'] is not None else 'N/A':>8}")
    print("="*100)
    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == '__main__':
    main()
