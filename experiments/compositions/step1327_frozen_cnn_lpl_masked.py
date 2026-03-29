"""
Step 1327 — Frozen CNN encoder + LPL on top (encoding quality vs update rule)
Leo mails 3759/3761, 2026-03-29.

## What this separates

Prior experiments confounded encoding quality and update rule power:
  CNN+Adam: rich encoding + strong update → cr=0.003
  LPL+avgpool: flat encoding + weak update → cr=0.93
  DFA+CNN: rich encoding + medium update → cr=0.66

This experiment freezes a pretrained CNN encoder and runs LPL on its features.

  CNN-LPL: frozen CNN features (256-dim) → W1/W2 LPL updates
  BASELINE-LPL: avgpool4 centered encoding (256-dim) → W1/W2 LPL updates

Kill: CNN-LPL cr ≈ BASELINE-LPL cr (both ≥ 0.9) → encoding NOT the bottleneck
Signal: CNN-LPL cr << BASELINE-LPL cr (CNN-LPL cr < 0.5) → encoding IS the bottleneck

## Warmup
Train CNN 1K steps on first ARC game (Adam, same as step 1305 architecture).
Freeze weights. Use as static feature extractor.

## Constitutional audit
R0: Deterministic init. CNN warmup uses fixed seed=0. W1/W2 identity init.
R1: PASS — LPL prediction errors self-computed from W1/W2 dynamics.
R2: PASS for LPL layers. VIOLATION for frozen CNN (not self-modifying) — acknowledged probe.
R3: PARTIAL — W1/W2 update every step. CNN frozen throughout experiment.
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

from substrates.step0674 import _enc_frame
from prism_masked import (select_games, seal_mapping, label_filename,
                           masked_game_list, masked_run_log,
                           format_speedup, write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1327)

STEP         = 1327
MAX_STEPS    = 2_000
MAX_SECONDS  = 120
WARMUP_STEPS = 1_000
SEED_A       = 0
SEED_B       = 1

CONDITIONS = ['cnn_lpl', 'baseline_lpl']
LABELS     = {'cnn_lpl': 'CNN-LPL', 'baseline_lpl': 'BASELINE-LPL'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1327')
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

_DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_BATCH_SIZE  = 64
_TRAIN_FREQ  = 5
_LR          = 0.0001
_SG_OUT_DIM  = 4101
N_KEYBOARD   = 7

# LPL hyperparameters
ENC_DIM = 256
H1_DIM  = 128
H2_DIM  = 64
ETA     = 0.001
INF_LR  = 0.05
K_INF   = 5
W_CLIP  = 100.0
T_SOFT  = 1.0          # Leo confirmed T=1.0

LOSS_CHECKPOINTS = [500, 1000, 2000]


# ---------------------------------------------------------------------------
# CNN model (same architecture as step 1323/1326)
# ---------------------------------------------------------------------------

class SgSelfSupModel(nn.Module):
    def __init__(self, input_channels=16, grid_size=64):
        super().__init__()
        self.grid_size = grid_size
        self.num_action_types = 5
        self.conv1 = nn.Conv2d(input_channels, 32,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64,  kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.action_pool    = nn.MaxPool2d(4, 4)
        self.action_fc      = nn.Linear(256 * 16 * 16, 512)
        self.action_head    = nn.Linear(512, self.num_action_types)
        self.coord_conv1    = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.coord_conv2    = nn.Conv2d(128, 64,  kernel_size=3, padding=1)
        self.coord_conv3    = nn.Conv2d(64, 32,  kernel_size=1)
        self.coord_conv4    = nn.Conv2d(32, 1,   kernel_size=1)
        self.dropout        = nn.Dropout(0.2)
        self.pred_head      = nn.Linear(256 + 6, 256)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        avg_features = x4.mean(dim=[2, 3])   # (batch, 256)
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
        return combined, avg_features


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
    if sg_idx < 5:
        vec[sg_idx] = 1.0
    else:
        vec[5] = 1.0
    return vec


# ---------------------------------------------------------------------------
# CNN warmup: train on first ARC game, return frozen model
# ---------------------------------------------------------------------------

def warmup_cnn(game_name):
    """Train CNN 1K steps on game_name with Adam. Freeze and return."""
    model = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
    buffer = deque(maxlen=200_000)
    buffer_hashes = set()
    rng = np.random.RandomState(42)

    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    obs = env.reset(seed=0)
    step = 0
    prev_one_hot = None
    prev_sg_idx = None
    fresh_episode = True
    t0 = time.time()

    while step < WARMUP_STEPS and (time.time() - t0) < 60:
        if obs is None:
            obs = env.reset(seed=0)
            prev_one_hot = None
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        if not _is_arc_obs(obs_arr):
            action = int(rng.randint(n_actions))
            obs_next, _, done, _ = env.step(action)
            step += 1
            obs = obs_next if not done else env.reset(seed=0)
            continue

        one_hot = _obs_to_one_hot(obs_arr)
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = model(tensor)
            logits = logits.squeeze(0).cpu()
        ap = torch.sigmoid(logits[:5])
        cp = torch.sigmoid(logits[5:]) / 4096.0
        combined = torch.cat([ap, cp])
        total = combined.sum().item()
        if total <= 1e-12:
            combined = torch.ones(_SG_OUT_DIM) / _SG_OUT_DIM
        else:
            combined = combined / total
        sg_idx = int(torch.multinomial(combined, 1).item())
        action = sg_idx if sg_idx < 5 else min(N_KEYBOARD + (sg_idx - 5), n_actions - 1)

        if prev_one_hot is not None and not fresh_episode:
            at_vec = _sg_to_action_type_vec(prev_sg_idx or 0)
            h = hashlib.md5(prev_one_hot.tobytes() +
                            np.array([sg_idx], np.int32).tobytes()).hexdigest()
            if h not in buffer_hashes:
                buffer_hashes.add(h)
                buffer.append({'state': prev_one_hot.copy(),
                                'action_type_vec': at_vec,
                                'next_state': one_hot.copy()})

        prev_one_hot = one_hot
        prev_sg_idx = sg_idx
        obs_next, _, done, _ = env.step(action)
        step += 1
        fresh_episode = False

        if len(buffer) >= _BATCH_SIZE and step % _TRAIN_FREQ == 0:
            buf_list = list(buffer)
            idx = rng.choice(len(buf_list), min(_BATCH_SIZE, len(buf_list)), replace=False)
            batch = [buf_list[i] for i in idx]
            states     = torch.from_numpy(np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
            next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
            at_vecs    = torch.from_numpy(np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)
            optimizer.zero_grad()
            _, avg_feats   = model(states)
            _, tgt_feats   = model(next_states)
            pred_in        = torch.cat([avg_feats, at_vecs], dim=1)
            pred_next      = model.pred_head(pred_in)
            loss = F.mse_loss(pred_next, tgt_feats)
            loss.backward()
            optimizer.step()

        obs = obs_next
        if done or obs is None:
            obs = env.reset(seed=0)
            prev_one_hot = None
            fresh_episode = True

    # Freeze
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Shared action selection from h1
# ---------------------------------------------------------------------------

def _h1_to_action(h1, n_actions):
    """Select action using h1 (128-dim). Hash modulo for large action spaces."""
    if n_actions <= H1_DIM:
        logits = h1[:n_actions].astype(np.float64) / T_SOFT
    else:
        indices = np.arange(n_actions) % H1_DIM
        logits = h1[indices].astype(np.float64) / T_SOFT
    logits -= logits.max()
    probs = np.exp(logits)
    total = probs.sum()
    if total < 1e-12:
        probs = np.ones(n_actions) / n_actions
    else:
        probs /= total
    return int(np.random.choice(n_actions, p=probs))


# ---------------------------------------------------------------------------
# LPL substrate base — shared W1/W2 logic
# ---------------------------------------------------------------------------

class BaseLplSubstrate:
    """LPL with W1/W2 and shared h1 action selection. Subclass provides features."""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self._W1_init = self.W1.copy()
        self._step = 0
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _get_features(self, obs_arr):
        raise NotImplementedError

    def _infer(self, enc):
        h1 = self.W1 @ enc
        h2 = self.W2 @ h1
        for _ in range(K_INF):
            enc_pred = self.W1.T @ h1
            e1 = enc - enc_pred
            h1_pred = self.W2.T @ h2
            e2 = h1 - h1_pred
            h1 = h1 + INF_LR * (self.W1 @ e1 - e2)
            h2 = h2 + INF_LR * (self.W2 @ e2)
        enc_pred = self.W1.T @ h1
        e1 = enc - enc_pred
        h1_pred = self.W2.T @ h2
        e2 = h1 - h1_pred
        return h1, h2, e1, e2

    def process(self, obs_arr):
        self._step += 1
        features, valid = self._get_features(obs_arr)

        if not valid:
            return int(np.random.randint(self.n_actions))

        h1, h2, e1, e2 = self._infer(features)
        action = _h1_to_action(h1, self.n_actions)

        # LPL updates on W1, W2 (no W3 — shared representation)
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)

        enc_norm_sq = float(np.dot(features, features))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def reset_loss_tracking(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)


# ---------------------------------------------------------------------------
# CNN-LPL substrate
# ---------------------------------------------------------------------------

class CnnLplSubstrate(BaseLplSubstrate):
    """LPL on frozen CNN features. CNN provides rich 256-dim features."""

    def __init__(self, n_actions, frozen_cnn):
        super().__init__(n_actions)
        self._frozen_cnn = frozen_cnn
        self._cnn_running_mean = np.zeros(ENC_DIM, np.float32)
        self._cnn_n_obs = 0

    def _get_features(self, obs_arr):
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        if not _is_arc_obs(obs_arr):
            # MBPP or non-ARC: skip LPL, return random action signal
            return np.zeros(ENC_DIM, np.float32), False

        one_hot = _obs_to_one_hot(obs_arr)
        tensor = torch.from_numpy(one_hot.astype(np.float32)).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            _, avg_features = self._frozen_cnn(tensor)
        feats = avg_features.squeeze(0).cpu().numpy()  # (256,)

        # Center CNN features (running mean subtraction for numerical stability)
        self._cnn_n_obs += 1
        a = 1.0 / self._cnn_n_obs
        self._cnn_running_mean = (1 - a) * self._cnn_running_mean + a * feats
        return (feats - self._cnn_running_mean), True


# ---------------------------------------------------------------------------
# Baseline-LPL substrate (avgpool centered encoding)
# ---------------------------------------------------------------------------

class BaselineLplSubstrate(BaseLplSubstrate):
    """LPL on avgpool4 centered encoding — same as step 1313 base."""

    def __init__(self, n_actions):
        super().__init__(n_actions)
        self._running_mean = np.zeros(ENC_DIM, np.float32)
        self._n_obs = 0

    def _get_features(self, obs_arr):
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        x = _enc_frame(obs_arr)
        self._n_obs += 1
        a = 1.0 / self._n_obs
        self._running_mean = (1 - a) * self._running_mean + a * x
        return (x - self._running_mean), True


# ---------------------------------------------------------------------------
# Game factory and solvers (copied from step 1326)
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

    arc_score  = compute_arc_score(level_first_step, solver_level_steps)
    action_kl  = compute_action_kl(action_log, n_actions)

    return {
        'steps_taken':     steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level':       max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score':       round(arc_score, 6),
        'RHAE':            round(arc_score, 6),
        'I3_cv':           i3_cv,
        'action_kl':       action_kl,
        'wdrift':          round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':  substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, condition, solver_level_steps, frozen_cnn):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    if condition == 'cnn_lpl':
        substrate = CnnLplSubstrate(n_actions=n_actions, frozen_cnn=frozen_cnn)
    else:
        substrate = BaselineLplSubstrate(n_actions=n_actions)

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
        'game':                   game_name,
        'condition':              condition,
        'try1':                   result_try1,
        'try2':                   result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio':      result_try1['compression_ratio'],
        'n_actions':              n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Step {STEP} — Frozen CNN encoder + LPL (encoding quality vs update rule)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"Conditions: CNN-LPL (frozen CNN features) vs BASELINE-LPL (avgpool)")
    print(f"Kill: CNN-LPL cr ≈ BASELINE-LPL cr → update rule is bottleneck")
    print(f"Signal: CNN-LPL cr < 0.5 << BASELINE-LPL cr → encoding IS bottleneck")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Warmup: train CNN on first ARC game
    first_arc = next((g for g in GAMES if g.lower() not in ('mbpp',) and not g.lower().startswith('mbpp_')), None)
    if first_arc:
        label_warmup = GAME_LABELS.get(first_arc, first_arc)
        print(f"Warmup: training CNN {WARMUP_STEPS} steps on {label_warmup} (Adam, freeze after)...")
        t_wu = time.time()
        frozen_cnn = warmup_cnn(first_arc)
        print(f"  Warmup done in {time.time()-t_wu:.1f}s. CNN frozen.")
    else:
        print("  No ARC game in draw — using random CNN init (frozen).")
        frozen_cnn = SgSelfSupModel(input_channels=16, grid_size=64).to(_DEVICE)
        for p in frozen_cnn.parameters():
            p.requires_grad_(False)
        frozen_cnn.eval()
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
    speedup_by_condition = {c: [] for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for condition in CONDITIONS:
            t0 = time.time()
            result = run_game(game, condition, solver_steps, frozen_cnn)
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
    print(f"STEP {STEP} — RESULT (Frozen CNN-LPL vs Baseline-LPL)\n")

    for cond in CONDITIONS:
        sp = final_speedup[cond]
        cr_vals = [r['compression_ratio'] for r in all_results
                   if r['condition'] == cond and r['compression_ratio'] is not None]
        cr_str = f"cr_mean={np.mean(cr_vals):.4f}" if cr_vals else "cr=N/A"
        print(f"  {LABELS[cond]}: speedup={format_speedup(sp)}  {cr_str}")
    print()

    print("Compression detail (try1):")
    for r in all_results:
        label = GAME_LABELS.get(r['game'], '?')
        cond  = LABELS[r['condition']]
        cr    = r['compression_ratio']
        traj  = r['try1']['pred_loss_traj']
        print(f"  {label}/{cond}: cr={cr}  traj={traj}")
    print()

    cnn_cr_vals  = [r['compression_ratio'] for r in all_results if r['condition'] == 'cnn_lpl'
                    and r['compression_ratio'] is not None]
    base_cr_vals = [r['compression_ratio'] for r in all_results if r['condition'] == 'baseline_lpl'
                    and r['compression_ratio'] is not None]
    cnn_cr_mean  = float(np.mean(cnn_cr_vals))  if cnn_cr_vals  else None
    base_cr_mean = float(np.mean(base_cr_vals)) if base_cr_vals else None

    print("KILL/SIGNAL ASSESSMENT:")
    if cnn_cr_mean is None:
        print(f"  >>> INCONCLUSIVE: CNN-LPL cr=N/A (no ARC L1 signal). Try larger game set.")
    elif cnn_cr_mean is not None and base_cr_mean is not None and cnn_cr_mean >= 0.9 and base_cr_mean >= 0.9:
        print(f"  >>> KILL: CNN-LPL cr={cnn_cr_mean:.4f} ≈ BASELINE cr={base_cr_mean:.4f}")
        print(f"  >>> Both ≥ 0.9 — encoding quality is NOT the bottleneck. Update rule is.")
    elif cnn_cr_mean is not None and cnn_cr_mean < 0.5:
        print(f"  >>> SIGNAL: CNN-LPL cr={cnn_cr_mean:.4f} < 0.5")
        if base_cr_mean is not None:
            print(f"  >>> vs BASELINE cr={base_cr_mean:.4f} — encoding IS the bottleneck.")
        print(f"  >>> Next: make the encoder self-modifying (R3-compliant).")
    else:
        print(f"  >>> PARTIAL: CNN-LPL cr={cnn_cr_mean}  BASELINE cr={base_cr_mean}")
        print(f"  >>> Neither clear kill nor clear signal.")
    print(sep)


if __name__ == '__main__':
    main()
