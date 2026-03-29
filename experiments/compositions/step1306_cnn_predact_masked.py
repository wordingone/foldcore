"""
Step 1306 — Close the prediction→action loop (predicted frame change as action score)
Leo mail 3682, 2026-03-29.

World model confirmed in Step 1305 (98% prediction improvement, wdrift=11.5).
1306 makes the substrate USE the world model to select actions.

Mechanism: for each candidate action a, compute predicted_delta = ||predict_next(avg_features, a_vec) - avg_features||.
Select action with HIGHEST predicted_delta. Seek the state predicted to change the encoding the most.
R1-compliant: self-generated criteria, no environment reward.

Conditions:
  SELFSUP-ACT: CNN + self-supervised forward prediction + predicted-delta action selection (new)
  SELFSUP-ENT: CNN + self-supervised forward prediction + entropy-driven action selection (1305 baseline)

Kill criteria (chain-level, masked):
  1. SELFSUP-ACT RHAE ≤ SELFSUP-ENT RHAE → predicted frame change doesn't help → KILL
  2. SELFSUP-ACT compression_ratio > 2× SELFSUP-ENT → action selection hurts learning → KILL
  3. action_KL(ACT) < 0.01 → predicted_delta concentrates on one action → collapsed → KILL

Protocol: random seed 1306, masked PRISM (Game A/B/C), 3 draws × 2 conditions = 18 runs, 10K steps, 5-min cap.
INFO logging suppressed in stdout (arcagi3 game class loading).
"""
import sys, os, time, json, random, logging
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

# Suppress arcagi3 INFO logs (game class loading spam)
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from collections import deque
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from substrates.step0674 import _enc_frame

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Random game selection from solved pool (seed 1306) ---
SOLVED_POOL = ['ft09', 'ls20', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
random.seed(1306)
FILTER_GAMES = sorted(random.sample(SOLVED_POOL, 3))
GAMES = FILTER_GAMES

# Masked labels: stdout never shows real game IDs
GAME_LABELS = {game: label for game, label in zip(GAMES, ['Game A', 'Game B', 'Game C'])}

N_DRAWS = 3
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64

# SG hyperparameters
_BUFFER_MAXLEN = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
_LR = 0.0001
_ACTION_ENTROPY_COEF = 0.0001
_COORD_ENTROPY_COEF = 0.00001
_SG_OUTPUT_DIM = 4101  # 5 discrete + 4096 click

N_KEYBOARD = 7   # PRISM keyboard offset for click actions

# SELFSUP-ACT: evaluate these many click positions per step
_N_CLICK_CANDIDATES = 64

# Prediction loss trajectory checkpoints
LOSS_CHECKPOINTS = [1000, 3000, 5000, 7000, 9000]

# R3 measurement (for reference, not used for CNN kill)
R3_STEP = 5000
R3_N_OBS = 30
R3_N_DIRS = 10
R3_EPSILON = 0.01

CONDITIONS = ['selfsup_act', 'selfsup_ent']
LABELS = {'selfsup_act': 'SELFSUP-ACT', 'selfsup_ent': 'SELFSUP-ENT'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1306')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

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
# CNN model
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
        """avg_features: (batch, 256), action_type_vec: (batch, 6) → (batch, 256)"""
        inp = torch.cat([avg_features, action_type_vec], dim=1)
        return self.pred_head(inp)

    def batch_predict_delta(self, avg_features_single, candidate_action_vecs):
        """Compute predicted_delta for multiple candidate actions.

        avg_features_single: (1, 256) — current features
        candidate_action_vecs: (n_candidates, 6)
        Returns: (n_candidates,) tensor of ||predicted_next - current||
        """
        n = candidate_action_vecs.shape[0]
        af_expanded = avg_features_single.expand(n, -1)  # (n, 256)
        predicted_next = self.predict_next(af_expanded, candidate_action_vecs)  # (n, 256)
        deltas = (predicted_next - avg_features_single).norm(dim=1)  # (n,)
        return deltas


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
# Shared substrate base
# ---------------------------------------------------------------------------

class SgSelfSupBase:
    """Shared CNN + forward prediction. Action selection differs by subclass."""

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

        # Episode-start snapshot: stored at creation, never reset
        self._episode_start_state_dict = {k: v.clone().cpu() for k, v in
                                           self._model.state_dict().items()}

        # Prediction loss trajectory
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

        # Current avg_features (set during process(), used for action selection)
        self._current_avg_features = None

    def _select_action(self, logits, avg_features_tensor):
        """Override in subclass for different action selection strategies."""
        raise NotImplementedError

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
            logits, avg_features = self._model(tensor)
            logits = logits.squeeze(0).cpu()
            self._current_avg_features = avg_features  # (1, 256) on device

        sg_idx = self._select_action(logits, self._current_avg_features)
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
        # Never reset model or episode_start_state_dict
        self._prev_one_hot = None
        self._prev_sg_idx = None
        self._current_avg_features = None

    def get_collapse_fraction(self):
        if self.step == 0:
            return 0.0
        return float(self._action_counts.max() / self.step)

    def compute_weight_drift(self):
        drift = 0.0
        for name, param in self._model.named_parameters():
            init_val = self._episode_start_state_dict[name]
            drift += (param.data.cpu() - init_val.cpu()).norm().item()
        return float(drift)

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

        loss_val = float(pred_loss.item())
        self._recent_losses.append(loss_val)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(list(self._recent_losses))), 6)


# ---------------------------------------------------------------------------
# SELFSUP-ENT: entropy-driven action selection (1305 baseline)
# ---------------------------------------------------------------------------

class SelfSupEntSubstrate(SgSelfSupBase):

    def _select_action(self, logits, avg_features_tensor):
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


# ---------------------------------------------------------------------------
# SELFSUP-ACT: predicted-delta action selection
# ---------------------------------------------------------------------------

class SelfSupActSubstrate(SgSelfSupBase):
    """Same world model as ENT. Action selection = argmax predicted_delta.

    Candidates: all 5 discrete action types + 64 random click positions = 69.
    For KB-only games (n_actions ≤ 7): only 5 or fewer discrete candidates.
    """

    def _select_action(self, logits, avg_features_tensor):
        if avg_features_tensor is None:
            # Fallback before first observation
            return int(self._rng.randint(_SG_OUTPUT_DIM))

        # Build candidate action vectors
        candidate_sg_indices = []
        candidate_vecs = []

        # All 5 discrete action types
        for i in range(5):
            candidate_sg_indices.append(i)
            vec = np.zeros(6, dtype=np.float32)
            vec[i] = 1.0
            candidate_vecs.append(vec)

        # 64 random click positions (SG indices 5 to 5+4095)
        n_clicks = min(_N_CLICK_CANDIDATES, 4096)
        click_indices = self._rng.choice(4096, n_clicks, replace=False)
        for ci in click_indices:
            sg_idx = 5 + int(ci)
            candidate_sg_indices.append(sg_idx)
            vec = np.zeros(6, dtype=np.float32)
            vec[5] = 1.0  # all clicks share action type 5
            candidate_vecs.append(vec)

        candidate_vecs_tensor = torch.from_numpy(
            np.stack(candidate_vecs)
        ).to(_DEVICE)  # (n_candidates, 6)

        with torch.no_grad():
            deltas = self._model.batch_predict_delta(avg_features_tensor, candidate_vecs_tensor)

        best_idx = int(deltas.argmax().item())
        return candidate_sg_indices[best_idx]


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

def run_episode(env, substrate, condition, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
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

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

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
    collapse_frac = substrate.get_collapse_fraction()
    wdrift = substrate.compute_weight_drift()

    pred_loss_traj = substrate.get_pred_loss_trajectory()
    compression_ratio = None
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
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'collapse_fraction': round(collapse_frac, 4),
        'wdrift': round(wdrift, 4),
        'pred_loss_traj': pred_loss_traj,
        'compression_ratio': compression_ratio,
    }


def make_substrate(condition, n_actions, seed):
    if condition == 'selfsup_act':
        return SelfSupActSubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'selfsup_ent':
        return SelfSupEntSubstrate(n_actions=n_actions, seed=seed)
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

    substrate = make_substrate(condition, n_actions, seed_a)

    result_a = run_episode(env, substrate, condition, n_actions, solver_level_steps, seed=seed_a)
    result_b = run_episode(env, substrate, condition, n_actions, solver_level_steps, seed=seed_b)

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
    print(f"Step 1306 — Close the prediction→action loop (predicted frame change as action score)")
    print(f"Device: {_DEVICE}")
    print(f"Games: {list(GAME_LABELS.values())} (real IDs masked in stdout)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Draws: {N_DRAWS} per game per condition")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
    print(f"SELFSUP-ACT: 5 discrete + {_N_CLICK_CANDIDATES} random clicks = {5+_N_CLICK_CANDIDATES} candidates/step")
    print()

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

    chain = {c: {
        'rhae_vals': [], 'wdrift_vals': [], 'action_kl_vals': [],
        'i3cv_vals': [], 'runtimes': [], 'compression_ratios': [],
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
            act = draw_results.get('selfsup_act', {}).get('episode_A', {})
            ent = draw_results.get('selfsup_ent', {}).get('episode_A', {})
            print(f"  {label} draw {draw}: "
                  f"ACT[wdrift={act.get('wdrift')},kl={act.get('action_kl')},RHAE={act.get('RHAE',0):.2e},cr={act.get('compression_ratio')}] "
                  f"ENT[wdrift={ent.get('wdrift')},kl={ent.get('action_kl')},RHAE={ent.get('RHAE',0):.2e},cr={ent.get('compression_ratio')}]  ({elapsed:.1f}s)")

        # Save raw per-game JSONL
        out_path = os.path.join(RESULTS_DIR, f"{game}_1306.jsonl")
        game_results = [r for r in all_results if r['game'] == game]
        with open(out_path, 'w') as f:
            for r in game_results:
                f.write(json.dumps(r) + '\n')
        print()

    # ---------------------------------------------------------------------------
    # Chain-level summary
    # ---------------------------------------------------------------------------
    print("\n" + "="*100)
    print("STEP 1306 — CHAIN-LEVEL SUMMARY (aggregate across all games)")
    print()

    chain_summary = {}
    for condition in CONDITIONS:
        d = chain[condition]
        mean_rhae = round(float(np.mean(d['rhae_vals'])), 6) if d['rhae_vals'] else 0.0
        mean_wdrift = round(float(np.mean(d['wdrift_vals'])), 4) if d['wdrift_vals'] else None
        mean_action_kl = round(float(np.mean(d['action_kl_vals'])), 4) if d['action_kl_vals'] else None
        mean_i3cv = round(float(np.mean(d['i3cv_vals'])), 4) if d['i3cv_vals'] else None
        mean_cr = round(float(np.mean(d['compression_ratios'])), 4) if d['compression_ratios'] else None

        loss_traj_summary = {}
        for ck in LOSS_CHECKPOINTS:
            vals = d['pred_loss_by_ck'][ck]
            loss_traj_summary[ck] = round(float(np.mean(vals)), 6) if vals else None

        chain_summary[condition] = {
            'mean_RHAE': mean_rhae,
            'mean_wdrift': mean_wdrift,
            'mean_action_KL': mean_action_kl,
            'mean_I3cv': mean_i3cv,
            'mean_compression_ratio': mean_cr,
            'pred_loss_trajectory': loss_traj_summary,
        }

        print(f"  {LABELS[condition]}:")
        print(f"    mean_RHAE={mean_rhae:.2e}  mean_wdrift={mean_wdrift}  mean_action_KL={mean_action_kl}  mean_I3cv={mean_i3cv}")
        print(f"    pred_loss_trajectory: {loss_traj_summary}")
        print(f"    mean_compression_ratio (loss@9K/loss@1K): {mean_cr}")

    print()

    # ---------------------------------------------------------------------------
    # Kill assessment
    # ---------------------------------------------------------------------------
    print("KILL ASSESSMENT (chain aggregates only):")
    act = chain_summary.get('selfsup_act', {})
    ent = chain_summary.get('selfsup_ent', {})

    act_rhae = act.get('mean_RHAE', 0.0)
    ent_rhae = ent.get('mean_RHAE', 0.0)
    act_cr = act.get('mean_compression_ratio')
    ent_cr = ent.get('mean_compression_ratio')
    act_kl = act.get('mean_action_KL')

    killed = False

    # Kill 1: ACT RHAE ≤ ENT RHAE
    if act_rhae <= ent_rhae:
        print(f"  >>> KILL: SELFSUP-ACT RHAE={act_rhae:.2e} ≤ SELFSUP-ENT RHAE={ent_rhae:.2e} → predicted frame change doesn't help")
        killed = True
    else:
        print(f"  >>> SIGNAL: SELFSUP-ACT RHAE={act_rhae:.2e} > SELFSUP-ENT RHAE={ent_rhae:.2e} → prediction-guided selection helps level progress")

    # Kill 2: ACT compression_ratio > 2× ENT
    if act_cr is not None and ent_cr is not None:
        if act_cr > 2 * ent_cr:
            print(f"  >>> KILL: ACT compression_ratio={act_cr} > 2× ENT={ent_cr} → action selection hurts learning")
            killed = True
        else:
            print(f"  >>> OK: ACT compression_ratio={act_cr} not > 2× ENT={ent_cr}")

    # Kill 3: action_KL(ACT) < 0.01
    if act_kl is not None and act_kl < 0.01:
        print(f"  >>> KILL: ACT action_KL={act_kl} < 0.01 → predicted_delta collapsed to one action")
        killed = True
    elif act_kl is not None:
        print(f"  >>> OK: ACT action_KL={act_kl} ≥ 0.01 → no collapse")

    if not killed:
        print("  >>> NO KILL triggered — prediction→action loop showing signal")

    print("="*100)

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
