"""
Step 1338 — Mode-TP: mode map (C23) as functional encoder for MLP + TP.
Leo mail 3797, 2026-03-29.

TP compresses perceptual sequences (what follows what) but doesn't encode
functional structure (what actions DO). Mode map discovers which obs elements
change together in response to actions — functional categorization.

zone_features (160-dim) = mode map output → MLP+TP input (replaces raw obs).
Persists try1→try2: zones discovered in try1 pre-seed try2 exploration.

Conditions:
  MODE-TP: Mode map + MLP (zone_features input) + TP. MBPP + 2 masked ARC.
  MLP-TP:  1337 architecture (raw obs input, no mode map). Same games. Control.

Primary metric: RHAE(try2) = mean(efficiency²) across all 3 games.
Kill:     MODE-TP RHAE ≤ MLP-TP RHAE → zone features don't help → KILL
Signal:   MODE-TP RHAE > 0 → functional categorization produces progress
Landmark: MODE-TP MBPP efficiency > 0 → first text progress

Additional diagnostic: zone_count + zone_stability (do try2 zones appear faster?).
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

GAMES, GAME_LABELS = select_games(seed=1338)

STEP        = 1338
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

CONDITIONS  = ['mode_tp', 'mlp_tp']
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1338')

_DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HIDDEN     = 512
_LR         = 1e-4
_TP_LR      = 0.01
_BUFFER_MAX = 200_000
_TRAIN_FREQ = 5
_BATCH_SIZE = 64
LOSS_CHECKPOINTS = [500, 1000, 2000]
N_KEYBOARD  = 7

# Mode map constants
_MM_UPDATE_FREQ  = 100   # recompute zones every N steps
_MM_CHANGE_THRESH = 0.01 # 1% change rate (calibrated in step1332)
_MM_N_ZONES      = 16
_MM_ZONE_DIMS    = 10    # [active, cx, cy, area, af0..af5]
_MM_ZONE_DIM     = _MM_N_ZONES * _MM_ZONE_DIMS  # 160
_MM_N_ACT_TYPES  = 6


# ---------------------------------------------------------------------------
# Obs encoding helpers
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
        return _obs_to_one_hot(arr).astype(np.float32).flatten()  # 65536
    return arr.flatten()  # MBPP: 256


def _action_type_vec(action, n_types=6):
    vec = np.zeros(6, dtype=np.float32)
    vec[int(action) % n_types] = 1.0
    return vec


def _sg_to_action_type_vec(sg_idx):
    vec = np.zeros(6, dtype=np.float32)
    vec[sg_idx if sg_idx < 5 else 5] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Mode map: frame differencing → CC zone discovery → 160-dim zone_features
# ---------------------------------------------------------------------------

class ModeMap:
    """Action-conditional frame differencing → connected component zones.

    Tracks per-action-type obs change frequency (ARC: 2D 64×64; MBPP: 1D 256).
    Every UPDATE_FREQ steps: CC isolation → N_ZONES × ZONE_DIMS zone_features.

    Zone features (160 dims):
        [active, cx_norm, cy_norm, area_norm, af0, af1, af2, af3, af4, af5] × 16
    cx/cy: normalized centroid [0,1]. cy=0 for 1D obs. af_i: action-type i affinity.

    Persists across try1→try2: accumulated change maps carry over.
    Only step counter + prev_obs reset (not zone data) when reset_for_try2() called.
    """

    def __init__(self):
        self._obs_type   = None   # '2d' or '1d', set on first update
        self._act_change = None   # (N_ACT_TYPES, h, w) or (N_ACT_TYPES, flat_dim)
        self._act_count  = np.zeros(_MM_N_ACT_TYPES, dtype=np.float32)
        self._prev_obs   = None
        self._step       = 0
        self._zone_feats = np.zeros(_MM_ZONE_DIM, dtype=np.float32)
        self._n_zones    = 0
        self._zone_stab  = []  # zone counts per recompute (for stability tracking)

    def update(self, obs_arr, action):
        """Update with new (obs, action). Call after each env.step()."""
        obs_arr = np.asarray(obs_arr, dtype=np.float32)
        at = int(action) % _MM_N_ACT_TYPES

        if self._obs_type is None:
            if _is_arc_obs(obs_arr):
                self._obs_type  = '2d'
                self._act_change = np.zeros((_MM_N_ACT_TYPES, 64, 64), dtype=np.float32)
                self._prev_obs   = obs_arr[0].copy()
            else:
                self._obs_type  = '1d'
                flat_dim = obs_arr.flatten().shape[0]
                self._act_change = np.zeros((_MM_N_ACT_TYPES, flat_dim), dtype=np.float32)
                self._prev_obs   = obs_arr.flatten().copy()

        self._act_count[at] += 1

        if self._prev_obs is not None:
            if self._obs_type == '2d':
                cur = obs_arr[0]
                diff = (np.abs(cur - self._prev_obs) > 0.5).astype(np.float32)
                self._act_change[at] += diff
                self._prev_obs = cur.copy()
            else:
                cur = obs_arr.flatten()
                diff = (np.abs(cur - self._prev_obs) > 1e-4).astype(np.float32)
                self._act_change[at] += diff
                self._prev_obs = cur.copy()

        self._step += 1
        if self._step % _MM_UPDATE_FREQ == 0:
            self._recompute_zones()

    def _recompute_zones(self):
        try:
            from scipy.ndimage import label as nd_label
        except ImportError:
            return

        if self._act_change is None:
            return

        # Total normalized change rate
        total = np.zeros_like(self._act_change[0])
        for at in range(_MM_N_ACT_TYPES):
            cnt = max(float(self._act_count[at]), 1.0)
            total += self._act_change[at] / cnt
        total /= _MM_N_ACT_TYPES

        active_mask = (total > _MM_CHANGE_THRESH).astype(np.uint8)
        if active_mask.sum() == 0:
            self._zone_feats[:] = 0
            self._n_zones = 0
            self._zone_stab.append(0)
            return

        labeled, n_zones_raw = nd_label(active_mask)
        n_zones = min(n_zones_raw, _MM_N_ZONES)
        feat = np.zeros(_MM_ZONE_DIM, dtype=np.float32)
        found = 0

        if self._obs_type == '2d':
            h, w = 64, 64
        else:
            h, w = self._act_change.shape[1], 1

        for z in range(1, n_zones_raw + 1):
            if found >= _MM_N_ZONES:
                break
            zone_mask = (labeled == z)
            if zone_mask.sum() == 0:
                continue

            if self._obs_type == '2d':
                rows, cols = np.where(zone_mask)
                cx = float(cols.mean()) / max(w - 1, 1)
                cy = float(rows.mean()) / max(h - 1, 1)
                area = float(zone_mask.sum()) / (h * w)
            else:
                indices = np.where(zone_mask)[0]
                flat_dim = self._act_change.shape[1]
                cx = float(indices.mean()) / max(flat_dim - 1, 1)
                cy = 0.0
                area = float(len(indices)) / max(flat_dim, 1)

            # Per-action affinity (normalized)
            affinities = np.zeros(_MM_N_ACT_TYPES, dtype=np.float32)
            for at in range(_MM_N_ACT_TYPES):
                cnt = max(float(self._act_count[at]), 1.0)
                affinities[at] = float(self._act_change[at][zone_mask].mean()) / cnt
            aff_sum = affinities.sum()
            if aff_sum > 1e-8:
                affinities /= aff_sum

            base = found * _MM_ZONE_DIMS
            feat[base + 0] = 1.0
            feat[base + 1] = cx
            feat[base + 2] = cy
            feat[base + 3] = area
            feat[base + 4:base + 10] = affinities
            found += 1

        self._zone_feats = feat
        self._n_zones = found
        self._zone_stab.append(found)

    def get_features(self):
        return self._zone_feats.copy()

    def n_zones_found(self):
        return self._n_zones

    def zone_stability(self):
        """Return zone count trajectory (how quickly zones stabilized)."""
        return list(self._zone_stab)

    def reset_for_try2(self):
        """Keep accumulated zone data. Reset step counter + prev_obs only."""
        self._step     = 0
        self._prev_obs = None


# ---------------------------------------------------------------------------
# Shared MLP model (used by both MODE-TP and MLP-TP)
# ---------------------------------------------------------------------------

class MlpSelfSupModel(nn.Module):
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
        h2 = F.relu(self.fc2(h1)); h2 = self.dropout(h2)
        h3 = F.relu(self.fc3(h2))
        return self.action_head(h3), h3

    def forward_all_layers(self, x):
        h0 = F.relu(self.input_proj(x))
        h1 = F.relu(self.fc1(h0))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        return h3, h2, h1, h0


def _make_mlp_g_and_opts(model, hidden=_HIDDEN, lr=_LR):
    g = nn.ModuleDict({
        'g3': nn.Linear(hidden, hidden).to(_DEVICE),
        'g2': nn.Linear(hidden, hidden).to(_DEVICE),
        'g1': nn.Linear(hidden, hidden).to(_DEVICE),
    })
    opt_pred = torch.optim.Adam(model.pred_head.parameters(), lr=lr)
    opt_g    = {k: torch.optim.Adam(v.parameters(), lr=lr) for k, v in g.items()}
    opt_f    = {
        'f_proj': torch.optim.Adam(model.input_proj.parameters(), lr=lr),
        'f1':     torch.optim.Adam(model.fc1.parameters(), lr=lr),
        'f2':     torch.optim.Adam(model.fc2.parameters(), lr=lr),
        'f3':     torch.optim.Adam(model.fc3.parameters(), lr=lr),
    }
    return g, opt_pred, opt_g, opt_f


def _tp_train_step_mlp(model, g, opt_pred, opt_g, opt_f, buffer, rng):
    """Shared TP train step for MLP substrates."""
    n   = len(buffer)
    buf = list(buffer)
    idx = rng.choice(n, min(_BATCH_SIZE, n), replace=False)
    batch = [buf[i] for i in idx]

    states      = torch.from_numpy(np.stack([b['state'].astype(np.float32) for b in batch])).to(_DEVICE)
    next_states = torch.from_numpy(np.stack([b['next_state'].astype(np.float32) for b in batch])).to(_DEVICE)
    action_vecs = torch.from_numpy(np.stack([b['action_type_vec'] for b in batch])).to(_DEVICE)

    with torch.no_grad():
        h3, h2, h1, h0 = model.forward_all_layers(states)
        _, target_h3_next = model(next_states)
        pred_in   = torch.cat([h3, action_vecs], dim=1)
        pred_next = model.pred_head(pred_in)
        pred_err  = pred_next - target_h3_next
        pred_loss = float((pred_err ** 2).mean())

        target_h3 = h3 - _TP_LR * pred_err
        target_h2 = g['g3'](target_h3)
        target_h1 = g['g2'](target_h2)
        target_h0 = g['g1'](target_h1)

    with torch.enable_grad():
        pred_in2 = torch.cat([h3.detach(), action_vecs.detach()], dim=1)
        loss_pred = F.mse_loss(model.pred_head(pred_in2), target_h3_next.detach())
        opt_pred.zero_grad(); loss_pred.backward(); opt_pred.step()

    for gk, h_in, h_out in [('g3', h3, h2), ('g2', h2, h1), ('g1', h1, h0)]:
        with torch.enable_grad():
            loss_g = F.mse_loss(g[gk](h_in.detach()), h_out.detach())
            opt_g[gk].zero_grad(); loss_g.backward(); opt_g[gk].step()

    for fk, layer, x_in, target in [
        ('f3', model.fc3, h2, target_h3), ('f2', model.fc2, h1, target_h2),
        ('f1', model.fc1, h0, target_h1), ('f_proj', model.input_proj, states, target_h0),
    ]:
        with torch.enable_grad():
            loss_local = F.mse_loss(F.relu(layer(x_in.detach())), target.detach())
            opt_f[fk].zero_grad(); loss_local.backward(); opt_f[fk].step()

    return pred_loss


# ---------------------------------------------------------------------------
# Mode-TP substrate: mode map → zone_features → MLP + TP
# ---------------------------------------------------------------------------

class ModeMapMlpTpSubstrate:
    """Mode map (C23) + MLP-TP. Zone features replace raw obs as MLP input.

    input_dim = 160 (fixed: N_ZONES × ZONE_DIMS).
    Model created in __init__ (no lazy init needed — input_dim is always 160).
    Zone map persists try1→try2.
    """
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._mode_map   = ModeMap()

        self._model = MlpSelfSupModel(_MM_ZONE_DIM, n_actions).to(_DEVICE)
        self._g, self._opt_pred, self._opt_g, self._opt_f = \
            _make_mlp_g_and_opts(self._model)

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_zone_enc = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

        # Zone stability tracking (try1 vs try2)
        self._try1_zone_stab = None
        self._try2_zone_stab = None
        self._in_try2 = False

    def reset_for_try2(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}
        self._try1_zone_stab = self._mode_map.zone_stability().copy()
        self._in_try2 = True
        self._mode_map.reset_for_try2()

    def process(self, obs_arr):
        self._step += 1
        self._prev_zone_enc = self._mode_map.get_features()

        tensor = torch.from_numpy(self._prev_zone_enc).unsqueeze(0).to(_DEVICE)
        with torch.no_grad():
            logits, _ = self._model(tensor)
            logits = logits.squeeze(0).cpu()

        probs  = torch.softmax(logits, dim=-1)
        action = int(torch.multinomial(probs, 1).item())
        return action % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_zone_enc is None:
            return
        # Update mode map with new obs + action (computes frame diff)
        self._mode_map.update(obs_next, action)
        zone_enc_next = self._mode_map.get_features()

        at_vec = _action_type_vec(action)
        h = hashlib.md5(self._prev_zone_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({
                'state':           self._prev_zone_enc.copy(),
                'action_type_vec': at_vec,
                'next_state':      zone_enc_next.copy(),
            })

        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = _tp_train_step_mlp(self._model, self._g, self._opt_pred,
                                       self._opt_g, self._opt_f, self._buffer, self._rng)
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_zone_enc = None

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

    def get_zone_stats(self):
        stab = self._mode_map.zone_stability()
        if self._in_try2:
            try2_stab = stab[len(self._try1_zone_stab):] if self._try1_zone_stab else stab
        else:
            try2_stab = []
        return {
            'n_zones':        self._mode_map.n_zones_found(),
            'try1_zone_stab': self._try1_zone_stab or stab,
            'try2_zone_stab': try2_stab,
        }


# ---------------------------------------------------------------------------
# MLP-TP substrate (control — same as step 1337, lazy init from obs_dim)
# ---------------------------------------------------------------------------

class MlpTpSubstrate:
    def __init__(self, n_actions):
        self.n_actions   = n_actions
        self._rng        = np.random.RandomState(42)
        self._model      = None
        self._g          = None
        self._opt_pred   = None
        self._opt_g      = None
        self._opt_f      = None

        self._buffer        = deque(maxlen=_BUFFER_MAX)
        self._buffer_hashes = set()
        self._prev_enc      = None
        self._train_counter = 0
        self._step          = 0
        self._ep_start_sd   = None
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in LOSS_CHECKPOINTS}

    def _init_model(self, input_dim):
        self._model = MlpSelfSupModel(input_dim, self.n_actions).to(_DEVICE)
        self._g, self._opt_pred, self._opt_g, self._opt_f = \
            _make_mlp_g_and_opts(self._model)
        self._ep_start_sd = {k: v.clone().cpu() for k, v in self._model.state_dict().items()}

    def reset_for_try2(self):
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
        probs  = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1).item()) % self.n_actions

    def update_after_step(self, obs_next, action, reward):
        if self._prev_enc is None or self._model is None:
            return
        enc_next = _encode_obs_mlp(obs_next)
        at_vec   = _action_type_vec(action)
        h = hashlib.md5(self._prev_enc.tobytes() +
                        np.array([action], np.int32).tobytes()).hexdigest()
        if h not in self._buffer_hashes:
            self._buffer_hashes.add(h)
            self._buffer.append({'state': self._prev_enc.copy(),
                                  'action_type_vec': at_vec,
                                  'next_state': enc_next.copy()})
        self._train_counter += 1
        if self._train_counter % _TRAIN_FREQ == 0 and len(self._buffer) >= _BATCH_SIZE:
            loss = _tp_train_step_mlp(self._model, self._g, self._opt_pred,
                                       self._opt_g, self._opt_f, self._buffer, self._rng)
            if loss is not None:
                self._recent_losses.append(loss)
                for ck in self._pred_loss_at:
                    if self._pred_loss_at[ck] is None and self._step >= ck:
                        self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None

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

    def get_zone_stats(self):
        return {'n_zones': None, 'try1_zone_stab': [], 'try2_zone_stab': []}


# ---------------------------------------------------------------------------
# Game factory + optimal_steps
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
        return mbpp_game.compute_solver_steps(problem_idx).get(1)
    return None


# ---------------------------------------------------------------------------
# Episode runner (level-masked)
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

    if condition == 'mode_tp':
        substrate = ModeMapMlpTpSubstrate(n_actions=n_actions)
    else:
        substrate = MlpTpSubstrate(n_actions=n_actions)

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

    zone_stats = substrate.get_zone_stats()

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
        'zone_stats':              zone_stats,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Mode-TP: functional categorization (C23) + MLP + TP")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Device: {_DEVICE}")
    print(f"Mode map: {_MM_N_ZONES} zones × {_MM_ZONE_DIMS} dims = {_MM_ZONE_DIM}-dim features")
    print(f"Conditions: {CONDITIONS}")
    print(f"Try1: {TRY1_STEPS} steps (seed 0), Try2: {TRY2_STEPS} steps (seed {TRY2_SEED})")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Tier 1: timing check
    first_arc = next((g for g in GAMES if not g.lower().startswith('mbpp')), None)
    if first_arc:
        first_label = GAME_LABELS[first_arc]
        print(f"Tier 1: timing on {first_label} (100 steps, MODE-TP)...")
        env_t = make_game(first_arc)
        try:
            na_t = int(env_t.n_actions)
        except AttributeError:
            na_t = 4103
        sub_t = ModeMapMlpTpSubstrate(n_actions=na_t)
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
            print(f"  WARNING: estimated {est_total:.0f}s > 300s cap")
        print()

    all_results = []
    rhae_by_condition    = {c: None for c in CONDITIONS}
    speedup_by_condition = {c: [] for c in CONDITIONS}
    eff_sq_by_game_cond  = {c: {} for c in CONDITIONS}

    for condition in CONDITIONS:
        print(f"=== Condition: {condition.upper()} ===")
        try2_progress = {}
        optimal_steps = {}

        for game_name, label in zip(GAMES, GAME_LABELS.values()):
            t_game = time.time()
            result = run_game(game_name, label, condition)
            elapsed = time.time() - t_game

            all_results.append(result)
            p2      = result['try2']['steps_to_first_progress']
            speedup = result['second_exposure_speedup']
            cr      = result['compression_ratio']
            eff     = result['efficiency_sq_try2']
            zs      = result['zone_stats']
            n_zones = zs.get('n_zones', 0) or 0

            try2_progress[label] = p2
            optimal_steps[label] = result['optimal_steps']
            eff_sq_by_game_cond[condition][label] = eff

            print(f"  {label}: speedup={format_speedup(speedup)}  cr={cr}  eff²={eff}"
                  f"  zones={n_zones}  ({elapsed:.1f}s)")
            speedup_by_condition[condition].append(speedup_for_chain(speedup))

            out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
            with open(out_path, 'a') as f:
                f.write(json.dumps(result, default=str) + '\n')

        rhae = compute_rhae_try2(try2_progress, optimal_steps)
        rhae_by_condition[condition] = rhae
        chain_speedup = sum(speedup_by_condition[condition]) / max(len(speedup_by_condition[condition]), 1)
        print(f"  → RHAE(try2) = {rhae:.4f}  (chain_speedup={chain_speedup:.4f})")
        print()

    # -------------------------------------------------------------------------
    # Report
    # -------------------------------------------------------------------------
    print("=" * 80)
    print(f"STEP {STEP} — RESULT (Mode-TP vs MLP-TP: functional categorization)")
    print()
    print(f"  MODE-TP  RHAE(try2) = {rhae_by_condition['mode_tp']:.4f}")
    print(f"  MLP-TP   RHAE(try2) = {rhae_by_condition['mlp_tp']:.4f}")
    print()

    print("Per-game efficiency²:")
    for label in GAME_LABELS.values():
        mode_eff = eff_sq_by_game_cond['mode_tp'].get(label, 0.0)
        mlp_eff  = eff_sq_by_game_cond['mlp_tp'].get(label, 0.0)
        print(f"  {label}: MODE={mode_eff:.6f}  MLP={mlp_eff:.6f}")
    print()

    print("Zone discovery diagnostic:")
    for result in all_results:
        if result['condition'] == 'mode_tp':
            lbl = result['label']
            zs  = result['zone_stats']
            n_zones = zs.get('n_zones', 0) or 0
            t1_stab = zs.get('try1_zone_stab', [])
            t2_stab = zs.get('try2_zone_stab', [])
            t1_steps_to_first = next((i*_MM_UPDATE_FREQ for i,z in enumerate(t1_stab) if z > 0), None)
            t2_steps_to_first = next((i*_MM_UPDATE_FREQ for i,z in enumerate(t2_stab) if z > 0), None)
            faster = (t2_steps_to_first is not None and t1_steps_to_first is not None
                      and t2_steps_to_first < t1_steps_to_first)
            print(f"  {lbl}: n_zones={n_zones}  t1_stab={t1_stab[:4]}  t2_stab={t2_stab[:4]}"
                  f"  {'FASTER-DISCOVERY' if faster else ''}")
    print()

    mode_rhae = rhae_by_condition['mode_tp']
    mlp_rhae  = rhae_by_condition['mlp_tp']
    mode_mbpp_eff = eff_sq_by_game_cond['mode_tp'].get('MBPP', 0.0)
    mode_mbpp_cr  = next((r['compression_ratio'] for r in all_results
                          if r['condition'] == 'mode_tp' and r['label'] == 'MBPP'), None)

    print("KILL/SIGNAL ASSESSMENT:")
    if mode_mbpp_eff > 0:
        print(f"  >>> LANDMARK: MODE-TP MBPP efficiency² = {mode_mbpp_eff:.6f} — FIRST TEXT PROGRESS.")
    elif mode_rhae > 0:
        print(f"  >>> SIGNAL: MODE-TP RHAE(try2) = {mode_rhae:.4f} > 0.")
    elif mode_rhae > mlp_rhae:
        print(f"  >>> CONTINUE: MODE-TP RHAE={mode_rhae:.4f} > MLP-TP RHAE={mlp_rhae:.4f}. Marginal improvement.")
    else:
        print(f"  >>> KILL: MODE-TP RHAE={mode_rhae:.4f} ≤ MLP-TP RHAE={mlp_rhae:.4f}. Zone features don't help.")
    print("=" * 80)

    chain_speedups = {c: round(sum(speedup_by_condition[c]) / max(len(speedup_by_condition[c]), 1), 4)
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
