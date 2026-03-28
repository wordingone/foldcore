"""
Step 1264 — LPL dynamics substrate: no selector, dynamics produce action

Remove the action selector entirely. W_action IS both encoding and selector.
Action falls out of the dynamics.

  enc = centered_obs               # (ENC_DIM,) — C4 centering
  h = tanh(W_h @ h + W_in @ enc)  # (H_DIM,) — C26 recurrent
  ext = [enc, h]                   # (EXT_DIM=320,)
  activation = W_action @ ext      # (n_actions,)
  activation[last_action] *= refractory_decay  # 0.8 — prevents locking
  action = argmax(activation)

  # LPL update on W_action (Hebbian + predictive, same rule as 1252)
  delta_hebb = eta * outer(activation, ext) - eta * (activation**2)[:,None] * W_action
  delta_pred = eta_p * outer(activation - prev_activation, ext)
  W_action += delta_hebb + delta_pred

No argmin. No GPR. No visit counts. No salience. No attention. No threshold.
W_action IS the encoding AND the action selector. LPL modifies W_action.
Refractory decay creates natural cycling (coverage).
LPL predictive term creates temporal structure (focus).

R3: W_action evolved by LPL -> Jacobian diff vs initial random W_action.
SAL: Spearman(per-action mean activation over all steps, per-action mean state-change).
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAMES = ['ls20', 'vc33', 'sp80']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

# Shared hyperparameters
ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM   # 320

# LPL hyperparameters
ETA_HEBB = 0.01       # Hebbian + Oja rate
ETA_PRED = 0.005      # predictive rate
REFRACTORY_DECAY = 0.8

# Instrumentation
I3_STEP = 200
I1_STEP = 1000
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50
ATTN_SNAPSHOT_STEPS = [1000, 5000]

DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1264')

# ─── Hash ───
_HASH_H = np.random.RandomState(42).randn(12, ENC_DIM).astype(np.float32)

def hash_enc(x: np.ndarray) -> int:
    bits = (_HASH_H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)


def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_n_actions(env, game_name: str) -> int:
    n = getattr(env, 'n_actions', None)
    if n is not None:
        return int(n)
    return {'ls20': 7, 'vc33': 4103, 'sp80': 4103, 'ft09': 4103}.get(game_name.lower(), 4103)


def load_game_diag(game_name: str) -> dict:
    with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
        return json.load(f)


def build_kb_profile(diag: dict) -> tuple:
    kb = diag.get('kb_responsiveness', {})
    delta = np.zeros(7, np.float32)
    responsive = np.zeros(7, bool)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            delta[i] = kb[key].get('delta_mean', 0.0)
            responsive[i] = kb[key].get('responsive', False)
    return delta, responsive


def action_entropy(action_seq: list, n_actions: int) -> float:
    if not action_seq:
        return 0.0
    counts = np.zeros(n_actions, np.float32)
    for a in action_seq:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, dtype=np.float32)
    between = np.array(between_dists, dtype=np.float32)
    if len(within) == 0 or len(between) == 0:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(all_dists))
        if float(np.mean(all_dists[perm[n_w:]]) - np.mean(all_dists[perm[:n_w]])) >= observed:
            count += 1
    return count / n_perms


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


# ─────────────────────────────────────────────────────────────
# SUBSTRATES
# ─────────────────────────────────────────────────────────────

class ControlC:
    """C4 + C14 only. Argmin baseline."""

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.action_counts = np.zeros(n_actions, np.float32)
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_running_mean):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        return x - frozen_running_mean

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = self._centered_encode(obs)
        self._last_enc = enc.copy()
        action = int(np.argmin(self.action_counts))
        self.action_counts[action] += 1
        self.step += 1
        return action

    def on_level_transition(self):
        pass

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.action_counts[:] = 0
        self._last_enc = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'step': self.step,
        }


class LPLSubstrate:
    """
    LPL dynamics substrate: W_action IS encoding AND selector.

    Architecture:
      enc (C4) -> h=tanh(W_h@h+W_in@enc) (C26) -> ext=[enc,h]
      activation = W_action @ ext
      activation[last_action] *= refractory_decay
      action = argmax(activation)
      LPL: W_action += delta_hebb (Oja) + delta_pred

    No argmin. No visit counts. No attention. No selector.
    R3 repr: activation (n_actions,) — changes as W_action evolves.
    SAL: per-action mean activation vs per-action mean state-change.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        rng = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed + 10000)

        # C4: Running mean centering
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # C26: Recurrent h
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # W_action: action = encoding dynamics (LPL-learned)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        W_action_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()  # for R3 fresh baseline

        # SAL tracking: per-action mean activation over ALL steps
        self._sal_sum = np.zeros(n_actions, np.float64)
        self._sal_steps = 0  # total steps (same denominator for all actions)

        # State-change tracking (per chosen action)
        self._delta_sum = np.zeros(n_actions, np.float64)
        self._delta_count = np.zeros(n_actions, np.int64)

        # LPL history
        self._prev_activation = np.zeros(n_actions, np.float32)
        self._prev_ext = None

        # Refractory
        self._last_action = None

        # Attention snapshots (activation stats)
        self._attn_snapshots = {}

        # History for repr log
        self._prev_repr = None

        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action) -> np.ndarray:
        """R3 read-only: activation with frozen state."""
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = _enc_frame(obs) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext  # (n_actions,) activation

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)

        # C4: Centered encoding
        enc = self._centered_encode(obs)

        # C26: Recurrent h
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])  # (EXT_DIM,)

        # Activation = encoding dynamics
        activation = self.W_action @ ext  # (n_actions,)

        # Refractory: decay last action's activation
        if self._last_action is not None:
            activation[self._last_action] *= REFRACTORY_DECAY

        # Action falls out of dynamics
        action = int(np.argmax(activation))

        # LPL update on W_action
        # Hebbian + Oja regularization
        delta_hebb = ETA_HEBB * np.outer(activation, ext)
        delta_oja = ETA_HEBB * (activation ** 2)[:, np.newaxis] * self.W_action
        # Predictive term
        act_diff = activation - self._prev_activation
        delta_pred = ETA_PRED * np.outer(act_diff, ext)
        self.W_action += delta_hebb - delta_oja + delta_pred

        # SAL: accumulate activation for all actions
        self._sal_sum += activation.astype(np.float64)
        self._sal_steps += 1

        # Attention snapshots (activation stats)
        if self.step in ATTN_SNAPSHOT_STEPS:
            self._attn_snapshots[self.step] = {
                'act_max': float(activation.max()),
                'act_mean': float(activation.mean()),
                'act_std': float(activation.std()),
            }

        # Store repr for I1
        self._prev_repr = activation.copy()

        # Update state
        self._prev_activation = activation.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self.step += 1

        return action

    def record_state_change(self, action: int, delta: float):
        if 0 <= action < self.n_actions:
            self._delta_sum[action] += delta
            self._delta_count[action] += 1

    def on_level_transition(self):
        self._prev_activation[:] = 0
        self._prev_ext = None
        # Keep h, W_action — learning persists across levels

    def reset(self, seed: int):
        self.running_mean[:] = 0
        self.n_obs = 0
        self.h[:] = 0
        self.W_action[:] = self.W_action_init  # reset to initial random
        self._sal_sum[:] = 0
        self._sal_steps = 0
        self._delta_sum[:] = 0
        self._delta_count[:] = 0
        self._prev_activation[:] = 0
        self._prev_ext = None
        self._last_action = None
        self._attn_snapshots = {}
        self._prev_repr = None
        self.step = 0

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }

    def set_state(self, state):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.h = state['h'].copy()
        self.W_action = state['W_action'].copy()
        self.step = state['step']


# ─────────────────────────────────────────────────────────────
# INSTRUMENTATION
# ─────────────────────────────────────────────────────────────

def compute_i3(action_counts_200, kb_delta):
    n = min(len(action_counts_200), 7)
    if n < 2:
        return None, None
    freq = action_counts_200[:n].astype(np.float32)
    ref = kb_delta[:n].astype(np.float32)
    if ref.max() - ref.min() < 1e-6:
        return None, None
    rho = spearman_rho(freq, ref)
    if rho is None:
        return None, None
    return rho, bool(rho > 0.5)


def compute_i1(repr_level_log):
    if len(repr_level_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    rng = np.random.RandomState(1)
    n = len(repr_level_log)
    within_dists, between_dists = [], []
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_level_log[i]
        r2, l2 = repr_level_log[j]
        d = cosine_dist(r1, r2)
        (within_dists if l1 == l2 else between_dists).append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists)
    wm = float(np.mean(within_dists))
    bm = float(np.mean(between_dists))
    return {'within': round(wm, 4), 'between': round(bm, 4),
            'p_value': round(p_val, 4), 'pass': bool(wm < bm and p_val < 0.05)}


def compute_i4(action_log, n_actions):
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        return action_entropy(window, n_actions) if len(window) >= 5 else None
    h100 = entropy_at(100)
    h5000 = entropy_at(5000)
    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}
    reduction_pct = float((h100 - h5000) / h100 * 100)
    return {'entropy_100': round(h100, 4), 'entropy_5000': round(h5000, 4),
            'reduction_pct': round(reduction_pct, 2), 'pass': bool(reduction_pct > 10.0)}


def compute_r3_jacobian(substrate, obs_sample, frozen_state, n_actions, rng_seed=42):
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    _is_lpl = isinstance(substrate, LPLSubstrate)

    if _is_lpl:
        frozen_rm = frozen_state['running_mean']
        frozen_h = frozen_state['h']
        frozen_W_action = frozen_state['W_action']
        fresh_rm = np.zeros(ENC_DIM, np.float32)
        fresh_h = np.zeros(H_DIM, np.float32)
        fresh_W_action = substrate.W_action_init  # initial random
    else:
        frozen_rm = frozen_state['running_mean']
        fresh_rm = np.zeros(ENC_DIM, np.float32)

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, dtype=np.float32)

        if _is_lpl:
            baseline_exp = substrate.get_internal_repr_readonly(
                obs_flat, frozen_rm, frozen_h, frozen_W_action)
        else:
            baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)

        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            if _is_lpl:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, frozen_rm, frozen_h, frozen_W_action)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, frozen_rm)
            J_exp_cols.append((pert - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)

        if _is_lpl:
            baseline_fresh = substrate.get_internal_repr_readonly(
                obs_flat, fresh_rm, fresh_h, fresh_W_action)
        else:
            baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm)

        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            if _is_lpl:
                pert = substrate.get_internal_repr_readonly(
                    perturbed, fresh_rm, fresh_h, fresh_W_action)
            else:
                pert = substrate.get_internal_repr_readonly(perturbed, fresh_rm)
            J_fresh_cols.append((pert - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diffs.append(float(np.linalg.norm(J_exp - J_fresh)))

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    jd = float(np.mean(diffs))
    return {'jacobian_diff': round(jd, 6), 'pass': bool(jd > 0.05), 'n_obs_used': len(diffs)}


def compute_sal(substrate):
    """Spearman(per-action mean activation over all steps, per-action mean state-change).
    Pass if rho > 0.3."""
    if not isinstance(substrate, LPLSubstrate):
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    if substrate._sal_steps == 0:
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}

    mean_activation = substrate._sal_sum / substrate._sal_steps  # (n_actions,)
    visited = substrate._delta_count > 0
    n_visited = int(visited.sum())
    if n_visited < 5:
        return {'rho': None, 'pass': False, 'n_actions_visited': n_visited}

    mean_delta = substrate._delta_sum[visited] / (substrate._delta_count[visited] + 1e-8)
    rho = spearman_rho(mean_activation[visited].tolist(), mean_delta.tolist())
    return {
        'rho': round(float(rho), 4) if rho is not None else None,
        'pass': bool(rho is not None and rho > 0.3),
        'n_actions_visited': n_visited,
    }


# ─────────────────────────────────────────────────────────────
# SINGLE RUN HARNESS
# ─────────────────────────────────────────────────────────────

def run_single(game_name, condition, draw, seed, n_actions, kb_delta):
    print(f"  {game_name.upper()} | {condition} | draw={draw} | seed={seed} ...", end='', flush=True)

    if condition == 'control_c':
        substrate = ControlC(n_actions=n_actions, seed=seed)
    elif condition == 'lpl_dynamics':
        substrate = LPLSubstrate(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    repr_log = []
    obs_store = []
    level_actions = {}

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    level_start_step = 0
    t_start = time.time()
    fresh_episode = True
    l1_step = l2_step = None

    while steps < MAX_STEPS:
        elapsed = time.time() - t_start
        if elapsed > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_flat = obs_arr.ravel()

        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        # I1 repr log: use activation (LPL) or enc (CtrlC)
        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_prev_repr') and substrate._prev_repr is not None:
                repr_log.append((substrate._prev_repr.copy(), level))
            elif hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), level))

        if steps == I3_STEP:
            # For LPL: action_counts not tracked, use action_log frequency
            if isinstance(substrate, LPLSubstrate):
                counts = np.zeros(min(7, n_actions), np.float32)
                for a in action_log:
                    if a < len(counts):
                        counts[a] += 1
                i3_action_counts = counts
            else:
                i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        # State-change tracking for SAL
        if obs is not None and not fresh_episode:
            next_flat = np.asarray(obs, dtype=np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            if isinstance(substrate, LPLSubstrate):
                substrate.record_state_change(action, delta)

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = steps
            if cl == 2 and l2_step is None:
                l2_step = steps
            level_actions[level + 1] = steps - level_start_step
            level = cl
            level_start_step = steps
            substrate.on_level_transition()

        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            level = 0
            level_start_step = steps

    elapsed = time.time() - t_start

    # ── Instrumentation ──
    if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
        repr_log.append((substrate._last_enc.copy(), level))
    elif hasattr(substrate, '_prev_repr') and substrate._prev_repr is not None:
        repr_log.append((substrate._prev_repr.copy(), level))

    i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta) if i3_action_counts is not None else (None, None)
    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)

    i5_pass = i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])

    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) \
        if (r3_snapshot and r3_obs_sample) else {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    sal_result = compute_sal(substrate)

    # Activation stats at end of run (LPL only)
    act_stats = {}
    if isinstance(substrate, LPLSubstrate) and substrate._sal_steps > 0:
        mean_act = substrate._sal_sum / substrate._sal_steps
        act_stats = {
            'act_mean': round(float(mean_act.mean()), 6),
            'act_std': round(float(mean_act.std()), 6),
            'act_max': round(float(mean_act.max()), 6),
        }

    result = {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': level,
        'L1_solved': bool(l1_step is not None),
        'L2_solved': bool(l2_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,

        'I3_spearman_rho': round(i3_rho, 4) if i3_rho is not None else None,
        'I3_pass': i3_pass,
        'I1_within_dist': i1_result['within'],
        'I1_between_dist': i1_result['between'],
        'I1_p_value': i1_result['p_value'],
        'I1_pass': i1_result['pass'],
        'I4_entropy_100': i4_result['entropy_100'],
        'I4_entropy_5000': i4_result['entropy_5000'],
        'I4_reduction_pct': i4_result['reduction_pct'],
        'I4_pass': i4_result['pass'],
        'I5_level_actions': i5_level_actions,
        'I5_pass': i5_pass,
        'R3_jacobian_diff': r3_result['jacobian_diff'],
        'R3_n_obs_used': r3_result['n_obs_used'],
        'R3_pass': r3_result['pass'],

        'SAL_rho': sal_result['rho'],
        'SAL_pass': sal_result['pass'],
        'SAL_n_actions_visited': sal_result['n_actions_visited'],

        'act_mean': act_stats.get('act_mean'),
        'act_std': act_stats.get('act_std'),
        'act_max': act_stats.get('act_max'),
    }

    l1_str = f"L1@{l1_step}" if l1_step else "L1=None"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    sal_str = f"SAL={sal_result['rho']:.3f}" if sal_result['rho'] is not None else "SAL=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    i4_str = f"I4={i4_result['reduction_pct']:.1f}%" if i4_result['reduction_pct'] is not None else "I4=null"
    print(f" {l1_str} | {i3_str} | {r3_str} | {sal_str} | {i4_str} | {elapsed:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    t_global = time.time()

    print("=" * 70)
    print("STEP 1264 — LPL DYNAMICS: NO SELECTOR, DYNAMICS PRODUCE ACTION")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Conditions: control_c, lpl_dynamics")
    print(f"Draws: {N_DRAWS} per condition per game = 30 runs")
    print(f"Budget: {MAX_STEPS} steps / {MAX_SECONDS}s per run")
    print()
    print("activation = W_action @ [enc, h]. action = argmax(activation).")
    print(f"LPL: eta={ETA_HEBB}, eta_p={ETA_PRED}, refractory={REFRACTORY_DECAY}.")
    print()

    for game_name in GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta, _ = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"ERROR: diagnostics for {game_name}: {e}")
            continue

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'─'*60}")

        for condition in ['control_c', 'lpl_dynamics']:
            print(f"\n  Condition: {condition}")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + (0 if condition == 'control_c' else 1)
                result = run_single(
                    game_name=game_name,
                    condition=condition,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                )
                all_results.append(result)
                with open(os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json"), 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1264 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    conditions_run = ['control_c', 'lpl_dynamics']
    stages = ['I3_pass', 'I1_pass', 'I4_pass', 'R3_pass', 'SAL_pass']
    print(f"{'Stage':<12}", end='')
    for c in conditions_run:
        print(f"  {'CtrlC' if c == 'control_c' else 'LPL':>8}", end='')
    print()
    print("-" * 35)
    for stage in stages:
        print(f"{stage:<12}", end='')
        for c in conditions_run:
            vals = [r[stage] for r in all_results if r['condition'] == c and r.get(stage) is not None]
            rate = sum(vals) / len(vals) if vals else None
            print(f"  {f'{rate:.2f}' if rate is not None else '  null':>8}", end='')
        print()

    print()
    print("L1 solved:")
    for game in GAMES:
        for cond in conditions_run:
            runs = [r for r in all_results if r['condition'] == cond and r['game'] == game]
            solved = sum(1 for r in runs if r['L1_solved'])
            steps = [r['l1_step'] for r in runs if r['L1_solved']]
            label = 'CtrlC' if cond == 'control_c' else 'LPL'
            print(f"  {game.upper()} {label}: {solved}/5  steps={steps}")

    print()
    print("SAL:")
    for game in GAMES:
        lpl_runs = [r for r in all_results if r['condition'] == 'lpl_dynamics' and r['game'] == game]
        rhos = [r['SAL_rho'] for r in lpl_runs if r.get('SAL_rho') is not None]
        rho_str = f"rho={np.mean(rhos):.3f}±{np.std(rhos):.3f}" if rhos else "rho=null"
        print(f"  {game.upper()}: {rho_str}")

    print()
    print("R3:")
    for game in GAMES:
        r3s = [r['R3_jacobian_diff'] for r in all_results
               if r['condition'] == 'lpl_dynamics' and r['game'] == game
               and r['R3_jacobian_diff'] is not None]
        print(f"  {game.upper()}: R3={np.mean(r3s):.4f}±{np.std(r3s):.4f}" if r3s else f"  {game.upper()}: null")

    print()
    print("I4 (entropy reduction):")
    for game in GAMES:
        lpl_runs = [r for r in all_results if r['condition'] == 'lpl_dynamics' and r['game'] == game]
        reductions = [r['I4_reduction_pct'] for r in lpl_runs if r.get('I4_reduction_pct') is not None]
        if reductions:
            print(f"  {game.upper()}: {np.mean(reductions):.1f}%±{np.std(reductions):.1f}%")
        else:
            print(f"  {game.upper()}: null")

    out_path = os.path.join(RESULTS_DIR, 'step1264_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1264,
            'conditions': conditions_run,
            'games': GAMES,
            'n_draws': N_DRAWS,
            'total_runs': len(all_results),
            'runs': all_results,
        }, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == '__main__':
    main()
