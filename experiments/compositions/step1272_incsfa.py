"""
Step 1272 — IncSFA for I1. Add incremental Slow Feature Analysis to Physarum+argmin.

Confirmed spec (Leo mail 3516, 2026-03-28). Wiring interpretation: ENC_DIM=256 (not 16 as
stated in spec — "(U16)" is a component reference, not dimensionality). Notified Leo via mail
3517.

Composition:
  raw pixels → avgpool4 → centered encoding (256 dims)
    → IncSFA (online derivative covariance, d=8 slow features)
    → [centered_enc (256); slow_features (8); h (64)] = 328 dims → W_action
    → Physarum flow (tube dynamics)
    → tube-weighted argmin → action

IncSFA:
  delta_enc = enc_t - enc_{t-1}
  C_dot = (1-ETA_SFA) * C_dot + ETA_SFA * outer(delta_enc, delta_enc)  [256×256]
  Every SFA_FREQ steps: W_sfa = bottom-8 eigenvectors of C_dot (smallest eigenvalues)
  slow_features = W_sfa.T @ enc_t  (8 dims)

Conditions: physarum_incsfa (new) + physarum_no_incsfa (step1271 config = control)
Games: SP80, FT09, TR87
Draws: 5 per condition per game = 30 runs
Budget: 10K steps, 300s per run
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAMES = ['sp80', 'ft09', 'tr87']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
SFA_DIM = 8                          # slow features
EXT_DIM = ENC_DIM + H_DIM           # 320 (no SFA)
EXT_DIM_SFA = ENC_DIM + H_DIM + SFA_DIM  # 328 (with SFA)

ETA_FLOW = 0.01
DECAY = 0.001
ETA_SFA = 0.01        # derivative covariance EMA rate
SFA_FREQ = 50         # recompute eigenvectors every N steps

I3_STEP = 200
I1_STEP = 1000
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50

PDIR = 'B:/M/the-search/experiments/results/prescriptions'
DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1272')

SOLVER_PRESCRIPTIONS = {
    'sp80': ('sp80_fullchain.json', 'sequence'),
    'ft09': ('ft09_fullchain.json', 'all_actions'),
    'tr87': ('tr87_fullchain.json', 'all_actions'),
}
SOLVER_TOTAL_ACTIONS = {'sp80': 107, 'ft09': 75, 'tr87': 123}


# ─── Utilities ───

def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_prescription(game_name: str):
    fname, field = SOLVER_PRESCRIPTIONS[game_name]
    path = os.path.join(PDIR, fname)
    with open(path) as f:
        d = json.load(f)
    return d[field]


def compute_solver_level_steps(game_name: str, seed: int = 1) -> dict:
    prescription = load_prescription(game_name)
    env = make_game(game_name)
    obs = env.reset(seed=seed)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    level = 0
    level_first_step = {}
    fresh_episode = True
    step = 0
    for action in prescription:
        action_int = int(action) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            obs = obs_next
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            obs = env.reset(seed=seed)
            fresh_episode = True
        else:
            obs = obs_next
    return level_first_step


def compute_arc_score(level_first_steps: dict, solver_level_steps: dict) -> float:
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def load_game_diag(game_name: str) -> dict:
    with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
        return json.load(f)


def build_kb_profile(diag: dict) -> np.ndarray:
    kb = diag.get('kb_responsiveness', {})
    delta = np.zeros(7, np.float32)
    for i, key in enumerate([f'ACTION{j}' for j in range(1, 8)]):
        if key in kb:
            delta[i] = kb[key].get('delta_mean', 0.0)
    return delta


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


# ─── Substrates ───

class PhysarumSubstrate:
    """Step 1271 config — no IncSFA. EXT_DIM=320."""

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        W_action_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()
        self.action_counts = np.zeros(n_actions, np.float32)
        self._sal_sum = np.zeros(n_actions, np.float64)
        self._sal_steps = 0
        self._delta_sum = np.zeros(n_actions, np.float64)
        self._delta_count = np.zeros(n_actions, np.int64)
        self._prev_enc_flow = None
        self._prev_ext = None
        self._last_action = None
        self._prev_repr = None
        self.step = 0
        self._ext_dim = EXT_DIM

    def _centered_encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _make_ext(self, enc: np.ndarray) -> np.ndarray:
        return np.concatenate([enc, self.h])

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action) -> np.ndarray:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = _enc_frame(obs) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_action @ ext

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = self._centered_encode(obs)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = self._make_ext(enc)
        activation = self.W_action @ ext
        tube_thickness = np.linalg.norm(self.W_action, axis=1)
        effective_count = self.action_counts / (1.0 + tube_thickness)
        action = int(np.argmin(effective_count))
        self.action_counts[action] += 1
        self._sal_sum += activation.astype(np.float64)
        self._sal_steps += 1
        self._prev_repr = activation.copy()
        self._prev_enc_flow = enc.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_flow(self, next_obs_raw):
        if self._prev_enc_flow is None or self._last_action is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean
        delta = enc_after - self._prev_enc_flow
        flow = float(np.linalg.norm(delta))
        self.W_action[self._last_action] += ETA_FLOW * flow * self._prev_ext
        self.W_action *= (1.0 - DECAY)

    def record_state_change(self, action: int, delta: float):
        if 0 <= action < self.n_actions:
            self._delta_sum[action] += delta
            self._delta_count[action] += 1

    def on_level_transition(self):
        self._prev_enc_flow = None
        self._prev_ext = None

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'h': self.h.copy(),
            'W_action': self.W_action.copy(),
            'step': self.step,
        }


class IncSFASubstrate(PhysarumSubstrate):
    """
    Physarum+argmin + IncSFA. EXT_DIM=328 (enc=256, h=64, sfa=8).

    IncSFA: online derivative covariance, bottom-d eigenvectors (slowest features).
      delta_enc = enc_t - enc_{t-1}
      C_dot = (1-ETA_SFA)*C_dot + ETA_SFA * outer(delta_enc, delta_enc)
      Every SFA_FREQ steps: W_sfa = bottom-d eigenvectors of C_dot
      slow_features = W_sfa.T @ enc_t
    """

    def __init__(self, n_actions: int, seed: int):
        super().__init__(n_actions, seed)
        # Override W_action for larger EXT_DIM_SFA
        rng_w = np.random.RandomState(seed + 10000)
        # Re-init W_action for EXT_DIM_SFA=328
        rng_w.randn(H_DIM, H_DIM)  # consume same random state as parent
        rng_w.randn(H_DIM, ENC_DIM)
        scale = 1.0 / np.sqrt(float(EXT_DIM_SFA))
        W_action_init = rng_w.randn(n_actions, EXT_DIM_SFA).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()

        # IncSFA state
        self.C_dot = np.zeros((ENC_DIM, ENC_DIM), np.float64)  # 256×256
        self.W_sfa = np.zeros((ENC_DIM, SFA_DIM), np.float64)  # 256×8 (column = eigvec)
        self._prev_enc_sfa = None   # enc_{t-1} for delta
        self._sfa_initialized = False
        self._ext_dim = EXT_DIM_SFA

        # Update SAL/delta sums for larger action space (already done in parent)

    def _make_ext(self, enc: np.ndarray) -> np.ndarray:
        slow_features = (self.W_sfa.T @ enc.astype(np.float64)).astype(np.float32)
        return np.concatenate([enc, self.h, slow_features])

    def _update_sfa(self, enc: np.ndarray):
        """Update C_dot and recompute W_sfa every SFA_FREQ steps."""
        if self._prev_enc_sfa is None:
            self._prev_enc_sfa = enc.copy()
            return
        delta = (enc - self._prev_enc_sfa).astype(np.float64)
        self._prev_enc_sfa = enc.copy()
        # Online update of derivative covariance
        self.C_dot = (1.0 - ETA_SFA) * self.C_dot + ETA_SFA * np.outer(delta, delta)
        # Recompute bottom-d eigenvectors periodically
        if self.step % SFA_FREQ == 0 and self.step > 0:
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(self.C_dot)
                # eigh returns ascending order: [:d] = smallest = slowest
                self.W_sfa = eigenvectors[:, :SFA_DIM].copy()
                self._sfa_initialized = True
            except np.linalg.LinAlgError:
                pass  # keep previous W_sfa

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_action) -> np.ndarray:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = _enc_frame(obs) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        slow = (self.W_sfa.T @ enc.astype(np.float64)).astype(np.float32)
        ext = np.concatenate([enc, h_new, slow])
        return frozen_W_action @ ext

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, dtype=np.float32)
        enc = self._centered_encode(obs)
        self._update_sfa(enc)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = self._make_ext(enc)  # 328 dims
        activation = self.W_action @ ext
        tube_thickness = np.linalg.norm(self.W_action, axis=1)
        effective_count = self.action_counts / (1.0 + tube_thickness)
        action = int(np.argmin(effective_count))
        self.action_counts[action] += 1
        self._sal_sum += activation.astype(np.float64)
        self._sal_steps += 1
        self._prev_repr = activation.copy()
        self._prev_enc_flow = enc.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self.step += 1
        return action

    def get_state(self):
        state = super().get_state()
        state['W_sfa'] = self.W_sfa.copy()
        state['C_dot_diag'] = np.diag(self.C_dot).copy()  # save diagonal only (not full 256×256)
        return state


# ─── Instrumentation ───

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
    if not obs_sample or frozen_state is None:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    frozen_rm = frozen_state['running_mean']
    frozen_h = frozen_state['h']
    frozen_W_action = frozen_state['W_action']
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W_action = substrate.W_action_init

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, dtype=np.float32)
        baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W_action)
        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            pert = substrate.get_internal_repr_readonly(perturbed, frozen_rm, frozen_h, frozen_W_action)
            J_exp_cols.append((pert - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)

        baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W_action)
        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            d = rng.randn(*obs_flat.shape).astype(np.float32)
            d /= (np.linalg.norm(d) + 1e-8)
            perturbed = obs_flat + R3_EPSILON * d
            pert = substrate.get_internal_repr_readonly(perturbed, fresh_rm, fresh_h, fresh_W_action)
            J_fresh_cols.append((pert - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diffs.append(float(np.linalg.norm(J_exp - J_fresh)))

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    jd = float(np.mean(diffs))
    return {'jacobian_diff': round(jd, 6), 'pass': bool(jd > 0.05), 'n_obs_used': len(diffs)}


def compute_sal(substrate):
    if not isinstance(substrate, (PhysarumSubstrate, IncSFASubstrate)):
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    if substrate._sal_steps == 0:
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    mean_activation = substrate._sal_sum / substrate._sal_steps
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


# ─── Single run harness ───

def run_single(game_name, condition, draw, seed, n_actions, kb_delta, solver_level_steps):
    print(f"  {game_name.upper()} | {condition} | draw={draw} | seed={seed} ...", end='', flush=True)

    if condition == 'physarum_incsfa':
        substrate = IncSFASubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'physarum_no_incsfa':
        substrate = PhysarumSubstrate(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    repr_log = []
    obs_store = []

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    max_level = 0
    level_start_step = 0
    level_first_step = {}
    level_actions = {}
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
        obs_flat = obs_arr.ravel()

        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps <= I1_STEP and steps % 20 == 0:
            if substrate._prev_repr is not None:
                repr_log.append((substrate._prev_repr.copy(), level))

        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        if obs is not None and not fresh_episode:
            next_flat = np.asarray(obs, dtype=np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            substrate.update_flow(obs)
            substrate.record_state_change(action, delta)

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_actions[level + 1] = steps - level_start_step
            if cl not in level_first_step:
                level_first_step[cl] = steps
            if cl > max_level:
                max_level = cl
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

    if substrate._prev_repr is not None:
        repr_log.append((substrate._prev_repr.copy(), level))

    i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta) if i3_action_counts is not None else (None, None)
    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)

    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    i5_pass = None
    i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])

    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) \
        if (r3_snapshot and r3_obs_sample) else {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    sal_result = compute_sal(substrate)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    arc_per_level = {}
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_step and level_first_step[lv] > 0:
            ratio = solver_step / level_first_step[lv]
            arc_per_level[int(lv)] = round(min(1.0, ratio * ratio), 6)

    # IncSFA diagnostics
    sfa_info = {}
    if isinstance(substrate, IncSFASubstrate):
        sfa_info['sfa_initialized'] = substrate._sfa_initialized
        if substrate._sfa_initialized:
            eigenvalues, _ = np.linalg.eigh(substrate.C_dot)
            sfa_info['C_dot_smallest_eigenvalues'] = [round(float(v), 6) for v in eigenvalues[:8]]
            sfa_info['C_dot_largest_eigenvalue'] = round(float(eigenvalues[-1]), 6)

    label = 'SFA' if condition == 'physarum_incsfa' else 'nSFA'
    i1_str = f"I1={'P' if i1_result['pass'] else 'F'}(w={i1_result['within']},b={i1_result['between']})" if i1_result['within'] is not None else "I1=null"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    arc_str = f"ARC={arc_score:.4f}" if arc_score > 0 else "ARC=0"
    print(f" [{label}] Lmax={max_level} | {i1_str} | {i3_str} | {r3_str} | {arc_str} | {elapsed:.1f}s")

    return {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'L2_solved': bool(l2_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'arc_per_level': arc_per_level,
        'solver_level_steps': {int(k): int(v) for k, v in solver_level_steps.items()},
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
        'sfa_info': sfa_info,
    }


# ─── Main ───

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1272 — IncSFA FOR I1")
    print("Physarum+argmin + slow features (ENC_DIM=256+SFA_DIM=8 → EXT_DIM=328)")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Conditions: physarum_incsfa, physarum_no_incsfa")
    print(f"Draws: {N_DRAWS} per condition per game = 30 runs")
    print(f"Budget: {MAX_STEPS} steps / {MAX_SECONDS}s per run")
    print(f"SFA: ETA_SFA={ETA_SFA}, SFA_DIM={SFA_DIM}, SFA_FREQ={SFA_FREQ}")
    print(f"NOTE: Using ENC_DIM=256 (actual); Leo's spec said 16 (likely notation confusion)")
    print()

    print("Computing solver per-level step counts...")
    solver_level_steps_by_game = {}
    for game_name in GAMES:
        try:
            slv = compute_solver_level_steps(game_name, seed=1)
            solver_level_steps_by_game[game_name] = slv
            print(f"  {game_name.upper()}: {slv}")
        except Exception as e:
            print(f"  {game_name.upper()}: ERROR — {e}")
            solver_level_steps_by_game[game_name] = {}
    print()

    all_results = []
    t_global = time.time()

    for game_name in GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', 4103)
        except Exception as e:
            print(f"ERROR loading diagnostics for {game_name}: {e}")
            continue

        solver_level_steps = solver_level_steps_by_game.get(game_name, {})
        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'─'*60}")

        for condition in ['physarum_incsfa', 'physarum_no_incsfa']:
            print(f"\n  Condition: {condition}")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + (1 if condition == 'physarum_incsfa' else 2)
                result = run_single(
                    game_name=game_name,
                    condition=condition,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                    solver_level_steps=solver_level_steps,
                )
                all_results.append(result)
                fname = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1272 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    summary = []
    for result in all_results:
        summary.append({k: result[k] for k in [
            'game', 'condition', 'draw', 'seed', 'max_level',
            'L1_solved', 'arc_score',
            'I3_pass', 'I1_pass', 'I4_pass', 'I5_pass', 'R3_pass', 'SAL_pass',
            'R3_jacobian_diff', 'I3_spearman_rho', 'SAL_rho',
            'I1_within_dist', 'I1_between_dist', 'I1_p_value',
        ]})
    with open(os.path.join(RESULTS_DIR, 'step1272_results.json'), 'w') as f:
        json.dump({'runs': summary, 'total_elapsed': round(total_elapsed, 1)}, f, indent=2)

    print("I1 Summary (KEY METRIC):")
    for game in GAMES:
        for cond in ['physarum_incsfa', 'physarum_no_incsfa']:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            passes = [r['I1_pass'] for r in runs if r.get('I1_pass') is not None]
            within = [r['I1_within_dist'] for r in runs if r['I1_within_dist'] is not None]
            between = [r['I1_between_dist'] for r in runs if r['I1_between_dist'] is not None]
            label = 'SFA' if 'incsfa' in cond else 'nSFA'
            rate = sum(passes) / len(passes) if passes else 0
            w_mean = np.mean(within) if within else None
            b_mean = np.mean(between) if between else None
            print(f"  {game.upper()} {label}: I1={rate:.2f} ({sum(passes)}/{len(passes)}) "
                  f"within={w_mean:.4f if w_mean else 'null'} between={b_mean:.4f if b_mean else 'null'}")

    print("\nARC + L1 Summary:")
    for game in GAMES:
        for cond in ['physarum_incsfa', 'physarum_no_incsfa']:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            arcs = [r['arc_score'] for r in runs]
            l1s = [r['L1_solved'] for r in runs]
            label = 'SFA' if 'incsfa' in cond else 'nSFA'
            print(f"  {game.upper()} {label}: ARC={np.mean(arcs):.4f}  L1={sum(l1s)}/{len(l1s)}")

    print("\nR3 Summary:")
    for game in GAMES:
        for cond in ['physarum_incsfa', 'physarum_no_incsfa']:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            passes = [r['R3_pass'] for r in runs if r.get('R3_pass') is not None]
            label = 'SFA' if 'incsfa' in cond else 'nSFA'
            rate = sum(passes) / len(passes) if passes else 0
            print(f"  {game.upper()} {label} R3: {rate:.2f} ({sum(passes)}/{len(passes)})")

    print(f"\nSTEP 1272 DONE")


if __name__ == '__main__':
    main()
