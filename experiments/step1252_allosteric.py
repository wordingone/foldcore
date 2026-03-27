"""
Step 1252 — Allosteric Substrate (encoding IS action selection)

R3 hypothesis: LPL modifies W such that the Jacobian ∂(enc)/∂obs differs between
fresh and experienced substrates (R3 confirmed). The real test is I1: does LPL produce
state-distinguishing encodings? If LPL's predictive term drives temporally-adjacent
encodings to be similar while between-level encodings differ, I1 should pass.

Core principle (allosteric): One matrix W serves as BOTH encoder AND policy.
  enc[a] = W[a] · centered_obs: how strongly action a resonates with current obs.
  Selection: GPR-style threshold on |enc|. Same W drives both.
  Update: LPL (Halvagal & Zenke 2023) — Hebbian + predictive terms.

When W changes, encoding AND action salience change simultaneously. No gap between
internal representation and action selection (the Step 1251 bottleneck).

Design notes vs spec:
- Bootstrap fix: explicit N_bootstrap=200 steps of random action before GPR kicks in.
  Reason: with n_actions=4103, random W produces max(|s|) > threshold immediately
  (extreme-value statistics: max ~ 2.9σ vs threshold ~ 1.4σ). Bootstrap phase in spec
  never activates for click games without this fix.
- W persists across level transitions (trajectory context, same as Step 1251 WiringA/h).
- Games: ls20, vc33, sp80 as specified. 5 draws each vs ControlC. 30 runs total.
"""
import sys, os, time, json, math, copy
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Config ───
GAMES = ['ls20', 'vc33', 'sp80']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256  # avgpool4 output dimension

# Allosteric substrate hyperparameters (frozen)
ETA_H = 0.01       # Hebbian learning rate
ETA_P = 0.01       # Predictive learning rate
K_THRESH = 1.0     # GPR threshold multiplier (theta = mean(|s|) + K*std(|s|))
N_BOOTSTRAP = 200  # Steps of mandatory random exploration before GPR kicks in
W_INIT_SCALE = 0.01  # W init std (small enough to not cause NaN early)
W_GRAD_CLIP = 1.0  # Gradient norm clip per step

# Instrumentation steps (same as Step 1251)
I3_STEP = 200
I1_STEP = 1000
I4_STEPS = [100, 5000]
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01
I4_WINDOW = 50

# Paths
DIAG_DIR = os.path.join(os.path.dirname(__file__), 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'allosteric_experiment')

# ─── Hash (same as Step 1251) ───
_HASH_H = np.random.RandomState(42).randn(12, ENC_DIM).astype(np.float32)

def hash_enc(x: np.ndarray) -> int:
    bits = (_HASH_H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)


# ─── Game factory ───
def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def get_game_n_actions(game_name: str) -> int:
    _table = {'ls20': 7, 'vc33': 4103, 'sp80': 4103, 'ft09': 4103}
    return _table.get(game_name.lower(), 4103)


# ─── Diagnostics ───
def load_game_diag(game_name: str) -> dict:
    path = os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')
    with open(path) as f:
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


# ─── Instrumentation utils (same as Step 1251) ───
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
    within = np.array(within_dists, np.float32)
    between = np.array(between_dists, np.float32)
    if len(within) < 2 or len(between) < 2:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = 0
    for _ in range(n_perms):
        perm = rng.permutation(len(all_dists))
        perm_w = all_dists[perm[:n_w]]
        perm_b = all_dists[perm[n_w:]]
        if float(np.mean(perm_b) - np.mean(perm_w)) >= observed:
            count += 1
    return count / n_perms


def spearman_rho(x, y):
    if len(x) < 2 or len(y) < 2:
        return None
    n = len(x)
    rx = np.argsort(np.argsort(x)).astype(np.float32)
    ry = np.argsort(np.argsort(y)).astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


def compute_i3(action_counts_200: np.ndarray, kb_delta: np.ndarray) -> tuple:
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


def compute_i1(repr_level_log: list) -> dict:
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
        if l1 == l2:
            within_dists.append(d)
        else:
            between_dists.append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists, n_perms=500)
    within_mean = float(np.mean(within_dists))
    between_mean = float(np.mean(between_dists))
    i1_pass = bool(within_mean < between_mean and p_val < 0.05)
    return {
        'within': round(within_mean, 4),
        'between': round(between_mean, 4),
        'p_value': round(p_val, 4),
        'pass': i1_pass,
    }


def compute_i4(action_log: list, n_actions: int) -> dict:
    def entropy_at(step):
        if step > len(action_log):
            return None
        window = action_log[max(0, step - I4_WINDOW): step]
        if len(window) < 5:
            return None
        return action_entropy(window, n_actions)
    h100 = entropy_at(100)
    h5000 = entropy_at(5000)
    if h100 is None or h5000 is None or h100 < 1e-6:
        return {'entropy_100': h100, 'entropy_5000': h5000, 'reduction_pct': None, 'pass': False}
    reduction_pct = float((h100 - h5000) / h100 * 100)
    return {
        'entropy_100': round(h100, 4),
        'entropy_5000': round(h5000, 4) if h5000 is not None else None,
        'reduction_pct': round(reduction_pct, 2),
        'pass': bool(reduction_pct > 10.0),
    }


# ─────────────────────────────────────────────────────────────
# ALLOSTERIC SUBSTRATE
# ─────────────────────────────────────────────────────────────

class AlloSubstrate:
    """
    Allosteric substrate — encoding IS action selection.

    W: (n_actions × ENC_DIM). Single matrix serves as both:
      - Encoder: enc = W @ centered_obs  (n_actions-dimensional)
      - Policy:  salience[a] = enc[a] — action with highest |salience| is selected

    Update rule (LPL — Halvagal & Zenke 2023):
      delta_W_hebb = eta_h * (outer(enc, centered_obs) - enc^2[:, None] * W)  [Oja-style]
      delta_W_pred = eta_p * outer(enc - prev_enc, centered_obs)  [predictive smoothing]
      W += delta_W_hebb + delta_W_pred  [with gradient norm clip]

    Selection (GPR-style):
      theta = mean(|enc|) + K_THRESH * std(|enc|) + EPS
      if step < N_BOOTSTRAP or max(|enc|) < theta: random action
      else: argmax(|enc|) — "on-center off-surround" winner

    Bootstrap note: with n_actions=4103, random W produces max(|enc|) ~ 2.9σ > theta ~ 1.4σ.
    The explicit N_bootstrap=200 steps ensures uniform coverage before GPR kicks in,
    regardless of W initialization magnitude.

    W persists across level transitions (trajectory context).
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._seed = seed
        self._rng = np.random.RandomState(seed)
        rng_init = np.random.RandomState(seed + 99999)

        # W: allosteric matrix — both encoder and policy
        self.W = rng_init.randn(n_actions, ENC_DIM).astype(np.float32) * W_INIT_SCALE

        # Running mean for centered encoding
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # LPL state: previous encoding for predictive term
        self._prev_enc = None  # (n_actions,)

        # Action visit counts (for I3 instrumentation)
        self.action_counts = np.zeros(n_actions, np.float32)

        # GPR state tracking
        self._in_selection_mode = False  # True once bootstrap phase ends
        self._selection_start_step = None  # When informed selection began

        # Last encoding for I1 repr log
        self._last_enc = None  # (n_actions,)

        self.step = 0

    def _centered_encode(self, obs: np.ndarray) -> tuple:
        """C4: avgpool4 + running mean centering. Returns (centered_obs, raw_x)."""
        x = _enc_frame(obs)
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean, x

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray,
                                   frozen_W: np.ndarray) -> np.ndarray:
        """
        Read-only allosteric encoding for R3 Jacobian.
        Returns enc = W_frozen @ (avgpool4(obs) - running_mean_frozen).
        Shape: (n_actions,).
        Does NOT update any state.
        """
        x = _enc_frame(np.asarray(obs_raw, np.float32))
        centered = x - frozen_running_mean
        return frozen_W @ centered  # (n_actions,)

    def _lpl_update(self, enc: np.ndarray, centered_obs: np.ndarray):
        """LPL (Halvagal & Zenke 2023): Hebbian + predictive update of W."""
        # Hebbian term (Oja-style normalization prevents runaway)
        # delta_W_hebb = eta_h * (outer(enc, centered_obs) - enc^2[:, None] * W)
        hebb = np.outer(enc, centered_obs) - (enc ** 2)[:, None] * self.W
        delta_W = ETA_H * hebb

        # Predictive term (temporal smoothing)
        if self._prev_enc is not None:
            pred = np.outer(enc - self._prev_enc, centered_obs)
            delta_W += ETA_P * pred

        # Gradient norm clip (per step)
        norm = float(np.linalg.norm(delta_W))
        if norm > W_GRAD_CLIP:
            delta_W *= W_GRAD_CLIP / norm

        self.W += delta_W

        # Clip W itself to prevent NaN
        np.clip(self.W, -100.0, 100.0, out=self.W)

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, np.float32)

        # Centered encoding
        centered_obs, _ = self._centered_encode(obs)

        # Allosteric encoding (enc IS the salience vector)
        enc = self.W @ centered_obs  # (n_actions,)
        self._last_enc = enc.copy()

        # Action selection
        if self.step < N_BOOTSTRAP:
            # Mandatory bootstrap: random action
            action = self._rng.randint(0, self.n_actions)
        else:
            # GPR-style threshold selection
            s_abs = np.abs(enc)
            theta = float(np.mean(s_abs) + K_THRESH * np.std(s_abs)) + 1e-8
            if float(np.max(s_abs)) >= theta:
                if not self._in_selection_mode:
                    self._in_selection_mode = True
                    self._selection_start_step = self.step
                action = int(np.argmax(s_abs))
            else:
                action = self._rng.randint(0, self.n_actions)

        self.action_counts[action] += 1

        # LPL update
        self._lpl_update(enc, centered_obs)
        self._prev_enc = enc.copy()

        self.step += 1
        return action

    def on_level_transition(self):
        # W persists across levels (trajectory context)
        # Reset prev_enc to avoid spurious predictive signal across level boundary
        self._prev_enc = None

    def reset(self, seed: int):
        rng_init = np.random.RandomState(seed + 99999)
        self.W = rng_init.randn(self.n_actions, ENC_DIM).astype(np.float32) * W_INIT_SCALE
        self.running_mean[:] = 0
        self.n_obs = 0
        self._prev_enc = None
        self.action_counts[:] = 0
        self._in_selection_mode = False
        self._selection_start_step = None
        self._last_enc = None
        self.step = 0
        self._rng = np.random.RandomState(seed)

    def get_state(self) -> dict:
        return {
            'W': self.W.copy(),
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'action_counts': self.action_counts.copy(),
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.W = state['W'].copy()
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.action_counts = state['action_counts'].copy()
        self.step = state['step']


# ─────────────────────────────────────────────────────────────
# R3 JACOBIAN FOR ALLOSTERIC SUBSTRATE
# ─────────────────────────────────────────────────────────────

def compute_r3_jacobian_allo(substrate: AlloSubstrate,
                              obs_sample: list,
                              frozen_state: dict,
                              rng_seed: int = 42) -> dict:
    """
    R3 Jacobian sketch for allosteric substrate.
    Internal repr = enc = W @ centered_obs (n_actions-dimensional).
    Experienced: frozen_W = W at step 5000, frozen_running_mean.
    Fresh: W_init (re-initialized from seed), running_mean=0.
    PASS: Frobenius diff > 0.05.
    """
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    frozen_rm = frozen_state['running_mean']
    frozen_W = frozen_state['W']

    # Fresh W: same initialization scheme, same seed
    fresh_seed = substrate._seed
    rng_fresh = np.random.RandomState(fresh_seed + 99999)
    fresh_W = rng_fresh.randn(substrate.n_actions, ENC_DIM).astype(np.float32) * W_INIT_SCALE
    fresh_rm = np.zeros(ENC_DIM, np.float32)

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, np.float32)

        # Experienced Jacobian sketch
        baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_W)
        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            pert = obs_flat + R3_EPSILON * direction
            pert_repr = substrate.get_internal_repr_readonly(pert, frozen_rm, frozen_W)
            J_exp_cols.append((pert_repr - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)  # [n_dirs, n_actions]

        # Fresh Jacobian sketch
        baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_W)
        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            pert = obs_flat + R3_EPSILON * direction
            pert_repr = substrate.get_internal_repr_readonly(pert, fresh_rm, fresh_W)
            J_fresh_cols.append((pert_repr - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diff = float(np.linalg.norm(J_exp - J_fresh))
        diffs.append(diff)

    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    jacobian_diff = float(np.mean(diffs))
    return {
        'jacobian_diff': round(jacobian_diff, 6),
        'pass': bool(jacobian_diff > 0.05),
        'n_obs_used': len(diffs),
    }


# ─────────────────────────────────────────────────────────────
# CONTROL C (from Step 1251 — re-implemented here for independence)
# ─────────────────────────────────────────────────────────────

class ControlC:
    """
    Control C — centered encoding + argmin only (same as Step 1251).
    Baseline: no allosteric matrix, no LPL, no learning.
    """

    def __init__(self, n_actions: int, seed: int):
        self.n_actions = n_actions
        self._seed = seed
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

    def get_internal_repr_readonly(self, obs_raw: np.ndarray,
                                   frozen_running_mean: np.ndarray) -> np.ndarray:
        x = _enc_frame(np.asarray(obs_raw, np.float32))
        return x - frozen_running_mean  # 256D

    def process(self, obs_raw) -> int:
        obs = np.asarray(obs_raw, np.float32)
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

    def get_state(self) -> dict:
        return {
            'running_mean': self.running_mean.copy(),
            'n_obs': self.n_obs,
            'action_counts': self.action_counts.copy(),
            'step': self.step,
        }

    def set_state(self, state: dict):
        self.running_mean = state['running_mean'].copy()
        self.n_obs = state['n_obs']
        self.action_counts = state['action_counts'].copy()
        self.step = state['step']


def compute_r3_jacobian_ctrl(substrate: ControlC,
                              obs_sample: list,
                              frozen_state: dict,
                              rng_seed: int = 42) -> dict:
    """R3 Jacobian for ControlC (256D enc)."""
    rng = np.random.RandomState(rng_seed)
    if not obs_sample:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    frozen_rm = frozen_state['running_mean']
    fresh_rm = np.zeros(ENC_DIM, np.float32)

    diffs = []
    for obs_raw in obs_sample[:R3_N_OBS]:
        obs_flat = np.asarray(obs_raw, np.float32)

        baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm)
        J_exp_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            pert = obs_flat + R3_EPSILON * direction
            J_exp_cols.append((substrate.get_internal_repr_readonly(pert, frozen_rm) - baseline_exp) / R3_EPSILON)
        J_exp = np.stack(J_exp_cols, axis=0)

        baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm)
        J_fresh_cols = []
        for _ in range(R3_N_DIRS):
            direction = rng.randn(*obs_flat.shape).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)
            pert = obs_flat + R3_EPSILON * direction
            J_fresh_cols.append((substrate.get_internal_repr_readonly(pert, fresh_rm) - baseline_fresh) / R3_EPSILON)
        J_fresh = np.stack(J_fresh_cols, axis=0)

        diffs.append(float(np.linalg.norm(J_exp - J_fresh)))

    jacobian_diff = float(np.mean(diffs)) if diffs else None
    return {
        'jacobian_diff': round(jacobian_diff, 6) if jacobian_diff else None,
        'pass': bool(jacobian_diff and jacobian_diff > 0.05),
        'n_obs_used': len(diffs),
    }


# ─────────────────────────────────────────────────────────────
# SINGLE RUN HARNESS
# ─────────────────────────────────────────────────────────────

def run_single(game_name: str, condition: str, draw: int, seed: int,
               n_actions: int, kb_delta: np.ndarray) -> dict:
    print(f"  Running {game_name.upper()} | {condition} | draw={draw} | seed={seed} ...", end='', flush=True)

    # Build substrate
    if condition == 'allosteric':
        substrate = AlloSubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'control_c':
        substrate = ControlC(n_actions=n_actions, seed=seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    repr_log = []         # (repr, level) for I1
    obs_store = []        # rolling buffer for R3
    level_actions = {}    # level → actions to solve

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None

    steps = 0
    level = 0
    level_start_step = 0
    t_start = time.time()
    fresh_episode = True
    l1_step = l2_step = None

    # Track allosteric diagnostics
    selection_mode_step = None  # When GPR selection first activated
    random_action_count = 0

    while steps < MAX_STEPS:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition()
            fresh_episode = True
            continue

        obs_arr = np.asarray(obs, np.float32)

        # Rolling obs buffer for R3
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        # Repr log for I1 (up to step I1_STEP)
        if steps <= I1_STEP and steps % 20 == 0:
            if hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
                repr_log.append((substrate._last_enc.copy(), level))

        # I3 snapshot
        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts[:min(7, n_actions)].copy()

        # R3 snapshot
        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_log.append(action)

        # Track when GPR selection started (allosteric only)
        if isinstance(substrate, AlloSubstrate):
            if substrate._in_selection_mode and selection_mode_step is None:
                selection_mode_step = substrate._selection_start_step
            if steps < N_BOOTSTRAP or not substrate._in_selection_mode:
                random_action_count += 1

        obs, reward, done, info = env.step(action)
        steps += 1

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

    if i3_action_counts is not None:
        i3_rho, i3_pass = compute_i3(i3_action_counts, kb_delta)
    else:
        i3_rho, i3_pass = None, None

    i1_result = compute_i1(repr_log)
    i4_result = compute_i4(action_log, n_actions)

    i5_pass = i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])

    if r3_snapshot is not None and r3_obs_sample:
        if isinstance(substrate, AlloSubstrate):
            r3_result = compute_r3_jacobian_allo(substrate, r3_obs_sample, r3_snapshot, n_actions)
        else:
            r3_result = compute_r3_jacobian_ctrl(substrate, r3_obs_sample, r3_snapshot)
    else:
        r3_result = {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}

    # Console
    l1_str = f"L1={l1_step}" if l1_step else "L1=None"
    i3_str = f"I3ρ={i3_rho:.2f}" if i3_rho is not None else "I3=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    i1_str = f"I1={'P' if i1_result['pass'] else 'F'}({i1_result['p_value']:.3f})"
    sel_str = f"sel@{selection_mode_step}" if selection_mode_step else "sel=boot"
    print(f" {l1_str} | {i3_str} | {i1_str} | I4={'P' if i4_result['pass'] else 'F'} | {r3_str} | {sel_str} | {elapsed:.1f}s")

    return {
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

        # Allosteric diagnostics
        'selection_mode_step': selection_mode_step,
        'random_action_count': random_action_count,

        'action_sequence': action_log,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = []
    t_global = time.time()

    print("=" * 70)
    print("STEP 1252 — ALLOSTERIC SUBSTRATE (encoding IS action selection)")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Conditions: allosteric, control_c")
    print(f"Draws: {N_DRAWS} per condition per game = 30 runs")
    print(f"Budget: {MAX_STEPS} steps / {MAX_SECONDS}s per run")
    print(f"Bootstrap: {N_BOOTSTRAP} random steps before GPR selection")
    print(f"Hyperparams: k={K_THRESH}, eta_h={ETA_H}, eta_p={ETA_P}")
    print()

    for game_name in GAMES:
        try:
            diag = load_game_diag(game_name)
            kb_delta, kb_responsive = build_kb_profile(diag)
            game_n_actions = diag.get('n_actions', get_game_n_actions(game_name))
        except Exception as e:
            print(f"WARNING: No diagnostics for {game_name}: {e}")
            game_n_actions = get_game_n_actions(game_name)
            kb_delta = np.zeros(7, np.float32)

        print(f"\n{'─'*60}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"W shape: ({game_n_actions} × {ENC_DIM}) = {game_n_actions*ENC_DIM:,} params")
        print(f"{'─'*60}")

        for condition in ['allosteric', 'control_c']:
            print(f"\n  Condition: {condition}")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + (10 if condition == 'allosteric' else 0)
                result = run_single(
                    game_name=game_name,
                    condition=condition,
                    draw=draw,
                    seed=seed,
                    n_actions=game_n_actions,
                    kb_delta=kb_delta,
                )
                all_results.append(result)

                incr_path = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(incr_path, 'w') as f:
                    json.dump({k: v for k, v in result.items() if k != 'action_sequence'}, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1252 COMPLETE — {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}")
    print()

    stages = ['I1_pass', 'I3_pass', 'I4_pass', 'I5_pass', 'R3_pass']
    print(f"{'Stage':<12} {'ControlC':>12} {'Allosteric':>12}  vs Step1251")
    print("-" * 55)

    # Step 1251 baselines (from phase1 results)
    step1251_ctrl = {'I1_pass': 0.00, 'I3_pass': 0.67, 'I4_pass': 0.00, 'I5_pass': None, 'R3_pass': 0.00}
    step1251_wa   = {'I1_pass': 0.00, 'I3_pass': 0.67, 'I4_pass': 0.00, 'I5_pass': None, 'R3_pass': 1.00}

    for stage in stages:
        ctrl = [r[stage] for r in all_results if r['condition'] == 'control_c' and r[stage] is not None]
        allo = [r[stage] for r in all_results if r['condition'] == 'allosteric' and r[stage] is not None]
        ctrl_rate = sum(ctrl) / len(ctrl) if ctrl else None
        allo_rate = sum(allo) / len(allo) if allo else None
        cr = f"{ctrl_rate:.2f}" if ctrl_rate is not None else "null"
        ar = f"{allo_rate:.2f}" if allo_rate is not None else "null"

        # Compare to Step 1251 WiringA
        s1251 = step1251_wa.get(stage)
        comp = ""
        if allo_rate is not None and s1251 is not None:
            if allo_rate > s1251 + 0.1:
                comp = "↑ vs WiringA"
            elif allo_rate < s1251 - 0.1:
                comp = "↓ vs WiringA"
            else:
                comp = "= WiringA"

        print(f"  {stage:<12} {cr:>12} {ar:>12}  {comp}")

    # L1 solve rates
    print(f"\nL1 solve rates:")
    for game in GAMES:
        for cond in ['control_c', 'allosteric']:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            l1 = sum(r['L1_solved'] for r in runs) / len(runs) if runs else 0
            print(f"  {game.upper()}/{cond}: L1={l1:.1%}")

    # Allosteric-specific: when did GPR selection kick in?
    print(f"\nGPR selection activation (allosteric only):")
    for game in GAMES:
        runs = [r for r in all_results if r['game'] == game and r['condition'] == 'allosteric']
        sel_steps = [r['selection_mode_step'] for r in runs if r.get('selection_mode_step') is not None]
        random_counts = [r['random_action_count'] for r in runs]
        if sel_steps:
            print(f"  {game.upper()}: selection_start mean={sum(sel_steps)/len(sel_steps):.0f} | random_actions mean={sum(random_counts)/len(random_counts):.0f}")
        else:
            print(f"  {game.upper()}: GPR threshold never exceeded (always random) | random_actions={sum(random_counts)/len(random_counts):.0f}")

    # Save
    out_path = os.path.join(RESULTS_DIR, 'step1252_results.json')
    with open(out_path, 'w') as f:
        json.dump({
            'step': 1252,
            'conditions': ['allosteric', 'control_c'],
            'games': GAMES,
            'n_draws': N_DRAWS,
            'hyperparams': {'k': K_THRESH, 'eta_h': ETA_H, 'eta_p': ETA_P, 'n_bootstrap': N_BOOTSTRAP},
            'total_runs': len(all_results),
            'runs': [{k: v for k, v in r.items() if k != 'action_sequence'} for r in all_results],
        }, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == '__main__':
    _env_file = r'C:\Users\Admin\.secrets\.env'
    if os.path.exists(_env_file):
        with open(_env_file) as f:
            for line in f:
                if line.strip().startswith('ARC_API_KEY='):
                    os.environ['ARC_API_KEY'] = line.strip().split('=', 1)[1].strip()
                    break
    main()
