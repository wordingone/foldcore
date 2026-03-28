"""
Step 1253b — Catalog Revisit: Allosteric Softmax (C#19) with corrected parameters.

Motivation: Step 1253 (allosteric softmax) was the ONE experiment where W drove action
selection and R3+I3 coexisted. Abandoned after 1 experiment for argmin. Never retested with:
  - eta_h=0.05 (fixed at Step 1282, was 0.01 in 1253)
  - I3_cv metric (fixed at Step 1280, I3_rho was broken)

If 1253b produces L1 with corrected params, 40+ experiments of network complexity were noise.
Simplest reflexive map was already there at Step 1253.

Architecture (ALLO condition):
  W: (n_actions × 256) single allosteric matrix. W IS both encoder and policy.
  enc = W @ centered_obs  (n_actions-dimensional allosteric encoding)
  T = max(1 / (std(enc) + eps), T_MIN)   [T_MIN=0.1, prevents extreme concentration]
  probs = softmax(|enc| / T)
  action = sample(probs)
  LPL update: Hebbian (Oja) + predictive
    dW_hebb = eta_h * (outer(enc, centered_obs) - enc^2[:, None] * W)   [Oja]
    dW_pred = eta_p * outer(enc - prev_enc, centered_obs)               [predictive]
    W += dW_hebb + dW_pred  (norm-clipped)
  eta_h = 0.05  ← was 0.01 in 1253 (KEY CHANGE)
  eta_p = 0.01  (unchanged)

Controls:
  CTL:    Pure argmin on visit counts. No learning. Legacy baseline.
  PE-EMA: pe_ema-weighted argmin from Step 1282 (old "confirmed composition").
           W_action (n_actions × 320) + recurrent h, LPL flow update (eta_h=0.05),
           selection: argmin(visit_counts - 0.1 * pe_ema).

What this tests:
  Did we waste 40+ experiments? If ALLO L1 on any game that CTL doesn't solve →
  simplest reflexive map was there at Step 1253. Network complexity unjustified.

Kill criteria:
  - ALLO L1 ≤ CTL L1 on all games → flat W can't match argmin even with corrected params.
  - ALLO I3_cv > 3 × CTL I3_cv on 3+ games → softmax kills coverage at eta_h=0.05.

Parallel thread to 1296 (reflexive network). Different architecture.

Budget: 10 ARC × 3 cond × 5 draws = 150 + 1 MBPP × 3 cond × 5 draws = 15 → 165 runs.
  ~8s/run (ALLO: LPL + softmax), ~5s (PE-EMA), ~3s (CTL) → ~1100s ≈ 18 min.
  Within 30-min chain cap.

Spec: Leo mail 3643, 2026-03-28. Parallel to Step 1296 (mail 3642/3644).
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

import numpy as np
from substrates.step0674 import _enc_frame

# --- Config ---
ARC_GAMES = ['ls20', 'ft09', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
MBPP_GAMES = ['mbpp_0']
GAMES = ARC_GAMES + MBPP_GAMES

N_DRAWS = 5
N_DRAWS_MBPP = 5

MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256    # avgpool4 output
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320 (for PE-EMA substrate)

# ALLO hyperparameters
ETA_H = 0.05    # KEY FIX: was 0.01 in original step 1253
ETA_P = 0.01
W_INIT_SCALE = 0.01
W_GRAD_CLIP = 1.0
T_MIN = 0.1     # minimum temperature (prevents extreme action concentration)

# PE-EMA hyperparameters (from step 1282)
ETA_H_FLOW = 0.05     # same as ALLO
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.05
SELECTION_ALPHA = 0.1
DECAY = 0.001

# Instrumentation
I3_STEP = 200
I1_SAMPLE_FREQ = 100
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['allo', 'ctl', 'pe_ema']
LABELS = {'allo': 'ALLO', 'ctl': 'CTL', 'pe_ema': 'PEEMA'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1253b')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'
DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')

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


# --- Utilities ---

def make_game(game_name):
    if game_name.lower().startswith('mbpp'):
        from mbpp_game import make as mbpp_make
        return mbpp_make(game_name.lower())
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
    if game_name.lower().startswith('mbpp'):
        from mbpp_game import compute_solver_steps
        idx = int(game_name.split('_', 1)[1]) if '_' in game_name else 0
        return compute_solver_steps(idx)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    obs = env.reset(seed=seed)
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
            fresh_episode = False; obs = obs_next; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step; level = cl
        if done:
            obs = env.reset(seed=seed); fresh_episode = True
        else:
            obs = obs_next
    return level_first_step


def compute_arc_score(level_first_steps, solver_level_steps):
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def action_entropy(action_seq, n_actions):
    if not action_seq:
        return 0.0
    counts = np.zeros(n_actions, np.float32)
    for a in action_seq:
        if 0 <= a < n_actions:
            counts[a] += 1
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, dtype=np.float32)
    between = np.array(between_dists, dtype=np.float32)
    if len(within) < 2 or len(between) < 2:
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


def compute_i1(repr_log):
    if len(repr_log) < 4:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    rng = np.random.RandomState(1)
    n = len(repr_log)
    within_dists, between_dists = [], []
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        r1, l1 = repr_log[i]
        r2, l2 = repr_log[j]
        d = cosine_dist(r1, r2)
        (within_dists if l1 == l2 else between_dists).append(d)
    if len(within_dists) < 2 or len(between_dists) < 2:
        return {'within': None, 'between': None, 'p_value': 1.0, 'pass': False}
    p_val = permutation_test_i1(within_dists, between_dists, n_perms=500)
    wm = float(np.mean(within_dists)); bm = float(np.mean(between_dists))
    return {'within': round(wm, 4), 'between': round(bm, 4),
            'p_value': round(p_val, 4), 'pass': bool(wm < bm and p_val < 0.05)}


# --- Substrate A: Allosteric Softmax (ALLO) ---

class AlloSoftmaxSubstrate:
    """W IS the substrate: (n_actions × ENC_DIM) allosteric matrix.
    LPL update (Hebbian Oja + predictive). Adaptive softmax selection.
    eta_h=0.05 (corrected from original step 1253's 0.01).
    """

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng = np.random.RandomState(seed)
        self._rng = rng
        rng_w = np.random.RandomState(seed + 99999)
        self.W = rng_w.randn(n_actions, ENC_DIM).astype(np.float32) * W_INIT_SCALE
        self.W_init = self.W.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self._prev_enc = None
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_W):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        centered = x - frozen_rm
        return frozen_W @ centered

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'W': self.W.copy(),
            'step': self.step,
        }

    def process(self, obs_raw):
        centered_obs = self._centered_encode(obs_raw)

        # Allosteric encoding (enc IS the salience/policy vector)
        enc = self.W @ centered_obs  # (n_actions,)
        self._last_enc = enc.copy()

        # Adaptive softmax selection (no bootstrap)
        std_enc = float(np.std(enc))
        T = max(1.0 / (std_enc + 1e-8), T_MIN)
        logits = np.abs(enc) * T
        logits -= logits.max()  # numerical stability
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self.n_actions, p=probs))

        # LPL update: Hebbian (Oja) + predictive
        hebb = np.outer(enc, centered_obs) - (enc ** 2)[:, None] * self.W
        delta_W = ETA_H * hebb
        if self._prev_enc is not None:
            delta_W += ETA_P * np.outer(enc - self._prev_enc, centered_obs)

        norm_dW = float(np.linalg.norm(delta_W))
        if norm_dW > W_GRAD_CLIP:
            delta_W *= W_GRAD_CLIP / norm_dW

        self.W += delta_W
        np.clip(self.W, -100.0, 100.0, out=self.W)

        self._prev_enc = enc.copy()
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        pass  # ALLO: all updates in process()

    def on_level_transition(self):
        self._prev_enc = None  # reset across level boundary


# --- Substrate B: Argmin Control (CTL) ---

class ArgminCtlSubstrate:
    """Pure visit count argmin. No learning. Legacy baseline."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._visit_counts = np.zeros(n_actions, np.float32)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self._last_enc = None
        self.step = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_W):
        return np.zeros(ENC_DIM, np.float32)  # no learned W

    def get_state(self):
        return {'running_mean': self.running_mean.copy(), 'step': self.step}

    def process(self, obs_raw):
        centered_obs = self._centered_encode(obs_raw)
        self._last_enc = centered_obs  # for I1 instrumentation
        action = int(np.argmin(self._visit_counts))
        self._visit_counts[action] += 1
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        pass

    def on_level_transition(self):
        pass


# --- Substrate C: PE-EMA Argmin (from Step 1282) ---

class PeEmaArgminSubstrate:
    """LPL PE substrate from Step 1282 (confirmed composition, eta_h=0.05).

    W_action: (n_actions × 320) = (n_actions × [ENC_DIM + H_DIM])
    Selection: argmin(visit_counts - 0.1 * pe_ema)
    Update: Hebbian flow update on W_action + pe_ema tracking (post-step).
    """

    def __init__(self, n_actions, seed):
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
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01
        self._visit_counts = np.zeros(n_actions, np.float32)
        self.pe_ema = np.zeros(n_actions, np.float32)
        self._prev_enc = None
        self._prev_ext = None
        self._last_action = None
        self._last_enc = None
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

    def process(self, obs_raw):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])
        self._last_enc = enc.copy()
        self._prev_ext_for_update = ext.copy()

        # pe_ema-weighted argmin
        score = self._visit_counts - SELECTION_ALPHA * self.pe_ema
        action = int(np.argmin(score))
        self._visit_counts[action] += 1
        self._prev_enc = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        """LPL flow update + pe_ema update (called after env.step)."""
        if self._prev_enc is None or action is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean

        # PE prediction
        pred_enc = self.W_pred @ self._prev_enc
        pe = float(np.linalg.norm(enc_after - pred_enc))
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc)

        # pe_ema update
        self.pe_ema[action] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[action] + PE_EMA_ALPHA * pe

        # LPL Hebbian flow update
        enc_delta = enc_after - self._prev_enc
        flow = float(np.linalg.norm(enc_delta))
        if hasattr(self, '_prev_ext_for_update') and self._prev_ext_for_update is not None:
            self.W_action[action] += ETA_H_FLOW * flow * self._prev_ext_for_update
            self.W_action *= (1.0 - DECAY)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_ext_for_update = None


# --- R3 computation ---

def compute_r3_allo(substrate, obs_sample, snapshot):
    """R3 for ALLO substrate: compare W vs W_init Jacobian."""
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot['running_mean']
    frozen_W = snapshot['W']
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_W = substrate.W_init.copy()
    obs_subset = obs_sample[-R3_N_OBS:]
    rng = np.random.RandomState(42)
    diffs = []
    for obs_arr in obs_subset:
        obs_flat = obs_arr.ravel()
        dirs = rng.randn(R3_N_DIRS, len(obs_flat)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        base_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_W)
        base_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_W)
        for d in dirs:
            pert = obs_flat + R3_EPSILON * d
            pe = substrate.get_internal_repr_readonly(pert, frozen_rm, frozen_W)
            pf = substrate.get_internal_repr_readonly(pert, fresh_rm, fresh_W)
            diffs.append(float(np.linalg.norm((pe - base_exp) - (pf - base_fresh))))
    if not diffs:
        return None, False
    mean_diff = float(np.mean(diffs))
    return round(mean_diff, 4), bool(mean_diff > 0.05)


def compute_r3_pe_ema(substrate, obs_sample, snapshot):
    """R3 for PE-EMA substrate: compare W_action vs W_action_init Jacobian."""
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot['running_mean']
    frozen_h = snapshot['h']
    frozen_W_action = snapshot['W_action']
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W_action = substrate.W_action_init.copy()
    obs_subset = obs_sample[-R3_N_OBS:]
    rng = np.random.RandomState(42)
    diffs = []
    for obs_arr in obs_subset:
        obs_flat = obs_arr.ravel()
        dirs = rng.randn(R3_N_DIRS, len(obs_flat)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
        base_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W_action)
        base_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W_action)
        for d in dirs:
            pert = obs_flat + R3_EPSILON * d
            pe = substrate.get_internal_repr_readonly(pert, frozen_rm, frozen_h, frozen_W_action)
            pf = substrate.get_internal_repr_readonly(pert, fresh_rm, fresh_h, fresh_W_action)
            diffs.append(float(np.linalg.norm((pe - base_exp) - (pf - base_fresh))))
    if not diffs:
        return None, False
    mean_diff = float(np.mean(diffs))
    return round(mean_diff, 4), bool(mean_diff > 0.05)


def make_substrate(condition, n_actions, seed):
    if condition == 'allo':
        return AlloSoftmaxSubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'ctl':
        return ArgminCtlSubstrate(n_actions=n_actions, seed=seed)
    elif condition == 'pe_ema':
        return PeEmaArgminSubstrate(n_actions=n_actions, seed=seed)
    raise ValueError(f"Unknown condition: {condition}")


# --- Run single ---

def run_single(game_name, condition, draw, seed, n_actions, solver_level_steps):
    label = LABELS[condition]
    is_mbpp = game_name.lower().startswith('mbpp')
    print(f"  {game_name.upper()} | {label} | draw={draw} | seed={seed} ...", end='', flush=True)

    substrate = make_substrate(condition, n_actions, seed)

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    obs_store = []
    repr_log = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    steps = 0
    level = 0
    max_level = 0
    level_first_step = {}
    level_actions_log = {}
    level_start_step = 0
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

        if steps % I1_SAMPLE_FREQ == 0 and hasattr(substrate, '_last_enc') and substrate._last_enc is not None:
            repr_log.append((substrate._last_enc.copy(), level))

        if steps == I3_STEP:
            i3_counts_at_200 = action_counts.copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        # Post-step update (PE-EMA needs next_obs)
        if obs_next is not None and not fresh_episode:
            next_flat = np.asarray(obs_next, dtype=np.float32).ravel()
            delta = float(np.linalg.norm(next_flat - obs_flat))
            substrate.update_after_step(obs_next, action, delta)

        obs = obs_next

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_actions_log[cl] = steps - level_start_step
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

    # I3 (cv)
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    # I1
    i1_result = compute_i1(repr_log)

    # I4
    i4_result = {'reduction_pct': None, 'pass': False}
    if len(action_log) >= 5100:
        e100 = action_entropy(action_log[:100], n_actions)
        e5000 = action_entropy(action_log[:5000], n_actions)
        if e100 > 1e-8:
            red = (e100 - e5000) / e100 * 100.0
            i4_result = {'reduction_pct': round(red, 2), 'pass': bool(red > 10.0)}

    # I5
    i5_pass = None
    if 1 in level_actions_log and 2 in level_actions_log:
        i5_pass = bool(level_actions_log[2] < level_actions_log[1])

    # R3
    r3_val, r3_pass = None, False
    if r3_snapshot is not None and r3_obs_sample:
        if condition == 'allo':
            r3_val, r3_pass = compute_r3_allo(substrate, r3_obs_sample, r3_snapshot)
        elif condition == 'pe_ema':
            r3_val, r3_pass = compute_r3_pe_ema(substrate, r3_obs_sample, r3_snapshot)
        # CTL: no learned W, R3 = 0 by definition

    # ARC score
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    l1_step = level_first_step.get(1)

    r3_str = f"R3={r3_val:.4f}" if r3_val is not None else "R3=0(no W)"
    i3_str = f"I3cv={i3_cv:.3f}" if i3_cv is not None else "I3cv=?"
    print(f" [{label}] Lmax={max_level} | {i3_str} | {r3_str} | "
          f"L1={l1_step is not None} | {elapsed:.1f}s")

    return {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'l1_step': l1_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'I3_cv': i3_cv,
        'I1_within': i1_result.get('within'),
        'I1_between': i1_result.get('between'),
        'I1_p_value': i1_result.get('p_value'),
        'I1_pass': i1_result.get('pass', False),
        'I4_reduction_pct': i4_result['reduction_pct'],
        'I4_pass': i4_result['pass'],
        'I5_pass': i5_pass,
        'R3_jacobian_diff': r3_val,
        'R3_pass': r3_pass,
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 1253b — Catalog Revisit: Allosteric Softmax (C#19) corrected params")
    print("ALLO:  W (n_actions×256), LPL (eta_h=0.05 FIXED), adaptive softmax")
    print("CTL:   Pure argmin on visit counts (legacy baseline)")
    print("PE-EMA: pe_ema-weighted argmin from Step 1282 (confirmed composition)")
    print("KEY CHANGE vs 1253: eta_h=0.05 (was 0.01), I3_cv metric (was I3_rho)")
    print("=" * 72)
    print(f"ARC games: {ARC_GAMES} | N_DRAWS={N_DRAWS}")
    print(f"MBPP: {MBPP_GAMES} | N_DRAWS_MBPP={N_DRAWS_MBPP}")
    print(f"Central question: ALLO L1 on any game that CTL doesn't solve?")
    print()

    print("Computing solver per-level step counts...")
    solver_steps = {}
    for g in GAMES:
        try:
            solver_steps[g] = compute_solver_level_steps(g, seed=1)
            print(f"  {g.upper()}: {solver_steps[g]}")
        except Exception as e:
            print(f"  {g.upper()}: ERROR -- {e}")
            solver_steps[g] = {}
    print()

    all_results = []
    t_global = time.time()

    for game_name in GAMES:
        is_mbpp = game_name.lower().startswith('mbpp')
        n_draws = N_DRAWS_MBPP if is_mbpp else N_DRAWS

        if is_mbpp:
            n_actions = 128
        else:
            try:
                with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
                    diag = json.load(f)
                n_actions = diag.get('n_actions', 4103)
            except Exception:
                n_actions = 4103

        slv = solver_steps.get(game_name, {})
        print(f"\n{'--'*36}")
        print(f"GAME: {game_name.upper()} | n_actions={n_actions} | draws={n_draws}")
        print(f"{'--'*36}")

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition} [{LABELS[condition]}]")
            for draw in range(1, n_draws + 1):
                seed = draw * 100 + CONDITIONS.index(condition)
                result = run_single(
                    game_name=game_name, condition=condition, draw=draw, seed=seed,
                    n_actions=n_actions, solver_level_steps=slv)
                all_results.append(result)
                fname = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*72}")
    print(f"STEP 1253b COMPLETE -- {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*72}\n")

    # L1 summary (central question)
    print("L1 Results (central question: ALLO beats CTL anywhere?):")
    for game in GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            l1s = sum(1 for r in runs if r.get('L1_solved'))
            arcs = [r['arc_score'] for r in runs]
            l1_steps = [r['l1_step'] for r in runs if r.get('l1_step') is not None]
            flag = " *** L1 ACHIEVED ***" if l1s > 0 else ""
            print(f"    [{label}] L1={l1s}/{len(runs)} | arc={np.mean(arcs):.4f} | "
                  f"avg_l1_step={int(np.mean(l1_steps)) if l1_steps else 'N/A'}{flag}")

    # I3 comparison
    print("\nI3_cv Summary (ALLO vs CTL — is softmax killing coverage?):")
    n_allo_cv_kill = 0
    for game in ARC_GAMES:
        allo_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'allo']
        ctl_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'ctl']
        allo_cv = [r['I3_cv'] for r in allo_runs if r.get('I3_cv') is not None]
        ctl_cv = [r['I3_cv'] for r in ctl_runs if r.get('I3_cv') is not None]
        if allo_cv and ctl_cv:
            mean_allo = float(np.mean(allo_cv))
            mean_ctl = float(np.mean(ctl_cv))
            kill_flag = mean_allo > 3 * mean_ctl
            if kill_flag:
                n_allo_cv_kill += 1
            print(f"  {game.upper()}: ALLO cv={mean_allo:.3f} | CTL cv={mean_ctl:.3f} "
                  f"{'KILL(ALLO>3xCTL)' if kill_flag else 'ok'}")

    if n_allo_cv_kill >= 3:
        print(f"\n*** KILL: ALLO I3_cv > 3x CTL on {n_allo_cv_kill} games ***")

    # R3 summary
    print("\nR3 Summary:")
    for game in ARC_GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            r3s = [r['R3_jacobian_diff'] for r in runs if r.get('R3_jacobian_diff') is not None]
            r3_pass = sum(1 for r in runs if r.get('R3_pass'))
            mean_r3 = float(np.mean(r3s)) if r3s else None
            r3_str = f"{mean_r3:.4f}" if mean_r3 is not None else "0(no W)"
            print(f"    [{label}] R3={r3_str} | pass={r3_pass}/{len(runs)}")

    # Kill criteria check
    print("\nKill criteria check:")
    all_allo_worse = True
    for game in ARC_GAMES:
        allo_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'allo']
        ctl_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'ctl']
        allo_l1 = sum(1 for r in allo_runs if r.get('L1_solved'))
        ctl_l1 = sum(1 for r in ctl_runs if r.get('L1_solved'))
        if allo_l1 > ctl_l1:
            all_allo_worse = False
            print(f"  {game.upper()}: ALLO={allo_l1} > CTL={ctl_l1} — ALLO BEATS CTL")
        else:
            print(f"  {game.upper()}: ALLO={allo_l1} vs CTL={ctl_l1} — no ALLO advantage")

    if all_allo_worse:
        print("\n*** KILL: ALLO L1 <= CTL on ALL games. Flat W can't match argmin. ***")
        print("*** Network complexity justified. ***")
    else:
        print("\n*** PASS: ALLO beats CTL on >= 1 game. ***")
        print("*** Original step 1253 was abandoned prematurely. ***")

    with open(os.path.join(RESULTS_DIR, 'step1253b_results.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_runs': len(all_results),
                   'conditions': CONDITIONS, 'games': GAMES}, f)

    print(f"\nSTEP 1253b DONE")


if __name__ == '__main__':
    main()
