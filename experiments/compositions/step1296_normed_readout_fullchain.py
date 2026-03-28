"""
Step 1296 — Phase 2b: Normalized activation in W_readout update, full PRISM chain.

Context: 1295 showed W_readout learning negligible (ro_drift≈0.000 on FT09/SP80) due to
activation scale (~0.02). Winner activation ~0.02 → update ~2e-5/element → drifts to zero.
MBPP drift=1.201 (128 actions, 78 visits/row) confirmed: mechanism works at small action space.

Fix: normalize activation before readout update.
  act_norm_vec = activation / max(||activation||, 1e-8)  → unit magnitude always
  W_readout[action] += 0.001 * pe * act_norm_vec

Architecture (all conditions):
  N=64, W_drive (64x320) Oja, W_inhibit frozen, W_pred for pe.
  W_recur: pe-gated three-factor + fill_diagonal(W_recur, -0.1)  [NEG-DIAG base, ALL]
  W_readout update (condition-dependent):
    LEARNED-RO:  W_readout[action] += 0.001 * pe * act_norm_vec  [if pe > threshold]
                 Row normalize after update.
    FROZEN-RO:   W_readout frozen (1295 NEG-DIAG/FROZEN-RO baseline).
    UNGATED-RO:  W_readout[action] += 0.001 * act_norm_vec  [always, no pe gate]
                 Row normalize after update.

Chain: full 10-game PRISM + MBPP (no preview-game restriction).
One variable vs 1295: activation normalization before readout update.

Kill criteria:
  - LEARNED-RO lock > 0.8 on FT09 → readout re-introduced lock
  - LEARNED-RO I3_cv > 3 × FROZEN-RO I3_cv on 2+ games → catastrophic concentration
  - UNGATED-RO ≈ LEARNED-RO on all metrics → pe gate unnecessary (direction change)

Predictions (HIGH confidence after scale fix):
  - ro_drift > 0.1 on FT09/LS20/SP80 (scale fix eliminates near-zero drift)
  - R3 > 0.05 on games where W_drive engages (ft09, vc33, sp80, sb26, lp85)
  - L1 on any game: LOW confidence (scale was the mechanism bug, but W_drive update unchanged)

Budget: 10 ARC × 3 cond × 5 draws = 150 + 1 MBPP × 3 cond × 5 draws = 15 → 165 runs
  ~8s/ARC + ~5s/MBPP → ~1275s ≈ 21 min (within 30-min chain cap).

Spec: Leo mails 3642 + 3644, 2026-03-28.
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

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

NETWORK_SIZE = 64
DT_SUB = 0.05
N_SUBSTEPS = 10
OJA_LR = 0.01
ETA_PRED = 0.01
W_RECUR_LR = 0.001
W_RO_LR = 0.001
PE_WINDOW = 500
NEG_DIAG_VAL = -0.1  # NEG-DIAG constraint for all conditions

SNAP_FREQ = 100
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['learned_ro', 'frozen_ro', 'ungated_ro']
LABELS = {'learned_ro': 'LRO', 'frozen_ro': 'FRO', 'ungated_ro': 'URO'}

# Known baselines from step1295 FROZEN-RO (= 1294 NEG-DIAG)
FROZEN_RO_BASELINE = {'ft09': {'lock': 0.549, 'k': 4.0, 'mi': 0.861, 'corr': -0.008}}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1296')
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


def spearman_rho(x, y):
    if len(x) < 2:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    if denom < 1e-8:
        return None
    return float(np.dot(rx, ry) / denom)


def compute_sequence_mi(winner_seq, max_lag=10):
    if len(winner_seq) < max_lag + 2:
        return None
    mi_vals = []
    for k in range(1, max_lag + 1):
        n = len(winner_seq) - k
        joint = {}; p_i = {}; p_j = {}
        for t in range(n):
            a, b = winner_seq[t], winner_seq[t + k]
            joint[(a, b)] = joint.get((a, b), 0) + 1
            p_i[a] = p_i.get(a, 0) + 1
            p_j[b] = p_j.get(b, 0) + 1
        total = float(n)
        mi = 0.0
        for (a, b), cnt in joint.items():
            pab = cnt / total; pa = p_i[a] / total; pb = p_j[b] / total
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * np.log(pab / (pa * pb))
        mi_vals.append(mi)
    return float(np.mean(mi_vals)) if mi_vals else None


def compute_attractor_count(activation_snaps):
    if len(activation_snaps) < 4:
        return None, None
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X = np.array(activation_snaps, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        X_norm = X / norms
        best_k, best_sil = 1, -1.0
        for k in range(2, min(9, len(X))):
            km = KMeans(n_clusters=k, n_init=3, random_state=0, max_iter=100)
            labels = km.fit_predict(X_norm)
            if len(set(labels)) > 1:
                sil = float(silhouette_score(X_norm, labels, metric='cosine'))
                if sil > best_sil:
                    best_sil, best_k = sil, k
        return best_k, round(best_sil, 4)
    except Exception:
        return None, None


def compute_attractor_state_corr(activation_snaps, snap_steps, total_steps):
    if len(activation_snaps) < 8:
        return None
    try:
        from sklearn.cluster import KMeans
        X = np.array(activation_snaps, dtype=np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        X_norm = X / norms
        km = KMeans(n_clusters=3, n_init=3, random_state=0, max_iter=100)
        labels = km.fit_predict(X_norm).astype(float)
        norm_steps = np.array(snap_steps, dtype=float) / max(total_steps, 1)
        return spearman_rho(labels, norm_steps)
    except Exception:
        return None


def compute_state_discrimination(activation_snaps, levels):
    if len(activation_snaps) < 4:
        return None, None, None
    rng = np.random.RandomState(42)
    pairs = list(zip(activation_snaps, levels))
    within, between = [], []
    n = len(pairs)
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        d = cosine_dist(pairs[i][0], pairs[j][0])
        (within if pairs[i][1] == pairs[j][1] else between).append(d)
    if len(within) < 2 or len(between) < 2:
        return None, None, None
    wm = float(np.mean(within)); bm = float(np.mean(between))
    return round(wm, 4), round(bm, 4), bool(wm < bm)


# --- Substrate ---

class NormedReadoutSubstrate:
    """N=64 NEG-DIAG competitive network with activation-normalized W_readout learning.

    All conditions use NEG-DIAG base (fill_diagonal(W_recur, -0.1) always).

    Key change vs step1295: activation normalized to unit magnitude before readout update.
      act_norm_vec = activation / ||activation||  → eliminates scale sensitivity.

    W_readout conditions:
      learned_ro:  W_readout[action] += W_RO_LR * pe * act_norm_vec  [if pe > threshold]
      frozen_ro:   W_readout frozen (1295 FROZEN-RO / 1294 NEG-DIAG baseline)
      ungated_ro:  W_readout[action] += W_RO_LR * act_norm_vec  [always, no pe gate]
    """

    def __init__(self, n_actions, seed, mode='learned_ro'):
        self.n_actions = n_actions
        self.mode = mode
        rng_w = np.random.RandomState(seed + 10000)

        # Encoding pipeline
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        # Recurrent state
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # Prediction model (for pe)
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01

        # Competitive network
        self.W_drive = rng_w.randn(NETWORK_SIZE, EXT_DIM).astype(np.float32) * 0.01
        self.W_drive_init = self.W_drive.copy()
        self.W_inhibit = rng_w.uniform(0.5, 1.5, (NETWORK_SIZE, NETWORK_SIZE)).astype(np.float32)
        np.fill_diagonal(self.W_inhibit, 0.0)

        # W_readout: (n_actions × NETWORK_SIZE)
        self.W_readout = rng_w.randn(n_actions, NETWORK_SIZE).astype(np.float32) * 0.01
        self.W_readout_init = self.W_readout.copy()

        self.activation = np.zeros(NETWORK_SIZE, np.float32)
        self.prev_activation = np.zeros(NETWORK_SIZE, np.float32)

        # W_recur: NEG-DIAG base for ALL conditions
        self.W_recur = rng_w.randn(NETWORK_SIZE, NETWORK_SIZE).astype(np.float32) * 0.01
        np.fill_diagonal(self.W_recur, NEG_DIAG_VAL)  # -0.1 diagonal always

        self._prev_enc = None
        self._last_action = None
        self.step = 0

        # pe tracking
        self._pe_window = []
        self._pe_threshold = 0.0
        self.pe_spike_count = 0
        self.pe_total_count = 0

        # Capacity tracking
        self.winner_seq = []
        self.activation_snaps = []
        self._snap_levels = []
        self._snap_steps = []
        self._surprise_steps = []
        self._winner_change_steps = set()

        # W_readout change tracking
        self.ro_update_count = 0

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_drive):
        """R3: encode obs with experienced W_drive vs fresh W_drive."""
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_drive @ ext

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_drive': self.W_drive.copy(),
            'W_readout': self.W_readout.copy(),
            'step': self.step,
        }

    def compute_ro_drift(self):
        """Frobenius norm of W_readout - W_readout_init."""
        diff = self.W_readout - self.W_readout_init
        return round(float(np.linalg.norm(diff, 'fro')), 4)

    def process(self, obs_raw, current_level=0):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])

        # Compute pe
        pe_scalar = 0.0
        if self._prev_enc is not None:
            pred = self.W_pred @ self._prev_enc
            pe_scalar = float(np.linalg.norm(enc - pred))
            pred_err = enc - pred
            self.W_pred += ETA_PRED * np.outer(pred_err, self._prev_enc)
            self._pe_window.append(pe_scalar)
            if len(self._pe_window) > PE_WINDOW:
                self._pe_window.pop(0)
            self._pe_threshold = float(np.median(self._pe_window))
            self.pe_total_count += 1
            if pe_scalar > self._pe_threshold:
                self.pe_spike_count += 1
                self._surprise_steps.append(self.step)

        # Network dynamics (NEG-DIAG W_recur for all conditions)
        prev_act = self.prev_activation
        act = self.activation.copy()
        for _ in range(N_SUBSTEPS):
            drive = self.W_drive @ ext
            recurrent = self.W_recur @ prev_act
            inhibit = self.W_inhibit @ act
            d_act = -act + np.maximum(drive + recurrent - inhibit, 0.0)
            act = np.maximum(act + DT_SUB * d_act, 0.0)

        prev_winner = int(np.argmax(self.activation)) if self.step > 0 else -1
        self.activation = act
        winner = int(np.argmax(self.activation))

        if self.step > 0 and winner != prev_winner:
            self._winner_change_steps.add(self.step)

        # Action selection from W_readout (all conditions)
        action_scores = self.W_readout @ self.activation
        action = int(np.argmax(action_scores))

        # Oja plasticity on W_drive (winner row)
        win_act = float(self.activation[winner])
        if win_act > 1e-8:
            self.W_drive[winner] += OJA_LR * win_act * (ext - self.W_drive[winner])

        # W_recur: pe-gated three-factor + NEG-DIAG constraint (ALL conditions)
        if pe_scalar > self._pe_threshold and self._pe_threshold > 0.0 and win_act > 1e-8:
            dW = W_RECUR_LR * np.outer(self.activation, prev_act) * pe_scalar
            self.W_recur += dW
            norms = np.linalg.norm(self.W_recur, axis=1, keepdims=True)
            self.W_recur /= np.maximum(norms, 1.0)
        # Always enforce NEG-DIAG after any potential update
        np.fill_diagonal(self.W_recur, NEG_DIAG_VAL)

        # W_readout update (condition-dependent)
        # KEY CHANGE vs 1295: normalize activation to unit magnitude before update
        act_norm_val = float(np.linalg.norm(self.activation))
        if act_norm_val > 1e-8:
            act_norm_vec = self.activation / act_norm_val  # unit magnitude signal
            if self.mode == 'learned_ro':
                if pe_scalar > self._pe_threshold and self._pe_threshold > 0.0:
                    self.W_readout[action] += W_RO_LR * pe_scalar * act_norm_vec
                    norms = np.linalg.norm(self.W_readout, axis=1, keepdims=True)
                    self.W_readout /= np.maximum(norms, 1.0)
                    self.ro_update_count += 1
            elif self.mode == 'ungated_ro':
                self.W_readout[action] += W_RO_LR * act_norm_vec
                norms = np.linalg.norm(self.W_readout, axis=1, keepdims=True)
                self.W_readout /= np.maximum(norms, 1.0)
                self.ro_update_count += 1

        # Tracking
        self.prev_activation = self.activation.copy()
        self.winner_seq.append(winner)
        if self.step % SNAP_FREQ == 0:
            self.activation_snaps.append(self.activation.copy())
            self._snap_levels.append(current_level)
            self._snap_steps.append(self.step)

        self._prev_enc = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def on_level_transition(self):
        pass

    def compute_recur_sparsity(self):
        total = NETWORK_SIZE * NETWORK_SIZE
        active = float(np.sum(np.abs(self.W_recur) > 0.01))
        return round(active / total, 4)


# --- R3 computation ---

def compute_r3(substrate, obs_sample, snapshot):
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot['running_mean']
    frozen_h = snapshot['h']
    frozen_W = snapshot['W_drive']
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W = substrate.W_drive_init.copy()
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
            diffs.append(float(np.linalg.norm((pe - base_exp) - (pf - base_fresh))))
    if not diffs:
        return None, False
    mean_diff = float(np.mean(diffs))
    return round(mean_diff, 4), bool(mean_diff > 0.05)


# --- Run single ---

def run_single(game_name, condition, draw, seed, n_actions, solver_level_steps):
    label = LABELS[condition]
    is_mbpp = game_name.lower().startswith('mbpp')
    print(f"  {game_name.upper()} | {label} | draw={draw} | seed={seed} ...", end='', flush=True)

    substrate = NormedReadoutSubstrate(n_actions=n_actions, seed=seed, mode=condition)

    env = make_game(game_name)
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
    level_start_step = 0
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

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr, current_level=level) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
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

    # I3
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    # I4
    i4_result = {'reduction_pct': None, 'pass': False}
    if len(action_log) >= 5100:
        e100 = action_entropy(action_log[:100], n_actions)
        e5000 = action_entropy(action_log[:5000], n_actions)
        if e100 > 1e-8:
            red = (e100 - e5000) / e100 * 100.0
            i4_result = {'reduction_pct': round(red, 2), 'pass': bool(red > 10.0)}

    # R3
    r3_val, r3_pass = compute_r3(substrate, r3_obs_sample, r3_snapshot)

    # ARC score
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    l1_step = level_first_step.get(1)

    # Capacity diagnostics
    winner_seq = substrate.winner_seq
    positive_lock_fraction = None
    num_active_winners = None
    sequence_mi = None
    if winner_seq:
        counts_w = {}
        for w in winner_seq:
            counts_w[w] = counts_w.get(w, 0) + 1
        positive_lock_fraction = round(max(counts_w.values()) / len(winner_seq), 4)
        num_active_winners = len(counts_w)
        sequence_mi = compute_sequence_mi(winner_seq, max_lag=10)
        if sequence_mi is not None:
            sequence_mi = round(sequence_mi, 6)

    # Attractor diagnostics
    attractor_k, attractor_sil = None, None
    state_disc_within, state_disc_between, state_disc_pass = None, None, None
    attractor_state_corr = None
    recur_sparsity = substrate.compute_recur_sparsity()
    ro_drift = substrate.compute_ro_drift()
    ro_updates = substrate.ro_update_count

    if substrate.activation_snaps:
        attractor_k, attractor_sil = compute_attractor_count(substrate.activation_snaps)
        if len(set(substrate._snap_levels)) >= 2:
            state_disc_within, state_disc_between, state_disc_pass = compute_state_discrimination(
                substrate.activation_snaps, substrate._snap_levels)
        attractor_state_corr = compute_attractor_state_corr(
            substrate.activation_snaps, substrate._snap_steps, steps)
        if attractor_state_corr is not None:
            attractor_state_corr = round(attractor_state_corr, 4)

    # Print line
    lock_str = f"lock={positive_lock_fraction:.3f}" if positive_lock_fraction is not None else "lock=?"
    i3_str = f"I3cv={i3_cv:.3f}" if i3_cv is not None else "I3cv=?"
    r3_str = f"R3={r3_val:.4f}" if r3_val is not None else "R3=?"
    sil_str = f"sil={attractor_sil}(k={attractor_k})" if attractor_sil is not None else ""
    ro_str = f"ro_drift={ro_drift:.3f}(n={ro_updates})"
    print(f" [{label}] Lmax={max_level} | {lock_str} | {i3_str} | {r3_str} "
          f"{sil_str} {ro_str} | {elapsed:.1f}s")

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
        'I4_reduction_pct': i4_result['reduction_pct'],
        'I4_pass': i4_result['pass'],
        'R3_jacobian_diff': r3_val,
        'R3_pass': r3_pass,
        'positive_lock_fraction': positive_lock_fraction,
        'num_active_winners': num_active_winners,
        'sequence_mi': sequence_mi,
        'attractor_k': attractor_k,
        'attractor_silhouette': attractor_sil,
        'state_disc_within': state_disc_within,
        'state_disc_between': state_disc_between,
        'state_disc_pass': state_disc_pass,
        'attractor_state_corr': attractor_state_corr,
        'recur_sparsity': recur_sparsity,
        'ro_drift': ro_drift,
        'ro_update_count': ro_updates,
        'pe_spike_count': substrate.pe_spike_count,
        'pe_total_count': substrate.pe_total_count,
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 1296 — Phase 2b: Normalized activation in W_readout update")
    print("LEARNED-RO:  W_readout[action] += 0.001 * pe * act_norm_vec  [pe-gated]")
    print("FROZEN-RO:   W_readout frozen (1295 FROZEN-RO baseline)")
    print("UNGATED-RO:  W_readout[action] += 0.001 * act_norm_vec  [always]")
    print("ALL conditions: NEG-DIAG base (W_recur diag=-0.1 always)")
    print("KEY CHANGE vs 1295: activation normalized before readout update")
    print("=" * 72)
    print(f"ARC games: {ARC_GAMES} | N_DRAWS={N_DRAWS}")
    print(f"MBPP: {MBPP_GAMES} | N_DRAWS_MBPP={N_DRAWS_MBPP}")
    print(f"Known FROZEN-RO (1295 NEG-DIAG): FT09 lock={FROZEN_RO_BASELINE['ft09']['lock']}, "
          f"k={FROZEN_RO_BASELINE['ft09']['k']}, mi={FROZEN_RO_BASELINE['ft09']['mi']}")
    print(f"Prediction (HIGH confidence): ro_drift > 0.1 on all games (scale fix)")
    print(f"Prediction (LOW confidence): L1 on any game (W_drive signal unchanged)")
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
    print(f"STEP 1296 COMPLETE -- {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*72}\n")

    # L1 summary (the central question)
    print("L1 Results (central question — any L1 from LEARNED-RO?):")
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

    # R3 summary
    print("\nR3 Summary (HIGH confidence: ro_drift now measurable):")
    for game in ARC_GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            r3s = [r['R3_jacobian_diff'] for r in runs if r.get('R3_jacobian_diff') is not None]
            r3_pass = sum(1 for r in runs if r.get('R3_pass'))
            mean_r3 = float(np.mean(r3s)) if r3s else None
            r3_str = f"{mean_r3:.4f}" if mean_r3 is not None else "?"
            print(f"    [{label}] R3={r3_str} | pass={r3_pass}/{len(runs)}")

    # ro_drift summary (key diagnostic for scale fix)
    print("\nro_drift Summary (PRIMARY: confirms scale fix worked):")
    for game in GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            drifts = [r['ro_drift'] for r in runs if r.get('ro_drift') is not None]
            updates = [r['ro_update_count'] for r in runs if r.get('ro_update_count') is not None]
            mean_drift = float(np.mean(drifts)) if drifts else None
            mean_updates = float(np.mean(updates)) if updates else None
            drift_display = f"{mean_drift:.3f}" if mean_drift is not None else "?"
            updates_display = f"{mean_updates:.1f}" if mean_updates is not None else "?"
            prev_1295 = "(1295: ~0.000)" if cond == 'learned_ro' and game in ['ft09', 'sp80', 'ls20'] else ""
            print(f"    [{label}] ro_drift={drift_display} (n_updates={updates_display}) {prev_1295}")

    # Lock + MI summary
    print("\nLock + MI Summary:")
    for game in GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            locks = [r['positive_lock_fraction'] for r in runs if r.get('positive_lock_fraction') is not None]
            mis = [r['sequence_mi'] for r in runs if r.get('sequence_mi') is not None]
            sils = [r['attractor_silhouette'] for r in runs if r.get('attractor_silhouette') is not None]
            ks = [r['attractor_k'] for r in runs if r.get('attractor_k') is not None]
            mean_lock = float(np.mean(locks)) if locks else None
            mean_mi = float(np.mean(mis)) if mis else None
            mean_sil = float(np.mean(sils)) if sils else None
            mean_k = float(np.mean(ks)) if ks else None
            lock_display = f"{mean_lock:.3f}" if mean_lock is not None else "?"
            mi_display = f"{mean_mi:.4f}" if mean_mi is not None else "?"
            sil_display = f"{mean_sil:.3f}(k={mean_k:.1f})" if mean_sil is not None else "?"
            print(f"    [{label}] lock={lock_display} | mi={mi_display} | sil={sil_display}")

    # Kill criteria check
    print("\nKill criteria check:")
    any_kill = False
    for game in ARC_GAMES:
        lro_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'learned_ro']
        fro_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'frozen_ro']
        if not lro_runs:
            continue
        lro_locks = [r['positive_lock_fraction'] for r in lro_runs if r.get('positive_lock_fraction') is not None]
        lro_i3 = [r['I3_cv'] for r in lro_runs if r.get('I3_cv') is not None]
        fro_i3 = [r['I3_cv'] for r in fro_runs if r.get('I3_cv') is not None]

        mean_lro_lock = float(np.mean(lro_locks)) if lro_locks else None
        mean_lro_i3 = float(np.mean(lro_i3)) if lro_i3 else None
        mean_fro_i3 = float(np.mean(fro_i3)) if fro_i3 else None

        lock_kill = mean_lro_lock is not None and game == 'ft09' and mean_lro_lock > 0.8
        i3_kill = (mean_lro_i3 is not None and mean_fro_i3 is not None and
                   mean_lro_i3 > 3 * mean_fro_i3)

        lock_display = f"{mean_lro_lock:.3f}" if mean_lro_lock is not None else "?"
        i3_display = f"{mean_lro_i3:.3f}" if mean_lro_i3 is not None else "?"
        if lock_kill or i3_kill:
            any_kill = True
        print(f"  {game.upper()} LRO: lock={lock_display} {'KILL(lock>0.8)' if lock_kill else 'ok'} | "
              f"I3cv={i3_display} {'KILL(I3 explode)' if i3_kill else 'ok'}")

    if any_kill:
        print("\n*** KILL criteria triggered — see above ***")
    else:
        print("\nNo kill criteria triggered.")

    # UNGATED vs LEARNED comparison
    print("\nUNGATED-RO vs LEARNED-RO (is pe gate necessary?):")
    n_eq_games = 0
    for game in ARC_GAMES:
        for metric in ['positive_lock_fraction', 'sequence_mi', 'attractor_silhouette']:
            lro_vals = [r.get(metric) for r in all_results
                        if r['game'] == game and r['condition'] == 'learned_ro' and r.get(metric) is not None]
            uro_vals = [r.get(metric) for r in all_results
                        if r['game'] == game and r['condition'] == 'ungated_ro' and r.get(metric) is not None]
            if lro_vals and uro_vals:
                diff = abs(np.mean(lro_vals) - np.mean(uro_vals))
                eq_flag = " [~EQUAL, gate unnecessary]" if diff < 0.02 else ""
                if eq_flag:
                    n_eq_games += 1
                print(f"  {game.upper()} {metric}: LRO={np.mean(lro_vals):.4f} | "
                      f"URO={np.mean(uro_vals):.4f}{eq_flag}")

    with open(os.path.join(RESULTS_DIR, 'step1296_results.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_runs': len(all_results),
                   'conditions': CONDITIONS, 'games': GAMES,
                   'frozen_ro_baseline': FROZEN_RO_BASELINE}, f)

    print(f"\nSTEP 1296 DONE")


if __name__ == '__main__':
    main()
