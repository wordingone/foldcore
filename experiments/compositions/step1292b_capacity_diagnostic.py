"""
Step 1292b — Minimal Reflexive Network: Capacity Diagnostic.

Separate from Step 1292 (winnerless competition on action space).
This tests CAPACITY — whether a competitive network (N=64, fixed) develops
internal representations from game observations.

Architecture (MINIMAL condition):
  N = 64 fixed network neurons
  W_drive  (64 x 320): input coupling (encoding → network), Oja plasticity
  W_inhibit (64 x 64): lateral inhibition, frozen, initialized U(0.5, 1.5), diag=0
  W_readout (n_actions x 64): network → action space, FROZEN random

  Per game step: 10 sub-steps of competitive dynamics
    drive = W_drive @ ext
    inhibit = W_inhibit @ activation
    d_act = -activation + max(drive - inhibit, 0)
    activation = max(activation + dt * d_act, 0)
  action = argmax(W_readout @ activation)
  winner = argmax(activation)
  W_drive[winner] += OJA_LR * activation[winner] * (ext - W_drive[winner])

Control (LINEAR condition): W_action (n_actions x 320), LPL Hebbian (eta=0.05),
  action = argmax(W_action @ ext). Known to collapse (Prop 30).

Capacity diagnostics (primary):
  1. positive_lock_fraction: max_winner_count / total_steps
  2. num_active_winners: distinct winners over run
  3. sequence_mi: mean MI between winner at t and winner at t+k (k=1..10)
  4. attractor_count: silhouette score on activation snapshots (MINIMAL only)
  5. state_discrimination: within vs between level cosine dist on activation (MINIMAL only)

Games: ls20 (KB, 7 actions), ft09 (click, 4103), sp80 (click, uniform response)
Draws: 5 per condition per game = 30 runs
Budget: 10K steps, 5-min cap

Spec: Leo mail 3630, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# --- Config ---
GAMES = ['ls20', 'ft09', 'sp80']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

NETWORK_SIZE = 64   # fixed for MINIMAL
DT_SUB = 0.05       # LV sub-step integration
N_SUBSTEPS = 10     # sub-steps per game step
OJA_LR = 0.01       # Oja plasticity rate

ETA_LINEAR = 0.05   # LPL Hebbian for LINEAR
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.05
SELECTION_ALPHA = 0.1

SNAP_FREQ = 100      # activation snapshot every N steps
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['minimal', 'linear']
LABELS = {'minimal': 'MIN', 'linear': 'LIN'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1292b')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'
DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')

SOLVER_PRESCRIPTIONS = {
    'ls20': ('ls20_fullchain.json', 'all_actions'),
    'ft09': ('ft09_fullchain.json', 'all_actions'),
    'sp80': ('sp80_fullchain.json', 'sequence'),
}


# --- Utilities ---

def make_game(game_name):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_prescription(game_name):
    fname, field = SOLVER_PRESCRIPTIONS[game_name]
    with open(os.path.join(PDIR, fname)) as f:
        d = json.load(f)
    return d[field]


def compute_solver_level_steps(game_name, seed=1):
    prescription = load_prescription(game_name)
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


def permutation_test_i1(within_dists, between_dists, n_perms=500, seed=0):
    rng = np.random.RandomState(seed)
    within = np.array(within_dists, dtype=np.float32)
    between = np.array(between_dists, dtype=np.float32)
    if len(within) == 0 or len(between) == 0:
        return 1.0
    observed = float(np.mean(between) - np.mean(within))
    all_dists = np.concatenate([within, between])
    n_w = len(within)
    count = sum(
        1 for _ in range(n_perms)
        if float(np.mean(all_dists[rng.permutation(len(all_dists))[n_w:]]) -
                 np.mean(all_dists[rng.permutation(len(all_dists))[:n_w]])) >= observed
    )
    return count / n_perms


def compute_sequence_mi(winner_seq, max_lag=10):
    """Mean MI between winner at t and winner at t+k, k=1..max_lag."""
    if len(winner_seq) < max_lag + 2:
        return None
    mi_vals = []
    for k in range(1, max_lag + 1):
        n = len(winner_seq) - k
        joint = {}
        p_i = {}; p_j = {}
        for t in range(n):
            a, b = winner_seq[t], winner_seq[t + k]
            joint[(a, b)] = joint.get((a, b), 0) + 1
            p_i[a] = p_i.get(a, 0) + 1
            p_j[b] = p_j.get(b, 0) + 1
        total = float(n)
        mi = 0.0
        for (a, b), cnt in joint.items():
            pab = cnt / total
            pa = p_i[a] / total
            pb = p_j[b] / total
            if pab > 0 and pa > 0 and pb > 0:
                mi += pab * np.log(pab / (pa * pb))
        mi_vals.append(mi)
    return float(np.mean(mi_vals)) if mi_vals else None


def compute_attractor_count(activation_snaps):
    """Silhouette score on activation snapshots for k=2..8. Best k reported."""
    if len(activation_snaps) < 4:
        return None, None
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        X = np.array(activation_snaps, dtype=np.float32)
        # Normalize rows
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


def compute_state_discrimination(activation_snaps, levels):
    """Within vs between level cosine distance on activation snapshots."""
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


# --- Substrates ---

class MinimalSubstrate:
    """N=64 competitive network with Oja plasticity on W_drive."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self._rng = np.random.RandomState(seed)

        # Encoding pipeline (shared)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        # Competitive network
        self.W_drive = rng_w.randn(NETWORK_SIZE, EXT_DIM).astype(np.float32) * 0.01
        self.W_drive_init = self.W_drive.copy()
        self.W_inhibit = rng_w.uniform(0.5, 1.5, (NETWORK_SIZE, NETWORK_SIZE)).astype(np.float32)
        np.fill_diagonal(self.W_inhibit, 0.0)
        self.W_readout = rng_w.randn(n_actions, NETWORK_SIZE).astype(np.float32) * 0.01
        self.activation = np.zeros(NETWORK_SIZE, np.float32)

        self._prev_enc = None
        self._prev_ext = None
        self._last_action = None
        self._last_winner = None
        self.step = 0

        # Capacity tracking
        self.winner_seq = []
        self.activation_snaps = []  # (activation, level) every SNAP_FREQ steps
        self._snap_levels = []

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_drive):
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_drive @ ext  # (NETWORK_SIZE,)

    def get_state(self):
        return {
            'running_mean': self.running_mean.copy(),
            'h': self.h.copy(),
            'W_drive': self.W_drive.copy(),
            'step': self.step,
        }

    def process(self, obs_raw, current_level=0):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])

        # Run N_SUBSTEPS of competitive dynamics
        act = self.activation.copy()
        for _ in range(N_SUBSTEPS):
            drive = self.W_drive @ ext
            inhibit = self.W_inhibit @ act
            d_act = -act + np.maximum(drive - inhibit, 0.0)
            act = np.maximum(act + DT_SUB * d_act, 0.0)
        self.activation = act

        # Action from readout (frozen)
        action_scores = self.W_readout @ self.activation
        action = int(np.argmax(action_scores))
        winner = int(np.argmax(self.activation))

        # Oja plasticity on winner row of W_drive
        win_act = float(self.activation[winner])
        if win_act > 1e-8:
            self.W_drive[winner] += OJA_LR * win_act * (ext - self.W_drive[winner])

        # Tracking
        self.winner_seq.append(winner)
        if self.step % SNAP_FREQ == 0:
            self.activation_snaps.append(self.activation.copy())
            self._snap_levels.append(current_level)

        self._prev_enc = enc.copy()
        self._prev_ext = ext.copy()
        self._last_action = action
        self._last_winner = winner
        self.step += 1
        return action

    def on_level_transition(self):
        pass


class LinearSubstrate:
    """W_action (n_actions x 320), argmax selection, LPL Hebbian update. Known to collapse."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        rng_w = np.random.RandomState(seed + 10000)
        self._rng = np.random.RandomState(seed)

        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)

        self.W_action = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * 0.01
        self.W_action_init = self.W_action.copy()
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01
        self.pe_ema = np.zeros(n_actions, np.float32)
        self.action_counts = np.zeros(n_actions, np.float32)

        self._prev_enc = None
        self._prev_ext = None
        self._prev_enc_flow = None
        self._last_action = None
        self.step = 0

        # Positive lock tracking
        self.winner_seq = []  # action sequence

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

    def process(self, obs_raw, current_level=0):
        enc = self._centered_encode(obs_raw)
        self.h = np.tanh(self.W_h @ self.h + self.W_in @ enc)
        ext = np.concatenate([enc, self.h])

        # argmax selection (Step 1264 style — known to collapse)
        scores = self.W_action @ ext
        action = int(np.argmax(scores))

        self.action_counts[action] += 1
        self.winner_seq.append(action)

        self._prev_enc = enc.copy()
        self._prev_ext = ext.copy()
        self._prev_enc_flow = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_flow(self, next_obs_raw):
        if self._prev_enc_flow is None or self._last_action is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean
        pred_enc = self.W_pred @ self._prev_enc_flow
        pe = float(np.linalg.norm(enc_after - pred_enc))
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc_flow)
        a = self._last_action
        self.pe_ema[a] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[a] + PE_EMA_ALPHA * pe
        delta = enc_after - self._prev_enc_flow
        flow = float(np.linalg.norm(delta))
        self.W_action[self._last_action] += ETA_LINEAR * flow * self._prev_ext

    def on_level_transition(self):
        self._prev_enc_flow = None


# --- R3 computation ---

def compute_r3(substrate, obs_sample, snapshot, condition):
    if not obs_sample or snapshot is None:
        return None, False
    if condition == 'minimal':
        frozen_rm = snapshot['running_mean']
        frozen_h = snapshot['h']
        frozen_W = snapshot['W_drive']
        fresh_W = substrate.W_drive_init.copy()
    else:
        frozen_rm = snapshot['running_mean']
        frozen_h = snapshot['h']
        frozen_W = snapshot['W_action']
        fresh_W = substrate.W_action_init.copy()
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
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

def run_single(game_name, condition, draw, seed, n_actions, solver_level_steps,
               kb_delta=None):
    label = LABELS[condition]
    print(f"  {game_name.upper()} | {label} | draw={draw} | seed={seed} ...", end='', flush=True)

    if condition == 'minimal':
        substrate = MinimalSubstrate(n_actions=n_actions, seed=seed)
    else:
        substrate = LinearSubstrate(n_actions=n_actions, seed=seed)

    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    obs_store = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    repr_log_enc = []  # (enc, level) for I1

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
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps % 100 == 0 and hasattr(substrate, '_prev_enc') and substrate._prev_enc is not None:
            repr_log_enc.append((substrate._prev_enc.copy(), level))

        if steps == 200:
            i3_counts_at_200 = substrate.action_counts.copy() if hasattr(substrate, 'action_counts') else None

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        action = substrate.process(obs_arr, current_level=level) % n_actions
        action_log.append(action)

        obs, reward, done, info = env.step(action)
        steps += 1

        obs_flat = obs_arr.ravel()
        if obs is not None and not fresh_episode:
            if hasattr(substrate, 'update_flow'):
                substrate.update_flow(obs)

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
    r3_val, r3_pass = compute_r3(substrate, r3_obs_sample, r3_snapshot, condition)

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

    # MINIMAL-only diagnostics
    attractor_k, attractor_sil = None, None
    state_disc_within, state_disc_between, state_disc_pass = None, None, None
    if condition == 'minimal' and hasattr(substrate, 'activation_snaps') and substrate.activation_snaps:
        attractor_k, attractor_sil = compute_attractor_count(substrate.activation_snaps)
        if len(set(substrate._snap_levels)) >= 2:
            state_disc_within, state_disc_between, state_disc_pass = compute_state_discrimination(
                substrate.activation_snaps, substrate._snap_levels)

    # Print line
    lock_str = f"lock={positive_lock_fraction:.2f}" if positive_lock_fraction is not None else "lock=?"
    cyc_str = f"cyc={num_active_winners}" if num_active_winners is not None else "cyc=?"
    i3_str = f"I3cv={i3_cv:.3f}" if i3_cv is not None else "I3cv=?"
    i4_str = f"I4={i4_result['reduction_pct']:.1f}%" if i4_result['reduction_pct'] is not None else "I4=?"
    r3_str = f"R3={r3_val:.4f}" if r3_val is not None else "R3=?"
    sil_str = f"sil={attractor_sil}(k={attractor_k})" if attractor_sil is not None else ""
    print(f" [{label}] Lmax={max_level} | {lock_str} | {cyc_str} | {i3_str} | {i4_str} | {r3_str} {sil_str} | {elapsed:.1f}s")

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
        # Capacity diagnostics
        'positive_lock_fraction': positive_lock_fraction,
        'num_active_winners': num_active_winners,
        'sequence_mi': sequence_mi,
        'attractor_k': attractor_k,
        'attractor_silhouette': attractor_sil,
        'state_disc_within': state_disc_within,
        'state_disc_between': state_disc_between,
        'state_disc_pass': state_disc_pass,
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1292b — Minimal Reflexive Network: Capacity Diagnostic")
    print(f"MINIMAL: N={NETWORK_SIZE} competitive network, Oja plasticity")
    print("LINEAR: W_action argmax + LPL Hebbian (known collapse baseline)")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Draws: {N_DRAWS} per condition per game = {N_DRAWS * 2 * len(GAMES)} total runs")
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
        try:
            diag = {}
            with open(os.path.join(DIAG_DIR, f'{game_name.lower()}_diagnostic.json')) as f:
                diag = json.load(f)
            n_actions = diag.get('n_actions', 4103)
        except Exception:
            n_actions = 4103

        slv = solver_steps.get(game_name, {})
        print(f"\n{'--'*35}")
        print(f"GAME: {game_name.upper()} | n_actions={n_actions}")
        print(f"{'--'*35}")

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition} [{LABELS[condition]}]")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + CONDITIONS.index(condition)
                result = run_single(
                    game_name=game_name, condition=condition, draw=draw, seed=seed,
                    n_actions=n_actions, solver_level_steps=slv)
                all_results.append(result)
                fname = os.path.join(RESULTS_DIR, f"{game_name}_{condition}_draw{draw}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*70}")
    print(f"STEP 1292b COMPLETE -- {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    print("Capacity Diagnostics Summary:")
    for game in GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            locks = [r['positive_lock_fraction'] for r in runs if r.get('positive_lock_fraction') is not None]
            cycs = [r['num_active_winners'] for r in runs if r.get('num_active_winners') is not None]
            mis = [r['sequence_mi'] for r in runs if r.get('sequence_mi') is not None]
            sils = [r['attractor_silhouette'] for r in runs if r.get('attractor_silhouette') is not None]
            l1s = sum(1 for r in runs if r.get('L1_solved'))
            mean_lock = float(np.mean(locks)) if locks else None
            mean_cyc = float(np.mean(cycs)) if cycs else None
            mean_mi = float(np.mean(mis)) if mis else None
            mean_sil = float(np.mean(sils)) if sils else None
            lock_str = f"lock={mean_lock:.2f}" if mean_lock is not None else "lock=?"
            cyc_str = f"cyc={mean_cyc:.1f}" if mean_cyc is not None else "cyc=?"
            mi_str = f"mi={mean_mi:.4f}" if mean_mi is not None else "mi=?"
            sil_str = f"sil={mean_sil:.3f}" if mean_sil is not None else "sil=N/A"
            print(f"    [{label}] L1={l1s}/5 | {lock_str} | {cyc_str} | {mi_str} | {sil_str}")

    print("\nL1 + I4 + R3 Summary:")
    for game in GAMES:
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            l1s = sum(1 for r in runs if r.get('L1_solved'))
            i4s = [r['I4_reduction_pct'] for r in runs if r.get('I4_reduction_pct') is not None]
            r3s = [r['R3_jacobian_diff'] for r in runs if r.get('R3_jacobian_diff') is not None]
            mean_i4 = float(np.mean(i4s)) if i4s else None
            mean_r3 = float(np.mean(r3s)) if r3s else None
            i4_str = f"I4={mean_i4:.1f}%" if mean_i4 is not None else "I4=?"
            r3_str = f"R3={mean_r3:.4f}" if mean_r3 is not None else "R3=?"
            print(f"  {game.upper()} [{label}]: L1={l1s}/5 | {i4_str} | {r3_str}")

    with open(os.path.join(RESULTS_DIR, 'step1292b_results.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_runs': len(all_results),
                   'conditions': CONDITIONS, 'games': GAMES}, f)

    print(f"\nSTEP 1292b DONE")


if __name__ == '__main__':
    main()
