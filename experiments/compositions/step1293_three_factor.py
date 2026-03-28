"""
Step 1293 — Three-Factor Plasticity: Prediction-Gated Recurrent Learning.

Tests whether prediction-gated three-factor plasticity on W_recur produces
meaningful internal structure (game-phase-aware attractors).

Architecture (all conditions):
  N = 64 neurons
  W_drive   (64 x 320): input coupling, Oja plasticity
  W_inhibit (64 x 64):  lateral inhibition, frozen U(0.5, 1.5), diag=0
  W_readout (n_actions x 64): frozen (L1=0 expected, diagnostic only)
  W_recur   (64 x 64):  recurrent connections (condition-dependent update)
  W_pred    (256 x 256): prediction model for pe computation

Per game step:
  pe = ||enc_t - W_pred @ enc_{t-1}||   (scalar prediction error)
  W_pred updated: W_pred += ETA_PRED * outer(enc_t - W_pred@enc_{t-1}, enc_{t-1})
  N_SUBSTEPS of: drive = W_drive@ext + W_recur@prev_activation - W_inhibit@act
  action = argmax(W_readout @ activation)
  W_drive[winner] += Oja update
  W_recur update per condition:
    THREE_FACTOR: dW = W_RECUR_LR * outer(activation, prev_activation) * pe  [if pe > median]
    UNGATED:      dW = W_RECUR_LR * outer(activation, prev_activation)
    FROZEN:       no update (W_recur = 0 always)

Diagnostics (beyond 1292b):
  - attractor_state_corr: Spearman(cluster_id, normalized_step) — do attractors track game phase?
  - recur_sparsity: fraction of |W_recur[i,j]| > 0.01 at end of run
  - surprise_reorg_ratio: rate of winner change after pe spikes vs baseline

Games: ls20, ft09, sp80
Draws: 4 per condition per game = 36 runs (budget: ~288s < 5 min cap)
Note: 5 draws would exceed 5-min cap (~360s). Using 4 draws.

Spec: Leo mail 3634, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# --- Config ---
GAMES = ['ls20', 'ft09', 'sp80']
N_DRAWS = 4
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
PE_WINDOW = 500       # rolling window for pe median threshold

SNAP_FREQ = 100
R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['three_factor', 'ungated', 'frozen']
LABELS = {'three_factor': '3F', 'ungated': 'UNG', 'frozen': 'FRZ'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1293')
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
    """Spearman correlation between cluster label and normalized step index.
    If > 0.3: attractors track game phase (temporal structure)."""
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

class ThreeFactorSubstrate:
    """N=64 competitive network with optional three-factor recurrent plasticity.

    mode: 'three_factor' | 'ungated' | 'frozen'
      three_factor: W_recur += W_RECUR_LR * outer(act, prev_act) * pe  [if pe > median]
      ungated:      W_recur += W_RECUR_LR * outer(act, prev_act)
      frozen:       W_recur = 0 (no recurrence, same as 1292b MINIMAL)
    """

    def __init__(self, n_actions, seed, mode='three_factor'):
        self.n_actions = n_actions
        self.mode = mode
        rng_w = np.random.RandomState(seed + 10000)

        # Encoding pipeline
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
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
        self.W_readout = rng_w.randn(n_actions, NETWORK_SIZE).astype(np.float32) * 0.01
        self.activation = np.zeros(NETWORK_SIZE, np.float32)
        self.prev_activation = np.zeros(NETWORK_SIZE, np.float32)

        # Recurrent weights
        if mode == 'frozen':
            self.W_recur = np.zeros((NETWORK_SIZE, NETWORK_SIZE), np.float32)
        else:
            self.W_recur = rng_w.randn(NETWORK_SIZE, NETWORK_SIZE).astype(np.float32) * 0.01

        self._prev_enc = None
        self._last_action = None
        self.step = 0

        # pe tracking for threshold
        self._pe_window = []          # rolling pe values for median
        self._pe_threshold = 0.0
        self.pe_spike_count = 0
        self.pe_total_count = 0

        # Capacity tracking
        self.winner_seq = []
        self.activation_snaps = []
        self._snap_levels = []
        self._snap_steps = []

        # Surprise-reorganization tracking
        self._surprise_steps = []     # steps where pe > threshold
        self._winner_change_steps = set()  # steps where winner changed

    def _centered_encode(self, obs):
        x = _enc_frame(np.asarray(obs, dtype=np.float32))
        self.n_obs += 1
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def get_internal_repr_readonly(self, obs_raw, frozen_rm, frozen_h, frozen_W_drive):
        """R3: use W_drive @ ext (same as 1292b for comparability)."""
        enc = _enc_frame(np.asarray(obs_raw, dtype=np.float32)) - frozen_rm
        h_new = np.tanh(self.W_h @ frozen_h + self.W_in @ enc)
        ext = np.concatenate([enc, h_new])
        return frozen_W_drive @ ext

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

        # Compute pe (only if we have prev_enc)
        pe_scalar = 0.0
        if self._prev_enc is not None:
            pred = self.W_pred @ self._prev_enc
            pe_scalar = float(np.linalg.norm(enc - pred))
            # Update W_pred
            pred_err = enc - pred
            self.W_pred += ETA_PRED * np.outer(pred_err, self._prev_enc)
            # Update pe rolling window and threshold
            self._pe_window.append(pe_scalar)
            if len(self._pe_window) > PE_WINDOW:
                self._pe_window.pop(0)
            self._pe_threshold = float(np.median(self._pe_window))
            self.pe_total_count += 1
            if pe_scalar > self._pe_threshold:
                self.pe_spike_count += 1
                self._surprise_steps.append(self.step)

        # Network dynamics (N_SUBSTEPS) with recurrent input from prev_activation
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

        # Action from frozen readout
        action_scores = self.W_readout @ self.activation
        action = int(np.argmax(action_scores))

        # Oja plasticity on W_drive (winner row)
        win_act = float(self.activation[winner])
        if win_act > 1e-8:
            self.W_drive[winner] += OJA_LR * win_act * (ext - self.W_drive[winner])

        # W_recur update
        if self.mode != 'frozen' and win_act > 1e-8:
            if self.mode == 'three_factor':
                if pe_scalar > self._pe_threshold and self._pe_threshold > 0.0:
                    dW = W_RECUR_LR * np.outer(self.activation, prev_act) * pe_scalar
                    self.W_recur += dW
                    # Row normalization: cap each row norm at 1.0
                    norms = np.linalg.norm(self.W_recur, axis=1, keepdims=True)
                    self.W_recur /= np.maximum(norms, 1.0)
            elif self.mode == 'ungated':
                dW = W_RECUR_LR * np.outer(self.activation, prev_act)
                self.W_recur += dW
                norms = np.linalg.norm(self.W_recur, axis=1, keepdims=True)
                self.W_recur /= np.maximum(norms, 1.0)

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
        """Fraction of |W_recur[i,j]| > 0.01 (non-negligible connections)."""
        if self.mode == 'frozen':
            return 0.0
        total = NETWORK_SIZE * NETWORK_SIZE
        active = float(np.sum(np.abs(self.W_recur) > 0.01))
        return round(active / total, 4)

    def compute_surprise_reorg_ratio(self):
        """Rate of winner change within 10 steps after pe spike vs baseline rate."""
        if not self._surprise_steps or len(self.winner_seq) < 20:
            return None
        changes = self._winner_change_steps
        total_steps = len(self.winner_seq)
        baseline_rate = len(changes) / max(total_steps, 1)

        reorg_count = 0
        for s in self._surprise_steps:
            # Check if winner changes in next 10 steps
            window_end = min(s + 11, total_steps)
            window_changes = sum(1 for t in range(s + 1, window_end) if t in changes)
            if window_changes > 0:
                reorg_count += 1

        spike_reorg_rate = reorg_count / len(self._surprise_steps)
        return round(spike_reorg_rate / max(baseline_rate * 10, 1e-8), 4)


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
    print(f"  {game_name.upper()} | {label} | draw={draw} | seed={seed} ...", end='', flush=True)

    substrate = ThreeFactorSubstrate(n_actions=n_actions, seed=seed, mode=condition)

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
    surprise_reorg = substrate.compute_surprise_reorg_ratio()

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
    lock_str = f"lock={positive_lock_fraction:.2f}" if positive_lock_fraction is not None else "lock=?"
    cyc_str = f"cyc={num_active_winners}" if num_active_winners is not None else "cyc=?"
    i3_str = f"I3cv={i3_cv:.3f}" if i3_cv is not None else "I3cv=?"
    i4_str = f"I4={i4_result['reduction_pct']:.1f}%" if i4_result['reduction_pct'] is not None else "I4=?"
    r3_str = f"R3={r3_val:.4f}" if r3_val is not None else "R3=?"
    sil_str = f"sil={attractor_sil}(k={attractor_k})" if attractor_sil is not None else ""
    corr_str = f"corr={attractor_state_corr}" if attractor_state_corr is not None else ""
    spr_str = f"spr={recur_sparsity:.3f}" if recur_sparsity is not None else ""
    print(f" [{label}] Lmax={max_level} | {lock_str} | {cyc_str} | {i3_str} | {i4_str} | "
          f"{r3_str} {sil_str} {corr_str} {spr_str} | {elapsed:.1f}s")

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
        # New 1293 diagnostics
        'attractor_state_corr': attractor_state_corr,
        'recur_sparsity': recur_sparsity,
        'surprise_reorg_ratio': surprise_reorg,
        'pe_spike_count': substrate.pe_spike_count,
        'pe_total_count': substrate.pe_total_count,
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1293 — Three-Factor Plasticity on Recurrent Weights")
    print("THREE-FACTOR: W_recur += lr * outer(act, prev_act) * pe  [if pe > median]")
    print("UNGATED:      W_recur += lr * outer(act, prev_act)  [always]")
    print("FROZEN:       W_recur = 0  [1292b baseline]")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Draws: {N_DRAWS} per condition per game = {N_DRAWS * len(CONDITIONS) * len(GAMES)} total runs")
    print(f"Note: N_DRAWS=4 (not 5) to stay within 5-min budget cap")
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
    print(f"STEP 1293 COMPLETE -- {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    # Summary
    print("Capacity + Attractor Summary:")
    for game in GAMES:
        print(f"\n  {game.upper()}:")
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            sils = [r['attractor_silhouette'] for r in runs if r.get('attractor_silhouette') is not None]
            locks = [r['positive_lock_fraction'] for r in runs if r.get('positive_lock_fraction') is not None]
            mis = [r['sequence_mi'] for r in runs if r.get('sequence_mi') is not None]
            corrs = [r['attractor_state_corr'] for r in runs if r.get('attractor_state_corr') is not None]
            sprs = [r['recur_sparsity'] for r in runs if r.get('recur_sparsity') is not None]
            reorgs = [r['surprise_reorg_ratio'] for r in runs if r.get('surprise_reorg_ratio') is not None]
            l1s = sum(1 for r in runs if r.get('L1_solved'))
            mean_sil = float(np.mean(sils)) if sils else None
            mean_lock = float(np.mean(locks)) if locks else None
            mean_mi = float(np.mean(mis)) if mis else None
            mean_corr = float(np.mean(corrs)) if corrs else None
            mean_spr = float(np.mean(sprs)) if sprs else None
            mean_reorg = float(np.mean(reorgs)) if reorgs else None
            ks = [r['attractor_k'] for r in runs if r.get('attractor_k') is not None]
            mean_k = float(np.mean(ks)) if ks else None
            print(f"    [{label}] L1={l1s}/{N_DRAWS} | "
                  f"sil={mean_sil:.3f}(k={mean_k:.1f})" if mean_sil is not None else f"    [{label}] L1={l1s}/{N_DRAWS} | sil=?", end='')
            print(f" | lock={mean_lock:.2f}" if mean_lock is not None else " | lock=?", end='')
            print(f" | mi={mean_mi:.4f}" if mean_mi is not None else " | mi=?", end='')
            print(f" | corr={mean_corr:.3f}" if mean_corr is not None else " | corr=?", end='')
            print(f" | spr={mean_spr:.3f}" if mean_spr is not None else " | spr=?", end='')
            print(f" | reorg={mean_reorg:.3f}" if mean_reorg is not None else " | reorg=?")

    print("\nKill criteria check:")
    for game in GAMES:
        tf_runs = [r for r in all_results if r['game'] == game and r['condition'] == 'three_factor']
        sils = [r['attractor_silhouette'] for r in tf_runs if r.get('attractor_silhouette') is not None]
        locks = [r['positive_lock_fraction'] for r in tf_runs if r.get('positive_lock_fraction') is not None]
        ks = [r['attractor_k'] for r in tf_runs if r.get('attractor_k') is not None]
        mean_k = float(np.mean(ks)) if ks else None
        mean_lock = float(np.mean(locks)) if locks else None
        k1 = mean_k is not None and mean_k <= 1.5
        k2 = mean_lock is not None and mean_lock > 0.8
        print(f"  {game.upper()} 3F: k={mean_k:.1f} {'KILL(collapse)' if k1 else 'ok'} | "
              f"lock={mean_lock:.2f} {'KILL(lock)' if k2 else 'ok'}" if mean_k is not None else
              f"  {game.upper()} 3F: no data")

    # UNGATED vs THREE-FACTOR comparison
    print("\nUNGATED vs THREE-FACTOR comparison (is pe gate meaningful?):")
    for game in GAMES:
        for metric in ['attractor_silhouette', 'positive_lock_fraction', 'sequence_mi']:
            tf_vals = [r.get(metric) for r in all_results
                       if r['game'] == game and r['condition'] == 'three_factor'
                       and r.get(metric) is not None]
            ung_vals = [r.get(metric) for r in all_results
                        if r['game'] == game and r['condition'] == 'ungated'
                        and r.get(metric) is not None]
            if tf_vals and ung_vals:
                print(f"  {game.upper()} {metric}: 3F={np.mean(tf_vals):.4f} | UNG={np.mean(ung_vals):.4f}")

    with open(os.path.join(RESULTS_DIR, 'step1293_results.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_runs': len(all_results),
                   'conditions': CONDITIONS, 'games': GAMES}, f)

    print(f"\nSTEP 1293 DONE")


if __name__ == '__main__':
    main()
