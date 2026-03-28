"""
Step 1291 — Refractory cycling with W_action activation.

One mechanism change from Step 1290: replace softmax(W_action@ext/T) with
argmax(W_action@ext - refractory), where refractory accumulates on chosen action
and decays exponentially (half-life ~7 steps).

Mechanism (REF condition):
  activation = W_action @ ext                  # allosteric signal
  effective = activation - self.refractory     # subtract accumulated suppression
  action = argmax(effective)                   # winner-take-all
  self.refractory *= 0.9                       # decay all (half-life ~7 steps)
  self.refractory[action] += 1.0              # suppress chosen action

W_action: LPL Hebbian, eta_h=0.05, NO DECAY (same as 1290 fix).
No pe_ema, no visit counts, no argmin. W_action + refractory is entire selector.

Difference from previous refractory attempts (1264-1265): multiplicative 0.8x on
LAST action only, no accumulation, no decay across all actions. This: additive
+1.0 on chosen, exponential decay on ALL, accumulated history.

Difference from 1290: argmax + refractory (deterministic, MUST try other actions
as suppression accumulates) vs softmax (stochastic, can return to same action).

Kill criteria:
  Total L1 < 10/50 AND no new game L1 → KILL
  VC33 0/5 AND FT09 ≤ 2/5 → KILL (W_action concentrates on wrong actions)
  FT09 0/5 AND no compensating gain elsewhere → KILL

Games:  ft09, ls20, vc33, tr87, sp80, sb26, tu93, cn04, cd82, lp85
Draws:  5 per condition per game = 100 runs
Budget: 10K steps, one config

Spec: Leo mail 3627, 2026-03-28.
"""
import sys, os, time, json
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

import numpy as np
from substrates.step0674 import _enc_frame

# --- Config ---
GAMES = ['ft09', 'ls20', 'vc33', 'tr87', 'sp80', 'sb26', 'tu93', 'cn04', 'cd82', 'lp85']
N_DRAWS = 5
MAX_STEPS = 10_000
MAX_SECONDS = 300

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

ETA_PRED = 0.01
PE_EMA_ALPHA = 0.05
SELECTION_ALPHA = 0.1
ETA_H = 0.05
REFRACTORY_DECAY = 0.9
REFRACTORY_BUMP = 1.0

I3_STEP = 200
I1_SAMPLE_FREQ = 100
R3_STEP = 5000
R3_N_OBS = 100
R3_N_DIRS = 20
R3_EPSILON = 0.01

PDIR = 'B:/M/the-search/experiments/results/prescriptions'
DIAG_DIR = os.path.join('B:/M/the-search/experiments', 'results', 'game_diagnostics')
RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1291')

SOLVER_PRESCRIPTIONS = {
    'ft09': ('ft09_fullchain.json', 'all_actions'),
    'ls20': ('ls20_fullchain.json', 'all_actions'),
    'vc33': ('vc33_fullchain.json', 'all_actions_encoded'),
    'tr87': ('tr87_fullchain.json', 'all_actions'),
    'sp80': ('sp80_fullchain.json', 'sequence'),
    'sb26': ('sb26_fullchain.json', 'all_actions'),
    'tu93': ('tu93_fullchain.json', 'all_actions'),
    'cn04': ('cn04_fullchain.json', 'sequence'),
    'cd82': ('cd82_fullchain.json', 'sequence'),
    'lp85': ('lp85_fullchain.json', 'full_sequence'),
}

CONDITIONS = ['refractory', 'lpl_pe_ctrl']
LABELS = {'refractory': 'REF', 'lpl_pe_ctrl': 'CTL'}


# --- Utilities ---

def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_prescription(game_name: str):
    fname, field = SOLVER_PRESCRIPTIONS[game_name]
    with open(os.path.join(PDIR, fname)) as f:
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
    ACTION_OFFSET = {'ls20': -1, 'vc33': 7}
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level = 0
    level_first_step = {}
    fresh_episode = True
    step = 0
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
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


# --- Substrate ---

class LPLREFSubstrate:
    """LPL substrate with refractory cycling (REF) or argmin+pe_ema (CTL).

    REF condition: action = argmax(W_action @ ext - refractory);
                   refractory *= 0.9; refractory[action] += 1.0.
    CTL condition: action = argmin(counts - alpha * pe_ema).
    """

    def __init__(self, n_actions: int, seed: int, use_refractory: bool = False):
        self.n_actions = n_actions
        self.use_refractory = use_refractory

        rng_w = np.random.RandomState(seed + 10000)
        self._rng = np.random.RandomState(seed)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.W_h = rng_w.randn(H_DIM, H_DIM).astype(np.float32) * 0.1
        self.W_in = rng_w.randn(H_DIM, ENC_DIM).astype(np.float32) * 0.1
        self.h = np.zeros(H_DIM, np.float32)
        scale = 1.0 / np.sqrt(float(EXT_DIM))
        W_action_init = rng_w.randn(n_actions, EXT_DIM).astype(np.float32) * scale
        self.W_action = W_action_init.copy()
        self.W_action_init = W_action_init.copy()

        # CTL-only fields (kept for unified CTL condition)
        self.action_counts = np.zeros(n_actions, np.float32)
        self.pe_ema = np.zeros(n_actions, np.float32)
        self.W_pred = rng_w.randn(ENC_DIM, ENC_DIM).astype(np.float32) * 0.01

        # REF-only field
        self.refractory = np.zeros(n_actions, np.float32)

        # Tracking
        self._sal_sum = np.zeros(n_actions, np.float64)
        self._sal_steps = 0
        self._delta_sum = np.zeros(n_actions, np.float64)
        self._delta_count = np.zeros(n_actions, np.int64)
        self._prev_enc_flow = None
        self._prev_ext = None
        self._last_action = None
        self._prev_repr = None
        self._prev_enc = None
        self.step = 0

        # Refractory + activation tracking (REF only)
        self._ref_stats = []   # [(step, mean_ref, max_ref, num_suppressed)] every 100 steps
        self._act_stats = []   # [(step, act_mean, act_std, act_max)] every 100 steps

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
        salience = self.W_action @ ext

        if self.use_refractory:
            activation = salience  # W_action @ ext already computed
            effective = activation - self.refractory
            action = int(np.argmax(effective))
            self.refractory *= REFRACTORY_DECAY
            self.refractory[action] += REFRACTORY_BUMP
            if self.step % 100 == 0:
                mean_ref = float(np.mean(self.refractory))
                max_ref = float(np.max(self.refractory))
                num_sup = int(np.sum(self.refractory > 0.1))
                act_mean = float(np.mean(activation))
                act_std = float(np.std(activation))
                act_max = float(np.max(activation))
                self._ref_stats.append((self.step, round(mean_ref, 4), round(max_ref, 4), num_sup))
                self._act_stats.append((self.step, round(act_mean, 6), round(act_std, 6), round(act_max, 6)))
        else:
            score = self.action_counts - SELECTION_ALPHA * self.pe_ema
            action = int(np.argmin(score))

        self.action_counts[action] += 1
        self._sal_sum += salience.astype(np.float64)
        self._sal_steps += 1
        self._prev_repr = salience.copy()
        self._prev_enc = enc.copy()
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
        pred_enc = self.W_pred @ self._prev_enc_flow
        pe = float(np.linalg.norm(enc_after - pred_enc))
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc_flow)
        a = self._last_action
        self.pe_ema[a] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[a] + PE_EMA_ALPHA * pe
        delta = enc_after - self._prev_enc_flow
        flow = float(np.linalg.norm(delta))
        self.W_action[self._last_action] += ETA_H * flow * self._prev_ext
        # DECAY removed from W_action � Step 1290 fix

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

    def mean_refractory_stats(self):
        """Return mean num_suppressed across all logged steps."""
        if not self._ref_stats:
            return None
        return float(np.mean([s[3] for s in self._ref_stats]))


# --- Instrumentation ---

def compute_i3_rho(action_counts_200, kb_delta):
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


def compute_i3_cv(action_counts_200, n_actions):
    counts = action_counts_200[:n_actions].astype(np.float64)
    mean = counts.mean()
    if mean < 1e-8:
        return None
    return round(float(counts.std() / mean), 6)


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
    if len(action_log) < 5100:
        return {'entropy_100': None, 'entropy_5000': None, 'reduction_pct': None, 'pass': False}
    e100 = action_entropy(action_log[:100], n_actions)
    e5000 = action_entropy(action_log[:5000], n_actions)
    if e100 < 1e-8:
        return {'entropy_100': e100, 'entropy_5000': e5000, 'reduction_pct': None, 'pass': False}
    reduction = (e100 - e5000) / e100 * 100.0
    return {'entropy_100': round(e100, 4), 'entropy_5000': round(e5000, 4),
            'reduction_pct': round(reduction, 2), 'pass': bool(reduction > 10.0)}


def compute_sal(substrate):
    if not hasattr(substrate, '_sal_steps') or substrate._sal_steps == 0:
        return {'rho': None, 'pass': False, 'n_actions_visited': 0}
    sal_mean = substrate._sal_sum / substrate._sal_steps
    n = min(substrate.n_actions, 7)
    delta_sum = substrate._delta_sum[:n]
    delta_count = substrate._delta_count[:n]
    visited = int((delta_count > 0).sum())
    if visited < 2:
        return {'rho': None, 'pass': False, 'n_actions_visited': visited}
    sal_top = sal_mean[:n]
    avg_delta = np.where(delta_count > 0, delta_sum / np.maximum(delta_count, 1), 0.0)
    rho = spearman_rho(sal_top.astype(float), avg_delta.astype(float))
    if rho is None:
        return {'rho': None, 'pass': False, 'n_actions_visited': visited}
    return {'rho': round(float(rho), 4), 'pass': bool(rho > 0.2), 'n_actions_visited': visited}


def compute_r3_jacobian(substrate, obs_sample, snapshot, n_actions):
    if not obs_sample or snapshot is None:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
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
        baseline_exp = substrate.get_internal_repr_readonly(obs_flat, frozen_rm, frozen_h, frozen_W_action)
        baseline_fresh = substrate.get_internal_repr_readonly(obs_flat, fresh_rm, fresh_h, fresh_W_action)
        for d in dirs:
            perturbed = obs_flat + R3_EPSILON * d
            pert_exp = substrate.get_internal_repr_readonly(perturbed, frozen_rm, frozen_h, frozen_W_action)
            pert_fresh = substrate.get_internal_repr_readonly(perturbed, fresh_rm, fresh_h, fresh_W_action)
            jac_exp = (pert_exp - baseline_exp) / R3_EPSILON
            jac_fresh = (pert_fresh - baseline_fresh) / R3_EPSILON
            diffs.append(float(np.linalg.norm(jac_exp - jac_fresh)))
    if not diffs:
        return {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    mean_diff = float(np.mean(diffs))
    return {'jacobian_diff': round(mean_diff, 4), 'pass': bool(mean_diff > 0.05),
            'n_obs_used': len(obs_subset)}


# --- Run single ---

def run_single(game_name, condition, draw, seed, n_actions, kb_delta, solver_level_steps):
    label = LABELS[condition]
    print(f"  {game_name.upper()} | {label} | draw={draw} | seed={seed} ...", end='', flush=True)

    use_refractory = (condition == 'refractory')
    substrate = LPLREFSubstrate(n_actions=n_actions, seed=seed, use_refractory=use_refractory)
    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    repr_log_act = []
    repr_log_enc = []
    obs_store = []

    i3_action_counts = None
    r3_snapshot = None
    r3_obs_sample = None
    pe_snapshots = {}
    PE_SNAPSHOT_STEPS = {100, 1000, 5000}

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

        if steps % I1_SAMPLE_FREQ == 0:
            if substrate._prev_repr is not None:
                repr_log_act.append((substrate._prev_repr.copy(), level))
            if substrate._prev_enc is not None:
                repr_log_enc.append((substrate._prev_enc.copy(), level))

        if steps == I3_STEP:
            i3_action_counts = substrate.action_counts.copy()

        if steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        if steps in PE_SNAPSHOT_STEPS:
            pe = substrate.pe_ema
            n7 = min(7, n_actions)
            pe_snapshots[steps] = {
                'mean': round(float(pe.mean()), 6),
                'max': round(float(pe.max()), 6),
                'top7': [round(float(pe[i]), 6) for i in range(n7)],
            }

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
        repr_log_act.append((substrate._prev_repr.copy(), level))
    if substrate._prev_enc is not None:
        repr_log_enc.append((substrate._prev_enc.copy(), level))

    i3_rho, i3_rho_pass = compute_i3_rho(i3_action_counts[:min(7, n_actions)], kb_delta) if i3_action_counts is not None else (None, None)
    i3_cv = compute_i3_cv(i3_action_counts, n_actions) if i3_action_counts is not None else None

    i1_act_result = compute_i1(repr_log_act)
    i1_enc_result = compute_i1(repr_log_enc)
    i4_result = compute_i4(action_log, n_actions)

    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    i5_pass = None
    i5_level_actions = None
    if l1_step is not None:
        i5_level_actions = {int(k): int(v) for k, v in level_actions.items() if k >= 1}
        if 1 in i5_level_actions and 2 in i5_level_actions:
            i5_pass = bool(i5_level_actions[2] < i5_level_actions[1])

    r3_result = compute_r3_jacobian(substrate, r3_obs_sample, r3_snapshot, n_actions) if r3_snapshot and r3_obs_sample else {'jacobian_diff': None, 'pass': False, 'n_obs_used': 0}
    sal_result = compute_sal(substrate)
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    arc_per_level = {}
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_step and level_first_step[lv] > 0:
            ratio = solver_step / level_first_step[lv]
            arc_per_level[int(lv)] = round(min(1.0, ratio * ratio), 6)

    mean_ref_sup = substrate.mean_refractory_stats()

    def i1_str(r, tag):
        if r['within'] is not None:
            return f"{tag}={'P' if r['pass'] else 'F'}(w={r['within']},b={r['between']})"
        return f"{tag}=null"
    n_repr_enc = len(repr_log_enc)
    n_levels_enc = len(set(lv for _, lv in repr_log_enc))
    i3_str = f"I3rho={i3_rho:.2f}/cv={i3_cv:.4f}" if i3_rho is not None and i3_cv is not None else f"I3_cv={i3_cv}"
    i4_str = f"I4={i4_result['reduction_pct']:.1f}%" if i4_result['reduction_pct'] is not None else "I4=null"
    r3_str = f"R3={r3_result['jacobian_diff']:.4f}" if r3_result['jacobian_diff'] else "R3=null"
    ref_str = f" nsup={mean_ref_sup:.1f}" if mean_ref_sup is not None else ""
    arc_str = f"ARC={arc_score:.4f}" if arc_score > 0 else "ARC=0"
    print(f" [{label}] Lmax={max_level} | {i3_str} | {i4_str} | {r3_str} |{ref_str} {arc_str} | {elapsed:.1f}s")

    counts_at_200 = i3_action_counts[:min(7, n_actions)].tolist() if i3_action_counts is not None else None

    return {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed': seed,
        'use_refractory': use_refractory,
        'mean_refractory_suppressed': round(mean_ref_sup, 2) if mean_ref_sup is not None else None,
        'refractory_stats': substrate._ref_stats if hasattr(substrate, '_ref_stats') else None,
        'activation_stats': substrate._act_stats if hasattr(substrate, '_act_stats') else None,
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
        'repr_log_enc_n': n_repr_enc,
        'repr_log_enc_levels': n_levels_enc,
        'I3_rho': round(i3_rho, 4) if i3_rho is not None else None,
        'I3_rho_pass': i3_rho_pass,
        'I3_cv': i3_cv,
        'I3_counts_at_200': counts_at_200,
        'I1_enc_within': i1_enc_result['within'],
        'I1_enc_between': i1_enc_result['between'],
        'I1_enc_p_value': i1_enc_result['p_value'],
        'I1_enc_pass': i1_enc_result['pass'],
        'I1_act_within': i1_act_result['within'],
        'I1_act_between': i1_act_result['between'],
        'I1_act_p_value': i1_act_result['p_value'],
        'I1_act_pass': i1_act_result['pass'],
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
        'pe_snapshots': {str(k): v for k, v in pe_snapshots.items()},
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("STEP 1291 — Refractory cycling with W_action activation")
    print("Mechanism: action = argmax(W_action @ ext - refractory); ref *= 0.9; ref[a] += 1.0")
    print("=" * 70)
    print(f"Games: {GAMES}")
    print(f"Draws: {N_DRAWS} per condition per game = {N_DRAWS * len(CONDITIONS) * len(GAMES)} total runs")
    print()

    print("Computing solver per-level step counts...")
    solver_level_steps_by_game = {}
    for game_name in GAMES:
        try:
            slv = compute_solver_level_steps(game_name, seed=1)
            solver_level_steps_by_game[game_name] = slv
            print(f"  {game_name.upper()}: {slv}")
        except Exception as e:
            print(f"  {game_name.upper()}: ERROR -- {e}")
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
        print(f"\n{'--'*35}")
        print(f"GAME: {game_name.upper()} | n_actions={game_n_actions}")
        print(f"{'--'*35}")

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition} [{LABELS[condition]}]")
            for draw in range(1, N_DRAWS + 1):
                seed = draw * 100 + CONDITIONS.index(condition)
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
    print(f"STEP 1291 COMPLETE -- {len(all_results)} runs in {total_elapsed:.1f}s")
    print(f"{'='*70}\n")

    print("L1 + I4 Summary (key metrics):")
    for game in GAMES:
        parts_l1 = []
        parts_i4 = []
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            l1s = sum(1 for r in runs if r.get('L1_solved') == True)
            i4s = [r['I4_reduction_pct'] for r in runs if r.get('I4_reduction_pct') is not None]
            mean_i4 = float(np.mean(i4s)) if i4s else None
            parts_l1.append(f"{label}={l1s}/5")
            parts_i4.append(f"{label}={mean_i4:.1f}%" if mean_i4 is not None else f"{label}=null")
        print(f"  {game.upper()} L1: " + " | ".join(parts_l1) + "  I4: " + " | ".join(parts_i4))

    print("\nR3 Summary:")
    for game in GAMES:
        parts = []
        for cond in CONDITIONS:
            label = LABELS[cond]
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            r3s = [r['R3_jacobian_diff'] for r in runs if r.get('R3_jacobian_diff') is not None]
            passes = sum(1 for r in runs if r.get('R3_pass') == True)
            mean_r3 = float(np.mean(r3s)) if r3s else None
            parts.append(f"{label}={mean_r3:.4f}({passes}/5)" if mean_r3 is not None else f"{label}=null")
        print(f"  {game.upper()}: " + " | ".join(parts))

    print("\nMean suppressed actions (REF condition only):")
    for game in GAMES:
        runs = [r for r in all_results if r['game'] == game and r['condition'] == 'refractory']
        sups = [r['mean_refractory_suppressed'] for r in runs if r.get('mean_refractory_suppressed') is not None]
        mean_sup = float(np.mean(sups)) if sups else None
        print(f"  {game.upper()}: {mean_sup:.1f}" if mean_sup is not None else f"  {game.upper()}: null")

    with open(os.path.join(RESULTS_DIR, 'step1291_results.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_runs': len(all_results),
                   'conditions': CONDITIONS}, f)

    print(f"\nSTEP 1291 DONE")


if __name__ == '__main__':
    main()
