"""
Step 1298 — Full PRISM Baseline Sweep (Jun directive, Leo mail 3649).

Three reference conditions across 10 ARC games + MBPP, 20 draws each.
This becomes THE reference for all future experiments.

Conditions:
  RANDOM  — uniform random action selection. The floor.
  ARGMIN  — pure visit-count argmin. Coverage ceiling. R2-violating, RHAE-dead.
  PE-EMA  — Step 1282 composition (argmin + 0.1*pe_ema, LPL eta_h=0.05). Best we have.

Protocol:
  - 11 chain positions: 10 ARC games + 1 MBPP
  - 20 draws per game per condition
  - Each draw: TWO episodes on same substrate (seed_A then seed_B, no reset between)
    Episode A = first exposure; Episode B = second exposure (substrate keeps state from A)
  - 3 conditions × 11 games × 20 draws = 660 substrate instances, 1320 episodes total

Measurements:
  Standard: R3 (episode A), I3_cv (episode A step 200), L1/L2/max_level (both episodes)
  Progression (new, from experiment-integrity.md):
    - RHAE: (actions_optimal / actions_taken)², averaged across levels
    - Second-exposure speedup: steps_to_L1(A) vs steps_to_L1(B). 20 pairs per game.
    - Post-transition KL: action dist 100 steps before vs 100 steps after L1 (episode A).
    - MBPP char stats: char freq at step 100 vs step 1000 (KL from uniform).

Budget: ~660 substrate instances × ~2 × ~4s avg = ~88 min (≈ 100 min with overhead).
Jun directive: Jun-authorized override of 30-min chain cap.

Spec: Leo mail 3649, 2026-03-28.
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

N_DRAWS = 20
MAX_STEPS = 10_000      # per episode
MAX_SECONDS = 300       # per episode (5-min cap)

ENC_DIM = 256
H_DIM = 64
EXT_DIM = ENC_DIM + H_DIM  # 320

# PE-EMA hyperparameters (from step 1282 / step 1253b)
ETA_H_FLOW = 0.05
ETA_PRED = 0.01
PE_EMA_ALPHA = 0.05
SELECTION_ALPHA = 0.1
DECAY = 0.001

R3_STEP = 5000
R3_N_OBS = 50
R3_N_DIRS = 20
R3_EPSILON = 0.01

CONDITIONS = ['random', 'argmin', 'pe_ema']
LABELS = {'random': 'RAND', 'argmin': 'ARGMIN', 'pe_ema': 'PEEMA'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1298')
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


def compute_rhae(level_first_steps, solver_level_steps, total_actions):
    """RHAE: (actions_optimal / actions_taken)^2, averaged across levels."""
    scores = []
    for lv, solver_step in solver_level_steps.items():
        if lv in level_first_steps and level_first_steps[lv] > 0:
            # actions_optimal = solver_step (minimum actions to reach level)
            # actions_taken = level_first_steps[lv]
            ratio = solver_step / level_first_steps[lv]
            scores.append(min(1.0, ratio * ratio))
    return float(np.mean(scores)) if scores else 0.0


def compute_kl_from_uniform(counts, n):
    """KL divergence from uniform distribution."""
    total = counts.sum()
    if total == 0:
        return None
    probs = counts / total
    uniform = 1.0 / n
    kl = 0.0
    for p in probs:
        if p > 0:
            kl += p * np.log(p / uniform)
    return float(kl)


def compute_post_transition_kl(action_log, l1_step, n_actions, window=100):
    """KL divergence of action distribution: 100 steps before vs 100 steps after L1."""
    if l1_step is None or l1_step < window:
        return None
    pre = action_log[max(0, l1_step - window):l1_step]
    post = action_log[l1_step:min(len(action_log), l1_step + window)]
    if len(pre) < 10 or len(post) < 10:
        return None
    pre_counts = np.zeros(n_actions, np.float32)
    post_counts = np.zeros(n_actions, np.float32)
    for a in pre:
        if 0 <= a < n_actions:
            pre_counts[a] += 1
    for a in post:
        if 0 <= a < n_actions:
            post_counts[a] += 1
    pre_total = pre_counts.sum()
    post_total = post_counts.sum()
    if pre_total == 0 or post_total == 0:
        return None
    pre_p = (pre_counts + 1e-8) / (pre_total + 1e-8 * n_actions)
    post_p = (post_counts + 1e-8) / (post_total + 1e-8 * n_actions)
    kl = float(np.sum(post_p * np.log(post_p / pre_p)))
    return round(kl, 4)


# --- Substrates ---

class RandomSubstrate:
    """Uniform random action selection. The floor baseline."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        self.step = 0

    def process(self, obs_raw):
        self.step += 1
        return int(self._rng.randint(self.n_actions))

    def on_level_transition(self):
        pass

    def get_state(self):
        return {}

    def get_internal_repr_readonly(self, obs_raw, *args):
        return np.zeros(1, np.float32)

    @property
    def W_action_init(self):
        return None


class ArgminSubstrate:
    """Pure visit-count argmin. Coverage ceiling. R2-violating, RHAE-dead."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._visit_counts = np.zeros(n_actions, np.float32)
        self.step = 0

    def process(self, obs_raw):
        action = int(np.argmin(self._visit_counts))
        self._visit_counts[action] += 1
        self.step += 1
        return action

    def on_level_transition(self):
        pass

    def get_state(self):
        return {}

    def get_internal_repr_readonly(self, obs_raw, *args):
        return np.zeros(1, np.float32)

    @property
    def W_action_init(self):
        return None


class PeEmaSubstrate:
    """PE-EMA composition from Step 1282 (best non-random selector).

    W_action: (n_actions × 320) = (n_actions × [ENC_DIM + H_DIM])
    Selection: argmin(visit_counts - 0.1 * pe_ema)
    Update: LPL Hebbian flow update + pe_ema tracking.
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
        self._prev_ext_for_update = None
        self._last_action = None
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
        self._prev_ext_for_update = ext.copy()
        score = self._visit_counts - SELECTION_ALPHA * self.pe_ema
        action = int(np.argmin(score))
        self._visit_counts[action] += 1
        self._prev_enc = enc.copy()
        self._last_action = action
        self.step += 1
        return action

    def update_after_step(self, next_obs_raw, action, delta):
        if self._prev_enc is None or action is None:
            return
        next_obs = np.asarray(next_obs_raw, dtype=np.float32)
        enc_after = _enc_frame(next_obs) - self.running_mean
        pred_enc = self.W_pred @ self._prev_enc
        pe = float(np.linalg.norm(enc_after - pred_enc))
        pred_error = enc_after - pred_enc
        self.W_pred += ETA_PRED * np.outer(pred_error, self._prev_enc)
        self.pe_ema[action] = (1.0 - PE_EMA_ALPHA) * self.pe_ema[action] + PE_EMA_ALPHA * pe
        enc_delta = enc_after - self._prev_enc
        flow = float(np.linalg.norm(enc_delta))
        if self._prev_ext_for_update is not None:
            self.W_action[action] += ETA_H_FLOW * flow * self._prev_ext_for_update
            self.W_action *= (1.0 - DECAY)

    def on_level_transition(self):
        self._prev_enc = None
        self._prev_ext_for_update = None


def make_substrate(condition, n_actions, seed):
    if condition == 'random':
        return RandomSubstrate(n_actions, seed)
    elif condition == 'argmin':
        return ArgminSubstrate(n_actions, seed)
    elif condition == 'pe_ema':
        return PeEmaSubstrate(n_actions, seed)
    else:
        raise ValueError(f"Unknown condition: {condition}")


# --- R3 computation ---

def compute_r3(substrate, obs_sample, snapshot):
    """R3 for PE-EMA substrate."""
    if not obs_sample or snapshot is None:
        return None, False
    frozen_rm = snapshot.get('running_mean')
    frozen_h = snapshot.get('h')
    frozen_W = snapshot.get('W_action')
    if frozen_rm is None or frozen_h is None or frozen_W is None:
        return 0.0, False
    fresh_rm = np.zeros(ENC_DIM, np.float32)
    fresh_h = np.zeros(H_DIM, np.float32)
    fresh_W = substrate.W_action_init.copy()
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


# --- Run episode ---

def run_episode(game_name, substrate, condition, seed, n_actions, solver_level_steps,
                take_r3_snapshot=False, is_mbpp=False):
    """Run one episode (10K steps) on existing substrate.

    Returns: dict with episode results + optional r3_snapshot for later computation.
    """
    env = make_game(game_name)
    obs = env.reset(seed=seed)

    action_log = []
    obs_store = []
    r3_snapshot = None
    r3_obs_sample = None
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    # MBPP char tracking
    mbpp_chars_100 = None
    mbpp_chars_1000 = None

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
        obs_store.append(obs_arr)
        if len(obs_store) > 200:
            obs_store.pop(0)

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()
        if take_r3_snapshot and steps == R3_STEP:
            r3_snapshot = substrate.get_state()
            r3_obs_sample = list(obs_store)

        # MBPP character stats
        if is_mbpp:
            if steps == 100:
                mbpp_chars_100 = action_counts.copy()
            if steps == 1000:
                mbpp_chars_1000 = action_counts.copy()

        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if hasattr(substrate, 'update_after_step') and obs_next is not None:
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

    # I3
    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    l1_step = level_first_step.get(1)
    l2_step = level_first_step.get(2)
    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    rhae = compute_rhae(level_first_step, solver_level_steps, steps)

    # Post-transition KL (if L1 achieved)
    post_trans_kl = compute_post_transition_kl(action_log, l1_step, n_actions)

    # MBPP char stats
    mbpp_kl_100 = None
    mbpp_kl_1000 = None
    if is_mbpp:
        if mbpp_chars_100 is not None:
            mbpp_kl_100 = compute_kl_from_uniform(mbpp_chars_100, n_actions)
        if mbpp_chars_1000 is not None:
            mbpp_kl_1000 = compute_kl_from_uniform(mbpp_chars_1000, n_actions)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'L1_solved': bool(l1_step is not None),
        'l1_step': l1_step,
        'l2_step': l2_step,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(rhae, 6),
        'I3_cv': i3_cv,
        'post_transition_kl': post_trans_kl,
        'mbpp_kl_at_100': round(mbpp_kl_100, 4) if mbpp_kl_100 is not None else None,
        'mbpp_kl_at_1000': round(mbpp_kl_1000, 4) if mbpp_kl_1000 is not None else None,
        '_r3_snapshot': r3_snapshot,
        '_r3_obs_sample': r3_obs_sample,
    }


# --- Run single draw (2 episodes) ---

def run_draw(game_name, condition, draw, seed_a, seed_b, n_actions, solver_level_steps):
    label = LABELS[condition]
    is_mbpp = game_name.lower().startswith('mbpp')
    print(f"  {game_name.upper()} | {label} | draw={draw} | seedA={seed_a} seedB={seed_b} ...",
          end='', flush=True)
    t_draw_start = time.time()

    substrate = make_substrate(condition, n_actions, seed_a)

    # Episode A: first exposure
    ep_a = run_episode(game_name, substrate, condition, seed_a, n_actions, solver_level_steps,
                       take_r3_snapshot=True, is_mbpp=is_mbpp)

    # Episode B: second exposure (same substrate, no reset — carries state from A)
    ep_b = run_episode(game_name, substrate, condition, seed_b, n_actions, solver_level_steps,
                       take_r3_snapshot=False, is_mbpp=False)

    draw_elapsed = time.time() - t_draw_start

    # R3 computation (from episode A snapshot)
    r3_val, r3_pass = 0.0, False
    if condition == 'pe_ema':
        r3_val, r3_pass = compute_r3(substrate, ep_a['_r3_obs_sample'], ep_a['_r3_snapshot'])

    # Second-exposure speedup
    speedup_ratio = None
    if ep_a['l1_step'] is not None and ep_b['l1_step'] is not None:
        # speedup_ratio = A / B. > 1 means B solved faster (second exposure helped).
        speedup_ratio = round(ep_a['l1_step'] / ep_b['l1_step'], 4)

    l1_a = ep_a['L1_solved']
    l1_b = ep_b['L1_solved']
    l1_both = l1_a and l1_b
    l1_only_a = l1_a and not l1_b
    l1_only_b = not l1_a and l1_b

    print(f" [{label}] A:Lmax={ep_a['max_level']} B:Lmax={ep_b['max_level']} | "
          f"I3cv={ep_a['I3_cv'] or '?':.3f} | R3={r3_val:.4f} | "
          f"speedup={'?' if speedup_ratio is None else f'{speedup_ratio:.2f}x'} | "
          f"{draw_elapsed:.1f}s")

    return {
        'game': game_name.lower(),
        'condition': condition,
        'draw': draw,
        'seed_a': seed_a,
        'seed_b': seed_b,
        'draw_elapsed_seconds': round(draw_elapsed, 2),

        # Episode A metrics (primary)
        'A_steps_taken': ep_a['steps_taken'],
        'A_max_level': ep_a['max_level'],
        'A_L1_solved': ep_a['L1_solved'],
        'A_l1_step': ep_a['l1_step'],
        'A_l2_step': ep_a['l2_step'],
        'A_arc_score': ep_a['arc_score'],
        'A_RHAE': ep_a['RHAE'],
        'A_I3_cv': ep_a['I3_cv'],
        'A_post_transition_kl': ep_a['post_transition_kl'],
        'A_mbpp_kl_at_100': ep_a['mbpp_kl_at_100'],
        'A_mbpp_kl_at_1000': ep_a['mbpp_kl_at_1000'],

        # Episode B metrics (second exposure)
        'B_steps_taken': ep_b['steps_taken'],
        'B_max_level': ep_b['max_level'],
        'B_L1_solved': ep_b['L1_solved'],
        'B_l1_step': ep_b['l1_step'],
        'B_l2_step': ep_b['l2_step'],
        'B_arc_score': ep_b['arc_score'],
        'B_RHAE': ep_b['RHAE'],
        'B_I3_cv': ep_b['I3_cv'],

        # Cross-episode metrics
        'speedup_ratio': speedup_ratio,  # A/B: > 1 = second exposure faster
        'L1_both_episodes': l1_both,
        'L1_only_A': l1_only_a,
        'L1_only_B': l1_only_b,

        # R3 (episode A)
        'R3_jacobian_diff': r3_val,
        'R3_pass': r3_pass,

        # Compatibility fields (for comparisons with previous experiments)
        'L1_solved': ep_a['L1_solved'],
        'L2_solved': ep_a['l2_step'] is not None,
        'max_level': ep_a['max_level'],
        'arc_score': ep_a['arc_score'],
        'RHAE': ep_a['RHAE'],
        'I3_cv': ep_a['I3_cv'],
    }


# --- Main ---

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("STEP 1298 — Full PRISM Baseline Sweep (Jun directive)")
    print("RANDOM:  uniform random. The floor.")
    print("ARGMIN:  visit-count argmin. Coverage ceiling. R2-violating.")
    print("PE-EMA:  Step 1282 composition. Best non-random selector.")
    print("Protocol: 20 draws × 2 episodes per draw (seed_A + seed_B, same substrate)")
    print("=" * 72)
    print(f"Games: {GAMES}")
    print(f"N_DRAWS={N_DRAWS} | MAX_STEPS={MAX_STEPS}/episode | MAX_SECONDS={MAX_SECONDS}/episode")
    print(f"Total substrate instances: {N_DRAWS * len(CONDITIONS) * len(GAMES)}")
    print(f"Total episodes: {N_DRAWS * len(CONDITIONS) * len(GAMES) * 2}")
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
        print(f"GAME: {game_name.upper()} | n_actions={n_actions} | draws={N_DRAWS}")
        print(f"{'--'*36}")

        for condition in CONDITIONS:
            print(f"\n  Condition: {condition} [{LABELS[condition]}]")
            for draw in range(1, N_DRAWS + 1):
                seed_a = draw * 100
                seed_b = draw * 100 + 50
                result = run_draw(
                    game_name=game_name, condition=condition, draw=draw,
                    seed_a=seed_a, seed_b=seed_b,
                    n_actions=n_actions, solver_level_steps=slv)
                all_results.append(result)
                fname = os.path.join(RESULTS_DIR,
                                     f"{game_name}_{condition}_draw{draw:02d}.json")
                with open(fname, 'w') as f:
                    json.dump(result, f, indent=2)

    total_elapsed = time.time() - t_global
    print(f"\n{'='*72}")
    print(f"STEP 1298 COMPLETE -- {len(all_results)} draws in {total_elapsed:.1f}s")
    print(f"{'='*72}\n")

    # Summary table
    print("SUMMARY TABLE — THE REFERENCE:")
    print(f"{'GAME':5} | {'COND':6} | {'L1_A':6} | {'L1_B':6} | {'SPEEDUP':8} | {'I3cv':6} | {'RHAE':10} | {'R3':7}")
    print("-" * 70)
    for game in GAMES:
        for cond in CONDITIONS:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            n = len(runs)
            l1_a = sum(1 for r in runs if r.get('A_L1_solved'))
            l1_b = sum(1 for r in runs if r.get('B_L1_solved'))
            speedups = [r['speedup_ratio'] for r in runs if r.get('speedup_ratio') is not None]
            i3cv_vals = [r.get('A_I3_cv') or 0 for r in runs]
            rhae_vals = [r.get('A_RHAE') or 0 for r in runs]
            r3_vals = [r.get('R3_jacobian_diff') or 0 for r in runs]
            mean_speedup = sum(speedups)/len(speedups) if speedups else None
            mean_i3cv = sum(i3cv_vals)/n
            mean_rhae = sum(rhae_vals)/n
            mean_r3 = sum(r3_vals)/n
            speedup_str = f"{mean_speedup:.2f}x" if mean_speedup is not None else "N/A (0 L1 pairs)"
            print(f"{game:5} | {cond:6} | {l1_a:3}/{n} | {l1_b:3}/{n} | {speedup_str:8} | "
                  f"{mean_i3cv:6.3f} | {mean_rhae:.2e} | {mean_r3:.4f}")

    # Second-exposure speedup summary
    print("\nSecond-Exposure Speedup (pairs where BOTH A and B reached L1):")
    for cond in CONDITIONS:
        label = LABELS[cond]
        all_speedups = []
        for game in GAMES:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            all_speedups.extend([r['speedup_ratio'] for r in runs
                                  if r.get('speedup_ratio') is not None])
        if all_speedups:
            mean_sp = sum(all_speedups) / len(all_speedups)
            n_pairs = len(all_speedups)
            print(f"  [{label}] n_pairs={n_pairs} | mean_speedup={mean_sp:.3f}x "
                  f"({'faster' if mean_sp > 1.0 else 'no speedup'})")
            if mean_sp > 1.0:
                print(f"  *** SECOND-EXPOSURE SPEEDUP DETECTED for {label} ***")
        else:
            print(f"  [{label}] 0 pairs with both A+B L1 — no speedup measurable")

    # RHAE summary
    print("\nRHAE Summary (all should be ≈0 — establishing the floor):")
    for cond in CONDITIONS:
        label = LABELS[cond]
        runs_all = [r for r in all_results if r['condition'] == cond]
        rhae_vals = [r.get('A_RHAE') or 0 for r in runs_all]
        if rhae_vals:
            mean_rhae = sum(rhae_vals) / len(rhae_vals)
            max_rhae = max(rhae_vals)
            print(f"  [{label}] mean_RHAE={mean_rhae:.2e} | max_RHAE={max_rhae:.2e}")

    # MBPP char stats
    print("\nMBPP Character Stats (KL from uniform — learning signal even without L1):")
    for cond in CONDITIONS:
        mbpp_runs = [r for r in all_results
                     if r['game'] == 'mbpp_0' and r['condition'] == cond]
        kl_100 = [r.get('A_mbpp_kl_at_100') for r in mbpp_runs
                  if r.get('A_mbpp_kl_at_100') is not None]
        kl_1000 = [r.get('A_mbpp_kl_at_1000') for r in mbpp_runs
                   if r.get('A_mbpp_kl_at_1000') is not None]
        if kl_100 and kl_1000:
            print(f"  [{LABELS[cond]}] step100 KL={sum(kl_100)/len(kl_100):.4f} | "
                  f"step1000 KL={sum(kl_1000)/len(kl_1000):.4f} "
                  f"({'increasing' if sum(kl_1000)/len(kl_1000) > sum(kl_100)/len(kl_100) else 'stable/decreasing'})")

    # Save consolidated summary
    summary = {}
    for game in GAMES:
        summary[game] = {}
        for cond in CONDITIONS:
            runs = [r for r in all_results if r['game'] == game and r['condition'] == cond]
            if not runs:
                continue
            n = len(runs)
            l1_a = sum(1 for r in runs if r.get('A_L1_solved'))
            l1_b = sum(1 for r in runs if r.get('B_L1_solved'))
            speedups = [r['speedup_ratio'] for r in runs if r.get('speedup_ratio') is not None]
            rhae_vals = [r.get('A_RHAE') or 0 for r in runs]
            i3cv_vals = [r.get('A_I3_cv') or 0 for r in runs]
            r3_vals = [r.get('R3_jacobian_diff') or 0 for r in runs]
            summary[game][cond] = {
                'n_draws': n,
                'L1_rate_A': round(l1_a / n, 3),
                'L1_rate_B': round(l1_b / n, 3),
                'mean_speedup': round(sum(speedups)/len(speedups), 4) if speedups else None,
                'n_speedup_pairs': len(speedups),
                'mean_RHAE': round(sum(rhae_vals)/n, 8),
                'mean_I3_cv': round(sum(i3cv_vals)/n, 4),
                'mean_R3': round(sum(r3_vals)/n, 6),
            }

    with open(os.path.join(RESULTS_DIR, 'step1298_summary.json'), 'w') as f:
        json.dump({'total_elapsed': round(total_elapsed, 1), 'n_draws': len(all_results),
                   'summary': summary}, f, indent=2)

    print(f"\nSTEP 1298 DONE")


if __name__ == '__main__':
    main()
