"""
Step 1312 — Multi-layer LPL + three-factor W3 + T_chain transfer metric
Leo mail 3710, 2026-03-29.

Two changes from step 1310 (BASE):

Change 1: Three-factor W3 update (3F condition)
  BASE: W3 += eta * outer(action_onehot, h2)           (pure Hebbian)
  3F:   W3 += eta * pe_next * outer(action_onehot, h2)  (pe-modulated)
  pe_next = ||e1_next||² / ||enc_next||² — prediction error of NEXT state
  Same coupling law as W1/W2 updates. R2 compliant (pe_next is byproduct).
  NOTE: formula (high pe → large update) may differ from description; formula as spec'ed.

Change 2: T_chain metric (R4 — transfer / second-exposure)
  Per draw: run episode A (2K steps, experienced substrate) then episode B (1K steps, same substrate).
  Also run episode B (1K steps) with FRESH substrate (no prior experience).
  T = pred_loss_fresh_B@1K / pred_loss_experienced_B@1K
  T > 1: experience on A helped predict B (transfer). T = 1: no transfer.
  T_chain = chain-level mean T across all games × draws.

Kill criteria:
  Primary:   3F T_chain ≤ BASE T_chain → three-factor doesn't improve transfer → KILL
  Secondary: 3F cr ≥ BASE cr → modulation hurts prediction → KILL
  Tertiary:  3F action_KL < 0.01 → collapsed → KILL

Protocol: seed 1312, masked PRISM (MBPP + 2 masked ARC), 3 draws × 2 conditions = 18 runs
          + 3 games × 3 draws = 9 fresh-B runs. Total 27 runs.
"""
import sys, os, time, json, logging

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame
from prism_masked import select_games, seal_mapping, label_filename, masked_game_list, masked_draw_log

GAMES, GAME_LABELS = select_games(seed=1312)

N_DRAWS = 3
MAX_STEPS_A = 2_000   # episode A
MAX_STEPS_B = 1_000   # episode B (experienced + fresh)
MAX_SECONDS = 300

LOSS_CHECKPOINTS_A = [500, 1000, 2000]
RECORD_STEP_B = 1000   # episode-local step for T_chain measurement

CONDITIONS = ['3f', 'base']
LABELS = {'3f': '3F', 'base': 'BASE'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1312')
PDIR = 'B:/M/the-search/experiments/results/prescriptions'

SOLVER_PRESCRIPTIONS = {
    'ls20':  ('ls20_fullchain.json',  'all_actions'),
    'ft09':  ('ft09_fullchain.json',  'all_actions'),
    'vc33':  ('vc33_fullchain.json',  'all_actions_encoded'),
    'tr87':  ('tr87_fullchain.json',  'all_actions'),
    'sp80':  ('sp80_fullchain.json',  'all_actions'),
    'sb26':  ('sb26_fullchain.json',  'all_actions'),
    'tu93':  ('tu93_fullchain.json',  'all_actions'),
    'cn04':  ('cn04_fullchain.json',  'sequence'),
    'cd82':  ('cd82_fullchain.json',  'all_actions'),
    'lp85':  ('lp85_fullchain.json',  'all_actions'),
}

ACTION_OFFSET = {'ls20': -1, 'vc33': 7}

# Hyperparameters (identical to step 1310)
ENC_DIM = 256
H1_DIM  = 128
H2_DIM  = 64
ETA     = 0.001
INF_LR  = 0.05
K_INF   = 5
W_INIT_SCALE = 0.01
W_CLIP  = 100.0
T_MIN   = 0.01


# ---------------------------------------------------------------------------
# Game factory
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


def load_prescription(game_name):
    if game_name.lower() not in SOLVER_PRESCRIPTIONS:
        return None
    fname, field = SOLVER_PRESCRIPTIONS[game_name.lower()]
    path = os.path.join(PDIR, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get(field)


def compute_solver_level_steps(game_name, seed=1):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=seed)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103
    offset = ACTION_OFFSET.get(game_name.lower(), 0)
    level, level_first_step, step, fresh_episode = 0, {}, 0, True
    for action in prescription:
        action_int = (int(action) + offset) % n_actions
        obs_next, reward, done, info = env.step(action_int)
        step += 1
        if fresh_episode:
            fresh_episode = False
            continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level_first_step[cl] = step
            level = cl
        if done:
            env.reset(seed=seed)
            fresh_episode = True
    return level_first_step


# ---------------------------------------------------------------------------
# BASE substrate (identical to step 1310 MLPLSubstrate)
# ---------------------------------------------------------------------------

class BaseSubstrate:
    """Multi-layer PC, pure Hebbian W3. Same as step 1310."""

    def __init__(self, n_actions, seed, checkpoints=None):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed + 99999)

        self.W1 = rng_w.randn(H1_DIM, ENC_DIM).astype(np.float32) * W_INIT_SCALE
        self.W2 = rng_w.randn(H2_DIM, H1_DIM).astype(np.float32) * W_INIT_SCALE
        self.W3 = rng_w.randn(n_actions, H2_DIM).astype(np.float32) * W_INIT_SCALE

        self._W1_init = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

        ckpts = checkpoints if checkpoints is not None else LOSS_CHECKPOINTS_A
        self._recent_pred_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in ckpts}

    def _centered_encode(self, obs_raw, update_mean=True):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        if update_mean:
            self.n_obs += 1
            a = 1.0 / self.n_obs
            self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def _infer(self, enc):
        h1 = self.W1 @ enc
        h2 = self.W2 @ h1
        for _ in range(K_INF):
            enc_pred = self.W1.T @ h1
            e1 = enc - enc_pred
            h1_pred = self.W2.T @ h2
            e2 = h1 - h1_pred
            h1 = h1 + INF_LR * (self.W1 @ e1 - e2)
            h2 = h2 + INF_LR * (self.W2 @ e2)
        enc_pred = self.W1.T @ h1
        e1 = enc - enc_pred
        h1_pred = self.W2.T @ h2
        e2 = h1 - h1_pred
        return h1, h2, e1, e2

    def process(self, obs_raw):
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        action_scores = self.W3 @ h2
        std_s = float(np.std(action_scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = action_scores * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        # W1, W2, W3 updates (pure Hebbian W3)
        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        self.W3 += ETA * np.outer(action_onehot, h2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass  # all learning in process()

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_recent_pred_loss(self):
        """Current mean pred_loss from recent 50 steps — for T_chain measurement."""
        if self._recent_pred_losses:
            return float(np.mean(self._recent_pred_losses))
        return None

    def get_dream_loss_trajectory(self):
        return {}


# ---------------------------------------------------------------------------
# 3F substrate (pe_next-modulated W3)
# ---------------------------------------------------------------------------

class ThreeFactorSubstrate(BaseSubstrate):
    """Multi-layer PC + three-factor W3 update.

    W3 += eta * pe_next * outer(action_onehot, h2)

    pe_next = ||e1_next||² / ||enc_next||² (prediction error of NEXT state).
    Computed in update_after_step. W3 update deferred until pe_next is known.
    W1, W2 still updated in process() (same as BASE).
    """

    def __init__(self, n_actions, seed, checkpoints=None):
        super().__init__(n_actions, seed, checkpoints)
        self._pending_h2 = None
        self._pending_action_onehot = None

    def process(self, obs_raw):
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        action_scores = self.W3 @ h2
        std_s = float(np.std(action_scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = action_scores * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        # W1, W2 updates (same as BASE)
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)

        # W3 deferred: save h2 and action for update_after_step
        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0
        self._pending_h2 = h2.copy()
        self._pending_action_onehot = action_onehot

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        if self._pending_h2 is None:
            return

        # Encode next obs (no running_mean update — obs_next is ephemeral)
        x_next = _enc_frame(np.asarray(obs_next, dtype=np.float32))
        enc_next = x_next - self.running_mean

        # Inference on next state (read-only — don't update W1/W2 here)
        h1_next = self.W1 @ enc_next
        h2_next = self.W2 @ h1_next
        for _ in range(K_INF):
            enc_pred_next = self.W1.T @ h1_next
            e1_next = enc_next - enc_pred_next
            h1_pred_next = self.W2.T @ h2_next
            e2_next = h1_next - h1_pred_next
            h1_next = h1_next + INF_LR * (self.W1 @ e1_next - e2_next)
            h2_next = h2_next + INF_LR * (self.W2 @ e2_next)
        enc_pred_next = self.W1.T @ h1_next
        e1_next = enc_next - enc_pred_next

        enc_norm_sq_next = float(np.dot(enc_next, enc_next))
        pe_next = float(np.dot(e1_next, e1_next)) / (enc_norm_sq_next + 1e-8)

        # Three-factor W3 update: pe_next modulates the Hebbian association
        self.W3 += ETA * pe_next * np.outer(self._pending_action_onehot, self._pending_h2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        self._pending_h2 = None
        self._pending_action_onehot = None


def make_substrate(condition, n_actions, seed, checkpoints=None):
    if condition == '3f':
        return ThreeFactorSubstrate(n_actions=n_actions, seed=seed, checkpoints=checkpoints)
    return BaseSubstrate(n_actions=n_actions, seed=seed, checkpoints=checkpoints)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_arc_score(level_first_step, solver_level_steps):
    if not level_first_step or not solver_level_steps:
        return 0.0
    scores = []
    for lvl, s_step in solver_level_steps.items():
        a_step = level_first_step.get(lvl)
        if a_step is not None and s_step > 0:
            scores.append((s_step / a_step) ** 2)
    if not scores:
        return 0.0
    return round(float(np.mean(scores)), 6)


def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions:
            late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p = (late_c + 1e-8) / (late_c.sum() + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed,
                max_steps=MAX_STEPS_A, record_step=None):
    """Run one episode. record_step: episode-local step at which to snapshot pred_loss for T_chain."""
    obs = env.reset(seed=seed)
    action_log = []
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

    steps = 0
    episode_step = 0   # episode-local counter for T_chain
    level = 0
    max_level = 0
    level_first_step = {}
    t_start = time.time()
    fresh_episode = True
    recorded_pred_loss = None

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
        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)
        episode_step += 1

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if obs_next is not None:
            substrate.update_after_step(obs_next, action, reward)

        # T_chain: record pred_loss at episode-local step
        if record_step is not None and episode_step == record_step:
            recorded_pred_loss = substrate.get_recent_pred_loss()

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

    i3_cv = None
    if i3_counts_at_200 is not None:
        counts = i3_counts_at_200[:n_actions].astype(float)
        mean_c = counts.mean()
        if mean_c > 1e-8:
            i3_cv = round(float(counts.std() / mean_c), 4)

    arc_score = compute_arc_score(level_first_step, solver_level_steps)
    action_kl = compute_action_kl(action_log, n_actions)
    wdrift = substrate.compute_weight_drift()
    pred_loss_traj = substrate.get_pred_loss_trajectory()

    compression_ratio = None
    l_early = pred_loss_traj.get(500)
    l_late = pred_loss_traj.get(2000)
    if l_early is not None and l_late is not None and l_early > 1e-8:
        compression_ratio = round(l_late / l_early, 4)

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(wdrift, 4),
        'pred_loss_traj': pred_loss_traj,
        'compression_ratio': compression_ratio,
        'recorded_pred_loss': recorded_pred_loss,  # for T_chain
    }


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(condition, game_name, draw_idx, solver_level_steps):
    """Run episode A (2K) + episode B experienced (1K) on same substrate."""
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_a = draw_idx * 2
    seed_b = draw_idx * 2 + 1

    substrate = make_substrate(condition, n_actions, seed_a,
                                checkpoints=LOSS_CHECKPOINTS_A)

    # Episode A: 2K steps
    result_a = run_episode(env, substrate, n_actions, solver_level_steps,
                           seed=seed_a, max_steps=MAX_STEPS_A)

    # Episode B experienced: 1K steps on same substrate (carries state from A)
    result_b_exp = run_episode(env, substrate, n_actions, solver_level_steps,
                                seed=seed_b, max_steps=MAX_STEPS_B,
                                record_step=RECORD_STEP_B)

    speedup = None
    l1_a = result_a['level_first_step'].get(1)
    l1_b = result_b_exp['level_first_step'].get(1)
    if l1_a is not None and l1_b is not None:
        speedup = round(l1_a / l1_b, 3)
    elif l1_a is None and l1_b is not None:
        speedup = float('inf')

    return {
        'game': game_name,
        'draw': draw_idx,
        'condition': condition,
        'episode_A': result_a,
        'episode_B_experienced': result_b_exp,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
        'pred_loss_experienced_B': result_b_exp['recorded_pred_loss'],
    }


def run_fresh_b(game_name, draw_idx, solver_level_steps):
    """Run episode B with fresh BASE substrate (no prior experience from A).
    Used as T_chain denominator — shared across conditions for the same game-draw."""
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_b = draw_idx * 2 + 1
    fresh_seed = draw_idx * 2 + 10000   # distinct from experienced seed

    substrate = make_substrate('base', n_actions, fresh_seed,
                                checkpoints=[500, 1000])

    result = run_episode(env, substrate, n_actions, solver_level_steps,
                          seed=seed_b, max_steps=MAX_STEPS_B,
                          record_step=RECORD_STEP_B)

    return result['recorded_pred_loss']


# ---------------------------------------------------------------------------
# Kill assessment
# ---------------------------------------------------------------------------

def kill_assessment(chain_summary, t_chain_3f, t_chain_base):
    s3f = chain_summary['3f']
    sbase = chain_summary['base']

    cr_3f = s3f.get('mean_compression_ratio')
    cr_base = sbase.get('mean_compression_ratio')
    kl_3f = s3f.get('mean_action_KL')

    lines = []
    kill = False

    # Primary: T_chain comparison
    if t_chain_3f is not None and t_chain_base is not None:
        if t_chain_3f <= t_chain_base:
            lines.append(
                f"  >>> KILL: 3F T_chain={t_chain_3f:.4f} ≤ BASE T_chain={t_chain_base:.4f} "
                f"→ three-factor doesn't improve transfer"
            )
            kill = True
        else:
            lines.append(
                f"  >>> SIGNAL: 3F T_chain={t_chain_3f:.4f} > BASE T_chain={t_chain_base:.4f} "
                f"→ pe modulation improves transfer"
            )
        # Check T > 1.0 (Leo prediction 1)
        for cond_name, t_val in [('3F', t_chain_3f), ('BASE', t_chain_base)]:
            pred = "CONFIRMED" if t_val > 1.0 else "WRONG"
            lines.append(f"  >>> PREDICTION 1 ({cond_name} T_chain > 1.0): {pred} (T={t_val:.4f})")

    # Secondary: compression ratio comparison
    if cr_3f is not None and cr_base is not None:
        if cr_3f >= cr_base:
            lines.append(
                f"  >>> KILL: 3F cr={cr_3f:.4f} ≥ BASE cr={cr_base:.4f} → modulation hurts prediction"
            )
            kill = True
        else:
            lines.append(
                f"  >>> SIGNAL: 3F cr={cr_3f:.4f} < BASE cr={cr_base:.4f} → prediction intact"
            )

    # action_KL comparison (Leo prediction 3)
    kl_base = sbase.get('mean_action_KL')
    if kl_3f is not None and kl_base is not None:
        kl_diff = abs(kl_3f - kl_base)
        pred3 = "CONFIRMED" if kl_diff > 0.1 else "WRONG"
        lines.append(
            f"  >>> PREDICTION 3 (3F action_KL ≠ BASE): {pred3} "
            f"(3F={kl_3f:.4f}, BASE={kl_base:.4f}, diff={kl_diff:.4f})"
        )

    # Collapse check
    if kl_3f is not None:
        if kl_3f < 0.01:
            lines.append(f"  >>> KILL: 3F action_KL={kl_3f:.4f} < 0.01 → collapsed")
            kill = True
        else:
            lines.append(f"  >>> OK: 3F action_KL={kl_3f:.4f} ≥ 0.01")

    rhae_3f = s3f.get('mean_RHAE', 0.0)
    rhae_base = sbase.get('mean_RHAE', 0.0)
    lines.append(f"  >>> SECONDARY: 3F RHAE={rhae_3f:.2e} vs BASE RHAE={rhae_base:.2e}")

    if kill:
        lines.append("  >>> KILL TRIGGERED")
    else:
        lines.append("  >>> NO KILL triggered")

    return lines, kill


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _mean_loss_traj(all_results, cond, checkpoints):
    by_ck = {ck: [] for ck in checkpoints}
    for r in all_results:
        if r['condition'] != cond:
            continue
        traj = r['episode_A'].get('pred_loss_traj', {})
        for ck in checkpoints:
            v = traj.get(ck)
            if v is not None:
                by_ck[ck].append(v)
    return {str(ck): (round(float(np.mean(v)), 6) if v else None)
            for ck, v in by_ck.items()}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1312 — Multi-layer LPL + three-factor W3 + T_chain")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"3F: W3 += eta * pe_next * outer(action_onehot, h2)")
    print(f"BASE: W3 += eta * outer(action_onehot, h2)  (step 1310 baseline)")
    print(f"T_chain: pred_loss(fresh_B@1K) / pred_loss(experienced_B@1K)")
    print(f"Episode A: {MAX_STEPS_A} steps | Episode B: {MAX_STEPS_B} steps")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} substrate runs + {len(GAMES) * N_DRAWS} fresh-B runs")
    print()

    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    all_results = []
    # T_chain data: per draw, per condition
    t_data = {'3f': [], 'base': []}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for draw_idx in range(N_DRAWS):
            t0 = time.time()

            # Run experienced substrates (both conditions)
            for condition in CONDITIONS:
                result = run_draw(condition, game, draw_idx, solver_steps)
                all_results.append(result)

            # Run shared fresh-B
            pred_loss_fresh = run_fresh_b(game, draw_idx, solver_steps)

            # Compute T per condition for this draw
            for result in all_results[-(len(CONDITIONS)):]:
                cond = result['condition']
                pl_exp = result['pred_loss_experienced_B']
                if pred_loss_fresh is not None and pl_exp is not None and pl_exp > 1e-8:
                    t_val = pred_loss_fresh / pl_exp
                    t_data[cond].append(t_val)

            elapsed_draw = time.time() - t0
            print(masked_draw_log(label, draw_idx, elapsed_draw))

        # Write per-game JSONL (timing + chain metrics only — no per-draw stats)
        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1312))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r) + '\n')

    # Chain-level aggregates
    chain_data = {c: {'RHAE': [], 'wdrift': [], 'action_KL': [], 'I3cv': [],
                      'compression_ratio': []} for c in CONDITIONS}
    for r in all_results:
        cond = r['condition']
        ep = r['episode_A']
        chain_data[cond]['RHAE'].append(ep.get('RHAE', 0.0) or 0.0)
        chain_data[cond]['wdrift'].append(ep.get('wdrift', 0.0) or 0.0)
        chain_data[cond]['action_KL'].append(ep.get('action_kl', 0.0) or 0.0)
        chain_data[cond]['I3cv'].append(ep.get('I3_cv', 0.0) or 0.0)
        if ep.get('compression_ratio') is not None:
            chain_data[cond]['compression_ratio'].append(ep['compression_ratio'])

    chain_summary = {}
    for cond in CONDITIONS:
        d = chain_data[cond]
        chain_summary[cond] = {
            'mean_RHAE': round(float(np.mean(d['RHAE'])) if d['RHAE'] else 0.0, 8),
            'mean_wdrift': round(float(np.mean(d['wdrift'])) if d['wdrift'] else 0.0, 4),
            'mean_action_KL': round(float(np.mean(d['action_KL'])) if d['action_KL'] else 0.0, 4),
            'mean_I3cv': round(float(np.mean(d['I3cv'])) if d['I3cv'] else 0.0, 4),
            'mean_compression_ratio': round(float(np.mean(d['compression_ratio'])), 4)
                if d['compression_ratio'] else None,
            'pred_loss_trajectory': _mean_loss_traj(all_results, cond, LOSS_CHECKPOINTS_A),
        }

    # T_chain
    t_chain_3f = round(float(np.mean(t_data['3f'])), 4) if t_data['3f'] else None
    t_chain_base = round(float(np.mean(t_data['base'])), 4) if t_data['base'] else None

    summary = {
        'chain_summary': chain_summary,
        'T_chain': {'3f': t_chain_3f, 'base': t_chain_base},
        'n_draws': N_DRAWS,
        'games_label_only': masked_game_list(GAME_LABELS),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1312 — CHAIN-LEVEL SUMMARY\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        s = chain_summary[cond]
        print(f"  {lbl}:")
        print(f"    mean_RHAE={s['mean_RHAE']:.2e}  mean_wdrift={s['mean_wdrift']}  "
              f"mean_action_KL={s['mean_action_KL']}  mean_I3cv={s['mean_I3cv']}")
        pt = s['pred_loss_trajectory']
        print(f"    pred_loss_trajectory: {pt}")
        cr = s['mean_compression_ratio']
        print(f"    mean_compression_ratio (loss@2K/loss@500): {cr}")
    print()
    t3f_str = f"{t_chain_3f:.4f}" if t_chain_3f is not None else "N/A"
    tbase_str = f"{t_chain_base:.4f}" if t_chain_base is not None else "N/A"
    print(f"  T_chain: 3F={t3f_str}  BASE={tbase_str}")
    print()
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary, t_chain_3f, t_chain_base)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
