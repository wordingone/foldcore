"""
Step 1313 — Multi-layer LPL, deterministic init, seed-free, T_chain
Leo mail 3715, 2026-03-29.

Architecture: identical to step 1310/1312 (3-layer W&B predictive coding).

Initialization (NEW — deterministic, no seeds):
  W1 = np.eye(128, 256) * 0.1   identity scaled, truncated to shape
  W2 = np.eye(64, 128)  * 0.1
  W3 = np.zeros((n_actions, 64))
  No np.random.seed. Environment breaks symmetry through observations.

Protocol (seed-free):
  3 games (MBPP + 2 masked ARC), 1 run per game per condition.
  Total: 2 conditions × 3 games = 6 runs + 3 fresh-B runs = 9 runs.

Conditions:
  MLPL: multi-layer LPL, deterministic init, W1/W2/W3 updated
  RAND: same deterministic init, no updates (frozen throughout)

T_chain (pure experience effect):
  Per game: run A (2K steps MLPL) → run B experienced (1K steps same substrate)
  Also run B fresh (1K steps fresh deterministic substrate)
  T = pred_loss(fresh_B@1K) / pred_loss(experienced_B@1K)
  T > 1: experience on A helps predict B. Not an init artifact (both start from same weights).

Kill criteria:
  MLPL T_chain ≤ 1.0 → no transfer → KILL
  MLPL cr ≥ 1.0 → prediction doesn't improve from deterministic init → KILL

Predictions (Leo):
  1. T_chain > 1.0 (experience genuinely helps)
  2. cr < 1.0 (prediction improves from deterministic init — dynamics dominate init)
  If both hold: organism is real. One chemistry, genuine transfer, no seed dependence.
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
from prism_masked import select_games, seal_mapping, label_filename, masked_game_list, masked_run_log

GAMES, GAME_LABELS = select_games(seed=1313)

MAX_STEPS_A    = 2_000
MAX_STEPS_B    = 1_000
MAX_SECONDS    = 300

LOSS_CHECKPOINTS_A = [500, 1000, 2000]
RECORD_STEP_B      = 1000

CONDITIONS = ['mlpl', 'rand']
LABELS     = {'mlpl': 'MLPL', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1313')
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

ENC_DIM = 256
H1_DIM  = 128
H2_DIM  = 64
ETA     = 0.001
INF_LR  = 0.05
K_INF   = 5
W_CLIP  = 100.0
T_MIN   = 0.01

# Episode seeds (game environment seeds — benchmark requirement, not substrate seeds)
SEED_A = 0
SEED_B = 1


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


def compute_solver_level_steps(game_name):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)
    prescription = load_prescription(game_name)
    if prescription is None:
        return {}
    env = make_game(game_name)
    env.reset(seed=1)
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
            env.reset(seed=1)
            fresh_episode = True
    return level_first_step


# ---------------------------------------------------------------------------
# Substrate — deterministic init, no seeds
# ---------------------------------------------------------------------------

class MLPLSubstrate:
    """Multi-layer PC, deterministic init. W1/W2/W3 updated via local Hebbian rules."""

    def __init__(self, n_actions, checkpoints=None):
        self.n_actions = n_actions

        # Deterministic weight init — no seeds
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)

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
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

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
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_recent_pred_loss(self):
        if self._recent_pred_losses:
            return float(np.mean(self._recent_pred_losses))
        return None


class RandSubstrate(MLPLSubstrate):
    """Same deterministic init as MLPL, no weight updates. Pure random actions via W3@h2."""

    def process(self, obs_raw):
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # No weight updates — frozen throughout
        action_scores = self.W3 @ h2
        std_s = float(np.std(action_scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = action_scores * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        return action


def make_substrate(condition, n_actions, checkpoints=None):
    if condition == 'mlpl':
        return MLPLSubstrate(n_actions=n_actions, checkpoints=checkpoints)
    return RandSubstrate(n_actions=n_actions, checkpoints=checkpoints)


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
    episode_step = 0
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
        'recorded_pred_loss': recorded_pred_loss,
    }


# ---------------------------------------------------------------------------
# Game runner (no draws — 1 run per game per condition)
# ---------------------------------------------------------------------------

def run_game(condition, game_name, solver_level_steps):
    """Run episode A (2K) + episode B experienced (1K) on same substrate."""
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = make_substrate(condition, n_actions, checkpoints=LOSS_CHECKPOINTS_A)

    result_a = run_episode(env, substrate, n_actions, solver_level_steps,
                           seed=SEED_A, max_steps=MAX_STEPS_A)

    result_b_exp = run_episode(env, substrate, n_actions, solver_level_steps,
                                seed=SEED_B, max_steps=MAX_STEPS_B,
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
        'condition': condition,
        'episode_A': result_a,
        'episode_B_experienced': result_b_exp,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
        'pred_loss_experienced_B': result_b_exp['recorded_pred_loss'],
    }


def run_fresh_b(game_name, solver_level_steps):
    """Run episode B with fresh MLPL substrate (no prior experience from A).
    T_chain denominator. Deterministic init — no seed needed."""
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = make_substrate('mlpl', n_actions, checkpoints=[500, 1000])

    result = run_episode(env, substrate, n_actions, solver_level_steps,
                          seed=SEED_B, max_steps=MAX_STEPS_B,
                          record_step=RECORD_STEP_B)

    return result['recorded_pred_loss']


# ---------------------------------------------------------------------------
# Kill assessment
# ---------------------------------------------------------------------------

def kill_assessment(chain_summary, t_chain_mlpl):
    s = chain_summary['mlpl']
    cr = s.get('mean_compression_ratio')
    kl = s.get('mean_action_KL')

    lines = []
    kill = False

    # Primary: T_chain
    if t_chain_mlpl is not None:
        if t_chain_mlpl <= 1.0:
            lines.append(
                f"  >>> KILL: MLPL T_chain={t_chain_mlpl:.4f} ≤ 1.0 → experience doesn't help"
            )
            kill = True
        else:
            lines.append(
                f"  >>> SIGNAL: MLPL T_chain={t_chain_mlpl:.4f} > 1.0 → genuine transfer confirmed"
            )
        pred1 = "CONFIRMED" if t_chain_mlpl > 1.0 else "WRONG"
        lines.append(f"  >>> PREDICTION 1 (T_chain > 1.0): {pred1} (T={t_chain_mlpl:.4f})")

    # Primary: compression ratio
    if cr is not None:
        if cr >= 1.0:
            lines.append(
                f"  >>> KILL: MLPL cr={cr:.4f} ≥ 1.0 → prediction doesn't improve from deterministic init"
            )
            kill = True
        else:
            lines.append(
                f"  >>> SIGNAL: MLPL cr={cr:.4f} < 1.0 → compression from deterministic init confirmed"
            )
        pred2 = "CONFIRMED" if cr < 1.0 else "WRONG"
        lines.append(f"  >>> PREDICTION 2 (cr < 1.0): {pred2} (cr={cr:.4f})")

    # Collapse check
    if kl is not None:
        if kl < 0.01:
            lines.append(f"  >>> KILL: MLPL action_KL={kl:.4f} < 0.01 → collapsed")
            kill = True
        else:
            lines.append(f"  >>> OK: MLPL action_KL={kl:.4f} ≥ 0.01")

    rhae = s.get('mean_RHAE', 0.0)
    rand_rhae = chain_summary['rand'].get('mean_RHAE', 0.0)
    lines.append(f"  >>> SECONDARY: MLPL RHAE={rhae:.2e} vs RAND RHAE={rand_rhae:.2e}")

    if kill:
        lines.append("  >>> KILL TRIGGERED")
    else:
        lines.append("  >>> NO KILL — organism confirmed: one chemistry, genuine transfer, deterministic init")

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
    print(f"Step 1313 — Multi-layer LPL, deterministic init, seed-free, T_chain")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Init: W1=eye(128,256)*0.1, W2=eye(64,128)*0.1, W3=zeros(n_actions,64)")
    print(f"No seeds. Environment breaks symmetry through observations.")
    print(f"Episode A: {MAX_STEPS_A} steps | Episode B: {MAX_STEPS_B} steps")
    print(f"Total: {len(CONDITIONS) * len(GAMES)} substrate runs + {len(GAMES)} fresh-B runs = "
          f"{len(CONDITIONS) * len(GAMES) + len(GAMES)} runs")
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
    t_mlpl_vals = []

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        t0 = time.time()
        for condition in CONDITIONS:
            result = run_game(condition, game, solver_steps)
            all_results.append(result)

        pred_loss_fresh = run_fresh_b(game, solver_steps)

        # T_chain for MLPL only
        mlpl_result = next(r for r in all_results if r['game'] == game and r['condition'] == 'mlpl')
        pl_exp = mlpl_result['pred_loss_experienced_B']
        if pred_loss_fresh is not None and pl_exp is not None and pl_exp > 1e-8:
            t_mlpl_vals.append(pred_loss_fresh / pl_exp)

        elapsed = time.time() - t0
        print(masked_run_log(label, elapsed))

        # Write per-game JSONL
        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1313))
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

    t_chain_mlpl = round(float(np.mean(t_mlpl_vals)), 4) if t_mlpl_vals else None

    summary = {
        'chain_summary': chain_summary,
        'T_chain': {'mlpl': t_chain_mlpl},
        'games_label_only': masked_game_list(GAME_LABELS),
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1313 — CHAIN-LEVEL SUMMARY\n")
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
    t_str = f"{t_chain_mlpl:.4f}" if t_chain_mlpl is not None else "N/A"
    print(f"  T_chain: MLPL={t_str}")
    print()
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary, t_chain_mlpl)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
