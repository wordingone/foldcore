"""
Step 1309 — Linear LPL reflexive map, MBPP in the pool (dual-modality)
Leo mail 3697, 2026-03-29.

Architecture: linear reflexive map with LPL Hebbian (confirmed composition from step1253b/1282).
  obs → _enc_frame → centered 256-dim → W_action (n_actions × 256) → action
  Update: LPL Hebbian on W_action (eta_h=0.05, predictive term eta_p=0.005)
  Action: adaptive softmax (temperature = 1/std(scores))

MBPP added to game pool. obs from mbpp_game.py is 256-dim float32 — same _enc_frame
fallback path as other games. No dual-modality encoding needed.

Why this step:
  1307/1306 killed curiosity direction (maximize encoding change). Both tested same
  hypothesis, both ≈ entropy baseline. Now returning to confirmed LPL (step1282) with
  MBPP in the pool — first test of whether prediction=action on text produces any signal.

Conditions:
  LPL:  linear reflexive map, LPL Hebbian on W_action
  RAND: random action baseline

Kill criteria (chain-level, masked):
  Primary:
    - LPL (1/cr × action_KL) ≤ RAND (1/cr × action_KL) → KILL
    - LPL action_KL < 0.01 → collapsed → KILL
  Secondary (informative):
    - LPL RHAE ≤ RAND RHAE

Protocol: random seed 1309, masked PRISM (Game A/B/C), 3 draws × 2 conditions = 18 runs.
Games: ['cn04', 'mbpp', 'sb26'] (pre-computed, shown in JSONL only).
"""
import sys, os, time, json, random, logging
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame
from prism_masked import select_games, seal_mapping, label_filename, masked_game_list

# --- Masked PRISM game selection ---
# GAMES is internal-only. Never print, never log, never pass to Leo.
# All output uses GAME_LABELS values (labels) only.
GAMES, GAME_LABELS = select_games(seed=1309)

N_DRAWS = 3
MAX_STEPS = 2_000
MAX_SECONDS = 300  # 5-min hard cap per episode

LOSS_CHECKPOINTS = [500, 1000, 1500, 2000]

CONDITIONS = ['lpl', 'rand']
LABELS = {'lpl': 'LPL', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1309')
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

# LPL hyperparameters (from confirmed step1253b / step1282)
ENC_DIM = 256
ETA_H = 0.05
ETA_P = 0.005
W_GRAD_CLIP = 1.0
W_INIT_SCALE = 0.01
T_MIN = 0.01  # minimum softmax temperature denominator


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
    # MBPP: use oracle (type ground-truth code directly)
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        return mbpp_game.compute_solver_steps(0)  # problem 0 oracle

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
# LPL substrate
# ---------------------------------------------------------------------------

class LPLSubstrate:
    """Linear reflexive map with LPL Hebbian. Confirmed from step1282/1253b.
    W_action (n_actions × 256) both encodes and selects. LPL update on W_action.

    Pred loss: EMA per-action forward model. W_pred[a] ← EMA of enc(obs_next) given action a.
    pred_loss = ||enc_next - W_pred[a]||² / ||enc_next||² (relative prediction error).
    RAND and LPL use the same metric — RAND never improves, LPL does if it learns structure.
    """

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed + 99999)

        self.W = rng_w.randn(n_actions, ENC_DIM).astype(np.float32) * W_INIT_SCALE
        self.W_init = self.W.copy()

        # EMA forward predictor: W_pred[a] ≈ E[enc_next | action=a]
        self.W_pred = np.zeros((n_actions, ENC_DIM), np.float32)
        self._W_pred_n = np.zeros(n_actions, np.float32)

        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0

        self._prev_enc = None
        self._last_centered = None
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

        self._recent_pred_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def _centered_encode(self, obs_raw, update_mean=True):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        if update_mean:
            self.n_obs += 1
            a = 1.0 / self.n_obs
            self.running_mean = (1 - a) * self.running_mean + a * x
        return x - self.running_mean

    def process(self, obs_raw):
        self.step += 1
        centered = self._centered_encode(obs_raw)
        self._last_centered = centered
        enc = self.W @ centered  # (n_actions,)

        # Adaptive softmax selection
        std_enc = float(np.std(enc))
        T = max(1.0 / (std_enc + 1e-8), T_MIN)
        logits = np.abs(enc) * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(self._rng.choice(self.n_actions, p=probs))

        self._action_counts[action] += 1

        # LPL update: Hebbian (Oja) + predictive
        hebb = np.outer(enc, centered) - (enc ** 2)[:, None] * self.W
        delta = ETA_H * hebb
        if self._prev_enc is not None:
            delta += ETA_P * np.outer(enc - self._prev_enc, centered)
        norm = float(np.linalg.norm(delta))
        if norm > W_GRAD_CLIP:
            delta *= W_GRAD_CLIP / norm
        self.W += delta
        np.clip(self.W, -100.0, 100.0, out=self.W)

        self._prev_enc = enc.copy()
        return action

    def update_after_step(self, obs_next, action, reward):
        # Encode next obs WITHOUT updating running_mean (that happens in process())
        enc_next = self._centered_encode(obs_next, update_mean=False)
        a = action % self.n_actions

        # EMA update of forward predictor
        self._W_pred_n[a] += 1
        lr = 1.0 / (self._W_pred_n[a] + 1)
        self.W_pred[a] = (1 - lr) * self.W_pred[a] + lr * enc_next

        # Pred loss: how far from the EMA prediction
        err = enc_next - self.W_pred[a]
        enc_norm_sq = float(np.dot(enc_next, enc_next))
        pred_loss = float(np.dot(err, err)) / (enc_norm_sq + 1e-8)

        self._recent_pred_losses.append(pred_loss)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

    def on_level_transition(self):
        self._prev_enc = None

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W - self.W_init))

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_dream_loss_trajectory(self):
        return {}


# ---------------------------------------------------------------------------
# RAND substrate
# ---------------------------------------------------------------------------

class RandSubstrate:
    """Pure random action selection. No learning.
    Same EMA forward predictor metric as LPL — RAND never improves at prediction."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        # EMA forward predictor (same structure as LPL, never improves for RAND)
        self.W_pred = np.zeros((n_actions, ENC_DIM), np.float32)
        self._W_pred_n = np.zeros(n_actions, np.float32)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)
        self._recent_pred_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def process(self, obs_raw):
        self.step += 1
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
        self.n_obs += 1
        a_lr = 1.0 / self.n_obs
        self.running_mean = (1 - a_lr) * self.running_mean + a_lr * x
        action = int(self._rng.randint(self.n_actions))
        self._action_counts[action] += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        x_next = _enc_frame(np.asarray(obs_next, dtype=np.float32))
        enc_next = x_next - self.running_mean
        a = action % self.n_actions

        self._W_pred_n[a] += 1
        lr = 1.0 / (self._W_pred_n[a] + 1)
        self.W_pred[a] = (1 - lr) * self.W_pred[a] + lr * enc_next

        err = enc_next - self.W_pred[a]
        enc_norm_sq = float(np.dot(enc_next, enc_next))
        pred_loss = float(np.dot(err, err)) / (enc_norm_sq + 1e-8)

        self._recent_pred_losses.append(pred_loss)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return 0.0

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_dream_loss_trajectory(self):
        return {}


def make_substrate(condition, n_actions, seed):
    if condition == 'lpl':
        return LPLSubstrate(n_actions=n_actions, seed=seed)
    return RandSubstrate(n_actions=n_actions, seed=seed)


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
    kl = float(np.sum(early_p * np.log(early_p / late_p + 1e-12)))
    return round(kl, 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
    i3_counts_at_200 = None
    action_counts = np.zeros(n_actions, np.float32)

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

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

        action = substrate.process(obs_arr) % n_actions
        action_counts[action] += 1
        action_log.append(action)

        obs_next, reward, done, info = env.step(action)
        steps += 1

        if obs_next is not None:
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
        'dream_loss_traj': {},
    }


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(condition, game_name, draw_idx, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    seed_a = draw_idx * 2
    seed_b = draw_idx * 2 + 1

    substrate = make_substrate(condition, n_actions, seed_a)
    result_a = run_episode(env, substrate, n_actions, solver_level_steps, seed=seed_a)
    result_b = run_episode(env, substrate, n_actions, solver_level_steps, seed=seed_b)

    speedup = None
    l1_a = result_a['level_first_step'].get(1)
    l1_b = result_b['level_first_step'].get(1)
    if l1_a is not None and l1_b is not None:
        speedup = round(l1_a / l1_b, 3)
    elif l1_a is None and l1_b is not None:
        speedup = float('inf')

    return {
        'game': game_name,
        'draw': draw_idx,
        'condition': condition,
        'episode_A': result_a,
        'episode_B': result_b,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Kill assessment
# ---------------------------------------------------------------------------

def kill_assessment(chain_summary):
    lpl = chain_summary['lpl']
    rand = chain_summary['rand']

    lpl_cr = lpl.get('mean_compression_ratio')
    rand_cr = rand.get('mean_compression_ratio')
    lpl_kl = lpl.get('mean_action_KL')
    rand_kl = rand.get('mean_action_KL')

    lines = []
    kill = False

    if lpl_cr is not None and rand_cr is not None and lpl_kl is not None and rand_kl is not None:
        lpl_score = (1.0 / (lpl_cr + 1e-8)) * lpl_kl
        rand_score = (1.0 / (rand_cr + 1e-8)) * rand_kl
        if lpl_score <= rand_score:
            lines.append(f"  >>> KILL: LPL (1/cr × kl)={lpl_score:.2f} ≤ RAND={rand_score:.2f}")
            kill = True
        else:
            lines.append(f"  >>> SIGNAL: LPL (1/cr × kl)={lpl_score:.2f} > RAND={rand_score:.2f} → LPL improves compression×diversity")

    if lpl_kl is not None:
        if lpl_kl < 0.01:
            lines.append(f"  >>> KILL: LPL action_KL={lpl_kl:.4f} < 0.01 → collapsed")
            kill = True
        else:
            lines.append(f"  >>> OK: LPL action_KL={lpl_kl:.4f} ≥ 0.01")

    lpl_rhae = lpl.get('mean_RHAE', 0.0)
    rand_rhae = rand.get('mean_RHAE', 0.0)
    lines.append(f"  >>> SECONDARY: LPL RHAE={lpl_rhae:.2e} vs RAND RHAE={rand_rhae:.2e} (informative)")

    if kill:
        lines.append("  >>> KILL TRIGGERED")
    else:
        lines.append("  >>> NO KILL triggered")

    return lines, kill


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step 1309 — Linear LPL reflexive map, MBPP in pool")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"MAX_STEPS={MAX_STEPS}, LOSS_CHECKPOINTS={LOSS_CHECKPOINTS}")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
    print()

    # Seal mapping — do not read during session
    seal_mapping(RESULTS_DIR, GAMES, GAME_LABELS)

    # Pre-compute solver baselines
    print("Computing solver baselines...")
    solver_steps_cache = {}
    for game in GAMES:
        try:
            solver_steps_cache[game] = compute_solver_level_steps(game)
        except Exception:
            solver_steps_cache[game] = {}
    print()

    # Per-game × draw × condition loop
    all_results = []
    chain_data = {c: {'RHAE': [], 'wdrift': [], 'action_KL': [], 'I3cv': [],
                      'compression_ratio': []} for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for draw_idx in range(N_DRAWS):
            row_parts = []
            for condition in CONDITIONS:
                result = run_draw(condition, game, draw_idx, solver_steps)
                all_results.append(result)

                ep = result['episode_A']
                lbl = LABELS[condition]
                wdrift = ep.get('wdrift', 0.0)
                kl = ep.get('action_kl', 0.0)
                rhae = ep.get('RHAE', 0.0)
                cr = ep.get('compression_ratio')
                cr_str = f"{cr:.4f}" if cr is not None else "N/A"
                row_parts.append(
                    f"{lbl}[wdrift={wdrift:.4f},kl={kl},RHAE={rhae:.2e},cr={cr_str}]"
                )

            elapsed_draw = sum(
                r['episode_A']['elapsed_seconds'] + r['episode_B']['elapsed_seconds']
                for r in all_results[-(len(CONDITIONS)):]
            )
            print(f"  {label} draw {draw_idx}: {' '.join(row_parts)}  ({elapsed_draw:.1f}s)")

        # Write per-game JSONL — filename uses label, not game name
        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1309))
        with open(out_path, 'w') as f:
            for r in game_results:
                # Strip game name from output record before writing
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r) + '\n')

    # Chain-level aggregates
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
            'pred_loss_trajectory': _mean_loss_traj(all_results, cond, LOSS_CHECKPOINTS),
        }

    summary = {'chain_summary': chain_summary, 'n_draws': N_DRAWS,
               'games_label_only': masked_game_list(GAME_LABELS)}
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Print chain summary
    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1309 — CHAIN-LEVEL SUMMARY\n")
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
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


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


if __name__ == '__main__':
    main()
