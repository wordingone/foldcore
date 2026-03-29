"""
Step 1311 — Multi-layer predictive coding, extended budget (5K steps)
Leo mail 3707, 2026-03-29.

Same architecture as step 1310 (W&B 2017 multi-layer PC). Only change: 5K steps.
Tests whether the compression trajectory (still declining at step 2000: 0.891) continues.

Kill criteria (chain-level, masked):
  Primary:   MLPL cr@5K ≥ 0.9282 (step 1310 cr@2K) → compression plateaued → KILL
  Secondary: MLPL cr@5K ≥ RAND cr → worse than random → KILL
  Tertiary:  MLPL action_KL < 0.01 → collapsed → KILL

Leo predictions:
  1. cr@5K < 0.8 (trajectory still declining at 2K, expects continuation)
  2. wdrift > 0.5 (more steps = more weight change)
  3. MBPP pred_loss shows strongest decline (text has sequential structure)

Protocol: random seed 1311, masked PRISM (MBPP + 2 ARC games), 3 draws × 2 conditions = 18 runs.
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

# --- Masked PRISM game selection ---
GAMES, GAME_LABELS = select_games(seed=1311)

N_DRAWS = 3
MAX_STEPS = 5_000
MAX_SECONDS = 300

LOSS_CHECKPOINTS = [500, 1000, 2000, 5000]

CONDITIONS = ['mlpl', 'rand']
LABELS = {'mlpl': 'MLPL', 'rand': 'RAND'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1311')
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

# Step 1310 reference cr (kill threshold)
CR_1310 = 0.9282


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
# MLPL substrate (identical to step 1310)
# ---------------------------------------------------------------------------

class MLPLSubstrate:
    """Multi-layer predictive coding (Whittington & Bogacz 2017). Same as step 1310."""

    def __init__(self, n_actions, seed):
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

        self._recent_pred_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

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
        for ck in LOSS_CHECKPOINTS:
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

    def get_dream_loss_trajectory(self):
        return {}


# ---------------------------------------------------------------------------
# RAND substrate (identical to step 1310)
# ---------------------------------------------------------------------------

class RandSubstrate:
    """Pure random action selection. W1, W2 fixed."""

    def __init__(self, n_actions, seed):
        self.n_actions = n_actions
        self._rng = np.random.RandomState(seed)
        rng_w = np.random.RandomState(seed + 99999)

        self.W1 = rng_w.randn(H1_DIM, ENC_DIM).astype(np.float32) * W_INIT_SCALE
        self.W2 = rng_w.randn(H2_DIM, H1_DIM).astype(np.float32) * W_INIT_SCALE
        rng_w.randn(n_actions, H2_DIM)

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
        a = 1.0 / self.n_obs
        self.running_mean = (1 - a) * self.running_mean + a * x
        enc = x - self.running_mean

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
        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_pred_losses.append(pred_loss)
        for ck in LOSS_CHECKPOINTS:
            if self._pred_loss_at[ck] is None and self.step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_pred_losses)), 6)

        action = int(self._rng.randint(self.n_actions))
        self._action_counts[action] += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return 0.0

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_dream_loss_trajectory(self):
        return {}


def make_substrate(condition, n_actions, seed):
    if condition == 'mlpl':
        return MLPLSubstrate(n_actions=n_actions, seed=seed)
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

        if steps == 200:
            i3_counts_at_200 = action_counts.copy()

        obs_arr = np.asarray(obs, dtype=np.float32)
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
    l_late = pred_loss_traj.get(5000)
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
    mlpl = chain_summary['mlpl']
    rand = chain_summary['rand']

    mlpl_cr = mlpl.get('mean_compression_ratio')
    rand_cr = rand.get('mean_compression_ratio')
    mlpl_kl = mlpl.get('mean_action_KL')
    rand_kl = rand.get('mean_action_KL')

    lines = []
    kill = False

    # Primary: cr@5K ≥ step 1310 cr@2K (0.9282) → plateaued
    if mlpl_cr is not None:
        if mlpl_cr >= CR_1310:
            lines.append(
                f"  >>> KILL: MLPL cr@5K={mlpl_cr:.4f} ≥ 1310 cr@2K={CR_1310} → compression plateaued"
            )
            kill = True
        else:
            lines.append(
                f"  >>> SIGNAL: MLPL cr@5K={mlpl_cr:.4f} < 1310 cr@2K={CR_1310} → compression continues"
            )

    # Secondary: worse than random
    if (mlpl_cr is not None and rand_cr is not None
            and mlpl_kl is not None and rand_kl is not None):
        mlpl_score = (1.0 / (mlpl_cr + 1e-8)) * mlpl_kl
        rand_score = (1.0 / (rand_cr + 1e-8)) * rand_kl
        cmp = '>' if mlpl_score > rand_score else '≤'
        tag = 'SIGNAL' if mlpl_score > rand_score else 'MISS'
        lines.append(
            f"  >>> {tag}: MLPL (1/cr × kl)={mlpl_score:.2f} {cmp} RAND={rand_score:.2f}"
        )
        if mlpl_score <= rand_score:
            lines.append("  >>> KILL: MLPL score ≤ RAND score")
            kill = True

    # Collapse check
    if mlpl_kl is not None:
        if mlpl_kl < 0.01:
            lines.append(f"  >>> KILL: MLPL action_KL={mlpl_kl:.4f} < 0.01 → collapsed")
            kill = True
        else:
            lines.append(f"  >>> OK: MLPL action_KL={mlpl_kl:.4f} ≥ 0.01")

    mlpl_rhae = mlpl.get('mean_RHAE', 0.0)
    rand_rhae = rand.get('mean_RHAE', 0.0)
    lines.append(f"  >>> SECONDARY: MLPL RHAE={mlpl_rhae:.2e} vs RAND RHAE={rand_rhae:.2e}")

    # Leo's predictions (informative, not kill criteria)
    if mlpl_cr is not None:
        pred1 = "CONFIRMED" if mlpl_cr < 0.8 else "WRONG"
        lines.append(f"  >>> PREDICTION 1 (cr@5K < 0.8): {pred1} (cr={mlpl_cr:.4f})")
    if mlpl.get('mean_wdrift') is not None:
        pred2 = "CONFIRMED" if mlpl['mean_wdrift'] > 0.5 else "WRONG"
        lines.append(f"  >>> PREDICTION 2 (wdrift > 0.5): {pred2} (wdrift={mlpl['mean_wdrift']:.4f})")

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
    print(f"Step 1311 — Multi-layer predictive coding, extended budget (5K steps)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"Architecture: enc(256)→W1(128×256)→h1→W2(64×128)→h2→W3(n×64)→action")
    print(f"K_INF={K_INF}, ETA={ETA}, INF_LR={INF_LR}")
    print(f"MAX_STEPS={MAX_STEPS}, LOSS_CHECKPOINTS={LOSS_CHECKPOINTS}")
    print(f"pred_loss = ||e1||²/||enc||² (Layer 1 reconstruction error)")
    print(f"Kill threshold: cr@5K ≥ {CR_1310} (step 1310 cr@2K)")
    print(f"Total: {len(CONDITIONS) * len(GAMES) * N_DRAWS} runs")
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
    chain_data = {c: {'RHAE': [], 'wdrift': [], 'action_KL': [], 'I3cv': [],
                      'compression_ratio': []} for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for draw_idx in range(N_DRAWS):
            t0 = time.time()
            for condition in CONDITIONS:
                result = run_draw(condition, game, draw_idx, solver_steps)
                all_results.append(result)

            elapsed_draw = time.time() - t0
            print(masked_draw_log(label, draw_idx, elapsed_draw))

        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, 1311))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r) + '\n')

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

    summary = {
        'chain_summary': chain_summary,
        'n_draws': N_DRAWS,
        'games_label_only': masked_game_list(GAME_LABELS),
        'kill_threshold_cr': CR_1310,
    }
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP 1311 — CHAIN-LEVEL SUMMARY\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        s = chain_summary[cond]
        print(f"  {lbl}:")
        print(f"    mean_RHAE={s['mean_RHAE']:.2e}  mean_wdrift={s['mean_wdrift']}  "
              f"mean_action_KL={s['mean_action_KL']}  mean_I3cv={s['mean_I3cv']}")
        pt = s['pred_loss_trajectory']
        print(f"    pred_loss_trajectory: {pt}")
        cr = s['mean_compression_ratio']
        print(f"    mean_compression_ratio (loss@5K/loss@500): {cr}")
    print()
    print("KILL ASSESSMENT (chain aggregates only):")
    kill_lines, kill = kill_assessment(chain_summary)
    for line in kill_lines:
        print(line)
    print(sep)
    print(f"\nResults saved to {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
