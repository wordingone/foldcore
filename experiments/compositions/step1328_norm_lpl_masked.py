"""
Step 1328 — Normalized LPL error (instability vs fundamental weakness)
Leo mail 3763, 2026-03-29.

One-line fix from step 1313 base: normalize e1 by ||enc|| before W1 update.

  Before: W1 += eta * outer(h1, e1)           — e1 unbounded, grows with enc magnitude
  After:  W1 += eta * outer(h1, e1_norm)       — e1_norm = e1 / (||enc|| + eps), scale-invariant

Kill: NORM-LPL cr ≈ BASE-LPL cr (both ≥ 0.9) → coupling law itself is too weak, not instability
Signal: NORM-LPL cr << BASE-LPL cr (NORM cr < 0.5) → instability was the problem, LPL works when stabilized

Note: Leo's pseudocode had outer(e1_norm, h1) → shape (256,128). W1 is (128,256).
Corrected to outer(h1, e1_norm) = (128,256) to match W1 shape. Intent unchanged.

Constitutional audit:
R0: Deterministic init (identity scaled). No seeds.
R1: PASS — prediction error e1 self-computed from W1 dynamics.
R2: PASS — same as step 1310. No backward mechanism.
R3: W1/W2 updated every step.
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
from prism_masked import (select_games, seal_mapping, label_filename,
                           masked_game_list, masked_run_log,
                           format_speedup, write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1328)

STEP        = 1328
MAX_STEPS   = 2_000
MAX_SECONDS = 300
SEED_A      = 0
SEED_B      = 1

CONDITIONS = ['norm_lpl', 'base_lpl']
LABELS     = {'norm_lpl': 'NORM-LPL', 'base_lpl': 'BASE-LPL'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1328')
PDIR        = 'B:/M/the-search/experiments/results/prescriptions'

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

LOSS_CHECKPOINTS = [500, 1000, 2000]


# ---------------------------------------------------------------------------
# Substrates
# ---------------------------------------------------------------------------

class LplSubstrate:
    """Multi-layer LPL, deterministic init. Base class — subclass overrides _lpl_update."""

    def __init__(self, n_actions, checkpoints=None):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)
        self._W1_init    = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self._step = 0
        ckpts = checkpoints if checkpoints is not None else LOSS_CHECKPOINTS
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at  = {ck: None for ck in ckpts}

    def _centered_encode(self, obs_raw):
        x = _enc_frame(np.asarray(obs_raw, dtype=np.float32))
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

    def _lpl_update(self, h1, h2, e1, e2, enc):
        """Override in subclass to change update rule."""
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)

    def process(self, obs_raw):
        self._step += 1
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

        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0

        self._lpl_update(h1, h2, e1, e2, enc)
        self.W3 += ETA * np.outer(action_onehot, h2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def reset_loss_tracking(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)


class NormLplSubstrate(LplSubstrate):
    """Normalized LPL: e1 divided by ||enc|| + eps before W1 update."""

    def _lpl_update(self, h1, h2, e1, e2, enc):
        enc_norm = float(np.linalg.norm(enc))
        e1_norm = e1 / (enc_norm + 1e-8)    # scale-invariant error signal
        self.W1 += ETA * np.outer(h1, e1_norm)
        self.W2 += ETA * np.outer(h2, e2)   # W2 unchanged (h1/e2 already bounded by W2 scale)


class BaseLplSubstrate(LplSubstrate):
    """Unnormalized LPL (same as step 1313 base). Control."""
    pass  # _lpl_update inherited unchanged


def make_substrate(condition, n_actions):
    if condition == 'norm_lpl':
        return NormLplSubstrate(n_actions=n_actions)
    return BaseLplSubstrate(n_actions=n_actions)


# ---------------------------------------------------------------------------
# Game factory and solvers
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


def compute_arc_score(level_first_step, solver_level_steps):
    if not level_first_step or not solver_level_steps:
        return 0.0
    scores = []
    for lvl, s_step in solver_level_steps.items():
        a_step = level_first_step.get(lvl)
        if a_step is not None and s_step > 0:
            scores.append((s_step / a_step) ** 2)
    return round(float(np.mean(scores)), 6) if scores else 0.0


def compute_action_kl(action_log, n_actions):
    if len(action_log) < 400:
        return None
    early_c = np.zeros(n_actions, np.float32)
    late_c  = np.zeros(n_actions, np.float32)
    for a in action_log[:200]:
        if 0 <= a < n_actions:
            early_c[a] += 1
    for a in action_log[-200:]:
        if 0 <= a < n_actions:
            late_c[a] += 1
    early_p = (early_c + 1e-8) / (early_c.sum() + 1e-8 * n_actions)
    late_p  = (late_c  + 1e-8) / (late_c.sum()  + 1e-8 * n_actions)
    return round(float(np.sum(early_p * np.log(early_p / late_p + 1e-12))), 4)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, solver_level_steps, seed):
    obs = env.reset(seed=seed)
    action_log = []
    action_counts = np.zeros(n_actions, np.float32)
    i3_counts_at_200 = None
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

    return {
        'steps_taken':     steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level':       max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score':       round(arc_score, 6),
        'RHAE':            round(arc_score, 6),
        'I3_cv':           i3_cv,
        'action_kl':       action_kl,
        'wdrift':          round(substrate.compute_weight_drift(), 4),
        'pred_loss_traj':  substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
    }


# ---------------------------------------------------------------------------
# Game runner
# ---------------------------------------------------------------------------

def run_game(game_name, condition, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = make_substrate(condition, n_actions)
    result_try1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)
    substrate.reset_loss_tracking()
    result_try2 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_B)

    l1_try1 = result_try1['level_first_step'].get(1)
    l1_try2 = result_try2['level_first_step'].get(1)

    if l1_try1 is not None and l1_try2 is not None and l1_try2 > 0:
        speedup = round(l1_try1 / l1_try2, 4)
    elif l1_try1 is None and l1_try2 is not None:
        speedup = float('inf')
    elif l1_try1 is not None and l1_try2 is None:
        speedup = 0.0
    else:
        speedup = None

    return {
        'game':                    game_name,
        'condition':               condition,
        'try1':                    result_try1,
        'try2':                    result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio':       result_try1['compression_ratio'],
        'n_actions':               n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Normalized LPL error (instability vs fundamental weakness)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"NORM-LPL: e1 / (||enc|| + eps) before W1 update")
    print(f"BASE-LPL: unnormalized (step 1313 base)")
    print(f"Kill: NORM cr ≈ BASE cr → coupling law too weak")
    print(f"Signal: NORM cr << BASE cr → instability was the problem")
    print(f"6 runs (3 games × 2 conditions), try1+try2 ({MAX_STEPS} steps each).")
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
    speedup_by_condition = {c: [] for c in CONDITIONS}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        for condition in CONDITIONS:
            t0 = time.time()
            result = run_game(game, condition, solver_steps)
            all_results.append(result)
            speedup_by_condition[condition].append(result['second_exposure_speedup'])
            elapsed = time.time() - t0
            print(masked_run_log(f"{label}/{LABELS[condition]}", elapsed))

        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        game_results = [r for r in all_results if GAME_LABELS.get(r['game']) == label]
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r, default=str) + '\n')

    def chain_agg(vals):
        finite = [v for v in vals if v is not None and v != float('inf') and v > 0]
        if finite:
            return round(float(np.mean(finite)), 4)
        if any(v == float('inf') for v in vals if v is not None):
            return float('inf')
        return None

    final_speedup = {c: chain_agg(speedup_by_condition[c]) for c in CONDITIONS}
    write_experiment_results(RESULTS_DIR, STEP, final_speedup, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT (NORM-LPL vs BASE-LPL)\n")

    for cond in CONDITIONS:
        sp = final_speedup[cond]
        cr_vals = [r['compression_ratio'] for r in all_results
                   if r['condition'] == cond and r['compression_ratio'] is not None]
        cr_str = f"cr_mean={np.mean(cr_vals):.4f}" if cr_vals else "cr=N/A"
        print(f"  {LABELS[cond]}: speedup={format_speedup(sp)}  {cr_str}")
    print()

    print("Compression detail (try1):")
    for r in all_results:
        label = GAME_LABELS.get(r['game'], '?')
        cr    = r['compression_ratio']
        traj  = r['try1']['pred_loss_traj']
        cond  = LABELS[r['condition']]
        print(f"  {label}/{cond}: cr={cr}  traj={traj}")
    print()

    norm_cr_vals = [r['compression_ratio'] for r in all_results if r['condition'] == 'norm_lpl'
                    and r['compression_ratio'] is not None]
    base_cr_vals = [r['compression_ratio'] for r in all_results if r['condition'] == 'base_lpl'
                    and r['compression_ratio'] is not None]
    norm_cr = float(np.mean(norm_cr_vals)) if norm_cr_vals else None
    base_cr = float(np.mean(base_cr_vals)) if base_cr_vals else None

    print("KILL/SIGNAL ASSESSMENT:")
    if norm_cr is None and base_cr is None:
        print(f"  >>> INCONCLUSIVE: cr=N/A both conditions.")
    elif norm_cr is not None and base_cr is not None and norm_cr >= 0.9 and base_cr >= 0.9:
        print(f"  >>> KILL: NORM cr={norm_cr:.4f} ≈ BASE cr={base_cr:.4f}")
        print(f"  >>> Both ≥ 0.9 — coupling law is fundamentally too weak, not just unstable.")
    elif norm_cr is not None and base_cr is not None and norm_cr < base_cr * 0.7:
        print(f"  >>> SIGNAL: NORM cr={norm_cr:.4f} << BASE cr={base_cr:.4f}")
        print(f"  >>> Instability was the problem — normalized LPL compresses meaningfully.")
    else:
        print(f"  >>> PARTIAL: NORM cr={norm_cr}  BASE cr={base_cr}")
        print(f"  >>> Normalization helps but not enough for signal threshold.")
    print(sep)


if __name__ == '__main__':
    main()
