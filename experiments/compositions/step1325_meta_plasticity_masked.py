"""
Step 1325 — Catalog NEW: Meta-learned plasticity (substrate discovers its own update rule).
Leo mail 3751, 2026-03-29.

## Question

LPL is R2-compliant but too weak. Can the substrate DISCOVER a stronger update rule
through the same prediction-error coupling? theta = [alpha, beta, gamma] parameterizes
W1's update as a combination of Hebbian, anti-Hebbian, and decay terms. theta updates
from the same prediction error that updates W1 (R2-compliant).

## Architecture

Multi-layer LPL base (W1, W2, W3) with META condition adding parameterized W1 update:

Fixed update (BASE = step 1313):
    W1 += eta * outer(h1, e1)                    # standard Hebbian

Parameterized update (META):
    term_hebbian    = outer(h1, e1)              # standard Hebbian
    term_anti       = -outer(h1, h1) @ W1        # anti-Hebbian decorrelation
    term_decay      = -W1                         # weight decay

    dW1 = eta * (theta[0]*term_hebb + theta[1]*term_anti + theta[2]*term_decay)
    W1 += dW1

    # Theta update: credit each term by first-order prediction error reduction
    # Leo's formula: correlation = dot(term.ravel(), -delta_e1.ravel())
    # Fix: Leo's delta_e1 would need shape (128,256) for the dot to work.
    # Implementation: first-order credit assignment (mathematically equivalent in intent):
    #   credit_i = dot(e1, term_i.T @ h1)  [scalar — first-order reduction in ||e1||^2]
    # For Hebbian: credit = ||e1||^2 * ||h1||^2 (always positive = correctly reinforced).
    # For anti-Hebbian and decay: credit only if they also reduce prediction error.
    for i, term in enumerate([term_hebb, term_anti, term_decay]):
        credit = float(np.dot(e1, term.T @ h1))
        theta[i] += eta_theta * credit
    theta = clip(theta, -5, 5)

## Constitutional audit
| R0 | theta=[1,0,0] = pure Hebbian start (deterministic) |
| R1 | PASS — all from prediction error |
| R2 | PASS — theta updates from same coupling law as W1 |
| R3 | W1 modified by parameterized rule; theta modifies the rule itself |
| R4 | second_exposure_speedup |
| R5 | PASS |
| R6 | theta=[1,0,0] fixed → BASE (step 1313) |

NEW component: meta-learned plasticity (not in C1-C33 catalog). Justification in spec.

## Protocol
- MBPP + 2 masked ARC, seed-free, 1 run per game
- 2K steps try1, 2K steps try2 (same substrate — weights persist)
- Conditions: META vs BASE
- One metric: second_exposure_speedup

## Kill criteria
- META speedup ≤ BASE speedup → KILL
- theta converges to [1,0,0] on all games → nothing discovered → KILL (in diagnostics)
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

GAMES, GAME_LABELS = select_games(seed=1325)

STEP = 1325
MAX_STEPS   = 2_000   # per try
MAX_SECONDS = 120     # per episode (hard cap)

CONDITIONS = ['meta', 'base']
LABELS     = {'meta': 'META', 'base': 'BASE'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1325')
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

ENC_DIM   = 256
H1_DIM    = 128
H2_DIM    = 64
ETA       = 0.001
ETA_THETA = 0.0001    # small: theta updates are large in magnitude, need slow rate
INF_LR    = 0.05
K_INF     = 5
W_CLIP    = 100.0
THETA_CLIP = 5.0
T_MIN     = 0.01

LOSS_CHECKPOINTS = [500, 1000, 2000]
SEED_A = 0
SEED_B = 1


# ---------------------------------------------------------------------------
# Game factory and solvers (same as step 1313)
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


# ---------------------------------------------------------------------------
# Substrates
# ---------------------------------------------------------------------------

class BaseSubstrate:
    """Multi-layer LPL, standard Hebbian (theta=[1,0,0] fixed). Step 1313 base."""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self._step = 0
        self._action_counts = np.zeros(n_actions, np.float32)
        self._recent_losses = deque(maxlen=50)
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

    def reset_loss_tracking(self):
        self._step = 0
        self._recent_losses.clear()
        self._pred_loss_at = {ck: None for ck in LOSS_CHECKPOINTS}

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

    def process(self, obs_raw):
        self._step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        scores = self.W3 @ h2
        std_s = float(np.std(scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = scores * T
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
        self._recent_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def get_compression_ratio(self):
        l_early = self._pred_loss_at.get(500)
        l_late  = self._pred_loss_at.get(2000)
        if l_early is not None and l_late is not None and l_early > 1e-8:
            return round(l_late / l_early, 4)
        return None

    def get_pred_loss_trajectory(self):
        return dict(self._pred_loss_at)

    def get_theta(self):
        return [1.0, 0.0, 0.0]  # fixed for BASE


class MetaSubstrate(BaseSubstrate):
    """Multi-layer LPL with parameterized W1 update rule.

    theta = [alpha, beta, gamma] — weights for Hebbian, anti-Hebbian, decay terms.
    theta starts at [1, 0, 0] (pure Hebbian) and updates via credit assignment.

    Credit formula: credit_i = dot(e1, term_i.T @ h1)
    This is the first-order reduction in ||e1||^2 from term_i.
    Leo's formula: np.dot(term.ravel(), -delta_e1.ravel()) — shape mismatch fix.
    See mail 3752 for explanation.
    """

    def __init__(self, n_actions):
        super().__init__(n_actions)
        self.theta = np.array([1.0, 0.0, 0.0], dtype=np.float64)  # [alpha, beta, gamma]
        self._theta_history = []   # snapshots at checkpoints for diagnostics

    def process(self, obs_raw):
        self._step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # Action selection (same as BASE)
        scores = self.W3 @ h2
        std_s = float(np.std(scores))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        logits = scores * T
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        # Parameterized W1 update
        term_hebb  = np.outer(h1, e1)                    # (H1_DIM, ENC_DIM)
        term_anti  = -(np.outer(h1, h1) @ self.W1)      # (H1_DIM, ENC_DIM)
        term_decay = -self.W1                             # (H1_DIM, ENC_DIM)
        terms = [term_hebb, term_anti, term_decay]

        dW1 = ETA * (self.theta[0] * term_hebb
                   + self.theta[1] * term_anti
                   + self.theta[2] * term_decay)
        self.W1 = self.W1 + dW1

        # Theta update: credit assignment via first-order prediction error reduction
        # credit_i = dot(e1, term_i.T @ h1) — scalar measure of how much term_i
        # reduces ||e1||^2 in the direction of h1
        for i, term in enumerate(terms):
            credit = float(np.dot(e1, term.T @ h1))
            self.theta[i] += ETA_THETA * credit
        np.clip(self.theta, -THETA_CLIP, THETA_CLIP, out=self.theta)

        # Standard W2, W3 updates (unchanged)
        action_onehot = np.zeros(self.n_actions, np.float32)
        action_onehot[action] = 1.0
        self.W2 += ETA * np.outer(h2, e2)
        self.W3 += ETA * np.outer(action_onehot, h2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)
        np.clip(self.W3, -W_CLIP, W_CLIP, out=self.W3)

        # Loss tracking
        enc_norm_sq = float(np.dot(enc, enc))
        pred_loss = float(np.dot(e1, e1)) / (enc_norm_sq + 1e-8)
        self._recent_losses.append(pred_loss)
        for ck in self._pred_loss_at:
            if self._pred_loss_at[ck] is None and self._step >= ck:
                self._pred_loss_at[ck] = round(float(np.mean(self._recent_losses)), 6)
                self._theta_history.append({'step': self._step,
                                            'theta': self.theta.tolist()})

        return action

    def get_theta(self):
        return self.theta.tolist()

    def get_theta_history(self):
        return list(self._theta_history)


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

    return {
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'pred_loss_traj': substrate.get_pred_loss_trajectory(),
        'compression_ratio': substrate.get_compression_ratio(),
        'theta': substrate.get_theta(),
    }


# ---------------------------------------------------------------------------
# Game runner — try1 then try2 on SAME substrate (weights and theta persist)
# ---------------------------------------------------------------------------

def run_game(game_name, condition, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    if condition == 'meta':
        substrate = MetaSubstrate(n_actions=n_actions)
    else:
        substrate = BaseSubstrate(n_actions=n_actions)

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

    theta_final = substrate.get_theta()
    theta_history = substrate.get_theta_history() if hasattr(substrate, 'get_theta_history') else []

    return {
        'game': game_name,
        'condition': condition,
        'try1': result_try1,
        'try2': result_try2,
        'second_exposure_speedup': speedup,
        'compression_ratio': result_try1['compression_ratio'],
        'n_actions': n_actions,
        'theta_final': theta_final,
        'theta_history': theta_history,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Meta-learned plasticity (R2-compliant)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"R2 PASS — theta updates from same coupling law as W1")
    print(f"Primary metric: second_exposure_speedup. Conditions: META vs BASE")
    print(f"6 runs (3 games × 2 conditions), try1+try2 per run ({MAX_STEPS} steps each).")
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
            cond_label = LABELS[condition]
            print(masked_run_log(f"{label}/{cond_label}", elapsed))

        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        game_results = [r for r in all_results if GAME_LABELS.get(r['game']) == label]
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r, default=str) + '\n')

    # Chain aggregates per condition
    def chain_speedup(vals):
        finite = [v for v in vals if v is not None and v != float('inf') and v > 0]
        if finite:
            return round(float(np.mean(finite)), 4)
        if any(v == float('inf') for v in vals if v is not None):
            return float('inf')
        return None

    final_speedup = {c: chain_speedup(speedup_by_condition[c]) for c in CONDITIONS}
    write_experiment_results(RESULTS_DIR, STEP, final_speedup, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT\n")
    for cond in CONDITIONS:
        sp = final_speedup[cond]
        print(f"  {LABELS[cond]}: second_exposure_speedup = {format_speedup(sp)}")
    print()

    # Per-game theta summary (for stdout visibility — not kill criterion)
    print("  Theta diagnostics (META condition final theta=[alpha, anti, decay]):")
    for r in all_results:
        if r['condition'] == 'meta':
            label = GAME_LABELS.get(r['game'], '?')
            th = r['theta_final']
            th_str = f"[{th[0]:.3f}, {th[1]:.3f}, {th[2]:.3f}]"
            print(f"    {label}: theta={th_str}")
    print()

    print("KILL ASSESSMENT:")
    meta_sp = final_speedup['meta']
    base_sp = final_speedup['base']
    meta_numeric = meta_sp if (meta_sp is not None and meta_sp != float('inf')) else (999 if meta_sp == float('inf') else -1)
    base_numeric = base_sp if (base_sp is not None and base_sp != float('inf')) else (999 if base_sp == float('inf') else -1)

    if meta_sp is None:
        print("  >>> KILL: META speedup=N/A (no L1 reached)")
    elif meta_numeric <= base_numeric:
        print(f"  >>> KILL: META speedup={format_speedup(meta_sp)} ≤ BASE speedup={format_speedup(base_sp)}")
        print("  >>> Meta-learning doesn't improve second-exposure performance.")
    else:
        print(f"  >>> SIGNAL: META speedup={format_speedup(meta_sp)} > BASE speedup={format_speedup(base_sp)}")
        print("  >>> NO KILL — meta-learned rule improves speedup. Investigate theta drift.")
    print(sep)
    print(f"\nResults: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
