"""
Step 1321 — Shared representation: action selection from h1 (W2 eliminated, no W3)
Leo mail 3739, 2026-03-29.

## Insight

CNN SELFSUP-ENT worked (RHAE=2.4e-5) because conv layers are SHARED between prediction
and action. Multi-layer LPL fails because W1/W2 (prediction) and W3 (action) are separate.
10 W3 mechanisms failed because W3 is disconnected from the representation W1 builds.

## Fix: eliminate W3. Actions read directly from h1.

action_logits = h1  (128-dim refined representation from W1 @ enc)
W1 updates from prediction error → h1 changes → action distribution changes automatically.
One matrix. One coupling law. Prediction and action share the same computation.

## Architecture

Shared substrate:
  enc (256) → W1 (128×256) → h1 (128) [5-step inference refinement]
               W2 (64×128) → h2 (64)  [prediction pathway, updated via e2]

  Action: softmax(h1[:n_actions]) for n_actions ≤ 128
          softmax(h1[a % 128] for a in range(n_actions)) for n_actions > 128
  Updates: W1 += ETA * outer(h1, e1), W2 += ETA * outer(h2, e2). No W3.

R2 compliance: h1 = W1 @ enc is the prediction computation. Action reads h1 directly.
When W1 updates, both encoding quality AND action selection change in the same step.

Wiring changes (flagged to Leo, mail 3741):
  1. No relu on h1 for action selection (prevents zero-logit collapse on 7-action KB games)
  2. Softmax instead of argmax (collapse prevention — same as 1313 base pattern)
  3. Hash h1 to larger action spaces: logits[a] = h1[a % 128]

## Protocol
- MBPP + 2 masked ARC, seed-free, 1 run per game (try1: 2K, try2: 2K same substrate)
- 6 runs (3 games × 2 conditions: SHARED vs BASE)
- Primary metric: second_exposure_speedup = steps_to_L1(try1) / steps_to_L1(try2)
- All diagnostics in diagnostics.json (NOT in stdout, NOT in kill decisions)

## Kill criteria
- SHARED speedup ≤ BASE speedup → sharing doesn't help → KILL

## Predictions (Leo)
1. SHARED action distribution changes over episode (h1 evolves as W1 learns → actions evolve)
2. SHARED may reach L1 on easy games (h1 develops structure from prediction → actions improve)
3. If L1 reached: speedup > 1 possible (try2 starts with better W1 → better h1 from step 0)
"""
import sys, os, time, json, logging

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')

logging.disable(logging.INFO)

import numpy as np
from substrates.step0674 import _enc_frame
from prism_masked import (select_games, seal_mapping, label_filename,
                           masked_game_list, masked_run_log,
                           format_speedup, write_experiment_results)

GAMES, GAME_LABELS = select_games(seed=1321)

STEP = 1321
MAX_STEPS   = 2_000   # per try
MAX_SECONDS = 300

CONDITIONS = ['shared', 'base']
LABELS     = {'shared': 'SHARED', 'base': 'BASE'}

RESULTS_DIR = os.path.join('B:/M/the-search/experiments/compositions', 'results_1321')
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

SEED_A = 0   # try1 episode seed
SEED_B = 1   # try2 episode seed


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
# Substrates
# ---------------------------------------------------------------------------

class BaseSubstrate:
    """Multi-layer LPL with Hebbian W3 — 1313/1320 baseline. Deterministic init."""

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        self.W3 = np.zeros((n_actions, H2_DIM), dtype=np.float32)
        self._W1_init = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

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
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # Action from W3 @ h2 (standard base)
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

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))


class SharedSubstrate:
    """Multi-layer LPL, NO W3. Action selection reads directly from h1.

    h1 = refined 128-dim representation from W1 @ enc (5 inference steps).
    W1 updates from prediction error → h1 changes → action selection changes.
    W2 updates from prediction error (same as base).
    No W3. No separate action matrix. One coupling law.

    Action selection:
      n_actions ≤ 128: logits = h1[:n_actions]
      n_actions > 128: logits[a] = h1[a % 128]  (hash to larger space)
    Then softmax with T = 1/std (same as base).

    R2: action reads from h1 which IS the W1 prediction computation. W1 modified →
    h1 changes → both prediction accuracy AND action distribution change simultaneously.
    """

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.W1 = np.eye(H1_DIM, ENC_DIM, dtype=np.float32) * 0.1
        self.W2 = np.eye(H2_DIM, H1_DIM, dtype=np.float32) * 0.1
        # No W3
        self._W1_init = self.W1.copy()
        self.running_mean = np.zeros(ENC_DIM, np.float32)
        self.n_obs = 0
        self.step = 0
        self._action_counts = np.zeros(n_actions, np.float32)

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
        self.step += 1
        enc = self._centered_encode(obs_raw)
        h1, h2, e1, e2 = self._infer(enc)

        # Action from h1 directly (shared representation — no W3)
        if self.n_actions <= H1_DIM:
            logits = h1[:self.n_actions].astype(np.float32)
        else:
            # Hash: logits[a] = h1[a % H1_DIM]
            indices = np.arange(self.n_actions) % H1_DIM
            logits = h1[indices].astype(np.float32)

        std_s = float(np.std(logits))
        T = max(1.0 / (std_s + 1e-8), T_MIN)
        l = logits * T
        l -= l.max()
        probs = np.exp(l)
        probs /= probs.sum()
        action = int(np.random.choice(self.n_actions, p=probs))
        self._action_counts[action] += 1

        # Update W1 and W2 only — no W3
        self.W1 += ETA * np.outer(h1, e1)
        self.W2 += ETA * np.outer(h2, e2)
        np.clip(self.W1, -W_CLIP, W_CLIP, out=self.W1)
        np.clip(self.W2, -W_CLIP, W_CLIP, out=self.W2)

        return action

    def update_after_step(self, obs_next, action, reward):
        pass

    def on_level_transition(self):
        pass

    def compute_weight_drift(self):
        return float(np.linalg.norm(self.W1 - self._W1_init))


def make_substrate(condition, n_actions):
    if condition == 'shared':
        return SharedSubstrate(n_actions=n_actions)
    return BaseSubstrate(n_actions=n_actions)


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

def run_episode(env, substrate, n_actions, solver_level_steps, seed, max_steps=MAX_STEPS):
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
        'steps_taken': steps,
        'elapsed_seconds': round(elapsed, 2),
        'max_level': max_level,
        'level_first_step': {int(k): int(v) for k, v in level_first_step.items()},
        'arc_score': round(arc_score, 6),
        'RHAE': round(arc_score, 6),
        'I3_cv': i3_cv,
        'action_kl': action_kl,
        'wdrift': round(substrate.compute_weight_drift(), 4),
    }


# ---------------------------------------------------------------------------
# Game runner — try1 then try2 on same substrate
# ---------------------------------------------------------------------------

def run_game(condition, game_name, solver_level_steps):
    env = make_game(game_name)
    try:
        n_actions = int(env.n_actions)
    except AttributeError:
        n_actions = 4103

    substrate = make_substrate(condition, n_actions)

    result_try1 = run_episode(env, substrate, n_actions, solver_level_steps, seed=SEED_A)
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
        'game': game_name,
        'condition': condition,
        'try1': result_try1,
        'try2': result_try2,
        'second_exposure_speedup': speedup,
        'n_actions': n_actions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Step {STEP} — Shared representation: action from h1 (no W3)")
    print(f"Games: {masked_game_list(GAME_LABELS)} (ARC games masked)")
    print(f"Conditions: {CONDITIONS}")
    print(f"SHARED: W1+W2 only, action = softmax(h1[:n_actions]) — no W3")
    print(f"BASE: standard multi-layer LPL (W1+W2+W3, 1313 style)")
    print(f"Primary metric: second_exposure_speedup (steps_to_L1[try1] / steps_to_L1[try2])")
    print(f"6 runs (3 games × 2 conditions), try1+try2 per run (4K steps each).")
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
    speedup_by_game_cond = {}

    for game in GAMES:
        label = GAME_LABELS[game]
        print(f"=== {label} ===")
        solver_steps = solver_steps_cache[game]

        t0 = time.time()
        for condition in CONDITIONS:
            result = run_game(condition, game, solver_steps)
            all_results.append(result)
            speedup_by_game_cond[(label, condition)] = result['second_exposure_speedup']

        elapsed = time.time() - t0
        print(masked_run_log(label, elapsed))

        game_results = [r for r in all_results if r['game'] == game]
        out_path = os.path.join(RESULTS_DIR, label_filename(label, STEP))
        with open(out_path, 'w') as f:
            for r in game_results:
                masked_r = {k: v for k, v in r.items() if k != 'game'}
                masked_r['label'] = label
                f.write(json.dumps(masked_r, default=str) + '\n')

    # Chain speedup per condition
    speedup_by_condition = {}
    for cond in CONDITIONS:
        values = [v for (_, c), v in speedup_by_game_cond.items()
                  if c == cond and v is not None and v != float('inf') and v > 0]
        if values:
            speedup_by_condition[cond] = round(float(np.mean(values)), 4)
        else:
            any_inf = any(v == float('inf') for (_, c), v in speedup_by_game_cond.items()
                          if c == cond)
            speedup_by_condition[cond] = float('inf') if any_inf else None

    write_experiment_results(RESULTS_DIR, STEP, speedup_by_condition, all_results, CONDITIONS)

    sep = '=' * 100
    print(f"\n{sep}")
    print(f"STEP {STEP} — RESULT\n")
    for cond in CONDITIONS:
        lbl = LABELS[cond]
        sp = speedup_by_condition.get(cond)
        print(f"  {lbl}: second_exposure_speedup = {format_speedup(sp)}")
    print()

    print("  Per-game breakdown:")
    for label in sorted(set(l for l, _ in speedup_by_game_cond)):
        for cond in CONDITIONS:
            sp = speedup_by_game_cond.get((label, cond))
            print(f"    {label} / {LABELS[cond]}: {format_speedup(sp)}")
    print()

    sp_sh = speedup_by_condition.get('shared')
    sp_base = speedup_by_condition.get('base')

    print("KILL ASSESSMENT:")
    kill = False
    if sp_sh is None and sp_base is None:
        print("  >>> BOTH: second_exposure_speedup = N/A (L1 not reached)")
        print("  >>> KILL — no L1 in either condition")
        kill = True
    elif sp_sh is None:
        print(f"  >>> KILL: SHARED speedup = N/A, BASE = {format_speedup(sp_base)}")
        print("  >>> KILL — SHARED can't reach L1 while BASE can")
        kill = True
    elif sp_base is None:
        print(f"  >>> SIGNAL: SHARED speedup = {format_speedup(sp_sh)}, BASE = N/A")
        print("  >>> NO KILL — SHARED reaches L1, BASE can't")
    elif sp_sh > sp_base:
        print(f"  >>> SIGNAL: SHARED speedup = {format_speedup(sp_sh)} > BASE = {format_speedup(sp_base)}")
        print("  >>> NO KILL — shared representation improves second-exposure learning")
    else:
        print(f"  >>> KILL: SHARED speedup = {format_speedup(sp_sh)} ≤ BASE = {format_speedup(sp_base)}")
        print("  >>> KILL — sharing W1 for action doesn't improve learning speed")
        kill = True

    if not kill:
        print("  >>> Diagnostics in diagnostics.json (not kill criteria)")
    print(sep)
    print(f"\nResults: {RESULTS_DIR}\n")


if __name__ == '__main__':
    main()
