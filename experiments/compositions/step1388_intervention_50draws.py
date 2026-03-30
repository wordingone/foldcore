"""
Step 1388 — Intervention tracking, 50 draws (power increase from 1387).
Leo mail 3972, 2026-03-30. Same mechanism as 1387 (confirmed working), 50 draws for statistical power.

Both paths closed (1379-1386). Root cause: pixel-level prediction learning can't
distinguish action effects from game dynamics (animations overwhelm action signal).

New approach: measure ONLY the pixel change at the action's TARGET POSITION.
Simplest possible intervention — "did anything change WHERE I acted?"

Architecture: No SSM. No neural network. Simple tabular tracker.
- For ARC click actions: 8×8 region grid (64 regions on 64×64 canvas).
  effect_magnitude[region] += |pixel_change_at_target_5x5_window|
- For keyboard/MBPP: per-action effect tracking (global obs change).
- Try2: softmax(effect_magnitude / T) over action space.

I3 stage instrumentation:
  I3: CV of region visit distribution at step 200 in try2 (both conditions).
  I1/I4/I5/R3: N/A (no encoding substrate per gate 3 resolution, mail 3970).

R2: Flagged as PROBE. RANDOM condition IS the R6 deletion test (no effect tables).

Conditions (30 draws, paired):
  INTERVENTION: try1 argmin coverage + effect tracking; try2 softmax(effect_magnitude / T).
  RANDOM:       try1 argmin coverage; try2 uniform random.

Seeds: 14230-14279.
"""

import os
import sys
import json
import time
import math
import numpy as np

sys.path.insert(0, 'B:/M/the-search/experiments/compositions')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')
sys.path.insert(0, 'B:/M/the-search/environments')
sys.path.insert(0, 'B:/M/the-search')

from prism_masked import (
    select_games, seal_mapping, label_filename, det_weights,
    compute_progress_speedup, compute_rhae_try2,
    mask_result_row, ARC_OPTIMAL_STEPS_PROXY,
    masked_game_list,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STEP        = 1388
N_DRAWS     = 50
DRAW_SEEDS  = [14230 + i for i in range(N_DRAWS)]
TRY1_STEPS  = 2000
TRY2_STEPS  = 2000
TRY2_SEED   = 4
MAX_SECONDS = 280

RESULTS_DIR     = os.path.join('B:/M/the-search/experiments/compositions', f'results_{STEP}')
MLP_TP_BASELINE = 4.59e-5

N_REGIONS     = 64    # 8×8 grid on 64×64 canvas
N_KEYBOARD    = 7     # action IDs 0..6 are keyboard/special
TEMPERATURE   = 3.0

CONDITIONS    = ['INTERVENTION', 'RANDOM']

MAX_N_ACTIONS = 4103
I3_STEP       = 200
I4_EARLY_MAX  = 100
I4_LATE_MIN   = 1900
TIER1_STEPS   = 200


# ---------------------------------------------------------------------------
# Game helpers
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


def get_optimal_steps(game_name, seed):
    gn = game_name.lower().strip()
    if gn == 'mbpp' or gn.startswith('mbpp_'):
        import mbpp_game
        problem_idx = int(seed) % mbpp_game.N_EVAL_PROBLEMS
        solver = mbpp_game.compute_solver_steps(problem_idx)
        return solver.get(1)
    return ARC_OPTIMAL_STEPS_PROXY


# ---------------------------------------------------------------------------
# Intervention helpers
# ---------------------------------------------------------------------------

def _is_arc_obs(arr):
    return arr.ndim == 3 and arr.shape[-2] == 64 and arr.shape[-1] == 64


def _action_to_region(action_id):
    """Map ARC click action → 8×8 region index (0..63). None for non-click."""
    if action_id < N_KEYBOARD:
        return None
    click_idx = action_id - N_KEYBOARD
    if click_idx >= 64 * 64:
        return None
    x = click_idx % 64
    y = click_idx // 64
    return (x // 8) + (y // 8) * 8


def _build_arc_scores(n_actions, region_effect_mag, keyboard_effect_mag):
    """Vectorized: score[action] for ARC game. Returns (n_actions,) float64."""
    scores = np.zeros(n_actions, dtype=np.float64)
    # Keyboard actions
    kb = min(N_KEYBOARD, n_actions, len(keyboard_effect_mag))
    scores[:kb] = keyboard_effect_mag[:kb]
    # Click actions
    n_clicks = min(n_actions - N_KEYBOARD, 64 * 64)
    if n_clicks > 0:
        ci = np.arange(n_clicks, dtype=np.int64)
        xs = ci % 64
        ys = ci // 64
        regions = (xs // 8) + (ys // 8) * 8
        scores[N_KEYBOARD:N_KEYBOARD + n_clicks] = region_effect_mag[regions]
    return scores


def _entropy_from_actions(actions, n_actions):
    if not actions or n_actions == 0:
        return None
    counts = np.zeros(n_actions, dtype=np.float64)
    for a in actions:
        counts[a] += 1.0
    counts /= counts.sum()
    h = -np.sum(counts[counts > 0] * np.log(counts[counts > 0]))
    return float(h) if h > 0 else None


# ---------------------------------------------------------------------------
# Intervention substrate
# ---------------------------------------------------------------------------

class InterventionSubstrate:
    """
    Tabular effect tracker. No neural network.

    Try1: argmin coverage (for both conditions). Accumulates effect_magnitude.
    Try2 (INTERVENTION): softmax(effect_magnitude / T).
    Try2 (RANDOM):       uniform random.
    """

    def __init__(self, n_actions, mode):
        self.n_actions = n_actions
        self.mode      = mode  # 'intervention' or 'random'

        # Effect tables — shared try1 → try2
        self._region_effect_mag  = np.zeros(N_REGIONS, dtype=np.float64)
        self._keyboard_effect_mag = np.zeros(max(N_KEYBOARD, n_actions), dtype=np.float64)

        # I3: region visit distribution at step I3_STEP in try2
        self._try2_region_visits = np.zeros(N_REGIONS, dtype=np.int64)
        self._try2_visit_at_i3   = None  # snapshot at step I3_STEP

        # I4
        self._try2_actions_early = []
        self._try2_actions_late  = []

        # Episode state
        self._obs_prev       = None
        self._in_try2        = False
        self._step           = 0
        self._visit_count    = np.zeros(n_actions, dtype=np.int64)
        self._current_level  = 0
        self._try2_max_level = 0
        self._is_arc         = None  # lazily detected

        # Diagnostics
        self._effect_events = 0
        self._total_effect  = 0.0

    def _detect_arc(self, obs_arr):
        if self._is_arc is None:
            a = np.asarray(obs_arr, dtype=np.float32)
            self._is_arc = _is_arc_obs(a)
        return self._is_arc

    def _measure_effect(self, obs_before, obs_after, action_id):
        """Localized pixel change at action target. Returns float."""
        if obs_before is None or obs_after is None:
            return 0.0
        b = np.asarray(obs_before, dtype=np.float32)
        a = np.asarray(obs_after,  dtype=np.float32)
        if _is_arc_obs(b):
            # Use first channel (64×64)
            bf = b[0] if b.ndim == 3 else b
            af = a[0] if a.ndim == 3 else a
            if action_id >= N_KEYBOARD:
                # Localized: 5×5 window at click target
                click_idx = action_id - N_KEYBOARD
                x = click_idx % 64
                y = click_idx // 64
                x0, x1 = max(0, x - 2), min(64, x + 3)
                y0, y1 = max(0, y - 2), min(64, y + 3)
                return float(np.mean(np.abs(af[y0:y1, x0:x1] - bf[y0:y1, x0:x1])))
            else:
                return float(np.mean(np.abs(af - bf)))
        else:
            # MBPP/other: global change on 1D obs
            bf = b.flatten()
            af = a.flatten()
            return float(np.mean(np.abs(af - bf)))

    def process(self, obs_arr):
        is_arc = self._detect_arc(obs_arr)

        if self._in_try2 and self.mode == 'intervention':
            # Softmax over effect magnitudes
            if is_arc:
                scores = _build_arc_scores(
                    self.n_actions, self._region_effect_mag, self._keyboard_effect_mag)
            else:
                scores = self._keyboard_effect_mag[:self.n_actions].copy()
            scores /= TEMPERATURE
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()
            action = int(np.random.choice(self.n_actions, p=probs))
        elif self._in_try2 and self.mode == 'random':
            # Uniform random
            action = int(np.random.randint(self.n_actions))
        else:
            # Try1 (both conditions): argmin visit coverage
            min_c      = self._visit_count.min()
            candidates = np.where(self._visit_count == min_c)[0]
            action     = int(np.random.choice(candidates))

        self._visit_count[action] += 1

        if self._in_try2:
            region = _action_to_region(action)
            if region is not None:
                self._try2_region_visits[region] += 1
            if self._step == I3_STEP:
                self._try2_visit_at_i3 = self._try2_region_visits.copy()
            if self._step <= I4_EARLY_MAX:
                self._try2_actions_early.append(action)
            elif self._step >= I4_LATE_MIN:
                self._try2_actions_late.append(action)

        self._obs_prev = obs_arr
        self._step    += 1
        return action

    def update_after_step(self, obs_next, action, reward):
        """Record pixel effect of the action taken."""
        if self._obs_prev is None or obs_next is None:
            return
        effect = self._measure_effect(self._obs_prev, obs_next, action)
        if effect > 0:
            self._effect_events += 1
            self._total_effect  += effect
        is_arc = self._detect_arc(np.asarray(self._obs_prev, dtype=np.float32))
        if is_arc:
            region = _action_to_region(action)
            if region is not None:
                self._region_effect_mag[region] += effect
            else:
                if action < len(self._keyboard_effect_mag):
                    self._keyboard_effect_mag[action] += effect
        else:
            if action < len(self._keyboard_effect_mag):
                self._keyboard_effect_mag[action] += effect

    def on_level_transition(self, new_level=None):
        self._obs_prev = None
        if new_level is not None:
            self._current_level = new_level
            if self._in_try2:
                self._try2_max_level = max(self._try2_max_level, new_level)

    def prepare_for_try2(self):
        self._in_try2        = True
        self._step           = 0
        self._obs_prev       = None
        self._visit_count[:] = 0
        self._current_level  = 0
        # Effect tables carry over (try1 → try2)
        # PRNG seeded externally after this call

    def compute_stage_metrics(self):
        # I3: CV of region visit distribution at step I3_STEP
        i3_cv = None
        if self._try2_visit_at_i3 is not None:
            v = self._try2_visit_at_i3.astype(np.float64)
            mean = v.mean()
            if mean > 0:
                i3_cv = round(float(v.std() / mean), 4)

        # I4: entropy change
        h_early = _entropy_from_actions(self._try2_actions_early, self.n_actions)
        h_late  = _entropy_from_actions(self._try2_actions_late,  self.n_actions)
        i4_red  = None
        if h_early and h_late and h_early > 0:
            i4_red = round((h_early - h_late) / h_early, 4)

        # Effect diagnostics
        nz_regions      = int(np.sum(self._region_effect_mag > 0))
        effect_mag_norm = round(float(np.linalg.norm(self._region_effect_mag)), 4)

        return {
            'i3_cv':           i3_cv,
            'i4_h_early':      round(h_early, 4) if h_early else None,
            'i4_h_late':       round(h_late,  4) if h_late  else None,
            'i4_reduction':    i4_red,
            'i1_within':       None,
            'i1_between':      None,
            'i1_pass':         None,
            'i5_max_level':    self._try2_max_level,
            'r3_weight_diff':  None,
            'nz_regions':      nz_regions,
            'effect_mag_norm': effect_mag_norm,
            'effect_events':   self._effect_events,
        }


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, substrate, n_actions, seed, max_steps):
    obs = env.reset(seed=seed)
    steps  = 0
    level  = 0
    steps_to_first_progress = None
    t_start     = time.time()
    fresh_episode = True

    while steps < max_steps:
        if time.time() - t_start > MAX_SECONDS:
            break
        if obs is None:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        action  = substrate.process(obs_arr) % n_actions
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
            if steps_to_first_progress is None:
                steps_to_first_progress = steps
            level = cl
            substrate.on_level_transition(new_level=cl)
        if done:
            obs = env.reset(seed=seed)
            substrate.on_level_transition(new_level=0)
            level = 0
            fresh_episode = True
        else:
            obs = obs_next

    return steps_to_first_progress, round(time.time() - t_start, 2)


# ---------------------------------------------------------------------------
# Binomial p-value (one-sided)
# ---------------------------------------------------------------------------

def binomial_p_one_sided(wins, n):
    if n == 0:
        return 1.0
    from math import comb
    p = sum(comb(n, k) * (0.5 ** n) for k in range(wins, n + 1))
    return round(p, 6)


# ---------------------------------------------------------------------------
# Draw runner
# ---------------------------------------------------------------------------

def run_draw(draw_idx, draw_seed, cond_name, max_steps):
    games, game_labels = select_games(seed=draw_seed)
    draw_dir = os.path.join(RESULTS_DIR, cond_name, f'draw{draw_idx}')
    os.makedirs(draw_dir, exist_ok=True)
    seal_mapping(draw_dir, games, game_labels)

    draw_results    = []
    try2_progress   = {}
    optimal_steps_d = {}
    mode = cond_name.lower()  # 'intervention' or 'random'

    for game_idx, (game_name, label) in enumerate(zip(games, game_labels.values())):
        env = make_game(game_name)
        try:
            n_actions = int(env.n_actions)
        except AttributeError:
            n_actions = MAX_N_ACTIONS

        substrate = InterventionSubstrate(n_actions=n_actions, mode=mode)

        p1, t1 = run_episode(env, substrate, n_actions, seed=0,         max_steps=max_steps)
        substrate.prepare_for_try2()
        np.random.seed(draw_seed * 1000 + 1)   # PRNG fix
        p2, t2 = run_episode(env, substrate, n_actions, seed=TRY2_SEED, max_steps=max_steps)

        stage = substrate.compute_stage_metrics()

        speedup = compute_progress_speedup(p1, p2)
        opt     = get_optimal_steps(game_name, TRY2_SEED)
        eff_sq  = 0.0
        if p2 is not None and opt is not None and opt > 0:
            eff_sq = round(min(1.0, opt / p2) ** 2, 6)

        try2_progress[label]   = p2
        optimal_steps_d[label] = opt

        row = {
            'draw': draw_idx, 'label': label, 'game': game_name,
            'condition': cond_name,
            'p1': p1, 'p2': p2, 'speedup': speedup,
            'eff_sq': eff_sq, 'optimal_steps': opt,
            't1': t1, 't2': t2,
            'i3_cv':           stage['i3_cv'],
            'i4_h_early':      stage['i4_h_early'],
            'i4_h_late':       stage['i4_h_late'],
            'i4_reduction':    stage['i4_reduction'],
            'i1_within':       stage['i1_within'],
            'i1_between':      stage['i1_between'],
            'i1_pass':         stage['i1_pass'],
            'i5_max_level':    stage['i5_max_level'],
            'r3_weight_diff':  stage['r3_weight_diff'],
            'nz_regions':      stage['nz_regions'],
            'effect_mag_norm': stage['effect_mag_norm'],
            'effect_events':   stage['effect_events'],
        }
        masked_row = mask_result_row(row, game_labels)

        fn = os.path.join(draw_dir, label_filename(label, STEP))
        with open(fn, 'w') as f:
            f.write(json.dumps(masked_row) + '\n')

        draw_results.append(masked_row)

    rhae         = compute_rhae_try2(try2_progress, optimal_steps_d)
    nz_r_mean    = round(float(np.mean([r.get('nz_regions', 0)       for r in draw_results])), 1)
    eff_mean     = round(float(np.mean([r.get('effect_mag_norm', 0.0) for r in draw_results])), 4)

    print(f"  [{cond_name}] Draw {draw_idx} RHAE={rhae:.6e}  "
          f"nz_regions={nz_r_mean:.0f}  effect_mag={eff_mean:.4f}")
    return round(rhae, 7), draw_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Tier 1: timing check ──────────────────────────────────────────────
    print("=== TIER 1: timing check ===")
    games_t1, _ = select_games(seed=DRAW_SEEDS[0])
    env_t1      = make_game(games_t1[0])
    try:
        na_t1 = int(env_t1.n_actions)
    except AttributeError:
        na_t1 = MAX_N_ACTIONS

    sub_t1 = InterventionSubstrate(n_actions=na_t1, mode='intervention')
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=50)   # warmup
    t0 = time.time()
    run_episode(env_t1, sub_t1, na_t1, seed=0, max_steps=TIER1_STEPS)
    tier1_elapsed = time.time() - t0

    ms_per_step  = tier1_elapsed / TIER1_STEPS * 1000
    n_eps_total  = N_DRAWS * 3 * 2 * len(CONDITIONS)
    est_total_s  = ms_per_step / 1000 * TRY1_STEPS * n_eps_total
    print(f"  {TIER1_STEPS} steps: {tier1_elapsed:.2f}s  ({ms_per_step:.2f}ms/step)")
    print(f"  Estimated total: {est_total_s:.0f}s ({est_total_s/60:.1f} min)")

    if est_total_s > MAX_SECONDS:
        max_steps = max(200, int(
            (MAX_SECONDS * 0.85) / (ms_per_step / 1000 * n_eps_total)
        ))
        print(f"  Budget exceeded — capping at {max_steps} steps")
    else:
        max_steps = TRY1_STEPS
        print(f"  Under budget — full {max_steps} steps")

    # ── Full experiment: INTERVENTION vs RANDOM, 30 draws ─────────────────
    print(f"\n=== STEP {STEP}: INTERVENTION vs RANDOM, {N_DRAWS} draws, {max_steps} steps ===")
    print(f"Seeds: {DRAW_SEEDS[0]}-{DRAW_SEEDS[-1]}")

    rhae_by_cond = {c: [] for c in CONDITIONS}

    for cond_name in CONDITIONS:
        print(f"\n--- Condition: {cond_name} ---")
        t_cond = time.time()
        for di, ds in enumerate(DRAW_SEEDS):
            rhae, _ = run_draw(di, ds, cond_name, max_steps)
            rhae_by_cond[cond_name].append(round(rhae, 7))
        print(f"  {cond_name} done in {time.time()-t_cond:.0f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {
        'step':            STEP,
        'n_draws':         N_DRAWS,
        'draw_seeds':      DRAW_SEEDS,
        'max_steps_used':  max_steps,
        'ms_per_step':     round(ms_per_step, 2),
        'mlp_tp_baseline': MLP_TP_BASELINE,
        'conditions':      {},
    }

    print(f"\n=== RESULTS ===")
    for cond_name in CONDITIONS:
        rhae_list  = rhae_by_cond[cond_name]
        chain_mean = round(sum(rhae_list) / len(rhae_list), 7)
        nz         = sum(1 for r in rhae_list if r > 0)
        print(f"\n{cond_name}:  chain_mean={chain_mean:.6e}  nz={nz}/{N_DRAWS}")
        summary['conditions'][cond_name] = {
            'chain_mean': chain_mean,
            'nz':         nz,
            'rhae_list':  rhae_list,
        }

    # Paired sign test: INTERVENTION vs RANDOM
    int_rhae = rhae_by_cond['INTERVENTION']
    rnd_rhae = rhae_by_cond['RANDOM']
    wins   = sum(1 for a, b in zip(int_rhae, rnd_rhae) if a > b)
    losses = sum(1 for a, b in zip(int_rhae, rnd_rhae) if a < b)
    ties   = N_DRAWS - wins - losses
    p_val  = binomial_p_one_sided(wins, wins + losses)

    print(f"\nPaired sign test (INTERVENTION vs RANDOM):")
    print(f"  Wins={wins}  Losses={losses}  Ties={ties}  p={p_val:.6f}")

    if wins > losses and p_val <= 0.10:
        verdict = 'SIGNAL'
    elif wins < losses:
        verdict = 'INTERVENTION_WORSE'
    else:
        verdict = 'KILL'

    summary['paired_wins']   = wins
    summary['paired_losses'] = losses
    summary['paired_ties']   = ties
    summary['p_value']       = p_val
    summary['verdict']       = verdict

    fn = os.path.join(RESULTS_DIR, 'summary.json')
    with open(fn, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {fn}")

    print(f"\nVERDICT: {verdict}")
    for cond_name in CONDITIONS:
        c = summary['conditions'][cond_name]
        print(f"  {cond_name}  chain_mean: {c['chain_mean']:.6e}")


if __name__ == '__main__':
    main()
