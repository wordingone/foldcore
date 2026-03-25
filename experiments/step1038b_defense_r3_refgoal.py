"""
Step 1038b -- Defense R3: Parametric Goal + Solved-State Reference (l_pi)

Extension of 1038. Three self-modifying parameters:
  sigma (spatial scale): [0.5, 16.0] - gaussian kernel width
  alpha (temporal weight): [0.9, 0.999] - history vs recency
  ref_weight (reference blend): [0.0, 0.8] - NEW: how much to trust previous
    level's solved state as goal for current level

KEY INSIGHT: After L1 transition, the pre-transition frame approximates the
SOLVED state. Storing this as a reference gives the substrate a SEED for L2's
goal that's not just "most common observation." SPSA on ref_weight discovers
whether this reference is useful per game.

This addresses the L2+ bottleneck: freq_mode predicts the INITIAL (unsolved)
state, not the SOLVED state. The reference provides the solved state.

Same bootloader, same SPSA mechanism as 1038. Only addition: solved_ref
and ref_weight parameter.

Kill: ref_weight stays at 0 (reference useless) AND no L2+ -> reference
  doesn't help, l_pi R3 still insufficient
Success: ref_weight > 0 for some games AND L2+ emerges

All 3 games, randomized order, 5 seeds, 10K steps/game.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# --- Config ---
WARMUP_STEPS = 3000
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
ZONE_RADIUS = 3
MAX_STEPS = 10_000
TIME_CAP = 120
CF_CHANGE_THRESH = 0.1

# --- R3 SPSA Config ---
SIGMA_INIT = 1.0
ALPHA_INIT = 0.999
REF_WEIGHT_INIT = 0.0  # start with no reference influence
SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_DELTA_RW = 0.05   # perturbation for ref_weight
SPSA_LR = 0.02

GAME_FILTER = ['ft09', 'ls20', 'vc33']  # preview games only

GAME_ACTIONS = list(GameAction)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
N_KB_ACTIONS = 7


class DefenseR3SubstrateV2:
    """l_pi R3 with solved-state reference."""

    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB_ACTIONS, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._click_xy = (32, 32)
        self._last_click_xy = None
        self._frame_before = None

        # R3 parameters (l_pi self-modifying)
        self.sigma = SIGMA_INIT
        self.alpha = ALPHA_INIT
        self.ref_weight = REF_WEIGHT_INIT

        # Solved state reference (set on level transition)
        self.solved_ref = None

        # Diagnostics
        self.r3_updates = 0
        self.levels_seen = 0

    def _goal(self, obs, sigma=None, alpha=None, ref_weight=None):
        if sigma is None: sigma = self.sigma
        if alpha is None: alpha = self.alpha
        if ref_weight is None: ref_weight = self.ref_weight

        s = max(0.3, sigma)
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        for c in range(16):
            smoothed[:, :, c] = gaussian_filter(
                self.freq_hist[:, :, c].astype(np.float32), sigma=s)
        spatial_goal = np.argmax(smoothed, axis=2).astype(np.float32)

        if self.novelty_map is not None:
            temporal_goal = alpha * spatial_goal + (1.0 - alpha) * self.novelty_map
        else:
            temporal_goal = spatial_goal

        # Blend with solved reference if available
        if self.solved_ref is not None and ref_weight > 0.01:
            return (1.0 - ref_weight) * temporal_goal + ref_weight * self.solved_ref
        return temporal_goal

    def _r3_spsa_update(self, frame_before, frame_after, cx, cy):
        obs = np.array(frame_before[0], dtype=np.float32)
        obs_after = np.array(frame_after[0], dtype=np.float32)

        r = ZONE_RADIUS
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)

        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        # --- SPSA for sigma ---
        s_plus = np.clip(self.sigma + SPSA_DELTA_SIGMA, 0.5, 16.0)
        s_minus = np.clip(self.sigma - SPSA_DELTA_SIGMA, 0.5, 16.0)
        if s_plus != s_minus:
            goal_sp = self._goal(obs, sigma=s_plus)
            goal_sm = self._goal(obs, sigma=s_minus)
            mm_sp = float(np.abs(obs[y0:y1, x0:x1] - goal_sp[y0:y1, x0:x1]).mean())
            mm_sm = float(np.abs(obs[y0:y1, x0:x1] - goal_sm[y0:y1, x0:x1]).mean())
            if zone_changed:
                score_sp, score_sm = mm_sp, mm_sm
            else:
                score_sp, score_sm = -mm_sp, -mm_sm
            grad_s = (score_sp - score_sm) / (s_plus - s_minus)
            self.sigma = float(np.clip(self.sigma + SPSA_LR * grad_s, 0.5, 16.0))

        # --- SPSA for alpha ---
        a_plus = np.clip(self.alpha + SPSA_DELTA_ALPHA, 0.9, 0.999)
        a_minus = np.clip(self.alpha - SPSA_DELTA_ALPHA, 0.9, 0.999)
        if a_plus != a_minus:
            goal_ap = self._goal(obs, alpha=a_plus)
            goal_am = self._goal(obs, alpha=a_minus)
            mm_ap = float(np.abs(obs[y0:y1, x0:x1] - goal_ap[y0:y1, x0:x1]).mean())
            mm_am = float(np.abs(obs[y0:y1, x0:x1] - goal_am[y0:y1, x0:x1]).mean())
            if zone_changed:
                score_ap, score_am = mm_ap, mm_am
            else:
                score_ap, score_am = -mm_ap, -mm_am
            grad_a = (score_ap - score_am) / (a_plus - a_minus)
            self.alpha = float(np.clip(self.alpha + SPSA_LR * grad_a, 0.9, 0.999))

        # --- SPSA for ref_weight (only when reference exists) ---
        if self.solved_ref is not None:
            rw_plus = np.clip(self.ref_weight + SPSA_DELTA_RW, 0.0, 0.8)
            rw_minus = np.clip(self.ref_weight - SPSA_DELTA_RW, 0.0, 0.8)
            if rw_plus != rw_minus:
                goal_rp = self._goal(obs, ref_weight=rw_plus)
                goal_rm = self._goal(obs, ref_weight=rw_minus)
                mm_rp = float(np.abs(
                    obs[y0:y1, x0:x1] - goal_rp[y0:y1, x0:x1]).mean())
                mm_rm = float(np.abs(
                    obs[y0:y1, x0:x1] - goal_rm[y0:y1, x0:x1]).mean())
                if zone_changed:
                    score_rp, score_rm = mm_rp, mm_rm
                else:
                    score_rp, score_rm = -mm_rp, -mm_rm
                grad_rw = (score_rp - score_rm) / (rw_plus - rw_minus)
                self.ref_weight = float(np.clip(
                    self.ref_weight + SPSA_LR * grad_rw, 0.0, 0.8))

        self.r3_updates += 1

    def step(self, frame):
        arr = np.array(frame[0], dtype=np.float32)
        obs_int = np.array(frame[0], dtype=np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        if self._frame_before is not None and self._last_click_xy is not None:
            cx, cy = self._last_click_xy
            self._r3_spsa_update(self._frame_before, frame, cx, cy)

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            self._click_xy = (32, 32)
            self._frame_before = frame
            self._last_click_xy = (32, 32)
            return 'click'

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        r_idx, c_idx = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r_idx, c_idx, obs_int] += 1
        alpha_nov = 0.999
        self.novelty_map = alpha_nov * self.novelty_map + (1 - alpha_nov) * arr

        goal = self._goal(arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)

        self.prev_obs = arr.copy()

        # Warmup: random
        if self.step_count < WARMUP_STEPS:
            if self.rng.random() < 0.9:
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self._frame_before = frame
                self._last_click_xy = (cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                self._frame_before = None
                self._last_click_xy = None
                return ('kb', kb)

        # Post-warmup: score-based
        click_score = float(np.max(smoothed))
        kb_scores = np.zeros(N_KB_ACTIONS)
        for k in range(N_KB_ACTIONS):
            kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
        best_kb = int(np.argmax(kb_scores))
        best_kb_score = kb_scores[best_kb]

        if self.rng.random() < 0.1:
            if self.rng.random() < 0.5:
                cx, cy = CLICK_GRID[self.rng.randint(64)]
                self._click_xy = (cx, cy)
                self._frame_before = frame
                self._last_click_xy = (cx, cy)
                self.prev_action_type = 'click'
                return 'click'
            else:
                kb = self.rng.randint(N_KB_ACTIONS)
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
                self._frame_before = None
                self._last_click_xy = None
                return ('kb', kb)

        if click_score >= best_kb_score:
            idx = np.argmax(smoothed)
            y, x = np.unravel_index(idx, (64, 64))
            self._click_xy = (int(x), int(y))
            y0 = max(0, y - SUPPRESS_RADIUS)
            y1 = min(64, y + SUPPRESS_RADIUS + 1)
            x0 = max(0, x - SUPPRESS_RADIUS)
            x1 = min(64, x + SUPPRESS_RADIUS + 1)
            self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
            self._frame_before = frame
            self._last_click_xy = (int(x), int(y))
            self.prev_action_type = 'click'
            return 'click'
        else:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = best_kb
            self._frame_before = None
            self._last_click_xy = None
            return ('kb', best_kb)

    def on_level_transition(self, post_frame=None):
        self.levels_seen += 1
        # Store pre-transition state as solved reference
        if self.prev_obs is not None:
            self.solved_ref = self.prev_obs.copy()
        self.freq_hist[:] = 0
        self.suppress[:] = 0
        # R3 parameters PERSIST (sigma, alpha, ref_weight carry forward)
        # ref_weight starts getting SPSA signal now that solved_ref exists


def run_game(arc, game_id, seed, substrate):
    np.random.seed(seed)
    env = arc.make(game_id)
    obs = env.reset()
    t0 = time.time()
    max_levels = 0
    level_steps = {}
    total_steps = 0

    for step_i in range(MAX_STEPS):
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        result = substrate.step(obs.frame)
        total_steps += 1
        levels_before = obs.levels_completed

        if result == 'click':
            cx, cy = substrate._click_xy
            obs = env.step(GAME_ACTIONS[6], data={"x": cx, "y": cy})
        else:
            _, kb_idx = result
            obs = env.step(GAME_ACTIONS[kb_idx + 1], data={})

        if obs and obs.levels_completed > levels_before:
            if obs.levels_completed > max_levels:
                max_levels = obs.levels_completed
                level_steps[max_levels] = step_i + 1
            substrate.on_level_transition()

    elapsed = time.time() - t0
    return {
        'max_levels': max_levels,
        'level_steps': level_steps,
        'steps': total_steps,
        'elapsed': round(elapsed, 1),
        'sigma': round(substrate.sigma, 3),
        'alpha': round(substrate.alpha, 4),
        'ref_weight': round(substrate.ref_weight, 3),
        'has_ref': substrate.solved_ref is not None,
        'r3_updates': substrate.r3_updates,
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    games = {}
    for e in envs:
        gname = e.game_id.split('-')[0]
        if gname in GAME_FILTER and gname not in games:
            games[gname] = e.game_id
    game_names = sorted(games.keys())

    print("=== Step 1038b: Defense R3 -- Parametric Goal + Solved Reference (l_pi) ===")
    print("R3: SPSA on sigma, alpha, ref_weight (3 params)")
    print(f"Config: warmup={WARMUP_STEPS}, spsa_lr={SPSA_LR}")
    print(f"Init: sigma={SIGMA_INIT}, alpha={ALPHA_INIT}, ref_weight={REF_WEIGHT_INIT}")
    print(f"Games: {', '.join(f'{g} ({games[g]})' for g in game_names)}")
    print()

    all_results = {g: [] for g in game_names}

    for seed in range(5):
        rng = np.random.RandomState(seed)
        order = list(game_names)
        rng.shuffle(order)
        print(f"--- Seed {seed} (order: {' > '.join(order)}) ---")
        substrate = DefenseR3SubstrateV2(seed)

        for gname in order:
            r = run_game(arc, games[gname], seed, substrate)
            lvl_str = ""
            if r['level_steps']:
                lvl_str = "  " + "  ".join(
                    f"L{l}@{s}" for l, s in sorted(r['level_steps'].items()))
            ref_str = f"rw={r['ref_weight']:.2f}" if r['has_ref'] else "rw=N/A"
            print(f"  {gname}: L={r['max_levels']}  steps={r['steps']}  "
                  f"s={r['sigma']:.2f}  a={r['alpha']:.4f}  {ref_str}  "
                  f"{r['elapsed']}s{lvl_str}")
            all_results[gname].append(r)

    print("\n=== Summary ===")
    any_l2 = False
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        counts = {}
        for lvl in range(1, 6):
            c = sum(1 for r in results if r['max_levels'] >= lvl)
            if c > 0: counts[lvl] = c
        max_l = max(r['max_levels'] for r in results)
        parts = [f"L{l}: {c}/5" for l, c in sorted(counts.items())]
        print(f"  {gname}: {', '.join(parts) if parts else 'L0 only'}  (max={max_l})")
        if max_l >= 2: any_l2 = True

    print("\n=== R3 Diagnostic (l_pi: 3 parameters) ===")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        avg_s = np.mean([r['sigma'] for r in results])
        avg_a = np.mean([r['alpha'] for r in results])
        rw_vals = [r['ref_weight'] for r in results if r['has_ref']]
        avg_rw = np.mean(rw_vals) if rw_vals else 0.0
        n_ref = sum(1 for r in results if r['has_ref'])
        print(f"  {gname}: sigma={avg_s:.3f}  alpha={avg_a:.4f}  "
              f"ref_weight={avg_rw:.3f} ({n_ref}/5 have ref)")

    # Differentiation
    all_sigma = {g: np.mean([r['sigma'] for r in all_results[g]]) for g in game_names}
    max_sigma_diff = max(abs(all_sigma[g1] - all_sigma[g2])
                         for g1 in game_names for g2 in game_names if g1 != g2)

    if any_l2:
        print("\n  SIGNAL: l_pi R3 with solved reference produces L2+!")
    else:
        print("\n  FAIL: No L2+.")

    print(f"  sigma max_diff: {max_sigma_diff:.3f}")

    print("\n=== Comparison ===")
    print("  1037b (prosecution l_1): ft09 1/5 L1, vc33 0/5, ls20 0/5. No L2+.")
    print("  1038  (defense l_pi):    ft09 1/5 L1, vc33 0/5, ls20 0/5. No L2+.")
    for gname in sorted(all_results.keys()):
        results = all_results[gname]
        l1 = sum(1 for r in results if r['max_levels'] >= 1)
        l2 = sum(1 for r in results if r['max_levels'] >= 2)
        print(f"  1038b (defense l_pi+ref): {gname} {l1}/5 L1, {l2}/5 L2")

    print("\nStep 1038b DONE")


if __name__ == "__main__":
    main()
