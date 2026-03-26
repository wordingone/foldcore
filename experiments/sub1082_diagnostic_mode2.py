"""
sub1082_diagnostic_mode2.py — Diagnostic: WHY can't responsive games be solved?

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1082 --substrate experiments/sub1082_diagnostic_mode2.py

FAMILY: diagnostic (not a substrate experiment)
R3 HYPOTHESIS: N/A — measurement only.

PURPOSE: For Mode 2 games (responsive, actions differentiate, but 0% L1), measure:
  1. How many distinct informative actions? (MI > threshold count)
  2. Does repeated best-action application progress the game?
  3. Does observation drift directionally or oscillate?
  4. Does the substrate ever get close to L1 without detecting it?

Output: per-game diagnostic table printed to log.

KILL: N/A — measurement only
SUCCESS: Clear diagnosis of why detection → solve fails
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS
N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
MI_EMA = 0.95
MI_EPSILON = 1e-8
MI_THRESH = 0.01  # lower threshold for counting informative actions


def _click_action(x, y):
    return N_KB + y * 64 + x

def _obs_to_blocks(obs):
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


class DiagnosticMode2Substrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self.prev_obs = None
        self._prev_action = None
        self._prev_blocks = None
        self._initial_obs = None
        self._initial_blocks = None

        # MI tracking
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)

        # Diagnostic measurements
        self._phase = 'explore'  # explore → repeat_best → measure_drift
        self._explore_steps = 500
        self._repeat_steps = 300
        self._best_action = 0
        self._distances_from_initial = []  # track distance from initial obs over time
        self._per_action_progress = {}  # action → list of (dist_before, dist_after)
        self._obs_mean_history = []  # track obs mean for drift detection
        self._repeat_distances = []  # distances during best-action repetition

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _init_mi_stats(self, n_actions):
        self._mi_mu = np.zeros((n_actions, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((n_actions, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(n_actions, dtype=np.float32)

    def set_game(self, n_actions: int):
        if self._initial_obs is not None and self.step_count > 100:
            self._print_diagnostic()
        self._game_number += 1
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()
        self._init_mi_stats(n_actions)

    def _update_mi(self, action, delta_blocks):
        if action >= len(self._mi_mu):
            return
        self._mi_count[action] += 1
        alpha = 1.0 - MI_EMA
        self._mi_mu[action] = MI_EMA * self._mi_mu[action] + alpha * delta_blocks
        residual = delta_blocks - self._mi_mu[action]
        self._mi_var[action] = MI_EMA * self._mi_var[action] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta_blocks ** 2)

    def _compute_mi(self):
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _dist_from_initial(self, blocks):
        if self._initial_blocks is None:
            return 0.0
        return float(np.sum(np.abs(blocks - self._initial_blocks)))

    def _print_diagnostic(self):
        print(f"\n{'='*70}", flush=True)
        print(f"MODE 2 DIAGNOSTIC — GAME_{self._game_number} (n_actions={self._n_actions})", flush=True)
        print(f"{'='*70}", flush=True)

        # 1. Informative actions count
        self._compute_mi()
        n_informative = 0
        action_mi_scores = []
        for a in range(min(self._n_actions, len(self._mi_mu))):
            if self._mi_count[a] > 5:
                score = float(np.sum(self._mi_values * np.abs(self._mi_mu[a])))
                action_mi_scores.append((a, score))
                if score > MI_THRESH:
                    n_informative += 1

        print(f"\n1. INFORMATIVE ACTIONS (MI > {MI_THRESH}):", flush=True)
        print(f"   Count: {n_informative}/{len(action_mi_scores)} tracked actions", flush=True)
        action_mi_scores.sort(key=lambda x: -x[1])
        for a, s in action_mi_scores[:10]:
            label = f"kb{a}" if a < N_KB else f"click({(a-N_KB)%64},{(a-N_KB)//64})"
            print(f"   {label:>20s}: MI_score={s:.6f} (n={int(self._mi_count[a])})", flush=True)

        # 2. Per-action progress toward initial state
        print(f"\n2. PER-ACTION PROGRESS (toward initial obs):", flush=True)
        for a in sorted(self._per_action_progress.keys()):
            pairs = self._per_action_progress[a]
            if len(pairs) < 3:
                continue
            progress_vals = [before - after for before, after in pairs]
            mean_progress = np.mean(progress_vals)
            label = f"kb{a}" if a < N_KB else f"click"
            direction = "TOWARD" if mean_progress > 0 else "AWAY"
            print(f"   {label}: mean_progress={mean_progress:+.4f} ({direction} initial, n={len(pairs)})", flush=True)

        # 3. Observation drift during best-action repetition
        print(f"\n3. BEST-ACTION REPETITION (action={self._best_action}, {len(self._repeat_distances)} steps):", flush=True)
        if len(self._repeat_distances) > 10:
            rd = np.array(self._repeat_distances)
            first_10 = rd[:10].mean()
            last_10 = rd[-10:].mean()
            monotonic_count = sum(1 for i in range(1, len(rd)) if rd[i] < rd[i-1])
            print(f"   Distance from initial — start: {first_10:.4f}, end: {last_10:.4f}", flush=True)
            print(f"   Trend: {'DECREASING (approaching)' if last_10 < first_10 else 'INCREASING (diverging)' if last_10 > first_10 else 'FLAT'}", flush=True)
            print(f"   Monotonic decreases: {monotonic_count}/{len(rd)-1} ({100*monotonic_count/max(1,len(rd)-1):.0f}%)", flush=True)

            # Check for oscillation
            if len(rd) > 20:
                diffs = np.diff(rd)
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                print(f"   Sign changes: {sign_changes}/{len(diffs)-1} ({'OSCILLATING' if sign_changes > len(diffs)*0.4 else 'DIRECTIONAL'})", flush=True)
        else:
            print(f"   Insufficient data", flush=True)

        # 4. Distance from initial over full trajectory
        print(f"\n4. DISTANCE FROM INITIAL OVER TIME:", flush=True)
        if self._distances_from_initial:
            dd = np.array(self._distances_from_initial)
            min_dist = dd.min()
            min_idx = int(dd.argmin())
            print(f"   Min distance: {min_dist:.4f} at step {min_idx}", flush=True)
            print(f"   Initial distance: {dd[0]:.4f}", flush=True)
            print(f"   Final distance: {dd[-1]:.4f}", flush=True)
            # Did we ever get close?
            close_threshold = dd[0] * 0.1  # within 10% of initial
            close_count = int(np.sum(dd < close_threshold))
            print(f"   Steps within 10% of initial: {close_count}/{len(dd)}", flush=True)

        # 5. Observation mean drift
        print(f"\n5. OBS MEAN DRIFT:", flush=True)
        if len(self._obs_mean_history) > 10:
            omh = np.array(self._obs_mean_history)
            print(f"   Start mean: {omh[0]:.4f}, End mean: {omh[-1]:.4f}", flush=True)
            print(f"   Range: [{omh.min():.4f}, {omh.max():.4f}]", flush=True)
            # Monotonic drift check
            if len(omh) > 50:
                first_half = omh[:len(omh)//2].mean()
                second_half = omh[len(omh)//2:].mean()
                print(f"   First half mean: {first_half:.4f}, Second half: {second_half:.4f}", flush=True)

        print(f"{'='*70}\n", flush=True)

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        self.step_count += 1
        blocks = _obs_to_blocks(arr)

        # Store initial observation
        if self._initial_obs is None:
            self._initial_obs = arr.copy()
            self._initial_blocks = blocks.copy()

        # MI tracking
        if self._prev_blocks is not None and self._prev_action is not None:
            delta_blocks = blocks - self._prev_blocks
            self._update_mi(self._prev_action, delta_blocks)

        # Track distance from initial
        dist = self._dist_from_initial(blocks)
        self._distances_from_initial.append(dist)
        self._obs_mean_history.append(float(arr.mean()))

        # Track per-action progress
        if self._prev_action is not None and self._prev_blocks is not None:
            dist_before = self._dist_from_initial(self._prev_blocks)
            dist_after = dist
            a = self._prev_action
            if a not in self._per_action_progress:
                self._per_action_progress[a] = []
            if len(self._per_action_progress[a]) < 200:
                self._per_action_progress[a].append((dist_before, dist_after))

        self.prev_obs = arr.copy()
        self._prev_blocks = blocks.copy()

        # ── Phase 1: EXPLORE (random actions for MI statistics) ──
        if self.step_count <= self._explore_steps:
            if self.step_count <= N_KB * 30:
                action = (self.step_count - 1) % N_KB
            elif self._supports_click and self.step_count <= N_KB * 30 + len(CLICK_GRID) * 5:
                idx = (self.step_count - N_KB * 30 - 1) % len(CLICK_GRID)
                cx, cy = CLICK_GRID[idx]
                action = _click_action(cx, cy)
            else:
                if self._supports_click and self._rng.random() < 0.7:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    action = _click_action(cx, cy)
                else:
                    action = self._rng.randint(N_KB)

            if self.step_count == self._explore_steps:
                self._compute_mi()
                # Find best action by MI score
                best_score = -1
                for a in range(min(self._n_actions, len(self._mi_mu))):
                    if self._mi_count[a] > 3:
                        score = float(np.sum(self._mi_values * np.abs(self._mi_mu[a])))
                        if score > best_score:
                            best_score = score
                            self._best_action = a
                self._phase = 'repeat_best'

        # ── Phase 2: REPEAT BEST (apply best action repeatedly) ──
        elif self.step_count <= self._explore_steps + self._repeat_steps:
            action = self._best_action
            self._repeat_distances.append(dist)
            if self.step_count == self._explore_steps + self._repeat_steps:
                self._phase = 'measure_drift'

        # ── Phase 3: MEASURE DRIFT (alternating actions for drift detection) ──
        else:
            # Cycle through all actions to measure their individual effects
            action = (self.step_count - self._explore_steps - self._repeat_steps) % N_KB

        self._prev_action = action
        return action

    def on_level_transition(self):
        if self._initial_obs is not None and self.step_count > 100:
            self._print_diagnostic()
        self._init_state()
        self._init_mi_stats(self._n_actions)


CONFIG = {
    "explore_steps": 500,
    "repeat_steps": 300,
    "v82_features": "mode-2 diagnostic: MI action count, best-action repetition, drift detection, distance-from-initial tracking",
}

SUBSTRATE_CLASS = DiagnosticMode2Substrate
