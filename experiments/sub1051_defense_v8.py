"""
sub1051_defense_v8.py — Defense v8: evolutionary sequence search (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1051 --substrate experiments/sub1051_defense_v8.py

FAMILY: parametric-goal
R3 HYPOTHESIS: ℓ₁ evolutionary sequence discovery with SPSA-tuned goal fitness.
  Fair counter to prosecution v7. Same evolutionary mechanism, different R3.

  (1+1)-ES evolves a population of action sequences.
  Each sequence is executed completely, then scored by goal-mismatch reduction.
  SPSA tunes the goal parameters (ℓ₁ R3).

  Defense fitness: does this sequence move observation TOWARD the goal?
  As SPSA converges to the correct goal, evolution selects sequences that solve.

Phase 1 (0-500): Keyboard sweep + random clicks. Build change_map.
Phase 2 (500-8000): Evolutionary sequence search with SPSA-tuned fitness.
Phase 3 (8000+): Exploit top sequences + SPSA-guided fallback.

KILL: ALL games 0%
SUCCESS: L1 > 0 on any game
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210
WARMUP_END = 500
EVO_PHASE_END = 8000
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# SPSA R3 parameters
SIGMA_INIT = 1.0
ALPHA_INIT = 0.999
SPSA_DELTA_SIGMA = 1.0
SPSA_DELTA_ALPHA = 0.01
SPSA_LR = 0.02

# Evolutionary parameters
POP_SIZE = 10
SEQ_MIN = 3
SEQ_MAX = 7
MUTATE_EVERY = 10

# Action encoding
N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x


def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)


class DefenseV8Substrate:
    """
    ℓ₁ R3 v8: evolutionary sequence search + SPSA-tuned goal fitness.
    SPSA adapts goal parameters, evolution finds sequences that reach the goal.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        # Evolution state
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        # Exploit state
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_INIT
            self.alpha = ALPHA_INIT
            self.r3_updates = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.freq_hist = np.zeros((64, 64, 16), dtype=np.int32)
        self.novelty_map = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        for _ in range(length):
            if self._supports_click and self._rng.random() < 0.7:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                seq.append(_click_action(cx, cy))
            else:
                seq.append(self._rng.randint(N_KB))
        return seq

    def _mutate_sequence(self, seq):
        seq = list(seq)
        mut = self._rng.randint(4)
        if mut == 0 and len(seq) > SEQ_MIN:
            idx = self._rng.randint(len(seq))
            seq.pop(idx)
        elif mut == 1 and len(seq) < SEQ_MAX:
            idx = self._rng.randint(len(seq) + 1)
            if self._supports_click and self._rng.random() < 0.7:
                g = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                seq.insert(idx, _click_action(g[0], g[1]))
            else:
                seq.insert(idx, self._rng.randint(N_KB))
        elif mut == 2:
            idx = self._rng.randint(len(seq))
            if self._supports_click and self._rng.random() < 0.7:
                g = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                seq[idx] = _click_action(g[0], g[1])
            else:
                seq[idx] = self._rng.randint(N_KB)
        elif mut == 3:
            idx = self._rng.randint(len(seq))
            xy = _decode_click(seq[idx])
            if xy is not None:
                cx = max(0, min(63, xy[0] + self._rng.randint(-4, 5)))
                cy = max(0, min(63, xy[1] + self._rng.randint(-4, 5)))
                seq[idx] = _click_action(cx, cy)
        return seq

    def _init_population(self):
        self._evo_pop = [self._random_sequence() for _ in range(POP_SIZE)]
        self._evo_scores = [0.0] * POP_SIZE
        self._evo_counts = [0] * POP_SIZE
        self._evo_initialized = True

    def _goal(self, obs, sigma=None, alpha=None):
        if sigma is None: sigma = self.sigma
        if alpha is None: alpha = self.alpha
        smoothed = np.zeros((64, 64, 16), dtype=np.float32)
        s = max(0.3, sigma)
        for c_idx in range(16):
            smoothed[:, :, c_idx] = gaussian_filter(
                self.freq_hist[:, :, c_idx].astype(np.float32), sigma=s)
        freq_goal = np.argmax(smoothed, axis=2).astype(np.float32)
        if self.novelty_map is not None:
            freq_goal = alpha * freq_goal + (1.0 - alpha) * self.novelty_map
        return freq_goal

    def _fitness(self, obs_start, obs_end):
        """Goal-mismatch reduction fitness (ℓ₁). Positive = moved toward goal."""
        goal = self._goal(obs_end)
        mismatch_start = np.abs(obs_start - goal)
        mismatch_end = np.abs(obs_end - goal)
        reduction = mismatch_start - mismatch_end  # positive = improvement
        return float(np.sum(self.change_map * reduction))

    def _r3_spsa_update(self, obs_before, obs_after, cx, cy):
        """SPSA gradient on sigma and alpha."""
        r = 3
        y0, y1 = max(0, cy - r), min(64, cy + r + 1)
        x0, x1 = max(0, cx - r), min(64, cx + r + 1)
        local_change = float(np.abs(
            obs_after[y0:y1, x0:x1] - obs_before[y0:y1, x0:x1]).mean())
        zone_changed = local_change > CF_CHANGE_THRESH

        def _step(val, delta, lo, hi, kwarg):
            p_plus = np.clip(val + delta, lo, hi)
            p_minus = np.clip(val - delta, lo, hi)
            if p_plus == p_minus:
                return val
            g_p = self._goal(obs_before, **{kwarg: p_plus})
            g_m = self._goal(obs_before, **{kwarg: p_minus})
            mm_p = float(np.abs(obs_before[y0:y1, x0:x1] - g_p[y0:y1, x0:x1]).mean())
            mm_m = float(np.abs(obs_before[y0:y1, x0:x1] - g_m[y0:y1, x0:x1]).mean())
            s_p = mm_p if zone_changed else -mm_p
            s_m = mm_m if zone_changed else -mm_m
            grad = (s_p - s_m) / (p_plus - p_minus)
            return float(np.clip(val + SPSA_LR * grad, lo, hi))

        self.sigma = _step(self.sigma, SPSA_DELTA_SIGMA, 0.5, 16.0, 'sigma')
        self.alpha = _step(self.alpha, SPSA_DELTA_ALPHA, 0.9, 0.999, 'alpha')
        self.r3_updates += 1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))
        arr = obs
        obs_int = obs.astype(np.int32)
        self.step_count += 1
        self.suppress = np.maximum(0, self.suppress - 1)

        # SPSA update from clicks
        if self._prev_obs_arr is not None and self._prev_action is not None:
            click_xy = _decode_click(self._prev_action)
            if click_xy is not None:
                self._r3_spsa_update(self._prev_obs_arr, arr, click_xy[0], click_xy[1])

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self.novelty_map = arr.copy()
            r, c = np.arange(64)[:, None], np.arange(64)[None, :]
            self.freq_hist[r, c, obs_int] += 1
            action = 0
            self.prev_action_type = 'kb'
            self.prev_kb_idx = 0
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff

        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (
                0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)

        r_idx, c_idx = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq_hist[r_idx, c_idx, obs_int] += 1
        self.novelty_map = 0.999 * self.novelty_map + 0.001 * arr
        self.prev_obs = arr.copy()

        # ── Phase 1: warmup (0-500) ──
        if self.step_count < WARMUP_END:
            if self.step_count < KB_PHASE_END:
                kb = (self.step_count - 1) % N_KB
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            elif self._supports_click and self._rng.random() < 0.9:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 2: evolutionary sequence search (500-8000) ──
        if self.step_count < EVO_PHASE_END:
            if not self._evo_initialized:
                self._init_population()

            if self._evo_exec_idx == 0:
                self._evo_obs_start = arr.copy()

            seq = self._evo_pop[self._evo_current]
            action = seq[self._evo_exec_idx]
            if action >= self._n_actions:
                action = self._rng.randint(self._n_actions)
            self._evo_exec_idx += 1

            if self._evo_exec_idx >= len(seq):
                score = self._fitness(self._evo_obs_start, arr)
                idx = self._evo_current
                self._evo_counts[idx] += 1
                a = 0.3 if self._evo_counts[idx] > 1 else 1.0
                self._evo_scores[idx] = (1 - a) * self._evo_scores[idx] + a * score
                self._evo_total_evals += 1

                if self._evo_total_evals % MUTATE_EVERY == 0 and self._evo_total_evals > POP_SIZE:
                    worst = int(np.argmin(self._evo_scores))
                    best = int(np.argmax(self._evo_scores))
                    if worst != best:
                        self._evo_pop[worst] = self._mutate_sequence(self._evo_pop[best])
                        self._evo_scores[worst] = self._evo_scores[best] * 0.5
                        self._evo_counts[worst] = 0

                self._evo_current = (self._evo_current + 1) % POP_SIZE
                self._evo_exec_idx = 0

            if action < N_KB:
                self.prev_action_type = 'kb'
                self.prev_kb_idx = action
            else:
                self.prev_action_type = 'click'
                self.prev_kb_idx = None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # ── Phase 3: exploit top sequences + SPSA-guided fallback ──
        if not self._top_sequences and self._evo_pop:
            ranked = sorted(range(POP_SIZE), key=lambda i: -self._evo_scores[i])
            self._top_sequences = [self._evo_pop[i] for i in ranked[:5] if self._evo_scores[i] > 0]
            if not self._top_sequences:
                self._top_sequences = [self._evo_pop[ranked[0]]]

        if self._rng.random() < 0.1:
            if self._supports_click and self._rng.random() < 0.5:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
            else:
                kb = self._rng.randint(N_KB)
                action = kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = kb
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        if self._top_sequences and self._rng.random() < 0.7:
            seq = self._top_sequences[self._exploit_current]
            action = seq[self._exploit_exec_idx]
            if action >= self._n_actions:
                action = self._rng.randint(self._n_actions)
            self._exploit_exec_idx += 1
            if self._exploit_exec_idx >= len(seq):
                self._exploit_exec_idx = 0
                self._exploit_current = (self._exploit_current + 1) % len(self._top_sequences)
        else:
            goal = self._goal(arr)
            mismatch = np.abs(arr - goal) * self.change_map
            suppress_mask = (self.suppress == 0).astype(np.float32)
            mismatch *= suppress_mask
            smoothed = uniform_filter(mismatch, size=KERNEL)

            click_score = float(np.max(smoothed)) if self._supports_click else -1.0
            kb_scores = np.zeros(N_KB)
            for k in range(N_KB):
                kb_scores[k] = np.sum(self.kb_influence[k] * smoothed)
            best_kb = int(np.argmax(kb_scores))

            if self._supports_click and click_score >= kb_scores[best_kb]:
                idx = np.argmax(smoothed)
                y, x = np.unravel_index(idx, (64, 64))
                action = _click_action(int(x), int(y))
                y0, y1 = max(0, y - SUPPRESS_RADIUS), min(64, y + SUPPRESS_RADIUS + 1)
                x0, x1 = max(0, x - SUPPRESS_RADIUS), min(64, x + SUPPRESS_RADIUS + 1)
                self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
                self.prev_action_type = 'click'
            else:
                action = best_kb
                self.prev_action_type = 'kb'
                self.prev_kb_idx = best_kb

        if action < N_KB:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        else:
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
        self._prev_obs_arr = arr.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.freq_hist[:] = 0
        self.novelty_map = None
        self.prev_obs = None
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        # Keep SPSA params + kb_influence


CONFIG = {
    "warmup": WARMUP_END,
    "evo_phase": EVO_PHASE_END,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "sigma_init": SIGMA_INIT,
    "spsa_lr": SPSA_LR,
    "v8_features": "evolutionary sequence search + SPSA-tuned goal fitness (l_1)",
}

SUBSTRATE_CLASS = DefenseV8Substrate
