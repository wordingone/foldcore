"""
sub1080_defense_v20.py — Defense v20: goal-contrast solve for Mode 2 games (ℓ₁).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1080 --substrate experiments/sub1080_defense_v20.py

FAMILY: goal-contrast
R3 HYPOTHESIS: Mode 2 games (responsive but unsolved) fail because the cascade
  evolves sequences by CHANGE MAGNITUDE, not PROGRESS DIRECTION. A pixel that
  changes a lot might be oscillating, not progressing. Goal-contrast: track the
  observation mode (most frequent pixel value) as a soft goal, and score actions
  by whether they move observations TOWARD that goal. The mode IS the solved
  state — Level 1 = return to the screen's "natural" configuration.

  Step 1079 showed Mode 2 games have real signal (var=0.37, actions differentiate)
  but 0% L1. The substrate sees change but can't convert to solutions.

  SPSA on block-reduced features (64D, ℓ₁). No encoding modification.

KILL: No improvement on Mode 2 games (responsive games still at 0%)
SUCCESS: Any Mode 2 game breaks 0%
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS

# Phase boundaries
KB_PROBE_END = 800
CLICK_PROBE_END = 1400
SEQ_PROBE_END = 2400

# MI detection (read-only — no attention modification)
MI_THRESH = 0.05
MI_EMA = 0.95
MI_EPSILON = 1e-8
SUSTAIN_STEPS = 10

# Goal-contrast parameters
GOAL_UPDATE_INTERVAL = 50   # recompute goal mode every N steps
GOAL_MOMENTUM = 0.9         # EMA for goal stability
PROGRESS_WINDOW = 20        # compare obs over this window for direction

POP_SIZE = 12
SEQ_MIN = 3
SEQ_MAX = 15
MUTATE_EVERY = 10

N_KB = 7
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def _click_action(x, y):
    return N_KB + y * 64 + x

def _decode_click(action):
    if action < N_KB:
        return None
    idx = action - N_KB
    return (idx % 64, idx // 64)

def _obs_to_blocks(obs):
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


class DefenseV20Substrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        # MI stats (read-only)
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._prev_blocks = None

        # Goal-contrast state
        self._goal_obs = None          # soft goal: mode of observations (64x64)
        self._goal_blocks = None       # block-reduced goal
        self._obs_history = []         # recent observations for progress tracking
        self._progress_scores = {}     # per-action progress toward goal

        # Observation tracking
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)

        # Frequency tables for goal
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self._raw_goal = None

        # KB influence
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None

        self._detected_type = None
        self._best_click_regions = []
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._archive = []
        self._archive_max = 20
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def _init_mi_stats(self, n_actions):
        self._mi_mu = np.zeros((n_actions, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((n_actions, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(n_actions, dtype=np.float32)

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()
        self._init_mi_stats(n_actions)

    def _update_mi(self, action, delta_blocks):
        if self._mi_mu is None:
            self._init_mi_stats(self._n_actions)
        if action >= len(self._mi_mu):
            return
        a = action
        self._mi_count[a] += 1
        alpha = 1.0 - MI_EMA
        self._mi_mu[a] = MI_EMA * self._mi_mu[a] + alpha * delta_blocks
        residual = delta_blocks - self._mi_mu[a]
        self._mi_var[a] = MI_EMA * self._mi_var[a] + alpha * (residual ** 2)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta_blocks ** 2)

    def _compute_mi(self):
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _mi_action_score(self, action):
        if self._mi_mu is None or action >= len(self._mi_mu):
            return 0.0
        return float(np.sum(self._mi_values * np.abs(self._mi_mu[action])))

    # ── GOAL-CONTRAST ──

    def _update_goal(self, arr):
        """Update soft goal as the pixel-wise mode of all observations."""
        goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        if self._goal_obs is None:
            self._goal_obs = goal
        else:
            self._goal_obs = GOAL_MOMENTUM * self._goal_obs + (1 - GOAL_MOMENTUM) * goal
        self._goal_blocks = _obs_to_blocks(self._goal_obs)

    def _goal_distance(self, arr):
        """Distance from current observation to goal in block space."""
        if self._goal_blocks is None:
            return 0.0
        blocks = _obs_to_blocks(arr)
        return float(np.sum(np.abs(blocks - self._goal_blocks)))

    def _progress_fitness(self, obs_start, obs_end):
        """Fitness = reduction in distance to goal (progress toward goal).
        Positive = moved closer to goal. Negative = moved away."""
        if self._goal_blocks is None:
            # Fallback to change magnitude
            return float(np.sum(np.abs(obs_end - obs_start)))
        blocks_start = _obs_to_blocks(obs_start)
        blocks_end = _obs_to_blocks(obs_end)
        dist_start = float(np.sum(np.abs(blocks_start - self._goal_blocks)))
        dist_end = float(np.sum(np.abs(blocks_end - self._goal_blocks)))
        progress = dist_start - dist_end  # positive = got closer
        # Mix progress with change detection (need some change signal)
        change = float(np.sum(np.abs(blocks_end - blocks_start)))
        return progress + 0.1 * change

    def _track_action_progress(self, action, arr):
        """Track per-action progress toward goal."""
        if self._goal_blocks is None:
            return
        dist = self._goal_distance(arr)
        if action not in self._progress_scores:
            self._progress_scores[action] = {'total': 0.0, 'count': 0, 'prev_dist': dist}
        ps = self._progress_scores[action]
        if ps['count'] > 0:
            progress = ps['prev_dist'] - dist  # positive = action moved closer
            ps['total'] += progress
        ps['count'] += 1
        ps['prev_dist'] = dist

    # ── EVOLUTION (with goal-contrast fitness) ──

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        for _ in range(length):
            if self._supports_click and self._rng.random() < 0.7:
                if self._best_click_regions and self._rng.random() < 0.5:
                    cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
                    seq.append(_click_action(cx, cy))
                else:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    seq.append(_click_action(cx, cy))
            else:
                # Bias toward actions with positive progress
                if self._progress_scores and self._rng.random() < 0.4:
                    best_actions = sorted(self._progress_scores.keys(),
                                         key=lambda a: self._progress_scores[a]['total'] / max(1, self._progress_scores[a]['count']),
                                         reverse=True)
                    top = best_actions[:max(1, len(best_actions)//3)]
                    seq.append(top[self._rng.randint(len(top))])
                else:
                    seq.append(self._rng.randint(N_KB))
        return seq

    def _mutate_sequence(self, seq):
        seq = list(seq)
        mut = self._rng.randint(4)
        if mut == 0 and len(seq) > SEQ_MIN:
            seq.pop(self._rng.randint(len(seq)))
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
        if self._archive:
            n_from = min(POP_SIZE // 2, len(self._archive))
            archive_sorted = sorted(self._archive, key=lambda x: -x[0])
            pop = [self._mutate_sequence(archive_sorted[i][1]) for i in range(n_from)]
            pop += [self._random_sequence() for _ in range(POP_SIZE - n_from)]
            self._evo_pop = pop
        else:
            self._evo_pop = [self._random_sequence() for _ in range(POP_SIZE)]
        self._evo_scores = [0.0] * POP_SIZE
        self._evo_counts = [0] * POP_SIZE
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_total_evals = 0
        self._evo_initialized = True

    def _do_kb_bootloader(self, arr):
        goal = self._goal_obs if self._goal_obs is not None else (self._raw_goal if self._raw_goal is not None else arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.kb_influence[k] * mismatch) + 0.5 * self._mi_action_score(k)
            # Add progress bias
            if k in self._progress_scores and self._progress_scores[k]['count'] > 3:
                kb_scores[k] += self._progress_scores[k]['total'] / self._progress_scores[k]['count']
        action = int(np.argmax(kb_scores))
        if self._rng.random() < 0.1:
            action = self._rng.randint(N_KB)
        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _do_click_exploit(self, arr):
        goal = self._goal_obs if self._goal_obs is not None else (self._raw_goal if self._raw_goal is not None else arr)
        mismatch = np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        smoothed = uniform_filter(mismatch, size=KERNEL)
        if self._rng.random() < 0.1:
            if self._best_click_regions:
                cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
            else:
                cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
            return _click_action(cx, cy)
        idx = np.argmax(smoothed)
        y, x = np.unravel_index(idx, (64, 64))
        action = _click_action(int(x), int(y))
        y0, y1 = max(0, y - SUPPRESS_RADIUS), min(64, y + SUPPRESS_RADIUS + 1)
        x0, x1 = max(0, x - SUPPRESS_RADIUS), min(64, x + SUPPRESS_RADIUS + 1)
        self.suppress[y0:y1, x0:x1] = SUPPRESS_DURATION
        return action

    def _do_evolution(self, arr):
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
            # Use PROGRESS fitness instead of change magnitude
            score = self._progress_fitness(self._evo_obs_start, arr)
            idx = self._evo_current
            self._evo_counts[idx] += 1
            a = 0.3 if self._evo_counts[idx] > 1 else 1.0
            self._evo_scores[idx] = (1 - a) * self._evo_scores[idx] + a * score
            self._evo_total_evals += 1
            if score > 0:
                self._archive.append((score, list(seq)))
                if len(self._archive) > self._archive_max:
                    self._archive.sort(key=lambda x: -x[0])
                    self._archive = self._archive[:self._archive_max]
            if self._evo_total_evals % MUTATE_EVERY == 0 and self._evo_total_evals > POP_SIZE:
                worst = int(np.argmin(self._evo_scores))
                best = int(np.argmax(self._evo_scores))
                if worst != best:
                    self._evo_pop[worst] = self._mutate_sequence(self._evo_pop[best])
                    self._evo_scores[worst] = self._evo_scores[best] * 0.5
                    self._evo_counts[worst] = 0
            self._evo_current = (self._evo_current + 1) % POP_SIZE
            self._evo_exec_idx = 0
        return action

    def _do_exploit_sequences(self, arr):
        if not self._top_sequences:
            if self._archive:
                self._archive.sort(key=lambda x: -x[0])
                self._top_sequences = [s for _, s in self._archive[:5]]
            if not self._top_sequences:
                self._top_sequences = [self._random_sequence()]
        if self._rng.random() < 0.1:
            return self._rng.randint(self._n_actions)
        seq = self._top_sequences[self._exploit_current]
        action = seq[self._exploit_exec_idx]
        if action >= self._n_actions:
            action = self._rng.randint(self._n_actions)
        self._exploit_exec_idx += 1
        if self._exploit_exec_idx >= len(seq):
            self._exploit_exec_idx = 0
            self._exploit_current = (self._exploit_current + 1) % len(self._top_sequences)
        return action

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
        blocks = _obs_to_blocks(arr)

        # MI tracking (read-only)
        if self._prev_blocks is not None and self._prev_action is not None:
            delta_blocks = blocks - self._prev_blocks
            self._update_mi(self._prev_action, delta_blocks)
            # Track action progress toward goal
            self._track_action_progress(self._prev_action, arr)
            if self.step_count % 100 == 0:
                self._compute_mi()

        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._prev_blocks = blocks.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            action = 0
            self.prev_action_type = 'kb'
            self.prev_kb_idx = 0
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        diff = np.abs(arr - self.prev_obs)
        self.change_map = ALPHA_CHANGE * self.change_map + (1 - ALPHA_CHANGE) * diff
        if self.prev_action_type == 'kb' and self.prev_kb_idx is not None:
            self.kb_influence[self.prev_kb_idx] = (0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self.prev_obs = arr.copy()
        self._prev_blocks = blocks.copy()

        # Update goal-contrast target periodically
        if self.step_count % GOAL_UPDATE_INTERVAL == 0:
            self._update_goal(arr)

        # ── Standard probe cascade ──
        if self._detected_type is not None:
            action = self._exploit(arr)
        elif self.step_count < KB_PROBE_END:
            action = self._rng.randint(N_KB)
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        elif self.step_count == KB_PROBE_END:
            self._compute_mi()
            if float(self._mi_values.max()) > MI_THRESH:
                self._detected_type = 'kb'
            action = self._do_kb_bootloader(arr) if self._detected_type == 'kb' else self._rng.randint(self._n_actions)
        elif self.step_count < CLICK_PROBE_END:
            if self._supports_click:
                click_phase = (self.step_count - KB_PROBE_END) // SUSTAIN_STEPS
                grid_idx = click_phase % len(CLICK_GRID)
                cx, cy = CLICK_GRID[grid_idx]
                action = _click_action(cx, cy)
                self.prev_action_type = 'click'
            else:
                self._detected_type = 'kb'
                action = self._do_kb_bootloader(arr)
        elif self.step_count == CLICK_PROBE_END:
            if self._detected_type is None:
                self._compute_mi()
                if float(self._mi_values.max()) > MI_THRESH:
                    self._detected_type = 'click'
                    high_mi = np.argwhere(self._mi_values > np.percentile(self._mi_values, 75))
                    for bi in high_mi:
                        by, bx = bi[0] // N_BLOCKS, bi[0] % N_BLOCKS
                        self._best_click_regions.append((bx * BLOCK_SIZE + 4, by * BLOCK_SIZE + 4))
                else:
                    self._detected_type = 'unknown'
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
        elif self.step_count < SEQ_PROBE_END:
            action = self._do_evolution(arr)
        elif self.step_count == SEQ_PROBE_END:
            if self._detected_type is None or self._detected_type == 'unknown':
                if self._archive and max(s for s, _ in self._archive) > 0:
                    self._detected_type = 'seq'
                else:
                    self._detected_type = 'unknown'
            action = self._do_evolution(arr)
        else:
            action = self._exploit(arr)

        if action < N_KB:
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        else:
            self.prev_action_type = 'click'
            self.prev_kb_idx = None
        self._prev_obs_arr = arr.copy()
        self._prev_action = action
        return action

    def _exploit(self, arr):
        if self._detected_type == 'kb':
            return self._do_kb_bootloader(arr)
        elif self._detected_type == 'click':
            return self._do_click_exploit(arr)
        elif self._detected_type == 'seq':
            if self.step_count < SEQ_PROBE_END + 3000:
                return self._do_evolution(arr)
            else:
                return self._do_exploit_sequences(arr)
        else:
            if self.step_count % 3 == 0:
                return self._do_kb_bootloader(arr)
            elif self._supports_click and self.step_count % 3 == 1:
                return self._do_click_exploit(arr)
            else:
                return self._do_evolution(arr)

    def on_level_transition(self):
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.raw_freq[:] = 0
        self.prev_obs = None
        self._prev_blocks = None
        self._raw_goal = None
        self._goal_obs = None
        self._goal_blocks = None
        self._obs_history = []
        self._progress_scores = {}
        self._detected_type = None
        self._best_click_regions = []
        self._evo_pop = []
        self._evo_scores = []
        self._evo_counts = []
        self._evo_current = 0
        self._evo_exec_idx = 0
        self._evo_obs_start = None
        self._evo_total_evals = 0
        self._evo_initialized = False
        self._archive = []
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        self._init_mi_stats(self._n_actions)
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)


CONFIG = {
    "goal_update_interval": GOAL_UPDATE_INTERVAL,
    "goal_momentum": GOAL_MOMENTUM,
    "mi_thresh": MI_THRESH,
    "v20_features": "goal-contrast fitness (progress toward mode) + MI detection (read-only) + progress-biased evolution (l1)",
}

SUBSTRATE_CLASS = DefenseV20Substrate
