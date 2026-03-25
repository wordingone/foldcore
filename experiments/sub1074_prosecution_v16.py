"""
sub1074_prosecution_v16.py — Prosecution v16: MI-based action informativeness (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1074 --substrate experiments/sub1074_prosecution_v16.py

FAMILY: attention-gated
R3 HYPOTHESIS: Magnitude-based detection (z-score, Mahalanobis) misses games where
  actions cause CONSISTENT but sub-threshold changes. Mutual information between
  actions and per-dim observation deltas captures reliability, not magnitude.
  An action shifting a pixel by 0.001 every time has high MI even at 0 z-score.

  MI_d = 0.5 * log(var_total[d] / mean_a(var_a[d])) — Gaussian MI approximation.
  Attention concentrates on dims with highest MI (ℓ_π).
  Action selection maximizes sum_d(MI_d * |mu_a[d]|).

  Ref: IDA (Mazzaglia & Cohen, 2024) — information-driven affordance discovery.
  Our version: R1-compliant (running EMA statistics only, no SGD).

KILL: No improvement over v15 on non-opaque games AND no signal on opaque games
SUCCESS: Signal on currently-opaque games (0% wall breached)
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8

# Block reduction for MI: 8×8 blocks → 64 dims
BLOCK_SIZE = 8
N_BLOCKS = 8
N_DIMS = N_BLOCKS * N_BLOCKS  # 64

# Probe boundaries
MI_WARMUP = 300          # longer warmup for MI statistics
KB_PROBE_END = 800
CLICK_PROBE_END = 1400
SEQ_PROBE_END = 2400

# MI detection
MI_THRESH = 0.05         # bits — threshold for "informative" dimension
MI_EMA = 0.95            # EMA decay for MI statistics
MI_EPSILON = 1e-8        # floor for variance ratio

# Sustained probe
SUSTAIN_STEPS = 15       # longer holds for consistent effect detection

# Attention R3 parameters
ATT_INIT = 0.5
ATT_LR = 0.03
ATT_MIN = 0.01
ATT_MAX = 1.0

# Evolutionary parameters
POP_SIZE = 12
SEQ_MIN = 3
SEQ_MAX = 15
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


def _obs_to_blocks(obs):
    """Reduce 64x64 observation to 64-dim block means."""
    blocks = np.zeros(N_DIMS, dtype=np.float32)
    for by in range(N_BLOCKS):
        for bx in range(N_BLOCKS):
            y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
            x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
            blocks[by * N_BLOCKS + bx] = obs[y0:y1, x0:x1].mean()
    return blocks


class ProsecutionV16Substrate:
    """
    ℓ_π R3 v16: MI-based action informativeness + attention encoding.
    Tracks per-action, per-dim delta statistics (EMA). Computes Gaussian MI
    approximation. Attention concentrates on high-MI dims (ℓ_π).
    Action selection maximizes MI-weighted expected effect.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        # MI statistics (block-reduced, per action)
        # mu_a[a, d] = EMA mean of delta after action a on dim d
        self._mi_mu = None       # (n_actions, N_DIMS)
        self._mi_var = None      # (n_actions, N_DIMS) per-action variance
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)  # total variance
        self._mi_count = None    # (n_actions,) count per action
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)  # computed MI per dim
        self._mi_per_action = None  # (n_actions, N_DIMS) MI contribution per action

        # Previous block observation
        self._prev_blocks = None

        # Change detection
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)

        # Attention (ℓ_π encoding) — per block, driven by MI
        self.block_attention = np.full(N_DIMS, ATT_INIT, dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)

        # Frequency tracking
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self._raw_goal = None
        self._gated_goal = None

        # KB influence
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None

        # Cascade detection
        self._detected_type = None
        self._kb_mi_signal = 0.0
        self._click_mi_signal = 0.0
        self._best_click_regions = []

        # Evolution
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
        """Initialize MI tracking arrays for current action space."""
        self._mi_mu = np.zeros((n_actions, N_DIMS), dtype=np.float32)
        self._mi_var = np.full((n_actions, N_DIMS), 1e-4, dtype=np.float32)
        self._mi_count = np.zeros(n_actions, dtype=np.float32)
        self._mi_per_action = np.zeros((n_actions, N_DIMS), dtype=np.float32)

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self._init_state()
        self._init_mi_stats(n_actions)

    def _update_mi(self, action, delta_blocks):
        """Update MI statistics for action-dim pair."""
        if self._mi_mu is None:
            self._init_mi_stats(self._n_actions)
        if action >= len(self._mi_mu):
            return

        # EMA update for per-action mean and variance
        a = action
        self._mi_count[a] += 1
        alpha = 1.0 - MI_EMA
        self._mi_mu[a] = MI_EMA * self._mi_mu[a] + alpha * delta_blocks
        residual = delta_blocks - self._mi_mu[a]
        self._mi_var[a] = MI_EMA * self._mi_var[a] + alpha * (residual ** 2)

        # EMA update for total variance (across all actions)
        self._mi_var_total = MI_EMA * self._mi_var_total + alpha * (delta_blocks ** 2)

    def _compute_mi(self):
        """Compute Gaussian MI approximation per dim."""
        if self._mi_mu is None:
            return
        # Mean within-action variance
        active = self._mi_count > 5  # only actions with enough samples
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)  # (N_DIMS,)
        # MI = 0.5 * log(var_total / mean_within_var)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _r3_mi_attention_update(self):
        """Update attention from MI values (ℓ_π)."""
        self._compute_mi()
        max_mi = float(self._mi_values.max())
        if max_mi < MI_THRESH:
            return
        # Attention from MI
        median_mi = max(float(np.median(
            self._mi_values[self._mi_values > 0])), MI_THRESH) \
            if np.any(self._mi_values > 0) else MI_THRESH
        new_att = np.clip(self._mi_values / median_mi, ATT_MIN, ATT_MAX)
        self.block_attention = (1 - ATT_LR) * self.block_attention + ATT_LR * new_att
        self._upsample_block_attention()
        self.r3_updates += 1
        self.att_updates_total += int(np.sum(self._mi_values > MI_THRESH))

    def _upsample_block_attention(self):
        for by in range(N_BLOCKS):
            for bx in range(N_BLOCKS):
                y0, y1 = by * BLOCK_SIZE, (by + 1) * BLOCK_SIZE
                x0, x1 = bx * BLOCK_SIZE, (bx + 1) * BLOCK_SIZE
                self.attention[y0:y1, x0:x1] = self.block_attention[by * N_BLOCKS + bx]

    def _mi_action_score(self, action):
        """Score an action by MI-weighted expected effect."""
        if self._mi_mu is None or action >= len(self._mi_mu):
            return 0.0
        return float(np.sum(self._mi_values * np.abs(self._mi_mu[action])))

    def _best_mi_action(self, action_set):
        """Choose action from set with highest MI-weighted expected effect."""
        scores = [self._mi_action_score(a) for a in action_set]
        if max(scores) < 1e-10:
            return action_set[self._rng.randint(len(action_set))]
        return action_set[int(np.argmax(scores))]

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        for _ in range(length):
            if self._supports_click and self._rng.random() < 0.7:
                if self._best_click_regions and self._rng.random() < 0.5:
                    cx, cy = self._best_click_regions[
                        self._rng.randint(len(self._best_click_regions))]
                    seq.append(_click_action(cx, cy))
                else:
                    cx, cy = CLICK_GRID[self._rng.randint(len(CLICK_GRID))]
                    seq.append(_click_action(cx, cy))
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

    def _fitness(self, obs_start, obs_end):
        """MI-weighted attention change (ℓ_π fitness)."""
        return float(np.sum(self.attention * np.abs(obs_end - obs_start)))

    def _do_kb_bootloader(self, arr):
        goal = self._gated_goal if self._gated_goal is not None else arr
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        # MI-informed action selection
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            influence_score = np.sum(self.attention * self.kb_influence[k] * mismatch)
            mi_score = self._mi_action_score(k)
            kb_scores[k] = influence_score + 0.5 * mi_score
        action = int(np.argmax(kb_scores))
        if self._rng.random() < 0.1:
            action = self._rng.randint(N_KB)
        self.prev_action_type = 'kb'
        self.prev_kb_idx = action
        return action

    def _do_click_exploit(self, arr):
        goal = self._gated_goal if self._gated_goal is not None else arr
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
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
            score = self._fitness(self._evo_obs_start, arr)
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

        # Block reduction
        blocks = _obs_to_blocks(arr)

        # MI statistics update
        if self._prev_blocks is not None and self._prev_action is not None:
            delta_blocks = blocks - self._prev_blocks
            self._update_mi(self._prev_action, delta_blocks)
            # Periodic MI computation + attention update
            if self.step_count % 50 == 0:
                self._r3_mi_attention_update()

        # Frequency tracking
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0
        self.gated_freq[r, c, obs_int] += self.attention

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
            self._prev_blocks = blocks.copy()
            self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
            self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
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

        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
        self.prev_obs = arr.copy()
        self._prev_blocks = blocks.copy()

        # ── MI WARMUP PHASE (0-300): cycle actions, build MI stats ──
        if self.step_count <= MI_WARMUP:
            # Sustained KB cycling for MI statistics
            kb = ((self.step_count - 1) // SUSTAIN_STEPS) % N_KB
            action = kb
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        # Check MI signal at end of warmup
        if self.step_count == MI_WARMUP + 1:
            self._compute_mi()
            max_mi = float(self._mi_values.max())
            n_informative = int(np.sum(self._mi_values > MI_THRESH))
            self._kb_mi_signal = max_mi

        # ── Already detected → exploit ──
        if self._detected_type is not None:
            action = self._exploit(arr)
        # ── KB PROBE (300-800): MI-informed ──
        elif self.step_count < KB_PROBE_END:
            # Use MI to guide KB probing — try highest-MI action more
            if self.step_count % SUSTAIN_STEPS == 0:
                self._compute_mi()
            kb_actions = list(range(N_KB))
            action = self._best_mi_action(kb_actions) if self._rng.random() > 0.3 else self._rng.randint(N_KB)
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action
        elif self.step_count == KB_PROBE_END:
            self._compute_mi()
            max_mi = float(self._mi_values.max())
            if max_mi > MI_THRESH:
                self._detected_type = 'kb'
            action = self._do_kb_bootloader(arr) if self._detected_type == 'kb' else self._rng.randint(self._n_actions)
        # ── CLICK PROBE (800-1400) ──
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
                max_mi = float(self._mi_values.max())
                if max_mi > MI_THRESH:
                    self._detected_type = 'click'
                    # Find responsive blocks from MI
                    high_mi_blocks = np.argwhere(self._mi_values > np.percentile(self._mi_values, 75))
                    for bi in high_mi_blocks:
                        by, bx = bi[0] // N_BLOCKS, bi[0] % N_BLOCKS
                        cx = bx * BLOCK_SIZE + BLOCK_SIZE // 2
                        cy = by * BLOCK_SIZE + BLOCK_SIZE // 2
                        self._best_click_regions.append((cx, cy))
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
        # ── SEQUENCE PROBE (1400-2400) ──
        elif self.step_count < SEQ_PROBE_END:
            action = self._do_evolution(arr)
        elif self.step_count == SEQ_PROBE_END:
            if self._detected_type is None:
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
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._prev_blocks = None
        self._raw_goal = None
        self._gated_goal = None
        self._detected_type = None
        self._kb_mi_signal = 0.0
        self._click_mi_signal = 0.0
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
        # Reset MI stats for new level but keep attention (ℓ_π transfer)
        self._init_mi_stats(self._n_actions)
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        # Keep block_attention + kb_influence across levels


CONFIG = {
    "mi_warmup": MI_WARMUP,
    "probes": f"warmup(0-{MI_WARMUP})/kb({MI_WARMUP}-{KB_PROBE_END})/click({KB_PROBE_END}-{CLICK_PROBE_END})/seq({CLICK_PROBE_END}-{SEQ_PROBE_END})/exploit({SEQ_PROBE_END}+)",
    "mi_thresh": MI_THRESH,
    "mi_ema": MI_EMA,
    "sustain_steps": SUSTAIN_STEPS,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v16_features": "MI-based action informativeness + MI-attention encoding + sustained probes + adaptive cascade (l_pi)",
}

SUBSTRATE_CLASS = ProsecutionV16Substrate
