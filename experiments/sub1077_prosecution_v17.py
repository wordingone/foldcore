"""
sub1077_prosecution_v17.py — Prosecution v17: paired-action MI probing (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1077 --substrate experiments/sub1077_prosecution_v17.py

FAMILY: attention-gated
R3 HYPOTHESIS: Opaque games (0% wall) only respond to multi-step action sequences,
  not single actions. Paired-action MI probing discovers synergistic pairs where
  MI(a_i,a_j) > MI(a_i) + MI(a_j). Attention (ℓ_π) over action pairs finds
  sequential structure invisible to single-action probing.

  Ref: Gibson 1988 — infant motor babbling discovers affordances via combinations.

KILL: No improvement on 0% games
SUCCESS: Any currently-opaque game breaks 0%
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

# Probe boundaries — longer warmup for pair probing
PAIR_WARMUP = 500
KB_PROBE_END = 1000
CLICK_PROBE_END = 1600
SEQ_PROBE_END = 2600

# MI detection
MI_THRESH = 0.05
MI_EMA = 0.95
MI_EPSILON = 1e-8
SUSTAIN_STEPS = 10

# Pair probing
PAIR_HOLD = 3          # steps between pair elements
MAX_PAIRS = 50         # max tracked pairs
SYNERGY_THRESH = 0.02  # MI_pair - MI_sum threshold

# Attention
ATT_INIT = 0.5
ATT_LR = 0.03
ATT_MIN = 0.01
ATT_MAX = 1.0

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


class ProsecutionV17Substrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self._mi_mu = None
        self._mi_var = None
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_count = None
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._prev_blocks = None

        # Pair MI tracking
        self._pair_keys = []      # list of (a_i, a_j) tuples
        self._pair_mu = {}        # (a_i,a_j) -> EMA mean delta (N_DIMS,)
        self._pair_var = {}       # (a_i,a_j) -> EMA var (N_DIMS,)
        self._pair_count = {}
        self._pair_mi = {}        # computed MI per pair
        self._synergistic_pairs = []
        self._pair_phase_action = None  # first action of current pair
        self._pair_phase_step = 0
        self._pair_obs_before = None

        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.block_attention = np.full(N_DIMS, ATT_INIT, dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)

        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self._raw_goal = None
        self._gated_goal = None
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
        # Generate pair candidates
        kb_actions = list(range(N_KB))
        pairs = []
        for i in kb_actions:
            for j in kb_actions:
                if i != j:
                    pairs.append((i, j))
        if self._supports_click:
            click_sample = [_click_action(cx, cy) for cx, cy in CLICK_GRID[:16]]
            for c in click_sample[:8]:
                for k in kb_actions[:4]:
                    pairs.append((k, c))
                    pairs.append((c, k))
            for i in range(min(8, len(click_sample))):
                for j in range(i+1, min(8, len(click_sample))):
                    pairs.append((click_sample[i], click_sample[j]))
        self._rng.shuffle(pairs)
        self._pair_keys = pairs[:MAX_PAIRS]

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

    def _update_pair_mi(self, pair_key, delta_blocks):
        alpha = 1.0 - MI_EMA
        if pair_key not in self._pair_mu:
            self._pair_mu[pair_key] = np.zeros(N_DIMS, dtype=np.float32)
            self._pair_var[pair_key] = np.full(N_DIMS, 1e-4, dtype=np.float32)
            self._pair_count[pair_key] = 0
        self._pair_count[pair_key] += 1
        self._pair_mu[pair_key] = MI_EMA * self._pair_mu[pair_key] + alpha * delta_blocks
        residual = delta_blocks - self._pair_mu[pair_key]
        self._pair_var[pair_key] = MI_EMA * self._pair_var[pair_key] + alpha * (residual ** 2)

    def _compute_mi(self):
        if self._mi_mu is None:
            return
        active = self._mi_count > 5
        if active.sum() < 2:
            return
        mean_within_var = self._mi_var[active].mean(axis=0)
        ratio = self._mi_var_total / np.maximum(mean_within_var, MI_EPSILON)
        self._mi_values = np.maximum(0.5 * np.log(np.maximum(ratio, 1.0)), 0.0)

    def _compute_pair_synergy(self):
        self._synergistic_pairs = []
        for pk in self._pair_keys:
            if pk not in self._pair_count or self._pair_count[pk] < 3:
                continue
            a_i, a_j = pk
            # Single-action MI contribution
            single_sum = np.zeros(N_DIMS, dtype=np.float32)
            if a_i < len(self._mi_mu) and self._mi_count[a_i] > 3:
                ratio_i = self._mi_var_total / np.maximum(self._mi_var[a_i], MI_EPSILON)
                single_sum += np.maximum(0.5 * np.log(np.maximum(ratio_i, 1.0)), 0.0)
            if a_j < len(self._mi_mu) and self._mi_count[a_j] > 3:
                ratio_j = self._mi_var_total / np.maximum(self._mi_var[a_j], MI_EPSILON)
                single_sum += np.maximum(0.5 * np.log(np.maximum(ratio_j, 1.0)), 0.0)
            # Pair MI
            pair_var = self._pair_var[pk]
            ratio_p = self._mi_var_total / np.maximum(pair_var.mean() + MI_EPSILON, MI_EPSILON)
            pair_mi = max(0.0, float(0.5 * np.log(max(ratio_p.mean(), 1.0))))
            single_mi = float(single_sum.mean())
            synergy = pair_mi - single_mi
            self._pair_mi[pk] = pair_mi
            if synergy > SYNERGY_THRESH:
                self._synergistic_pairs.append((synergy, pk))
        self._synergistic_pairs.sort(reverse=True)

    def _r3_mi_attention_update(self):
        self._compute_mi()
        max_mi = float(self._mi_values.max())
        if max_mi < MI_THRESH:
            return
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
        if self._mi_mu is None or action >= len(self._mi_mu):
            return 0.0
        return float(np.sum(self._mi_values * np.abs(self._mi_mu[action])))

    def _random_sequence(self):
        length = self._rng.randint(SEQ_MIN, SEQ_MAX + 1)
        seq = []
        # Inject synergistic pairs into sequences
        if self._synergistic_pairs and self._rng.random() < 0.5:
            _, (a_i, a_j) = self._synergistic_pairs[
                self._rng.randint(min(5, len(self._synergistic_pairs)))]
            seq = [a_i, a_j]
            length -= 2
        for _ in range(max(0, length)):
            if self._supports_click and self._rng.random() < 0.7:
                if self._best_click_regions and self._rng.random() < 0.5:
                    cx, cy = self._best_click_regions[self._rng.randint(len(self._best_click_regions))]
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
            if self._synergistic_pairs and self._rng.random() < 0.3:
                _, (a_i, a_j) = self._synergistic_pairs[
                    self._rng.randint(min(5, len(self._synergistic_pairs)))]
                seq.insert(idx, a_i)
                seq.insert(min(idx+1, len(seq)), a_j)
            elif self._supports_click and self._rng.random() < 0.7:
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
        return float(np.sum(self.attention * np.abs(obs_end - obs_start)))

    def _do_kb_bootloader(self, arr):
        goal = self._gated_goal if self._gated_goal is not None else arr
        mismatch = self.attention * np.abs(arr - goal) * self.change_map
        suppress_mask = (self.suppress == 0).astype(np.float32)
        mismatch *= suppress_mask
        kb_scores = np.zeros(N_KB)
        for k in range(N_KB):
            kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * mismatch) + 0.5 * self._mi_action_score(k)
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
        blocks = _obs_to_blocks(arr)

        if self._prev_blocks is not None and self._prev_action is not None:
            delta_blocks = blocks - self._prev_blocks
            self._update_mi(self._prev_action, delta_blocks)
            # Pair MI update
            if self._pair_phase_action is not None and self._pair_obs_before is not None:
                self._pair_phase_step += 1
                if self._pair_phase_step >= PAIR_HOLD:
                    pair_delta = blocks - _obs_to_blocks(self._pair_obs_before)
                    pk = (self._pair_phase_action, self._prev_action)
                    if pk in self._pair_keys or len(self._pair_mu) < MAX_PAIRS:
                        self._update_pair_mi(pk, pair_delta)
                    self._pair_phase_action = None
                    self._pair_obs_before = None
                    self._pair_phase_step = 0
            if self.step_count % 50 == 0:
                self._r3_mi_attention_update()

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
            self.kb_influence[self.prev_kb_idx] = (0.9 * self.kb_influence[self.prev_kb_idx] + 0.1 * diff)
        self._raw_goal = np.argmax(self.raw_freq, axis=2).astype(np.float32)
        self._gated_goal = np.argmax(self.gated_freq, axis=2).astype(np.float32)
        self.prev_obs = arr.copy()
        self._prev_blocks = blocks.copy()

        # ── PAIR WARMUP (0-500): single + pair probing ──
        if self.step_count <= PAIR_WARMUP:
            pair_idx = (self.step_count // (PAIR_HOLD + 1)) % max(1, len(self._pair_keys))
            step_in_pair = self.step_count % (PAIR_HOLD + 1)
            if step_in_pair == 0 and pair_idx < len(self._pair_keys):
                a_i, _ = self._pair_keys[pair_idx]
                action = a_i if a_i < self._n_actions else self._rng.randint(self._n_actions)
                self._pair_phase_action = action
                self._pair_obs_before = arr.copy()
                self._pair_phase_step = 0
            elif step_in_pair == PAIR_HOLD and pair_idx < len(self._pair_keys):
                _, a_j = self._pair_keys[pair_idx]
                action = a_j if a_j < self._n_actions else self._rng.randint(self._n_actions)
            else:
                action = self._rng.randint(N_KB)
            if self.step_count == PAIR_WARMUP:
                self._compute_mi()
                self._compute_pair_synergy()
            self.prev_action_type = 'kb' if action < N_KB else 'click'
            self.prev_kb_idx = action if action < N_KB else None
            self._prev_obs_arr = arr.copy()
            self._prev_action = action
            return action

        if self._detected_type is not None:
            action = self._exploit(arr)
        elif self.step_count < KB_PROBE_END:
            if self._synergistic_pairs and self._rng.random() < 0.4:
                _, (a_i, a_j) = self._synergistic_pairs[
                    self._rng.randint(min(5, len(self._synergistic_pairs)))]
                action = a_i if (self.step_count % 2 == 0) else a_j
                if action >= self._n_actions:
                    action = self._rng.randint(self._n_actions)
            else:
                action = self._rng.randint(N_KB)
            self.prev_action_type = 'kb'
            self.prev_kb_idx = action if action < N_KB else None
        elif self.step_count == KB_PROBE_END:
            self._compute_mi()
            if float(self._mi_values.max()) > MI_THRESH or len(self._synergistic_pairs) > 0:
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
            action = self._do_click_exploit(arr) if self._detected_type == 'click' else self._rng.randint(self._n_actions)
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
        self._pair_phase_action = None
        self._pair_obs_before = None
        self._pair_phase_step = 0
        self._init_mi_stats(self._n_actions)
        self._mi_var_total = np.zeros(N_DIMS, dtype=np.float32)
        self._mi_values = np.zeros(N_DIMS, dtype=np.float32)
        self._pair_mu = {}
        self._pair_var = {}
        self._pair_count = {}
        self._pair_mi = {}
        # Keep synergistic_pairs + block_attention across levels (ℓ_π transfer)


CONFIG = {
    "pair_warmup": PAIR_WARMUP,
    "pair_hold": PAIR_HOLD,
    "max_pairs": MAX_PAIRS,
    "synergy_thresh": SYNERGY_THRESH,
    "mi_thresh": MI_THRESH,
    "v17_features": "paired-action MI probing + synergy detection + MI-attention (l_pi)",
}

SUBSTRATE_CLASS = ProsecutionV17Substrate
