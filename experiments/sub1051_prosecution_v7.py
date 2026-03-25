"""
sub1051_prosecution_v7.py — Prosecution v7: evolutionary sequence search (ℓ_π).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1051 --substrate experiments/sub1051_prosecution_v7.py

FAMILY: attention-gated
R3 HYPOTHESIS: ℓ_π evolutionary sequence discovery with attention-weighted fitness.
  Single actions and pairs (v5) both fail on click games (100+ runs at 0%).
  Hypothesis: some games require 3-7 step action SEQUENCES as atomic units.

  (1+1)-ES evolves a population of action sequences.
  Each sequence is executed completely, then scored by attention-weighted change.
  Attention adapts via counterfactual advantage (ℓ_π R3).

  As attention learns which dimensions are informative, evolutionary pressure
  SHIFTS — selecting sequences that affect the RIGHT pixels, not just any pixels.

Phase 1 (0-500): Keyboard sweep + random clicks. Build change_map + attention.
Phase 2 (500-8000): Evolutionary sequence search.
  - Population of 10 sequences (length 3-7).
  - Round-robin evaluation: execute sequence, score, accumulate EMA.
  - Every 10 evaluations: mutate worst sequence using best as parent.
Phase 3 (8000+): Exploit top sequences + attention-gated fallback.

KILL: ALL games 0% → sequences don't solve it either
SUCCESS: L1 > 0 on any previously-0% game
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from scipy.ndimage import uniform_filter

# ─── Hyperparameters (ONE config for all games) ───
KB_PHASE_END = 210
WARMUP_END = 500
EVO_PHASE_END = 8000
ALPHA_CHANGE = 0.99
KERNEL = 5
SUPPRESS_RADIUS = 3
SUPPRESS_DURATION = 8
CF_CHANGE_THRESH = 0.1

# Attention R3 parameters
ATT_INIT = 0.5
ATT_LR = 0.02
ATT_MIN = 0.01
ATT_MAX = 1.0

# Evolutionary parameters
POP_SIZE = 10
SEQ_MIN = 3
SEQ_MAX = 7
MUTATE_EVERY = 10  # mutate worst every N evaluations

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


class ProsecutionV7Substrate:
    """
    ℓ_π R3 v7: evolutionary sequence search + attention-weighted fitness.
    Attention weights adapt fitness function during evolution (R3).
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._supports_click = False
        self._init_state()

    def _init_state(self):
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._raw_goal = None
        self._gated_goal = None
        # Evolution state
        self._evo_pop = []          # list of action sequences
        self._evo_scores = []       # EMA fitness scores
        self._evo_counts = []       # evaluation counts
        self._evo_current = 0       # which sequence being evaluated
        self._evo_exec_idx = 0      # position in current sequence
        self._evo_obs_start = None  # obs at start of sequence
        self._evo_total_evals = 0
        self._evo_initialized = False
        # Exploit state
        self._top_sequences = []
        self._exploit_exec_idx = 0
        self._exploit_current = 0
        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._supports_click = n_actions > N_KB
        self.raw_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.gated_freq = np.zeros((64, 64, 16), dtype=np.float32)
        self.attention = np.full((64, 64), ATT_INIT, dtype=np.float32)
        self.change_map = np.zeros((64, 64), dtype=np.float32)
        self.prev_obs = None
        self.suppress = np.zeros((64, 64), dtype=np.int32)
        self.kb_influence = np.zeros((N_KB, 64, 64), dtype=np.float32)
        self.prev_kb_idx = None
        self.prev_action_type = None
        self.step_count = 0
        self._prev_obs_arr = None
        self._prev_action = None
        self._raw_goal = None
        self._gated_goal = None
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

    def _fitness(self, obs_start, obs_end):
        """Attention-weighted total change (ℓ_π fitness)."""
        return float(np.sum(self.attention * np.abs(obs_end - obs_start)))

    def _r3_attention_update(self, obs_before, obs_after):
        diff = np.abs(obs_after - obs_before)
        changed = diff > CF_CHANGE_THRESH
        n_changed = int(np.sum(changed))
        if n_changed == 0 or self._raw_goal is None or self._gated_goal is None:
            return
        raw_error = np.abs(self._raw_goal - obs_after)
        gated_error = np.abs(self._gated_goal - obs_after)
        advantage = raw_error - gated_error
        self.attention[changed] += ATT_LR * advantage[changed]
        self.attention = np.clip(self.attention, ATT_MIN, ATT_MAX)
        self.r3_updates += 1
        self.att_updates_total += n_changed

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

        # R3 update
        if self._prev_obs_arr is not None and self._prev_action is not None:
            self._r3_attention_update(self._prev_obs_arr, arr)

        # Update frequency histograms
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.raw_freq[r, c, obs_int] += 1.0
        self.gated_freq[r, c, obs_int] += self.attention

        if self.prev_obs is None:
            self.prev_obs = arr.copy()
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

            # Starting new sequence evaluation?
            if self._evo_exec_idx == 0:
                self._evo_obs_start = arr.copy()

            # Get current sequence
            seq = self._evo_pop[self._evo_current]
            action = seq[self._evo_exec_idx]
            if action >= self._n_actions:
                action = self._rng.randint(self._n_actions)
            self._evo_exec_idx += 1

            # End of sequence?
            if self._evo_exec_idx >= len(seq):
                # Score this sequence (using NEXT obs, but we don't have it yet)
                # Score using current obs as proxy (off by one step, acceptable)
                score = self._fitness(self._evo_obs_start, arr)
                idx = self._evo_current
                self._evo_counts[idx] += 1
                alpha = 0.3 if self._evo_counts[idx] > 1 else 1.0
                self._evo_scores[idx] = (1 - alpha) * self._evo_scores[idx] + alpha * score
                self._evo_total_evals += 1

                # Mutate worst every MUTATE_EVERY evaluations
                if self._evo_total_evals % MUTATE_EVERY == 0 and self._evo_total_evals > POP_SIZE:
                    worst = int(np.argmin(self._evo_scores))
                    best = int(np.argmax(self._evo_scores))
                    if worst != best:
                        self._evo_pop[worst] = self._mutate_sequence(self._evo_pop[best])
                        self._evo_scores[worst] = self._evo_scores[best] * 0.5
                        self._evo_counts[worst] = 0

                # Move to next sequence (round-robin)
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

        # ── Phase 3: exploit top sequences + attention fallback ──
        if not self._top_sequences and self._evo_pop:
            ranked = sorted(range(POP_SIZE), key=lambda i: -self._evo_scores[i])
            self._top_sequences = [self._evo_pop[i] for i in ranked[:5] if self._evo_scores[i] > 0]
            if not self._top_sequences:
                self._top_sequences = [self._evo_pop[ranked[0]]]

        # Epsilon-greedy
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

        # Execute top sequences (70%) or attention-gated singles (30%)
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
            goal = self._gated_goal
            mismatch = self.attention * np.abs(arr - goal) * self.change_map
            suppress_mask = (self.suppress == 0).astype(np.float32)
            mismatch *= suppress_mask
            smoothed = uniform_filter(mismatch, size=KERNEL)

            click_score = float(np.max(smoothed)) if self._supports_click else -1.0
            kb_scores = np.zeros(N_KB)
            for k in range(N_KB):
                kb_scores[k] = np.sum(self.attention * self.kb_influence[k] * smoothed)
            best_kb = int(np.argmax(kb_scores))
            best_kb_score = kb_scores[best_kb]

            if self._supports_click and click_score >= best_kb_score:
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
        self.raw_freq[:] = 0
        self.gated_freq[:] = 0
        self.prev_obs = None
        self._raw_goal = None
        self._gated_goal = None
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
        # Keep attention + kb_influence (learned priors)


CONFIG = {
    "warmup": WARMUP_END,
    "evo_phase": EVO_PHASE_END,
    "pop_size": POP_SIZE,
    "seq_range": f"{SEQ_MIN}-{SEQ_MAX}",
    "att_init": ATT_INIT,
    "att_lr": ATT_LR,
    "v7_features": "evolutionary sequence search + attention-weighted fitness (l_pi)",
}

SUBSTRATE_CLASS = ProsecutionV7Substrate
