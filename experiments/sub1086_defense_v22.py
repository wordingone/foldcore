"""
sub1086_defense_v22.py — Multi-scale reactive switching with action memory

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1086 --substrate experiments/sub1086_defense_v22.py

FAMILY: Multi-scale reactive (defense-only architecture)
Tagged: defense (ℓ₁)
R3 HYPOTHESIS: Multi-scale observation comparison with short-term action memory
breaks more games than single-scale reactive switching. If fixed multi-scale
comparison works, learned encoding (prosecution's alpha) is unnecessary — the
substrate just needs to look at the right resolution. R3 test: does the substrate
need to self-modify its comparison metric, or is multi-resolution fixed comparison
sufficient?

BUILDS ON Step 1084 (defense v21):
- v21 solved GAME_4 (100%, ARC=0.2973) but missed 2/3 ARC games
- v21 limitation: single scale (avgpool8, 64D). Some games may need finer detail.
- v21 limitation: no memory of which actions worked before — only immediate comparison.

ARCHITECTURE (genuinely different from prosecution's attention-trajectory):
- THREE encoding scales: avgpool4 (256D), avgpool8 (64D), avgpool16 (16D)
- Active scale selection: use the scale showing MOST change (highest delta variance)
  - Compute delta at each scale after each action
  - Track running variance of deltas per scale
  - Use scale with highest variance (most informative resolution)
- Short-term action memory (last K=20 actions + their effects):
  - Ring buffer of (action, progress_at_active_scale)
  - Action scores = mean progress from last K occurrences
  - NOT learned EMA — just raw recent history
- Reactive switching: same as v21 but at selected scale with action memory

WHY DIFFERENT FROM PROSECUTION:
- No learned parameters (no alpha, no W_pred, no attention weights)
- No trajectory buffer with attention retrieval
- Scale selection is deterministic from variance, not learned
- Action memory is raw recent history, not EMA or attention-weighted

KILL: worse than v21 (no games solved) → KILL.
SUCCESS: solve same game as v21 + any additional game.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_KB = 7
EXPLORE_STEPS = 50
MAX_PATIENCE = 20
ACTION_MEMORY_K = 20  # remember last K action effects
SCALE_WINDOW = 50     # steps to evaluate scale variance

# Three scales
SCALES = [
    (4, 16),   # avgpool4: 16x16 = 256D (fine)
    (8, 8),    # avgpool8: 8x8 = 64D (medium)
    (16, 4),   # avgpool16: 4x4 = 16D (coarse)
]


def _obs_to_enc(obs, block_size, n_blocks):
    """Pool observation at given scale."""
    n_dims = n_blocks * n_blocks
    enc = np.zeros(n_dims, dtype=np.float32)
    for by in range(n_blocks):
        for bx in range(n_blocks):
            y0, y1 = by * block_size, (by + 1) * block_size
            x0, x1 = bx * block_size, (bx + 1) * block_size
            enc[by * n_blocks + bx] = obs[y0:y1, x0:x1].mean()
    return enc


class MultiScaleReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        # Per-scale state
        self._enc_0 = [None, None, None]  # initial encoding per scale
        self._prev_enc = [None, None, None]
        self._prev_dist = [None, None, None]
        # Scale selection
        self._scale_deltas = [[] for _ in range(3)]  # recent deltas per scale
        self._active_scale = 1  # start with medium (avgpool8)
        # Action state
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        # Action memory: ring buffer of (action, progress)
        self._action_history = []

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions = min(n_actions, N_KB)
        self._init_state()

    def _encode_all_scales(self, obs):
        """Encode observation at all three scales."""
        encs = []
        for block_size, n_blocks in SCALES:
            encs.append(_obs_to_enc(obs, block_size, n_blocks))
        return encs

    def _dist_to_initial(self, enc, scale_idx):
        """L1 distance to initial at given scale."""
        if self._enc_0[scale_idx] is None:
            return 0.0
        return float(np.sum(np.abs(enc - self._enc_0[scale_idx])))

    def _select_active_scale(self):
        """Pick scale with highest delta variance (most informative)."""
        best_var = -1
        best_scale = self._active_scale
        for i in range(3):
            if len(self._scale_deltas[i]) < 10:
                continue
            var = float(np.var(self._scale_deltas[i][-SCALE_WINDOW:]))
            if var > best_var:
                best_var = var
                best_scale = i
        self._active_scale = best_scale

    def _action_score_from_memory(self, action):
        """Mean progress for this action from recent history."""
        recent = [(a, p) for a, p in self._action_history if a == action]
        if len(recent) < 2:
            return 0.0
        return float(np.mean([p for _, p in recent]))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, self._n_actions))

        self.step_count += 1
        encs = self._encode_all_scales(obs)

        # Store initial encodings
        for i in range(3):
            if self._enc_0[i] is None:
                self._enc_0[i] = encs[i].copy()
                self._prev_enc[i] = encs[i].copy()
                self._prev_dist[i] = 0.0

        if self.step_count == 1:
            self._current_action = self._rng.randint(self._n_actions)
            return self._current_action

        # Compute deltas at all scales
        for i in range(3):
            delta = float(np.sum(np.abs(encs[i] - self._prev_enc[i])))
            self._scale_deltas[i].append(delta)

        # Select active scale periodically
        if self.step_count % 50 == 0:
            self._select_active_scale()

        s = self._active_scale
        dist = self._dist_to_initial(encs[s], s)

        # Initial exploration
        if self.step_count <= EXPLORE_STEPS:
            action = self.step_count % self._n_actions
            for i in range(3):
                self._prev_enc[i] = encs[i].copy()
                self._prev_dist[i] = self._dist_to_initial(encs[i], i)
            return action

        # Record action effect in memory
        progress = self._prev_dist[s] - dist  # positive = moved toward initial
        self._action_history.append((self._current_action, progress))
        if len(self._action_history) > ACTION_MEMORY_K:
            self._action_history.pop(0)

        # Reactive policy with action memory
        made_progress = progress > 1e-4
        no_change = abs(progress) < 1e-6

        self._steps_on_action += 1

        if made_progress:
            self._consecutive_progress += 1
            self._patience = min(3 + self._consecutive_progress, MAX_PATIENCE)
            self._actions_tried_this_round = 0
        else:
            self._consecutive_progress = 0

            if self._steps_on_action >= self._patience or no_change:
                self._actions_tried_this_round += 1
                self._steps_on_action = 0
                self._patience = 3

                if self._actions_tried_this_round >= self._n_actions:
                    # All actions tried — pick best from memory
                    best_a = self._current_action
                    best_score = -999
                    for a in range(self._n_actions):
                        sc = self._action_score_from_memory(a)
                        if sc > best_score:
                            best_score = sc
                            best_a = a
                    if best_score > 1e-4:
                        self._current_action = best_a
                    else:
                        self._current_action = self._rng.randint(self._n_actions)
                    self._actions_tried_this_round = 0
                else:
                    self._current_action = (self._current_action + 1) % self._n_actions

        for i in range(3):
            self._prev_enc[i] = encs[i].copy()
            self._prev_dist[i] = self._dist_to_initial(encs[i], i)

        return self._current_action

    def on_level_transition(self):
        self._enc_0 = [None, None, None]
        self._prev_enc = [None, None, None]
        self._prev_dist = [None, None, None]
        self._current_action = 0
        self._patience = 3
        self._consecutive_progress = 0
        self._steps_on_action = 0
        self._actions_tried_this_round = 0
        self._action_history = []
        self._scale_deltas = [[] for _ in range(3)]


CONFIG = {
    "scales": "avgpool4(256D) + avgpool8(64D) + avgpool16(16D)",
    "action_memory_k": ACTION_MEMORY_K,
    "scale_window": SCALE_WINDOW,
    "explore_steps": EXPLORE_STEPS,
    "max_patience": MAX_PATIENCE,
    "family": "multi-scale reactive",
    "tag": "defense v22 (ℓ₁ multi-scale + action memory, zero learned params)",
}

SUBSTRATE_CLASS = MultiScaleReactiveSubstrate
