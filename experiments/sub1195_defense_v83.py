"""
sub1195_defense_v83.py — Anti-correlated action pairs (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1195 --substrate experiments/sub1195_defense_v83.py

FAMILY: Anti-correlated action pairs. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: Step 1082 diagnostic found OSCILLATION in games — 47/74
sign changes under best-action repetition. Actions produce directional
effects but DON'T ACCUMULATE because the game bounces back.

v80/v82 rank INDIVIDUAL actions. But oscillating games might need
ACTION ALTERNATION — two actions with OPPOSITE effects that together
drive the state in a direction neither achieves alone.

v83 strategy:
- Phase 1 (100 steps): random exploration, track per-action SIGNED change
  vectors (like v82).
- After exploration: compute pairwise cosine similarity of action effects.
  Find the most ANTI-CORRELATED pair (most negative cosine). These are
  actions with opposite effects — one pushes state "left", other "right".
- Phase 2: alternate between the anti-correlated pair. Also try the top
  individual action (v80-style). Cycle through: pair alternation, then
  individual top actions.

If oscillation is the bottleneck, alternating opposite actions should
break through where single-action strategies can't.

20% epsilon-greedy throughout.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (no improvement over random).
SUCCESS: avg L1 > 3.3/5 (beats v80) OR solves a 0% game.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EXPLORE_STEPS = 100
EPSILON = 0.2
CHANGE_THRESH = 0.1


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class AntiCorrelatedPairSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._exploring = True

        # Per-action signed change vectors
        self._action_signed_sum = {}   # action -> np.array(256)
        self._action_abs_sum = {}      # action -> np.array(256)
        self._action_count = {}        # action -> int

        # Exploit state
        self._strategy_list = []  # list of actions to cycle through
        self._current_idx = 0
        self._hold_count = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _build_strategy(self):
        """Build alternation strategy from exploration data."""
        # Compute mean signed change vectors for each action
        action_vecs = {}
        action_mags = {}
        for a in self._action_signed_sum:
            count = max(self._action_count.get(a, 1), 1)
            mean_vec = self._action_signed_sum[a] / count
            action_vecs[a] = mean_vec
            action_mags[a] = float(np.sum(self._action_abs_sum[a] / count))

        actions = list(action_vecs.keys())
        if len(actions) < 2:
            n_kb = min(self._n_actions_env, N_KB)
            self._strategy_list = list(range(n_kb))
            return

        # Find the most anti-correlated pair
        best_pair = None
        best_cos = 1.0  # want most negative

        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                a1, a2 = actions[i], actions[j]
                v1, v2 = action_vecs[a1], action_vecs[a2]
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 < 1e-6 or norm2 < 1e-6:
                    continue
                cos = float(np.dot(v1, v2) / (norm1 * norm2))
                # Prefer anti-correlated pairs with large magnitudes
                score = cos - 0.1 * (action_mags[a1] + action_mags[a2])
                if score < best_cos:
                    best_cos = score
                    best_pair = (a1, a2)

        # Build strategy: alternating pair first, then top individual actions
        strategy = []

        if best_pair is not None and best_cos < 0:
            # Anti-correlated pair found — alternate A, B, A, B
            a1, a2 = best_pair
            strategy.extend([a1, a2, a1, a2])

        # Add top individual actions by magnitude (v80-style)
        sorted_by_mag = sorted(action_mags.items(), key=lambda x: -x[1])
        for a, mag in sorted_by_mag:
            if mag > CHANGE_THRESH and a not in strategy:
                strategy.append(a)

        if not strategy:
            n_kb = min(self._n_actions_env, N_KB)
            strategy = list(range(n_kb))

        self._strategy_list = strategy

    def _transition_to_exploit(self):
        self._exploring = False
        self._build_strategy()
        self.r3_updates += 1
        self.att_updates_total += 1

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Compute signed change
        delta_signed = enc - self._prev_enc
        delta_abs = np.abs(delta_signed)

        # Record stats for previous action
        a = self._prev_action
        if a not in self._action_signed_sum:
            self._action_signed_sum[a] = np.zeros(N_DIMS, dtype=np.float32)
            self._action_abs_sum[a] = np.zeros(N_DIMS, dtype=np.float32)
            self._action_count[a] = 0
        self._action_signed_sum[a] += delta_signed
        self._action_abs_sum[a] += delta_abs
        self._action_count[a] += 1

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === EXPLOIT: cycle through strategy list ===

        # Epsilon-greedy
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_action = action
            return action

        # Cycle through strategy — advance every step for pairs,
        # hold longer for individual actions
        self._hold_count += 1

        # For pair section (first 4 entries), alternate every step
        # For individual actions, hold for 10 steps
        if self._current_idx < 4 and len(self._strategy_list) > 4:
            # Pair alternation — advance every step
            self._current_idx = (self._current_idx + 1) % len(self._strategy_list)
            self._hold_count = 0
        elif self._hold_count > 10:
            # Individual action — advance after patience
            self._current_idx = (self._current_idx + 1) % len(self._strategy_list)
            self._hold_count = 0

        action = self._strategy_list[self._current_idx % len(self._strategy_list)]
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._hold_count = 0
        # Keep strategy across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "family": "anti-correlated action pairs",
    "tag": "defense v83 (ℓ₁ anti-correlated pairs: find actions with OPPOSITE effects via signed change cosine, alternate between them. Tests if oscillating games need action ALTERNATION, not holding. Step 1082 diagnostic: 47/74 sign changes = oscillation.)",
}

SUBSTRATE_CLASS = AntiCorrelatedPairSubstrate
