"""
sub1192_defense_v82.py — Directional consistency reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1192 --substrate experiments/sub1192_defense_v82.py

FAMILY: Directional consistency. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: v80 measures raw change MAGNITUDE — which actions cause
the most pixel change? But not all change is meaningful. Random noise
causes large deltas that cancel over time. An action that consistently
pushes pixels in the SAME direction is more likely "doing something real."

v82 measures COHERENCE of each action's effect:
- For each action, track both SIGNED change (sum of enc_after - enc_before)
  and ABSOLUTE change (sum of |enc_after - enc_before|) per dimension.
- Consistency = |mean_signed| / (mean_abs + eps), averaged across dims.
  - Near 1.0: action always changes in same direction (coherent effect)
  - Near 0.0: changes cancel out (random noise, oscillation)
- Rank actions by: consistency × magnitude (coherent AND large preferred)

This tests: does v80's advantage come from finding actions with LARGE
effects, or from finding actions with COHERENT effects? v82 should
outperform v80 on games where coherent actions exist but are masked by
noise, and match v80 on games where raw magnitude is the right signal.

Phase 1 (100 steps): random exploration, track signed+abs change per action.
Phase 2: ranked cycling by coherence score. Switch when current action's
recent coherence drops (the action stopped having a consistent effect).
20% epsilon-greedy throughout.

ZERO learned parameters (defense: ℓ₁).

KILL: avg L1 ≤ 3.0/5 (no improvement over random).
SUCCESS: avg L1 > 3.3/5 (beats v80) OR same L1 with better ARC.
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
COHERENCE_EPS = 1e-6


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class DirectionalConsistencySubstrate:
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

        # Per-action directional statistics (256D vectors)
        self._action_signed_sum = {}   # action -> np.array(256) signed change
        self._action_abs_sum = {}      # action -> np.array(256) abs change
        self._action_count = {}        # action -> int

        # Exploit state
        self._ranked_actions = []
        self._current_idx = 0
        self._recent_coherence = 0.0
        self._recent_count = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _coherence_score(self, action):
        """Compute coherence: |mean_signed| / (mean_abs + eps), averaged across dims."""
        if action not in self._action_signed_sum:
            return 0.0
        count = max(self._action_count.get(action, 1), 1)
        mean_signed = self._action_signed_sum[action] / count  # 256D
        mean_abs = self._action_abs_sum[action] / count         # 256D

        # Per-dim coherence ratio
        coherence_per_dim = np.abs(mean_signed) / (mean_abs + COHERENCE_EPS)
        # Average coherence across dims
        avg_coherence = float(np.mean(coherence_per_dim))

        # Magnitude: mean absolute change across dims
        avg_magnitude = float(np.mean(mean_abs))

        # Combined score: coherent AND large changes preferred
        return avg_coherence * avg_magnitude

    def _rebuild_ranking(self):
        """Rank actions by coherence × magnitude (highest first)."""
        action_scores = []
        for a in self._action_signed_sum:
            score = self._coherence_score(a)
            action_scores.append((score, a))
        action_scores.sort(reverse=True)
        self._ranked_actions = [a for score, a in action_scores if score > 0.001]

        # Fallback: keyboard actions
        if not self._ranked_actions:
            n_kb = min(self._n_actions_env, N_KB)
            self._ranked_actions = list(range(n_kb))

    def _transition_to_exploit(self):
        self._exploring = False
        self._rebuild_ranking()
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

        # Compute signed and absolute change vectors
        delta_signed = enc - self._prev_enc          # 256D signed
        delta_abs = np.abs(delta_signed)             # 256D absolute

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

        # Periodically rebuild ranking (new actions from epsilon)
        if self.step_count % 500 == 0:
            self._rebuild_ranking()

        # === EXPLOIT: coherence-ranked cycling ===

        # Epsilon-greedy exploration
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_action = action
            self._recent_coherence = 0.0
            self._recent_count = 0
            return action

        # Track recent coherence of current action
        current_delta_mag = float(np.sum(delta_abs))
        if current_delta_mag > 0.01:
            # Compute instantaneous coherence with action's historical direction
            current_action = self._ranked_actions[self._current_idx]
            if current_action in self._action_signed_sum:
                count = max(self._action_count[current_action], 1)
                mean_dir = self._action_signed_sum[current_action] / count
                mean_dir_norm = np.linalg.norm(mean_dir)
                if mean_dir_norm > COHERENCE_EPS:
                    # Cosine similarity between current change and historical direction
                    cos_sim = float(np.dot(delta_signed, mean_dir) / (np.linalg.norm(delta_signed) * mean_dir_norm + COHERENCE_EPS))
                    self._recent_coherence += cos_sim
                    self._recent_count += 1

        # Switch when recent coherence is negative (action going opposite direction)
        # or after patience exhausted
        should_switch = False
        if self._recent_count >= 5:
            avg_recent = self._recent_coherence / self._recent_count
            if avg_recent < 0.0:
                should_switch = True
        if self._recent_count > 20:
            should_switch = True

        if should_switch:
            self._current_idx = (self._current_idx + 1) % len(self._ranked_actions)
            self._recent_coherence = 0.0
            self._recent_count = 0

        action = self._ranked_actions[self._current_idx]
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._recent_coherence = 0.0
        self._recent_count = 0
        # Keep directional statistics across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "explore_steps": EXPLORE_STEPS,
    "epsilon": EPSILON,
    "family": "directional consistency",
    "tag": "defense v82 (ℓ₁ directional consistency: rank actions by coherence × magnitude. Coherence = |mean_signed_change| / mean_abs_change per dim. Tests if COHERENT change beats raw MAGNITUDE (v80's signal).)",
}

SUBSTRATE_CLASS = DirectionalConsistencySubstrate
