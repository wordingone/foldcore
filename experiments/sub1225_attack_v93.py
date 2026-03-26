"""
sub1225_attack_v93.py — State-conditioned rankings (ATTACK: ℓ_π)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1225 --substrate experiments/sub1225_attack_v93.py

FAMILY: State-conditioned action ranking. Tagged: ATTACK (tests prosecution).
R3 HYPOTHESIS: v91 (ATTACK) proved minimal ℓ_π (EMA) = v80. The 800b theorem
says GLOBAL per-action stats fail for state-dependent games. What if we
condition rankings on a COARSE state descriptor?

v93 = v80 recipe but with STATE-CONDITIONED rankings:
  - Partition observation space into 4 coarse states based on which QUADRANT
    of the 16x16 encoding has the highest mean activation
  - Maintain separate per-action change stats for each coarse state
  - During exploit, use the ranking for the CURRENT coarse state

This is a genuine ℓ_π substrate: it learns different action rankings for
different observation regions. But it's NOT per-observation memory (banned):
it uses a FIXED 4-way partition, not a growing graph.

If v93 > v80: state-conditioning helps. Prosecution has a point — learning
WHERE you are matters. 800b theorem correctly predicts global stats fail.
If v93 = v80: coarse state partition too crude. Need finer resolution.
If v93 < v80: state conditioning with too few samples degrades rankings.

KILL: avg L1 ≤ 3.3/5 (state conditioning doesn't help at 4 states).
SUCCESS: avg L1 > 3.3/5 (prosecution claim supported).
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

EPSILON = 0.2
CHANGE_THRESH = 0.1
EXPLORE_STEPS = 100
N_COARSE_STATES = 4  # quadrant-based


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _coarse_state(enc):
    """Map 256D encoding to one of 4 coarse states (quadrants)."""
    grid = enc.reshape(N_BLOCKS, N_BLOCKS)  # 16x16
    q_means = [
        grid[:8, :8].mean(),   # top-left
        grid[:8, 8:].mean(),   # top-right
        grid[8:, :8].mean(),   # bottom-left
        grid[8:, 8:].mean(),   # bottom-right
    ]
    return int(np.argmax(q_means))


class StateConditionedSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._prev_enc = None
        self._prev_action = 0
        self._prev_coarse = 0
        self._exploring = True

        # Per-(coarse_state, action) change statistics
        self._state_action_change_sum = {}  # (state, action) -> float
        self._state_action_change_count = {}  # (state, action) -> int

        # Also keep global stats as fallback
        self._action_change_sum = {}
        self._action_change_count = {}

        # Per-coarse-state ranked action lists
        self._state_ranked_actions = {}  # state -> [actions]

        # Exploit state
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._init_state()

    def _build_rankings(self):
        """Build per-state rankings from accumulated stats."""
        self.r3_updates += 1
        self.att_updates_total += 1

        # Build ranking for each observed coarse state
        for s in range(N_COARSE_STATES):
            action_avgs = []
            for a in range(min(self._n_actions_env, N_KB)):
                key = (s, a)
                if key in self._state_action_change_sum:
                    count = max(self._state_action_change_count.get(key, 1), 1)
                    avg = self._state_action_change_sum[key] / count
                    action_avgs.append((avg, a))

            if action_avgs:
                action_avgs.sort(reverse=True)
                ranked = [a for _, a in action_avgs if _ > CHANGE_THRESH]
                if ranked:
                    self._state_ranked_actions[s] = ranked
                    continue

            # Fallback: use global stats
            global_avgs = []
            for a, total in self._action_change_sum.items():
                count = max(self._action_change_count.get(a, 1), 1)
                global_avgs.append((total / count, a))

            if global_avgs:
                global_avgs.sort(reverse=True)
                ranked = [a for _, a in global_avgs if _ > CHANGE_THRESH]
                if ranked:
                    self._state_ranked_actions[s] = ranked
                    continue

            # Final fallback
            n_kb = min(self._n_actions_env, N_KB)
            self._state_ranked_actions[s] = list(range(n_kb))

    def _transition_to_exploit(self):
        self._exploring = False
        self._build_rankings()
        self._current_idx = 0

    def _get_ranked_actions(self, coarse_state):
        """Get ranked actions for current coarse state."""
        if coarse_state in self._state_ranked_actions:
            return self._state_ranked_actions[coarse_state]
        # Fallback
        n_kb = min(self._n_actions_env, N_KB)
        return list(range(n_kb))

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)
        coarse = _coarse_state(enc)

        if self._prev_enc is None:
            self._prev_enc = enc.copy()
            self._prev_coarse = coarse
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        # Measure change
        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # Record stats for previous action in BOTH state-conditioned and global
        a = self._prev_action
        s = self._prev_coarse

        key = (s, a)
        self._state_action_change_sum[key] = self._state_action_change_sum.get(key, 0.0) + delta
        self._state_action_change_count[key] = self._state_action_change_count.get(key, 0) + 1

        self._action_change_sum[a] = self._action_change_sum.get(a, 0.0) + delta
        self._action_change_count[a] = self._action_change_count.get(a, 0) + 1

        # === EXPLORE PHASE ===
        if self._exploring:
            if self.step_count >= EXPLORE_STEPS:
                self._transition_to_exploit()
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_coarse = coarse
                self._prev_action = action
                return action

        # Periodic rebuild
        if self.step_count % 500 == 0:
            self._build_rankings()
            self._current_idx = 0

        # === EXPLOIT: v80-style cycling with state-conditioned rankings ===
        if self._rng.random() < EPSILON:
            action = int(self._rng.randint(0, self._n_actions_env))
            self._prev_enc = enc.copy()
            self._prev_coarse = coarse
            self._prev_action = action
            return action

        ranked = self._get_ranked_actions(coarse)
        # Clamp index
        self._current_idx = self._current_idx % len(ranked)

        self._current_change_sum += delta
        self._current_hold_count += 1

        current_avg = self._current_change_sum / max(self._current_hold_count, 1)
        if current_avg < CHANGE_THRESH and self._current_hold_count > 5:
            self._current_idx = (self._current_idx + 1) % len(ranked)
            self._current_change_sum = 0.0
            self._current_hold_count = 0
            self._patience = 0
        elif self._current_hold_count > 20:
            self._patience += 1
            if self._patience > 3:
                self._current_idx = (self._current_idx + 1) % len(ranked)
                self._current_change_sum = 0.0
                self._current_hold_count = 0
                self._patience = 0

        action = ranked[self._current_idx]
        self._prev_enc = enc.copy()
        self._prev_coarse = coarse
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._prev_enc = None
        self._current_idx = 0
        self._current_change_sum = 0.0
        self._current_hold_count = 0
        self._patience = 0
        # Keep state-conditioned stats across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "epsilon": EPSILON,
    "change_thresh": CHANGE_THRESH,
    "explore_steps": EXPLORE_STEPS,
    "n_coarse_states": N_COARSE_STATES,
    "family": "state-conditioned ranking",
    "tag": "ATTACK v93 (ℓ_π state-conditioned: v80 recipe but per-(quadrant, action) change stats. 4 coarse states from quadrant max activation. Tests 800b theorem prediction: state-conditioned rankings > global rankings.)",
}

SUBSTRATE_CLASS = StateConditionedSubstrate
