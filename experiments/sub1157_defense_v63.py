"""
sub1157_defense_v63.py — Action-space-adaptive reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1157 --substrate experiments/sub1157_defense_v63.py

FAMILY: Action-space-adaptive reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: set_game(n_actions) provides ONE piece of metadata:
the action space size. Games with n_actions=7 (keyboard-only) behave
differently from games with n_actions>7 (click-capable). What if the
substrate uses DIFFERENT strategies for each type?

- Keyboard-only (n_actions≤7): pure reactive argmin (v30 behavior).
  These games have small discrete action spaces — reactive cycling works.
- Click-capable (n_actions>7): empowerment-based click targeting.
  These games have huge click spaces — need to identify responsive regions.

This is the first substrate to explicitly branch on action space type.
Both branches use fixed protocols (ℓ₁). The branching rule itself is fixed.

Architecture:
- if n_actions ≤ 7: v30-style reactive argmin over keyboard actions
- if n_actions > 7: empowerment scan over 16 click regions + keyboard,
  then exploit top-4 distinguishable actions with reactive switching

ZERO learned parameters (defense: ℓ₁). Fixed branching + fixed protocols.

KILL: ARC ≤ v30.
SUCCESS: Action-space-adaptive strategy > uniform strategy.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
SAMPLES_PER_ACTION = 30
DISTINGUISHABILITY_EPS = 1e-6


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class AdaptiveReactiveSubstrate:
    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions_env = N_KB
        self._has_clicks = False
        self._game_number = 0
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')

        # Mode: 'keyboard' or 'click'
        self._mode = 'keyboard'
        self._n_active = N_KB

        # Keyboard reactive state
        self._kb_action = 0
        self._kb_patience = 0

        # Click empowerment state
        self._click_actions = []
        self._estimating = False
        self._est_action_idx = 0
        self._est_sample_count = 0
        self._action_means = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_m2 = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_vars = np.zeros((N_KB + N_CLICK_REGIONS, N_DIMS), dtype=np.float64)
        self._action_counts = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.int32)
        self._empowerment = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)
        self._best_actions = []
        self._exploit_idx = 0
        self._exploit_patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

        # Branch on action space
        if self._has_clicks:
            self._mode = 'click'
            self._estimating = True
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._mode = 'keyboard'
            self._n_active = min(n_actions, N_KB)

    def _discover_regions(self, enc):
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _update_action_stats(self, action_idx, enc):
        n = self._action_counts[action_idx] + 1
        self._action_counts[action_idx] = n
        delta = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_means[action_idx] += delta / n
        delta2 = enc.astype(np.float64) - self._action_means[action_idx]
        self._action_m2[action_idx] += delta * delta2
        if n > 1:
            self._action_vars[action_idx] = self._action_m2[action_idx] / (n - 1)

    def _compute_empowerment(self):
        self._estimating = False
        self.r3_updates += 1
        self.att_updates_total += 1

        for a in range(self._n_active):
            if self._action_counts[a] < 2:
                continue
            total_dist = 0.0
            n_comp = 0
            for b in range(self._n_active):
                if b == a or self._action_counts[b] < 2:
                    continue
                mean_diff = self._action_means[a] - self._action_means[b]
                pooled_var = (self._action_vars[a] + self._action_vars[b]) / 2.0 + DISTINGUISHABILITY_EPS
                d_sq = np.sum(mean_diff ** 2 / pooled_var)
                total_dist += np.sqrt(d_sq)
                n_comp += 1
            if n_comp > 0:
                self._empowerment[a] = total_dist / n_comp

        scored = [(self._empowerment[a], a) for a in range(self._n_active)]
        scored.sort(reverse=True)
        self._best_actions = [a for emp, a in scored[:6] if emp > 0.01]
        if not self._best_actions:
            self._best_actions = list(range(min(self._n_actions_env, N_KB)))

    def _keyboard_reactive(self, enc):
        """v30-style reactive: cycle actions, switch on distance improvement."""
        dist = np.sum(np.abs(enc - self._enc_0))

        if dist >= self._prev_dist:
            # No improvement — try next action
            self._kb_action = (self._kb_action + 1) % self._n_active
            self._kb_patience = 0
        else:
            self._kb_patience += 1
            if self._kb_patience > 10:
                # Stuck improving — try switching anyway
                self._kb_patience = 0
                self._kb_action = (self._kb_action + 1) % self._n_active

        self._prev_dist = dist
        return self._kb_action

    def process(self, obs: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 3 and obs.shape[0] == 1:
            obs = obs[0]
        if obs.shape != (64, 64):
            return int(self._rng.randint(0, min(self._n_actions_env, N_KB)))

        self.step_count += 1
        enc = _obs_to_enc(obs)

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            if self._mode == 'click':
                self._discover_regions(enc)
            return 0

        # KEYBOARD MODE: pure reactive
        if self._mode == 'keyboard':
            action = self._keyboard_reactive(enc)
            self._prev_enc = enc.copy()
            return action

        # CLICK MODE: empowerment estimation then exploit
        if self._estimating:
            self._update_action_stats(self._est_action_idx, enc)
            self._est_sample_count += 1

            if self._est_sample_count >= SAMPLES_PER_ACTION:
                self._est_sample_count = 0
                self._est_action_idx += 1
                if self._est_action_idx >= self._n_active:
                    self._compute_empowerment()
                    self._prev_enc = enc.copy()
                    if self._best_actions:
                        return self._idx_to_env_action(self._best_actions[0])
                    return 0

            self._prev_enc = enc.copy()
            return self._idx_to_env_action(self._est_action_idx)

        # Click exploit: cycle through best empowered actions with reactive switching
        dist = np.sum(np.abs(enc - self._enc_0))
        a = self._best_actions[self._exploit_idx]
        self._exploit_patience += 1

        if dist >= self._prev_dist:
            self._exploit_patience += 1

        if self._exploit_patience >= 15:
            self._exploit_patience = 0
            self._exploit_idx = (self._exploit_idx + 1) % len(self._best_actions)

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        return self._idx_to_env_action(a)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._kb_action = 0
        self._kb_patience = 0
        self._exploit_idx = 0
        self._exploit_patience = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "samples_per_action": SAMPLES_PER_ACTION,
    "family": "action-space-adaptive reactive",
    "tag": "defense v63 (ℓ₁ adaptive: keyboard→reactive argmin, click→empowerment scan+exploit. Branch on n_actions metadata.)",
}

SUBSTRATE_CLASS = AdaptiveReactiveSubstrate
