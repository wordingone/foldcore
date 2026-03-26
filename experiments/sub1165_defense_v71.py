"""
sub1165_defense_v71.py — Entropy-reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1165 --substrate experiments/sub1165_defense_v71.py

FAMILY: Entropy-reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: ALL prior defense substrates use distance-to-initial as the
progress metric. This assumes "closer to start = better." But for many
games, progress means ORGANIZING the screen — sorting elements, completing
patterns, matching colors. The initial state is disordered (high entropy)
and the goal is ordered (low entropy).

This substrate uses ENTROPY as the progress metric:
- Compute Shannon entropy of the 256D encoded observation
- H = -sum(p_i * log(p_i)) where p_i is normalized frequency of enc values
- Progress = decreasing entropy (screen becoming more organized)
- Reactive cycling: switch action when entropy stops decreasing

FUNDAMENTALLY DIFFERENT from all prior substrates:
- v30: dist-to-initial (spatial metric — how FAR from start?)
- v59: empowerment (controllability — how DISTINGUISHABLE are actions?)
- v71: entropy (organization — how ORDERED is the screen?)

These are three independent axes of "progress." Entropy captures games
where the goal is ORDER, not proximity to start.

ZERO learned parameters (defense: ℓ₁). Fixed entropy computation.

KILL: ARC ≤ v30.
SUCCESS: Entropy-reactive solves games that distance-reactive misses.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 8
N_BINS = 16  # histogram bins for entropy computation


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _compute_entropy(enc, n_bins=N_BINS):
    """Shannon entropy of encoded observation histogram."""
    # Bin the 256 values into n_bins buckets
    enc_min, enc_max = enc.min(), enc.max()
    if enc_max - enc_min < 1e-8:
        return 0.0  # uniform screen = zero entropy
    bins = np.linspace(enc_min, enc_max, n_bins + 1)
    hist, _ = np.histogram(enc, bins=bins)
    # Normalize to probability distribution
    total = hist.sum()
    if total == 0:
        return 0.0
    probs = hist / total
    probs = probs[probs > 0]  # remove zeros for log
    return -float(np.sum(probs * np.log2(probs)))


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class EntropyReactiveSubstrate:
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
        self._prev_entropy = float('inf')
        self._prev_dist = float('inf')  # also track dist for comparison

        self._n_active = N_KB

        # Reactive state
        self._current_action = 0
        self._patience = 0

        # Click regions
        self._click_actions = []
        self._regions_set = False

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _discover_regions(self, enc):
        screen_mean = enc.mean()
        saliency = np.abs(enc - screen_mean)
        sorted_blocks = np.argsort(saliency)[::-1]
        click_regions = list(sorted_blocks[:N_CLICK_REGIONS].astype(int))
        self._click_actions = [_block_to_click_action(b) for b in click_regions]
        if self._has_clicks:
            self._n_active = N_KB + N_CLICK_REGIONS
        else:
            self._n_active = min(self._n_actions_env, N_KB)
        self._regions_set = True

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

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
            self._prev_entropy = _compute_entropy(enc)
            self._discover_regions(enc)
            return 0

        entropy = _compute_entropy(enc)

        # Reactive cycling based on ENTROPY (not distance)
        # Improvement = entropy decreased (screen more organized)
        if entropy >= self._prev_entropy:
            # No improvement — try next action
            self._current_action = (self._current_action + 1) % self._n_active
            self._patience = 0
        else:
            self._patience += 1
            if self._patience > 10:
                self._patience = 0
                self._current_action = (self._current_action + 1) % self._n_active

        self._prev_entropy = entropy
        self._prev_enc = enc.copy()
        return self._idx_to_env_action(self._current_action)

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_entropy = float('inf')
        self._prev_dist = float('inf')
        self._current_action = 0
        self._patience = 0


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "n_bins": N_BINS,
    "family": "entropy-reactive",
    "tag": "defense v71 (ℓ₁ entropy-reactive: minimize Shannon entropy of observation. Tests whether progress = organization, not proximity to initial state.)",
}

SUBSTRATE_CLASS = EntropyReactiveSubstrate
