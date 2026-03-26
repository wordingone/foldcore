"""
sub1151_defense_v58.py — Periodic alternation scan (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1151 --substrate experiments/sub1151_defense_v58.py

FAMILY: Periodic interaction scan (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: ALL 50 prior substrates execute actions ONE-AT-A-TIME or in
short sequences. Some games may respond to PERIODIC interaction — rapid
alternation between two actions creating a rhythmic signal. If a game
responds to frequency (like a rhythm game or toggle mechanism), alternation
at the right frequency would produce pixel change where individual actions
don't. A pair (a,b) that produces change under rapid alternation where
neither a nor b individually does → periodic interaction is the bottleneck.

Architecture:
- Phase 1 (steps 0-50): keyboard scan, record per-action change (baseline)
- Phase 2 (steps 50-~750): periodic alternation scan
  - For each unique pair {a, b} where a < b ∈ {0..6} (keyboard only):
    - Alternate a,b,a,b,... for 30 steps, record pixel change per step
  - 7C2 = 21 pairs × 30 steps = 630 steps
  - Also test with clicks: 7 keyboard × top-4 click = 28 pairs × 30 = 840
  - Compare: alternation change rate vs individual change rates
    - If alt_change > max(individual_a, individual_b) → PERIODIC SIGNAL
- Phase 3 (remaining): exploit periodic patterns
  - Repeat highest-response alternation pairs

ZERO learned parameters (defense: ℓ₁). Fixed scan protocol.

KILL: No periodic signals found AND ARC ≤ v30.
SUCCESS: Periodic interaction discovered → frequency-based mechanics exist.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7
N_CLICK_REGIONS = 16
KB_SCAN_END = 50
ALT_STEPS = 30  # steps per alternation test
MAX_CLICK_PAIRS = 4  # top-4 click regions for pairing


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class PeriodicScanSubstrate:
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

        # Phase 1: individual action change
        self._individual_change = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.float32)
        self._individual_counts = np.zeros(N_KB + N_CLICK_REGIONS, dtype=np.int32)

        # Phase 2: alternation scan
        self._alt_pairs = []  # list of (a, b) pairs to test
        self._alt_pair_idx = 0
        self._alt_step = 0  # step within current alternation
        self._alt_total_change = 0.0  # accumulated change during alternation
        self._alt_results = {}  # (a, b) → avg change per step
        self._scanning = True

        # Phase 3: exploit
        self._best_pairs = []  # sorted by alternation response
        self._exploit_pair_idx = 0
        self._exploit_patience = 0

        # Click regions
        self._click_actions = []
        self._n_active = N_KB
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

        # Build alternation pair list: all keyboard pairs (a < b)
        self._alt_pairs = []
        n_kb = min(self._n_actions_env, N_KB)
        for a in range(n_kb):
            for b in range(a + 1, n_kb):
                self._alt_pairs.append((a, b))
        # Same-action pairs (a, a) — test if rapid repeating matters
        for a in range(n_kb):
            self._alt_pairs.append((a, a))
        # Keyboard-click pairs
        if self._has_clicks:
            for a in range(n_kb):
                for ci in range(min(MAX_CLICK_PAIRS, len(self._click_actions))):
                    self._alt_pairs.append((a, N_KB + ci))

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _build_exploit_set(self):
        """Find alternation pairs with highest response."""
        self._scanning = False
        response_list = []

        for (a, b), avg_change in self._alt_results.items():
            # Compare to individual baselines
            ind_a = self._individual_change[a] / max(self._individual_counts[a], 1)
            ind_b = self._individual_change[b] / max(self._individual_counts[b], 1)
            baseline = max(ind_a, ind_b)
            response_list.append((avg_change, baseline, a, b))

        response_list.sort(reverse=True)
        self._best_pairs = [(a, b) for _, _, a, b in response_list[:12]]
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

        if self._enc_0 is None:
            self._enc_0 = enc.copy()
            self._prev_enc = enc.copy()
            self._discover_regions(enc)
            return 0

        change = np.sum(np.abs(enc - self._prev_enc))

        # Phase 1: individual action scan
        if self.step_count <= KB_SCAN_END:
            action_idx = (self.step_count - 1) % self._n_active
            prev_action = (self.step_count - 2) % self._n_active
            self._individual_change[prev_action] += change
            self._individual_counts[prev_action] += 1
            self._prev_enc = enc.copy()
            return self._idx_to_env_action(action_idx)

        # Phase 2: alternation scan
        if self._scanning and self._alt_pair_idx < len(self._alt_pairs):
            a, b = self._alt_pairs[self._alt_pair_idx]

            # Accumulate change during alternation
            self._alt_total_change += change
            self._alt_step += 1

            if self._alt_step >= ALT_STEPS:
                # Record average change per step for this pair
                self._alt_results[(a, b)] = self._alt_total_change / ALT_STEPS
                self._alt_pair_idx += 1
                self._alt_step = 0
                self._alt_total_change = 0.0

            # Alternate: even steps → a, odd steps → b
            if self._alt_step % 2 == 0:
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a)
            else:
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(b)

        # Transition from scan to exploit
        if self._scanning:
            self._build_exploit_set()

        # Phase 3: exploit best alternation pairs
        if self._best_pairs:
            a, b = self._best_pairs[self._exploit_pair_idx]
            self._exploit_patience += 1

            if self._exploit_patience >= ALT_STEPS * 3:  # 90 steps per pair
                self._exploit_patience = 0
                self._exploit_pair_idx = (self._exploit_pair_idx + 1) % len(self._best_pairs)

            # Alternate
            if self._exploit_patience % 2 == 0:
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a)
            else:
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(b)

        # Fallback: keyboard cycling
        n_active = min(self._n_actions_env, N_KB)
        action = (self.step_count // 5) % n_active
        self._prev_enc = enc.copy()
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._alt_step = 0
        self._alt_total_change = 0.0
        self._exploit_pair_idx = 0
        self._exploit_patience = 0
        # Keep scan results across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "kb_scan_end": KB_SCAN_END,
    "alt_steps": ALT_STEPS,
    "family": "periodic interaction scan",
    "tag": "defense v58 (ℓ₁ periodic alternation scan: rapid a,b,a,b... for 30 steps per pair, exploit highest-response)",
}

SUBSTRATE_CLASS = PeriodicScanSubstrate
