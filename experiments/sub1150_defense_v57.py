"""
sub1150_defense_v57.py — 2-action sequence scan (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1150 --substrate experiments/sub1150_defense_v57.py

FAMILY: Action sequence scan (NEW). Tagged: defense (ℓ₁).
R3 HYPOTHESIS: ALL 49 prior substrates test INDIVIDUAL actions. Some games
may require ACTION SEQUENCES (e.g., press key1 THEN key2) to trigger any
response. A 2-action sequence that produces pixel change where neither
individual action does → sequential interaction is the bottleneck.

Architecture:
- Phase 1 (steps 0-50): keyboard scan, record per-action change (baseline)
- Phase 2 (steps 50-~1100): systematic 2-action sequence scan
  - For each pair (a1, a2) where a1, a2 ∈ {0..6} (keyboard only):
    - Execute a1, then a2, record total pixel change across both steps
  - 7×7 = 49 pairs × 2 steps each = 98 steps (plus overhead)
  - Also test keyboard→click sequences if game has clicks:
    - 7 keyboard × 16 clicks = 112 pairs × 2 steps = 224 steps
  - Compare: sequence change vs sum of individual changes
    - If sequence_change > individual_sum → SYNERGY (sequence required)
- Phase 3 (remaining): exploit synergistic sequences
  - Repeat the highest-synergy 2-sequences with reactive switching

ZERO learned parameters (defense: ℓ₁). Fixed scan protocol.

KILL: No synergistic sequences found AND ARC ≤ v30.
SUCCESS: Synergistic sequences discovered → sequential interaction is the key.
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
MAX_PATIENCE = 20


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


def _block_to_click_action(block_idx):
    by = block_idx // N_BLOCKS
    bx = block_idx % N_BLOCKS
    px = bx * BLOCK_SIZE + BLOCK_SIZE // 2
    py = by * BLOCK_SIZE + BLOCK_SIZE // 2
    return N_KB + px + py * 64


class SequenceScanSubstrate:
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

        # Phase 2: sequence scan
        self._sequences = []  # list of (a1, a2) pairs to test
        self._sequence_idx = 0
        self._sequence_step = 0  # 0 = execute a1, 1 = execute a2
        self._sequence_changes = {}  # (a1, a2) → total change
        self._pre_seq_enc = None  # encoding before sequence started
        self._scanning = True

        # Phase 3: exploit
        self._synergistic_seqs = []  # sorted by synergy
        self._exploit_seq_idx = 0
        self._exploit_step = 0  # 0 or 1 within current sequence
        self._exploit_patience = 0
        self._exploit_patience_max = 10

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

        # Build sequence list: all keyboard pairs first
        self._sequences = []
        for a1 in range(min(self._n_actions_env, N_KB)):
            for a2 in range(min(self._n_actions_env, N_KB)):
                self._sequences.append((a1, a2))
        # Then keyboard→click pairs (if clicks available)
        if self._has_clicks:
            for a1 in range(min(self._n_actions_env, N_KB)):
                for ci, ca in enumerate(self._click_actions[:8]):  # top-8 clicks only
                    self._sequences.append((a1, N_KB + ci))

    def _idx_to_env_action(self, idx):
        if idx < N_KB:
            return idx
        click_idx = idx - N_KB
        if click_idx < len(self._click_actions):
            return self._click_actions[click_idx]
        return self._rng.randint(min(self._n_actions_env, N_KB))

    def _build_exploit_set(self):
        """Find sequences with synergy: change > sum of individual changes."""
        self._scanning = False
        synergy_list = []

        for (a1, a2), seq_change in self._sequence_changes.items():
            ind1 = self._individual_change[a1] / max(self._individual_counts[a1], 1)
            ind2 = self._individual_change[a2] / max(self._individual_counts[a2], 1)
            expected = ind1 + ind2
            synergy = seq_change - expected
            if seq_change > 0.5:  # minimum change threshold
                synergy_list.append((synergy, seq_change, a1, a2))

        synergy_list.sort(reverse=True)
        self._synergistic_seqs = [(a1, a2) for _, _, a1, a2 in synergy_list[:16]]
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
            return 0  # first action

        change = np.sum(np.abs(enc - self._prev_enc))

        # Phase 1: individual action scan
        if self.step_count <= KB_SCAN_END:
            action_idx = (self.step_count - 1) % self._n_active
            # Record change from previous action
            prev_action = (self.step_count - 2) % self._n_active
            self._individual_change[prev_action] += change
            self._individual_counts[prev_action] += 1
            self._prev_enc = enc.copy()
            return self._idx_to_env_action(action_idx)

        # Phase 2: sequence scan
        if self._scanning and self._sequence_idx < len(self._sequences):
            a1, a2 = self._sequences[self._sequence_idx]

            if self._sequence_step == 0:
                # About to execute a1 — record pre-sequence encoding
                self._pre_seq_enc = enc.copy()
                self._sequence_step = 1
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a1)
            else:
                # Just executed a1, now execute a2
                # Record change from a1
                change_after_a1 = np.sum(np.abs(enc - self._pre_seq_enc))
                self._sequence_step = 0
                self._sequence_idx += 1

                # We'll record total sequence change on the NEXT step
                # For now, store partial
                self._sequence_changes[(a1, a2)] = change_after_a1

                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a2)

        # Transition from scan to exploit
        if self._scanning:
            # Record final sequence change
            if self._sequence_idx > 0 and self._sequence_idx <= len(self._sequences):
                last_a1, last_a2 = self._sequences[self._sequence_idx - 1]
                total_change = np.sum(np.abs(enc - self._pre_seq_enc)) if self._pre_seq_enc is not None else 0
                self._sequence_changes[(last_a1, last_a2)] = total_change

            self._build_exploit_set()

        # Phase 3: exploit synergistic sequences
        if self._synergistic_seqs:
            a1, a2 = self._synergistic_seqs[self._exploit_seq_idx]

            if self._exploit_step == 0:
                self._exploit_step = 1
                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a1)
            else:
                self._exploit_step = 0
                self._exploit_patience += 1

                if self._exploit_patience >= self._exploit_patience_max:
                    self._exploit_patience = 0
                    self._exploit_seq_idx = (self._exploit_seq_idx + 1) % len(self._synergistic_seqs)

                self._prev_enc = enc.copy()
                return self._idx_to_env_action(a2)

        # Fallback: keyboard cycling
        n_active = min(self._n_actions_env, N_KB)
        action = (self.step_count // 5) % n_active
        self._prev_enc = enc.copy()
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._pre_seq_enc = None
        self._sequence_step = 0  # reset mid-sequence state
        self._exploit_seq_idx = 0
        self._exploit_step = 0
        self._exploit_patience = 0
        # Keep scan results and synergistic sequences across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "n_click_regions": N_CLICK_REGIONS,
    "kb_scan_end": KB_SCAN_END,
    "family": "action sequence scan",
    "tag": "defense v57 (ℓ₁ 2-action sequence scan: test all keyboard pairs + top-8 click combos, exploit synergistic sequences)",
}

SUBSTRATE_CLASS = SequenceScanSubstrate
