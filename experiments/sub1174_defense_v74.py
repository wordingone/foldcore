"""
sub1174_defense_v74.py — Sequence-reactive (defense: ℓ₁)

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1174 --substrate experiments/sub1174_defense_v74.py

FAMILY: Sequence reactive. Tagged: defense (ℓ₁).
R3 HYPOTHESIS: All substrates try SINGLE actions and measure response.
But some games may require ACTION SEQUENCES (2-3 actions in order) to
produce any pixel change. A game with a "confirm" button needs:
action A (select) → action B (confirm). Neither alone produces change.

This substrate tests whether the 0% wall is a SEQUENCE DISCOVERY problem:
- Phase 1: try all single KB actions + random clicks (like v73)
- Phase 2: for actions that produced NO change, try 2-action PAIRS
  (A then B, measuring change after B)
- Exploit: reactive cycling over responsive singles AND pairs

If the 0% wall games require sequences, this should break at least one.
If sequences don't help, the 0% wall is truly about UNDISCOVERABLE targets
(not about action complexity).

DIFFERENT from all prior substrates:
- All prior: single action → measure change → decide
- v74: action PAIRS as atomic units → measure change → decide

ZERO learned parameters (defense: ℓ₁). Fixed sequence testing protocol.

KILL: No improvement over v73/random on 0% games.
SUCCESS: Any 0% game responds to a sequence that doesn't respond to singles.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

BLOCK_SIZE = 4
N_BLOCKS = 16
N_DIMS = N_BLOCKS * N_BLOCKS  # 256
N_KB = 7

SINGLE_EXPLORE = 100       # steps for single-action exploration
PAIR_EXPLORE = 200          # steps for pair exploration
RESPONSE_THRESH = 0.3       # min pixel change to count as responsive
MAX_RESPONSIVE = 20         # max responsive actions/pairs to track


def _obs_to_enc(obs):
    """avgpool4: 64x64 -> 16x16 = 256D."""
    return obs.reshape(N_BLOCKS, BLOCK_SIZE, N_BLOCKS, BLOCK_SIZE).mean(axis=(1, 3)).ravel().astype(np.float32)


class SequenceReactiveSubstrate:
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

        # Phase tracking
        self._phase = 'single'  # 'single' -> 'pair' -> 'exploit'
        self._prev_action = 0

        # Single exploration
        self._responsive_singles = {}  # action -> max_change
        self._unresponsive_singles = set()

        # Pair exploration
        self._pair_step = 0       # 0 = first action, 1 = second action
        self._pair_first = 0      # first action of current pair
        self._pair_enc_before = None  # enc before pair started
        self._responsive_pairs = []  # [(a1, a2, change), ...]

        # Exploit phase
        self._exploit_sequence = []  # list of (action,) or (a1, a2) tuples
        self._exploit_idx = 0
        self._exploit_sub_idx = 0   # position within current sequence
        self._exploit_patience = 0

        if not hasattr(self, 'r3_updates'):
            self.r3_updates = 0
            self.att_updates_total = 0

    def set_game(self, n_actions: int):
        self._game_number += 1
        self._n_actions_env = n_actions
        self._has_clicks = n_actions > N_KB
        self._init_state()

    def _transition_to_pair(self):
        """After single exploration, test pairs of unresponsive actions."""
        self._phase = 'pair'
        self._pair_step = 0
        self.r3_updates += 1

    def _transition_to_exploit(self):
        """Build exploit sequence from responsive singles + pairs."""
        self._phase = 'exploit'
        self.att_updates_total += 1

        self._exploit_sequence = []

        # Add responsive singles (sorted by change)
        if self._responsive_singles:
            sorted_singles = sorted(
                self._responsive_singles.items(),
                key=lambda x: -x[1]
            )[:MAX_RESPONSIVE]
            for a, _ in sorted_singles:
                self._exploit_sequence.append((a,))

        # Add responsive pairs (sorted by change)
        if self._responsive_pairs:
            sorted_pairs = sorted(
                self._responsive_pairs,
                key=lambda x: -x[2]
            )[:MAX_RESPONSIVE // 2]
            for a1, a2, _ in sorted_pairs:
                self._exploit_sequence.append((a1, a2))

        # Fallback: keyboard cycling
        if not self._exploit_sequence:
            n_kb = min(self._n_actions_env, N_KB)
            for a in range(n_kb):
                self._exploit_sequence.append((a,))

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
            self._prev_action = int(self._rng.randint(0, self._n_actions_env))
            return self._prev_action

        delta = float(np.sum(np.abs(enc - self._prev_enc)))

        # === SINGLE EXPLORATION PHASE ===
        if self._phase == 'single':
            # Record response of previous action
            if delta > RESPONSE_THRESH:
                prev = self._prev_action
                if prev in self._responsive_singles:
                    self._responsive_singles[prev] = max(
                        self._responsive_singles[prev], delta
                    )
                else:
                    self._responsive_singles[prev] = delta
            else:
                self._unresponsive_singles.add(self._prev_action)

            if self.step_count >= SINGLE_EXPLORE:
                self._transition_to_pair()
                # Fall through to pair phase
            else:
                action = int(self._rng.randint(0, self._n_actions_env))
                self._prev_enc = enc.copy()
                self._prev_action = action
                return action

        # === PAIR EXPLORATION PHASE ===
        if self._phase == 'pair':
            if self._pair_step == 0:
                # Starting a new pair: pick first action
                # Preferentially test KB actions (likely game controls)
                n_kb = min(self._n_actions_env, N_KB)
                self._pair_first = int(self._rng.randint(0, n_kb))
                self._pair_enc_before = enc.copy()
                self._pair_step = 1
                self._prev_enc = enc.copy()
                self._prev_action = self._pair_first
                return self._pair_first
            else:
                # Second action of pair: measure change from before pair
                pair_change = float(np.sum(np.abs(enc - self._pair_enc_before)))
                if pair_change > RESPONSE_THRESH:
                    # This pair produced change!
                    second = int(self._rng.randint(0, self._n_actions_env))
                    self._responsive_pairs.append(
                        (self._pair_first, self._prev_action, pair_change)
                    )
                # Pick second action for a new pair test
                second = int(self._rng.randint(0, self._n_actions_env))
                self._pair_step = 0

                if self.step_count >= SINGLE_EXPLORE + PAIR_EXPLORE:
                    self._transition_to_exploit()
                    # Fall through to exploit
                else:
                    self._prev_enc = enc.copy()
                    self._prev_action = second
                    return second

        # === EXPLOIT PHASE: reactive cycling over responsive actions/pairs ===
        dist = float(np.sum(np.abs(enc - self._enc_0)))

        if not self._exploit_sequence:
            self._transition_to_exploit()

        current_seq = self._exploit_sequence[self._exploit_idx]

        # Execute current position in sequence
        if self._exploit_sub_idx < len(current_seq):
            action = current_seq[self._exploit_sub_idx]
            self._exploit_sub_idx += 1

            # If sequence complete, evaluate
            if self._exploit_sub_idx >= len(current_seq):
                self._exploit_sub_idx = 0
                # Reactive: check progress
                if dist >= self._prev_dist:
                    self._exploit_idx = (self._exploit_idx + 1) % len(self._exploit_sequence)
                    self._exploit_patience = 0
                else:
                    self._exploit_patience += 1
                    if self._exploit_patience > 10:
                        self._exploit_patience = 0
                        self._exploit_idx = (self._exploit_idx + 1) % len(self._exploit_sequence)
        else:
            action = current_seq[0]
            self._exploit_sub_idx = 1 if len(current_seq) > 1 else 0

        self._prev_dist = dist
        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        self._enc_0 = None
        self._prev_enc = None
        self._prev_dist = float('inf')
        self._exploit_idx = 0
        self._exploit_sub_idx = 0
        self._exploit_patience = 0
        # Keep discovered actions/pairs across levels


CONFIG = {
    "n_dims": N_DIMS,
    "block_size": BLOCK_SIZE,
    "single_explore": SINGLE_EXPLORE,
    "pair_explore": PAIR_EXPLORE,
    "response_thresh": RESPONSE_THRESH,
    "max_responsive": MAX_RESPONSIVE,
    "family": "sequence reactive",
    "tag": "defense v74 (ℓ₁ sequence-reactive: tests if 0% wall games need ACTION PAIRS (A then B) rather than single actions. Random singles + pair exploration → reactive exploitation.)",
}

SUBSTRATE_CLASS = SequenceReactiveSubstrate
