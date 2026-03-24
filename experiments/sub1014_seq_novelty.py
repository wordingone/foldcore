"""
sub1014_seq_novelty.py — Sequence Novelty Learner (Direction 1, game-agnostic base).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1014 --substrate experiments/sub1014_seq_novelty.py

FAMILY: sequence-novelty (new)
R3 HYPOTHESIS: The substrate discovers productive ACTION SEQUENCES from observation novelty.
  R3 on action ordering — modifies WHICH sequences to try based on which produced novel
  observations. Sequences of K actions that produce novel observations score high; the
  substrate learns to extend these sequences. This addresses the universal sequencing gap
  that 800b/coverage-based mechanisms can't solve.

GAME-AGNOSTIC BASE (Jun directive 2026-03-24):
  NO avgpool16, NO 800b, NO alpha, NO recurrent h.
  Only mechanism: sequence novelty learning.

  Encoding: raw 64x64 flattened → centered (x - running_mean, alpha=0.1)
  Novelty: min L2 distance to last 100 buffered observations
  Sequence score: EMA of novelty achieved when last K actions were taken
  Action: argmax over next-action candidates based on sequence continuations

KILL: FT09=0 AND VC33=0 AND LS20=0
ALIVE: FT09 or VC33 L1 > 0 on ANY seed = unprecedented post-ban
NOTE: LS20 may be 0 (no coverage mechanism). That is acceptable — sequencing test.
BUDGET: 10K steps/game, 10 seeds
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

# ─── Hyperparameters ───
ENC_DIM = 64 * 64                # raw 64x64 flattened
RUNNING_MEAN_ALPHA = 0.1         # encoding centering rate
NOVELTY_BUFFER_MAX = 500         # max observations in novelty buffer
NOVELTY_WINDOW = 100             # compare against last N buffered obs
NOVELTY_THRESHOLD_QUANTILE = 0.5 # add to buffer if novelty > median recent novelty
SEQ_LEN = 3                      # K: sequence length for action history
SEQ_ALPHA = 0.1                  # EMA rate for sequence score updates
EPS = 0.30                       # random exploration rate

CONFIG = {
    "ENC_DIM": ENC_DIM,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA,
    "NOVELTY_BUFFER_MAX": NOVELTY_BUFFER_MAX,
    "SEQ_LEN": SEQ_LEN,
    "SEQ_ALPHA": SEQ_ALPHA,
    "EPS": EPS,
}


class SequenceNoveltySubstrate:
    """
    Sequence Novelty Learner — game-agnostic base.

    Core mechanism:
    1. Encode obs as raw flattened + centered (NO avgpool16)
    2. Compute novelty = min L2 distance to recent buffer
    3. Update sequence_scores[last K actions] += SEQ_ALPHA * (novelty - score)
    4. Select action that extends the highest-scoring known sequence
    5. Epsilon-greedy exploration

    The sequence key is a tuple of the last K action indices — NOT observation-conditioned.
    Works for any game where productive action sequences produce novel observations.
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4

        # Encoding
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)

        # Novelty buffer: list of past enc vectors
        self._novelty_buffer = []

        # Sequence scores: tuple(last K actions) → float score
        self._sequence_scores = {}

        # Action history (last 1000 actions)
        self._action_history = []

        # Recent novelty for adaptive threshold
        self._recent_novelty = []

    def set_game(self, n_actions: int):
        """Reset per-game state. Keep sequence knowledge across games (game-agnostic)."""
        self._n_actions = n_actions
        # Keep running_mean, novelty_buffer, sequence_scores across games
        # (transfer of sequence knowledge is part of the hypothesis)

    def _encode(self, obs):
        """Raw 64x64 flatten + centered running mean. NO avgpool16."""
        arr = np.asarray(obs, dtype=np.float32)
        # Flatten to 64*64 regardless of channels
        if arr.ndim == 3:
            if arr.shape[0] < arr.shape[1]:   # (C, H, W)
                arr = arr.mean(axis=0)
            else:                              # (H, W, C)
                arr = arr.mean(axis=2)
        # Resize/flatten to ENC_DIM
        flat = arr.flatten()
        if len(flat) != ENC_DIM:
            # Interpolate to ENC_DIM if needed
            indices = np.linspace(0, len(flat) - 1, ENC_DIM).astype(int)
            flat = flat[indices]
        # Update running mean + center
        self._running_mean = (
            (1 - RUNNING_MEAN_ALPHA) * self._running_mean + RUNNING_MEAN_ALPHA * flat
        )
        return (flat - self._running_mean).astype(np.float32)

    def _compute_novelty(self, enc):
        """Min L2 distance to recent buffer entries."""
        if not self._novelty_buffer:
            return 1.0
        window = self._novelty_buffer[-NOVELTY_WINDOW:]
        dists = [float(np.linalg.norm(enc - b)) for b in window]
        return min(dists)

    def _update_novelty_buffer(self, enc, novelty):
        """Add enc to buffer if novel enough."""
        if len(self._novelty_buffer) < 50 or novelty > (
            np.median(self._recent_novelty[-50:]) if self._recent_novelty else 0.0
        ):
            self._novelty_buffer.append(enc.copy())
            if len(self._novelty_buffer) > NOVELTY_BUFFER_MAX:
                self._novelty_buffer.pop(0)

    def process(self, obs: np.ndarray) -> int:
        enc = self._encode(obs)
        novelty = self._compute_novelty(enc)

        # Track recent novelty for adaptive threshold
        self._recent_novelty.append(novelty)
        if len(self._recent_novelty) > 200:
            self._recent_novelty.pop(0)

        # Update sequence score for the sequence that PRODUCED this observation
        if len(self._action_history) >= SEQ_LEN:
            seq = tuple(self._action_history[-SEQ_LEN:])
            if seq not in self._sequence_scores:
                self._sequence_scores[seq] = 0.0
            self._sequence_scores[seq] += SEQ_ALPHA * (novelty - self._sequence_scores[seq])

        # Add to novelty buffer
        self._update_novelty_buffer(enc, novelty)

        # ── Action selection ──
        if self._rng.random() < EPS or not self._sequence_scores:
            action = int(self._rng.randint(0, self._n_actions))
        else:
            # Score each candidate next action: does extending history[-K+1:] + [a] score high?
            prefix = tuple(self._action_history[-(SEQ_LEN - 1):])
            scores = []
            for a in range(self._n_actions):
                candidate = prefix + (a,)
                scores.append(self._sequence_scores.get(candidate, 0.0))
            action = int(np.argmax(scores))

        self._action_history.append(action)
        if len(self._action_history) > 1000:
            self._action_history = self._action_history[-1000:]

        return action

    def on_level_transition(self):
        """Clear action history on level transition (new episode = new sequence context)."""
        self._action_history = []


SUBSTRATE_CLASS = SequenceNoveltySubstrate
