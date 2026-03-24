"""
sub1007_attention_traj.py — Attention-over-trajectory substrate.

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1007 --substrate experiments/sub1007_attention_traj.py

FAMILY: Attention-over-trajectory
R3 HYPOTHESIS: softmax(q @ K^T) @ V provides state-conditioned action selection
  without per-state storage, breaking Theorem 4's wall.
  If attention retrieval over a time-indexed buffer selects actions by similarity
  to past states, then FT09/VC33 will show L1 signal (delta-conditioned selection
  works across games), because the substrate retrieves what worked in similar states
  rather than averaging across all states (which dilutes SNR to 0 per Theorem 4).
  Falsified if: FT09 and VC33 remain 0% (same as 800b/994 ceiling).

KILL: chain_score < 1006 baseline (3 phases) OR FT09=0 AND VC33=0
SUCCESS: games_with_signal > 3 (any FT09 or VC33 L1)
BUDGET: 10K steps/game, 10 seeds, all 5 phases (Split-CIFAR x2, LS20, FT09, VC33)

Jun directive 2026-03-24:
- Substrate defines the class. Harness (run_experiment.py) is the constant.
- Game order randomized per seed — cannot be bypassed
- Results saved to chain_results/runs/ — enforced by harness
- chain_kill verdict: PASS / KILL / FAIL vs baseline_994.json

Ban check (Leo, 2026-03-24):
- Buffer is TIME-indexed, not state-action indexed
- No visit counts, no edge dicts, no per-state storage
- Attention COMPUTES state-conditioned preferences at query time
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
ENC_DIM = 256
MAX_BUFFER = 2000
ETA_W = 0.01
EPS = 0.20
TEMP = 1.0  # attention temperature scaling (applied to sqrt(ENC_DIM) denominator)
RUNNING_MEAN_ALPHA = 0.01  # EMA rate for running mean


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


# ─── Substrate ───
class AttentionTrajectorySubstrate:
    """
    Attention over trajectory buffer for state-conditioned action selection.

    Buffer stores (enc_t, outcome_t) pairs indexed by time.
    Query: softmax(K @ q / sqrt(256)) @ V → action_scores → argmax.
    Buffer persists across games (trajectory memory of ALL experience).
    W_pred resets on game switch (different action space dims).

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4

        # Trajectory buffer: time-indexed, persists across games
        # Each entry: (enc, action_idx, delta) — avoids n_actions mismatch across games
        self._buffer_K = []   # list of enc arrays (ENC_DIM,)
        self._buffer_A = []   # list of int action indices
        self._buffer_D = []   # list of float deltas

        # Running mean for centering (global, persists across games)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)
        self._n_obs = 0

        # W_pred (prediction error, resets on game switch)
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)

        self._prev_enc = None
        self._prev_action = None

    def set_game(self, n_actions: int):
        """Called on game switch. Buffer persists; W_pred resets (new action space)."""
        self._n_actions = n_actions
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None

    def _encode(self, obs):
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._n_obs += 1
        a = RUNNING_MEAN_ALPHA
        self._running_mean = (1 - a) * self._running_mean + a * enc_raw
        return (enc_raw - self._running_mean).astype(np.float32)

    def process(self, obs: np.ndarray) -> int:
        enc = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            # Compute delta for previous action
            delta = float(np.linalg.norm(enc - self._prev_enc))

            # Store in time-indexed buffer
            self._buffer_K.append(self._prev_enc.copy())
            self._buffer_A.append(self._prev_action)
            self._buffer_D.append(delta)

            # Cap buffer: drop oldest
            if len(self._buffer_K) > MAX_BUFFER:
                self._buffer_K.pop(0)
                self._buffer_A.pop(0)
                self._buffer_D.pop(0)

            # W_pred update (gradient ascent — keeps errors volatile for alpha calibration)
            pred = self._W_pred @ self._prev_enc
            error = enc - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self._W_pred -= ETA_W * np.outer(error, self._prev_enc)

        # Action selection
        if len(self._buffer_K) > 10 and self._rng.random() >= EPS:
            K = np.stack(self._buffer_K)              # (T, ENC_DIM)
            A = np.array(self._buffer_A, dtype=np.int32)  # (T,)
            D = np.array(self._buffer_D, dtype=np.float32)  # (T,)

            # Attention: which past states look like current?
            scores = K @ enc / (ENC_DIM ** 0.5)       # (T,)
            attn = _softmax(scores)                    # (T,)

            # Retrieve: attention-weighted delta per action (current game only)
            action_scores = np.zeros(self._n_actions, dtype=np.float32)
            weighted_D = attn * D                      # (T,)
            for t in range(len(A)):
                if A[t] < self._n_actions:
                    action_scores[A[t]] += weighted_D[t]
            action = int(np.argmax(action_scores))
        else:
            action = int(self._rng.randint(0, self._n_actions))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        """Buffer persists. Clear prev to avoid cross-episode delta."""
        self._prev_enc = None
        self._prev_action = None


CONFIG = {
    "ENC_DIM": ENC_DIM,
    "MAX_BUFFER": MAX_BUFFER,
    "ETA_W": ETA_W,
    "EPS": EPS,
    "TEMP": TEMP,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA,
}

SUBSTRATE_CLASS = AttentionTrajectorySubstrate
