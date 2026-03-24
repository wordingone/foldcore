"""
sub1009_attn_traj_v2.py — Attention Trajectory v2 (Extraction 5, fixes Step 1007).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1009 --substrate experiments/sub1009_attn_traj_v2.py

FAMILY: attention-trajectory (experiment 2/20)
R3 HYPOTHESIS: State-conditioned retrieval provides temporal credit for sequential actions,
  bypassing Theorem 4's global running-mean SNR collapse.
  State-similar past transitions should predict which actions are useful in the current context.

CHANGES FROM STEP 1007 (root cause fixes):
  1. RUNNING_MEAN_ALPHA = 0.1  (was 0.01 — slow centering starved encoding quality, root cause of 0/10)
  2. MAX_BUFFER = 200           (was 2000 — less noise, 10x faster attention)
  3. TEMP = 0.1                 (was 1.0 — sharper retrieval, decisive selection)
  4. WARMUP = 100               (was 0 — pure random before buffer populates)
  5. BUFFER_FALLBACK = 50       (new — use 800b when buffer too small)
  6. Buffer resets on game switch (was persistent — prevents cross-game noise in v2)
  7. deque instead of list for O(1) append/pop (was O(n) list.pop(0))

KILL: LS20 L1 = 0 on all seeds (bootstrap still broken) AND chain < baseline
ALIVE: LS20 L1 > 0 on any seed = bootstrap fixed
SUCCESS: FT09 or VC33 L1 > 0 = unprecedented
BUDGET: 10K steps/game, 10 seeds, all 5 phases
"""
import sys
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from collections import deque
from substrates.step0674 import _enc_frame

# ─── Hyperparameters ───
ENC_DIM = 256
MAX_BUFFER = 200              # was 2000 — smaller, faster, lower noise
ETA_W = 0.01
EPS = 0.20
TEMP = 0.1                    # was 1.0 — sharper attention
RUNNING_MEAN_ALPHA = 0.1      # was 0.01 — faster encoding adaptation
WARMUP = 100                  # pure random steps before attention activates
BUFFER_FALLBACK = 50          # 800b when buffer size < this
ALPHA_EMA_800B = 0.10         # 800b delta EMA rate (fallback)
INIT_DELTA = 1.0

CONFIG = {
    "ENC_DIM": ENC_DIM,
    "MAX_BUFFER": MAX_BUFFER,
    "ETA_W": ETA_W,
    "EPS": EPS,
    "TEMP": TEMP,
    "RUNNING_MEAN_ALPHA": RUNNING_MEAN_ALPHA,
    "WARMUP": WARMUP,
    "BUFFER_FALLBACK": BUFFER_FALLBACK,
}


def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def _softmax_sel(vals, temp, rng):
    x = np.array(vals, dtype=np.float32) / temp
    x -= x.max()
    e = np.exp(x)
    p = e / (e.sum() + 1e-12)
    return int(rng.choice(len(vals), p=p))


# ─── Substrate ───

class AttentionTrajV2:
    """
    Attention Trajectory v2: fixes Step 1007 bootstrap failure.

    Key design decisions:
    - Buffer resets on game switch (v2: no cross-game contamination)
    - 100-step warmup of pure random per game (buffer must populate before use)
    - 800b softmax fallback when buffer too small (< 50 entries)
    - Running mean alpha=0.1 (10× faster than 1007's 0.01)
    - Buffer size 200 (10× smaller → 10× faster attention, lower noise)
    - Temperature 0.1 (sharper → more decisive retrieval)

    Interface contract (ChainRunner-compatible):
      process(obs: np.ndarray) -> int
      on_level_transition()
      set_game(n_actions: int)
    """

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4
        self._step_in_game = 0

        # Trajectory buffer (resets on game switch — v2 design)
        self._buf_K = deque(maxlen=MAX_BUFFER)   # past encodings (ENC_DIM,)
        self._buf_A = deque(maxlen=MAX_BUFFER)   # past action indices
        self._buf_D = deque(maxlen=MAX_BUFFER)   # past deltas (float)

        # Running mean (global — persists across games for domain adaptation signal)
        self._running_mean = np.zeros(ENC_DIM, dtype=np.float32)

        # W_pred (resets on game switch — different action space)
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)

        # 800b fallback: delta per action
        self._delta_per_action = np.full(4, INIT_DELTA, dtype=np.float32)

        self._prev_enc = None
        self._prev_action = None

    def set_game(self, n_actions: int):
        """Called on game switch. Reset per-game state; keep global running mean."""
        self._n_actions = n_actions
        self._step_in_game = 0
        # Buffer resets (v2: no cross-game contamination)
        self._buf_K.clear()
        self._buf_A.clear()
        self._buf_D.clear()
        self._W_pred = np.zeros((ENC_DIM, ENC_DIM), dtype=np.float32)
        self._delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self._prev_enc = None
        self._prev_action = None

    def _encode(self, obs):
        """Encode observation with global running mean centering (alpha=0.1)."""
        enc_raw = _enc_frame(np.asarray(obs, dtype=np.float32))
        self._running_mean = (
            (1 - RUNNING_MEAN_ALPHA) * self._running_mean + RUNNING_MEAN_ALPHA * enc_raw
        )
        return (enc_raw - self._running_mean).astype(np.float32)

    def process(self, obs: np.ndarray) -> int:
        self._step_in_game += 1
        enc = self._encode(obs)

        if self._prev_enc is not None and self._prev_action is not None:
            delta = float(np.linalg.norm(enc - self._prev_enc))

            # Append to buffer (deque handles maxlen automatically)
            self._buf_K.append(self._prev_enc.copy())
            self._buf_A.append(self._prev_action)
            self._buf_D.append(delta)

            # W_pred update (gradient ascent for prediction error signal)
            pred = self._W_pred @ self._prev_enc
            error = enc - pred
            err_norm = float(np.linalg.norm(error))
            if err_norm > 10.0:
                error = error * (10.0 / err_norm)
            if not np.any(np.isnan(error)):
                self._W_pred -= ETA_W * np.outer(error, self._prev_enc)

            # 800b delta update (fallback tracker)
            a = self._prev_action
            if a < len(self._delta_per_action):
                self._delta_per_action[a] = (
                    (1 - ALPHA_EMA_800B) * self._delta_per_action[a] + ALPHA_EMA_800B * delta
                )

        # ── Action selection ──
        n_buf = len(self._buf_K)
        in_warmup = self._step_in_game <= WARMUP
        use_fallback = n_buf < BUFFER_FALLBACK

        if in_warmup:
            # Pure random during warmup
            action = int(self._rng.randint(0, self._n_actions))
        elif use_fallback or self._rng.random() < EPS:
            if use_fallback:
                # 800b softmax when buffer not full yet
                action = _softmax_sel(self._delta_per_action, 0.1, self._rng)
            else:
                # Epsilon random
                action = int(self._rng.randint(0, self._n_actions))
        else:
            # Attention over buffer (O(n_buf) where n_buf <= 200)
            K = np.stack(self._buf_K)                        # (T, ENC_DIM)
            A = np.array(self._buf_A, dtype=np.int32)        # (T,)
            D = np.array(self._buf_D, dtype=np.float32)      # (T,)

            # Attention: which past states are similar to current?
            scores = K @ enc / (ENC_DIM ** 0.5) / TEMP       # (T,)
            attn = _softmax(scores)                           # (T,)

            # Weighted delta accumulation per action
            action_scores = np.zeros(self._n_actions, dtype=np.float32)
            weighted_D = attn * D                             # (T,)
            for t in range(len(A)):
                if A[t] < self._n_actions:
                    action_scores[A[t]] += weighted_D[t]
            action = int(np.argmax(action_scores))

        self._prev_enc = enc.copy()
        self._prev_action = action
        return action

    def on_level_transition(self):
        """Clear prev to avoid cross-episode delta corruption."""
        self._prev_enc = None
        self._prev_action = None


SUBSTRATE_CLASS = AttentionTrajV2
