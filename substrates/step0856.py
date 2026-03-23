"""
step0856.py -- StateEntropy856: obs histogram entropy maximization.

R3 hypothesis: obs_histogram (D(s), per-obs count, NOT per-(obs,action)) enables
entropy-maximizing exploration that transfers across seeds.

State: obs_histogram[obs_hash] = visit_count (global, per-obs).
Action: pick action predicted to produce obs with LOWEST count in histogram.
Graph ban: histogram keyed by obs_hash only. PASS.

D(s) = {W, obs_histogram, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01
K_NAV = 12


class StateEntropy856(BaseSubstrate):
    """Histogram-based entropy maximization.

    Tracks obs visit counts globally. Picks action whose predicted next
    obs has lowest count (least visited region).
    D(s) = {W, obs_histogram, running_mean}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01   # M
        self._H = rng.randn(K_NAV, DIM).astype(np.float32)         # I: fixed
        self.running_mean = np.zeros(DIM, np.float32)               # M
        self._n_obs = 0
        self.obs_histogram = {}   # M: obs_hash -> count
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _encode_for_pred(self, obs: np.ndarray) -> np.ndarray:
        return _enc_frame(obs) - self.running_mean

    def predict_next(self, enc: np.ndarray, action: int) -> np.ndarray:
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        return self.W @ np.concatenate([enc, a_oh])

    def _hash_enc(self, x_c: np.ndarray) -> int:
        bits = (self._H @ x_c > 0).astype(np.uint8)
        return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x

        # Update histogram at current obs
        h = self._hash_enc(x)
        self.obs_histogram[h] = self.obs_histogram.get(h, 0) + 1

        # Update W from previous transition
        if self._prev_enc is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

        # Action: pick action predicted to lead to least-visited obs
        best_a, best_score = 0, float('inf')
        for a in range(self._n_actions):
            pred = self.predict_next(x, a)
            pred_h = self._hash_enc(pred)
            count = self.obs_histogram.get(pred_h, 0)
            if count < best_score:
                best_score = count
                best_a = a

        self._prev_enc = x.copy()
        self._prev_action = best_a
        return best_a

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "obs_histogram": dict(self.obs_histogram),
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self.obs_histogram = dict(state["obs_histogram"])
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed: int) -> None:
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def on_level_transition(self) -> None:
        self._prev_enc = None
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W_hebbian", "class": "M",
             "justification": "W updated by every transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "obs_histogram", "class": "M",
             "justification": "Per-obs visit count. Modified by every step. System-driven."},
            {"name": "_H_hash_planes", "class": "I",
             "justification": "Fixed LSH. Removing breaks obs identification."},
            {"name": "min_count_rule", "class": "I",
             "justification": "argmin predicted_next count. Removing -> no entropy bias."},
        ]
