"""
step0809b.py — CyclingForwardModel809b: action cycling + forward model in parallel.

R3 hypothesis: action cycling guarantees systematic coverage (solves step788/803 finding
that LS20 rewards action 0 first); W forward model learns dynamics in parallel and
its D(s) = {W} transfers across seeds.

Key insight from steps 788/803:
- Round-robin (788): gets 0/seed. Too rigid, can't repeat action 0.
- Cycling from cold (803): gets 226/seed because cold start = "action 0 first" everywhere.
- This substrate: cycling for coverage + W learns while cycling. R3_cf via prediction accuracy.

D(s) = {W, running_mean, hash_to_idx}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01
K_NAV = 12


class CyclingForwardModel809b(BaseSubstrate):
    """Action cycling + Hebbian forward model learning in parallel.

    Action selection: per-obs-hash cycling (systematic coverage).
    Dynamics learning: W Hebbian update on every transition.
    D(s) = {W, running_mean, hash_to_idx}. L(s) = empty.
    R3_cf: prediction accuracy (warm W predicts better than cold W).
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01  # M
        self._H = rng.randn(K_NAV, DIM).astype(np.float32)       # I: fixed hash planes
        self.running_mean = np.zeros(DIM, np.float32)              # M
        self._n_obs = 0
        self.hash_to_idx = {}   # M: per-obs cycling counter
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

        # Update W from previous transition (Hebbian)
        if self._prev_enc is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

        # Action: per-obs-hash cycling (systematic coverage)
        h = self._hash_enc(x)
        idx = self.hash_to_idx.get(h, 0)
        action = idx % self._n_actions
        self.hash_to_idx[h] = idx + 1

        self._prev_enc = x.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "hash_to_idx": dict(self.hash_to_idx),
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self.hash_to_idx = dict(state["hash_to_idx"])
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
            {"name": "hash_to_idx", "class": "M",
             "justification": "Per-obs cycling counter. Modified by every step. System-driven."},
            {"name": "_H_hash_planes", "class": "I",
             "justification": "Fixed LSH planes for obs hashing. Removing breaks per-obs identification."},
            {"name": "cycling_rule", "class": "I",
             "justification": "idx%n_actions cycling. Removing -> random or fixed action."},
        ]
