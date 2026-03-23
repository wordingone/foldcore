"""
step0778.py — GlobalForwardModel778: random actions + Hebbian forward model.

R3 hypothesis: global W matrix captures transferable dynamics even under
random action selection. W ∈ R^{256×(256+n_actions)} updated Hebbianly.
D(s) = {W, running_mean}. L(s) = ∅.

Correction (mail 2565): W input = 256-dim encoded obs, NOT 64-dim truncation.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01


class GlobalForwardModel778(BaseSubstrate):
    """Random action + global Hebbian forward model W ∈ R^{256×(256+n_actions)}.

    D(s) = {W, running_mean}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01  # M
        self.running_mean = np.zeros(DIM, np.float32)              # M
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._rng = np.random.RandomState(seed + 1)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)  # 256-dim
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _encode_for_pred(self, obs: np.ndarray) -> np.ndarray:
        return _enc_frame(obs) - self.running_mean

    def predict_next(self, enc: np.ndarray, action: int) -> np.ndarray:
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        return self.W @ inp

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x

        if self._prev_enc is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule (gradient descent on MSE): minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

        action = self._rng.randint(0, self._n_actions)
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
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
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
             "justification": "W updated by every (obs,action)->obs' transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "random_action", "class": "I",
             "justification": "No selection mechanism. Irreducible baseline."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. Could be 0.001 or 0.1. System doesn't choose."},
            {"name": "outer_product_rule", "class": "U",
             "justification": "Hebbian: outer(next, input). Could be delta rule. System doesn't choose."},
            {"name": "action_onehot", "class": "U",
             "justification": "One-hot action encoding. Could be ordinal. System doesn't choose."},
        ]
