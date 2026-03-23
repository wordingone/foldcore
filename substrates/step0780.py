"""
step0780.py -- PredictionContrast780: action selection via max predicted change.

R3 hypothesis: choosing actions that maximize predicted state change finds novel
states without visit counts. W in R^{{256x(256+n_actions)}} same as 778.
Action: argmax_a ||W concat(obs,a_oh) - obs||_2.

Correction (mail 2565): W input = 256-dim encoded obs, NOT hash integer.
D(s) = {{W, running_mean}}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01


class PredictionContrast780(BaseSubstrate):
    def __init__(self, n_actions=4, seed=0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01
        self.running_mean = np.zeros(DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None

    def _encode(self, obs):
        x = _enc_frame(obs)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _encode_for_pred(self, obs):
        return _enc_frame(obs) - self.running_mean

    def predict_next(self, enc, action):
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        inp = np.concatenate([enc, a_oh])
        return self.W @ inp

    def process(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x
        if self._prev_enc is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            pred = self.predict_next(x, a)
            score = float(np.sum((pred - x)**2))
            if score > best_score:
                best_score = score
                best_a = a
        self._prev_enc = x.copy()
        self._prev_action = best_a
        return best_a

    @property
    def n_actions(self):
        return self._n_actions

    def get_state(self):
        return {"W": self.W.copy(), "running_mean": self.running_mean.copy(),
                "_n_obs": self._n_obs,
                "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
                "_prev_action": self._prev_action}

    def set_state(self, state):
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed):
        self._prev_enc = None; self._prev_action = None; self._last_enc = None

    def on_level_transition(self):
        self._prev_enc = None; self._prev_action = None

    def frozen_elements(self):
        return [
            {"name": "W_hebbian", "class": "M",
             "justification": "W updated by every transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "max_predicted_change_rule", "class": "I",
             "justification": "argmax ||W(obs,a)-obs||. Removing loses all structure."},
        ]
