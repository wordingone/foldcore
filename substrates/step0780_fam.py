"""
step0780_fam.py -- FamiliarSuccessor780fam: argmin_a ||W(enc,a) - running_mean||².

R3 hypothesis: argmin distance to running_mean selects actions that bring
the agent toward the most-observed (average) state. This is a "go home" policy.

Compare to step780 (argmax = novelty) and step780_inv (argmin distance to current enc).
780_fam targets the GLOBAL mean, not local mean — different from 780_inv.

Leo mail 2582: "Step 780_fam (familiar successor): argmin_a ||W(enc,a) - running_mean||."

D(s) = {W, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01


class FamiliarSuccessor780fam(BaseSubstrate):
    """argmin_a ||W(enc,a) - running_mean||² — predict toward global mean state.

    'Go home' policy: always choose action predicted to return to average.
    Opposite of novelty-seeking. Tests if LS20 rewards mean-regression.

    D(s) = {W, running_mean}. L(s) = empty.
    Delta rule W update (gradient descent on prediction error).
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
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

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x

        if self._prev_enc is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

        # argmin_a ||W(enc,a) - running_mean||^2
        # running_mean in raw space, but enc is centered: enc = raw - running_mean
        # So running_mean in centered space = 0. Target is zero vector.
        # Equivalent to argmin_a ||W(enc,a)||^2 (go toward zero = go toward mean)
        best_a, best_score = 0, float('inf')
        for a in range(self._n_actions):
            pred = self.predict_next(x, a)
            # Distance to running_mean in centered enc space = distance to 0
            score = float(np.sum(pred ** 2))
            if score < best_score:
                best_score = score
                best_a = a
        action = best_a

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
            {"name": "W_delta_rule", "class": "M",
             "justification": "W updated by gradient descent on prediction error. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "argmin_to_mean_rule", "class": "I",
             "justification": "argmin ||W(enc,a)||^2 selects action predicted to return to mean. Removing -> no systematic action preference."},
        ]
