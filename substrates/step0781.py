"""
step0781.py — EnsembleDisagreement781: K=3 W matrices, action by max variance.

R3 hypothesis: ensemble variance of forward model predictions correlates with
novelty without counting visits. K=3 independent W matrices (different random
init). Action = argmax_a var(W_1(obs,a), W_2(obs,a), W_3(obs,a)).

R1 check: variance is internal dynamics signal, not external objective.
D(s) = {W_1, W_2, W_3, running_mean}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

D = 64
ETA = 0.01
K_ENSEMBLE = 3


class EnsembleDisagreement781(BaseSubstrate):
    """K=3 forward models. Action = argmax_a variance of K predictions.

    Variance across ensemble = uncertainty = novelty signal.
    D(s) = {W_1..W_K, running_mean}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        d_in = D + n_actions
        self.Ws = []
        for k in range(K_ENSEMBLE):
            rng = np.random.RandomState(seed * 1000 + k)
            self.Ws.append(rng.randn(D, d_in).astype(np.float32) * 0.01)  # M
        self.running_mean = np.zeros(D, np.float32)  # M
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)[:D]
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def _ensemble_variance(self, x: np.ndarray, a: int) -> float:
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[a] = 1.0
        inp = np.concatenate([x, a_oh])
        preds = np.stack([W @ inp for W in self.Ws])  # (K, D)
        return float(preds.var(axis=0).mean())

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)

        # Hebbian update all ensemble members
        if self._prev_enc is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            for W in self.Ws:
                W += ETA * np.outer(x, inp)

        # Select action with highest ensemble disagreement
        best_a, best_var = 0, -1.0
        for a in range(self._n_actions):
            v = self._ensemble_variance(x, a)
            if v > best_var:
                best_var = v
                best_a = a

        self._prev_enc = x.copy()
        self._prev_action = best_a
        return best_a

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "Ws": [W.copy() for W in self.Ws],
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.Ws = [W.copy() for W in state["Ws"]]
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]

    def reset(self, seed: int) -> None:
        self._prev_enc = None
        self._prev_action = None

    def on_level_transition(self) -> None:
        self._prev_enc = None
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W1_hebbian", "class": "M",
             "justification": "W_1 updated by every (obs,action)->obs' transition. System-driven."},
            {"name": "W2_hebbian", "class": "M",
             "justification": "W_2 updated by every (obs,action)->obs' transition. System-driven."},
            {"name": "W3_hebbian", "class": "M",
             "justification": "W_3 updated by every (obs,action)->obs' transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "max_variance_rule", "class": "I",
             "justification": "argmax_a variance. Removing -> no disagreement signal -> collapses to random."},
            {"name": "k_ensemble_size", "class": "U",
             "justification": "K=3. Could be 2, 5, 10. System doesn't choose."},
            {"name": "enc_truncate_64", "class": "U",
             "justification": "64-dim. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
        ]
