"""
step0779.py — MomentumExplorer779: momentum action + Hebbian forward model.

R3 hypothesis: momentum creates longer directional trajectories → W learns
richer directional dynamics → R3_cf > 778 (random). Same W as 778 but
action = repeat_last 70%, random 30%.

D(s) = {W, running_mean}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

D = 64
ETA = 0.01
MOMENTUM_PROB = 0.70


class MomentumExplorer779(BaseSubstrate):
    """Forward model W + momentum action (70% repeat, 30% random).

    Same W update as 778. Action selection adds momentum.
    D(s) = {W, running_mean}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = D + n_actions
        self.W = rng.randn(D, d_in).astype(np.float32) * 0.01  # M
        self.running_mean = np.zeros(D, np.float32)              # M
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_action = 0
        self._rng = np.random.RandomState(seed + 1)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)[:D]
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)

        if self._prev_enc is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            self.W += ETA * np.outer(x, inp)

        # Momentum: 70% repeat last, 30% random
        if self._rng.random() < MOMENTUM_PROB:
            action = self._last_action
        else:
            action = self._rng.randint(0, self._n_actions)

        self._last_action = action
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
            "_last_action": self._last_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_action = state["_prev_action"]
        self._last_action = state["_last_action"]

    def reset(self, seed: int) -> None:
        self._prev_enc = None
        self._prev_action = None
        self._last_action = 0

    def on_level_transition(self) -> None:
        self._prev_enc = None
        self._prev_action = None
        self._last_action = 0

    def frozen_elements(self) -> list:
        return [
            {"name": "W_hebbian", "class": "M",
             "justification": "W updated by every (obs,action)->obs' transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "momentum_action", "class": "U",
             "justification": "70% repeat last. Could be 50%, 90%. System doesn't choose probability."},
            {"name": "enc_truncate_64", "class": "U",
             "justification": "Truncates 256-dim to 64. System doesn't choose d."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
            {"name": "outer_product_rule", "class": "U",
             "justification": "Hebbian outer product. System doesn't choose update rule."},
            {"name": "action_onehot", "class": "U",
             "justification": "One-hot action encoding. System doesn't choose."},
        ]
