"""
step0800b.py -- EpsilonActionChange800b: 80% argmax delta + 20% random.

R3 hypothesis: per-action change tracking (step800) may collapse to a single
action (argmax gets stuck after first large-change event). Adding 20% random
prevents collapse while keeping the productive-action signal.

Compare: step800 (pure argmax), step806v2 (pure argmin W), random (36.4/seed).

D(s) = {delta_per_action, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
EMA_ALPHA = 0.1
EPSILON_RANDOM = 0.20  # 20% random to prevent collapse
INIT_DELTA = 1.0


class EpsilonActionChange800b(BaseSubstrate):
    """80% argmax per-action EMA change + 20% random exploration.

    Prevents action collapse while maintaining the productive-click signal.
    Random component ensures all actions sampled occasionally → EMA updates
    for all actions → delta_per_action stays calibrated.

    D(s) = {delta_per_action, running_mean}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        self.delta_per_action = np.full(n_actions, INIT_DELTA, dtype=np.float32)
        self.running_mean = np.zeros(DIM, np.float32)
        self._n_obs = 0
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._rng = np.random.RandomState(seed)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x

        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sqrt(np.sum((x - self._prev_enc) ** 2)))
            a = self._prev_action
            self.delta_per_action[a] = (
                (1 - EMA_ALPHA) * self.delta_per_action[a] + EMA_ALPHA * change
            )

        if self._rng.random() < EPSILON_RANDOM:
            action = self._rng.randint(0, self._n_actions)
        else:
            action = int(np.argmax(self.delta_per_action))

        self._prev_enc = x.copy()
        self._prev_action = action
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "delta_per_action": self.delta_per_action.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.delta_per_action = state["delta_per_action"].copy()
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
            {"name": "delta_per_action", "class": "M",
             "justification": "EMA of per-action observation change magnitude. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "argmax_change_rule", "class": "I",
             "justification": "80% argmax(delta[a]). Removing -> pure random baseline."},
            {"name": "epsilon_random", "class": "I",
             "justification": "20% random prevents action collapse. Removing -> pure argmax (step800)."},
        ]
