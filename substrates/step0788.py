"""
step0788.py — GlobalActionBalance788: argmin global action count.

R3 hypothesis: balancing action usage globally (not per-state) provides
better-than-random exploration. State: action_count[a] for each action.
GLOBAL, not per-(state,action). Graph ban: PASS.

D(s) = {action_count}. L(s) = empty.
Kill: if L1=0/10 (global balancing doesn't navigate).
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256


class GlobalActionBalance788(BaseSubstrate):
    """Global action frequency balancing: argmin(action_count[a]).

    Picks the globally least-used action regardless of state.
    D(s) = {action_count}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        self.action_count = np.zeros(n_actions, np.float32)  # M: global counts
        self.running_mean = np.zeros(DIM, np.float32)         # M: obs centering
        self._n_obs = 0

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = _enc_frame(observation)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x

        action = int(np.argmin(self.action_count))
        self.action_count[action] += 1
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "action_count": self.action_count.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self.action_count = state["action_count"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]

    def reset(self, seed: int) -> None:
        pass  # preserve action_count (D(s))

    def on_level_transition(self) -> None:
        pass

    def frozen_elements(self) -> list:
        return [
            {"name": "action_count", "class": "M",
             "justification": "Global action counts modified by every action taken. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "argmin_action_count", "class": "I",
             "justification": "argmin global count. Removing -> round-robin or random. Irreducible for balance."},
        ]
