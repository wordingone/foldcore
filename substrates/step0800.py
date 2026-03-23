"""
step0800.py -- PerActionChangePursuit800: argmax of global per-action EMA change.

R3 hypothesis: tracking ||enc(obs_{t+1}) - enc(obs_t)|| per action (globally,
not per state) identifies which actions produce observation changes.
On FT09, productive clicks (8/68) cause large changes; non-productive = small.
argmax(delta[a]) selects the historically most-changing action.

D(s) = {delta_per_action, running_mean}. L(s) = empty.
Global per-action, NOT per-(state,action). Passes graph ban.

Leo mail 2585: "per-action GLOBAL change tracking."
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
EMA_ALPHA = 0.1   # EMA decay: recent experience weighted
INIT_DELTA = 1.0  # Initial delta (explore before converging)


class PerActionChangePursuit800(BaseSubstrate):
    """argmax_a EMA[||enc(obs_{t+1}) - enc(obs_t)||] when action a was taken.

    Identifies productive actions by tracking observation change magnitude.
    FT09: productive clicks (high delta) vs non-productive clicks (low delta).
    LS20: directional actions should have higher delta than click actions.

    R3 mechanism: D(s) self-modifies which actions to try next based on
    accumulated per-action change experience. Action selection changes with D.

    D(s) = {delta_per_action, running_mean}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        # Initialize all deltas equal (uniform exploration until learned)
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

        # Update delta for prev action (EMA of change magnitude)
        if self._prev_enc is not None and self._prev_action is not None:
            change = float(np.sqrt(np.sum((x - self._prev_enc) ** 2)))
            a = self._prev_action
            self.delta_per_action[a] = (
                (1 - EMA_ALPHA) * self.delta_per_action[a] + EMA_ALPHA * change
            )

        # argmax: pick action with highest historical change
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
             "justification": "EMA of per-action observation change magnitude. System-driven by environment responses."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "argmax_change_rule", "class": "I",
             "justification": "argmax(delta[a]) selects historically most-changing action. Removing -> no systematic action preference."},
            {"name": "ema_alpha", "class": "I",
             "justification": "EMA decay 0.1. Removing -> no temporal weighting."},
        ]
