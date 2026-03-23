"""
step0803.py -- ObsHashCycling803: per-observation action cycling.

R3 hypothesis: tracking which action to try next per observation hash
(D(s) = {hash_to_action_idx}) enables structured exploration that
transfers -- cold substrate never visited these hashes, warm has.

State: hash_to_action_idx[h] = next_action_index. Per-OBSERVATION,
not per-(state,action). Action = hash_to_action_idx[h] % n_actions,
then increment.

D(s) = {hash_to_action_idx}. L(s) = empty (no visit counts).
Kill: if L1=0/10.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
K_NAV = 12


class ObsHashCycling803(BaseSubstrate):
    """Per-observation action cycling via LSH hash.

    Encodes obs to 256-dim, hashes to 12-bit key, cycles through
    actions 0..n_actions-1 at each unique observation.
    D(s) = {hash_to_action_idx}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        self._H = rng.randn(K_NAV, DIM).astype(np.float32)  # fixed hash planes
        self.hash_to_action_idx = {}   # M: per-obs cycling counter
        self.running_mean = np.zeros(DIM, np.float32)  # M: obs centering
        self._n_obs = 0

    def _hash_obs(self, x_c: np.ndarray) -> int:
        bits = (self._H @ x_c > 0).astype(np.uint8)
        return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = _enc_frame(observation)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        x_c = x - self.running_mean

        h = self._hash_obs(x_c)
        idx = self.hash_to_action_idx.get(h, 0)
        action = idx % self._n_actions
        self.hash_to_action_idx[h] = idx + 1
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "hash_to_action_idx": dict(self.hash_to_action_idx),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self.hash_to_action_idx = dict(state["hash_to_action_idx"])
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]

    def reset(self, seed: int) -> None:
        pass  # preserve hash_to_action_idx (D(s))

    def on_level_transition(self) -> None:
        pass

    def frozen_elements(self) -> list:
        return [
            {"name": "hash_to_action_idx", "class": "M",
             "justification": "Per-obs cycling counter. Modified by every obs processed. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "_H", "class": "I",
             "justification": "Fixed LSH planes. Removing -> no structured obs hashing. Irreducible."},
            {"name": "obs_hash_cycling", "class": "I",
             "justification": "Cycling idx % n_actions. Removing -> random or fixed action. Irreducible for structure."},
        ]
