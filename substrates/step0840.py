"""
step0840.py -- AntColony840: anti-pheromone exploration.

R3 hypothesis: pheromone_map[obs_hash] (per-obs, not per-(obs,action)) enables
exploration without visit counts. Decay prevents convergence.

State: pheromone_map[obs_hash] = counter. Each visit sets counter += DEPOSIT.
Counter decays * DECAY each step.
Action: predict successor obs for each action via W. Pick action whose predicted
successor has LOWEST pheromone (least recently visited region).

Graph ban: pheromone keyed by obs_hash ONLY. PASS.
D(s) = {W, pheromone_map, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01
K_NAV = 12
DEPOSIT = 10.0
DECAY = 0.99


class AntColony840(BaseSubstrate):
    """Anti-pheromone: avoid recently visited obs regions.

    W predicts next obs per action. Pick action toward lowest pheromone region.
    D(s) = {W, pheromone_map, running_mean}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01  # M
        self._H = rng.randn(K_NAV, DIM).astype(np.float32)       # I: fixed hash
        self.running_mean = np.zeros(DIM, np.float32)              # M
        self._n_obs = 0
        self.pheromone_map = {}   # M: obs_hash -> pheromone level
        self._prev_enc = None
        self._prev_action = None
        self._last_enc = None
        self._rng = np.random.RandomState(seed + 1)

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

        # Deposit pheromone at current obs
        h = self._hash_enc(x)
        self.pheromone_map[h] = self.pheromone_map.get(h, 0.0) + DEPOSIT

        # Decay all pheromone
        self.pheromone_map = {k: v * DECAY for k, v in self.pheromone_map.items() if v * DECAY > 0.01}

        # Update W from previous transition
        if self._prev_enc is not None and self._prev_action is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

        # Action: pick action whose predicted successor has lowest pheromone
        best_a, best_score = 0, float('inf')
        for a in range(self._n_actions):
            pred = self.predict_next(x, a)
            pred_h = self._hash_enc(pred)
            score = self.pheromone_map.get(pred_h, 0.0)
            if score < best_score:
                best_score = score
                best_a = a

        self._prev_enc = x.copy()
        self._prev_action = best_a
        return best_a

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "pheromone_map": dict(self.pheromone_map),
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self.pheromone_map = dict(state["pheromone_map"])
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
            {"name": "pheromone_map", "class": "M",
             "justification": "Per-obs pheromone deposited and decayed. System-driven."},
            {"name": "_H_hash_planes", "class": "I",
             "justification": "Fixed LSH for obs hashing. Removing breaks pheromone targeting."},
            {"name": "antipheromone_rule", "class": "I",
             "justification": "argmin pheromone of predicted successor. Removing -> no novelty bias."},
            {"name": "deposit_decay", "class": "U",
             "justification": "DEPOSIT=10, DECAY=0.99. System does not choose these values."},
        ]
