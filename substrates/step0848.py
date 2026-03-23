"""
step0848.py -- MaxEntropy848: maximum entropy action selection.

R3 hypothesis: maintaining high-entropy action distribution prevents convergence
while novelty feedback adaptively guides exploration.

State: action_logits ∈ R^{n_actions}. Action = sample from softmax(logits).
Update: increase logit of action that produced novel obs (hash not in histogram),
        decrease logit of action that produced familiar obs.
Entropy reg: logits pulled toward uniform distribution.

D(s) = {action_logits, obs_histogram, running_mean}. L(s) = empty.
Graph ban: obs_histogram keyed by obs_hash only. PASS.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
K_NAV = 12
ETA_LOGIT = 0.1   # logit update magnitude
ETA_ENT = 0.01    # entropy regularization (pull toward uniform)
TEMP = 1.0        # softmax temperature


class MaxEntropy848(BaseSubstrate):
    """Max-entropy action selection with novelty-based logit updates.

    action_logits updated based on whether selected action produced novel obs.
    Entropy regularization prevents collapse to single action.
    D(s) = {action_logits, obs_histogram, running_mean}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        self._H = rng.randn(K_NAV, DIM).astype(np.float32)   # I: fixed hash planes
        self.action_logits = np.zeros(n_actions, np.float32)  # M: action preferences
        self.obs_histogram = {}   # M: obs_hash -> count
        self.running_mean = np.zeros(DIM, np.float32)         # M
        self._n_obs = 0
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

    def _hash_enc(self, x_c: np.ndarray) -> int:
        bits = (self._H @ x_c > 0).astype(np.uint8)
        return int(np.packbits(bits[:8], bitorder='big').tobytes().hex(), 16)

    def process(self, observation) -> int:
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        self._last_enc = x

        # Update logits based on novelty of current obs (result of prev action)
        h = self._hash_enc(x)
        count = self.obs_histogram.get(h, 0)
        if self._prev_action is not None:
            if count == 0:  # novel obs → reinforce action that led here
                self.action_logits[self._prev_action] += ETA_LOGIT
            else:  # familiar obs → penalize (already explored)
                self.action_logits[self._prev_action] -= ETA_LOGIT * 0.5

        # Update histogram
        self.obs_histogram[h] = count + 1

        # Entropy regularization: pull logits toward uniform
        uniform = np.zeros(self._n_actions, np.float32)
        self.action_logits += ETA_ENT * (uniform - self.action_logits)

        # Sample action from softmax(logits / TEMP)
        logits = self.action_logits / TEMP
        logits = logits - logits.max()  # numerical stability
        probs = np.exp(logits)
        probs = probs / probs.sum()
        action = int(self._rng.choice(self._n_actions, p=probs))

        self._prev_action = action
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "action_logits": self.action_logits.copy(),
            "obs_histogram": dict(self.obs_histogram),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.action_logits = state["action_logits"].copy()
        self.obs_histogram = dict(state["obs_histogram"])
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_action = state["_prev_action"]

    def reset(self, seed: int) -> None:
        self._prev_action = None
        self._last_enc = None

    def on_level_transition(self) -> None:
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "action_logits", "class": "M",
             "justification": "Logits updated by novelty feedback every step. System-driven."},
            {"name": "obs_histogram", "class": "M",
             "justification": "Per-obs visit count. Modified every step. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "_H_hash_planes", "class": "I",
             "justification": "Fixed LSH planes for obs hashing. Removing breaks novelty detection."},
            {"name": "novelty_logit_update", "class": "I",
             "justification": "Increase logit on novel obs, decrease on familiar. Core mechanism."},
            {"name": "entropy_regularization", "class": "I",
             "justification": "Pull toward uniform prevents collapse. Removing -> convergent action (U22)."},
        ]
