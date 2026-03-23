"""
step0806v2.py -- EpsilonPrediction806v2: 80% random + 20% argmax predicted change.

R3 hypothesis: random action provides persistence (needed for LS20/FT09),
while 20% prediction-contrast uses trained W to occasionally choose informative actions.
This combines the robustness of random with the learning of forward model.

Different from step855b (epsilon-compression): uses W prediction directly, not learning progress.

D(s) = {W, running_mean}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01
EPSILON_RANDOM = 0.80  # 80% random, 20% prediction-contrast


class EpsilonPrediction806v2(BaseSubstrate):
    """80% random + 20% argmax predicted change.

    Random provides persistence for LS20-like games.
    argmax ||W(enc,a)-enc|| uses trained W for informative action selection.
    D(s) = {W, running_mean}. L(s) = empty.

    R2 note: delta rule computes ∇_W||W@inp - x||^2. Target x is environmental
    observation (R5 ground truth), not external reward. Self-supervised error
    correction. R2 tension flagged: I (irreducible for convergent learning).
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01  # M
        self.running_mean = np.zeros(DIM, np.float32)              # M
        self._n_obs = 0
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

        # 80% random, 20% argmax predicted change
        if self._rng.random() < EPSILON_RANDOM:
            action = self._rng.randint(0, self._n_actions)
        else:
            best_a, best_score = 0, -1.0
            for a in range(self._n_actions):
                pred = self.predict_next(x, a)
                score = float(np.sum((pred - x) ** 2))
                if score > best_score:
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
             "justification": "W updated by gradient descent on prediction error. System-driven.",
             "R2_note": "Delta rule computes ∇_W||W@inp - x||^2. Target x is environmental obs (R5). Self-supervised. R2 tension flagged."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean adapts to obs distribution. System-driven."},
            {"name": "epsilon_random", "class": "I",
             "justification": "80% random provides persistence. Removing -> pure prediction-contrast = L1=0."},
            {"name": "argmax_change_rule", "class": "I",
             "justification": "20% argmax ||W(enc,a)-enc||. Removing -> pure random (baseline)."},
        ]
