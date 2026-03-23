"""
step0855b.py — EpsilonCompressionProgress855b: compression progress with epsilon-random.

R3 hypothesis: 80% compression progress + 20% random action prevents action collapse
while maintaining anti-noisy-TV property.

Fix for step855 action collapse: pure compression progress argmax converges to one
action too quickly. Epsilon-random ensures diversity without losing the learning signal.

D(s) = {W, running_mean, acc_ema, prev_acc_ema}. L(s) = empty.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

DIM = 256
ETA = 0.01
WINDOW = 50
EMA_ALPHA = 0.1
EPSILON = 0.20  # 20% random action


class EpsilonCompressionProgress855b(BaseSubstrate):
    """Compression progress + epsilon-random.

    80% of steps: argmax compression progress.
    20% of steps: uniform random action (diversity guarantee).
    Anti-noisy-TV property retained: learning signal guides 80%.

    D(s) = {W, running_mean, acc_ema, prev_acc_ema}. L(s) = empty.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = DIM + n_actions
        self.W = rng.randn(DIM, d_in).astype(np.float32) * 0.01  # M
        self.running_mean = np.zeros(DIM, np.float32)              # M
        self._n_obs = 0
        self.acc_ema = np.zeros(n_actions, np.float32)       # M
        self.prev_acc_ema = np.zeros(n_actions, np.float32)  # M
        self._progress_ema = np.zeros(n_actions, np.float32)  # M
        self._step = 0
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
        self._step += 1

        if self._prev_enc is not None and self._prev_action is not None:
            a = self._prev_action
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[a] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            pred = self.W @ inp

            err = float(np.sum((pred - x) ** 2))
            norm = float(np.sum(x ** 2)) + 1e-8
            acc = max(0.0, 1.0 - err / norm)

            # Delta rule: minimize ||W@inp - x||^2
            pred_err = self.W @ inp - x
            self.W -= ETA * np.outer(pred_err, inp)

            old_acc = self.acc_ema[a]
            self.acc_ema[a] = (1 - EMA_ALPHA) * old_acc + EMA_ALPHA * acc
            progress = self.acc_ema[a] - self.prev_acc_ema[a]
            self._progress_ema[a] = (1 - EMA_ALPHA) * self._progress_ema[a] + EMA_ALPHA * progress

            if self._step % WINDOW == 0:
                self.prev_acc_ema = self.acc_ema.copy()

        # Epsilon-random: 20% random, 80% compression progress argmax
        if self._rng.random() < EPSILON:
            action = self._rng.randint(0, self._n_actions)
        elif self._step > WINDOW and np.max(self._progress_ema) > 0:
            action = int(np.argmax(self._progress_ema))
        else:
            action = self._rng.randint(0, self._n_actions)

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
            "acc_ema": self.acc_ema.copy(),
            "prev_acc_ema": self.prev_acc_ema.copy(),
            "_progress_ema": self._progress_ema.copy(),
            "_step": self._step,
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self.acc_ema = state["acc_ema"].copy()
        self.prev_acc_ema = state["prev_acc_ema"].copy()
        self._progress_ema = state["_progress_ema"].copy()
        self._step = state["_step"]
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
            {"name": "acc_ema", "class": "M",
             "justification": "Per-action prediction accuracy EMA. System-driven."},
            {"name": "prev_acc_ema", "class": "M",
             "justification": "Lagged accuracy for progress. System-driven."},
            {"name": "progress_ema", "class": "M",
             "justification": "Compression progress per action. System-driven."},
            {"name": "argmax_progress_rule", "class": "I",
             "justification": "argmax progress for 80%. Removing loses learning-based guidance."},
            {"name": "epsilon_random", "class": "I",
             "justification": "20% random prevents collapse. Removing -> convergent action (kills 855 pure)."},
        ]
