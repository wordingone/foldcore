"""
step0787.py — ReservoirComputing787: reservoir with Hebbian readout + decay.

R3 hypothesis: reservoir with anti-convergence mechanism (W_out decay) avoids
U22 kill that hit Steps 437d+. Fixed random W_res (spectral radius 0.95).
W_out: Hebbian update WITH DECAY (W_out *= 0.999 each step).
Decay prevents W_out convergence → maintains dynamic action selection.

h' = tanh(W_res @ h + W_in @ obs).
Action = argmax(W_out @ h).
W_out += eta * outer(a_oh, h). W_out *= decay.

D(s) = {h, W_out}. L(s) = ∅. W_res and W_in are fixed (U).
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

H_DIM = 256
D_OBS = 64
ETA = 0.01
DECAY = 0.999
SPECTRAL_RADIUS = 0.95


def _make_reservoir(dim: int, radius: float, rng) -> np.ndarray:
    W = rng.randn(dim, dim).astype(np.float32)
    eigvals = np.linalg.eigvals(W)
    W = W / np.max(np.abs(eigvals)) * radius
    return W.astype(np.float32)


class ReservoirComputing787(BaseSubstrate):
    """Fixed reservoir W_res + learned W_out with exponential decay.

    h' = tanh(W_res @ h + W_in @ obs).
    Action = argmax(W_out @ h).
    W_out += eta * outer(a_oh, h). W_out *= decay.
    D(s) = {h, W_out}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        self.W_res = _make_reservoir(H_DIM, SPECTRAL_RADIUS, rng)  # U: fixed
        self.W_in = rng.randn(H_DIM, D_OBS).astype(np.float32) * 0.1  # U: fixed
        self.W_out = rng.randn(n_actions, H_DIM).astype(np.float32) * 0.01  # M
        self.h = np.zeros(H_DIM, np.float32)  # M: reservoir state

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        return _enc_frame(obs)[:D_OBS]

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)

        # Reservoir update
        self.h = np.tanh(self.W_res @ self.h + self.W_in @ x).astype(np.float32)

        # Action selection
        logits = self.W_out @ self.h
        action = int(np.argmax(logits)) % self._n_actions

        # Hebbian update W_out with decay
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        self.W_out += ETA * np.outer(a_oh, self.h)
        self.W_out *= DECAY  # anti-convergence

        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "h": self.h.copy(),
            "W_out": self.W_out.copy(),
        }

    def set_state(self, state: dict) -> None:
        self.h = state["h"].copy()
        self.W_out = state["W_out"].copy()

    def reset(self, seed: int) -> None:
        self.h = np.zeros(H_DIM, np.float32)
        # Preserve W_out (D(s)) across resets

    def on_level_transition(self) -> None:
        self.h = np.zeros(H_DIM, np.float32)

    def frozen_elements(self) -> list:
        return [
            {"name": "h_reservoir_state", "class": "M",
             "justification": "h updated every step: tanh(W_res@h + W_in@obs). System-driven."},
            {"name": "W_out_hebbian_decay", "class": "M",
             "justification": "W_out updated Hebbianly with decay each step. System-driven."},
            {"name": "W_res_fixed", "class": "U",
             "justification": "Fixed random reservoir. Could be structured or learned. System doesn't choose."},
            {"name": "W_in_fixed", "class": "U",
             "justification": "Fixed random input projection. System doesn't choose."},
            {"name": "spectral_radius", "class": "U",
             "justification": "0.95. Could be 0.8 or 1.1. System doesn't choose."},
            {"name": "h_dim", "class": "U",
             "justification": "H_DIM=256. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
            {"name": "decay_rate", "class": "U",
             "justification": "decay=0.999. Could be 0.99 or no decay. System doesn't choose."},
            {"name": "argmax_action", "class": "U",
             "justification": "argmax(W_out@h). Could be softmax sampling. System doesn't choose."},
        ]
