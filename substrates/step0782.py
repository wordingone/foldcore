"""
step0782.py — HebbianRecurrent782: recurrent network with Hebbian updates.

R3 hypothesis: recurrent dynamics capture temporal structure → R3_cf > 0.
State: h ∈ R^128, W_x ∈ R^{128×d_obs} (Hebbian), W_a ∈ R^{n_actions×128} (Hebbian).
W_h ∈ R^{128×128} FIXED random (spectral radius 0.95).
h' = tanh(W_h @ h + W_x @ obs). W_x += eta * outer(h', obs). W_a += eta * outer(a_oh, h).
Action: argmax(W_a @ h).

All Hebbian. R2 compliant. No graph. No visit counts.
D(s) = {h, W_x, W_a}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

H_DIM = 128
D_OBS = 64
ETA = 0.01
SPECTRAL_RADIUS = 0.95


def _make_reservoir(dim: int, radius: float, rng) -> np.ndarray:
    W = rng.randn(dim, dim).astype(np.float32)
    eigvals = np.linalg.eigvals(W)
    W = W / np.max(np.abs(eigvals)) * radius
    return W.astype(np.float32)


class HebbianRecurrent782(BaseSubstrate):
    """Recurrent Hebbian network. Fixed reservoir W_h, learned W_x and W_a.

    h' = tanh(W_h @ h + W_x @ obs).
    W_x += eta * outer(h', obs).  (M)
    W_a += eta * outer(a_oh, h).  (M)
    Action: argmax(W_a @ h).
    D(s) = {h, W_x, W_a}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        self.W_h = _make_reservoir(H_DIM, SPECTRAL_RADIUS, rng)  # U: fixed random
        self.W_x = rng.randn(H_DIM, D_OBS).astype(np.float32) * 0.01  # M
        self.W_a = rng.randn(n_actions, H_DIM).astype(np.float32) * 0.01  # M
        self.h = np.zeros(H_DIM, np.float32)  # M: recurrent state

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        return _enc_frame(obs)[:D_OBS]

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)

        # Recurrent update
        h_new = np.tanh(self.W_h @ self.h + self.W_x @ x).astype(np.float32)

        # Hebbian update W_x: associate new hidden state with current obs
        self.W_x += ETA * np.outer(h_new, x)
        # Action before W_a update (use current h)
        logits = self.W_a @ self.h
        action = int(np.argmax(logits)) % self._n_actions
        a_oh = np.zeros(self._n_actions, np.float32)
        a_oh[action] = 1.0
        # Hebbian update W_a: associate action with hidden state
        self.W_a += ETA * np.outer(a_oh, self.h)

        self.h = h_new
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "h": self.h.copy(),
            "W_x": self.W_x.copy(),
            "W_a": self.W_a.copy(),
        }

    def set_state(self, state: dict) -> None:
        self.h = state["h"].copy()
        self.W_x = state["W_x"].copy()
        self.W_a = state["W_a"].copy()

    def reset(self, seed: int) -> None:
        # Preserve W_x, W_a (D(s)). Reset recurrent state h.
        self.h = np.zeros(H_DIM, np.float32)

    def on_level_transition(self) -> None:
        self.h = np.zeros(H_DIM, np.float32)

    def frozen_elements(self) -> list:
        return [
            {"name": "h_recurrent_state", "class": "M",
             "justification": "h updated every step by tanh(W_h@h + W_x@obs). System-driven."},
            {"name": "W_x_hebbian", "class": "M",
             "justification": "W_x updated: outer(h', obs) each step. System-driven."},
            {"name": "W_a_hebbian", "class": "M",
             "justification": "W_a updated: outer(a_oh, h) each step. System-driven."},
            {"name": "W_h_reservoir", "class": "U",
             "justification": "Fixed random reservoir. Could be structured, identity, or trained. System doesn't choose."},
            {"name": "tanh_activation", "class": "U",
             "justification": "tanh. Could be ReLU, sigmoid. System doesn't choose."},
            {"name": "spectral_radius", "class": "U",
             "justification": "0.95. Could be 0.8, 1.1. System doesn't choose."},
            {"name": "h_dim", "class": "U",
             "justification": "H_DIM=128. Could be 64 or 256. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
            {"name": "argmax_action", "class": "U",
             "justification": "argmax(W_a@h). Could be softmax sampling. System doesn't choose."},
        ]
