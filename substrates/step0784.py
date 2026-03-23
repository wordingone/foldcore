"""
step0784.py — EncodingOnly784: minimal D-only substrate.

R3 hypothesis: running mean adaptation alone (obs distribution centering)
produces positive R3_cf. This is the cleanest test of whether D(s) = {encoding
adaptation} transfers, with L(s) = ∅ (no visit counts, no forward model).

674 encoding: avgpool16, running_mean centering, LSH k=12. No graph. Random action.
D(s) = {running_mean}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

K = 12   # LSH planes (matches 674 nav)
DIM = 256


class EncodingOnly784(BaseSubstrate):
    """674 encoding with running mean. No graph, no forward model. Random action.

    D(s) = {running_mean}. L(s) = ∅.
    Minimal test: does obs distribution centering alone transfer?
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        self.H_nav = rng.randn(K, DIM).astype(np.float32)  # U: frozen random planes
        self.running_mean = np.zeros(DIM, np.float32)       # M: obs distribution
        self._n_obs = 0
        self._rng = np.random.RandomState(seed + 1)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)  # 256-dim, within-frame centered
        # Update global running mean (across all obs seen)
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        self._encode(observation)  # updates running_mean
        action = self._rng.randint(0, self._n_actions)
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
        }

    def set_state(self, state: dict) -> None:
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]

    def reset(self, seed: int) -> None:
        # Preserve running_mean (D(s)) — only clear episode-local state
        pass

    def on_level_transition(self) -> None:
        pass

    def frozen_elements(self) -> list:
        return [
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution across all steps. System-driven."},
            {"name": "random_action", "class": "I",
             "justification": "No selection mechanism. Removing = identical random walk. Irreducible baseline."},
            {"name": "avgpool16_enc", "class": "U",
             "justification": "16x16 average pooling. Could be 8x8 or conv. System doesn't choose."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "K=12 frozen random LSH planes. System doesn't choose count or direction."},
            {"name": "running_mean_alpha", "class": "U",
             "justification": "Online mean: alpha=1/n. Could be exponential decay. System doesn't choose."},
        ]
