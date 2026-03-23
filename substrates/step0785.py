"""
step0785.py — ForwardModelRefinement785: forward model + transition refinement.

R3 hypothesis: combining dynamics prediction (W) with encoding adaptation
(transition-triggered refinement, K=12→20) yields two active D components →
stronger R3_cf than either alone.

674 encoding + transition refinement (ref dict from C stats). Plus global W
forward model from 778. Action: prediction-contrast (780 style).
Graph ban: W is global (not per-(state,action)). ref dict is per-observation
(not per-(state,action)). PASS.

D(s) = {W, running_mean, ref}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

D = 64
DIM = 256
K_NAV = 12
ETA = 0.01
MIN_OBS_REFINE = 8


class ForwardModelRefinement785(BaseSubstrate):
    """Forward model W + transition-triggered hash refinement.

    Encoding: avgpool16 + running_mean + LSH K=12 with adaptive refinement.
    Forward model W: same Hebbian update as 778.
    Action: prediction-contrast (780) — argmax_a ||W(obs,a) - obs||.
    D(s) = {W, running_mean, ref}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        # Forward model
        d_in = D + n_actions
        self.W = rng.randn(D, d_in).astype(np.float32) * 0.01  # M
        self.running_mean = np.zeros(DIM, np.float32)             # M
        self._n_obs = 0
        # Encoding / refinement
        self.H_nav = rng.randn(K_NAV, DIM).astype(np.float32)  # U: frozen
        self.ref = {}   # M: refined hash planes per node
        self.C = {}     # M: transition centroid stats per (node, successor)
        self.G_counts = {}  # M: (node, succ) -> count (for refinement trigger)
        self._prev_full = None   # full DIM encoding
        self._prev_enc64 = None  # truncated D=64 encoding
        self._prev_action = None
        self._prev_node = None

    def _hash_node(self, x: np.ndarray) -> int:
        n = int(np.packbits((self.H_nav @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def _encode_full(self, obs: np.ndarray):
        x = _enc_frame(obs)  # 256-dim
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        x_centered = x - self.running_mean
        node = self._hash_node(x_centered)
        return x_centered, node

    def _maybe_refine(self, prev_node, curr_node, prev_x: np.ndarray, curr_x: np.ndarray):
        """Trigger refinement: if transition (prev_node→curr_node) seen ≥ MIN_OBS_REFINE."""
        key = (prev_node, curr_node)
        self.G_counts[key] = self.G_counts.get(key, 0) + 1
        # Update centroid stats
        s, c = self.C.get(key, (np.zeros(DIM, np.float64), 0))
        self.C[key] = (s + curr_x.astype(np.float64), c + 1)

        # Refinement: if this node has 2+ successors with ≥ MIN_OBS each, split
        if isinstance(prev_node, int) and prev_node not in self.ref:
            successors = {}
            for (pn, cn), cnt in self.G_counts.items():
                if pn == prev_node:
                    successors[cn] = cnt
            if len(successors) >= 2:
                top = sorted(successors, key=successors.get, reverse=True)[:2]
                c0 = self.C.get((prev_node, top[0]))
                c1 = self.C.get((prev_node, top[1]))
                if c0 and c1 and c0[1] >= 3 and c1[1] >= 3:
                    diff = (c0[0] / c0[1]) - (c1[0] / c1[1])
                    nm = np.linalg.norm(diff)
                    if nm > 1e-8:
                        self.ref[prev_node] = (diff / nm).astype(np.float32)

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x_full, node = self._encode_full(observation)
        x64 = x_full[:D]

        # Update forward model W
        if self._prev_enc64 is not None:
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc64, a_oh])
            # Delta rule: minimize ||W@inp - x64||^2
            pred_err = self.W @ inp - x64
            self.W -= ETA * np.outer(pred_err, inp)

        # Update refinement
        if self._prev_node is not None and self._prev_full is not None:
            self._maybe_refine(self._prev_node, node, self._prev_full, x_full)

        # Action: prediction-contrast
        best_a, best_score = 0, -1.0
        for a in range(self._n_actions):
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[a] = 1.0
            inp = np.concatenate([x64, a_oh])
            pred = self.W @ inp
            score = float(np.sum((pred - x64) ** 2))
            if score > best_score:
                best_score = score
                best_a = a

        self._prev_full = x_full.copy()
        self._prev_enc64 = x64.copy()
        self._prev_action = best_a
        self._prev_node = node
        return best_a

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        import copy
        return {
            "W": self.W.copy(),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "ref": copy.deepcopy(self.ref),
            "C": copy.deepcopy(self.C),
            "G_counts": dict(self.G_counts),
            "_prev_full": self._prev_full.copy() if self._prev_full is not None else None,
            "_prev_enc64": self._prev_enc64.copy() if self._prev_enc64 is not None else None,
            "_prev_action": self._prev_action,
            "_prev_node": self._prev_node,
        }

    def set_state(self, state: dict) -> None:
        import copy
        self.W = state["W"].copy()
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self.ref = copy.deepcopy(state["ref"])
        self.C = copy.deepcopy(state["C"])
        self.G_counts = dict(state["G_counts"])
        self._prev_full = state["_prev_full"].copy() if state["_prev_full"] is not None else None
        self._prev_enc64 = state["_prev_enc64"].copy() if state["_prev_enc64"] is not None else None
        self._prev_action = state["_prev_action"]
        self._prev_node = state["_prev_node"]

    def reset(self, seed: int) -> None:
        self._prev_full = None
        self._prev_enc64 = None
        self._prev_action = None
        self._prev_node = None

    def on_level_transition(self) -> None:
        self._prev_full = None
        self._prev_enc64 = None
        self._prev_action = None
        self._prev_node = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W_hebbian", "class": "M",
             "justification": "W updated by every transition. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "ref_hyperplanes", "class": "M",
             "justification": "Refinement planes derived from observed transition centroids. System-driven."},
            {"name": "G_counts_C_stats", "class": "M",
             "justification": "Transition counts/centroids drive refinement. System-driven."},
            {"name": "max_predicted_change_rule", "class": "I",
             "justification": "argmax_a ||W(obs,a)-obs||. Removing loses all structure."},
            {"name": "H_nav_planes", "class": "U",
             "justification": "K=12 frozen random LSH planes. System doesn't choose."},
            {"name": "enc_truncate_64", "class": "U",
             "justification": "64-dim for W. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
            {"name": "min_obs_refine", "class": "U",
             "justification": "MIN_OBS=8. System doesn't choose threshold."},
        ]
