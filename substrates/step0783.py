"""
step0783.py — TransitionHashSet783: track seen transitions without visit counts.

R3 hypothesis: tracking which transitions occurred (not how many times) enables
exploration without visit counts. State: set of (obs_hash, next_obs_hash) pairs.
NOT keyed by action. NOT per-(state,action). Graph ban CHECK: keyed by
(obs, obs_next) — cannot reconstruct per-(state,action) visit counts. PASS.

Action: for each action a, predict next obs hash via W. If hash ∈ seen_set →
already explored transition → skip. Pick first action whose predicted
transition hash is NOT in seen_set.

D(s) = {W, seen_transitions, running_mean}. L(s) = ∅.
"""
import numpy as np
from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame

D = 64
ETA = 0.01
MAX_SET_SIZE = 50_000  # prevent unbounded growth


def _hash_vec(v: np.ndarray) -> int:
    """Stable hash of a float32 vector via sign bits."""
    return int(np.packbits((v > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)


class TransitionHashSet783(BaseSubstrate):
    """Tracks seen (obs_hash, next_obs_hash) transitions.

    Action: pick first action whose predicted transition is novel.
    Falls back to random if all transitions seen.
    D(s) = {W, seen_transitions, running_mean}. L(s) = ∅.
    """

    def __init__(self, n_actions: int = 4, seed: int = 0):
        self._n_actions = n_actions
        self._seed = seed
        rng = np.random.RandomState(seed)
        d_in = D + n_actions
        self.W = rng.randn(D, d_in).astype(np.float32) * 0.01  # M: forward model
        self.seen_transitions: set = set()                       # M: novel tracker
        self.running_mean = np.zeros(D, np.float32)              # M
        self._n_obs = 0
        self._prev_enc = None
        self._prev_hash = None
        self._prev_action = None
        self._rng = np.random.RandomState(seed + 1)

    def _encode(self, obs: np.ndarray) -> np.ndarray:
        x = _enc_frame(obs)[:D]
        self._n_obs += 1
        alpha = 1.0 / self._n_obs
        self.running_mean = (1 - alpha) * self.running_mean + alpha * x
        return x - self.running_mean

    def process(self, observation) -> int:
        import numpy as np
        observation = np.asarray(observation, dtype=np.float32)
        x = self._encode(observation)
        curr_hash = _hash_vec(x)

        # Update seen_transitions and Hebbian W
        if self._prev_enc is not None:
            transition = (self._prev_hash, curr_hash)
            if len(self.seen_transitions) < MAX_SET_SIZE:
                self.seen_transitions.add(transition)
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[self._prev_action] = 1.0
            inp = np.concatenate([self._prev_enc, a_oh])
            self.W += ETA * np.outer(x, inp)

        # Action: pick first action with novel predicted transition
        action = None
        for a in range(self._n_actions):
            a_oh = np.zeros(self._n_actions, np.float32)
            a_oh[a] = 1.0
            inp = np.concatenate([x, a_oh])
            pred_next = self.W @ inp
            pred_hash = _hash_vec(pred_next)
            t = (curr_hash, pred_hash)
            if t not in self.seen_transitions:
                action = a
                break

        if action is None:
            action = self._rng.randint(0, self._n_actions)  # all seen → random

        self._prev_enc = x.copy()
        self._prev_hash = curr_hash
        self._prev_action = action
        return action

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def get_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "seen_transitions": set(self.seen_transitions),
            "running_mean": self.running_mean.copy(),
            "_n_obs": self._n_obs,
            "_prev_enc": self._prev_enc.copy() if self._prev_enc is not None else None,
            "_prev_hash": self._prev_hash,
            "_prev_action": self._prev_action,
        }

    def set_state(self, state: dict) -> None:
        self.W = state["W"].copy()
        self.seen_transitions = set(state["seen_transitions"])
        self.running_mean = state["running_mean"].copy()
        self._n_obs = state["_n_obs"]
        self._prev_enc = state["_prev_enc"].copy() if state["_prev_enc"] is not None else None
        self._prev_hash = state["_prev_hash"]
        self._prev_action = state["_prev_action"]

    def reset(self, seed: int) -> None:
        self._prev_enc = None
        self._prev_hash = None
        self._prev_action = None

    def on_level_transition(self) -> None:
        self._prev_enc = None
        self._prev_hash = None
        self._prev_action = None

    def frozen_elements(self) -> list:
        return [
            {"name": "W_hebbian", "class": "M",
             "justification": "W updated by every transition. System-driven."},
            {"name": "seen_transitions", "class": "M",
             "justification": "Set of seen (obs_hash, next_hash) pairs. Grows with experience. System-driven."},
            {"name": "running_mean", "class": "M",
             "justification": "Running mean tracks obs distribution. System-driven."},
            {"name": "first_novel_action_rule", "class": "U",
             "justification": "Picks first (not best) novel action. Could be most-novel or random among novel. System doesn't choose."},
            {"name": "hash_via_sign_bits", "class": "U",
             "justification": "Sign-bit hash of 64-dim vec. Could be any locality-sensitive hash. System doesn't choose."},
            {"name": "enc_truncate_64", "class": "U",
             "justification": "64-dim. System doesn't choose."},
            {"name": "eta_learning_rate", "class": "U",
             "justification": "eta=0.01. System doesn't choose."},
            {"name": "max_set_size", "class": "U",
             "justification": "MAX=50K. System doesn't choose."},
        ]
