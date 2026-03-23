"""BaseSubstrate adapter for SelfRef — the self-referential codebook.

Killed: Step 417-420. R3 FAIL (5 U elements). Uses cosine matching + attract
(LVQ mechanism — codebook ban applies). Kept as evidence for constraint map.
"""
import copy
import numpy as np
import torch
from substrates.base import BaseSubstrate, Observation
from substrates.selfref.selfref import SelfRef


class SelfRefAdapter(BaseSubstrate):
    """Wraps SelfRef into BaseSubstrate protocol."""

    def __init__(self, d=256, n_act=4, device='cpu'):
        self._d = d
        self._n_act = n_act
        self._device = device
        self._sub = SelfRef(d, device=device)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        x = torch.from_numpy(obs.flatten()[:self._d].astype(np.float32))
        if len(x) < self._d:
            x = torch.nn.functional.pad(x, (0, self._d - len(x)))
        return self._sub.step(x, self._n_act)

    def get_state(self):
        return {
            "V": self._sub.V.clone().cpu().numpy(),
            "d": self._sub.d,
        }

    def set_state(self, state):
        self._sub.V = torch.from_numpy(state["V"]).to(self._device)
        self._sub.d = state["d"]

    def frozen_elements(self):
        return [
            {"name": "V", "class": "M", "justification": "Codebook entries modified by attract on every step"},
            {"name": "cosine_matching", "class": "I", "justification": "Dot product + argmax is the compare operation. Removing = blind."},
            {"name": "chain_self_reference", "class": "I", "justification": "V @ V[w0] re-match is the chain. Removing = no self-reference."},
            {"name": "argmax_action", "class": "I", "justification": "Action from chain endpoint index. Removing = no action."},
            {"name": "F_normalize", "class": "U", "justification": "Unit sphere normalization. Cosine-specific. Could use L2 or unnormalized."},
            {"name": "lr_formula", "class": "U", "justification": "lr = 1 - sim. Designer-chosen. Could be constant or adaptive."},
            {"name": "chain_depth_2", "class": "U", "justification": "Two-step chain (obs→w0→w1). Could be 1 or 3. Not derived from data."},
            {"name": "spawn_threshold_median", "class": "U", "justification": "Median of Gram matrix max. Self-derived but formula is frozen."},
            {"name": "attract_lerp", "class": "U", "justification": "Linear interpolation attract. Could be any update rule."},
        ]

    def reset(self, seed: int):
        torch.manual_seed(seed)
        self._sub = SelfRef(self._d, device=self._device)

    def set_state(self, state):
        self._sub.V = torch.from_numpy(state["V"]).to(self._device)

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
