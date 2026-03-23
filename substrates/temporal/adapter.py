"""BaseSubstrate adapter for TemporalPrediction — prediction matrix substrate.

Killed: Step 437d. Convergence: pred_err→0 → W frozen → action locked (U22 violation).
TemporalMinimal has 0 U elements in self-assessment but never tested on navigation.
Kept as evidence for convergence kills exploration (U22).
"""
import copy
import numpy as np
import torch
from substrates.base import BaseSubstrate, Observation
from substrates.temporal.temporal import TemporalPrediction, TemporalMinimal


class TemporalPredictionAdapter(BaseSubstrate):
    """Wraps TemporalPrediction into BaseSubstrate protocol."""

    def __init__(self, d=256, n_act=4, device='cpu'):
        self._d = d
        self._n_act = n_act
        self._device = device
        self._sub = TemporalPrediction(d, n_act, device)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)[:self._d]
        if len(flat) < self._d:
            flat = np.pad(flat, (0, self._d - len(flat)))
        x = torch.from_numpy(flat)
        return self._sub.step(x)

    def get_state(self):
        return {
            "W": self._sub.W.clone().cpu().numpy(),
            "prev": self._sub.prev.clone().cpu().numpy() if self._sub.prev is not None else None,
            "pred_err": self._sub.pred_err,
        }

    def set_state(self, state):
        self._sub.W = torch.from_numpy(state["W"]).to(self._device)
        self._sub.prev = torch.from_numpy(state["prev"]).to(self._device) if state["prev"] is not None else None
        self._sub.pred_err = state["pred_err"]

    def frozen_elements(self):
        return [
            {"name": "W", "class": "M", "justification": "Prediction matrix updated by LMS rule every step"},
            {"name": "prev", "class": "M", "justification": "Previous observation stored for prediction"},
            {"name": "matmul_predict", "class": "I", "justification": "W@prev is the prediction. Removing = no prediction."},
            {"name": "subtract_error", "class": "I", "justification": "x - pred is error signal. Removing = no learning."},
            {"name": "outer_product_update", "class": "I", "justification": "outer(err, prev) is the LMS gradient. Removing = no adaptation."},
            {"name": "argmax_action", "class": "I", "justification": "Action from prediction chain endpoint. Removing = no action."},
            {"name": "lms_normalization", "class": "U", "justification": "Dividing by prev.dot(prev). Could be raw Hebbian (TemporalMinimal variant)."},
            {"name": "chain_depth_2", "class": "U", "justification": "Two-step chain (W@x then W@(W@x)). Could be depth 1. Not derived from data."},
            {"name": "abs_before_argmax", "class": "U", "justification": "abs() before argmax. Could use raw values. Not justified."},
            {"name": "mod_n_actions", "class": "U", "justification": "% n_actions wrapping. Could use top-K or softmax."},
            {"name": "clamp_denom", "class": "U", "justification": "clamp(min=1e-8). Numerical stability. Arbitrary epsilon."},
        ]

    def reset(self, seed: int):
        torch.manual_seed(seed)
        self._sub = TemporalPrediction(self._d, self._n_act, self._device)

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass


class TemporalMinimalAdapter(BaseSubstrate):
    """Wraps TemporalMinimal — the 0-U-element variant (self-assessed)."""

    def __init__(self, d=256, n_act=4, device='cpu'):
        self._d = d
        self._n_act = n_act
        self._device = device
        self._sub = TemporalMinimal(d, n_act, device)

    def process(self, observation):
        if isinstance(observation, Observation):
            obs = observation.data
        else:
            obs = observation
        flat = obs.flatten().astype(np.float32)[:self._d]
        if len(flat) < self._d:
            flat = np.pad(flat, (0, self._d - len(flat)))
        x = torch.from_numpy(flat)
        return self._sub.step(x)

    def get_state(self):
        return {
            "W": self._sub.W.clone().cpu().numpy(),
            "prev": self._sub.prev.clone().cpu().numpy() if self._sub.prev is not None else None,
            "pred_err": self._sub.pred_err,
        }

    def set_state(self, state):
        self._sub.W = torch.from_numpy(state["W"]).to(self._device)
        self._sub.prev = torch.from_numpy(state["prev"]).to(self._device) if state["prev"] is not None else None
        self._sub.pred_err = state["pred_err"]

    def frozen_elements(self):
        return [
            {"name": "W", "class": "M", "justification": "Prediction matrix updated by raw Hebbian every step"},
            {"name": "prev", "class": "M", "justification": "Previous observation stored for prediction"},
            {"name": "matmul_predict", "class": "I", "justification": "W@prev is the prediction. Removing = blind."},
            {"name": "subtract_error", "class": "I", "justification": "x - pred is error signal. Removing = no learning."},
            {"name": "outer_product_update", "class": "I", "justification": "outer(err, prev) is raw Hebbian. The unique least-squares step."},
            {"name": "argmax_action", "class": "I", "justification": "Action from W@x endpoint. Removing = no action."},
            {"name": "mod_n_actions", "class": "I", "justification": "% n_actions wrapping. Required for valid action output."},
        ]

    def reset(self, seed: int):
        torch.manual_seed(seed)
        self._sub = TemporalMinimal(self._d, self._n_act, self._device)

    @property
    def n_actions(self):
        return self._n_act

    def on_level_transition(self):
        pass
