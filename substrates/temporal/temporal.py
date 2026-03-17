#!/usr/bin/env python3
"""
Temporal Prediction Substrate — Phase 2

State = prediction matrix W (d x d).
Step: predict x_{t+1} from x_t, compare reality, update W, act via chain.

Prediction error simultaneously provides:
  - Similarity metric (U20): matmul is continuous -> similar inputs -> similar outputs
  - Self-modification signal (R2): error-driven update changes prediction behavior
  - Self-test (R4): every prediction is tested against reality next step
  - Novelty detector: high error = unfamiliar transition

R3 AUDIT (honest):
  MODIFIED (2): W, prev
  IRREDUCIBLE (4): matmul, subtract, outer_product, argmax
  UNJUSTIFIED (5): update_normalization, chain_depth=2, abs, %n_actions, clamp
  R3: FAIL (5 unjustified). Fewest of any Phase 2 substrate, but still not zero.

Frozen frame: {matmul, subtract, outer_product, abs, argmax}
  All mathematical primitives. No domain-specific operations (no cosine,
  no attract, no spawn, no threshold formula).

8 lines of core logic. Not vectors. Not cosine. Not a codebook.
"""

import torch


class TemporalMinimal:
    """
    Maximum reduction variant. 5 lines of core logic.

    R3 AUDIT:
      MODIFIED (2): W, prev
      IRREDUCIBLE (5): matmul, subtract, outer_product, argmax, %n_actions
      UNJUSTIFIED (0): NONE
      R3: potentially PASS (requires external validation)

    Differences from TemporalPrediction:
      - No update normalization (raw Hebbian: outer(err, prev))
      - Depth-1 chain (W@x, not W@(W@x))
      - No abs() before argmax
    """

    def __init__(self, d, n_actions, device='cpu'):
        self.d = d
        self.n_actions = n_actions
        self.device = device
        self.W = torch.zeros(d, d, device=device)
        self.prev = None
        self.pred_err = 0.0

    def step(self, x):
        x = x.to(self.device).float()
        if self.prev is None:
            self.prev = x.clone()
            return 0

        err = x - self.W @ self.prev                          # predict + error
        self.pred_err = err.norm().item()
        self.W += torch.outer(err, self.prev)                  # raw Hebbian
        action = (self.W @ x).argmax().item() % self.n_actions # depth-1 chain
        self.prev = x.clone()
        return action


class TemporalPrediction:
    """
    State = W. W encodes transitions, defines metric, determines actions.
    The prediction model IS the substrate.
    """

    def __init__(self, d, n_actions, device='cpu'):
        self.d = d
        self.n_actions = n_actions
        self.device = device
        self.W = torch.zeros(d, d, device=device)   # prediction model
        self.prev = None
        self.pred_err = 0.0                          # diagnostic

    def step(self, x):
        x = x.to(self.device).float()

        if self.prev is None:
            self.prev = x.clone()
            return 0

        # PREDICT: what did W think x would be?
        pred = self.W @ self.prev
        err = x - pred
        self.pred_err = err.norm().item()

        # UPDATE: LMS rule (unique least-squares gradient step)
        denom = self.prev.dot(self.prev).clamp(min=1e-8)
        self.W += torch.outer(err, self.prev) / denom

        # ACT: two-step prediction chain through W
        p1 = self.W @ x
        p2 = self.W @ p1
        action = p2.abs().argmax().item() % self.n_actions

        self.prev = x.clone()
        return action
