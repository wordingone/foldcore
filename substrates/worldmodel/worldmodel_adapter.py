"""
worldmodel_adapter.py — Genesis World Model wrapped as BaseSubstrate.

KILL REASON (Phase 2, ~Step 580): Genesis is a video world model (64M parameters,
sub-1B target). It is fundamentally incompatible with the BaseSubstrate interface.

Specific failures:
  1. Genesis requires a trained checkpoint. The substrate cannot function without
     pre-training on video data (OpenVid, etc.). There is no unsupervised
     initialization that produces meaningful outputs.
  2. The model generates video frames (720p), not action selections. There is no
     principled mapping from generated frames → navigation actions.
  3. Genesis requires 24GB GPU VRAM for inference. On CPU, inference is ~0.01 FPS.
     The 5-minute experiment cap is violated for any non-trivial evaluation.
  4. Genesis is a different paradigm entirely: predictive/generative, not
     reactive/navigational. It predicts what the world will look like next,
     not what action to take.
  5. R3: The model weights (64M params) could be M if fine-tuned during inference.
     But Genesis is evaluated frozen post-training — weights are I, not M.

This adapter is a STUB. It cannot be instantiated without a checkpoint.
It documents WHY Genesis cannot be a BaseSubstrate.

ConstitutionalJudge verdict:
  R1: FAIL (supervised pre-training required; cannot function without external training signal)
  R2: FAIL (weights frozen post-training; no online adaptation)
  R3: FAIL (architecture not designed for navigation; action mapping is U)
  R5: PASS (inference computation frozen)
  NOTE: Genesis is not a substrate candidate. It is a world model — a different
  problem class. Kill was correct and final.

Usage:
  from substrates.worldmodel.worldmodel_adapter import WorldModelAdapter
  # NOTE: Will raise NotImplementedError — intentional.
  judge = ConstitutionalJudge()
  results = judge.audit(WorldModelAdapter)
"""
import numpy as np

from substrates.base import BaseSubstrate
from substrates.step0674 import _enc_frame, DIM


class WorldModelAdapter(BaseSubstrate):
    """Genesis world model adapter stub.

    CANNOT be instantiated for real inference — documents WHY Genesis
    is not a viable substrate.

    R3 hypothesis: FAIL. Requires checkpoint (R1=FAIL), frozen post-training
    (R2=FAIL), no action mapping (U), 13+ architecture hyperparameters (U).
    """

    def __init__(self, n_actions: int = 4, seed: int = 0, checkpoint: str = None):
        self._n_actions = n_actions
        self._seed = seed
        self._t = 0
        self._checkpoint = checkpoint

        if checkpoint is None:
            # Document the failure mode — don't crash immediately for judge audit
            self._stub_mode = True
        else:
            self._stub_mode = False
            raise NotImplementedError(
                "WorldModelAdapter requires a Genesis checkpoint. "
                "Genesis is killed — use this adapter only for R3 audit."
            )

    def process(self, observation) -> int:
        self._t += 1
        if self._stub_mode:
            # Stub: deterministic action from step count
            return self._t % self._n_actions
        raise NotImplementedError("WorldModelAdapter: no checkpoint available")

    def get_state(self) -> dict:
        return {"t": self._t, "stub_mode": self._stub_mode}

    def set_state(self, state: dict) -> None:
        self._t = state["t"]
        self._stub_mode = state["stub_mode"]

    def frozen_elements(self) -> list:
        return [
            {"name": "model_weights_64M", "class": "I",
             "justification": "64M frozen transformer weights (post-training). Removing destroys generation."},
            {"name": "pre_training_required", "class": "U",
             "justification": "Model requires pre-training on video data. Cannot function from random init. R1 VIOLATION."},
            {"name": "generated_frame_to_action", "class": "U",
             "justification": "No principled mapping from generated video frame to action index. System doesn't choose."},
            {"name": "model_architecture", "class": "U",
             "justification": "Transformer architecture (heads, layers, d_model) all unjustified for navigation."},
            {"name": "tokenizer", "class": "U",
             "justification": "Video tokenizer (VQVAE). Could be different tokenizer. System doesn't choose."},
            {"name": "training_data", "class": "U",
             "justification": "OpenVid training data. System doesn't choose what videos to train on."},
        ]

    def reset(self, seed: int) -> None:
        self._t = 0

    def on_level_transition(self) -> None:
        pass

    @property
    def n_actions(self) -> int:
        return self._n_actions
