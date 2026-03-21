"""Action-Conditioned World Model.

Extends InfiniteHorizonModel to accept action inputs.
Actions influence motion prediction - the model learns:
  "If I press RIGHT, the paddle moves right"

Supports:
1. Discrete actions (keyboard: up/down/left/right/none)
2. Continuous actions (mouse: dx, dy)
3. No actions (unconditional generation)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from genesis.pilot.motion_model_v2 import (
    spatial_transformer,
    BoundedSlotAttention,
    MotionEncoder,
    MotionDecoder,
)


class ActionEncoder(nn.Module):
    """Encode actions into feature space."""

    def __init__(self, num_actions: int = 5, action_dim: int = 32, continuous: bool = False):
        """
        Args:
            num_actions: Number of discrete actions (e.g., 5 = up/down/left/right/none)
            action_dim: Output dimension for action embedding
            continuous: If True, expect continuous actions [B, 2] (dx, dy)
        """
        super().__init__()
        self.continuous = continuous
        self.action_dim = action_dim

        if continuous:
            # Continuous action encoder (e.g., mouse movement)
            self.encoder = nn.Sequential(
                nn.Linear(2, action_dim),
                nn.GELU(),
                nn.Linear(action_dim, action_dim),
            )
        else:
            # Discrete action embedding
            self.embedding = nn.Embedding(num_actions, action_dim)

    def forward(self, actions):
        """
        Args:
            actions: [B] discrete or [B, 2] continuous

        Returns:
            action_emb: [B, action_dim]
        """
        if self.continuous:
            return self.encoder(actions)
        else:
            return self.embedding(actions)


class ActionConditionedModel(nn.Module):
    """World model with action conditioning.

    Architecture:
    1. Encode frame pair -> visual features
    2. Encode action -> action features
    3. Combine visual + action features
    4. Slot attention for object tracking
    5. Predict motion conditioned on action
    6. Warp + residual -> next frame
    """

    def __init__(
        self,
        base_channels: int = 48,
        num_slots: int = 8,
        slot_dim: int = 64,
        slot_decay: float = 0.95,
        num_actions: int = 5,
        action_dim: int = 32,
        continuous_actions: bool = False,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.num_actions = num_actions

        # Visual encoder
        self.encoder = MotionEncoder(in_channels=6, base_channels=base_channels)
        feat_channels = self.encoder.out_channels

        # Action encoder
        self.action_encoder = ActionEncoder(
            num_actions=num_actions,
            action_dim=action_dim,
            continuous=continuous_actions,
        )

        # Action -> spatial influence
        self.action_to_spatial = nn.Sequential(
            nn.Linear(action_dim, feat_channels),
            nn.GELU(),
            nn.Linear(feat_channels, feat_channels),
        )

        # Combine visual + action
        self.combine_action = nn.Conv2d(feat_channels * 2, feat_channels, 1)

        # Slot attention
        self.to_slot_input = nn.Conv2d(feat_channels, slot_dim, 1)
        self.slot_attention = BoundedSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            decay=slot_decay,
        )

        # Slot -> spatial
        self.slot_to_spatial = nn.Sequential(
            nn.Linear(slot_dim, feat_channels * 4),
            nn.GELU(),
            nn.Linear(feat_channels * 4, feat_channels),
        )

        self.combine_slot = nn.Conv2d(feat_channels * 2, feat_channels, 1)

        # Decoder
        self.decoder = MotionDecoder(in_channels=feat_channels, base_channels=base_channels)

        # Internal state
        self._slots = None

    def reset_state(self):
        """Reset internal state for new sequence."""
        self._slots = None

    def step(self, prev, curr, action=None):
        """Single prediction step.

        Args:
            prev: [B, 3, H, W] - previous frame
            curr: [B, 3, H, W] - current frame
            action: [B] discrete or [B, 2] continuous, or None for unconditional

        Returns:
            next_pred: [B, 3, H, W] - predicted next frame
        """
        B = curr.shape[0]

        # Encode visual
        feat = self.encoder(prev, curr)  # [B, C, 16, 16]

        # Encode and apply action
        if action is not None:
            action_emb = self.action_encoder(action)  # [B, action_dim]
            action_feat = self.action_to_spatial(action_emb)  # [B, C]
            action_feat = action_feat.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
            action_feat = action_feat.expand(-1, -1, feat.shape[2], feat.shape[3])

            feat = torch.cat([feat, action_feat], dim=1)
            feat = self.combine_action(feat)

        # Slot attention
        slot_input = self.to_slot_input(feat)
        slot_input = slot_input.flatten(2).transpose(1, 2)
        self._slots = self.slot_attention(slot_input, self._slots)

        # Slot influence
        slot_feat = self.slot_to_spatial(self._slots)
        slot_feat = slot_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        slot_feat = slot_feat.expand(-1, -1, feat.shape[2], feat.shape[3])

        combined = torch.cat([feat, slot_feat], dim=1)
        combined = self.combine_slot(combined)

        # Decode to motion + residual
        motion, residual = self.decoder(combined)

        # Warp
        warped = spatial_transformer(curr, motion)
        pred = torch.clamp(warped + residual, 0, 1)

        return pred

    def forward(self, frames, actions=None, rollout_steps=1):
        """Forward pass with optional actions.

        Args:
            frames: [B, T, 3, H, W]
            actions: [B, T-2] discrete or [B, T-2, 2] continuous, or None
            rollout_steps: Autoregressive rollout steps during training

        Returns:
            predictions: [B, T-2, 3, H, W]
        """
        B, T, C, H, W = frames.shape
        predictions = []

        self.reset_state()

        for t in range(T - 2):
            if t < rollout_steps or rollout_steps == 1:
                prev = frames[:, t]
                curr = frames[:, t + 1]
            else:
                prev = predictions[-2] if len(predictions) >= 2 else frames[:, t]
                curr = predictions[-1] if len(predictions) >= 1 else frames[:, t + 1]

            # Get action for this timestep
            action = None
            if actions is not None:
                if actions.dim() == 2:  # [B, T-2] discrete
                    action = actions[:, t]
                else:  # [B, T-2, 2] continuous
                    action = actions[:, t]

            pred = self.step(prev, curr, action)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def generate(self, seed_frames, actions, num_steps=None):
        """Generate frames conditioned on actions.

        Args:
            seed_frames: [B, 2, 3, H, W] - initial frames
            actions: [B, num_steps] or [B, num_steps, 2] - actions to execute
            num_steps: Override number of steps (default: len(actions))

        Returns:
            generated: [B, num_steps, 3, H, W]
        """
        self.reset_state()
        self.eval()

        B = seed_frames.shape[0]

        if num_steps is None:
            num_steps = actions.shape[1]

        history = [seed_frames[:, i] for i in range(seed_frames.shape[1])]
        generated = []

        with torch.no_grad():
            for step in range(num_steps):
                prev = history[-2]
                curr = history[-1]

                # Get action
                if actions.dim() == 2:
                    action = actions[:, step] if step < actions.shape[1] else actions[:, -1]
                else:
                    action = actions[:, step] if step < actions.shape[1] else actions[:, -1]

                pred = self.step(prev, curr, action)
                generated.append(pred)

                history.append(pred)
                if len(history) > 3:
                    history.pop(0)

        return torch.stack(generated, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def infer_actions_from_motion(frames, num_actions=5):
    """Infer discrete actions from frame motion (for datasets without actions).

    Uses optical flow direction to guess action:
    0 = none, 1 = up, 2 = down, 3 = left, 4 = right
    """
    B, T, C, H, W = frames.shape
    actions = torch.zeros(B, T - 2, dtype=torch.long, device=frames.device)

    for t in range(T - 2):
        # Simple motion estimation: center of mass change
        curr = frames[:, t + 1].mean(dim=1)  # [B, H, W]
        next_frame = frames[:, t + 2].mean(dim=1)

        # Compute center of mass
        y_grid = torch.arange(H, device=frames.device).float().view(1, H, 1)
        x_grid = torch.arange(W, device=frames.device).float().view(1, 1, W)

        curr_sum = curr.sum(dim=(1, 2), keepdim=True) + 1e-6
        next_sum = next_frame.sum(dim=(1, 2), keepdim=True) + 1e-6

        curr_y = (curr * y_grid).sum(dim=(1, 2)) / curr_sum.squeeze()
        curr_x = (curr * x_grid).sum(dim=(1, 2)) / curr_sum.squeeze()
        next_y = (next_frame * y_grid).sum(dim=(1, 2)) / next_sum.squeeze()
        next_x = (next_frame * x_grid).sum(dim=(1, 2)) / next_sum.squeeze()

        dy = next_y - curr_y
        dx = next_x - curr_x

        # Discretize
        for b in range(B):
            if abs(dy[b]) > abs(dx[b]):
                if dy[b] > 1:
                    actions[b, t] = 2  # down
                elif dy[b] < -1:
                    actions[b, t] = 1  # up
            else:
                if dx[b] > 1:
                    actions[b, t] = 4  # right
                elif dx[b] < -1:
                    actions[b, t] = 3  # left
            # else: 0 = none

    return actions


if __name__ == '__main__':
    print("Testing Action-Conditioned Model...")

    # Test discrete actions
    model = ActionConditionedModel(
        base_channels=48,
        num_slots=8,
        slot_dim=64,
        num_actions=5,
        continuous_actions=False,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward with actions
    frames = torch.randn(2, 10, 3, 64, 64).sigmoid()
    actions = torch.randint(0, 5, (2, 8))  # [B, T-2]

    preds = model(frames, actions=actions)
    print(f"Forward with actions: {preds.shape}")

    # Test forward without actions
    preds_no_action = model(frames, actions=None)
    print(f"Forward without actions: {preds_no_action.shape}")

    # Test generation
    seed = frames[:, :2]
    gen_actions = torch.randint(0, 5, (2, 50))
    generated = model.generate(seed, gen_actions)
    print(f"Generate 50 steps: {generated.shape}")

    # Test action inference
    inferred = infer_actions_from_motion(frames)
    print(f"Inferred actions: {inferred.shape}")

    # Test continuous actions
    model_cont = ActionConditionedModel(
        base_channels=48,
        num_slots=8,
        slot_dim=64,
        action_dim=32,
        continuous_actions=True,
    )
    print(f"\nContinuous model params: {model_cont.count_parameters():,}")

    cont_actions = torch.randn(2, 8, 2)  # [B, T-2, 2]
    preds_cont = model_cont(frames, actions=cont_actions)
    print(f"Forward with continuous actions: {preds_cont.shape}")

    print("\nAll tests passed!")
