"""Micro Slot: Minimal slot attention model for 16x16 @ 8 frames.

Tests whether slot attention helps occlusion recovery at minimal scale.
Matches micro_baseline.py parameter count (~65K) for fair comparison.

Architecture:
- Encoder: 16x16 grayscale -> 4x4 features
- Slot attention: 4 slots (square + occluder + background + spare)
- Slots persist across frames (key difference from baseline)
- Decoder: slots -> 16x16 predictions
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroSlotAttention(nn.Module):
    """Minimal slot attention for micro experiments."""

    def __init__(self, num_slots=4, slot_dim=16, input_dim=16, num_iterations=3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # Learnable slot init
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)

        # Projections
        self.norm = nn.LayerNorm(input_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.scale = slot_dim ** -0.5

    def forward(self, inputs, slots=None):
        """
        Args:
            inputs: (B, N, input_dim) - spatial features
            slots: (B, K, slot_dim) - previous slots or None

        Returns:
            slots: (B, K, slot_dim)
            attn: (B, K, N)
        """
        B, N, _ = inputs.shape

        if slots is None:
            slots = self.slot_mu.expand(B, -1, -1).clone()

        inputs = self.norm(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.project_q(slots)

            attn = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = F.softmax(attn, dim=1)  # Compete over slots

            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)

        return slots, attn


class MicroSlot(nn.Module):
    """Micro slot model for 16x16 @ 8 frames.

    Target: ~65K params (match baseline)
    """

    def __init__(self, channels=16, num_slots=4, slot_dim=16):
        super().__init__()
        self.channels = channels
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Encoder: 16x16 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=4, stride=4),
            nn.GELU(),
        )

        # Slot attention
        self.slot_attention = MicroSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=channels,
            num_iterations=3,
        )

        # Decode slots to spatial
        self.slot_to_spatial = nn.Sequential(
            nn.Linear(slot_dim, channels * 4 * 4),
        )

        # Combine slots (learned mixing)
        self.slot_mixer = nn.Conv2d(num_slots * channels, channels, 1)

        # Decoder: 4x4 -> 16x16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, kernel_size=4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, T, 1, 16, 16]

        Returns:
            predictions: [B, T-2, 1, 16, 16]
        """
        B, T, C, H, W = frames.shape
        predictions = []

        # Slots persist across time
        slots = None

        for t in range(T - 2):
            # Use frames t and t+1 to predict t+2
            frame_t = frames[:, t]
            frame_t1 = frames[:, t + 1]

            # Encode current frame
            feat = self.encoder(frame_t1)  # [B, channels, 4, 4]
            feat_flat = feat.flatten(2).transpose(1, 2)  # [B, 16, channels]

            # Update slots
            slots, attn = self.slot_attention(feat_flat, slots)

            # Decode slots to spatial features
            slot_spatial = self.slot_to_spatial(slots)  # [B, K, channels*16]
            slot_spatial = slot_spatial.reshape(B, self.num_slots, self.channels, 4, 4)

            # Mix slots
            mixed = slot_spatial.reshape(B, self.num_slots * self.channels, 4, 4)
            mixed = self.slot_mixer(mixed)  # [B, channels, 4, 4]

            # Decode to frame
            pred = self.decoder(mixed)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing MicroSlot...")

    # Scale to match baseline (~65K params)
    model = MicroSlot(channels=48, num_slots=4, slot_dim=48)
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    frames = torch.randn(4, 8, 1, 16, 16)
    with torch.no_grad():
        preds = model(frames)

    print(f"Input: {frames.shape}")
    print(f"Output: {preds.shape}")
    print(f"Expected: (4, 6, 1, 16, 16)")
    print(f"Match: {preds.shape == (4, 6, 1, 16, 16)}")
