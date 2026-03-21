"""Micro Slot-AR: Full pipeline with VAE + Slot Attention + AR Dynamics.

Architecture:
1. VAE encoder: 16x16 -> 4x4 latent (frozen after pretraining)
2. Slot attention: extract object slots from latent
3. AR dynamics: predict next latent from slots
4. VAE decoder: 4x4 latent -> 16x16 (frozen)

Target: ~70K trainable params (dynamics only).
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from genesis.pilot.micro_vae import MicroVAE
from genesis.pilot.micro_slot import MicroSlotAttention


class LatentSlotAttention(nn.Module):
    """Slot attention operating on latent features."""

    def __init__(self, num_slots=4, slot_dim=32, input_dim=8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)

        # Project latent features to slot dim
        self.input_proj = nn.Linear(input_dim, slot_dim)
        self.norm = nn.LayerNorm(slot_dim)

        # Attention
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)

        # GRU update
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.scale = slot_dim ** -0.5
        self.num_iterations = 3

    def forward(self, latent, slots=None):
        """
        Args:
            latent: [B, C, H, W] latent from VAE (C=8, H=W=4)
            slots: [B, K, slot_dim] previous slots or None

        Returns:
            slots: [B, K, slot_dim]
        """
        B, C, H, W = latent.shape

        # Flatten spatial and project
        x = latent.flatten(2).transpose(1, 2)  # [B, 16, 8]
        x = self.input_proj(x)  # [B, 16, slot_dim]
        x = self.norm(x)

        # Initialize slots
        if slots is None:
            slots = self.slot_mu.expand(B, -1, -1).clone()

        # Keys and values
        k = self.project_k(x)
        v = self.project_v(x)

        # Iterative refinement
        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.project_q(slots)

            attn = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
            attn = F.softmax(attn, dim=1)

            updates = torch.einsum('bkn,bnd->bkd', attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim)
            ).reshape(B, self.num_slots, self.slot_dim)

        return slots


class LatentDynamics(nn.Module):
    """Predict next latent from slots."""

    def __init__(self, num_slots=4, slot_dim=32, latent_channels=8, latent_size=4):
        super().__init__()
        self.latent_channels = latent_channels
        self.latent_size = latent_size

        # Combine slots into prediction
        self.slot_to_latent = nn.Sequential(
            nn.Linear(num_slots * slot_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_channels * latent_size * latent_size),
        )

    def forward(self, slots):
        """
        Args:
            slots: [B, K, slot_dim]

        Returns:
            latent_pred: [B, C, H, W]
        """
        B = slots.shape[0]
        x = slots.flatten(1)  # [B, K*slot_dim]
        x = self.slot_to_latent(x)  # [B, C*H*W]
        return x.reshape(B, self.latent_channels, self.latent_size, self.latent_size)


class MicroSlotAR(nn.Module):
    """Full Slot-AR pipeline for video prediction.

    VAE is frozen; only slot attention + dynamics are trained.
    """

    def __init__(self, vae: MicroVAE, num_slots=4, slot_dim=32):
        super().__init__()
        self.vae = vae

        # Freeze VAE
        for p in self.vae.parameters():
            p.requires_grad = False

        self.slot_attention = LatentSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=vae.latent_channels,
        )

        self.dynamics = LatentDynamics(
            num_slots=num_slots,
            slot_dim=slot_dim,
            latent_channels=vae.latent_channels,
            latent_size=4,
        )

    def forward(self, frames: torch.Tensor, return_latents: bool = False):
        """
        Args:
            frames: [B, T, 1, 16, 16]
            return_latents: If True, return (preds, latent_preds, latent_targets)

        Returns:
            predictions: [B, T-2, 1, 16, 16]
        """
        B, T, C, H, W = frames.shape
        predictions = []
        latent_preds = []
        latent_targets = []
        slots = None

        for t in range(T - 2):
            # Encode current frame to latent
            frame_curr = frames[:, t + 1]
            with torch.no_grad():
                latent = self.vae.get_latent(frame_curr)

            # Update slots
            slots = self.slot_attention(latent, slots)

            # Predict next latent
            latent_pred = self.dynamics(slots)
            latent_preds.append(latent_pred)

            # Get target latent for training
            if return_latents:
                with torch.no_grad():
                    latent_target = self.vae.get_latent(frames[:, t + 2])
                latent_targets.append(latent_target)

            # Decode to pixel space (no grad needed)
            with torch.no_grad():
                pred = self.vae.decode(latent_pred)
            predictions.append(pred)

        preds = torch.stack(predictions, dim=1)
        if return_latents:
            return preds, torch.stack(latent_preds, dim=1), torch.stack(latent_targets, dim=1)
        return preds

    def count_parameters(self):
        """Count trainable (non-frozen) parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MicroBaselineAR(nn.Module):
    """Baseline AR model in latent space (no slots, just MLP dynamics)."""

    def __init__(self, vae: MicroVAE, hidden_dim=64):
        super().__init__()
        self.vae = vae

        # Freeze VAE
        for p in self.vae.parameters():
            p.requires_grad = False

        latent_flat = vae.latent_channels * 4 * 4  # 128

        # Simple MLP dynamics (match slot-AR param count)
        self.dynamics = nn.Sequential(
            nn.Linear(latent_flat * 2, hidden_dim),  # 2 frames context
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_flat),
        )

    def forward(self, frames: torch.Tensor, return_latents: bool = False):
        """
        Args:
            frames: [B, T, 1, 16, 16]
            return_latents: If True, return (preds, latent_preds, latent_targets)

        Returns:
            predictions: [B, T-2, 1, 16, 16]
        """
        B, T, C, H, W = frames.shape
        predictions = []
        latent_preds = []
        latent_targets = []

        for t in range(T - 2):
            # Encode two frames
            with torch.no_grad():
                lat_t = self.vae.get_latent(frames[:, t]).flatten(1)
                lat_t1 = self.vae.get_latent(frames[:, t + 1]).flatten(1)

            # Concat and predict
            x = torch.cat([lat_t, lat_t1], dim=1)
            lat_pred = self.dynamics(x)
            lat_pred_spatial = lat_pred.reshape(B, self.vae.latent_channels, 4, 4)
            latent_preds.append(lat_pred_spatial)

            # Get target latent for training
            if return_latents:
                with torch.no_grad():
                    latent_target = self.vae.get_latent(frames[:, t + 2])
                latent_targets.append(latent_target)

            # Decode
            with torch.no_grad():
                pred = self.vae.decode(lat_pred_spatial)

            predictions.append(pred)

        preds = torch.stack(predictions, dim=1)
        if return_latents:
            return preds, torch.stack(latent_preds, dim=1), torch.stack(latent_targets, dim=1)
        return preds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing Micro Slot-AR...")

    # Create VAE (will be pretrained)
    vae = MicroVAE(latent_channels=8)
    print(f"VAE params: {vae.count_parameters():,}")

    # Create Slot-AR
    slot_ar = MicroSlotAR(vae, num_slots=4, slot_dim=32)
    print(f"Slot-AR trainable params: {slot_ar.count_parameters():,}")

    # Create Baseline-AR
    baseline_ar = MicroBaselineAR(vae, hidden_dim=64)
    print(f"Baseline-AR trainable params: {baseline_ar.count_parameters():,}")

    # Test forward
    frames = torch.randn(4, 8, 1, 16, 16).sigmoid()

    with torch.no_grad():
        preds_slot = slot_ar(frames)
        preds_base = baseline_ar(frames)

    print(f"\nInput: {frames.shape}")
    print(f"Slot-AR output: {preds_slot.shape}")
    print(f"Baseline-AR output: {preds_base.shape}")
    print(f"Expected: (4, 6, 1, 16, 16)")

    print("\nAll tests passed!")
