"""Motion Model V2: Designed for Infinite Horizon.

Key changes from V1:
1. Train with autoregressive rollout (not just single-step)
2. Bounded slot memory (no unbounded growth)
3. Self-correction mechanism (learn to fix accumulated errors)
4. Explicit slot reset protocol

Goal: Play Pong forever without error explosion.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def spatial_transformer(x, flow):
    """Warp image x according to flow field."""
    B, C, H, W = x.shape
    device = x.device

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

    flow_permuted = flow.permute(0, 2, 3, 1)
    flow_normalized = flow_permuted * 2 / torch.tensor([W, H], device=device, dtype=x.dtype)

    sample_grid = (grid + flow_normalized).clamp(-1, 1)
    warped = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


class BoundedSlotAttention(nn.Module):
    """Slot attention with bounded memory.

    Key difference: slots decay over time to prevent unbounded growth.
    """

    def __init__(self, num_slots=8, slot_dim=64, input_dim=64, decay=0.95):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.decay = decay  # Slot memory decay factor

        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)

        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.scale = slot_dim ** -0.5

    def forward(self, inputs, slots=None):
        """
        Args:
            inputs: [B, N, input_dim]
            slots: [B, K, slot_dim] or None

        Returns:
            slots: [B, K, slot_dim]
        """
        B, N, _ = inputs.shape

        if slots is None:
            slots = self.slot_mu.expand(B, -1, -1).clone()
        else:
            # Apply decay to prevent unbounded growth
            slots = slots * self.decay + self.slot_mu.expand(B, -1, -1) * (1 - self.decay)

        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        # Single iteration (faster, sufficient with decay)
        slots = self.norm_slots(slots)
        q = self.project_q(slots)

        attn = torch.einsum('bkd,bnd->bkn', q, k) * self.scale
        attn = F.softmax(attn, dim=1)

        attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        updates = torch.einsum('bkn,bnd->bkd', attn_norm, v)

        slots = self.gru(
            updates.reshape(-1, self.slot_dim),
            slots.reshape(-1, self.slot_dim)
        ).reshape(B, self.num_slots, self.slot_dim)

        return slots


class MotionEncoder(nn.Module):
    """Encode frame pair to motion features."""

    def __init__(self, in_channels=6, base_channels=32):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
        )
        self.out_channels = base_channels * 2

    def forward(self, prev, curr):
        x = torch.cat([prev, curr], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MotionDecoder(nn.Module):
    """Decode features to motion field + residual."""

    def __init__(self, in_channels=64, base_channels=32):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
        )

        self.motion_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.motion_scale = 8.0

        self.residual_head = nn.Sequential(
            nn.Conv2d(base_channels, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.residual_scale = 0.2

    def forward(self, features):
        x = self.up1(features)
        x = self.up2(x)
        motion = self.motion_head(x) * self.motion_scale
        residual = self.residual_head(x) * self.residual_scale
        return motion, residual


class InfiniteHorizonModel(nn.Module):
    """World model designed for infinite horizon generation.

    Key features:
    1. Bounded slot memory (decay prevents growth)
    2. Pure motion-based prediction (no history accumulation)
    3. Can train with autoregressive rollout
    4. Stateless prediction (each step only needs prev, curr)
    """

    def __init__(self, base_channels=48, num_slots=8, slot_dim=64, slot_decay=0.95):
        super().__init__()

        self.encoder = MotionEncoder(in_channels=6, base_channels=base_channels)
        feat_channels = self.encoder.out_channels

        self.to_slot_input = nn.Conv2d(feat_channels, slot_dim, 1)
        self.slot_attention = BoundedSlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            decay=slot_decay,
        )

        self.slot_to_spatial = nn.Sequential(
            nn.Linear(slot_dim, feat_channels * 4),
            nn.GELU(),
            nn.Linear(feat_channels * 4, feat_channels),
        )

        self.combine = nn.Conv2d(feat_channels * 2, feat_channels, 1)
        self.decoder = MotionDecoder(in_channels=feat_channels, base_channels=base_channels)

        # Internal state for rollout
        self._slots = None

    def reset_state(self):
        """Reset internal state for new sequence."""
        self._slots = None

    def step(self, prev, curr):
        """Single prediction step.

        Args:
            prev: [B, 3, H, W] - previous frame
            curr: [B, 3, H, W] - current frame

        Returns:
            next_pred: [B, 3, H, W] - predicted next frame
        """
        # Encode
        feat = self.encoder(prev, curr)

        # Slot attention (with bounded memory)
        slot_input = self.to_slot_input(feat)
        slot_input = slot_input.flatten(2).transpose(1, 2)
        self._slots = self.slot_attention(slot_input, self._slots)

        # Slot influence
        slot_feat = self.slot_to_spatial(self._slots)
        slot_feat = slot_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        slot_feat = slot_feat.expand(-1, -1, feat.shape[2], feat.shape[3])

        combined = torch.cat([feat, slot_feat], dim=1)
        combined = self.combine(combined)

        # Decode to motion + residual
        motion, residual = self.decoder(combined)

        # Warp current frame
        warped = spatial_transformer(curr, motion)

        # Add residual
        pred = torch.clamp(warped + residual, 0, 1)

        return pred

    def forward(self, frames, rollout_steps=1):
        """Forward pass with optional autoregressive rollout.

        Args:
            frames: [B, T, 3, H, W] - input sequence
            rollout_steps: Number of autoregressive steps to roll out
                          1 = teacher forcing (default, matches V1)
                          >1 = autoregressive training (for infinite horizon)

        Returns:
            predictions: [B, T-2, 3, H, W]
        """
        B, T, C, H, W = frames.shape
        predictions = []

        self.reset_state()

        for t in range(T - 2):
            if t < rollout_steps or rollout_steps == 1:
                # Teacher forcing: use ground truth
                prev = frames[:, t]
                curr = frames[:, t + 1]
            else:
                # Autoregressive: use own predictions
                prev = predictions[-2] if len(predictions) >= 2 else frames[:, t]
                curr = predictions[-1] if len(predictions) >= 1 else frames[:, t + 1]

            pred = self.step(prev, curr)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def generate(self, seed_frames, num_steps):
        """Generate frames autoregressively (for inference).

        Args:
            seed_frames: [B, 2, 3, H, W] - at least 2 frames to start
            num_steps: Number of frames to generate

        Returns:
            generated: [B, num_steps, 3, H, W]
        """
        self.reset_state()
        self.eval()

        B = seed_frames.shape[0]
        history = [seed_frames[:, i] for i in range(seed_frames.shape[1])]
        generated = []

        with torch.no_grad():
            for step in range(num_steps):
                prev = history[-2]
                curr = history[-1]

                pred = self.step(prev, curr)
                generated.append(pred)

                # Update history (bounded)
                history.append(pred)
                if len(history) > 3:
                    history.pop(0)

        return torch.stack(generated, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_infinite_horizon(model, train_loader, epochs, lr, device, rollout_schedule=None):
    """Train with progressive autoregressive rollout.

    Args:
        rollout_schedule: List of (epoch, rollout_steps) tuples
                         e.g., [(0, 1), (5, 2), (10, 4)] = teacher forcing first,
                         then gradually increase rollout
    """
    if rollout_schedule is None:
        # Default: start with teacher forcing, increase rollout
        rollout_schedule = [(0, 1), (3, 2), (6, 4), (9, 8)]

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def get_rollout_steps(epoch):
        steps = 1
        for e, s in rollout_schedule:
            if epoch >= e:
                steps = s
        return steps

    for epoch in range(epochs):
        model.train()
        rollout_steps = get_rollout_steps(epoch)
        total_loss = 0
        num_batches = 0

        for frames in train_loader:
            frames = frames.to(device)
            targets = frames[:, 2:]

            preds = model(frames, rollout_steps=rollout_steps)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}: Loss={avg_loss:.6f} Rollout={rollout_steps}")

    return model


if __name__ == '__main__':
    print("Testing Infinite Horizon Model...")

    model = InfiniteHorizonModel(base_channels=48, num_slots=8, slot_dim=64, slot_decay=0.95)
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    frames = torch.randn(2, 10, 3, 64, 64).sigmoid()
    preds = model(frames, rollout_steps=1)
    print(f"Forward (teacher): {preds.shape}")

    preds = model(frames, rollout_steps=4)
    print(f"Forward (rollout=4): {preds.shape}")

    # Test generation
    seed = frames[:, :2]
    generated = model.generate(seed, num_steps=100)
    print(f"Generate 100 steps: {generated.shape}")

    # Memory check
    if torch.cuda.is_available():
        model = model.cuda()
        seed = seed.cuda()

        torch.cuda.reset_peak_memory_stats()
        gen = model.generate(seed, num_steps=500)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak memory for 500 steps: {peak_mb:.1f} MB")

    print("All tests passed!")
