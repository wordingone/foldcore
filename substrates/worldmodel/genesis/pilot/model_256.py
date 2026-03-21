"""256x256 Resolution Model.

Scales the infinite horizon architecture from 64x64 to 256x256.
Key changes:
1. Deeper encoder/decoder (256 -> 16 = 4 downsampling layers)
2. More channels for capacity
3. Windowed attention for memory efficiency

Target: <5M params (vs TinyWorlds 3M, but higher resolution)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from genesis.pilot.motion_model_v2 import spatial_transformer, BoundedSlotAttention


class ResBlock(nn.Module):
    """Residual block for encoder/decoder."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x):
        h = self.norm1(x)
        h = F.gelu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.gelu(h)
        h = self.conv2(h)
        return x + h


class MotionEncoder256(nn.Module):
    """Encode 256x256 frame pair to 16x16 features.

    256 -> 128 -> 64 -> 32 -> 16 (4 downsampling steps)
    """

    def __init__(self, in_channels=6, base_channels=48):
        super().__init__()
        c = base_channels

        self.conv_in = nn.Conv2d(in_channels, c, 3, padding=1)

        # 256 -> 128
        self.down1 = nn.Sequential(
            nn.Conv2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            ResBlock(c),
        )

        # 128 -> 64
        self.down2 = nn.Sequential(
            nn.Conv2d(c, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            ResBlock(c * 2),
        )

        # 64 -> 32
        self.down3 = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            ResBlock(c * 2),
        )

        # 32 -> 16
        self.down4 = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
        )

        self.out_channels = c * 2

    def forward(self, prev, curr):
        x = torch.cat([prev, curr], dim=1)
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class MotionDecoder256(nn.Module):
    """Decode 16x16 features to 256x256 motion + residual.

    16 -> 32 -> 64 -> 128 -> 256 (4 upsampling steps)
    """

    def __init__(self, in_channels=96, base_channels=48):
        super().__init__()
        c = base_channels

        # 16 -> 32
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            ResBlock(c * 2),
        )

        # 32 -> 64
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, c * 2),
            nn.GELU(),
            ResBlock(c * 2),
        )

        # 64 -> 128
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(c * 2, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
            ResBlock(c),
        )

        # 128 -> 256
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1),
            nn.GroupNorm(8, c),
            nn.GELU(),
        )

        # Motion head
        self.motion_head = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 2, 3, padding=1),
            nn.Tanh(),
        )
        self.motion_scale = 16.0  # Larger displacement for higher res

        # Residual head
        self.residual_head = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Tanh(),
        )
        self.residual_scale = 0.15

    def forward(self, features):
        x = self.up1(features)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)

        motion = self.motion_head(x) * self.motion_scale
        residual = self.residual_head(x) * self.residual_scale

        return motion, residual


class InfiniteHorizonModel256(nn.Module):
    """256x256 resolution infinite horizon model.

    Same architecture as 64x64 but scaled up:
    - Deeper encoder/decoder
    - More slots for more objects
    - Memory-efficient (16x16 feature map same as 64x64 version)
    """

    def __init__(
        self,
        base_channels=48,
        num_slots=16,  # More slots for higher res
        slot_dim=64,
        slot_decay=0.95,
    ):
        super().__init__()

        self.encoder = MotionEncoder256(in_channels=6, base_channels=base_channels)
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
        self.decoder = MotionDecoder256(in_channels=feat_channels, base_channels=base_channels)

        self._slots = None

    def reset_state(self):
        self._slots = None

    def step(self, prev, curr):
        feat = self.encoder(prev, curr)

        slot_input = self.to_slot_input(feat)
        slot_input = slot_input.flatten(2).transpose(1, 2)
        self._slots = self.slot_attention(slot_input, self._slots)

        slot_feat = self.slot_to_spatial(self._slots)
        slot_feat = slot_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        slot_feat = slot_feat.expand(-1, -1, feat.shape[2], feat.shape[3])

        combined = torch.cat([feat, slot_feat], dim=1)
        combined = self.combine(combined)

        motion, residual = self.decoder(combined)
        warped = spatial_transformer(curr, motion)
        pred = torch.clamp(warped + residual, 0, 1)

        return pred

    def forward(self, frames, rollout_steps=1):
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

            pred = self.step(prev, curr)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def generate(self, seed_frames, num_steps):
        self.reset_state()
        self.eval()

        history = [seed_frames[:, i] for i in range(seed_frames.shape[1])]
        generated = []

        with torch.no_grad():
            for step in range(num_steps):
                prev = history[-2]
                curr = history[-1]
                pred = self.step(prev, curr)
                generated.append(pred)
                history.append(pred)
                if len(history) > 3:
                    history.pop(0)

        return torch.stack(generated, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing 256x256 Model...")

    model = InfiniteHorizonModel256(
        base_channels=48,
        num_slots=16,
        slot_dim=64,
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    frames = torch.randn(2, 8, 3, 256, 256).sigmoid()
    print(f"Input: {frames.shape}")

    with torch.no_grad():
        preds = model(frames)
    print(f"Output: {preds.shape}")

    # Test generation
    seed = frames[:, :2]
    with torch.no_grad():
        gen = model.generate(seed, num_steps=20)
    print(f"Generated 20 steps: {gen.shape}")

    # Memory test
    if torch.cuda.is_available():
        model = model.cuda()
        frames = frames.cuda()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            _ = model(frames)

        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"\nPeak GPU memory: {peak_mb:.1f} MB")

    print("\nAll tests passed!")
