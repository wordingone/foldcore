"""Pilot Slot Model - Frame predictor with slot attention for object permanence.

Architecture:
- CNN encoder (same as baseline)
- Slot attention for object-centric representation
- Slots persist across time (key difference from baseline)
- Slot decoder to reconstruct frame

Hypothesis: Slots will track objects through occlusion, improving recovery MSE.

Target: ~25-30M parameters (within 20% of baseline)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import math

try:
    from genesis.pilot.slot_attention import SlotAttention, SlotDecoder
except ImportError:
    from slot_attention import SlotAttention, SlotDecoder


class FrameEncoder(nn.Module):
    """Encode frame pair to features (same as baseline)."""

    def __init__(self, in_channels=6, feature_dim=128):
        super().__init__()
        # Process concatenated frame pair
        # 64x64 → 8x8
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, frame_prev, frame_curr):
        """Encode frame pair.

        Args:
            frame_prev: (B, 3, H, W)
            frame_curr: (B, 3, H, W)

        Returns:
            features: (B, N, feature_dim) where N = 64 (8x8)
        """
        # Concatenate frames
        x = torch.cat([frame_prev, frame_curr], dim=1)

        # CNN backbone
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (B, feature_dim, 8, 8)

        # Flatten spatial dims
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x


class SlotTransformer(nn.Module):
    """Process slots with self-attention for temporal reasoning."""

    def __init__(self, slot_dim=64, num_layers=4, num_heads=4):
        super().__init__()
        self.slot_dim = slot_dim

        # Transformer for slot processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=slot_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, slots):
        """Apply transformer to slots.

        Args:
            slots: (B, K, slot_dim)

        Returns:
            slots: (B, K, slot_dim)
        """
        return self.transformer(slots)


class FrameDecoder(nn.Module):
    """Decode features to predicted frame."""

    def __init__(self, feature_dim=128):
        super().__init__()
        # Decoder: 8x8 → 64x64
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 64, 4, stride=2, padding=1),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        """Decode features to frame.

        Args:
            features: (B, N, feature_dim) where N = 64

        Returns:
            frame: (B, 3, 64, 64)
        """
        B, N, C = features.shape
        H = W = 8  # sqrt(64)

        # Reshape to spatial
        x = features.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Decode
        frame = self.decoder(x)

        return frame


class PilotSlotModel(nn.Module):
    """Pilot model with slot attention for object permanence.

    Key difference from baseline:
    - Slots persist across time, explicitly tracking objects
    - Occlusion doesn't lose object info (slot still exists)

    Target: ~25-30M parameters
    """

    def __init__(
        self,
        num_slots: int = 4,
        slot_dim: int = 64,
        feature_dim: int = 128,
        num_slot_iterations: int = 3,
        num_transformer_layers: int = 4,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Encoder
        self.encoder = FrameEncoder(in_channels=6, feature_dim=feature_dim)

        # Slot attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=feature_dim,
            num_iterations=num_slot_iterations,
            hidden_dim=feature_dim,
        )

        # Slot transformer for temporal reasoning
        self.slot_transformer = SlotTransformer(
            slot_dim=slot_dim,
            num_layers=num_transformer_layers,
            num_heads=4,
        )

        # Slot decoder (from slots back to spatial features)
        self.slot_decoder = SlotDecoder(
            slot_dim=slot_dim,
            output_dim=feature_dim,
            spatial_size=8,
            hidden_dim=feature_dim,
        )

        # Frame decoder
        self.frame_decoder = FrameDecoder(feature_dim=feature_dim)

    def forward(self, frames, return_slots: bool = False):
        """Full forward pass for training.

        Args:
            frames: (B, T, 3, H, W)
            return_slots: If True, also return slot history

        Returns:
            predictions: (B, T-2, 3, H, W)
            slots_history: (optional) list of (B, K, slot_dim) for each timestep
        """
        B, T, _, H, W = frames.shape
        device = frames.device

        predictions = []
        slots_history = []

        # Initialize slots (will persist across time)
        slots = None

        # Predict frames t+1 from (t-1, t) for t in [1, T-2]
        for t in range(1, T - 1):
            frame_prev = frames[:, t - 1]
            frame_curr = frames[:, t]

            # Encode frame pair
            features = self.encoder(frame_prev, frame_curr)  # (B, N, feature_dim)

            # Update slots via attention (slots persist!)
            slots, attn = self.slot_attention(features, slots=slots)

            # Process slots with transformer
            slots_processed = self.slot_transformer(slots)

            # Decode slots to spatial features
            decoded_features = self.slot_decoder(slots_processed)  # (B, N, feature_dim)

            # Decode to frame
            pred = self.frame_decoder(decoded_features)

            predictions.append(pred)
            if return_slots:
                slots_history.append(slots.detach().clone())

        # Stack: (B, T-2, 3, H, W)
        predictions = torch.stack(predictions, dim=1)

        if return_slots:
            return predictions, slots_history
        return predictions

    def forward_single(self, frame_prev, frame_curr, slots=None):
        """Single step forward (for inference with persistent slots).

        Args:
            frame_prev: (B, 3, H, W)
            frame_curr: (B, 3, H, W)
            slots: (B, K, slot_dim) previous slots (None to initialize)

        Returns:
            pred: (B, 3, H, W) predicted next frame
            slots: (B, K, slot_dim) updated slots (pass to next call)
        """
        # Encode
        features = self.encoder(frame_prev, frame_curr)

        # Update slots
        slots, attn = self.slot_attention(features, slots=slots)

        # Process and decode
        slots_processed = self.slot_transformer(slots)
        decoded_features = self.slot_decoder(slots_processed)
        pred = self.frame_decoder(decoded_features)

        return pred, slots

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self):
        """Get parameter count by component."""
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'slot_attention': self.slot_attention.count_parameters(),
            'slot_transformer': sum(p.numel() for p in self.slot_transformer.parameters()),
            'slot_decoder': self.slot_decoder.count_parameters(),
            'frame_decoder': sum(p.numel() for p in self.frame_decoder.parameters()),
            'total': self.count_parameters(),
        }


if __name__ == '__main__':
    # Verification
    print("Testing PilotSlotModel...")

    # Small config for rapid iteration (~700K params)
    model = PilotSlotModel(
        num_slots=4,
        slot_dim=64,
        feature_dim=128,
        num_slot_iterations=3,
        num_transformer_layers=4,
    )

    print(f"\nParameter breakdown:")
    for name, count in model.get_param_breakdown().items():
        print(f"  {name}: {count:,}")

    # Forward pass test
    frames = torch.randn(4, 10, 3, 64, 64)
    with torch.no_grad():
        preds, slots_history = model(frames, return_slots=True)

    print(f"\nInput shape: {frames.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Expected: (4, 8, 3, 64, 64)")
    print(f"Match: {preds.shape == (4, 8, 3, 64, 64)}")
    print(f"Slots history length: {len(slots_history)}")
    print(f"Slot shape: {slots_history[0].shape}")

    # Test single-step inference
    print("\nTesting single-step inference...")
    pred, slots = model.forward_single(frames[:, 0], frames[:, 1])
    print(f"Single pred shape: {pred.shape}")
    print(f"Slots shape: {slots.shape}")

    # Test slot persistence
    pred2, slots2 = model.forward_single(frames[:, 1], frames[:, 2], slots=slots)
    print(f"Slots persisted: {slots2.shape}")

    print("\nAll tests passed!")
