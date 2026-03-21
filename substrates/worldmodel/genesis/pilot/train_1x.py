"""Train Genesis on 1X World Model Dataset.

Streams 100+ hours of humanoid robot first-person video with continuous actions.
Uses chunked streaming to handle large dataset efficiently.

Dataset: https://huggingface.co/datasets/1x-technologies/worldmodel
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import json
import time

from genesis.pilot.streaming_1x import (
    MemoryMapped1XDataset,
    Streaming1XDataset,
    create_1x_dataloader,
    EpochTracker,
    download_1x_dataset,
)
from genesis.pilot.action_model import ActionConditionedModel


# =============================================================================
# TOKEN DECODER (MAGVIT2 or simple learned decoder)
# =============================================================================

class SimpleTokenDecoder(nn.Module):
    """Simple learned decoder from tokens to images.

    For full quality, use the provided MAGVIT2 decoder.
    This is a lightweight alternative for training.
    """

    def __init__(
        self,
        vocab_size: int = 2**18,
        token_dim: int = 256,
        image_size: int = 64,
        token_grid: int = 16,
    ):
        super().__init__()

        self.token_grid = token_grid
        self.image_size = image_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, token_dim)

        # Upsample 16x16 -> 64x64 (or target size)
        scale_factor = image_size // token_grid  # 4x for 64, 16x for 256

        layers = []
        current_dim = token_dim

        while scale_factor > 1:
            out_dim = current_dim // 2 if current_dim > 64 else current_dim
            layers.extend([
                nn.ConvTranspose2d(current_dim, out_dim, 4, stride=2, padding=1),
                nn.GroupNorm(8, out_dim),
                nn.GELU(),
            ])
            current_dim = out_dim
            scale_factor //= 2

        # Final conv to RGB
        layers.append(nn.Conv2d(current_dim, 3, 3, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, tokens):
        """
        Args:
            tokens: [B, T, H, W] or [B, H, W] integer tokens

        Returns:
            images: [B, T, 3, image_size, image_size] or [B, 3, H, W]
        """
        has_time = tokens.dim() == 4

        if has_time:
            B, T, H, W = tokens.shape
            tokens = tokens.view(B * T, H, W)
        else:
            B, H, W = tokens.shape

        # Embed tokens
        x = self.embedding(tokens)  # [B*T, H, W, D]
        x = x.permute(0, 3, 1, 2)  # [B*T, D, H, W]

        # Decode to image
        images = self.decoder(x)  # [B*T, 3, H', W']

        if has_time:
            _, C, H_out, W_out = images.shape
            images = images.view(B, T, C, H_out, W_out)

        return images


# =============================================================================
# CONTINUOUS ACTION MODEL (extends ActionConditionedModel)
# =============================================================================

class ContinuousActionModel(nn.Module):
    """World model with continuous action conditioning.

    Adapted for 1X dataset with 25D continuous actions.
    """

    def __init__(
        self,
        base_channels: int = 48,
        num_slots: int = 8,
        slot_dim: int = 64,
        slot_decay: float = 0.95,
        action_dim: int = 25,  # 1X has 25D actions
        image_size: int = 64,
    ):
        super().__init__()

        from genesis.pilot.motion_model_v2 import (
            MotionEncoder, MotionDecoder, BoundedSlotAttention, spatial_transformer
        )

        self.action_dim = action_dim
        self.spatial_transformer = spatial_transformer

        # Visual encoder
        self.encoder = MotionEncoder(in_channels=6, base_channels=base_channels)
        feat_channels = self.encoder.out_channels

        # Continuous action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.GELU(),
            nn.Linear(64, feat_channels),
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

        self._slots = None

    def reset_state(self):
        self._slots = None

    def step(self, prev, curr, action=None):
        """Single prediction step.

        Args:
            prev: [B, 3, H, W] - previous frame
            curr: [B, 3, H, W] - current frame
            action: [B, action_dim] - continuous action vector

        Returns:
            next_pred: [B, 3, H, W] - predicted next frame
        """
        # Encode visual
        feat = self.encoder(prev, curr)  # [B, C, 16, 16]

        # Encode and apply action
        if action is not None:
            action_feat = self.action_encoder(action)  # [B, C]
            action_feat = action_feat.unsqueeze(-1).unsqueeze(-1)
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

        # Decode
        motion, residual = self.decoder(combined)
        warped = self.spatial_transformer(curr, motion)
        pred = torch.clamp(warped + residual, 0, 1)

        return pred

    def forward(self, frames, actions=None, rollout_steps=1):
        """Forward pass with continuous actions.

        Args:
            frames: [B, T, 3, H, W]
            actions: [B, T-2, action_dim] continuous actions
            rollout_steps: Autoregressive rollout steps

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

            action = actions[:, t] if actions is not None else None
            pred = self.step(prev, curr, action)
            predictions.append(pred)

        return torch.stack(predictions, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model,
    token_decoder,
    loader,
    optimizer,
    device,
    epoch_tracker,
    rollout_steps=1,
    max_batches=None,
):
    """Train for one epoch (or partial epoch with streaming)."""
    model.train()
    token_decoder.eval()  # Decoder is frozen or separately trained

    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        tokens = batch["tokens"].to(device)  # [B, T, 16, 16]
        actions = batch.get("actions")
        if actions is not None:
            actions = actions.to(device)  # [B, T, 25]

        # Decode tokens to images
        with torch.no_grad():
            frames = token_decoder(tokens)  # [B, T, 3, H, W]

        # Get targets
        targets = frames[:, 2:]

        # Get action inputs (offset by 1 since we predict t+2 from t, t+1)
        if actions is not None:
            action_inputs = actions[:, 1:-1]  # Actions for predicting frames 2..T-1
        else:
            action_inputs = None

        # Forward
        preds = model(frames, actions=action_inputs, rollout_steps=rollout_steps)

        # Loss
        loss = F.mse_loss(preds, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Track epoch progress
        epoch_done = epoch_tracker.step()

        if batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            samples_per_sec = (batch_idx + 1) * tokens.shape[0] / elapsed
            print(f"  Batch {batch_idx}: loss={loss.item():.6f}, "
                  f"progress={epoch_tracker.progress:.1%}, "
                  f"speed={samples_per_sec:.1f} samples/s")

        if epoch_done:
            print(f"  Epoch {epoch_tracker.current_epoch} completed")
            break

    return total_loss / max(num_batches, 1)


def evaluate(model, token_decoder, loader, device, max_batches=100):
    """Evaluate model."""
    model.eval()
    token_decoder.eval()

    total_mse = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            tokens = batch["tokens"].to(device)
            actions = batch.get("actions")
            if actions is not None:
                actions = actions.to(device)

            frames = token_decoder(tokens)
            targets = frames[:, 2:]

            action_inputs = actions[:, 1:-1] if actions is not None else None
            preds = model(frames, actions=action_inputs)

            mse = F.mse_loss(preds, targets)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

            total_mse += mse.item()
            total_psnr += psnr.item()
            num_batches += 1

    return {
        "mse": total_mse / num_batches,
        "psnr": total_psnr / num_batches,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Genesis on 1X Dataset")
    parser.add_argument("--data-dir", default="data/1x_worldmodel",
                       help="Path to 1X data")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-length", type=int, default=16)
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming data loader")
    parser.add_argument("--chunk-size", type=int, default=5000,
                       help="Chunk size for streaming")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None,
                       help="Limit batches per epoch (for testing)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="checkpoints/1x")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset if not present")
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING GENESIS ON 1X WORLD MODEL DATASET")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Streaming: {args.streaming}")

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Download if requested
    if args.download:
        download_1x_dataset(args.data_dir)

    # Check data exists
    data_dir = Path(args.data_dir)
    if not (data_dir / "video.bin").exists():
        print(f"\nData not found at {data_dir}")
        print("Run with --download to download, or manually download from:")
        print("  huggingface-cli download 1x-technologies/worldmodel --repo-type dataset --local-dir data/1x_worldmodel")
        return

    # Create data loader
    print("\nLoading data...")
    if args.streaming:
        loader = create_1x_dataloader(
            str(data_dir),
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            streaming=True,
            chunk_size=args.chunk_size,
            num_workers=2,
        )
        # Estimate dataset size for epoch tracking
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            dataset_size = metadata["num_images"] // args.seq_length
        else:
            dataset_size = 100000  # Rough estimate
    else:
        loader = create_1x_dataloader(
            str(data_dir),
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            streaming=False,
            num_workers=4,
        )
        dataset_size = len(loader.dataset)

    print(f"Dataset size: ~{dataset_size:,} sequences")

    # Create models
    print("\nCreating models...")

    # Token decoder (lightweight version)
    token_decoder = SimpleTokenDecoder(
        vocab_size=2**18,
        token_dim=256,
        image_size=64,
        token_grid=16,
    ).to(device)
    print(f"Token decoder params: {sum(p.numel() for p in token_decoder.parameters()):,}")

    # Detect action dimension from data
    if args.streaming:
        # For streaming, create a temporary loader to check
        temp_dataset = Streaming1XDataset(str(data_dir), seq_length=args.seq_length, chunk_size=1000)
        sample_batch = next(iter(temp_dataset))
        action_dim = sample_batch["actions"].shape[-1] if "actions" in sample_batch else 25
        del temp_dataset
    else:
        sample_batch = loader.dataset[0]
        action_dim = sample_batch["actions"].shape[-1] if "actions" in sample_batch else 25

    print(f"Action dimension: {action_dim}")

    # World model with continuous actions
    model = ContinuousActionModel(
        base_channels=48,
        num_slots=8,
        slot_dim=64,
        slot_decay=0.95,
        action_dim=action_dim,
        image_size=64,
    ).to(device)
    print(f"World model params: {model.count_parameters():,}")

    # Optimizer (train both decoder and model)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(token_decoder.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )

    # Epoch tracker
    epoch_tracker = EpochTracker(dataset_size, args.batch_size)

    # Progressive rollout schedule
    rollout_schedule = [(0, 1), (1, 2), (2, 4)]

    def get_rollout(epoch):
        r = 1
        for e, s in rollout_schedule:
            if epoch >= e:
                r = s
        return r

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_psnr = 0

    for epoch in range(args.epochs):
        rollout = get_rollout(epoch)
        print(f"\nEpoch {epoch + 1}/{args.epochs} (rollout={rollout})")

        # Train
        train_loss = train_epoch(
            model, token_decoder, loader, optimizer, device,
            epoch_tracker, rollout_steps=rollout,
            max_batches=args.max_batches_per_epoch,
        )

        # Evaluate
        val_metrics = evaluate(model, token_decoder, loader, device, max_batches=50)

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, "
              f"val_mse={val_metrics['mse']:.6f}, val_psnr={val_metrics['psnr']:.2f} dB")

        # Save checkpoint
        if val_metrics["psnr"] > best_psnr:
            best_psnr = val_metrics["psnr"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "decoder_state_dict": token_decoder.state_dict(),
                "psnr": best_psnr,
            }, save_dir / "best_1x.pt")
            print(f"  -> Saved (best PSNR)")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
