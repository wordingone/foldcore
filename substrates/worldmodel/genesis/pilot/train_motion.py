"""Train Motion Model on Real Video Data.

Combines:
- Motion vector prediction (4000x efficiency)
- Slot attention (32x efficiency)
- Real video data (TinyWorlds Pong/Sonic/etc)

Usage:
    python genesis/pilot/train_motion.py --dataset pong --epochs 10
    python genesis/pilot/train_motion.py --dataset synthetic --epochs 5  # Quick test
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
import json

from genesis.pilot.motion_model import MotionSlotModel, MotionBaselineModel
from genesis.pilot.video_data import get_video_dataset, SyntheticVideoDataset


def train_epoch(model, loader, optimizer, device, epoch, log_interval=50):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_motion_loss = 0
    num_batches = 0

    for batch_idx, frames in enumerate(loader):
        frames = frames.to(device)
        B, T, C, H, W = frames.shape
        targets = frames[:, 2:]  # Predict frames 2 onwards

        # Forward
        if hasattr(model, 'slot_attention'):
            preds, motions = model(frames, return_intermediates=True)
            # Motion regularization: encourage smooth motion
            motion_reg = (motions[:, 1:] - motions[:, :-1]).abs().mean() * 0.01
        else:
            preds = model(frames)
            motion_reg = 0

        # MSE loss
        loss = F.mse_loss(preds, targets) + motion_reg

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        if isinstance(motion_reg, torch.Tensor):
            total_motion_loss += motion_reg.item()
        num_batches += 1

        if batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}/{len(loader)}: Loss={loss.item():.4f}")

    return {
        'loss': total_loss / num_batches,
        'motion_reg': total_motion_loss / num_batches,
    }


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_mse = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for frames in loader:
            frames = frames.to(device)
            targets = frames[:, 2:]

            preds = model(frames)

            # MSE
            mse = F.mse_loss(preds, targets)
            total_mse += mse.item()

            # PSNR
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()

            num_batches += 1

    return {
        'mse': total_mse / num_batches,
        'psnr': total_psnr / num_batches,
    }


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pong',
                        choices=['synthetic', 'pong', 'pole_position', 'sonic', 'picodoom', 'zelda'])
    parser.add_argument('--model', type=str, default='slot', choices=['slot', 'baseline'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-length', type=int, default=16)
    parser.add_argument('--base-channels', type=int, default=48)
    parser.add_argument('--num-slots', type=int, default=8)
    parser.add_argument('--slot-dim', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--save-dir', type=str, default='checkpoints/motion')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit dataset size for quick tests')
    args = parser.parse_args()

    print("=" * 70)
    print("GENESIS MOTION MODEL TRAINING")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    if args.dataset == 'synthetic':
        full_dataset = SyntheticVideoDataset(
            num_sequences=args.max_samples or 5000,
            seq_length=args.seq_length,
        )
    else:
        full_dataset = get_video_dataset(
            f"tinyworlds:{args.dataset}",
            seq_length=args.seq_length,
        )

    # Limit dataset size if requested
    if args.max_samples and len(full_dataset) > args.max_samples:
        full_dataset, _ = random_split(
            full_dataset,
            [args.max_samples, len(full_dataset) - args.max_samples],
            generator=torch.Generator().manual_seed(42)
        )

    # Split into train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    print(f"Total sequences: {len(full_dataset)}")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")

    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)

    if args.model == 'slot':
        model = MotionSlotModel(
            base_channels=args.base_channels,
            num_slots=args.num_slots,
            slot_dim=args.slot_dim,
        )
    else:
        model = MotionBaselineModel(base_channels=args.base_channels)

    model = model.to(args.device)
    params = model.count_parameters()
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_mse = float('inf')
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, args.device, epoch)
        train_time = time.time() - start_time

        # Validate
        val_metrics = evaluate(model, val_loader, args.device)

        # Log
        print(f"Train Loss: {train_metrics['loss']:.4f}")
        print(f"Val MSE: {val_metrics['mse']:.4f} | PSNR: {val_metrics['psnr']:.2f} dB")
        print(f"Time: {train_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save best
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            save_checkpoint(
                model, optimizer, epoch,
                {'train': train_metrics, 'val': val_metrics},
                save_dir / f"best_{args.model}_{args.dataset}.pt"
            )
            print(f"  -> New best! Saved checkpoint.")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_mse': val_metrics['mse'],
            'val_psnr': val_metrics['psnr'],
        })

        scheduler.step()

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Val MSE: {best_val_mse:.4f}")
    print(f"Model: {args.model} | Params: {params:,}")
    print(f"Dataset: {args.dataset}")

    # Save history
    with open(save_dir / f"history_{args.model}_{args.dataset}.json", 'w') as f:
        json.dump({
            'args': vars(args),
            'params': params,
            'best_val_mse': best_val_mse,
            'history': history,
        }, f, indent=2)

    print(f"\nCheckpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
