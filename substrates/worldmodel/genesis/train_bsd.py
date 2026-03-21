"""Train Genesis with BSD for horizon stability test."""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import time

from genesis.genesis_model import Genesis, GenesisConfig


def train_step(model, batch, optimizer, device):
    """Single training step."""
    model.train()

    if 'frames' in batch:
        video = batch['frames'].to(device)
    else:
        video = batch['video'].to(device)

    actions = batch.get('actions')
    if actions is not None:
        actions = actions.to(device)
        if actions.shape[1] >= video.shape[1]:
            actions = actions[:, :video.shape[1] - 1]

    out = model(video, actions)
    loss = out['total_loss']

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'total': loss.item(),
        'latent': out['latent_loss'].item(),
        'recon': out['recon_loss'].item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, max_batches=10):
    """Quick evaluation."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        actions = batch.get('actions')
        if actions is not None:
            actions = actions.to(device)
            if actions.shape[1] >= video.shape[1]:
                actions = actions[:, :video.shape[1] - 1]

        out = model(video, actions)
        total_loss += out['total_loss'].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', default='checkpoints/genesis_bsd')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING GENESIS WITH BSD")
    print("=" * 70)

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create BSD-enabled config
    config = GenesisConfig(
        image_size=64,
        num_frames=16,
        use_bsd=True,
        use_slots=True,
        num_slots=8,
        slot_dim=64,
        bsd_d_state=256,
        bsd_lambda_range=(0.9, 0.999),
        bsd_rotation_scale=0.1,
        num_layers=6,
        hidden_dim=512,
        num_heads=8,
        action_dim=18,
    )

    print(f"\nConfig: {config}")

    # Create model
    model = Genesis(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Data loader
    print("\nLoading data...")
    from genesis.pilot.stream_hf import create_streaming_loader
    train_loader = create_streaming_loader(
        dataset_name='jat',
        batch_size=args.batch_size,
        seq_length=config.num_frames,
        image_size=config.image_size,
        game='atari-breakout',
        shuffle=True,
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= 100:  # Limit batches per epoch for quick test
                break

            losses = train_step(model, batch, optimizer, device)
            epoch_loss += losses['total']
            num_batches += 1
            global_step += 1

            if batch_idx % 20 == 0:
                print(f"  Step {batch_idx}: loss={losses['total']:.4f}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch avg loss: {avg_loss:.4f}")

        # Save if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, output_dir / 'best_bsd.pt')
            print(f"  Saved best model (loss={avg_loss:.4f})")

    # Save final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
    }, output_dir / 'final_bsd.pt')

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
