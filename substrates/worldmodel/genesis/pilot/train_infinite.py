"""Train Infinite Horizon Model.

Goal: Train a model that can play Pong forever without error explosion.

Key technique: Progressive autoregressive rollout training.
- Start with teacher forcing (use ground truth inputs)
- Gradually increase rollout (use own predictions as inputs)
- This teaches the model to recover from its own errors
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
import numpy as np

from genesis.pilot.motion_model_v2 import InfiniteHorizonModel, train_infinite_horizon
from genesis.pilot.video_data import get_video_dataset


def test_infinite_horizon(model, dataset, num_steps, device):
    """Test model on long autoregressive rollout."""
    model.eval()

    # Get sequences for testing
    all_frames = []
    for i in range(min(50, len(dataset))):
        all_frames.append(dataset[i])

    long_sequence = torch.cat(all_frames, dim=0).unsqueeze(0).to(device)
    num_steps = min(num_steps, long_sequence.shape[1] - 3)

    # Generate
    seed = long_sequence[:, :2]
    generated = model.generate(seed, num_steps)

    # Compare to ground truth
    gt = long_sequence[:, 2:2+num_steps]

    # MSE over time
    mse_over_time = []
    for t in range(num_steps):
        mse = F.mse_loss(generated[:, t], gt[:, t]).item()
        mse_over_time.append(mse)

    mse = np.array(mse_over_time)

    # Compute metrics
    initial_mse = mse[:50].mean()
    final_mse = mse[-50:].mean() if len(mse) >= 100 else mse[-len(mse)//2:].mean()
    explosion_ratio = final_mse / initial_mse

    return {
        'initial_mse': initial_mse,
        'final_mse': final_mse,
        'explosion_ratio': explosion_ratio,
        'mean_mse': mse.mean(),
        'max_mse': mse.max(),
        'mse_over_time': mse_over_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=2000)
    parser.add_argument('--test-steps', type=int, default=500)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', type=str, default='checkpoints/infinite')
    args = parser.parse_args()

    print("=" * 70)
    print("INFINITE HORIZON TRAINING")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    full_dataset = get_video_dataset(f"tinyworlds:{args.dataset}", seq_length=16)

    if len(full_dataset) > args.max_samples:
        full_dataset, _ = random_split(
            full_dataset,
            [args.max_samples, len(full_dataset) - args.max_samples],
            generator=torch.Generator().manual_seed(42)
        )

    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Create model
    print("\nCreating model...")
    model = InfiniteHorizonModel(
        base_channels=48,
        num_slots=8,
        slot_dim=64,
        slot_decay=0.95
    )
    print(f"Parameters: {model.count_parameters():,}")

    # Progressive rollout schedule
    # Gradually increase how many steps we unroll during training
    rollout_schedule = [
        (0, 1),   # Epochs 0-4: teacher forcing
        (5, 2),   # Epochs 5-9: 2-step rollout
        (10, 4),  # Epochs 10-14: 4-step rollout
    ]

    print(f"\nRollout schedule: {rollout_schedule}")

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model = model.to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def get_rollout_steps(epoch):
        steps = 1
        for e, s in rollout_schedule:
            if epoch >= e:
                steps = s
        return steps

    best_horizon_ratio = float('inf')

    for epoch in range(args.epochs):
        model.train()
        rollout_steps = get_rollout_steps(epoch)
        total_loss = 0
        num_batches = 0

        for frames in train_loader:
            frames = frames.to(args.device)
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
        scheduler.step()

        # Test infinite horizon
        horizon_metrics = test_infinite_horizon(model, val_dataset, args.test_steps, args.device)

        print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.6f} | "
              f"Rollout={rollout_steps} | "
              f"Horizon Ratio={horizon_metrics['explosion_ratio']:.2f}x")

        # Save best
        if horizon_metrics['explosion_ratio'] < best_horizon_ratio:
            best_horizon_ratio = horizon_metrics['explosion_ratio']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'horizon_ratio': best_horizon_ratio,
            }, save_dir / f"best_infinite_{args.dataset}.pt")
            print(f"  -> New best! Saved checkpoint.")

    # Final test
    print("\n" + "=" * 70)
    print("FINAL INFINITE HORIZON TEST")
    print("=" * 70)

    final_metrics = test_infinite_horizon(model, val_dataset, args.test_steps, args.device)

    print(f"Initial MSE (first 50):  {final_metrics['initial_mse']:.6f}")
    print(f"Final MSE (last 50):     {final_metrics['final_mse']:.6f}")
    print(f"Explosion Ratio:         {final_metrics['explosion_ratio']:.2f}x")
    print(f"Mean MSE:                {final_metrics['mean_mse']:.6f}")
    print(f"Max MSE:                 {final_metrics['max_mse']:.6f}")

    if final_metrics['explosion_ratio'] < 2.0:
        print("\nPASS: Can play Pong (nearly) forever!")
    elif final_metrics['explosion_ratio'] < 5.0:
        print("\nPARTIAL: Error grows but stays manageable")
    else:
        print("\nFAIL: Error still explodes")

    print("\n" + "=" * 70)
    print(f"Best explosion ratio achieved: {best_horizon_ratio:.2f}x")
    print(f"Checkpoints saved to: {save_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
