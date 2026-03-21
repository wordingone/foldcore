"""Micro Compare: Quick comparison of slot vs baseline on occlusion recovery.

Runs both models for N epochs and compares recovery MSE.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from genesis.pilot.micro_data import MicroDataset, get_recovery_frames
from genesis.pilot.micro_baseline import MicroBaseline
from genesis.pilot.micro_slot import MicroSlot


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for frames, masks in loader:
        frames = frames.to(device)
        # Target: frames 2..T-1
        targets = frames[:, 2:]

        preds = model(frames)
        loss = nn.functional.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """Evaluate model, return overall MSE and recovery MSE."""
    model.eval()
    total_mse = 0
    recovery_mse = 0
    recovery_count = 0
    total_count = 0

    with torch.no_grad():
        for frames, masks in loader:
            frames = frames.to(device)
            masks = masks.to(device)
            targets = frames[:, 2:]

            preds = model(frames)

            # Overall MSE
            mse = (preds - targets) ** 2
            total_mse += mse.mean().item()
            total_count += 1

            # Recovery MSE (frames after occlusion)
            # Recovery mask for prediction frames (2..T-1)
            recovery = get_recovery_frames(masks)[:, 2:]

            if recovery.any():
                # Expand recovery mask to match spatial dims
                B, T, C, H, W = preds.shape
                recovery_expanded = recovery.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                recovery_expanded = recovery_expanded.expand(-1, -1, C, H, W)

                recovery_preds = preds[recovery_expanded].reshape(-1)
                recovery_targets = targets[recovery_expanded].reshape(-1)

                if len(recovery_preds) > 0:
                    recovery_mse += ((recovery_preds - recovery_targets) ** 2).mean().item()
                    recovery_count += 1

    return {
        'mse': total_mse / max(total_count, 1),
        'recovery_mse': recovery_mse / max(recovery_count, 1),
    }


def run_experiment(model_name, model, train_loader, val_loader, epochs, lr, device):
    """Run training experiment."""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*50}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_recovery = float('inf')
    results = []

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        if metrics['recovery_mse'] < best_recovery:
            best_recovery = metrics['recovery_mse']

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            **metrics
        })

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss={train_loss:.4f} MSE={metrics['mse']:.4f} "
              f"Recovery={metrics['recovery_mse']:.4f}")

    return {
        'name': model_name,
        'params': sum(p.numel() for p in model.parameters()),
        'final_mse': results[-1]['mse'],
        'final_recovery': results[-1]['recovery_mse'],
        'best_recovery': best_recovery,
        'history': results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")

    # Create datasets
    print("\nGenerating datasets...")
    train_data = MicroDataset(num_sequences=args.train_size, seed=42)
    val_data = MicroDataset(num_sequences=args.val_size, seed=1000)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Train: {len(train_data)} sequences")
    print(f"Val: {len(val_data)} sequences")

    # Run experiments
    results = []

    # Baseline
    baseline = MicroBaseline(channels=16, num_frames=8)
    results.append(run_experiment(
        'Baseline', baseline, train_loader, val_loader,
        args.epochs, args.lr, args.device
    ))

    # Slot (matched params)
    slot = MicroSlot(channels=48, num_slots=4, slot_dim=48)
    results.append(run_experiment(
        'Slot', slot, train_loader, val_loader,
        args.epochs, args.lr, args.device
    ))

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Params':>10} {'MSE':>10} {'Recovery':>12} {'Best Rec':>12}")
    print("-" * 60)

    for r in results:
        print(f"{r['name']:<12} {r['params']:>10,} {r['final_mse']:>10.4f} "
              f"{r['final_recovery']:>12.4f} {r['best_recovery']:>12.4f}")

    # Decision
    baseline_rec = results[0]['best_recovery']
    slot_rec = results[1]['best_recovery']
    improvement = (baseline_rec - slot_rec) / baseline_rec * 100

    print(f"\n{'='*60}")
    if improvement > 10:
        print(f"PASS: Slot {improvement:.1f}% better on recovery (>{10}% threshold)")
    elif improvement > 0:
        print(f"MARGINAL: Slot {improvement:.1f}% better (below 10% threshold)")
    else:
        print(f"FAIL: Baseline {-improvement:.1f}% better than Slot")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
