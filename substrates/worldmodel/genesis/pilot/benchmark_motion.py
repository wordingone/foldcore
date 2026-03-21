"""Benchmark Motion Model vs Baselines.

Compares our motion + slot model against:
1. Motion baseline (no slots)
2. Direct pixel prediction (no motion)

Reports: params, MSE, PSNR, efficiency ratio.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
from typing import List
import time

from genesis.pilot.motion_model import MotionSlotModel, MotionBaselineModel
from genesis.pilot.video_data import get_video_dataset


@dataclass
class BenchmarkResult:
    name: str
    params: int
    mse: float
    psnr: float
    train_time: float


class DirectPixelModel(nn.Module):
    """Baseline: Predict pixels directly (no motion, no slots).

    Fair comparison: similar capacity to motion model.
    """

    def __init__(self, base_channels=48):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GELU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []

        for t in range(T - 2):
            prev = frames[:, t]
            curr = frames[:, t + 1]

            x = torch.cat([prev, curr], dim=1)
            x = self.encoder(x)
            pred = self.decoder(x)
            preds.append(pred)

        return torch.stack(preds, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_model(model, train_loader, epochs, lr, device):
    """Train a model and return training time."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for frames in train_loader:
            frames = frames.to(device)
            targets = frames[:, 2:]
            preds = model(frames)
            loss = F.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return time.time() - start_time


def evaluate_model(model, val_loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_mse = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for frames in val_loader:
            frames = frames.to(device)
            targets = frames[:, 2:]
            preds = model(frames)

            mse = F.mse_loss(preds, targets)
            total_mse += mse.item()

            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            total_psnr += psnr.item()

            num_batches += 1

    return {
        'mse': total_mse / num_batches,
        'psnr': total_psnr / num_batches,
    }


def run_benchmark(args):
    """Run full benchmark comparison."""
    device = args.device
    print("=" * 70)
    print("GENESIS MOTION MODEL BENCHMARK")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}")
    print(f"Epochs: {args.epochs}")

    # Load data
    print("\nLoading data...")
    if args.dataset == 'synthetic':
        from genesis.pilot.video_data import SyntheticVideoDataset
        full_dataset = SyntheticVideoDataset(num_sequences=args.num_samples, seq_length=16)
    else:
        full_dataset = get_video_dataset(f"tinyworlds:{args.dataset}", seq_length=16)

    # Limit size
    if len(full_dataset) > args.num_samples:
        full_dataset, _ = random_split(
            full_dataset,
            [args.num_samples, len(full_dataset) - args.num_samples],
            generator=torch.Generator().manual_seed(42)
        )

    # Split
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Models to compare
    models = {
        'DirectPixel': DirectPixelModel(base_channels=48),
        'MotionBaseline': MotionBaselineModel(base_channels=48),
        'MotionSlot': MotionSlotModel(base_channels=48, num_slots=8, slot_dim=64),
    }

    results: List[BenchmarkResult] = []

    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training {name}...")
        params = model.count_parameters()
        print(f"Parameters: {params:,}")

        train_time = train_model(model, train_loader, args.epochs, args.lr, device)
        metrics = evaluate_model(model, val_loader, device)

        print(f"MSE: {metrics['mse']:.6f} | PSNR: {metrics['psnr']:.2f} dB")
        print(f"Time: {train_time:.1f}s")

        results.append(BenchmarkResult(
            name=name,
            params=params,
            mse=metrics['mse'],
            psnr=metrics['psnr'],
            train_time=train_time,
        ))

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'Params':>12} {'MSE':>12} {'PSNR':>10} {'Time':>10}")
    print("-" * 70)

    for r in results:
        print(f"{r.name:<20} {r.params:>12,} {r.mse:>12.6f} {r.psnr:>10.2f} {r.train_time:>9.1f}s")

    # Analysis
    print("\n" + "=" * 70)
    print("EFFICIENCY ANALYSIS")
    print("=" * 70)

    direct = results[0]
    motion_base = results[1]
    motion_slot = results[2]

    # Motion vs Direct
    motion_improvement = (direct.mse - motion_base.mse) / direct.mse * 100
    print(f"\nMotion vs Direct Pixel:")
    print(f"  MSE Improvement: {motion_improvement:+.1f}%")
    print(f"  Params: {motion_base.params:,} vs {direct.params:,}")

    # Slot vs No-Slot
    slot_improvement = (motion_base.mse - motion_slot.mse) / motion_base.mse * 100
    print(f"\nMotion+Slot vs Motion:")
    print(f"  MSE Improvement: {slot_improvement:+.1f}%")
    print(f"  Params: {motion_slot.params:,} vs {motion_base.params:,}")

    # Overall
    total_improvement = (direct.mse - motion_slot.mse) / direct.mse * 100
    print(f"\nMotion+Slot vs Direct:")
    print(f"  MSE Improvement: {total_improvement:+.1f}%")

    # Efficiency (accuracy per param)
    print("\n" + "=" * 70)
    print("EFFICIENCY RATIOS")
    print("=" * 70)

    for r in results:
        efficiency = (1 / r.mse) / r.params * 1e6
        print(f"{r.name}: {efficiency:.2f} accuracy/param (x10^6)")

    best = max(results, key=lambda x: (1/x.mse) / x.params)
    print(f"\nMost Efficient: {best.name}")

    print("\n" + "=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--num-samples', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
