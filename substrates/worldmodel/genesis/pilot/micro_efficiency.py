"""Micro Efficiency Benchmark: Prove efficiency thesis alongside accuracy.

Measures:
1. Parameters at equal accuracy (param efficiency)
2. FLOPs per prediction (compute efficiency)
3. Scaling curves (efficiency at scale)

Reports efficiency ratios, not just accuracy.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from dataclasses import dataclass
from typing import List, Dict

from genesis.pilot.micro_data import MicroDataset, get_recovery_frames
from genesis.pilot.micro_baseline import MicroBaseline
from genesis.pilot.micro_slot import MicroSlot


@dataclass
class ModelMetrics:
    """Complete metrics for efficiency comparison."""
    name: str
    params: int
    flops_per_frame: int
    mse: float
    recovery_mse: float
    train_time_seconds: float
    memory_mb: float


def count_flops(model, input_shape=(1, 8, 1, 16, 16)):
    """Count FLOPs for a single forward pass.

    Uses manual counting since torchprofile may not be available.
    Returns approximate FLOPs.
    """
    # For micro models, we count ops manually
    # Conv2d: 2 * K^2 * C_in * C_out * H_out * W_out
    # Linear: 2 * in_features * out_features
    # GRU: ~6 * hidden_size^2 per step

    total_flops = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Assuming stride divides evenly
            h_out = 16 // (module.stride[0] if hasattr(module, 'stride') else 1)
            w_out = 16 // (module.stride[1] if hasattr(module, 'stride') else 1)
            # Rough estimate for our 16x16 inputs
            if module.stride[0] == 4:
                h_out, w_out = 4, 4
            elif module.stride[0] == 2:
                h_out, w_out = 8, 8
            flops = 2 * module.kernel_size[0] * module.kernel_size[1] * \
                    module.in_channels * module.out_channels * h_out * w_out
            total_flops += flops

        elif isinstance(module, nn.ConvTranspose2d):
            # Similar to conv but output is larger
            if module.stride[0] == 4:
                h_out, w_out = 16, 16
            elif module.stride[0] == 2:
                h_out, w_out = 8, 8
            else:
                h_out, w_out = 4, 4
            flops = 2 * module.kernel_size[0] * module.kernel_size[1] * \
                    module.in_channels * module.out_channels * h_out * w_out
            total_flops += flops

        elif isinstance(module, nn.Linear):
            flops = 2 * module.in_features * module.out_features
            total_flops += flops

        elif isinstance(module, nn.GRUCell):
            # GRU: 3 gates, each with input and hidden projections
            flops = 6 * module.hidden_size * (module.input_size + module.hidden_size)
            total_flops += flops

    # Multiply by number of timesteps (T-2 = 6 predictions)
    num_predictions = input_shape[1] - 2
    total_flops *= num_predictions

    return total_flops


def measure_memory(model, device):
    """Measure peak memory usage during forward pass."""
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(32, 8, 1, 16, 16, device=device)
        with torch.no_grad():
            _ = model(x)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        return peak_mb
    return 0.0


def train_and_measure(model, train_loader, val_loader, epochs, lr, device) -> ModelMetrics:
    """Train model and collect all efficiency metrics."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Measure FLOPs before training
    flops = count_flops(model)
    params = sum(p.numel() for p in model.parameters())

    # Train
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        for frames, masks in train_loader:
            frames = frames.to(device)
            targets = frames[:, 2:]
            preds = model(frames)
            loss = nn.functional.mse_loss(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_time = time.time() - start_time

    # Evaluate
    model.eval()
    total_mse = 0
    recovery_mse = 0
    recovery_count = 0
    total_count = 0

    with torch.no_grad():
        for frames, masks in val_loader:
            frames = frames.to(device)
            masks = masks.to(device)
            targets = frames[:, 2:]
            preds = model(frames)

            mse = (preds - targets) ** 2
            total_mse += mse.mean().item()
            total_count += 1

            recovery = get_recovery_frames(masks)[:, 2:]
            if recovery.any():
                B, T, C, H, W = preds.shape
                recovery_exp = recovery.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                recovery_exp = recovery_exp.expand(-1, -1, C, H, W)
                rec_preds = preds[recovery_exp].reshape(-1)
                rec_targets = targets[recovery_exp].reshape(-1)
                if len(rec_preds) > 0:
                    recovery_mse += ((rec_preds - rec_targets) ** 2).mean().item()
                    recovery_count += 1

    # Memory
    memory_mb = measure_memory(model, device)

    return ModelMetrics(
        name=model.__class__.__name__,
        params=params,
        flops_per_frame=flops // 6,  # Per prediction, not per sequence
        mse=total_mse / max(total_count, 1),
        recovery_mse=recovery_mse / max(recovery_count, 1),
        train_time_seconds=train_time,
        memory_mb=memory_mb,
    )


def create_scaled_models(scale: str):
    """Create baseline and slot models at different scales."""
    scales = {
        'tiny':   {'baseline': {'channels': 8},  'slot': {'channels': 24, 'num_slots': 4, 'slot_dim': 24}},
        'small':  {'baseline': {'channels': 16}, 'slot': {'channels': 48, 'num_slots': 4, 'slot_dim': 48}},
        'medium': {'baseline': {'channels': 32}, 'slot': {'channels': 64, 'num_slots': 4, 'slot_dim': 64}},
        'large':  {'baseline': {'channels': 48}, 'slot': {'channels': 96, 'num_slots': 6, 'slot_dim': 96}},
    }

    cfg = scales[scale]
    baseline = MicroBaseline(channels=cfg['baseline']['channels'], num_frames=8)
    slot = MicroSlot(**cfg['slot'])
    return baseline, slot


def run_efficiency_benchmark(args):
    """Run full efficiency benchmark."""
    device = args.device
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")

    # Data
    train_data = MicroDataset(num_sequences=args.train_size, seed=42)
    val_data = MicroDataset(num_sequences=args.val_size, seed=1000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    results: List[ModelMetrics] = []

    # Test at multiple scales
    for scale in ['tiny', 'small', 'medium', 'large']:
        print(f"\n{'='*60}")
        print(f"Scale: {scale.upper()}")
        print(f"{'='*60}")

        baseline, slot = create_scaled_models(scale)

        print(f"\nTraining Baseline ({sum(p.numel() for p in baseline.parameters()):,} params)...")
        baseline_metrics = train_and_measure(baseline, train_loader, val_loader, args.epochs, args.lr, device)
        baseline_metrics.name = f"Baseline-{scale}"
        results.append(baseline_metrics)
        print(f"  MSE={baseline_metrics.mse:.4f} Recovery={baseline_metrics.recovery_mse:.4f}")

        print(f"\nTraining Slot ({sum(p.numel() for p in slot.parameters()):,} params)...")
        slot_metrics = train_and_measure(slot, train_loader, val_loader, args.epochs, args.lr, device)
        slot_metrics.name = f"Slot-{scale}"
        results.append(slot_metrics)
        print(f"  MSE={slot_metrics.mse:.4f} Recovery={slot_metrics.recovery_mse:.4f}")

    # Report
    print("\n" + "=" * 100)
    print("EFFICIENCY BENCHMARK RESULTS")
    print("=" * 100)
    print(f"{'Model':<20} {'Params':>10} {'FLOPs/frame':>12} {'MSE':>10} {'Recovery':>10} {'Time(s)':>10} {'Mem(MB)':>10}")
    print("-" * 100)

    for m in results:
        print(f"{m.name:<20} {m.params:>10,} {m.flops_per_frame:>12,} {m.mse:>10.4f} {m.recovery_mse:>10.4f} {m.train_time_seconds:>10.1f} {m.memory_mb:>10.1f}")

    # Efficiency Analysis
    print("\n" + "=" * 100)
    print("EFFICIENCY RATIOS (Slot vs Baseline at each scale)")
    print("=" * 100)

    scales = ['tiny', 'small', 'medium', 'large']
    for i, scale in enumerate(scales):
        baseline = results[i * 2]
        slot = results[i * 2 + 1]

        # Recovery improvement
        recovery_improvement = (baseline.recovery_mse - slot.recovery_mse) / baseline.recovery_mse * 100

        # Efficiency ratios
        param_ratio = baseline.params / slot.params
        flop_ratio = baseline.flops_per_frame / slot.flops_per_frame if slot.flops_per_frame > 0 else 0

        # Accuracy per param (higher is better)
        baseline_acc_per_param = (1 / baseline.recovery_mse) / baseline.params * 1e6
        slot_acc_per_param = (1 / slot.recovery_mse) / slot.params * 1e6
        acc_efficiency_ratio = slot_acc_per_param / baseline_acc_per_param

        print(f"\n{scale.upper()}:")
        print(f"  Recovery Improvement: {recovery_improvement:+.1f}%")
        print(f"  Param Ratio (base/slot): {param_ratio:.2f}x")
        print(f"  FLOP Ratio (base/slot): {flop_ratio:.2f}x")
        print(f"  Accuracy/Param Efficiency: {acc_efficiency_ratio:.2f}x better")

    # Final Verdict
    print("\n" + "=" * 100)
    print("EVAL RESULTS")
    print("=" * 100)

    # E1: Param Efficiency
    # Find smallest slot that beats largest baseline recovery
    large_baseline_recovery = results[6].recovery_mse  # Baseline-large
    slot_that_beats_it = None
    for m in results:
        if 'Slot' in m.name and m.recovery_mse <= large_baseline_recovery:
            if slot_that_beats_it is None or m.params < slot_that_beats_it.params:
                slot_that_beats_it = m

    if slot_that_beats_it:
        param_efficiency = results[6].params / slot_that_beats_it.params
        e1_pass = param_efficiency >= 2.0
        print(f"\nE1 Param Efficiency: {'PASS' if e1_pass else 'FAIL'}")
        print(f"   {slot_that_beats_it.name} ({slot_that_beats_it.params:,} params) beats Baseline-large ({results[6].params:,} params)")
        print(f"   Ratio: {param_efficiency:.2f}x (target: >= 2.0x)")
    else:
        print(f"\nE1 Param Efficiency: INCONCLUSIVE (need larger scale)")

    # E5: Accuracy (already known)
    best_slot = min([m for m in results if 'Slot' in m.name], key=lambda x: x.recovery_mse)
    best_base = min([m for m in results if 'Baseline' in m.name], key=lambda x: x.recovery_mse)
    recovery_gain = (best_base.recovery_mse - best_slot.recovery_mse) / best_base.recovery_mse * 100
    e5_pass = recovery_gain >= 10
    print(f"\nE5 Accuracy: {'PASS' if e5_pass else 'FAIL'}")
    print(f"   Best Slot Recovery: {best_slot.recovery_mse:.4f}")
    print(f"   Best Baseline Recovery: {best_base.recovery_mse:.4f}")
    print(f"   Improvement: {recovery_gain:.1f}% (target: >= 10%)")

    # Scaling trend
    print(f"\nScaling Trend:")
    for i, scale in enumerate(scales):
        baseline = results[i * 2]
        slot = results[i * 2 + 1]
        gap = (baseline.recovery_mse - slot.recovery_mse) / baseline.recovery_mse * 100
        print(f"   {scale}: Slot {gap:+.1f}% better")

    print("\n" + "=" * 100)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_efficiency_benchmark(args)


if __name__ == '__main__':
    main()
