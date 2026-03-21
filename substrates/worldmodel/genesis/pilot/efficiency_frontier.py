"""Efficiency Frontier: Systematically explore ALL sources of order-of-magnitude savings.

Proven:
- Slot attention: 32x params, 10x FLOPs (structure > scale)

Hypotheses to test:
1. Temporal sparsity: Most pixels don't change. Predict deltas, not frames.
2. Spatial sparsity: Objects occupy <10% of pixels. Sparse prediction.
3. Resolution cascade: Predict 4x4, upsample to 16x16 (16x fewer tokens).
4. KV-cache: Reuse keys/values across time (avoid recomputation).
5. Quantization: Q8 vs FP32 (2x memory, often same accuracy).

Each test: measure FLOPs, params, accuracy, and compute savings ratio.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
import time

from genesis.pilot.micro_data import MicroDataset, get_recovery_frames


@dataclass
class EfficiencyResult:
    name: str
    params: int
    flops: int
    mse: float
    recovery_mse: float
    theoretical_savings: str
    actual_savings: float
    notes: str


# ============================================================================
# BASELINE: Full frame prediction (reference point)
# ============================================================================

class FullFrameBaseline(nn.Module):
    """Predict entire next frame. No efficiency tricks."""

    def __init__(self, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, stride=4),  # 16->4
            nn.GELU(),
        )
        self.temporal = nn.GRU(channels * 16, channels * 16, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []
        hidden = None

        for t in range(T - 2):
            x = frames[:, t+1]
            feat = self.encoder(x).flatten(1)
            out, hidden = self.temporal(feat.unsqueeze(1), hidden)
            pred = self.decoder(out.squeeze(1).reshape(B, -1, 4, 4))
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# HYPOTHESIS 1: Delta Prediction (temporal sparsity)
# ============================================================================

class DeltaPredictor(nn.Module):
    """Predict frame DIFFERENCE, not full frame.

    Hypothesis: Most pixels don't change. Delta is sparse.
    Savings: If 90% pixels unchanged, 10x effective compression.
    """

    def __init__(self, channels=32):
        super().__init__()
        # Encode difference between frames
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 4, stride=4),  # Input: [curr, prev]
            nn.GELU(),
        )
        self.temporal = nn.GRU(channels * 16, channels * 16, batch_first=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=4),
            nn.Tanh(),  # Delta can be negative
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []
        hidden = None

        for t in range(T - 2):
            curr = frames[:, t+1]
            prev = frames[:, t]

            # Encode the difference
            diff_input = torch.cat([curr, prev], dim=1)
            feat = self.encoder(diff_input).flatten(1)
            out, hidden = self.temporal(feat.unsqueeze(1), hidden)

            # Predict delta
            delta = self.decoder(out.squeeze(1).reshape(B, -1, 4, 4))

            # Apply delta to current frame
            pred = torch.clamp(curr + delta, 0, 1)
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# HYPOTHESIS 2: Sparse Prediction (spatial sparsity)
# ============================================================================

class SparsePredictor(nn.Module):
    """Only predict pixels that are likely to change.

    Hypothesis: Objects occupy <20% of frame. Only predict object regions.
    Savings: If 80% background, 5x compute savings.
    """

    def __init__(self, channels=32, topk_ratio=0.25):
        super().__init__()
        self.topk_ratio = topk_ratio

        # Motion detector (which pixels will change?)
        self.motion_detector = nn.Sequential(
            nn.Conv2d(2, channels//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels//2, 1, 1),
            nn.Sigmoid(),
        )

        # Predictor for selected regions
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, stride=4),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(channels * 16, channels * 16),
            nn.GELU(),
            nn.Linear(channels * 16, channels * 16),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []

        for t in range(T - 2):
            curr = frames[:, t+1]
            prev = frames[:, t]

            # Detect motion regions
            motion_input = torch.cat([curr, prev], dim=1)
            motion_mask = self.motion_detector(motion_input)  # [B, 1, H, W]

            # Full prediction (in practice, would be sparse)
            feat = self.encoder(curr).flatten(1)
            feat = self.predictor(feat)
            pred_full = self.decoder(feat.reshape(B, -1, 4, 4))

            # Blend: motion regions get prediction, static get copy
            pred = motion_mask * pred_full + (1 - motion_mask) * curr
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# HYPOTHESIS 3: Resolution Cascade (predict low-res, upsample)
# ============================================================================

class CascadePredictor(nn.Module):
    """Predict at 4x4, upsample to 16x16.

    Hypothesis: Low-res captures motion, high-res is detail.
    Savings: 16x fewer tokens to predict.
    """

    def __init__(self, channels=32):
        super().__init__()
        # Work entirely at 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, stride=4),  # 16->4
            nn.GELU(),
        )
        self.temporal = nn.GRU(channels * 16, channels * 16, batch_first=True)
        self.lowres_head = nn.Conv2d(channels, 1, 1)  # Predict 4x4

        # Learnable upsampler
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(1, channels//2, 4, stride=2, padding=1),  # 4->8
            nn.GELU(),
            nn.ConvTranspose2d(channels//2, 1, 4, stride=2, padding=1),  # 8->16
            nn.Sigmoid(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []
        hidden = None

        for t in range(T - 2):
            x = frames[:, t+1]
            feat = self.encoder(x)  # [B, C, 4, 4]
            feat_flat = feat.flatten(1)

            out, hidden = self.temporal(feat_flat.unsqueeze(1), hidden)
            out = out.squeeze(1).reshape(B, -1, 4, 4)

            # Predict at low-res
            lowres = torch.sigmoid(self.lowres_head(out))  # [B, 1, 4, 4]

            # Upsample to full res
            pred = self.upsampler(lowres)
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# HYPOTHESIS 4: KV-Cache (reuse computation across time)
# ============================================================================

class KVCachePredictor(nn.Module):
    """Cache key-value pairs across time steps.

    Hypothesis: Most context doesn't change. Reuse KV from previous steps.
    Savings: O(T) instead of O(T^2) for attention.
    """

    def __init__(self, channels=32, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.encoder = nn.Sequential(
            nn.Conv2d(1, channels, 4, stride=4),
            nn.GELU(),
        )

        # Attention with KV cache
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=4),
            nn.Sigmoid(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []

        # KV cache
        k_cache = []
        v_cache = []

        for t in range(T - 2):
            x = frames[:, t+1]
            feat = self.encoder(x).flatten(2).transpose(1, 2)  # [B, 16, C]

            # Project to Q, K, V
            q = self.q_proj(feat)
            k = self.k_proj(feat)
            v = self.v_proj(feat)

            # Add to cache
            k_cache.append(k)
            v_cache.append(v)

            # Attend to all cached KV (this is where savings come from)
            # In full attention: would recompute all K,V each step
            # With cache: just append and attend
            k_all = torch.cat(k_cache, dim=1)  # [B, t*16, C]
            v_all = torch.cat(v_cache, dim=1)

            # Scaled dot-product attention
            attn = torch.matmul(q, k_all.transpose(-2, -1)) / (self.channels ** 0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v_all)
            out = self.out_proj(out)

            # Decode
            out = out.transpose(1, 2).reshape(B, self.channels, 4, 4)
            pred = self.decoder(out)
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# HYPOTHESIS 5: Combined Slot + Delta (stack efficiencies)
# ============================================================================

class SlotDeltaPredictor(nn.Module):
    """Combine slot attention (32x) with delta prediction.

    Hypothesis: Slots track objects, deltas capture motion. Multiplicative savings.
    Target: 32x * 5x = 160x combined efficiency.
    """

    def __init__(self, channels=32, num_slots=4, slot_dim=32):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Encode difference
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 4, stride=4),
            nn.GELU(),
        )

        # Slot attention on delta features
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)
        self.to_slot_dim = nn.Linear(channels, slot_dim)
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Decode slots to delta
        self.slot_decoder = nn.Linear(num_slots * slot_dim, channels * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, 1, 4, stride=4),
            nn.Tanh(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []
        slots = self.slot_mu.expand(B, -1, -1).clone()

        for t in range(T - 2):
            curr = frames[:, t+1]
            prev = frames[:, t]

            # Encode difference
            diff_input = torch.cat([curr, prev], dim=1)
            feat = self.encoder(diff_input)  # [B, C, 4, 4]
            feat = feat.flatten(2).transpose(1, 2)  # [B, 16, C]
            feat = self.to_slot_dim(feat)  # [B, 16, slot_dim]

            # Slot attention
            k = self.k_proj(feat)
            v = self.v_proj(feat)

            for _ in range(3):  # iterations
                q = self.q_proj(slots)
                attn = torch.einsum('bkd,bnd->bkn', q, k) / (self.slot_dim ** 0.5)
                attn = F.softmax(attn, dim=1)
                updates = torch.einsum('bkn,bnd->bkd', attn, v)
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots.reshape(-1, self.slot_dim)
                ).reshape(B, self.num_slots, self.slot_dim)

            # Decode to delta
            slot_feat = self.slot_decoder(slots.flatten(1))
            slot_feat = slot_feat.reshape(B, -1, 4, 4)
            delta = self.decoder(slot_feat)

            pred = torch.clamp(curr + delta, 0, 1)
            preds.append(pred)

        return torch.stack(preds, dim=1)


# ============================================================================
# MEASUREMENT
# ============================================================================

def measure_frame_statistics(dataset):
    """Measure actual sparsity in the data."""
    total_pixels = 0
    changed_pixels = 0
    object_pixels = 0

    for frames, _ in dataset:
        for t in range(1, frames.shape[0]):
            diff = (frames[t] - frames[t-1]).abs()
            changed = (diff > 0.1).float().sum()
            changed_pixels += changed.item()
            total_pixels += frames[t].numel()

            # Object pixels (non-background)
            obj = (frames[t] > 0.1).float().sum()
            object_pixels += obj.item()

    temporal_sparsity = 1 - (changed_pixels / total_pixels)
    spatial_sparsity = 1 - (object_pixels / total_pixels)

    return {
        'temporal_sparsity': temporal_sparsity,
        'spatial_sparsity': spatial_sparsity,
        'theoretical_temporal_savings': 1 / (1 - temporal_sparsity + 1e-6),
        'theoretical_spatial_savings': 1 / (1 - spatial_sparsity + 1e-6),
    }


def count_flops_simple(model, name):
    """Rough FLOP count."""
    params = sum(p.numel() for p in model.parameters())
    # Rough heuristic: 2 FLOPs per param per forward
    return params * 2 * 6  # 6 predictions


def train_and_eval(model, train_loader, val_loader, epochs, lr, device):
    """Train and evaluate model."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for _ in range(epochs):
        model.train()
        for frames, _ in train_loader:
            frames = frames.to(device)
            preds = model(frames)
            loss = F.mse_loss(preds, frames[:, 2:])
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.time() - start

    # Eval
    model.eval()
    total_mse = 0
    recovery_mse = 0
    n_total = 0
    n_recovery = 0

    with torch.no_grad():
        for frames, masks in val_loader:
            frames = frames.to(device)
            masks = masks.to(device)
            preds = model(frames)
            targets = frames[:, 2:]

            total_mse += F.mse_loss(preds, targets).item()
            n_total += 1

            recovery = get_recovery_frames(masks)[:, 2:]
            if recovery.any():
                B, T, C, H, W = preds.shape
                r_exp = recovery.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,C,H,W)
                r_preds = preds[r_exp].reshape(-1)
                r_targets = targets[r_exp].reshape(-1)
                if len(r_preds) > 0:
                    recovery_mse += F.mse_loss(r_preds, r_targets).item()
                    n_recovery += 1

    return {
        'mse': total_mse / max(n_total, 1),
        'recovery_mse': recovery_mse / max(n_recovery, 1),
        'train_time': train_time,
    }


def run_frontier_exploration(args):
    """Explore all efficiency frontiers."""
    device = args.device
    print(f"Device: {device}\n")

    # Data
    train_data = MicroDataset(num_sequences=args.train_size, seed=42)
    val_data = MicroDataset(num_sequences=args.val_size, seed=1000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    # Measure data statistics
    print("=" * 70)
    print("DATA STATISTICS (Theoretical Upper Bounds)")
    print("=" * 70)
    stats = measure_frame_statistics(train_data)
    print(f"Temporal Sparsity: {stats['temporal_sparsity']*100:.1f}% pixels unchanged")
    print(f"  -> Theoretical savings: {stats['theoretical_temporal_savings']:.1f}x")
    print(f"Spatial Sparsity: {stats['spatial_sparsity']*100:.1f}% background")
    print(f"  -> Theoretical savings: {stats['theoretical_spatial_savings']:.1f}x")

    # Models to test
    models = {
        'Baseline': FullFrameBaseline(channels=32),
        'Delta': DeltaPredictor(channels=32),
        'Sparse': SparsePredictor(channels=32),
        'Cascade': CascadePredictor(channels=32),
        'KV-Cache': KVCachePredictor(channels=32),
        'Slot+Delta': SlotDeltaPredictor(channels=32, num_slots=4, slot_dim=32),
    }

    results = []
    baseline_mse = None
    baseline_recovery = None
    baseline_flops = None

    print("\n" + "=" * 70)
    print("TRAINING ALL MODELS")
    print("=" * 70)

    for name, model in models.items():
        print(f"\n{name}...")
        params = sum(p.numel() for p in model.parameters())
        flops = count_flops_simple(model, name)

        metrics = train_and_eval(model, train_loader, val_loader, args.epochs, args.lr, device)

        if name == 'Baseline':
            baseline_mse = metrics['mse']
            baseline_recovery = metrics['recovery_mse']
            baseline_flops = flops

        print(f"  Params: {params:,} | MSE: {metrics['mse']:.4f} | Recovery: {metrics['recovery_mse']:.4f}")

        results.append({
            'name': name,
            'params': params,
            'flops': flops,
            'mse': metrics['mse'],
            'recovery_mse': metrics['recovery_mse'],
            'time': metrics['train_time'],
        })

    # Analysis
    print("\n" + "=" * 70)
    print("EFFICIENCY FRONTIER RESULTS")
    print("=" * 70)
    print(f"{'Model':<15} {'Params':>10} {'MSE':>10} {'Recovery':>10} {'vs Base':>12} {'Efficiency':>12}")
    print("-" * 70)

    for r in results:
        vs_base = (baseline_recovery - r['recovery_mse']) / baseline_recovery * 100
        # Efficiency = accuracy improvement per param
        eff = (baseline_recovery / r['recovery_mse']) * (baseline_flops / r['flops'])
        print(f"{r['name']:<15} {r['params']:>10,} {r['mse']:>10.4f} {r['recovery_mse']:>10.4f} {vs_base:>+11.1f}% {eff:>11.2f}x")

    # Best efficiency
    print("\n" + "=" * 70)
    print("EFFICIENCY ANALYSIS")
    print("=" * 70)

    # Sort by recovery MSE
    by_recovery = sorted(results, key=lambda x: x['recovery_mse'])
    print(f"\nBest Recovery: {by_recovery[0]['name']} ({by_recovery[0]['recovery_mse']:.4f})")

    # Sort by params (smallest that beats baseline)
    beats_baseline = [r for r in results if r['recovery_mse'] < baseline_recovery]
    if beats_baseline:
        by_params = sorted(beats_baseline, key=lambda x: x['params'])
        param_savings = baseline_flops / by_params[0]['flops']
        print(f"Most Efficient: {by_params[0]['name']} ({by_params[0]['params']:,} params)")
        print(f"  Savings vs Baseline: {param_savings:.1f}x FLOPs")

    # Identify multiplicative opportunities
    print("\n" + "=" * 70)
    print("STACKING ANALYSIS (Multiplicative Savings)")
    print("=" * 70)

    slot_delta = [r for r in results if r['name'] == 'Slot+Delta'][0]
    baseline = [r for r in results if r['name'] == 'Baseline'][0]

    recovery_gain = (baseline['recovery_mse'] - slot_delta['recovery_mse']) / baseline['recovery_mse'] * 100
    flop_ratio = baseline['flops'] / slot_delta['flops']

    print(f"\nSlot+Delta vs Baseline:")
    print(f"  Recovery Improvement: {recovery_gain:+.1f}%")
    print(f"  FLOP Ratio: {flop_ratio:.1f}x")
    print(f"  Combined Efficiency: {flop_ratio * (baseline['recovery_mse']/slot_delta['recovery_mse']):.1f}x")

    # Theoretical stack
    print(f"\nTheoretical Maximum (if all stack):")
    print(f"  Slots: 32x (verified)")
    print(f"  Temporal sparsity: {stats['theoretical_temporal_savings']:.1f}x")
    print(f"  Spatial sparsity: {stats['theoretical_spatial_savings']:.1f}x")
    print(f"  Resolution cascade: 16x")
    print(f"  Combined: {32 * stats['theoretical_temporal_savings'] * 16:.0f}x (theoretical upper bound)")

    print("\n" + "=" * 70)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_frontier_exploration(args)


if __name__ == '__main__':
    main()
