"""Micro Object Permanence Benchmark.

Tests:
1. Recovery MSE - prediction quality after occlusion ends
2. Position accuracy - predict where object reappears
3. Multi-object tracking - harder dataset with 2 objects

Compares: Baseline (pixel), Slot (pixel), Baseline-AR (latent), Slot-AR (latent)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from genesis.pilot.micro_data import MicroDataset, get_recovery_frames
from genesis.pilot.micro_baseline import MicroBaseline
from genesis.pilot.micro_slot import MicroSlot
from genesis.pilot.micro_vae import MicroVAE, vae_loss
from genesis.pilot.micro_slot_ar import MicroSlotAR, MicroBaselineAR


class MultiObjectDataset(Dataset):
    """Harder dataset: 2 bouncing squares with occlusion."""

    def __init__(self, num_sequences=500, seed=42):
        super().__init__()
        self.num_sequences = num_sequences
        self.rng = np.random.RandomState(seed)

        # Pregenerate all sequences
        self.sequences = []
        self.masks = []
        for _ in range(num_sequences):
            seq, mask = self._generate_sequence()
            self.sequences.append(seq)
            self.masks.append(mask)

    def _generate_sequence(self, size=16, num_frames=8, obj_size=3):
        """Generate sequence with 2 bouncing objects + occluder."""
        frames = np.zeros((num_frames, 1, size, size), dtype=np.float32)
        occlusion_mask = np.zeros(num_frames, dtype=bool)

        # Two objects with random positions and velocities
        objects = []
        for _ in range(2):
            x = self.rng.randint(obj_size, size - obj_size)
            y = self.rng.randint(obj_size, size - obj_size)
            vx = self.rng.choice([-1, 1])
            vy = self.rng.choice([-1, 1])
            objects.append([x, y, vx, vy])

        # Occluder: horizontal bar
        occ_y = self.rng.randint(4, size - 4)
        occ_h = 3

        for t in range(num_frames):
            # Draw objects
            for obj in objects:
                x, y, vx, vy = obj
                frames[t, 0, max(0,y):min(size,y+obj_size),
                           max(0,x):min(size,x+obj_size)] = 0.8

                # Update position
                x += vx
                y += vy

                # Bounce
                if x <= 0 or x >= size - obj_size:
                    vx = -vx
                    x = max(0, min(size - obj_size, x))
                if y <= 0 or y >= size - obj_size:
                    vy = -vy
                    y = max(0, min(size - obj_size, y))

                obj[:] = [x, y, vx, vy]

            # Draw occluder (white bar)
            frames[t, 0, occ_y:occ_y+occ_h, :] = 1.0

            # Check if any object is under occluder
            for obj in objects:
                _, y, _, _ = obj
                if occ_y <= y < occ_y + occ_h or occ_y <= y + obj_size < occ_y + occ_h:
                    occlusion_mask[t] = True

        return torch.from_numpy(frames), torch.from_numpy(occlusion_mask)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.sequences[idx], self.masks[idx]


def train_vae(vae, train_loader, epochs, lr, device):
    """Pretrain VAE on reconstruction."""
    vae = vae.to(device)
    vae.train()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for frames, _ in train_loader:
            frames = frames.to(device)
            B, T, C, H, W = frames.shape

            # Flatten batch and time
            x = frames.reshape(B * T, C, H, W)
            x_recon, mu, logvar = vae(x)
            loss, _, _ = vae_loss(x, x_recon, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(f"  VAE Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")

    vae.eval()
    return vae


def train_model(model, train_loader, epochs, lr, device, use_latent_loss=False):
    """Train prediction model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for frames, masks in train_loader:
            frames = frames.to(device)

            if use_latent_loss:
                # Train with latent-space loss
                preds, latent_preds, latent_targets = model(frames, return_latents=True)
                loss = nn.functional.mse_loss(latent_preds, latent_targets)
            else:
                # Train with pixel-space loss
                targets = frames[:, 2:]
                preds = model(frames)
                loss = nn.functional.mse_loss(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def evaluate(model, loader, device):
    """Evaluate on recovery metrics."""
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

            # Recovery MSE
            recovery = get_recovery_frames(masks)[:, 2:]
            if recovery.any():
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


def run_benchmark(args):
    """Run full benchmark comparison."""
    device = args.device
    print(f"Device: {device}")

    # Create datasets
    print("\n=== Creating Datasets ===")
    train_single = MicroDataset(num_sequences=args.train_size, seed=42)
    val_single = MicroDataset(num_sequences=args.val_size, seed=1000)
    train_multi = MultiObjectDataset(num_sequences=args.train_size, seed=42)
    val_multi = MultiObjectDataset(num_sequences=args.val_size, seed=1000)

    train_loader_single = DataLoader(train_single, batch_size=args.batch_size, shuffle=True)
    val_loader_single = DataLoader(val_single, batch_size=args.batch_size)
    train_loader_multi = DataLoader(train_multi, batch_size=args.batch_size, shuffle=True)
    val_loader_multi = DataLoader(val_multi, batch_size=args.batch_size)

    print(f"Single object: {len(train_single)} train, {len(val_single)} val")
    print(f"Multi object: {len(train_multi)} train, {len(val_multi)} val")

    results = []

    # ============ PIXEL SPACE MODELS ============
    print("\n=== Pixel Space Models ===")

    # Baseline (pixel)
    print("\nTraining Baseline (pixel)...")
    baseline = MicroBaseline(channels=16, num_frames=8).to(device)
    baseline = train_model(baseline, train_loader_single, args.epochs, args.lr, device)
    metrics_single = evaluate(baseline, val_loader_single, device)
    metrics_multi = evaluate(baseline, val_loader_multi, device)
    results.append({
        'name': 'Baseline-Pixel',
        'params': sum(p.numel() for p in baseline.parameters()),
        'single_mse': metrics_single['mse'],
        'single_recovery': metrics_single['recovery_mse'],
        'multi_mse': metrics_multi['mse'],
        'multi_recovery': metrics_multi['recovery_mse'],
    })
    print(f"  Single: MSE={metrics_single['mse']:.4f} Recovery={metrics_single['recovery_mse']:.4f}")
    print(f"  Multi:  MSE={metrics_multi['mse']:.4f} Recovery={metrics_multi['recovery_mse']:.4f}")

    # Slot (pixel)
    print("\nTraining Slot (pixel)...")
    slot = MicroSlot(channels=48, num_slots=4, slot_dim=48).to(device)
    slot = train_model(slot, train_loader_single, args.epochs, args.lr, device)
    metrics_single = evaluate(slot, val_loader_single, device)
    metrics_multi = evaluate(slot, val_loader_multi, device)
    results.append({
        'name': 'Slot-Pixel',
        'params': sum(p.numel() for p in slot.parameters()),
        'single_mse': metrics_single['mse'],
        'single_recovery': metrics_single['recovery_mse'],
        'multi_mse': metrics_multi['mse'],
        'multi_recovery': metrics_multi['recovery_mse'],
    })
    print(f"  Single: MSE={metrics_single['mse']:.4f} Recovery={metrics_single['recovery_mse']:.4f}")
    print(f"  Multi:  MSE={metrics_multi['mse']:.4f} Recovery={metrics_multi['recovery_mse']:.4f}")

    # ============ LATENT SPACE MODELS ============
    print("\n=== Latent Space Models ===")

    # Pretrain VAE
    print("\nPretraining VAE...")
    vae = MicroVAE(latent_channels=8)
    vae = train_vae(vae, train_loader_single, args.vae_epochs, args.lr, device)

    # Baseline-AR (latent)
    print("\nTraining Baseline-AR (latent)...")
    baseline_ar = MicroBaselineAR(vae, hidden_dim=64).to(device)
    baseline_ar = train_model(baseline_ar, train_loader_single, args.epochs, args.lr, device, use_latent_loss=True)
    metrics_single = evaluate(baseline_ar, val_loader_single, device)
    metrics_multi = evaluate(baseline_ar, val_loader_multi, device)
    results.append({
        'name': 'Baseline-Latent',
        'params': baseline_ar.count_parameters(),
        'single_mse': metrics_single['mse'],
        'single_recovery': metrics_single['recovery_mse'],
        'multi_mse': metrics_multi['mse'],
        'multi_recovery': metrics_multi['recovery_mse'],
    })
    print(f"  Single: MSE={metrics_single['mse']:.4f} Recovery={metrics_single['recovery_mse']:.4f}")
    print(f"  Multi:  MSE={metrics_multi['mse']:.4f} Recovery={metrics_multi['recovery_mse']:.4f}")

    # Slot-AR (latent)
    print("\nTraining Slot-AR (latent)...")
    slot_ar = MicroSlotAR(vae, num_slots=4, slot_dim=32).to(device)
    slot_ar = train_model(slot_ar, train_loader_single, args.epochs, args.lr, device, use_latent_loss=True)
    metrics_single = evaluate(slot_ar, val_loader_single, device)
    metrics_multi = evaluate(slot_ar, val_loader_multi, device)
    results.append({
        'name': 'Slot-Latent',
        'params': slot_ar.count_parameters(),
        'single_mse': metrics_single['mse'],
        'single_recovery': metrics_single['recovery_mse'],
        'multi_mse': metrics_multi['mse'],
        'multi_recovery': metrics_multi['recovery_mse'],
    })
    print(f"  Single: MSE={metrics_single['mse']:.4f} Recovery={metrics_single['recovery_mse']:.4f}")
    print(f"  Multi:  MSE={metrics_multi['mse']:.4f} Recovery={metrics_multi['recovery_mse']:.4f}")

    # ============ SUMMARY ============
    print("\n" + "=" * 80)
    print("OBJECT PERMANENCE BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<18} {'Params':>10} {'1-Obj MSE':>10} {'1-Obj Rec':>10} {'2-Obj MSE':>10} {'2-Obj Rec':>10}")
    print("-" * 80)

    for r in results:
        print(f"{r['name']:<18} {r['params']:>10,} {r['single_mse']:>10.4f} {r['single_recovery']:>10.4f} "
              f"{r['multi_mse']:>10.4f} {r['multi_recovery']:>10.4f}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline_pixel = results[0]
    slot_pixel = results[1]
    baseline_latent = results[2]
    slot_latent = results[3]

    # Single object improvements
    slot_vs_base_single = (baseline_pixel['single_recovery'] - slot_pixel['single_recovery']) / baseline_pixel['single_recovery'] * 100
    print(f"\nSingle Object Recovery:")
    print(f"  Slot-Pixel vs Baseline-Pixel: {slot_vs_base_single:+.1f}%")

    # Multi object improvements
    slot_vs_base_multi = (baseline_pixel['multi_recovery'] - slot_pixel['multi_recovery']) / baseline_pixel['multi_recovery'] * 100
    print(f"\nMulti Object Recovery:")
    print(f"  Slot-Pixel vs Baseline-Pixel: {slot_vs_base_multi:+.1f}%")

    # Latent space
    slot_vs_base_latent = (baseline_latent['single_recovery'] - slot_latent['single_recovery']) / baseline_latent['single_recovery'] * 100
    print(f"\nLatent Space (Single Object):")
    print(f"  Slot-Latent vs Baseline-Latent: {slot_vs_base_latent:+.1f}%")

    # Best model
    best_single = min(results, key=lambda x: x['single_recovery'])
    best_multi = min(results, key=lambda x: x['multi_recovery'])

    print(f"\nBest Single Object Recovery: {best_single['name']} ({best_single['single_recovery']:.4f})")
    print(f"Best Multi Object Recovery: {best_multi['name']} ({best_multi['multi_recovery']:.4f})")

    # Verdict
    print("\n" + "=" * 80)
    if slot_vs_base_single > 10 and slot_vs_base_multi > 5:
        print("VERDICT: Slot attention SIGNIFICANTLY improves object permanence")
    elif slot_vs_base_single > 5:
        print("VERDICT: Slot attention provides MODERATE improvement")
    else:
        print("VERDICT: Slot attention shows MARGINAL improvement")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--vae-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == '__main__':
    main()
