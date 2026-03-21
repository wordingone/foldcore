"""Train 256x256 Infinite Horizon Model.

Scales up from 64x64 to 256x256 resolution.
Uses bilinear interpolation to resize TinyWorlds data.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from pathlib import Path
import numpy as np

from genesis.pilot.model_256 import InfiniteHorizonModel256
from genesis.pilot.video_data import get_video_dataset


class ResizedDataset(Dataset):
    """Wrapper that resizes video frames to target resolution."""

    def __init__(self, dataset, target_size=256):
        self.dataset = dataset
        self.target_size = target_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        frames = self.dataset[idx]  # [T, C, H, W]
        # Resize each frame
        T, C, H, W = frames.shape
        if H != self.target_size or W != self.target_size:
            frames = F.interpolate(
                frames,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            )
        return frames


def train_epoch(model, loader, optimizer, device, rollout_steps=1):
    model.train()
    total_loss = 0
    num_batches = 0

    for frames in loader:
        frames = frames.to(device)
        targets = frames[:, 2:]

        preds = model(frames, rollout_steps=rollout_steps)
        loss = F.mse_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, loader, device):
    model.eval()
    total_mse = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for frames in loader:
            frames = frames.to(device)
            targets = frames[:, 2:]

            preds = model(frames)
            mse = F.mse_loss(preds, targets)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

            total_mse += mse.item()
            total_psnr += psnr.item()
            num_batches += 1

    return {
        'mse': total_mse / num_batches,
        'psnr': total_psnr / num_batches,
    }


def test_horizon(model, dataset, num_steps, device):
    """Test infinite horizon on sequence."""
    model.eval()

    # Get long sequence
    all_frames = []
    for i in range(min(20, len(dataset))):
        all_frames.append(dataset[i])

    long_seq = torch.cat(all_frames, dim=0).unsqueeze(0).to(device)
    num_steps = min(num_steps, long_seq.shape[1] - 3)

    seed = long_seq[:, :2]
    generated = model.generate(seed, num_steps)

    gt = long_seq[:, 2:2+num_steps]

    mse_over_time = []
    for t in range(num_steps):
        mse = F.mse_loss(generated[:, t], gt[:, t]).item()
        mse_over_time.append(mse)

    mse = np.array(mse_over_time)
    initial = mse[:30].mean()
    final = mse[-30:].mean() if len(mse) >= 60 else mse[-len(mse)//2:].mean()

    return {
        'explosion_ratio': final / initial,
        'mean_mse': mse.mean(),
        'initial_mse': initial,
        'final_mse': final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='sonic')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)  # Smaller batch for 256x256
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-samples', type=int, default=1500)
    parser.add_argument('--horizon-steps', type=int, default=150)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints/256')
    args = parser.parse_args()

    print("=" * 70)
    print(f"TRAINING 256x256 MODEL ON {args.dataset.upper()}")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Max samples: {args.max_samples}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    base_dataset = get_video_dataset(f"tinyworlds:{args.dataset}", seq_length=12)

    # Wrap with resizer
    full_dataset = ResizedDataset(base_dataset, target_size=256)

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
    model = InfiniteHorizonModel256(
        base_channels=48,
        num_slots=16,
        slot_dim=64,
        slot_decay=0.95,
    )
    model = model.to(args.device)
    print(f"Parameters: {model.count_parameters():,}")

    # Memory check
    if args.device == 'cuda':
        test_batch = torch.randn(args.batch_size, 12, 3, 256, 256).to(args.device)
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(test_batch)
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"Peak GPU memory: {peak_mb:.1f} MB")
        del test_batch
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Progressive rollout
    rollout_schedule = [(0, 1), (3, 2), (6, 4)]

    def get_rollout(epoch):
        r = 1
        for e, s in rollout_schedule:
            if epoch >= e:
                r = s
        return r

    # Train
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_horizon = float('inf')

    for epoch in range(args.epochs):
        rollout = get_rollout(epoch)
        train_loss = train_epoch(model, train_loader, optimizer, args.device, rollout)
        val_metrics = evaluate(model, val_loader, args.device)
        horizon = test_horizon(model, val_dataset, args.horizon_steps, args.device)

        scheduler.step()

        print(f"Epoch {epoch+1:2d}: Loss={train_loss:.6f} | "
              f"Val MSE={val_metrics['mse']:.6f} PSNR={val_metrics['psnr']:.1f} | "
              f"Horizon={horizon['explosion_ratio']:.2f}x | R={rollout}")

        if horizon['explosion_ratio'] < best_horizon:
            best_horizon = horizon['explosion_ratio']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'horizon_ratio': best_horizon,
                'val_mse': val_metrics['mse'],
            }, save_dir / f"best_256_{args.dataset}.pt")
            print(f"  -> Saved (best horizon)")

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    final_val = evaluate(model, val_loader, args.device)
    final_horizon = test_horizon(model, val_dataset, args.horizon_steps, args.device)

    print(f"\nValidation:")
    print(f"  MSE:  {final_val['mse']:.6f}")
    print(f"  PSNR: {final_val['psnr']:.2f} dB")

    print(f"\nInfinite Horizon ({args.horizon_steps} steps):")
    print(f"  Explosion Ratio: {final_horizon['explosion_ratio']:.2f}x")
    print(f"  Initial MSE:     {final_horizon['initial_mse']:.6f}")
    print(f"  Final MSE:       {final_horizon['final_mse']:.6f}")

    if final_horizon['explosion_ratio'] < 2.0:
        print(f"\n  PASS: 256x256 model can play {args.dataset} indefinitely!")
    else:
        print(f"\n  PARTIAL: Error grows but may be acceptable")

    print(f"\nBest horizon ratio: {best_horizon:.2f}x")
    print(f"Checkpoints: {save_dir}")


if __name__ == '__main__':
    main()
