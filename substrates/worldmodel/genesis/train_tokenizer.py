"""Train Genesis tokenizer in isolation (Phase 1 of split training).

This script trains only the tokenizer using reconstruction loss.
The tokenizer learns to encode/decode video frames without dynamics prediction.

Usage:
    # MSE only (baseline)
    python genesis/train_tokenizer.py --epochs 50 --image-size 256 \
        --save-dir checkpoints/tokenizer_256_mse

    # With perceptual loss (for better CLIP-IQA)
    python genesis/train_tokenizer.py --epochs 50 --image-size 256 \
        --use-perceptual --perceptual-weight 0.1 \
        --save-dir checkpoints/tokenizer_256_perceptual
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Optional

from genesis.tokenizer.motion import MotionAwareTokenizer


def train_step(
    tokenizer: MotionAwareTokenizer,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: str,
    perceptual_loss_fn: Optional[nn.Module] = None,
    perceptual_weight: float = 0.1,
) -> dict:
    """Single training step for tokenizer only."""
    tokenizer.train()

    # Get video
    if 'frames' in batch:
        video = batch['frames'].to(device)
    else:
        video = batch['video'].to(device)

    # Encode to latent
    latents, intermediates = tokenizer.encode(video)

    # Decode back to video
    H, W = video.shape[-2:]
    reconstructed = tokenizer.decode(latents, (H, W))

    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(reconstructed, video)

    # Perceptual loss (optional)
    if perceptual_loss_fn is not None:
        perceptual_loss = perceptual_loss_fn(reconstructed, video)
        total_loss = recon_loss + perceptual_weight * perceptual_loss
    else:
        perceptual_loss = torch.tensor(0.0, device=device)
        total_loss = recon_loss

    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
    optimizer.step()

    # Compute PSNR
    with torch.no_grad():
        mse = F.mse_loss(reconstructed, video)
        psnr = 10 * torch.log10(1.0 / mse) if mse > 0 else torch.tensor(100.0)

    return {
        'recon_loss': recon_loss.item(),
        'perceptual_loss': perceptual_loss.item() if perceptual_loss_fn else 0.0,
        'total_loss': total_loss.item(),
        'psnr': psnr.item(),
    }


@torch.no_grad()
def evaluate(
    tokenizer: MotionAwareTokenizer,
    loader: DataLoader,
    device: str,
    max_batches: int = 10,
) -> dict:
    """Evaluate tokenizer reconstruction quality."""
    tokenizer.eval()

    total_loss = 0
    total_psnr = 0
    count = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break

        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        latents, _ = tokenizer.encode(video)
        H, W = video.shape[-2:]
        reconstructed = tokenizer.decode(latents, (H, W))

        loss = F.mse_loss(reconstructed, video)
        psnr = 10 * torch.log10(1.0 / loss) if loss > 0 else torch.tensor(100.0)

        total_loss += loss.item()
        total_psnr += psnr.item()
        count += 1

    return {
        'val_loss': total_loss / count if count > 0 else 0,
        'val_psnr': total_psnr / count if count > 0 else 0,
    }


def create_dataloader(
    dataset_name: str,
    batch_size: int,
    seq_length: int,
    image_size: int,
    game: Optional[str] = None,
) -> DataLoader:
    """Create data loader for training."""
    from genesis.pilot.stream_hf import create_streaming_loader

    # Default game for JAT dataset
    if dataset_name == 'jat' and game is None:
        game = 'atari-breakout'

    return create_streaming_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        seq_length=seq_length,
        image_size=image_size,
        game=game,
    )


def main():
    parser = argparse.ArgumentParser(description='Train Genesis tokenizer (Phase 1)')
    parser.add_argument('--dataset', type=str, default='jat', help='Dataset name')
    parser.add_argument('--game', type=str, default=None, help='Game name for JAT dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=16, help='Sequence length')
    parser.add_argument('--image-size', type=int, default=64, help='Image resolution')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='checkpoints/tokenizer',
                        help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--samples-per-epoch', type=int, default=5000,
                        help='Samples per epoch')
    # Perceptual loss options
    parser.add_argument('--use-perceptual', action='store_true',
                        help='Use perceptual loss (LPIPS/VGG) for better CLIP-IQA')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Weight for perceptual loss (default: 0.1)')
    parser.add_argument('--perceptual-type', type=str, default='vgg',
                        choices=['lpips', 'vgg'],
                        help='Type of perceptual loss (default: vgg)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('GENESIS TOKENIZER TRAINING (Phase 1)')
    print('=' * 70)
    print(f'Dataset: {args.dataset}')
    print(f'Resolution: {args.image_size}')
    print(f'Perceptual loss: {args.use_perceptual} ({args.perceptual_type}, weight={args.perceptual_weight})')
    print(f'Device: {device}')
    print('=' * 70)

    # Create tokenizer
    print('\nCreating tokenizer...')
    tokenizer = MotionAwareTokenizer(
        in_channels=3,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=args.image_size,
        adaptive_channels=True,
    ).to(device)

    params = sum(p.numel() for p in tokenizer.parameters())
    print(f'Tokenizer parameters: {params:,}')
    print(f'Latent channels: {tokenizer.latent_channels}')
    print(f'Latent size: {tokenizer.latent_size}x{tokenizer.latent_size}')

    # Initialize perceptual loss (if enabled)
    perceptual_loss_fn = None
    if args.use_perceptual:
        print(f'\nInitializing {args.perceptual_type.upper()} perceptual loss...')
        if args.perceptual_type == 'lpips':
            from genesis.tokenizer.losses import LPIPSLoss
            perceptual_loss_fn = LPIPSLoss().to(device)
        else:
            from genesis.tokenizer.losses import VGGPerceptualLoss
            perceptual_loss_fn = VGGPerceptualLoss().to(device)
        # Freeze perceptual loss network
        for param in perceptual_loss_fn.parameters():
            param.requires_grad = False
        print(f'  Perceptual loss ready (weight={args.perceptual_weight})')

    # Optimizer
    optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.samples_per_epoch // args.batch_size
    )

    # Resume
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        print(f'\nResuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        tokenizer.load_state_dict(ckpt['tokenizer_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('psnr', 0)

    # Data loader
    print('\nCreating data loader...')
    loader = create_dataloader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        image_size=args.image_size,
        game=args.game if args.dataset == 'jat' else None,
    )

    # Training loop
    print('\n' + '=' * 70)
    print('TRAINING')
    print('=' * 70)

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        samples_processed = 0
        epoch_loss = 0
        batch_count = 0

        for batch in loader:
            if samples_processed >= args.samples_per_epoch:
                break

            metrics = train_step(
                tokenizer, batch, optimizer, device,
                perceptual_loss_fn=perceptual_loss_fn,
                perceptual_weight=args.perceptual_weight,
            )
            scheduler.step()

            samples_processed += batch['frames'].shape[0] if 'frames' in batch else batch['video'].shape[0]
            epoch_loss += metrics['total_loss']
            batch_count += 1

            if batch_count % 25 == 0:
                elapsed = time.time() - epoch_start
                rate = samples_processed / elapsed if elapsed > 0 else 0
                perc_str = f" perc={metrics['perceptual_loss']:.4f}" if args.use_perceptual else ""
                print(f"Epoch {epoch+1}/{args.epochs} | Batch {batch_count} | "
                      f"recon={metrics['recon_loss']:.4f}{perc_str} psnr={metrics['psnr']:.1f}dB | "
                      f"{rate:.1f} samples/s")

        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0

        print(f'\n--- Epoch {epoch+1} Complete ({epoch_time:.1f}s) ---')
        print(f'Average loss: {avg_loss:.4f}')

        # Evaluate
        print('Evaluating...')
        val_metrics = evaluate(tokenizer, loader, device)
        print(f"Val Loss: {val_metrics['val_loss']:.4f}, PSNR: {val_metrics['val_psnr']:.2f} dB")

        # Save checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'tokenizer_state_dict': tokenizer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr': val_metrics['val_psnr'],
            'image_size': args.image_size,
            'latent_channels': tokenizer.latent_channels,
            'use_perceptual': args.use_perceptual,
            'perceptual_weight': args.perceptual_weight if args.use_perceptual else 0.0,
            'perceptual_type': args.perceptual_type if args.use_perceptual else None,
        }

        torch.save(ckpt, save_dir / 'latest_tokenizer.pt')

        if val_metrics['val_psnr'] > best_psnr:
            best_psnr = val_metrics['val_psnr']
            torch.save(ckpt, save_dir / 'best_tokenizer.pt')
            print(f'  -> Saved (best PSNR: {best_psnr:.2f} dB)')

        print()

    total_time = time.time() - start_time
    print('=' * 70)
    print('TOKENIZER TRAINING COMPLETE')
    print('=' * 70)
    print(f'Total time: {total_time:.1f}s')
    print(f'Best PSNR: {best_psnr:.2f} dB')
    print(f'Checkpoints: {save_dir}')


if __name__ == '__main__':
    main()
