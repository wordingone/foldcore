"""Train video tokenizer with reconstruction + perceptual + GAN losses."""

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

from genesis.config import TokenizerConfig
from genesis.tokenizer.decoder import VideoTokenizer
from genesis.tokenizer.discriminator import (
    MultiScaleDiscriminator,
    multiscale_hinge_loss_d,
    multiscale_hinge_loss_g,
)
from genesis.tokenizer.losses import TokenizerLoss, TemporalConsistencyLoss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_step_generator(
    tokenizer: VideoTokenizer,
    discriminator: MultiScaleDiscriminator,
    batch: dict,
    optimizer_g: torch.optim.Optimizer,
    loss_fn: TokenizerLoss,
    temporal_loss: TemporalConsistencyLoss,
    device: str,
    gan_start_step: int,
    current_step: int,
) -> dict:
    """Single generator training step."""
    tokenizer.train()
    discriminator.eval()

    # Get video batch
    if 'frames' in batch:
        video = batch['frames'].to(device)  # [B, T, C, H, W]
    else:
        video = batch['video'].to(device)

    # Forward pass
    out = tokenizer(video)
    recon = out['recon']
    latent = out['latent']
    codes = out['codes']

    # GAN loss (after warmup)
    gan_loss = None
    if current_step >= gan_start_step:
        with torch.no_grad():
            fake_outputs = discriminator(recon)
        gan_loss = multiscale_hinge_loss_g(fake_outputs)

    # Compute losses
    losses = loss_fn(recon, video, latent, codes, gan_loss)

    # Temporal consistency
    temp_loss = temporal_loss(recon, video)
    losses['temporal'] = temp_loss
    losses['total'] = losses['total'] + temp_loss

    # Backward
    optimizer_g.zero_grad()
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), 1.0)
    optimizer_g.step()

    return {k: v.item() for k, v in losses.items()}


def train_step_discriminator(
    tokenizer: VideoTokenizer,
    discriminator: MultiScaleDiscriminator,
    batch: dict,
    optimizer_d: torch.optim.Optimizer,
    device: str,
) -> dict:
    """Single discriminator training step."""
    tokenizer.eval()
    discriminator.train()

    # Get video batch
    if 'frames' in batch:
        video = batch['frames'].to(device)
    else:
        video = batch['video'].to(device)

    # Generate fake videos
    with torch.no_grad():
        out = tokenizer(video)
        recon = out['recon']

    # Discriminator outputs
    real_outputs = discriminator(video)
    fake_outputs = discriminator(recon.detach())

    # Hinge loss
    d_loss = multiscale_hinge_loss_d(real_outputs, fake_outputs)

    # Backward
    optimizer_d.zero_grad()
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
    optimizer_d.step()

    return {'d_loss': d_loss.item()}


@torch.no_grad()
def evaluate(
    tokenizer: VideoTokenizer,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> dict:
    """Evaluate tokenizer on validation data."""
    tokenizer.eval()

    total_mse = 0
    total_psnr = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        out = tokenizer(video)
        recon = out['recon']

        mse = F.mse_loss(recon, video)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        total_mse += mse.item()
        total_psnr += psnr.item()
        num_batches += 1

    if num_batches == 0:
        return {'mse': 0, 'psnr': 0}

    return {
        'mse': total_mse / num_batches,
        'psnr': total_psnr / num_batches,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def create_dataloader(
    dataset_name: str = 'jat',
    batch_size: int = 4,
    seq_length: int = 16,
    image_size: int = 256,
    **kwargs,
) -> DataLoader:
    """Create data loader for tokenizer training."""
    from genesis.pilot.stream_hf import create_streaming_loader

    return create_streaming_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        seq_length=seq_length,
        image_size=image_size,
        **kwargs,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Video Tokenizer')
    parser.add_argument('--dataset', default='jat', help='Dataset to use')
    parser.add_argument('--game', default='atari-breakout', help='Game for JAT')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples-per-epoch', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr-g', type=float, default=1e-4, help='Generator LR')
    parser.add_argument('--lr-d', type=float, default=4e-4, help='Discriminator LR')
    parser.add_argument('--seq-length', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=64, help='Training resolution')
    parser.add_argument('--target-size', type=int, default=256, help='Target resolution')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints/tokenizer')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--gan-start-epoch', type=int, default=2, help='Start GAN after N epochs')
    parser.add_argument('--d-steps', type=int, default=1, help='D steps per G step')
    parser.add_argument('--perceptual-weight', type=float, default=0.1)
    parser.add_argument('--gan-weight', type=float, default=0.1)
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    print('=' * 70)
    print('VIDEO TOKENIZER TRAINING')
    print('=' * 70)
    print(f'Dataset: {args.dataset}')
    print(f'Resolution: {args.image_size} (training) -> {args.target_size} (target)')
    print(f'Device: {args.device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Sequence length: {args.seq_length}')
    print('=' * 70)

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Create tokenizer
    print('\nCreating tokenizer...')

    # Adjust config for training resolution
    config = TokenizerConfig()

    # For smaller training resolution, reduce spatial downsample
    if args.image_size <= 64:
        config.spatial_downsample = 8  # 64 -> 8x8
        config.encoder_channels = [32, 64, 128, 256]
        config.decoder_channels = [256, 128, 64, 32]
        config.encoder_depths = [2, 2, 2, 2]
        config.decoder_depths = [2, 2, 2, 2]
    elif args.image_size <= 128:
        config.spatial_downsample = 8  # 128 -> 16x16
        config.encoder_channels = [48, 96, 192, 384]
        config.decoder_channels = [384, 192, 96, 48]

    tokenizer = VideoTokenizer(config).to(device)
    print(f'Tokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}')

    # Create discriminator
    print('Creating discriminator...')
    discriminator = MultiScaleDiscriminator(
        in_channels=3,
        num_scales=2,
        channels=[32, 64, 128, 256],
    ).to(device)
    print(f'Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}')

    # Loss functions
    loss_fn = TokenizerLoss(
        recon_weight=1.0,
        perceptual_weight=args.perceptual_weight,
        gan_weight=args.gan_weight,
        commitment_weight=0.25,
        use_lpips=False,  # VGG is sufficient and doesn't need extra install
    )
    temporal_loss = TemporalConsistencyLoss(weight=0.1)

    # Move loss modules to device
    loss_fn.perceptual = loss_fn.perceptual.to(device)

    # Optimizers
    optimizer_g = torch.optim.AdamW(tokenizer.parameters(), lr=args.lr_g, weight_decay=0.01)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d, weight_decay=0.01)

    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g, T_max=args.epochs * args.samples_per_epoch // args.batch_size
    )
    scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d, T_max=args.epochs * args.samples_per_epoch // args.batch_size
    )

    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        print(f'\nResuming from {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        tokenizer.load_state_dict(ckpt['tokenizer_state_dict'])
        if 'discriminator_state_dict' in ckpt:
            discriminator.load_state_dict(ckpt['discriminator_state_dict'])
        if 'optimizer_g_state_dict' in ckpt:
            optimizer_g.load_state_dict(ckpt['optimizer_g_state_dict'])
        if 'optimizer_d_state_dict' in ckpt:
            optimizer_d.load_state_dict(ckpt['optimizer_d_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_psnr = ckpt.get('psnr', 0)
        print(f'Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.2f}')

    # Create data loader
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

    gan_start_step = args.gan_start_epoch * args.samples_per_epoch // args.batch_size
    current_step = start_epoch * args.samples_per_epoch // args.batch_size
    epoch_losses = []
    start_time = time.time()
    samples_processed = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            # Check if we've processed enough samples for this epoch
            if samples_processed >= args.samples_per_epoch:
                break

            # Train discriminator
            if current_step >= gan_start_step:
                for _ in range(args.d_steps):
                    d_losses = train_step_discriminator(
                        tokenizer, discriminator, batch, optimizer_d, device
                    )
                scheduler_d.step()

            # Train generator
            g_losses = train_step_generator(
                tokenizer, discriminator, batch, optimizer_g,
                loss_fn, temporal_loss, device,
                gan_start_step, current_step,
            )
            scheduler_g.step()

            epoch_losses.append(g_losses)
            current_step += 1
            samples_processed += args.batch_size

            # Logging
            if batch_idx % args.log_every == 0:
                avg_total = sum(l['total'] for l in epoch_losses[-100:]) / max(len(epoch_losses[-100:]), 1)
                avg_recon = sum(l['recon'] for l in epoch_losses[-100:]) / max(len(epoch_losses[-100:]), 1)
                avg_perc = sum(l['perceptual'] for l in epoch_losses[-100:]) / max(len(epoch_losses[-100:]), 1)

                status = f'Epoch {epoch+1}/{args.epochs} | Batch {batch_idx}'
                status += f' | total={avg_total:.4f} recon={avg_recon:.4f} perc={avg_perc:.4f}'

                if current_step >= gan_start_step:
                    avg_gan = sum(l['gan'] for l in epoch_losses[-100:]) / max(len(epoch_losses[-100:]), 1)
                    status += f' gan={avg_gan:.4f}'

                elapsed = time.time() - start_time
                samples_per_sec = (current_step * args.batch_size) / elapsed
                status += f' | {samples_per_sec:.1f} samples/s'

                print(status)

        # End of epoch
        samples_processed = 0
        epoch_time = time.time() - epoch_start

        avg_loss = sum(l['total'] for l in epoch_losses) / max(len(epoch_losses), 1)
        print(f'\n--- Epoch {epoch+1} Complete ({epoch_time:.1f}s) ---')
        print(f'Average loss: {avg_loss:.4f}')

        # Evaluate
        print('Evaluating...')
        eval_loader = create_dataloader(
            dataset_name=args.dataset,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            image_size=args.image_size,
            game=args.game if args.dataset == 'jat' else None,
            shuffle=False,
        )
        metrics = evaluate(tokenizer, eval_loader, device, max_batches=30)
        print(f'Val MSE: {metrics["mse"]:.6f}, PSNR: {metrics["psnr"]:.2f} dB')

        # Save checkpoint
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            torch.save({
                'epoch': epoch + 1,
                'tokenizer_state_dict': tokenizer.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'psnr': best_psnr,
                'config': config,
            }, save_dir / 'best_tokenizer.pt')
            print(f'  -> Saved (best PSNR: {best_psnr:.2f} dB)')

        # Save latest checkpoint
        torch.save({
            'epoch': epoch + 1,
            'tokenizer_state_dict': tokenizer.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_g_state_dict': optimizer_g.state_dict(),
            'optimizer_d_state_dict': optimizer_d.state_dict(),
            'psnr': metrics['psnr'],
            'config': config,
        }, save_dir / 'latest_tokenizer.pt')

        epoch_losses = []

    # Final summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    total_time = time.time() - start_time
    print(f'Total time: {total_time:.1f}s ({total_time/3600:.2f} hours)')
    print(f'Best PSNR: {best_psnr:.2f} dB')
    print(f'Checkpoints: {save_dir}')


if __name__ == '__main__':
    main()
