"""Train motion-aware video tokenizer."""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import time

from genesis.tokenizer.motion import MotionAwareTokenizer, MotionAwareLoss
from genesis.tokenizer.losses import VGGPerceptualLoss


def train_step(
    model: MotionAwareTokenizer,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    loss_fn: MotionAwareLoss,
    perceptual_loss: VGGPerceptualLoss,
    device: str,
    perceptual_weight: float = 0.1,
) -> dict:
    """Single training step."""
    model.train()

    # Get video
    if 'frames' in batch:
        video = batch['frames'].to(device)  # [B, T, C, H, W]
    else:
        video = batch['video'].to(device)

    # Forward
    out = model(video)
    recon = out['recon']
    intermediates = out['intermediates']

    # Compute losses
    losses = loss_fn(recon, video, intermediates)

    # Perceptual loss
    perc_loss = perceptual_loss(recon, video)
    losses['perceptual'] = perc_loss
    losses['total'] = losses['total'] + perceptual_weight * perc_loss

    # Backward
    optimizer.zero_grad()
    losses['total'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


@torch.no_grad()
def evaluate(
    model: MotionAwareTokenizer,
    loader: DataLoader,
    device: str,
    max_batches: int = 50,
) -> dict:
    """Evaluate model."""
    model.eval()

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

        out = model(video)
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


def create_dataloader(
    dataset_name: str = 'jat',
    batch_size: int = 4,
    seq_length: int = 16,
    image_size: int = 64,
    **kwargs,
) -> DataLoader:
    """Create data loader."""
    from genesis.pilot.stream_hf import create_streaming_loader
    return create_streaming_loader(
        dataset_name=dataset_name,
        batch_size=batch_size,
        seq_length=seq_length,
        image_size=image_size,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description='Train Motion-Aware Tokenizer')
    parser.add_argument('--dataset', default='jat')
    parser.add_argument('--game', default='atari-breakout')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples-per-epoch', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-length', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints/motion_tokenizer')
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--keyframe-interval', type=int, default=8)
    args = parser.parse_args()

    print('=' * 70)
    print('MOTION-AWARE TOKENIZER TRAINING')
    print('=' * 70)
    print(f'Dataset: {args.dataset}')
    print(f'Resolution: {args.image_size}')
    print(f'Keyframe interval: {args.keyframe_interval}')
    print(f'Device: {args.device}')
    print('=' * 70)

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Create model
    print('\nCreating motion-aware tokenizer...')
    model = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=args.keyframe_interval,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {params:,}')

    # Test compression ratio
    test_shape = (1, args.seq_length, 3, args.image_size, args.image_size)
    compression = model.get_compression_ratio(test_shape)
    print(f'Compression ratio: {compression:.1f}x')

    # Loss functions
    loss_fn = MotionAwareLoss(
        recon_weight=1.0,
        flow_smooth_weight=0.1,
        residual_sparse_weight=0.05,
    )
    perceptual_loss = VGGPerceptualLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.samples_per_epoch // args.batch_size
    )

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

    best_psnr = 0
    epoch_losses = []
    start_time = time.time()
    samples_processed = 0
    batch_idx = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        samples_processed = 0

        for batch in loader:
            if samples_processed >= args.samples_per_epoch:
                break

            # Train step
            losses = train_step(
                model, batch, optimizer, loss_fn, perceptual_loss, device
            )
            scheduler.step()

            epoch_losses.append(losses)
            batch_idx += 1
            samples_processed += args.batch_size

            # Logging
            if batch_idx % args.log_every == 0:
                avg_total = sum(l['total'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                avg_recon = sum(l['recon'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                avg_flow = sum(l['flow_smooth'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                avg_res = sum(l['residual_sparse'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)

                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx * args.batch_size) / elapsed

                print(f'Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | '
                      f'total={avg_total:.4f} recon={avg_recon:.4f} '
                      f'flow={avg_flow:.4f} res={avg_res:.4f} | '
                      f'{samples_per_sec:.1f} samples/s')

        # End of epoch
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
        metrics = evaluate(model, eval_loader, device, max_batches=30)
        print(f'Val MSE: {metrics["mse"]:.6f}, PSNR: {metrics["psnr"]:.2f} dB')

        # Save checkpoint
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'compression_ratio': compression,
            }, save_dir / 'best_motion_tokenizer.pt')
            print(f'  -> Saved (best PSNR: {best_psnr:.2f} dB)')

        epoch_losses = []

    # Final summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'Total time: {time.time() - start_time:.1f}s')
    print(f'Best PSNR: {best_psnr:.2f} dB')
    print(f'Compression: {compression:.1f}x')
    print(f'Checkpoints: {save_dir}')


if __name__ == '__main__':
    main()
