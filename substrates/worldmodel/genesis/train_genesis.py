"""Train Genesis world model end-to-end."""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import time
from typing import Optional

from genesis.genesis_model import Genesis, GenesisConfig, create_genesis


def train_step(
    model: Genesis,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    device: str,
    perceptual_loss_fn=None,
    perceptual_weight: float = 0.1,
) -> dict:
    """Single training step."""
    model.train()

    # Get video and actions
    if 'frames' in batch:
        video = batch['frames'].to(device)
    else:
        video = batch['video'].to(device)

    actions = batch.get('actions')
    if actions is not None:
        actions = actions.to(device)
        # Ensure actions are one less than frames (for next-frame prediction)
        if actions.shape[1] >= video.shape[1]:
            actions = actions[:, :video.shape[1] - 1]

    # Forward pass
    out = model(video, actions)

    # Total loss from model
    loss = out['total_loss']

    # Add perceptual loss on decoded predictions (if enabled)
    perceptual_loss = 0.0
    if perceptual_loss_fn is not None and 'pred_video' in out and 'target_video' in out:
        perceptual_loss = perceptual_loss_fn(out['pred_video'], out['target_video'])
        loss = loss + perceptual_weight * perceptual_loss
        perceptual_loss = perceptual_loss.item()

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'total': loss.item(),
        'latent': out['latent_loss'].item(),
        'recon': out['recon_loss'].item(),
        'perceptual': perceptual_loss,
    }


@torch.no_grad()
def evaluate(
    model: Genesis,
    loader: DataLoader,
    device: str,
    max_batches: int = 30,
) -> dict:
    """Evaluate model."""
    model.eval()

    total_loss = 0
    total_psnr = 0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        actions = batch.get('actions')
        if actions is not None:
            actions = actions.to(device)
            if actions.shape[1] >= video.shape[1]:
                actions = actions[:, :video.shape[1] - 1]

        out = model(video, actions)

        # Compute PSNR
        mse = F.mse_loss(out['pred_video'], out['target_video'])
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        total_loss += out['total_loss'].item()
        total_psnr += psnr.item()
        num_batches += 1

    if num_batches == 0:
        return {'loss': 0, 'psnr': 0}

    return {
        'loss': total_loss / num_batches,
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
    parser = argparse.ArgumentParser(description='Train Genesis World Model')
    parser.add_argument('--dataset', default='jat')
    parser.add_argument('--game', default='atari-breakout')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--samples-per-epoch', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-length', type=int, default=16)
    parser.add_argument('--image-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-dir', default='checkpoints/genesis')
    parser.add_argument('--log-every', type=int, default=25)
    # Model config
    parser.add_argument('--model-size', default='small', choices=['tiny', 'small', 'medium', 'large'])
    parser.add_argument('--use-slots', action='store_true', default=True)
    parser.add_argument('--no-slots', dest='use_slots', action='store_false')
    parser.add_argument('--slot-norm-mode', default='decay', choices=['decay', 'layernorm', 'clip', 'none'],
                        help='Slot normalization mode for TRAINING (default: decay). '
                             'IMPORTANT: Use decay during training to learn diversity. '
                             'At inference, override with --slot-norm-mode clip for stability.')
    parser.add_argument('--tokenizer', default='motion', choices=['motion', 'vae'])
    parser.add_argument('--resume', type=str, default=None)
    # Split training options
    parser.add_argument('--freeze-tokenizer', action='store_true',
                        help='Freeze tokenizer weights (Phase 2 of split training)')
    parser.add_argument('--load-tokenizer', type=str, default=None,
                        help='Load pre-trained tokenizer from checkpoint (Phase 2)')
    # Perceptual loss options
    parser.add_argument('--use-perceptual', action='store_true',
                        help='Use perceptual loss (LPIPS/VGG) for better CLIP-IQA')
    parser.add_argument('--perceptual-weight', type=float, default=0.1,
                        help='Weight for perceptual loss (default: 0.1)')
    parser.add_argument('--perceptual-type', type=str, default='vgg',
                        choices=['lpips', 'vgg'],
                        help='Type of perceptual loss (default: vgg)')
    args = parser.parse_args()

    print('=' * 70)
    print('GENESIS WORLD MODEL TRAINING')
    print('=' * 70)
    print(f'Dataset: {args.dataset}')
    print(f'Model size: {args.model_size}')
    print(f'Use slots: {args.use_slots}')
    print(f'Slot norm mode: {args.slot_norm_mode}')
    print(f'Tokenizer: {args.tokenizer}')
    print(f'Resolution: {args.image_size}')
    print(f'Perceptual loss: {args.use_perceptual} ({args.perceptual_type}, weight={args.perceptual_weight})')
    print(f'Device: {args.device}')
    print('=' * 70)

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Create model
    print('\nCreating Genesis model...')
    model = create_genesis(
        size=args.model_size,
        image_size=args.image_size,
        use_slots=args.use_slots,
        slot_norm_mode=args.slot_norm_mode,
    )
    # Override tokenizer type
    model.config.tokenizer_type = args.tokenizer

    model = model.to(device)

    breakdown = model.get_parameter_breakdown()
    print(f'Parameters:')
    for k, v in breakdown.items():
        print(f'  {k}: {v:,}')

    # Split training: Load pre-trained tokenizer (Phase 2)
    if args.load_tokenizer:
        print(f'\nLoading pre-trained tokenizer from {args.load_tokenizer}')
        tok_ckpt = torch.load(args.load_tokenizer, map_location=device, weights_only=False)
        model.tokenizer.load_state_dict(tok_ckpt['tokenizer_state_dict'])
        print(f'  Loaded tokenizer (PSNR: {tok_ckpt.get("psnr", "N/A"):.2f} dB)')

    # Split training: Freeze tokenizer (Phase 2)
    if args.freeze_tokenizer:
        print('\nFreezing tokenizer weights (Phase 2 split training)')
        for param in model.tokenizer.parameters():
            param.requires_grad = False
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'  Trainable parameters: {trainable:,} (dynamics only)')

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

    # Optimizer (only train non-frozen params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * args.samples_per_epoch // args.batch_size
    )

    # Resume
    start_epoch = 0
    best_psnr = 0
    if args.resume:
        print(f'\nResuming from {args.resume}')
        # Use weights_only=False for PyTorch 2.6+ compatibility (trusted checkpoint)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
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

    epoch_losses = []
    start_time = time.time()
    batch_idx = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        samples_processed = 0

        for batch in loader:
            if samples_processed >= args.samples_per_epoch:
                break

            # Train step
            losses = train_step(
                model, batch, optimizer, device,
                perceptual_loss_fn=perceptual_loss_fn,
                perceptual_weight=args.perceptual_weight,
            )
            scheduler.step()

            epoch_losses.append(losses)
            batch_idx += 1
            samples_processed += args.batch_size

            # Logging
            if batch_idx % args.log_every == 0:
                avg_total = sum(l['total'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                avg_latent = sum(l['latent'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                avg_recon = sum(l['recon'] for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)

                elapsed = time.time() - start_time
                samples_per_sec = (batch_idx * args.batch_size) / elapsed

                perc_str = ""
                if args.use_perceptual:
                    avg_perc = sum(l.get('perceptual', 0) for l in epoch_losses[-50:]) / max(len(epoch_losses[-50:]), 1)
                    perc_str = f" perc={avg_perc:.4f}"

                print(f'Epoch {epoch+1}/{args.epochs} | Batch {batch_idx} | '
                      f'total={avg_total:.4f} latent={avg_latent:.4f} recon={avg_recon:.4f}{perc_str} | '
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
        print(f'Val Loss: {metrics["loss"]:.4f}, PSNR: {metrics["psnr"]:.2f} dB')

        # Save checkpoint
        if metrics['psnr'] > best_psnr:
            best_psnr = metrics['psnr']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'psnr': best_psnr,
                'config': model.config,
            }, save_dir / 'best_genesis.pt')
            print(f'  -> Saved (best PSNR: {best_psnr:.2f} dB)')

        # Save latest
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'psnr': metrics['psnr'],
            'config': model.config,
        }, save_dir / 'latest_genesis.pt')

        epoch_losses = []

    # Final summary
    print('\n' + '=' * 70)
    print('TRAINING COMPLETE')
    print('=' * 70)
    print(f'Total time: {time.time() - start_time:.1f}s')
    print(f'Best PSNR: {best_psnr:.2f} dB')
    print(f'Checkpoints: {save_dir}')


if __name__ == '__main__':
    main()
