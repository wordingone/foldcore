"""Validate Genesis world model with interactive playback."""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description='Validate Genesis World Model')
    parser.add_argument('--checkpoint', default='checkpoints/genesis/best_genesis.pt')
    parser.add_argument('--dataset', default='jat')
    parser.add_argument('--game', default='atari-breakout')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--context-frames', type=int, default=4)
    parser.add_argument('--generate-frames', type=int, default=12)
    parser.add_argument('--output-dir', default='outputs/genesis_validation')
    args = parser.parse_args()

    print('=' * 70)
    print('GENESIS WORLD MODEL VALIDATION')
    print('=' * 70)

    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f'\nLoading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt['config']
    print(f'Config: {config}')

    # Create model
    from genesis.genesis_model import Genesis
    model = Genesis(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f'\nModel loaded: {model.count_parameters():,} parameters')

    # Load validation data
    print('\nLoading validation data...')
    from genesis.pilot.stream_hf import create_streaming_loader
    loader = create_streaming_loader(
        dataset_name=args.dataset,
        batch_size=1,
        seq_length=args.context_frames + args.generate_frames,
        image_size=config.image_size,
        game=args.game if args.dataset == 'jat' else None,
        shuffle=False,
    )

    # Validation metrics
    all_psnr = []
    all_ssim = []

    print('\n' + '=' * 70)
    print('GENERATING PREDICTIONS')
    print('=' * 70)

    for sample_idx, batch in enumerate(loader):
        if sample_idx >= args.num_samples:
            break

        print(f'\nSample {sample_idx + 1}/{args.num_samples}')

        # Get video
        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        # Get actions
        actions = batch.get('actions')
        if actions is not None:
            actions = actions.to(device)

        B, T, C, H, W = video.shape
        context = video[:, :args.context_frames]
        target = video[:, args.context_frames:]

        # Encode actions for generation
        if actions is not None:
            gen_actions = actions[:, args.context_frames - 1:]
            if gen_actions.shape[1] > args.generate_frames:
                gen_actions = gen_actions[:, :args.generate_frames]
        else:
            gen_actions = torch.zeros(B, args.generate_frames, config.action_dim, device=device)

        # Generate
        model.reset_state()
        with torch.no_grad():
            generated = model.generate(
                context,
                gen_actions,
                num_frames=args.generate_frames,
            )

        # Extract generated frames (skip context)
        pred_frames = generated[:, args.context_frames:]
        target_frames = target[:, :pred_frames.shape[1]]

        # Compute metrics
        mse = F.mse_loss(pred_frames, target_frames)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
        all_psnr.append(psnr.item())

        print(f'  Context: {context.shape[1]} frames')
        print(f'  Generated: {pred_frames.shape[1]} frames')
        print(f'  PSNR: {psnr.item():.2f} dB')
        print(f'  MSE: {mse.item():.6f}')

        # Save sample visualization
        save_visualization(
            context[0].cpu(),
            pred_frames[0].cpu(),
            target_frames[0].cpu(),
            output_dir / f'sample_{sample_idx:02d}.png',
        )

    # Summary
    print('\n' + '=' * 70)
    print('VALIDATION SUMMARY')
    print('=' * 70)
    print(f'Samples evaluated: {len(all_psnr)}')
    print(f'Mean PSNR: {np.mean(all_psnr):.2f} dB')
    print(f'Std PSNR: {np.std(all_psnr):.2f} dB')
    print(f'Min PSNR: {np.min(all_psnr):.2f} dB')
    print(f'Max PSNR: {np.max(all_psnr):.2f} dB')
    print(f'\nVisualizations saved to: {output_dir}')

    # Test real-time generation capability
    print('\n' + '=' * 70)
    print('REAL-TIME GENERATION TEST')
    print('=' * 70)

    import time

    # Warmup
    dummy_context = torch.randn(1, 4, 3, 64, 64, device=device)
    dummy_actions = torch.zeros(1, 1, config.action_dim, device=device)

    for _ in range(3):
        model.reset_state()
        with torch.no_grad():
            _ = model.generate(dummy_context, dummy_actions, num_frames=1)

    # Benchmark
    num_trials = 10
    times = []

    for _ in range(num_trials):
        model.reset_state()
        torch.cuda.synchronize() if device == 'cuda' else None
        start = time.time()

        with torch.no_grad():
            _ = model.generate(dummy_context, dummy_actions, num_frames=1)

        torch.cuda.synchronize() if device == 'cuda' else None
        times.append(time.time() - start)

    avg_time = np.mean(times)
    fps = 1.0 / avg_time

    print(f'Single frame generation: {avg_time * 1000:.1f} ms')
    print(f'Effective FPS: {fps:.1f}')
    print(f'Target: 24 FPS -> {"PASS" if fps >= 24 else "FAIL"} (need {1000/24:.1f} ms/frame)')

    return all_psnr


def save_visualization(context, pred, target, path):
    """Save visualization comparing context, prediction, and target."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    num_context = context.shape[0]
    num_pred = pred.shape[0]

    # Select frames to show
    show_context = min(4, num_context)
    show_pred = min(6, num_pred)

    fig, axes = plt.subplots(3, max(show_context, show_pred), figsize=(2 * max(show_context, show_pred), 6))

    # Context row
    for i in range(show_context):
        ax = axes[0, i]
        frame = context[i].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        ax.set_title(f'Context {i+1}')
        ax.axis('off')

    for i in range(show_context, axes.shape[1]):
        axes[0, i].axis('off')

    # Prediction row
    for i in range(show_pred):
        ax = axes[1, i]
        frame = pred[i].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        ax.set_title(f'Pred {i+1}')
        ax.axis('off')

    for i in range(show_pred, axes.shape[1]):
        axes[1, i].axis('off')

    # Target row
    for i in range(min(show_pred, target.shape[0])):
        ax = axes[2, i]
        frame = target[i].permute(1, 2, 0).numpy()
        frame = np.clip(frame, 0, 1)
        ax.imshow(frame)
        ax.set_title(f'Target {i+1}')
        ax.axis('off')

    for i in range(min(show_pred, target.shape[0]), axes.shape[1]):
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    main()
