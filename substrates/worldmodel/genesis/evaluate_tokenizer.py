"""Evaluate tokenizer reconstruction quality using CLIP-IQA.

This script isolates tokenizer quality from dynamics quality.
Used to diagnose WHERE CLIP-IQA degradation occurs.

Usage:
    python genesis/evaluate_tokenizer.py \
        --checkpoint checkpoints/tokenizer_256/best_tokenizer.pt \
        --num-samples 256 --image-size 256
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from genesis.tokenizer.motion import MotionAwareTokenizer
from genesis.pilot.stream_hf import create_streaming_loader


@torch.no_grad()
def evaluate_tokenizer_quality(
    tokenizer: MotionAwareTokenizer,
    loader,
    device: str,
    num_samples: int = 256,
    save_samples: bool = False,
    output_dir: Path = None,
) -> dict:
    """Compute CLIP-IQA, MUSIQ, PSNR on tokenizer reconstructions."""
    tokenizer.eval()

    # Initialize metrics
    try:
        import pyiqa
        clipiqa = pyiqa.create_metric('clipiqa', device=device)
        musiq = pyiqa.create_metric('musiq', device=device)
        has_pyiqa = True
    except ImportError:
        print("Warning: pyiqa not available, using PSNR only")
        has_pyiqa = False

    scores = {'clipiqa': [], 'musiq': [], 'psnr': [], 'mse': []}
    samples_processed = 0

    print(f"Evaluating tokenizer on {num_samples} samples...")

    for batch in tqdm(loader, desc="Evaluating"):
        if samples_processed >= num_samples:
            break

        # Get video
        if 'frames' in batch:
            video = batch['frames'].to(device)
        else:
            video = batch['video'].to(device)

        B, T, C, H, W = video.shape

        # Encode and decode (reconstruction)
        latents, aux = tokenizer.encode(video)
        recon = tokenizer.decode(latents, (H, W))

        # Clamp reconstruction to valid range
        recon = recon.clamp(0, 1)

        # Sample middle frame for quality evaluation
        frame_idx = T // 2
        orig_frames = video[:, frame_idx]  # [B, C, H, W]
        recon_frames = recon[:, frame_idx]

        # Compute metrics per sample in batch
        for b in range(B):
            if samples_processed >= num_samples:
                break

            orig = orig_frames[b:b+1]
            rec = recon_frames[b:b+1]

            # PSNR (always available)
            mse = F.mse_loss(rec, orig)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))
            scores['psnr'].append(psnr.item())
            scores['mse'].append(mse.item())

            # Perceptual metrics (if pyiqa available)
            if has_pyiqa:
                try:
                    scores['clipiqa'].append(clipiqa(rec).item())
                    scores['musiq'].append(musiq(rec).item())
                except Exception as e:
                    print(f"Warning: pyiqa failed on sample {samples_processed}: {e}")

            samples_processed += 1

            # Optionally save sample images
            if save_samples and output_dir and samples_processed <= 16:
                from PIL import Image
                orig_img = (orig[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                recon_img = (rec[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                output_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(orig_img).save(output_dir / f"sample_{samples_processed:03d}_orig.png")
                Image.fromarray(recon_img).save(output_dir / f"sample_{samples_processed:03d}_recon.png")

    # Compute statistics
    results = {}
    for k, v in scores.items():
        if len(v) > 0:
            results[k] = {
                'mean': float(np.mean(v)),
                'std': float(np.std(v)),
                'n_samples': len(v),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Tokenizer Reconstruction Quality')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to tokenizer checkpoint')
    parser.add_argument('--num-samples', type=int, default=256,
                        help='Number of samples to evaluate')
    parser.add_argument('--image-size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--dataset', type=str, default='jat',
                        help='Dataset to use (jat, webvid, etc.)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output-dir', type=str, default='outputs/tokenizer_eval',
                        help='Output directory for results')
    parser.add_argument('--save-samples', action='store_true',
                        help='Save sample reconstructions')
    args = parser.parse_args()

    print("=" * 70)
    print("TOKENIZER QUALITY EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Resolution: {args.image_size}x{args.image_size}")
    print(f"Samples: {args.num_samples}")
    print(f"Dataset: {args.dataset}")
    print("=" * 70)

    # Load checkpoint
    device = args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create tokenizer with same config
    if 'config' in checkpoint:
        config = checkpoint['config']
        tokenizer = MotionAwareTokenizer(
            in_channels=3,
            hidden_channels=getattr(config, 'hidden_channels', 64),
            keyframe_interval=getattr(config, 'keyframe_interval', 4),
            image_size=args.image_size,
            adaptive_channels=True,
        )
    else:
        # Default config
        tokenizer = MotionAwareTokenizer(
            in_channels=3,
            hidden_channels=64,
            keyframe_interval=4,
            image_size=args.image_size,
            adaptive_channels=True,
        )

    # Load weights
    if 'tokenizer_state_dict' in checkpoint:
        tokenizer.load_state_dict(checkpoint['tokenizer_state_dict'])
    elif 'model_state_dict' in checkpoint:
        # Full model checkpoint - extract tokenizer weights
        state_dict = checkpoint['model_state_dict']
        tok_state = {k.replace('tokenizer.', ''): v for k, v in state_dict.items() if k.startswith('tokenizer.')}
        tokenizer.load_state_dict(tok_state)
    else:
        tokenizer.load_state_dict(checkpoint)

    tokenizer = tokenizer.to(device)
    tokenizer.eval()

    print(f"\nTokenizer loaded:")
    print(f"  Latent size: {tokenizer.latent_size}x{tokenizer.latent_size}")
    print(f"  Latent channels: {tokenizer.latent_channels}")
    print(f"  Parameters: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Create data loader
    print(f"\nCreating data loader ({args.dataset})...")
    loader = create_streaming_loader(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        seq_length=16,
        image_size=args.image_size,
    )

    # Evaluate
    output_dir = Path(args.output_dir)
    results = evaluate_tokenizer_quality(
        tokenizer=tokenizer,
        loader=loader,
        device=device,
        num_samples=args.num_samples,
        save_samples=args.save_samples,
        output_dir=output_dir if args.save_samples else None,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for metric, stats in results.items():
        print(f"{metric.upper():12s}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['n_samples']})")

    # Diagnostic decision
    print("\n" + "=" * 70)
    print("DIAGNOSTIC DECISION")
    print("=" * 70)

    if 'clipiqa' in results:
        clipiqa_mean = results['clipiqa']['mean']
        if clipiqa_mean >= 0.40:
            print(f"CLIP-IQA = {clipiqa_mean:.3f} >= 0.40")
            print("-> Tokenizer is GOOD. Problem is likely in DYNAMICS.")
            print("-> Proceed with Phase 2B: Add perceptual loss to dynamics training")
        else:
            print(f"CLIP-IQA = {clipiqa_mean:.3f} < 0.40")
            print("-> Tokenizer is the BOTTLENECK.")
            print("-> Proceed with Phase 2A: Add perceptual loss to tokenizer training")
    else:
        print("CLIP-IQA not available (pyiqa not installed)")
        print(f"PSNR = {results['psnr']['mean']:.2f} dB")
        print("-> Install pyiqa for perceptual quality metrics")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / 'tokenizer_quality.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == '__main__':
    main()
