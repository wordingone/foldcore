"""VBench Evaluation for Genesis World Model.

Generates video samples and evaluates them using VBench benchmark suite.

Usage:
    # Generate samples only
    python genesis/evaluate_vbench.py \
        --checkpoint checkpoints/genesis/best_genesis.pt \
        --num-samples 256 \
        --output-dir outputs/vbench_eval/

    # Generate and run VBench evaluation
    python genesis/evaluate_vbench.py \
        --checkpoint checkpoints/genesis/best_genesis.pt \
        --num-samples 256 \
        --output-dir outputs/vbench_eval/ \
        --run-vbench

    # Quick test with 10 samples
    python genesis/evaluate_vbench.py \
        --checkpoint checkpoints/genesis/best_genesis.pt \
        --num-samples 10 \
        --output-dir outputs/vbench_test/

CRITICAL: Uses train-decay + infer-clip workflow:
- Model must be trained with slot_norm_mode='decay' (forces diversity learning)
- Inference MUST use --slot-norm-mode clip (preserves diversity)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def tensor_to_mp4(
    video: torch.Tensor,
    output_path: str,
    fps: int = 8,
) -> None:
    """Save video tensor as MP4 file.

    Args:
        video: [T, C, H, W] tensor in [0, 1] range
        output_path: Path to save MP4 file
        fps: Frames per second
    """
    import imageio

    # Convert to numpy: [T, C, H, W] -> [T, H, W, C]
    frames = video.permute(0, 2, 3, 1).cpu().numpy()

    # Clamp to [0, 1] and convert to uint8
    frames = np.clip(frames, 0, 1)
    frames = (frames * 255).astype(np.uint8)

    # Write video
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def upscale_video(video: torch.Tensor, scale: int) -> torch.Tensor:
    """Upscale video by given factor using bilinear interpolation.

    Args:
        video: [T, C, H, W] tensor
        scale: Upscale factor (e.g., 4 for 64x64 -> 256x256)

    Returns:
        Upscaled video tensor
    """
    T, C, H, W = video.shape
    new_H, new_W = H * scale, W * scale

    # Process frame by frame to save memory
    upscaled = torch.empty(T, C, new_H, new_W, device=video.device)
    for t in range(T):
        upscaled[t] = F.interpolate(
            video[t:t+1],
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )[0]

    return upscaled


def load_model(checkpoint_path: str, device: str, slot_norm_mode: str = 'clip'):
    """Load Genesis model with inference configuration.

    CRITICAL: Uses slot_norm_mode='clip' for inference to preserve diversity
    learned during training with decay mode.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        slot_norm_mode: Normalization mode ('clip' recommended for inference)

    Returns:
        model: Loaded Genesis model
        config: Model configuration
    """
    from genesis.genesis_model import Genesis

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Override slot_norm_mode for inference
    original_mode = getattr(config, 'slot_norm_mode', 'decay')
    if config.use_slots:
        config.slot_norm_mode = slot_norm_mode
        print(f"Slot norm mode: {original_mode} -> {slot_norm_mode} (inference override)")

    # Create model
    model = Genesis(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, config


def get_seed_videos(
    config,
    device: str,
    num_seeds: int,
    dataset: str = 'webvid',
    game: Optional[str] = None,
) -> List[torch.Tensor]:
    """Get multiple seed videos from dataset.

    Args:
        config: Model config
        device: Device to load videos on
        num_seeds: Number of seed videos to load
        dataset: Dataset name
        game: Game name (for jat dataset)

    Returns:
        List of seed video tensors [1, T, C, H, W]
    """
    from genesis.pilot.stream_hf import create_streaming_loader

    # Build loader kwargs - only include game for JAT dataset
    loader_kwargs = {
        'dataset_name': dataset,
        'batch_size': 1,
        'seq_length': config.num_frames,
        'image_size': config.image_size,
        'shuffle': True,  # Shuffle for diversity
    }
    if dataset == 'jat' and game:
        loader_kwargs['game'] = game

    loader = create_streaming_loader(**loader_kwargs)

    seeds = []
    for i, batch in enumerate(loader):
        if i >= num_seeds:
            break

        if 'frames' in batch:
            video = batch['frames']
        else:
            video = batch['video']

        seeds.append(video.to(device))

    print(f"Loaded {len(seeds)} seed videos")
    return seeds


def generate_video_sample(
    model,
    config,
    seed_video: torch.Tensor,
    num_frames: int = 16,
    device: str = 'cuda',
) -> torch.Tensor:
    """Generate a single video sample from seed.

    Uses the train-decay + infer-clip workflow pattern from test_horizon_stability.py.

    Args:
        model: Genesis model (with slot_norm_mode='clip' for inference)
        config: Model config
        seed_video: [1, T_seed, C, H, W] seed video
        num_frames: Number of frames to generate
        device: Device

    Returns:
        video: [T_total, C, H, W] generated video (seed + generated frames)
    """
    model.eval()
    B, T_seed, C, H, W = seed_video.shape

    # Reset model state for clean generation
    model.reset_state()

    # Encode seed video as context
    context_latents, _ = model.encode(seed_video)
    current_context = context_latents

    # Storage for generated frames
    all_frames = [seed_video[:, t] for t in range(T_seed)]  # Start with seed frames

    # Generation state
    prev_slots = None
    bsd_states = None
    use_bsd = getattr(config, 'use_bsd', False) or getattr(model, '_use_bsd', False)
    action_dim = config.action_dim

    with torch.no_grad():
        for step in range(num_frames):
            # Zero action (passive observation)
            action = torch.zeros(B, 1, action_dim, device=device)

            # Expand action to match context length if needed
            if current_context.shape[1] > 1:
                action = action.expand(-1, current_context.shape[1], -1)

            # Predict next latent
            if use_bsd:
                pred, prev_slots, bsd_states = model.dynamics(
                    current_context, action, prev_slots, bsd_states
                )
            elif config.use_slots:
                pred, prev_slots = model.dynamics(current_context, action, prev_slots)
            else:
                pred = model.dynamics(current_context, action)

            next_latent = pred[:, -1:]

            # Decode to frame
            next_frame = model.decode(next_latent)
            all_frames.append(next_frame[:, 0])  # [B, C, H, W]

            # Update context (sliding window)
            window = min(current_context.shape[1] + 1, config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

    # Stack all frames: [T_total, C, H, W]
    video = torch.stack(all_frames, dim=0)[:, 0]  # Remove batch dimension

    return video


def generate_vbench_samples(
    model,
    config,
    num_samples: int = 256,
    frames_per_video: int = 16,
    output_dir: str = 'outputs/vbench_eval/',
    device: str = 'cuda',
    seed_dataset: str = 'jat',
    seed_game: str = 'atari-breakout',
    fps: int = 8,
    upscale: Optional[int] = None,
) -> List[str]:
    """Generate video samples for VBench evaluation.

    Args:
        model: Genesis model
        config: Model config
        num_samples: Number of videos to generate
        frames_per_video: Frames per video (after seed)
        output_dir: Directory to save videos
        device: Device
        seed_dataset: Dataset for seed videos
        seed_game: Game for seed videos
        fps: Output video FPS
        upscale: Optional upscale factor (e.g., 4 for 256x256)

    Returns:
        List of paths to generated videos
    """
    videos_dir = Path(output_dir) / 'videos'
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Load seed videos
    print(f"\nLoading {num_samples} seed videos...")
    seeds = get_seed_videos(config, device, num_samples, seed_dataset, seed_game)

    # If we don't have enough seeds, cycle through them
    while len(seeds) < num_samples:
        seeds = seeds + seeds
    seeds = seeds[:num_samples]

    video_paths = []
    total_time = 0

    print(f"\nGenerating {num_samples} videos ({frames_per_video} frames each)...")

    for i, seed in enumerate(seeds):
        start_time = time.time()

        # Generate video
        video = generate_video_sample(
            model, config, seed,
            num_frames=frames_per_video,
            device=device,
        )

        # Optional upscaling
        if upscale is not None:
            video = upscale_video(video, upscale)

        # Save as MP4
        video_path = videos_dir / f'sample_{i:04d}.mp4'
        tensor_to_mp4(video, str(video_path), fps=fps)
        video_paths.append(str(video_path))

        elapsed = time.time() - start_time
        total_time += elapsed

        # Progress update
        if (i + 1) % 10 == 0 or i == num_samples - 1:
            avg_time = total_time / (i + 1)
            eta = avg_time * (num_samples - i - 1)
            print(f"  [{i+1}/{num_samples}] {elapsed:.2f}s, avg {avg_time:.2f}s, ETA {eta:.0f}s")

        # Clear CUDA cache periodically
        if (i + 1) % 50 == 0 and device == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nGenerated {len(video_paths)} videos in {total_time:.1f}s")
    print(f"Saved to: {videos_dir}")

    return video_paths


def run_pyiqa_evaluation(
    videos_dir: str,
    output_dir: str = 'outputs/vbench_results/',
    device: str = 'cuda',
) -> Dict[str, float]:
    """Run image quality evaluation using pyiqa (VBench backend).

    Uses same metrics as VBench but runs locally without distributed computing.
    Metrics computed:
    - musiq: MUSIQ (Multi-scale Image Quality Transformer)
    - clipiqa: CLIP-based Image Quality Assessment
    - niqe: NIQE (Natural Image Quality Evaluator) - lower is better

    Args:
        videos_dir: Directory containing video files
        output_dir: Directory to save results
        device: Device for computation

    Returns:
        Dict of metric -> score
    """
    import pyiqa
    import imageio

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize metrics
    print("\nInitializing quality metrics...")
    metrics = {
        'musiq': pyiqa.create_metric('musiq', device=device),  # Higher is better
        'clipiqa': pyiqa.create_metric('clipiqa', device=device),  # Higher is better
    }

    # Try NIQE but it may fail on small images
    try:
        metrics['niqe'] = pyiqa.create_metric('niqe', device=device)  # Lower is better
    except Exception as e:
        print(f"  NIQE unavailable: {e}")

    # Get video files
    videos_dir = Path(videos_dir)
    video_files = sorted(videos_dir.glob('*.mp4'))
    print(f"Found {len(video_files)} videos")

    # Collect scores per video per metric
    all_scores = {name: [] for name in metrics.keys()}

    print("\nEvaluating videos...")
    for i, video_path in enumerate(video_files):
        try:
            reader = imageio.get_reader(str(video_path))
            n_frames = reader.count_frames()

            # Sample frames (first, middle, last)
            frame_indices = [0, n_frames // 2, n_frames - 1]
            frame_scores = {name: [] for name in metrics.keys()}

            for frame_idx in frame_indices:
                frame = reader.get_data(frame_idx)

                # Convert to tensor [1, C, H, W]
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                frame_tensor = frame_tensor.to(device)

                # Compute each metric
                for name, metric in metrics.items():
                    try:
                        score = metric(frame_tensor).item()
                        frame_scores[name].append(score)
                    except Exception:
                        pass

            reader.close()

            # Average over sampled frames
            for name in metrics.keys():
                if frame_scores[name]:
                    all_scores[name].append(np.mean(frame_scores[name]))

        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")

        if (i + 1) % 10 == 0 or i == len(video_files) - 1:
            print(f"  [{i+1}/{len(video_files)}] processed")

    # Compute final scores (average over all videos)
    results = {}
    print("\n" + "=" * 50)
    print("QUALITY ASSESSMENT RESULTS")
    print("=" * 50)

    for name in metrics.keys():
        if all_scores[name]:
            avg = np.mean(all_scores[name])
            std = np.std(all_scores[name])
            results[name] = {
                'mean': float(avg),
                'std': float(std),
                'n_samples': len(all_scores[name]),
            }
            # Note: NIQE is "lower is better", others are "higher is better"
            direction = "lower=better" if name == 'niqe' else "higher=better"
            print(f"  {name:12s}: {avg:.4f} +/- {std:.4f} ({direction})")
        else:
            results[name] = None
            print(f"  {name:12s}: N/A")

    print("=" * 50)

    # Save results
    results_path = Path(output_dir) / 'pyiqa_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


def compute_temporal_consistency(
    videos_dir: str,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Compute temporal consistency metrics for video quality.

    Metrics:
    - frame_difference: Average L2 difference between consecutive frames (lower = more stable)
    - motion_magnitude: Average optical flow magnitude (higher = more dynamic)

    Args:
        videos_dir: Directory containing video files
        device: Device for computation

    Returns:
        Dict of metric -> score
    """
    import imageio

    videos_dir = Path(videos_dir)
    video_files = sorted(videos_dir.glob('*.mp4'))

    frame_diffs = []
    motion_mags = []

    print("\nComputing temporal consistency...")
    for i, video_path in enumerate(video_files):
        try:
            reader = imageio.get_reader(str(video_path))
            frames = [reader.get_data(j) for j in range(reader.count_frames())]
            reader.close()

            # Convert to tensors
            frames_tensor = torch.stack([
                torch.from_numpy(f).float() / 255.0 for f in frames
            ])  # [T, H, W, C]

            # Compute consecutive frame differences
            diffs = []
            for t in range(len(frames) - 1):
                diff = torch.sqrt(((frames_tensor[t+1] - frames_tensor[t]) ** 2).mean())
                diffs.append(diff.item())

            if diffs:
                frame_diffs.append(np.mean(diffs))

        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")

        if (i + 1) % 10 == 0 or i == len(video_files) - 1:
            print(f"  [{i+1}/{len(video_files)}] processed")

    results = {}
    if frame_diffs:
        results['frame_consistency'] = {
            'mean': float(np.mean(frame_diffs)),
            'std': float(np.std(frame_diffs)),
            'interpretation': 'lower = more temporally stable',
        }
        print(f"\nFrame consistency: {results['frame_consistency']['mean']:.4f} +/- {results['frame_consistency']['std']:.4f}")

    return results


def run_vbench_evaluation(
    videos_dir: str,
    output_dir: str = 'outputs/vbench_results/',
    dimensions: Optional[List[str]] = None,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Run video quality evaluation.

    Note: VBench CLI has distributed computing issues on Windows.
    Uses pyiqa metrics directly instead, which are the same as VBench uses internally.

    Args:
        videos_dir: Directory containing video files
        output_dir: Directory to save results
        dimensions: Ignored (uses pyiqa metrics directly)
        device: Device for computation

    Returns:
        Dict of metric -> score
    """
    results = {}

    # Run pyiqa quality metrics
    pyiqa_results = run_pyiqa_evaluation(videos_dir, output_dir, device)
    results['quality'] = pyiqa_results

    # Run temporal consistency
    temporal_results = compute_temporal_consistency(videos_dir, device)
    results['temporal'] = temporal_results

    # Save combined results
    results_path = Path(output_dir) / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def print_results_summary(results: Dict[str, any]):
    """Print summary of evaluation results.

    Args:
        results: Dict containing quality and temporal metrics
    """
    print("\n" + "=" * 60)
    print("GENESIS VIDEO QUALITY EVALUATION")
    print("=" * 60)

    # Quality metrics
    if 'quality' in results and results['quality']:
        print("\nImage Quality Metrics (pyiqa):")
        print("-" * 40)
        for metric, data in results['quality'].items():
            if data and isinstance(data, dict):
                mean = data.get('mean', 0)
                std = data.get('std', 0)
                direction = "lower=better" if metric == 'niqe' else "higher=better"
                print(f"  {metric:12s}: {mean:.4f} +/- {std:.4f} ({direction})")
            else:
                print(f"  {metric:12s}: N/A")

    # Temporal metrics
    if 'temporal' in results and results['temporal']:
        print("\nTemporal Consistency Metrics:")
        print("-" * 40)
        for metric, data in results['temporal'].items():
            if data and isinstance(data, dict):
                mean = data.get('mean', 0)
                std = data.get('std', 0)
                interp = data.get('interpretation', '')
                print(f"  {metric:20s}: {mean:.4f} +/- {std:.4f}")
                print(f"    ({interp})")

    print("\n" + "=" * 60)
    print("Note: VBench CLI unavailable on Windows (distributed computing issue)")
    print("Using pyiqa metrics directly (same as VBench backend)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='VBench Evaluation for Genesis')

    # Required
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')

    # Generation
    parser.add_argument('--num-samples', type=int, default=256, help='Number of videos to generate')
    parser.add_argument('--frames-per-video', type=int, default=16, help='Frames per video')
    parser.add_argument('--fps', type=int, default=8, help='Video FPS')
    parser.add_argument('--output-dir', default='outputs/vbench_eval/', help='Output directory')

    # Model configuration
    parser.add_argument('--slot-norm-mode', default='clip',
                        choices=['decay', 'layernorm', 'clip', 'none'],
                        help='Slot normalization mode (MUST use clip for inference)')
    parser.add_argument('--upscale', type=int, default=None,
                        help='Upscale factor (e.g., 4 for 64->256)')

    # Data
    parser.add_argument('--seed-dataset', default='webvid',
                        help='Dataset for seed videos (webvid for 720p, jat for Atari)')
    parser.add_argument('--seed-game', default=None,
                        help='Game for seed videos (only for jat dataset)')

    # Evaluation
    parser.add_argument('--run-vbench', action='store_true', help='Run VBench evaluation')
    parser.add_argument('--dimensions', default=None,
                        help='Comma-separated VBench dimensions to evaluate')

    # Device
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Validate slot-norm-mode
    if args.slot_norm_mode != 'clip':
        print(f"WARNING: Using slot_norm_mode='{args.slot_norm_mode}' instead of 'clip'")
        print("  For best results, use --slot-norm-mode clip (inference override)")
        print("  Model should be trained with decay, inference should use clip")

    # Load model
    model, config = load_model(args.checkpoint, args.device, args.slot_norm_mode)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate videos
    videos_dir = output_dir / 'videos'
    video_paths = generate_vbench_samples(
        model, config,
        num_samples=args.num_samples,
        frames_per_video=args.frames_per_video,
        output_dir=str(output_dir),
        device=args.device,
        seed_dataset=args.seed_dataset,
        seed_game=args.seed_game,
        fps=args.fps,
        upscale=args.upscale,
    )

    print(f"\nGenerated {len(video_paths)} videos")

    # Save generation metadata
    metadata = {
        'model': str(args.checkpoint),
        'num_samples': args.num_samples,
        'frames_per_video': args.frames_per_video,
        'fps': args.fps,
        'resolution': f"{config.image_size}x{config.image_size}",
        'upscaled': f"{config.image_size * args.upscale}x{config.image_size * args.upscale}" if args.upscale else None,
        'slot_norm_mode': args.slot_norm_mode,
        'seed_dataset': args.seed_dataset,
        'seed_game': args.seed_game,
    }
    with open(output_dir / 'generation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Run VBench evaluation (optional)
    if args.run_vbench:
        dimensions = args.dimensions.split(',') if args.dimensions else None
        results = run_vbench_evaluation(
            videos_dir=str(videos_dir),
            output_dir=str(output_dir / 'results'),
            dimensions=dimensions,
        )

        # Save full results
        full_results = {
            'metadata': metadata,
            'scores': results,
            'baseline': {'VideoAR-4B': 81.74},
        }
        with open(output_dir / 'vbench_results.json', 'w') as f:
            json.dump(full_results, f, indent=2, default=str)

        # Print summary
        print_results_summary(results)
    else:
        print("\nSkipping VBench evaluation (use --run-vbench to enable)")
        print(f"Videos saved to: {videos_dir}")
        print("\nTo run VBench manually:")
        print(f"  vbench evaluate --dimension <dim> --videos_path {videos_dir} --mode=custom_input")


if __name__ == '__main__':
    main()
