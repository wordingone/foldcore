"""Play Sonic through Genesis - Interactive Visualization.

Loads trained model and generates infinite Sonic gameplay.
Saves as GIF/MP4 for viewing.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from PIL import Image
import imageio

from genesis.pilot.motion_model_v2 import InfiniteHorizonModel
from genesis.pilot.model_256 import InfiniteHorizonModel256
from genesis.pilot.video_data import get_video_dataset


def load_model(checkpoint_path, resolution=64, device='cuda'):
    """Load trained model from checkpoint."""
    if resolution == 256:
        model = InfiniteHorizonModel256(
            base_channels=48,
            num_slots=16,
            slot_dim=64,
            slot_decay=0.95,
        )
    else:
        model = InfiniteHorizonModel(
            base_channels=48,
            num_slots=12,
            slot_dim=64,
            slot_decay=0.95,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Horizon ratio at save: {checkpoint.get('horizon_ratio', 'N/A'):.2f}x")
    print(f"  Validation MSE: {checkpoint.get('val_mse', 'N/A'):.6f}")

    return model


def get_seed_frames(dataset, idx=0, num_seeds=2, target_size=None):
    """Get seed frames from dataset."""
    frames = dataset[idx]  # [T, C, H, W]

    if target_size and frames.shape[-1] != target_size:
        frames = F.interpolate(
            frames,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

    return frames[:num_seeds].unsqueeze(0)  # [1, 2, C, H, W]


def generate_video(model, seed_frames, num_steps, device):
    """Generate video frames from seed."""
    model.eval()
    seed_frames = seed_frames.to(device)

    with torch.no_grad():
        generated = model.generate(seed_frames, num_steps)

    # Combine seed + generated
    all_frames = torch.cat([seed_frames, generated], dim=1)
    return all_frames


def frames_to_images(frames, upscale=1):
    """Convert tensor frames to PIL images."""
    # frames: [1, T, C, H, W]
    frames = frames[0].cpu().numpy()  # [T, C, H, W]
    frames = (frames * 255).clip(0, 255).astype(np.uint8)
    frames = frames.transpose(0, 2, 3, 1)  # [T, H, W, C]

    images = []
    for frame in frames:
        img = Image.fromarray(frame)
        if upscale > 1:
            new_size = (img.width * upscale, img.height * upscale)
            img = img.resize(new_size, Image.NEAREST)
        images.append(img)

    return images


def save_gif(images, path, fps=15):
    """Save images as GIF."""
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000/fps),
        loop=0
    )
    print(f"Saved GIF: {path}")


def save_mp4(images, path, fps=15):
    """Save images as MP4."""
    frames = [np.array(img) for img in images]
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved MP4: {path}")


def compare_with_ground_truth(model, dataset, idx, num_steps, device, target_size=None):
    """Generate side-by-side comparison with ground truth."""
    # Get full sequence
    gt_frames = dataset[idx]  # [T, C, H, W]

    if target_size and gt_frames.shape[-1] != target_size:
        gt_frames = F.interpolate(
            gt_frames,
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        )

    # Get seed and generate
    seed = gt_frames[:2].unsqueeze(0).to(device)

    actual_steps = min(num_steps, gt_frames.shape[0] - 2)

    with torch.no_grad():
        generated = model.generate(seed, actual_steps)

    # Build comparison frames (GT on left, generated on right)
    comparison_frames = []

    # First show seed frames
    for i in range(2):
        frame = gt_frames[i].cpu().numpy()
        frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = frame.transpose(1, 2, 0)  # [H, W, C]

        # Create side by side (same frame for seed)
        h, w, c = frame.shape
        combined = np.zeros((h, w * 2 + 10, c), dtype=np.uint8)
        combined[:, :w] = frame
        combined[:, w+10:] = frame
        comparison_frames.append(Image.fromarray(combined))

    # Then show generated vs GT
    for i in range(actual_steps):
        gt_idx = i + 2
        if gt_idx >= gt_frames.shape[0]:
            break

        gt_frame = gt_frames[gt_idx].cpu().numpy()
        gt_frame = (gt_frame * 255).clip(0, 255).astype(np.uint8)
        gt_frame = gt_frame.transpose(1, 2, 0)

        gen_frame = generated[0, i].cpu().numpy()
        gen_frame = (gen_frame * 255).clip(0, 255).astype(np.uint8)
        gen_frame = gen_frame.transpose(1, 2, 0)

        h, w, c = gt_frame.shape
        combined = np.zeros((h, w * 2 + 10, c), dtype=np.uint8)
        combined[:, :w] = gt_frame
        combined[:, w+10:] = gen_frame
        comparison_frames.append(Image.fromarray(combined))

    return comparison_frames


def main():
    parser = argparse.ArgumentParser(description="Play Sonic through Genesis")
    parser.add_argument('--checkpoint', default='checkpoints/sonic/best_sonic.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--resolution', type=int, default=64, choices=[64, 256],
                       help='Model resolution')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of steps to generate')
    parser.add_argument('--seed-idx', type=int, default=0,
                       help='Index of sequence to use as seed')
    parser.add_argument('--output', default='sonic_genesis.gif',
                       help='Output file (gif or mp4)')
    parser.add_argument('--compare', action='store_true',
                       help='Show side-by-side comparison with ground truth')
    parser.add_argument('--upscale', type=int, default=4,
                       help='Upscale factor for visualization')
    parser.add_argument('--fps', type=int, default=15,
                       help='Output framerate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 60)
    print("GENESIS PLAYS SONIC")
    print("=" * 60)

    # Handle checkpoint path based on resolution
    if args.resolution == 256 and 'sonic/best_sonic' in args.checkpoint:
        args.checkpoint = 'checkpoints/256/best_256_sonic.pt'

    # Load model
    print(f"\nLoading {args.resolution}x{args.resolution} model...")
    model = load_model(args.checkpoint, args.resolution, args.device)

    # Load dataset
    print("\nLoading Sonic dataset...")
    dataset = get_video_dataset("tinyworlds:sonic", seq_length=16)
    print(f"Dataset has {len(dataset)} sequences")

    # Determine target size for data
    target_size = args.resolution if args.resolution == 256 else None

    if args.compare:
        print(f"\nGenerating comparison (GT vs Genesis) for {args.steps} steps...")
        images = compare_with_ground_truth(
            model, dataset, args.seed_idx, args.steps,
            args.device, target_size
        )
        # Upscale comparison frames
        if args.upscale > 1:
            images = [img.resize((img.width * args.upscale, img.height * args.upscale),
                                Image.NEAREST) for img in images]
    else:
        print(f"\nGenerating {args.steps} frames...")
        seed = get_seed_frames(dataset, args.seed_idx, num_seeds=2, target_size=target_size)
        frames = generate_video(model, seed, args.steps, args.device)
        images = frames_to_images(frames, upscale=args.upscale)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == '.gif':
        save_gif(images, str(output_path), fps=args.fps)
    else:
        save_mp4(images, str(output_path), fps=args.fps)

    print(f"\nGenerated {len(images)} frames")
    print(f"Output: {output_path.absolute()}")

    # Calculate MSE if comparing
    if args.compare:
        print("\nTo view: open the output file in any image viewer")
        print("Left = Ground Truth, Right = Genesis Generated")


if __name__ == '__main__':
    main()
