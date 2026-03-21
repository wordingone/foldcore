"""Test Infinite Horizon: Can Genesis play Pong forever?

Core requirement: No ending horizon. Generate indefinitely without:
1. Error accumulation blowing up
2. Memory growing unbounded
3. Quality degrading over time

Test: Roll out 1000+ frames and measure:
- MSE over time (should stay bounded)
- Memory usage (should stay constant)
- Visual coherence (qualitative)
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from genesis.pilot.motion_model import MotionSlotModel
from genesis.pilot.video_data import get_video_dataset


def rollout_autoregressive(model, seed_frames, num_steps, device):
    """Roll out model autoregressively for num_steps.

    Args:
        model: Trained motion model
        seed_frames: [1, T_seed, C, H, W] - initial frames to seed
        num_steps: Number of frames to generate
        device: torch device

    Returns:
        generated: [1, num_steps, C, H, W] - generated frames
        memory_usage: List of memory usage at each step
    """
    model.eval()
    B, T_seed, C, H, W = seed_frames.shape

    # Start with seed frames
    history = [seed_frames[:, i] for i in range(T_seed)]
    generated = []
    memory_usage = []

    with torch.no_grad():
        for step in range(num_steps):
            # Take last 2 frames as input
            # Model expects [B, T, C, H, W] with T >= 3
            # We'll feed 3 frames and predict 1
            if len(history) >= 2:
                # Build input: [prev, curr]
                prev = history[-2]
                curr = history[-1]

                # Run encoder + decoder directly for single-step prediction
                feat = model.encoder(prev, curr)

                # Slot attention (carry slots across time for object tracking)
                if not hasattr(rollout_autoregressive, 'slots'):
                    rollout_autoregressive.slots = None

                slot_input = model.to_slot_input(feat)
                slot_input = slot_input.flatten(2).transpose(1, 2)
                rollout_autoregressive.slots, _ = model.slot_attention(
                    slot_input,
                    rollout_autoregressive.slots
                )

                # Slot influence
                slot_feat = model.slot_to_spatial(rollout_autoregressive.slots)
                slot_feat = slot_feat.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
                slot_feat = slot_feat.expand(-1, -1, 16, 16)

                combined = torch.cat([feat, slot_feat], dim=1)
                combined = model.combine(combined)

                # Decode to motion + residual
                motion, residual = model.decoder(combined)

                # Warp current frame
                from genesis.pilot.motion_model import spatial_transformer
                warped = spatial_transformer(curr, motion)

                # Add residual
                pred = torch.clamp(warped + residual, 0, 1)

                # Append to history (keep bounded)
                history.append(pred)
                if len(history) > 10:  # Keep only last 10 frames
                    history.pop(0)

                generated.append(pred)

                # Track memory
                if device == 'cuda':
                    memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
                else:
                    memory_usage.append(0)

            if step % 100 == 0:
                print(f"  Step {step}/{num_steps}")

    # Reset slots for next rollout
    rollout_autoregressive.slots = None

    return torch.stack(generated, dim=1), memory_usage


def measure_rollout_quality(model, dataset, num_steps, device):
    """Measure prediction quality over long rollout.

    Compare generated frames to ground truth at each step.
    """
    model.eval()

    # Get a long sequence from dataset
    # We'll use multiple sequences concatenated
    all_frames = []
    for i in range(min(100, len(dataset))):
        frames = dataset[i]  # [T, C, H, W]
        all_frames.append(frames)

    # Concatenate into one long sequence
    long_sequence = torch.cat(all_frames, dim=0)  # [T_total, C, H, W]
    long_sequence = long_sequence.unsqueeze(0).to(device)  # [1, T_total, C, H, W]

    print(f"Ground truth length: {long_sequence.shape[1]} frames")

    # Limit to num_steps
    num_steps = min(num_steps, long_sequence.shape[1] - 3)

    # Seed with first 3 frames
    seed = long_sequence[:, :3]

    # Generate
    print("Generating...")
    generated, memory = rollout_autoregressive(model, seed, num_steps, device)

    # Compare to ground truth
    gt = long_sequence[:, 3:3+num_steps]

    # Compute MSE at different time horizons
    mse_over_time = []
    for t in range(num_steps):
        mse = F.mse_loss(generated[:, t], gt[:, t]).item()
        mse_over_time.append(mse)

    return {
        'mse_over_time': mse_over_time,
        'memory_usage': memory,
        'generated': generated,
        'ground_truth': gt,
    }


def plot_results(results, save_path):
    """Plot rollout quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MSE over time
    ax = axes[0, 0]
    ax.plot(results['mse_over_time'])
    ax.set_xlabel('Time step')
    ax.set_ylabel('MSE')
    ax.set_title('Prediction Error Over Time')
    ax.set_yscale('log')

    # Moving average MSE
    ax = axes[0, 1]
    window = 50
    mse = np.array(results['mse_over_time'])
    if len(mse) > window:
        ma = np.convolve(mse, np.ones(window)/window, mode='valid')
        ax.plot(ma)
    ax.set_xlabel('Time step')
    ax.set_ylabel('MSE (50-step moving avg)')
    ax.set_title('Smoothed Prediction Error')

    # Memory usage
    ax = axes[1, 0]
    if results['memory_usage']:
        ax.plot(results['memory_usage'])
        ax.set_xlabel('Time step')
        ax.set_ylabel('GPU Memory (MB)')
        ax.set_title('Memory Usage Over Time')

    # Sample frames
    ax = axes[1, 1]
    gen = results['generated']
    gt = results['ground_truth']

    # Show frames at different time points
    time_points = [0, len(gen[0])//4, len(gen[0])//2, len(gen[0])-1]
    time_points = [t for t in time_points if t < len(gen[0])]

    n_cols = len(time_points)
    comparison = []
    for t in time_points:
        gen_frame = gen[0, t].cpu().permute(1, 2, 0).numpy()
        gt_frame = gt[0, t].cpu().permute(1, 2, 0).numpy()
        # Stack vertically: GT on top, generated below
        combined = np.vstack([gt_frame, gen_frame])
        comparison.append(combined)

    if comparison:
        full_comparison = np.hstack(comparison)
        ax.imshow(full_comparison)
        ax.set_title(f'GT (top) vs Generated (bottom) at t={time_points}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/motion/best_slot_pong.pt')
    parser.add_argument('--dataset', type=str, default='pong')
    parser.add_argument('--num-steps', type=int, default=500)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("=" * 70)
    print("INFINITE HORIZON TEST")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Rollout steps: {args.num_steps}")
    print(f"Device: {args.device}")

    # Load model
    print("\nLoading model...")
    model = MotionSlotModel(base_channels=48, num_slots=8, slot_dim=64)

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"WARNING: No checkpoint found at {args.checkpoint}")
        print("Using untrained model for testing architecture only")

    model = model.to(args.device)
    model.eval()

    # Load dataset
    print("\nLoading dataset...")
    dataset = get_video_dataset(f"tinyworlds:{args.dataset}", seq_length=16)
    print(f"Dataset size: {len(dataset)} sequences")

    # Run rollout test
    print("\nRunning infinite horizon test...")
    results = measure_rollout_quality(model, dataset, args.num_steps, args.device)

    # Analyze results
    mse = np.array(results['mse_over_time'])
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\nMSE Statistics:")
    print(f"  Initial (t=0):      {mse[0]:.6f}")
    print(f"  Middle (t={len(mse)//2}):    {mse[len(mse)//2]:.6f}")
    print(f"  Final (t={len(mse)-1}):     {mse[-1]:.6f}")
    print(f"  Mean:               {mse.mean():.6f}")
    print(f"  Std:                {mse.std():.6f}")
    print(f"  Max:                {mse.max():.6f}")

    # Check for error explosion
    initial_mse = mse[:50].mean()
    final_mse = mse[-50:].mean()
    explosion_ratio = final_mse / initial_mse

    print(f"\nError Accumulation:")
    print(f"  Initial avg (first 50):  {initial_mse:.6f}")
    print(f"  Final avg (last 50):     {final_mse:.6f}")
    print(f"  Explosion ratio:         {explosion_ratio:.2f}x")

    if explosion_ratio < 2.0:
        print("\n  PASS: Error stays bounded (< 2x growth)")
    elif explosion_ratio < 5.0:
        print("\n  WARNING: Moderate error growth (2-5x)")
    else:
        print("\n  FAIL: Error explodes (> 5x growth)")

    # Memory check
    if results['memory_usage']:
        mem = np.array(results['memory_usage'])
        mem_growth = (mem[-1] - mem[0]) / mem[0] * 100 if mem[0] > 0 else 0
        print(f"\nMemory:")
        print(f"  Initial: {mem[0]:.1f} MB")
        print(f"  Final:   {mem[-1]:.1f} MB")
        print(f"  Growth:  {mem_growth:.1f}%")

        if mem_growth < 10:
            print("  PASS: Memory stays bounded")
        else:
            print("  FAIL: Memory grows unbounded")

    # Save plot
    plot_path = Path('checkpoints/motion/infinite_horizon_test.png')
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_results(results, plot_path)

    print("\n" + "=" * 70)
    print("INFINITE HORIZON REQUIREMENTS")
    print("=" * 70)
    print(f"  Error bounded:    {'PASS' if explosion_ratio < 5 else 'FAIL'}")
    print(f"  Memory bounded:   {'PASS' if not results['memory_usage'] or mem_growth < 10 else 'FAIL'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
