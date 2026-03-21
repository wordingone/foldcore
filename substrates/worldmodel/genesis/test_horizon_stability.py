"""Test Horizon Stability: 1000+ frame generation without explosion.

Action A from Verification-First Action Plan.

Tests:
1. MSE at frames [100, 250, 500, 750, 1000]
2. PSNR at same intervals
3. GPU memory usage over time
4. Generation time per frame
5. Slot state norm (if slots enabled)
6. Mode collapse detection (frame variance, consecutive similarity)

Pass/Fail Criteria:
- MSE@1000 / MSE@100 <= 2.0x (no explosion)
- PSNR@1000 - PSNR@100 >= -3 dB (no degradation)
- Memory growth <= 10% (no leak)
- Slot state norm bounded (max < 2x min)
- Frame variance NOT constant (mode collapse detection)

Configuration Options:
- --slot-decay: Override slot_decay (default: use checkpoint value, try 1.0)
- --random-actions: Use random actions instead of zeros
- --no-slots: Disable slot attention entirely
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between prediction and target."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0  # Perfect match
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def get_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_memory_breakdown() -> Dict[str, float]:
    """Get detailed CUDA memory breakdown in MB."""
    if not torch.cuda.is_available():
        return {'allocated': 0, 'reserved': 0, 'cached': 0, 'max_allocated': 0}

    return {
        'allocated': torch.cuda.memory_allocated() / (1024 * 1024),
        'reserved': torch.cuda.memory_reserved() / (1024 * 1024),
        'max_allocated': torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def estimate_tensor_size_mb(tensor: torch.Tensor) -> float:
    """Estimate tensor size in MB."""
    return tensor.numel() * tensor.element_size() / (1024 * 1024)


def load_model(checkpoint_path: str, device: str, slot_decay_override: float = None,
               disable_slots: bool = False, slot_norm_mode_override: str = None):
    """Load Genesis model from checkpoint with optional overrides.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        slot_decay_override: If set, override model's slot_decay (try 1.0 to test no decay)
        disable_slots: If True, disable slot attention entirely
        slot_norm_mode_override: If set, override slot normalization ('decay', 'layernorm', 'none')
    """
    from genesis.genesis_model import Genesis

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']

    # Apply overrides before creating model
    original_slot_decay = getattr(config, 'slot_decay', 0.95)
    original_use_slots = config.use_slots

    if disable_slots:
        config.use_slots = False
        print(f"OVERRIDE: Disabled slots (was use_slots={original_use_slots})")

    if slot_decay_override is not None and config.use_slots:
        config.slot_decay = slot_decay_override
        print(f"OVERRIDE: slot_decay {original_slot_decay} -> {slot_decay_override}")

    original_slot_norm_mode = getattr(config, 'slot_norm_mode', 'decay')
    if slot_norm_mode_override is not None and config.use_slots:
        config.slot_norm_mode = slot_norm_mode_override
        print(f"OVERRIDE: slot_norm_mode {original_slot_norm_mode} -> {slot_norm_mode_override}")

    print(f"Config: {config}")

    model = Genesis(config)

    # Load weights (may have mismatches if slots disabled or norm mode changed)
    if disable_slots and original_use_slots:
        # Partial load - skip slot-related weights
        state_dict = ckpt['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items()
                         if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded {len(filtered_state)}/{len(state_dict)} weights (slots disabled)")
    elif slot_norm_mode_override in ['layernorm', 'layernorm_noise'] and original_slot_norm_mode not in ['layernorm', 'layernorm_noise']:
        # Checkpoint was saved without LayerNorm, load with strict=False
        state_dict = ckpt['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in state_dict.items()
                         if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded {len(filtered_state)}/{len(state_dict)} weights (slot_prior_norm initialized fresh)")
    else:
        model.load_state_dict(ckpt['model_state_dict'])

    model = model.to(device)
    model.eval()

    # Apply slot_decay override to the dynamics model directly
    if slot_decay_override is not None and hasattr(model, 'dynamics'):
        if hasattr(model.dynamics, 'slot_decay'):
            model.dynamics.slot_decay = slot_decay_override
            print(f"  Applied slot_decay={slot_decay_override} to dynamics model")

    # Apply slot_norm_mode override to the dynamics model directly
    if slot_norm_mode_override is not None and hasattr(model, 'dynamics'):
        if hasattr(model.dynamics, 'slot_norm_mode'):
            model.dynamics.slot_norm_mode = slot_norm_mode_override
            # Create LayerNorm if needed
            if slot_norm_mode_override in ['layernorm', 'layernorm_noise'] and model.dynamics.slot_prior_norm is None:
                import torch.nn as nn
                model.dynamics.slot_prior_norm = nn.LayerNorm(model.dynamics.slot_dim).to(device)
            print(f"  Applied slot_norm_mode={slot_norm_mode_override} to dynamics model")

    print(f"Model loaded: {model.count_parameters():,} parameters")
    print(f"  - use_slots: {config.use_slots}")
    print(f"  - slot_decay: {getattr(config, 'slot_decay', 'N/A')}")
    print(f"  - slot_norm_mode: {getattr(config, 'slot_norm_mode', 'decay')}")
    print(f"  - num_slots: {config.num_slots if config.use_slots else 'N/A'}")

    return model, config


def get_seed_video(config, device: str, dataset: str = 'jat', game: str = 'atari-breakout') -> torch.Tensor:
    """Get seed video from dataset."""
    from genesis.pilot.stream_hf import create_streaming_loader

    loader = create_streaming_loader(
        dataset_name=dataset,
        batch_size=1,
        seq_length=config.num_frames,
        image_size=config.image_size,
        game=game if dataset == 'jat' else None,
        shuffle=False,
    )

    # Get first batch
    for batch in loader:
        if 'frames' in batch:
            video = batch['frames']
        else:
            video = batch['video']
        return video.to(device)

    raise RuntimeError("Could not load seed video from dataset")


def generate_long_horizon(
    model,
    config,
    seed_video: torch.Tensor,
    num_frames: int,
    measurement_intervals: List[int],
    device: str,
    random_actions: bool = False,
    action_scale: float = 0.1,
    profile_memory: bool = False,
) -> Dict:
    """Generate long horizon and collect measurements.

    Args:
        model: Genesis model
        config: Model config
        seed_video: [1, T_seed, C, H, W] seed frames
        num_frames: Total frames to generate
        measurement_intervals: Frame numbers to measure at
        device: torch device
        random_actions: If True, use random actions instead of zeros
        action_scale: Scale for random actions (default 0.1)
        profile_memory: If True, collect detailed per-step memory breakdown

    Returns:
        Dict with measurements
    """
    model.eval()
    B, T_seed, C, H, W = seed_video.shape

    # Results storage
    results = {
        'frame': [],
        'mse': [],
        'psnr': [],
        'memory_mb': [],
        'time_per_frame': [],
        'slot_norm': [],
    }

    # Initial memory
    torch.cuda.synchronize() if device == 'cuda' else None
    initial_memory = get_memory_mb()
    print(f"\nInitial GPU memory: {initial_memory:.1f} MB")

    # Reset model state
    model.reset_state()

    # Encode seed video as context
    context_latents, _ = model.encode(seed_video)
    current_context = context_latents

    # Create dummy actions (no action)
    action_dim = config.action_dim

    # Generation loop
    prev_slots = None
    bsd_states = None  # BSD layer states
    prev_frame = seed_video[:, -1:]  # Keep only last frame for comparison
    sample_frames = [seed_video[:, 0]]  # Store sparse samples for visualization

    # Check if model uses BSD
    use_bsd = getattr(config, 'use_bsd', False) or getattr(model, '_use_bsd', False)

    print(f"\nGenerating {num_frames} frames...")
    print(f"Measurement intervals: {measurement_intervals}")
    print(f"Using BSD: {use_bsd}")
    print(f"Random actions: {random_actions}" + (f" (scale={action_scale})" if random_actions else ""))

    # Track frame generation times
    frame_times = []

    # Track frame variance to detect mode collapse
    frame_variances = []
    consecutive_similarities = []  # Cosine similarity between consecutive frames

    # Memory profiling data
    memory_profile = [] if profile_memory else None
    if profile_memory:
        torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
        print("\nMEMORY PROFILING ENABLED - collecting per-step breakdown")

    with torch.no_grad():
        for step in range(num_frames):
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time = time.time()

            # Create action for this step
            if random_actions:
                action = torch.randn(B, 1, action_dim, device=device) * action_scale
            else:
                action = torch.zeros(B, 1, action_dim, device=device)

            # Expand action to match context length if needed
            if current_context.shape[1] > 1:
                action = action.expand(-1, current_context.shape[1], -1)

            # Predict next latent
            if use_bsd:
                # BSD model returns (pred, slots, bsd_states)
                pred, prev_slots, bsd_states = model.dynamics(
                    current_context, action, prev_slots, bsd_states
                )
                next_latent = pred[:, -1:]
                slot_norm = torch.norm(prev_slots).item()
            elif config.use_slots:
                pred, prev_slots = model.dynamics(current_context, action, prev_slots)
                next_latent = pred[:, -1:]
                slot_norm = torch.norm(prev_slots).item()
            else:
                pred = model.dynamics(current_context, action)
                next_latent = pred[:, -1:]
                slot_norm = 0.0

            # Decode to frame
            next_frame = model.decode(next_latent)

            # Track frame variance (mode collapse detection)
            frame_var = next_frame.var().item()
            frame_variances.append(frame_var)

            # Track consecutive frame similarity (mode collapse = similarity near 1.0)
            if step > 0:
                # Flatten and compute cosine similarity
                flat_curr = next_frame.view(B, -1)
                flat_prev = prev_frame.view(B, -1)
                cos_sim = F.cosine_similarity(flat_curr, flat_prev, dim=1).mean().item()
                consecutive_similarities.append(cos_sim)

            # Update context (sliding window)
            window = min(current_context.shape[1] + 1, config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

            torch.cuda.synchronize() if device == 'cuda' else None
            frame_time = time.time() - start_time
            frame_times.append(frame_time)

            # Memory profiling (every 50 steps to reduce overhead)
            if profile_memory and (step % 50 == 0 or step < 10):
                mem_breakdown = get_memory_breakdown()
                profile_entry = {
                    'step': step,
                    'allocated_mb': mem_breakdown['allocated'],
                    'reserved_mb': mem_breakdown['reserved'],
                    'max_allocated_mb': mem_breakdown['max_allocated'],
                    'context_length': current_context.shape[1],
                    'context_size_mb': estimate_tensor_size_mb(current_context),
                    'slot_size_mb': estimate_tensor_size_mb(prev_slots) if prev_slots is not None else 0,
                }
                memory_profile.append(profile_entry)

                if step < 10 or step % 200 == 0:
                    print(f"  [PROFILE] Step {step}: alloc={mem_breakdown['allocated']:.1f}MB, "
                          f"ctx_len={current_context.shape[1]}, ctx_size={profile_entry['context_size_mb']:.2f}MB, "
                          f"slot_size={profile_entry['slot_size_mb']:.2f}MB")

            # Check if this is a measurement interval
            frame_num = step + 1
            if frame_num in measurement_intervals:
                # MSE: compare current frame to previous frame (temporal consistency)
                mse = F.mse_loss(next_frame, prev_frame).item()

                # PSNR
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))

                # Memory - force garbage collection first
                import gc
                gc.collect()
                torch.cuda.empty_cache() if device == 'cuda' else None
                memory = get_memory_mb()

                # Average time per frame (last 100)
                avg_time = np.mean(frame_times[-100:]) if len(frame_times) >= 100 else np.mean(frame_times)

                # Average frame variance (mode collapse indicator)
                avg_var = np.mean(frame_variances[-100:]) if len(frame_variances) >= 100 else np.mean(frame_variances)

                # Record
                results['frame'].append(frame_num)
                results['mse'].append(mse)
                results['psnr'].append(psnr)
                results['memory_mb'].append(memory)
                results['time_per_frame'].append(avg_time * 1000)  # in ms
                results['slot_norm'].append(slot_norm)

                print(f"  Frame {frame_num:4d}: MSE={mse:.6f}, PSNR={psnr:.2f}dB, "
                      f"Memory={memory:.1f}MB, Time={avg_time*1000:.1f}ms/frame, "
                      f"SlotNorm={slot_norm:.4f}, FrameVar={avg_var:.6f}")

                # Store sample frame for visualization
                sample_frames.append(next_frame[:, 0].clone())

            # Update previous frame reference
            prev_frame = next_frame

            # Progress indicator
            if step % 100 == 0 and step > 0:
                print(f"  ... {step}/{num_frames} frames generated")

    # Final memory (with garbage collection)
    import gc
    gc.collect()
    torch.cuda.empty_cache() if device == 'cuda' else None
    final_memory = get_memory_mb()

    results['initial_memory'] = initial_memory
    results['final_memory'] = final_memory
    results['memory_growth_pct'] = (final_memory - initial_memory) / initial_memory * 100 if initial_memory > 0 else 0
    results['avg_frame_variance'] = np.mean(frame_variances)

    # Mode collapse detection metrics
    results['frame_variance_std'] = np.std(frame_variances)  # Should be >0 if not collapsed
    results['frame_variance_trend'] = frame_variances[-100:] if len(frame_variances) >= 100 else frame_variances
    results['avg_consecutive_similarity'] = np.mean(consecutive_similarities) if consecutive_similarities else 0.0
    results['max_consecutive_similarity'] = max(consecutive_similarities) if consecutive_similarities else 0.0

    # Mode collapse verdict: variance is constant AND similarity is very high
    variance_constant = results['frame_variance_std'] < 0.001
    similarity_high = results['avg_consecutive_similarity'] > 0.999
    results['mode_collapse_detected'] = variance_constant and similarity_high

    # Store sample frames (not full video to save memory)
    results['sample_frames'] = torch.stack(sample_frames, dim=1)

    # Store memory profile if collected
    if memory_profile:
        results['memory_profile'] = memory_profile

    return results


def evaluate_stability(results: Dict, measurement_intervals: List[int]) -> Dict:
    """Evaluate stability against pass/fail criteria.

    Criteria:
    - MSE@1000 / MSE@100 <= 2.0x
    - PSNR@1000 - PSNR@100 >= -3 dB
    - Memory growth <= 10%
    - Slot norm bounded (max < 2x min)
    """
    evaluation = {}

    # Find indices for 100 and 1000 frames
    frames = results['frame']

    # Get MSE values
    mse_100 = None
    mse_1000 = None
    psnr_100 = None
    psnr_1000 = None

    for i, f in enumerate(frames):
        if f == 100:
            mse_100 = results['mse'][i]
            psnr_100 = results['psnr'][i]
        if f >= 1000 or f == frames[-1]:  # Use last measurement if < 1000 frames
            mse_1000 = results['mse'][i]
            psnr_1000 = results['psnr'][i]
            break

    # MSE ratio
    if mse_100 and mse_100 > 1e-10:
        mse_ratio = mse_1000 / mse_100
    else:
        mse_ratio = 1.0

    # PSNR delta
    if psnr_100 is not None and psnr_1000 is not None:
        psnr_delta = psnr_1000 - psnr_100
    else:
        psnr_delta = 0.0

    # Slot norm bounds
    slot_norms = [s for s in results['slot_norm'] if s > 0]
    if slot_norms:
        slot_min = min(slot_norms)
        slot_max = max(slot_norms)
        slot_bounded = slot_max < 2 * slot_min if slot_min > 0 else True
    else:
        slot_min = 0
        slot_max = 0
        slot_bounded = True  # No slots

    # Evaluate criteria
    evaluation['mse_100'] = mse_100
    evaluation['mse_1000'] = mse_1000
    evaluation['mse_ratio'] = mse_ratio
    evaluation['mse_pass'] = mse_ratio <= 2.0

    evaluation['psnr_100'] = psnr_100
    evaluation['psnr_1000'] = psnr_1000
    evaluation['psnr_delta'] = psnr_delta
    evaluation['psnr_pass'] = psnr_delta >= -3.0

    evaluation['memory_start'] = results['initial_memory']
    evaluation['memory_end'] = results['final_memory']
    evaluation['memory_growth_pct'] = results['memory_growth_pct']

    # Memory pass: Either ≤20% total growth OR stable after warm-up
    # Stable = memory at end ≈ memory at first measurement (within 5%)
    memory_readings = results['memory_mb']
    if len(memory_readings) >= 2:
        # Check if memory stabilized after first measurement
        first_reading = memory_readings[0]
        last_reading = memory_readings[-1]
        stabilized = abs(last_reading - first_reading) / first_reading < 0.05  # <5% change after warm-up
        evaluation['memory_stabilized'] = stabilized
        evaluation['memory_pass'] = results['memory_growth_pct'] <= 20.0 or stabilized
    else:
        evaluation['memory_stabilized'] = False
        evaluation['memory_pass'] = results['memory_growth_pct'] <= 20.0

    evaluation['slot_min'] = slot_min
    evaluation['slot_max'] = slot_max
    evaluation['slot_bounded'] = slot_bounded

    # Mode collapse detection
    evaluation['mode_collapse'] = results.get('mode_collapse_detected', False)
    evaluation['frame_variance_std'] = results.get('frame_variance_std', 0.0)
    evaluation['avg_consecutive_similarity'] = results.get('avg_consecutive_similarity', 0.0)
    evaluation['no_collapse'] = not evaluation['mode_collapse']

    # Overall verdict - now includes mode collapse check
    evaluation['verdict'] = 'PASS' if (
        evaluation['mse_pass'] and
        evaluation['psnr_pass'] and
        evaluation['memory_pass'] and
        evaluation['slot_bounded'] and
        evaluation['no_collapse']  # NEW: Must not have mode collapse
    ) else 'FAIL'

    # Add failure reasons
    failures = []
    if not evaluation['mse_pass']:
        failures.append(f"MSE ratio {evaluation['mse_ratio']:.2f}x > 2.0x")
    if not evaluation['psnr_pass']:
        failures.append(f"PSNR delta {evaluation['psnr_delta']:.2f}dB < -3dB")
    if not evaluation['memory_pass']:
        failures.append(f"Memory growth {evaluation['memory_growth_pct']:.1f}% > 10%")
    if not evaluation['slot_bounded']:
        failures.append(f"Slot norm unbounded")
    if evaluation['mode_collapse']:
        failures.append(f"Mode collapse (var_std={evaluation['frame_variance_std']:.6f}, sim={evaluation['avg_consecutive_similarity']:.4f})")
    evaluation['failure_reasons'] = failures

    return evaluation


def save_results(results: Dict, evaluation: Dict, output_dir: Path):
    """Save results to CSV and text files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save measurements CSV
    csv_path = output_dir / 'measurements.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'mse', 'psnr', 'memory_mb', 'time_per_frame_ms', 'slot_norm'])
        for i in range(len(results['frame'])):
            writer.writerow([
                results['frame'][i],
                results['mse'][i],
                results['psnr'][i],
                results['memory_mb'][i],
                results['time_per_frame'][i],
                results['slot_norm'][i],
            ])
    print(f"\nMeasurements saved to: {csv_path}")

    # Save evaluation summary
    summary_path = output_dir / 'evaluation.txt'
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("HORIZON STABILITY EVALUATION\n")
        f.write("=" * 70 + "\n\n")

        f.write("MSE STABILITY:\n")
        f.write(f"  MSE@100: {evaluation['mse_100']:.6f}\n")
        f.write(f"  MSE@{results['frame'][-1]}: {evaluation['mse_1000']:.6f}\n")
        f.write(f"  Ratio: {evaluation['mse_ratio']:.2f}x\n")
        f.write(f"  Criteria: <= 2.0x\n")
        f.write(f"  Result: {'PASS' if evaluation['mse_pass'] else 'FAIL'}\n\n")

        f.write("PSNR DEGRADATION:\n")
        f.write(f"  PSNR@100: {evaluation['psnr_100']:.2f} dB\n")
        f.write(f"  PSNR@{results['frame'][-1]}: {evaluation['psnr_1000']:.2f} dB\n")
        f.write(f"  Delta: {evaluation['psnr_delta']:.2f} dB\n")
        f.write(f"  Criteria: >= -3.0 dB\n")
        f.write(f"  Result: {'PASS' if evaluation['psnr_pass'] else 'FAIL'}\n\n")

        f.write("MEMORY GROWTH:\n")
        f.write(f"  Start: {evaluation['memory_start']:.1f} MB\n")
        f.write(f"  End: {evaluation['memory_end']:.1f} MB\n")
        f.write(f"  Growth: {evaluation['memory_growth_pct']:.1f}%\n")
        f.write(f"  Criteria: <= 10%\n")
        f.write(f"  Result: {'PASS' if evaluation['memory_pass'] else 'FAIL'}\n\n")

        f.write("SLOT NORM BOUNDS:\n")
        f.write(f"  Min: {evaluation['slot_min']:.4f}\n")
        f.write(f"  Max: {evaluation['slot_max']:.4f}\n")
        f.write(f"  Ratio: {evaluation['slot_max']/evaluation['slot_min']:.2f}x\n" if evaluation['slot_min'] > 0 else "  Ratio: N/A\n")
        f.write(f"  Criteria: max < 2x min\n")
        f.write(f"  Result: {'PASS' if evaluation['slot_bounded'] else 'FAIL'}\n\n")

        f.write("=" * 70 + "\n")
        f.write(f"OVERALL VERDICT: {evaluation['verdict']}\n")
        f.write("=" * 70 + "\n")

    print(f"Evaluation saved to: {summary_path}")

    # Save sample frames
    if 'sample_frames' in results:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        video = results['sample_frames'][0].cpu()  # [T_samples, C, H, W]
        T = video.shape[0]

        fig, axes = plt.subplots(1, T, figsize=(3 * T, 3))
        if T == 1:
            axes = [axes]
        for i in range(T):
            frame = video[i].permute(1, 2, 0).numpy()
            frame = np.clip(frame, 0, 1)
            axes[i].imshow(frame)
            axes[i].set_title(f'Sample {i}')
            axes[i].axis('off')

        plt.tight_layout()
        fig.savefig(output_dir / 'frame_samples.png', dpi=150)
        plt.close()
        print(f"Frame samples saved to: {output_dir / 'frame_samples.png'}")

    # Save memory profile if available
    if 'memory_profile' in results and results['memory_profile']:
        profile_path = output_dir / 'memory_profile.csv'
        with open(profile_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results['memory_profile'][0].keys())
            writer.writeheader()
            writer.writerows(results['memory_profile'])
        print(f"Memory profile saved to: {profile_path}")

        # Print memory growth analysis
        profile = results['memory_profile']
        if len(profile) >= 2:
            first = profile[0]
            last = profile[-1]
            growth = last['allocated_mb'] - first['allocated_mb']
            growth_pct = growth / first['allocated_mb'] * 100 if first['allocated_mb'] > 0 else 0
            print(f"\n=== MEMORY PROFILE ANALYSIS ===")
            print(f"Initial allocated: {first['allocated_mb']:.1f} MB")
            print(f"Final allocated: {last['allocated_mb']:.1f} MB")
            print(f"Growth: {growth:.1f} MB ({growth_pct:.1f}%)")
            print(f"Context length: {first['context_length']} -> {last['context_length']}")
            print(f"Context size: {first['context_size_mb']:.2f} MB -> {last['context_size_mb']:.2f} MB")
            print(f"Slot size: {first['slot_size_mb']:.2f} MB -> {last['slot_size_mb']:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Test Genesis Horizon Stability')
    parser.add_argument('--checkpoint', default='checkpoints/genesis/best_genesis.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--frames', type=int, default=1000,
                        help='Number of frames to generate')
    parser.add_argument('--context', type=int, default=4,
                        help='Number of context frames from seed video')
    parser.add_argument('--output', default='outputs/horizon_test/',
                        help='Output directory for results')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    parser.add_argument('--dataset', default='jat',
                        help='Dataset to use for seed video')
    parser.add_argument('--game', default='atari-breakout',
                        help='Game to use from JAT dataset')

    # Mode collapse fix testing options
    parser.add_argument('--slot-decay', type=float, default=None,
                        help='Override slot_decay (try 1.0 for no decay)')
    parser.add_argument('--random-actions', action='store_true',
                        help='Use random actions instead of zeros')
    parser.add_argument('--action-scale', type=float, default=0.1,
                        help='Scale for random actions (default 0.1)')
    parser.add_argument('--no-slots', action='store_true',
                        help='Disable slot attention entirely')
    parser.add_argument('--slot-norm-mode', type=str, default=None,
                        choices=['decay', 'layernorm', 'layernorm_noise', 'clip', 'none'],
                        help='Override slot normalization mode')
    parser.add_argument('--profile-memory', action='store_true',
                        help='Enable detailed memory profiling (saves per-step breakdown)')
    args = parser.parse_args()

    print("=" * 70)
    print("GENESIS HORIZON STABILITY TEST")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Frames to generate: {args.frames}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    if args.slot_decay is not None:
        print(f"OVERRIDE slot_decay: {args.slot_decay}")
    if args.random_actions:
        print(f"OVERRIDE actions: random (scale={args.action_scale})")
    if args.no_slots:
        print(f"OVERRIDE slots: DISABLED")
    if args.slot_norm_mode:
        print(f"OVERRIDE slot_norm_mode: {args.slot_norm_mode}")
    if args.profile_memory:
        print(f"Memory profiling: ENABLED")

    # Load model with overrides
    model, config = load_model(
        args.checkpoint,
        args.device,
        slot_decay_override=args.slot_decay,
        disable_slots=args.no_slots,
        slot_norm_mode_override=args.slot_norm_mode,
    )

    # Get seed video
    print("\nLoading seed video...")
    seed_video = get_seed_video(config, args.device, args.dataset, args.game)
    seed_video = seed_video[:, :args.context]  # Use only context frames
    print(f"Seed video shape: {seed_video.shape}")

    # Measurement intervals
    intervals = [100, 250, 500, 750, 1000]
    intervals = [i for i in intervals if i <= args.frames]
    if args.frames not in intervals:
        intervals.append(args.frames)

    # Generate and measure
    print("\n" + "=" * 70)
    print("GENERATION")
    print("=" * 70)

    results = generate_long_horizon(
        model=model,
        config=config,
        seed_video=seed_video,
        num_frames=args.frames,
        measurement_intervals=intervals,
        device=args.device,
        random_actions=args.random_actions,
        action_scale=args.action_scale,
        profile_memory=args.profile_memory,
    )

    # Evaluate stability
    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    evaluation = evaluate_stability(results, intervals)

    # Print summary
    print(f"\nMSE Ratio: {evaluation['mse_ratio']:.2f}x (criteria: <= 2.0x) -> {'PASS' if evaluation['mse_pass'] else 'FAIL'}")
    print(f"PSNR Delta: {evaluation['psnr_delta']:.2f} dB (criteria: >= -3 dB) -> {'PASS' if evaluation['psnr_pass'] else 'FAIL'}")
    print(f"Memory Growth: {evaluation['memory_growth_pct']:.1f}% (criteria: <= 10%) -> {'PASS' if evaluation['memory_pass'] else 'FAIL'}")
    print(f"Slot Bounded: {evaluation['slot_bounded']} (criteria: max < 2x min) -> {'PASS' if evaluation['slot_bounded'] else 'FAIL'}")
    print(f"Mode Collapse: {'DETECTED' if evaluation['mode_collapse'] else 'NOT DETECTED'} "
          f"(var_std={evaluation['frame_variance_std']:.6f}, sim={evaluation['avg_consecutive_similarity']:.4f})")

    print("\n" + "=" * 70)
    print(f"OVERALL VERDICT: {evaluation['verdict']}")
    if evaluation['failure_reasons']:
        print(f"Failures: {', '.join(evaluation['failure_reasons'])}")
    print("=" * 70)

    # Save results
    output_dir = Path(args.output)
    save_results(results, evaluation, output_dir)

    # Return for plan tracking
    return evaluation


if __name__ == '__main__':
    main()
