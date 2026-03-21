"""Diagnostic script to investigate mode collapse at 256x256.

Tests:
1. Tokenizer roundtrip - does encode→decode preserve information?
2. Latent dynamics - are predicted latents actually changing?
3. Slot state - are slots frozen or evolving?
4. Action sensitivity - does changing actions change output?
5. Collapse onset - when exactly does it start?
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import argparse

from genesis.genesis_model import Genesis, GenesisConfig


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load model with clip mode override."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    config.slot_norm_mode = 'clip'  # Override for inference

    model = Genesis(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Also set dynamics model slot norm mode
    if hasattr(model.dynamics, 'slot_norm_mode'):
        model.dynamics.slot_norm_mode = 'clip'

    return model, config


def get_seed_video(model, device, num_frames=4):
    """Get a seed video from the dataset."""
    from genesis.pilot.stream_hf import create_streaming_loader
    loader = create_streaming_loader(
        dataset_name='jat',
        batch_size=1,
        seq_length=num_frames + 12,
        image_size=model.config.image_size,
        game='atari-breakout',
    )
    batch = next(iter(loader))
    return batch['frames'][:, :num_frames].to(device)


def test_tokenizer_roundtrip(model, seed_video, device):
    """Test 1: Does encode→decode preserve information?"""
    print("\n" + "="*70)
    print("TEST 1: TOKENIZER ROUNDTRIP")
    print("="*70)

    with torch.no_grad():
        # Encode
        latents, intermediates = model.encode(seed_video)
        print(f"Input shape: {seed_video.shape}")
        print(f"Latent shape: {latents.shape}")
        print(f"Latent stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}, std={latents.std():.4f}")

        # Decode
        reconstructed = model.decode(latents)
        print(f"Reconstructed shape: {reconstructed.shape}")

        # Compare
        mse = F.mse_loss(reconstructed, seed_video).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))

        # Per-frame comparison
        print(f"\nReconstruction quality:")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")

        # Check if frames are distinguishable after roundtrip
        frame_mses = []
        for t in range(seed_video.shape[1]):
            frame_mse = F.mse_loss(reconstructed[:, t], seed_video[:, t]).item()
            frame_mses.append(frame_mse)
            print(f"  Frame {t}: MSE={frame_mse:.6f}, PSNR={10*np.log10(1/(frame_mse+1e-8)):.2f} dB")

        # Check inter-frame variance preserved
        orig_var = seed_video.var(dim=1).mean().item()
        recon_var = reconstructed.var(dim=1).mean().item()
        print(f"\nInter-frame variance:")
        print(f"  Original: {orig_var:.6f}")
        print(f"  Reconstructed: {recon_var:.6f}")
        print(f"  Ratio: {recon_var/orig_var:.4f}")

        if psnr > 25:
            print("\n[PASS] Tokenizer preserves information reasonably well")
            return True, latents
        else:
            print("\n[FAIL] Tokenizer loses too much information")
            return False, latents


def test_latent_dynamics(model, seed_video, device, num_steps=20):
    """Test 2: Are predicted latents actually changing over time?"""
    print("\n" + "="*70)
    print("TEST 2: LATENT DYNAMICS")
    print("="*70)

    with torch.no_grad():
        # Encode seed
        latents, _ = model.encode(seed_video)
        print(f"Seed latent shape: {latents.shape}")

        # Generate latents step by step
        model.reset_state()
        current_context = latents

        latent_history = [latents[:, -1:].clone()]  # Last seed frame latent

        for step in range(num_steps):
            # Single step prediction
            if model.config.use_slots:
                pred, model.prev_slots = model.dynamics(current_context, None, model.prev_slots)
            else:
                pred = model.dynamics(current_context, None)

            next_latent = pred[:, -1:]
            latent_history.append(next_latent.clone())

            # Update context
            window = min(current_context.shape[1] + 1, model.config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

        # Analyze latent changes
        print(f"\nLatent evolution over {num_steps} steps:")

        latent_diffs = []
        latent_stds = []
        for i in range(1, len(latent_history)):
            diff = (latent_history[i] - latent_history[i-1]).abs().mean().item()
            latent_diffs.append(diff)
            latent_stds.append(latent_history[i].std().item())

        print(f"  Step  1-5 avg diff: {np.mean(latent_diffs[:5]):.6f}")
        print(f"  Step 6-10 avg diff: {np.mean(latent_diffs[5:10]):.6f}")
        print(f"  Step 11-20 avg diff: {np.mean(latent_diffs[10:]):.6f}")

        print(f"\nLatent std over time:")
        print(f"  Step  1: {latent_stds[0]:.6f}")
        print(f"  Step 10: {latent_stds[9]:.6f}")
        print(f"  Step 20: {latent_stds[-1]:.6f}")

        # Check if latents converge
        final_diff = latent_diffs[-1]
        if final_diff < 1e-5:
            print(f"\n[FAIL] Latents converge to fixed point (diff={final_diff:.8f})")
            return False, latent_history
        else:
            print(f"\n[PASS] Latents continue evolving (diff={final_diff:.6f})")
            return True, latent_history


def test_slot_evolution(model, seed_video, device, num_steps=20):
    """Test 3: Are slots frozen or evolving?"""
    print("\n" + "="*70)
    print("TEST 3: SLOT STATE EVOLUTION")
    print("="*70)

    if not model.config.use_slots:
        print("Slots not enabled, skipping test")
        return True, []

    with torch.no_grad():
        latents, _ = model.encode(seed_video)
        model.reset_state()
        current_context = latents

        slot_history = []

        for step in range(num_steps):
            pred, slots = model.dynamics(current_context, None, model.prev_slots)
            model.prev_slots = slots

            slot_history.append(slots.clone())

            next_latent = pred[:, -1:]
            window = min(current_context.shape[1] + 1, model.config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

        # Analyze slot changes
        print(f"\nSlot evolution over {num_steps} steps:")
        print(f"  Slot shape: {slot_history[0].shape}")

        slot_diffs = []
        slot_norms = []
        for i in range(1, len(slot_history)):
            diff = (slot_history[i] - slot_history[i-1]).abs().mean().item()
            slot_diffs.append(diff)
            slot_norms.append(slot_history[i].norm().item())

        print(f"  Step  1-5 avg diff: {np.mean(slot_diffs[:5]):.6f}")
        print(f"  Step 6-10 avg diff: {np.mean(slot_diffs[5:10]):.6f}")
        print(f"  Step 11-20 avg diff: {np.mean(slot_diffs[10:]):.6f}")

        print(f"\nSlot norm over time:")
        print(f"  Step  1: {slot_norms[0]:.4f}")
        print(f"  Step 10: {slot_norms[9]:.4f}")
        print(f"  Step 20: {slot_norms[-1]:.4f}")

        # Check individual slots
        print(f"\nPer-slot analysis (step 1 vs step 20):")
        slot_1 = slot_history[0][0]  # [num_slots, slot_dim]
        slot_20 = slot_history[-1][0]
        for s in range(min(4, slot_1.shape[0])):
            slot_diff = (slot_20[s] - slot_1[s]).abs().mean().item()
            print(f"  Slot {s}: diff={slot_diff:.6f}, norm_1={slot_1[s].norm():.4f}, norm_20={slot_20[s].norm():.4f}")

        final_diff = slot_diffs[-1]
        if final_diff < 1e-6:
            print(f"\n[FAIL] Slots converge to fixed point (diff={final_diff:.8f})")
            return False, slot_history
        else:
            print(f"\n[PASS] Slots continue evolving (diff={final_diff:.6f})")
            return True, slot_history


def test_action_sensitivity(model, seed_video, device):
    """Test 4: Does changing actions change output?"""
    print("\n" + "="*70)
    print("TEST 4: ACTION SENSITIVITY")
    print("="*70)

    with torch.no_grad():
        latents, _ = model.encode(seed_video)

        # Generate with no action
        model.reset_state()
        pred_no_action, _ = model.dynamics(latents, None, None)

        # Generate with action 0
        model.reset_state()
        action_0 = torch.zeros(1, latents.shape[1], dtype=torch.long, device=device)
        action_0_encoded = model.encode_action(action_0)
        pred_action_0, _ = model.dynamics(latents, action_0_encoded, None)

        # Generate with action 1
        model.reset_state()
        action_1 = torch.ones(1, latents.shape[1], dtype=torch.long, device=device)
        action_1_encoded = model.encode_action(action_1)
        pred_action_1, _ = model.dynamics(latents, action_1_encoded, None)

        # Compare
        diff_no_vs_0 = (pred_no_action - pred_action_0).abs().mean().item()
        diff_no_vs_1 = (pred_no_action - pred_action_1).abs().mean().item()
        diff_0_vs_1 = (pred_action_0 - pred_action_1).abs().mean().item()

        print(f"Output difference with different actions:")
        print(f"  No action vs action 0: {diff_no_vs_0:.6f}")
        print(f"  No action vs action 1: {diff_no_vs_1:.6f}")
        print(f"  Action 0 vs action 1: {diff_0_vs_1:.6f}")

        if diff_0_vs_1 < 1e-5:
            print(f"\n[FAIL] Model ignores action input")
            return False
        else:
            print(f"\n[PASS] Model responds to action changes")
            return True


def test_collapse_onset(model, seed_video, device, max_steps=100):
    """Test 5: When exactly does collapse start?"""
    print("\n" + "="*70)
    print("TEST 5: COLLAPSE ONSET DETECTION")
    print("="*70)

    with torch.no_grad():
        latents, _ = model.encode(seed_video)
        model.reset_state()
        current_context = latents

        frame_history = []

        # Decode initial
        initial_frame = model.decode(latents[:, -1:])
        frame_history.append(initial_frame.clone())

        collapse_step = None

        for step in range(max_steps):
            if model.config.use_slots:
                pred, model.prev_slots = model.dynamics(current_context, None, model.prev_slots)
            else:
                pred = model.dynamics(current_context, None)

            next_latent = pred[:, -1:]

            # Decode to pixel space
            next_frame = model.decode(next_latent)
            frame_history.append(next_frame.clone())

            # Check similarity to previous frame
            if len(frame_history) >= 2:
                sim = F.cosine_similarity(
                    frame_history[-1].flatten(1),
                    frame_history[-2].flatten(1)
                ).item()

                if step % 10 == 0:
                    var = next_frame.var().item()
                    print(f"  Step {step:3d}: similarity={sim:.6f}, frame_var={var:.6f}")

                # Detect collapse (similarity > 0.9999 for 5 consecutive frames)
                if sim > 0.9999 and collapse_step is None:
                    collapse_step = step

            # Update context
            window = min(current_context.shape[1] + 1, model.config.window_size // 4)
            current_context = torch.cat([current_context, next_latent], dim=1)[:, -window:]

        if collapse_step is not None:
            print(f"\n[FAIL] Collapse detected at step {collapse_step}")
            return False, collapse_step
        else:
            print(f"\n[PASS] No collapse detected in {max_steps} steps")
            return True, None


def test_64_vs_256_comparison(device):
    """Test 6: Compare behavior at 64x64 vs 256x256."""
    print("\n" + "="*70)
    print("TEST 6: RESOLUTION COMPARISON (64 vs 256)")
    print("="*70)

    # Check if 64x64 checkpoint exists
    ckpt_64 = Path('checkpoints/genesis/best_genesis.pt')
    ckpt_256 = Path('checkpoints/genesis_256/best_genesis.pt')

    if not ckpt_64.exists():
        print("64x64 checkpoint not found, skipping comparison")
        return None

    # Load both models
    model_64, _ = load_model(str(ckpt_64), device)
    model_256, _ = load_model(str(ckpt_256), device)

    # Get seed videos at each resolution
    seed_64 = get_seed_video(model_64, device, num_frames=4)
    seed_256 = get_seed_video(model_256, device, num_frames=4)

    print(f"\n64x64 model:")
    with torch.no_grad():
        latents_64, _ = model_64.encode(seed_64)
        print(f"  Latent shape: {latents_64.shape}")
        print(f"  Latent std: {latents_64.std():.6f}")

        # Single step
        model_64.reset_state()
        pred_64, slots_64 = model_64.dynamics(latents_64, None, None)
        diff_64 = (pred_64[:, -1] - latents_64[:, -1]).abs().mean().item()
        print(f"  Single step latent diff: {diff_64:.6f}")

    print(f"\n256x256 model:")
    with torch.no_grad():
        latents_256, _ = model_256.encode(seed_256)
        print(f"  Latent shape: {latents_256.shape}")
        print(f"  Latent std: {latents_256.std():.6f}")

        # Single step
        model_256.reset_state()
        pred_256, slots_256 = model_256.dynamics(latents_256, None, None)
        diff_256 = (pred_256[:, -1] - latents_256[:, -1]).abs().mean().item()
        print(f"  Single step latent diff: {diff_256:.6f}")

    print(f"\nComparison:")
    print(f"  Latent std ratio (256/64): {latents_256.std() / latents_64.std():.4f}")
    print(f"  Step diff ratio (256/64): {diff_256 / diff_64:.4f}")

    if diff_256 < diff_64 * 0.1:
        print(f"\n[WARNING] 256x256 dynamics much less active than 64x64")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/genesis_256/best_genesis.pt')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    print("="*70)
    print("GENESIS MODE COLLAPSE DIAGNOSTIC")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load model
    model, config = load_model(args.checkpoint, args.device)
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Image size: {config.image_size}")
    print(f"Slot norm mode: {config.slot_norm_mode}")

    # Get seed video
    seed_video = get_seed_video(model, args.device, num_frames=4)
    print(f"Seed video: {seed_video.shape}")

    # Run tests
    results = {}

    results['tokenizer'], latents = test_tokenizer_roundtrip(model, seed_video, args.device)
    results['latent_dynamics'], _ = test_latent_dynamics(model, seed_video, args.device)
    results['slot_evolution'], _ = test_slot_evolution(model, seed_video, args.device)
    results['action_sensitivity'] = test_action_sensitivity(model, seed_video, args.device)
    results['collapse_onset'], collapse_step = test_collapse_onset(model, seed_video, args.device)

    # Compare resolutions
    test_64_vs_256_comparison(args.device)

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")

    # Root cause analysis
    print("\n" + "="*70)
    print("ROOT CAUSE ANALYSIS")
    print("="*70)

    if not results['tokenizer']:
        print("[LIKELY CAUSE] Tokenizer loses too much information at 256x256")
        print("  -> Investigate encoder depth, latent capacity")

    if not results['latent_dynamics']:
        print("[LIKELY CAUSE] Dynamics model converges to fixed point")
        print("  -> Latent predictions stop changing, need noise injection or different training")

    if not results['slot_evolution']:
        print("[LIKELY CAUSE] Slots freeze to fixed values")
        print("  -> Slot attention converges, need different normalization or reset mechanism")

    if not results['action_sensitivity']:
        print("[CONTRIBUTING] Model ignores actions")
        print("  -> Action conditioning ineffective, but not primary cause of collapse")

    if collapse_step is not None and collapse_step < 10:
        print(f"[SEVERE] Collapse happens very quickly (step {collapse_step})")
        print("  -> Fundamental architecture issue, not just training")
    elif collapse_step is not None:
        print(f"[MODERATE] Collapse happens gradually (step {collapse_step})")
        print("  -> May be fixable with longer training or noise injection")


if __name__ == '__main__':
    main()
