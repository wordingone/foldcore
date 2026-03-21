"""Deep investigation of tokenizer behavior at 64x64 vs 256x256.

Questions to answer:
1. Is the encoder losing information, or the decoder failing to reconstruct?
2. Where in the encoder/decoder pipeline does information loss happen?
3. Is there a bug in the resolution-aware implementation?
4. What's the actual information content of the latents?
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from genesis.genesis_model import Genesis, GenesisConfig, create_genesis
from genesis.tokenizer.motion import MotionAwareTokenizer, compute_spatial_downsample


def get_test_frames(resolution, num_frames=4):
    """Get test frames from dataset."""
    from genesis.pilot.stream_hf import create_streaming_loader
    loader = create_streaming_loader(
        dataset_name='jat',
        batch_size=1,
        seq_length=num_frames,
        image_size=resolution,
        game='atari-breakout',
    )
    batch = next(iter(loader))
    return batch['frames'].cuda()


def analyze_tokenizer_standalone(resolution):
    """Test tokenizer in isolation (no dynamics model)."""
    print(f"\n{'='*70}")
    print(f"TOKENIZER ANALYSIS: {resolution}x{resolution}")
    print('='*70)

    # Create tokenizer directly
    downsample = compute_spatial_downsample(resolution)
    print(f"Spatial downsample factor: {downsample}x")
    print(f"Expected latent size: {resolution // downsample}x{resolution // downsample}")

    tokenizer = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=resolution,
    ).cuda()

    print(f"Tokenizer latent_size: {tokenizer.latent_size}")
    print(f"Tokenizer params: {sum(p.numel() for p in tokenizer.parameters()):,}")

    # Get test frames
    frames = get_test_frames(resolution, num_frames=4)
    print(f"Input shape: {frames.shape}")
    print(f"Input stats: min={frames.min():.4f}, max={frames.max():.4f}, mean={frames.mean():.4f}, std={frames.std():.4f}")

    with torch.no_grad():
        # Encode
        latents, intermediates = tokenizer.encode(frames)
        print(f"\nLatent shape: {latents.shape}")
        print(f"Latent stats: min={latents.min():.4f}, max={latents.max():.4f}, mean={latents.mean():.4f}, std={latents.std():.4f}")

        # Check latent information content
        latent_entropy = estimate_entropy(latents)
        print(f"Latent entropy estimate: {latent_entropy:.4f} bits")

        # Decode
        reconstructed = tokenizer.decode(latents, (resolution, resolution))
        print(f"\nReconstructed shape: {reconstructed.shape}")

        # Reconstruction quality
        mse = F.mse_loss(reconstructed, frames).item()
        psnr = 10 * np.log10(1.0 / (mse + 1e-8))
        ssim = compute_ssim(frames, reconstructed)

        print(f"\nReconstruction quality:")
        print(f"  MSE: {mse:.6f}")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")

        # Per-frame analysis
        print(f"\nPer-frame reconstruction:")
        for t in range(frames.shape[1]):
            frame_mse = F.mse_loss(reconstructed[:, t], frames[:, t]).item()
            frame_psnr = 10 * np.log10(1.0 / (frame_mse + 1e-8))
            print(f"  Frame {t}: PSNR={frame_psnr:.2f} dB")

        # Inter-frame variance preservation
        orig_var = frames.var(dim=1).mean().item()
        recon_var = reconstructed.var(dim=1).mean().item()
        print(f"\nInter-frame variance:")
        print(f"  Original: {orig_var:.6f}")
        print(f"  Reconstructed: {recon_var:.6f}")
        print(f"  Preservation ratio: {recon_var/orig_var:.4f}")

        # Check if different input frames produce different latents
        print(f"\nLatent differentiation (do different frames produce different latents?):")
        for t in range(1, frames.shape[1]):
            latent_diff = (latents[:, t] - latents[:, 0]).abs().mean().item()
            frame_diff = (frames[:, t] - frames[:, 0]).abs().mean().item()
            print(f"  Frame 0 vs {t}: frame_diff={frame_diff:.6f}, latent_diff={latent_diff:.6f}, ratio={latent_diff/(frame_diff+1e-8):.4f}")

    return {
        'resolution': resolution,
        'psnr': psnr,
        'ssim': ssim,
        'variance_ratio': recon_var / orig_var,
        'latent_std': latents.std().item(),
    }


def analyze_encoder_stages(resolution):
    """Trace information through encoder stages."""
    print(f"\n{'='*70}")
    print(f"ENCODER STAGE ANALYSIS: {resolution}x{resolution}")
    print('='*70)

    tokenizer = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=resolution,
    ).cuda()

    frames = get_test_frames(resolution, num_frames=1)
    frame = frames[:, 0]  # Single frame [B, C, H, W]

    print(f"Input: {frame.shape}, std={frame.std():.4f}")

    # Trace through keyframe encoder
    encoder = tokenizer.keyframe_encoder
    x = frame

    with torch.no_grad():
        # Manual forward through stages
        stage_outputs = []

        if hasattr(encoder, 'stages'):
            # ResolutionAwareKeyframeEncoder
            for i, stage in enumerate(encoder.stages):
                x = stage(x)
                stage_outputs.append({
                    'stage': i,
                    'shape': x.shape,
                    'std': x.std().item(),
                    'min': x.min().item(),
                    'max': x.max().item(),
                })
                print(f"  Stage {i}: {x.shape}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")

            x = encoder.final(x)
            print(f"  Final: {x.shape}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")
        else:
            # Original encoder - trace manually
            print("  (Original encoder architecture)")
            x = encoder.conv1(frame)
            print(f"  conv1: {x.shape}, std={x.std():.4f}")
            x = encoder.conv2(x)
            print(f"  conv2: {x.shape}, std={x.std():.4f}")
            x = encoder.conv3(x)
            print(f"  conv3: {x.shape}, std={x.std():.4f}")
            x = encoder.final(x)
            print(f"  final: {x.shape}, std={x.std():.4f}")

    return stage_outputs


def analyze_decoder_stages(resolution):
    """Trace information through decoder stages."""
    print(f"\n{'='*70}")
    print(f"DECODER STAGE ANALYSIS: {resolution}x{resolution}")
    print('='*70)

    tokenizer = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=resolution,
    ).cuda()

    # Get a latent
    frames = get_test_frames(resolution, num_frames=1)
    with torch.no_grad():
        latents, _ = tokenizer.encode(frames)
        latent = latents[:, 0]  # Single frame latent

    print(f"Latent input: {latent.shape}, std={latent.std():.4f}")

    decoder = tokenizer.keyframe_decoder

    with torch.no_grad():
        x = latent

        if hasattr(decoder, 'initial'):
            x = decoder.initial(x)
            print(f"  initial: {x.shape}, std={x.std():.4f}")

        if hasattr(decoder, 'stages'):
            for i, stage in enumerate(decoder.stages):
                x = stage(x)
                print(f"  Stage {i}: {x.shape}, std={x.std():.4f}, range=[{x.min():.4f}, {x.max():.4f}]")

            x = decoder.final(x)
            print(f"  final: {x.shape}, std={x.std():.4f}")
        else:
            print("  (Original decoder architecture)")


def check_gradient_flow(resolution):
    """Check if gradients flow properly through the tokenizer."""
    print(f"\n{'='*70}")
    print(f"GRADIENT FLOW CHECK: {resolution}x{resolution}")
    print('='*70)

    tokenizer = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=resolution,
    ).cuda()

    frames = get_test_frames(resolution, num_frames=1)

    # Forward pass
    latents, _ = tokenizer.encode(frames)
    reconstructed = tokenizer.decode(latents, (resolution, resolution))

    # Compute loss
    loss = F.mse_loss(reconstructed, frames)
    loss.backward()

    # Check gradients
    print("Encoder gradients:")
    for name, param in tokenizer.keyframe_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  {name}: NO GRADIENT")

    print("\nDecoder gradients:")
    for name, param in tokenizer.keyframe_decoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"  {name}: NO GRADIENT")


def compare_trained_vs_untrained(resolution):
    """Compare trained tokenizer (from checkpoint) vs fresh initialization."""
    print(f"\n{'='*70}")
    print(f"TRAINED vs UNTRAINED COMPARISON: {resolution}x{resolution}")
    print('='*70)

    # Fresh tokenizer
    fresh_tokenizer = MotionAwareTokenizer(
        in_channels=3,
        keyframe_channels=8,
        motion_channels=4,
        residual_channels=4,
        hidden_channels=64,
        keyframe_interval=8,
        image_size=resolution,
    ).cuda()

    # Load trained model
    ckpt_path = f'checkpoints/genesis_256/best_genesis.pt' if resolution == 256 else 'checkpoints/genesis/best_genesis.pt'
    if not Path(ckpt_path).exists():
        print(f"Checkpoint {ckpt_path} not found, skipping")
        return

    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    config = ckpt['config']

    trained_model = Genesis(config).cuda()
    trained_model.load_state_dict(ckpt['model_state_dict'])
    trained_tokenizer = trained_model.tokenizer

    frames = get_test_frames(resolution, num_frames=4)

    with torch.no_grad():
        # Fresh tokenizer
        fresh_latents, _ = fresh_tokenizer.encode(frames)
        fresh_recon = fresh_tokenizer.decode(fresh_latents, (resolution, resolution))
        fresh_mse = F.mse_loss(fresh_recon, frames).item()
        fresh_psnr = 10 * np.log10(1.0 / (fresh_mse + 1e-8))

        # Trained tokenizer
        trained_latents, _ = trained_tokenizer.encode(frames)
        trained_recon = trained_tokenizer.decode(trained_latents, (resolution, resolution))
        trained_mse = F.mse_loss(trained_recon, frames).item()
        trained_psnr = 10 * np.log10(1.0 / (trained_mse + 1e-8))

        print(f"Fresh tokenizer:   PSNR={fresh_psnr:.2f} dB, latent_std={fresh_latents.std():.4f}")
        print(f"Trained tokenizer: PSNR={trained_psnr:.2f} dB, latent_std={trained_latents.std():.4f}")

        if trained_psnr < fresh_psnr:
            print("\n[WARNING] Trained tokenizer is WORSE than untrained!")
            print("This suggests training corrupted the tokenizer, or reconstruction wasn't optimized.")


def estimate_entropy(tensor):
    """Rough entropy estimate of tensor values."""
    # Discretize to 256 bins
    vals = tensor.flatten().cpu().numpy()
    vals_norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
    hist, _ = np.histogram(vals_norm, bins=256, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(256)
    return entropy * 8  # Scale to bits


def compute_ssim(img1, img2):
    """Simple SSIM approximation."""
    # Flatten spatial dims
    img1_flat = img1.flatten(2)
    img2_flat = img2.flatten(2)

    mu1 = img1_flat.mean(dim=2)
    mu2 = img2_flat.mean(dim=2)
    sigma1 = img1_flat.std(dim=2)
    sigma2 = img2_flat.std(dim=2)

    # Covariance
    cov = ((img1_flat - mu1.unsqueeze(2)) * (img2_flat - mu2.unsqueeze(2))).mean(dim=2)

    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu1*mu2 + c1) * (2*cov + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))

    return ssim.mean().item()


def main():
    print("="*70)
    print("TOKENIZER DEEP INVESTIGATION")
    print("="*70)

    # Compare 64x64 vs 256x256
    results_64 = analyze_tokenizer_standalone(64)
    results_256 = analyze_tokenizer_standalone(256)

    # Detailed stage analysis
    analyze_encoder_stages(64)
    analyze_encoder_stages(256)

    analyze_decoder_stages(64)
    analyze_decoder_stages(256)

    # Compare trained vs untrained
    compare_trained_vs_untrained(64)
    compare_trained_vs_untrained(256)

    # Summary
    print("\n" + "="*70)
    print("INVESTIGATION SUMMARY")
    print("="*70)

    print(f"\n{'Metric':<25} {'64x64':>12} {'256x256':>12} {'Ratio':>12}")
    print("-"*65)
    print(f"{'PSNR (dB)':<25} {results_64['psnr']:>12.2f} {results_256['psnr']:>12.2f} {results_256['psnr']/results_64['psnr']:>12.2f}")
    print(f"{'SSIM':<25} {results_64['ssim']:>12.4f} {results_256['ssim']:>12.4f} {results_256['ssim']/results_64['ssim']:>12.2f}")
    print(f"{'Variance preservation':<25} {results_64['variance_ratio']:>12.4f} {results_256['variance_ratio']:>12.4f} {results_256['variance_ratio']/results_64['variance_ratio']:>12.2f}")
    print(f"{'Latent std':<25} {results_64['latent_std']:>12.4f} {results_256['latent_std']:>12.4f} {results_256['latent_std']/results_64['latent_std']:>12.2f}")

    print("\n" + "="*70)
    print("ROOT CAUSE HYPOTHESES")
    print("="*70)

    if results_256['psnr'] < results_64['psnr'] - 5:
        print("\n[FINDING] 256x256 reconstruction significantly worse than 64x64")

        if results_256['variance_ratio'] < results_64['variance_ratio'] * 0.6:
            print("[HYPOTHESIS] Encoder over-smooths at 256x256 - loses high-frequency detail")

        if results_256['latent_std'] < results_64['latent_std'] * 0.8:
            print("[HYPOTHESIS] Latent space is under-utilized at 256x256")


if __name__ == '__main__':
    main()
