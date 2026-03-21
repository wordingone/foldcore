"""Tests for pilot implementation."""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from genesis.pilot.baseline_model import PilotBaselineModel
from genesis.pilot.data import MovingMNISTOcclusion


def test_baseline_model():
    """Verify baseline model."""
    print("\n" + "="*60)
    print("Testing Baseline Model")
    print("="*60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = PilotBaselineModel().to(device)

    # Parameter count
    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    assert 20_000_000 <= params <= 30_000_000, f"Param count {params} outside target range"
    print("[OK] Parameter count in range [20M, 30M]")

    # Forward pass
    print("Running forward pass...")
    frames = torch.randn(1, 10, 3, 64, 64).to(device)
    with torch.no_grad():
        preds = model(frames)

    print(f"Input shape: {frames.shape}")
    print(f"Output shape: {preds.shape}")
    assert preds.shape == (1, 8, 3, 64, 64), f"Wrong output shape: {preds.shape}"
    print("[OK] Forward pass shape correct")

    # Gradients flow
    print("Testing backward pass...")
    frames_grad = torch.randn(1, 10, 3, 64, 64).to(device)
    preds_grad = model(frames_grad)
    loss = torch.nn.functional.mse_loss(preds_grad, torch.randn_like(preds_grad))
    loss.backward()
    print("[OK] Backward pass completes")

    # Check gradients exist
    has_grads = any(p.grad is not None for p in model.parameters())
    assert has_grads, "No gradients computed"
    print("[OK] Gradients flow through model")

    print("="*60)
    print("Baseline Model: PASS\n")


def test_dataset():
    """Verify dataset generation."""
    print("\n" + "="*60)
    print("Testing Dataset")
    print("="*60)

    print("Generating dataset (this may take a moment)...")
    dataset = MovingMNISTOcclusion(num_sequences=10, seq_length=20)

    print(f"Dataset size: {len(dataset)}")
    assert len(dataset) == 10
    print("[OK] Dataset size correct")

    seq = dataset[0]
    print(f"Sequence shape: {seq.shape}")
    assert seq.shape == (20, 3, 64, 64)
    print("[OK] Sequence shape correct")

    # Check value range
    assert seq.min() >= 0 and seq.max() <= 1, "Values outside [0, 1]"
    print("[OK] Values in [0, 1]")

    print("="*60)
    print("Dataset: PASS\n")


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "#"*60)
    print("# PILOT VERIFICATION")
    print("#"*60)

    try:
        test_baseline_model()
        test_dataset()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED")
        print("#"*60)
        print("\nNext steps:")
        print("1. python genesis/pilot/data.py  # Generate sample visualizations")
        print("2. python genesis/pilot/train.py --quick  # Quick smoke test")
        print("3. python genesis/pilot/train.py --epochs 20  # Full training")
        print("\n")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_all_tests()
