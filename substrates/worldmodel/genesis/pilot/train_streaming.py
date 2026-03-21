"""Train Genesis with True Streaming - No Full Download.

Streams data directly from HuggingFace, processes shard by shard.
Supports: JAT (Atari), 1X (Humanoid Robot), and other HF datasets.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time

from genesis.pilot.stream_hf import (
    create_streaming_loader,
    StreamingEpochManager,
    JATStreamDataset,
    HFBinaryStreamDataset,
)
from genesis.pilot.action_model import ActionConditionedModel


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_step(model, batch, optimizer, device, use_tokens=False):
    """Single training step."""
    model.train()

    if use_tokens:
        # Token-based input (1X dataset)
        tokens = batch["tokens"].to(device)
        # For now, skip token decoding - train on token prediction
        # This would need a token decoder for full training
        return {"loss": 0.0, "skipped": True}
    else:
        # Frame-based input (JAT, OpenX)
        frames = batch["frames"].to(device)  # [B, T, C, H, W]
        actions = batch.get("actions")
        if actions is not None:
            actions = actions.to(device)

    targets = frames[:, 2:]

    # Action inputs (predict frame t+2 from frames t, t+1 and action t+1)
    if actions is not None and actions.dim() >= 2:
        # Discrete actions -> one-hot or embedding handled by model
        action_inputs = actions[:, 1:-1]  # [B, T-2]
    else:
        action_inputs = None

    # Forward pass
    preds = model(frames, actions=action_inputs, rollout_steps=1)

    # Loss
    loss = F.mse_loss(preds, targets)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {"loss": loss.item()}


def evaluate(model, loader, device, max_batches=50, use_tokens=False):
    """Evaluate model on streaming data."""
    model.eval()

    total_mse = 0
    total_psnr = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break

            if use_tokens:
                continue

            frames = batch["frames"].to(device)
            actions = batch.get("actions")
            if actions is not None:
                actions = actions.to(device)

            targets = frames[:, 2:]
            action_inputs = actions[:, 1:-1] if actions is not None and actions.dim() >= 2 else None

            preds = model(frames, actions=action_inputs)

            mse = F.mse_loss(preds, targets)
            psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

            total_mse += mse.item()
            total_psnr += psnr.item()
            num_batches += 1

    if num_batches == 0:
        return {"mse": 0, "psnr": 0}

    return {
        "mse": total_mse / num_batches,
        "psnr": total_psnr / num_batches,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Genesis with Streaming Data")
    parser.add_argument("--dataset", default="jat",
                       choices=["jat", "1x"],
                       help="Dataset to stream")
    parser.add_argument("--game", default="atari-breakout",
                       help="Game for JAT dataset")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of epochs")
    parser.add_argument("--samples-per-epoch", type=int, default=5000,
                       help="Samples per epoch (streaming has no fixed size)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seq-length", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="checkpoints/streaming")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    print("=" * 70)
    print("GENESIS STREAMING TRAINING (NO DOWNLOAD)")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    if args.dataset == "jat":
        print(f"Game: {args.game}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Samples/epoch: {args.samples_per_epoch}")
    print("=" * 70)

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Determine if using tokens or frames
    use_tokens = args.dataset == "1x"

    # Create streaming data loader
    print("\nCreating streaming data loader...")
    if args.dataset == "jat":
        loader = create_streaming_loader(
            dataset_name="jat",
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            game=args.game,
            image_size=args.image_size,
        )
        action_dim = 18  # Atari has up to 18 discrete actions
        num_actions = 18
        continuous_actions = False
    else:  # 1x
        loader = create_streaming_loader(
            dataset_name="1x",
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            version="v2.0",
        )
        action_dim = 25
        num_actions = None
        continuous_actions = True

    # Create model
    print("\nCreating model...")
    model = ActionConditionedModel(
        base_channels=48,
        num_slots=8,
        slot_dim=64,
        slot_decay=0.95,
        num_actions=num_actions if not continuous_actions else 5,
        continuous_actions=continuous_actions,
    ).to(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Epoch manager
    epoch_mgr = StreamingEpochManager(
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
    )

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING (streaming from HuggingFace)")
    print("=" * 70)

    best_psnr = 0
    batch_idx = 0
    epoch_losses = []
    start_time = time.time()

    for batch in loader:
        # Training step
        result = train_step(model, batch, optimizer, device, use_tokens=use_tokens)

        if not result.get("skipped", False):
            epoch_losses.append(result["loss"])

        # Progress tracking
        epoch_done = epoch_mgr.step()
        batch_idx += 1

        # Logging
        if batch_idx % args.log_every == 0:
            elapsed = time.time() - start_time
            samples_per_sec = epoch_mgr.total_samples / elapsed
            avg_loss = sum(epoch_losses[-100:]) / max(len(epoch_losses[-100:]), 1)
            print(f"Batch {batch_idx}: loss={avg_loss:.6f}, "
                  f"progress={epoch_mgr.progress:.1%}, "
                  f"epoch={epoch_mgr.current_epoch + 1}, "
                  f"speed={samples_per_sec:.1f} samples/s")

        # End of epoch
        if epoch_done:
            avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            print(f"\n--- Epoch {epoch_mgr.current_epoch} Complete ---")
            print(f"Average loss: {avg_loss:.6f}")

            # Quick evaluation
            if not use_tokens:
                eval_loader = create_streaming_loader(
                    dataset_name=args.dataset,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    game=args.game if args.dataset == "jat" else None,
                    image_size=args.image_size,
                    shuffle=False,
                )
                metrics = evaluate(model, eval_loader, device, max_batches=30, use_tokens=use_tokens)
                print(f"Val MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f} dB")

                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    torch.save({
                        "epoch": epoch_mgr.current_epoch,
                        "model_state_dict": model.state_dict(),
                        "psnr": best_psnr,
                        "dataset": args.dataset,
                        "game": args.game if args.dataset == "jat" else None,
                    }, save_dir / f"best_{args.dataset}.pt")
                    print(f"  -> Saved (best PSNR)")

            epoch_losses = []

            if epoch_mgr.current_epoch >= args.epochs:
                break

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total samples: {epoch_mgr.total_samples:,}")
    print(f"Total time: {time.time() - start_time:.1f}s")
    if not use_tokens:
        print(f"Best PSNR: {best_psnr:.2f} dB")
    print(f"Checkpoints: {save_dir}")


if __name__ == "__main__":
    main()
