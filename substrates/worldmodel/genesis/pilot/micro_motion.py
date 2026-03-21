"""Micro Motion: Test motion vector prediction (video codec insight).

Video codecs predict WHERE objects move, not pixel values directly.
This separates "what" (object appearance) from "where" (motion).

Hypothesis: Predicting motion vectors + warping is more efficient than
predicting pixel deltas directly.

Risk: High. This works for compression (encoder sees future), unclear for prediction.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from genesis.pilot.micro_data import MicroDataset, get_recovery_frames


def spatial_transformer(x, flow):
    """Warp image x according to flow field.

    Args:
        x: [B, C, H, W] - source image
        flow: [B, 2, H, W] - displacement field (dx, dy)

    Returns:
        warped: [B, C, H, W]
    """
    B, C, H, W = x.shape

    # Create base grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, H, device=x.device),
        torch.linspace(-1, 1, W, device=x.device),
        indexing='ij'
    )
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 2]

    # Add flow (normalized to [-1, 1])
    flow_permuted = flow.permute(0, 2, 3, 1)  # [B, H, W, 2]
    # Scale flow to grid coordinates (flow is in pixels, grid is [-1, 1])
    flow_scaled = flow_permuted * 2 / torch.tensor([W, H], device=x.device).float()

    sample_grid = grid + flow_scaled
    sample_grid = sample_grid.clamp(-1, 1)

    # Sample
    warped = F.grid_sample(x, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)
    return warped


class MotionPredictor(nn.Module):
    """Predict motion vectors, then warp previous frame.

    Architecture:
    1. Encode two frames
    2. Predict flow field (where each pixel moves)
    3. Warp frame t to get prediction for t+1
    4. Predict residual (what warping missed)
    """

    def __init__(self, channels=32):
        super().__init__()

        # Encode frame pair
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
        )

        # Predict flow (2 channels: dx, dy)
        self.flow_head = nn.Sequential(
            nn.Conv2d(channels, channels//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels//2, 2, 3, padding=1),
            nn.Tanh(),  # Limit displacement to reasonable range
        )
        self.flow_scale = 4.0  # Max displacement in pixels

        # Residual prediction (what warping missed)
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels + 1, channels//2, 3, padding=1),  # +1 for warped
            nn.GELU(),
            nn.Conv2d(channels//2, 1, 3, padding=1),
            nn.Tanh(),
        )
        self.residual_scale = 0.2

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []

        for t in range(T - 2):
            prev = frames[:, t]
            curr = frames[:, t + 1]

            # Encode
            pair = torch.cat([curr, prev], dim=1)
            feat = self.encoder(pair)

            # Predict flow
            flow = self.flow_head(feat) * self.flow_scale

            # Warp current frame to predict next
            warped = spatial_transformer(curr, flow)

            # Predict residual
            residual_input = torch.cat([feat, warped], dim=1)
            residual = self.residual_head(residual_input) * self.residual_scale

            # Combine
            pred = torch.clamp(warped + residual, 0, 1)
            preds.append(pred)

        return torch.stack(preds, dim=1)


class SlotMotionPredictor(nn.Module):
    """Combine slot attention with motion prediction.

    Slots track WHAT objects are.
    Motion predicts WHERE they go.
    """

    def __init__(self, channels=32, num_slots=4, slot_dim=32):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 4, stride=4),  # 16->4
            nn.GELU(),
        )

        # Slot attention for object tracking
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.1)
        self.to_slot = nn.Linear(channels, slot_dim)
        self.q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Slot -> motion per slot
        self.slot_to_motion = nn.Sequential(
            nn.Linear(slot_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # dx, dy per slot
            nn.Tanh(),
        )
        self.motion_scale = 4.0

        # Slot -> mask (which pixels belong to which slot)
        self.slot_to_mask = nn.Sequential(
            nn.Linear(slot_dim, channels * 16),
        )

        # Upsample mask
        self.mask_upsample = nn.ConvTranspose2d(num_slots, num_slots, 4, stride=4)

        # Residual
        self.residual = nn.Sequential(
            nn.Conv2d(1, channels//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels//2, 1, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, frames):
        B, T, C, H, W = frames.shape
        preds = []
        slots = self.slot_mu.expand(B, -1, -1).clone()

        for t in range(T - 2):
            prev = frames[:, t]
            curr = frames[:, t + 1]

            # Encode
            pair = torch.cat([curr, prev], dim=1)
            feat = self.encoder(pair)  # [B, C, 4, 4]
            feat_flat = feat.flatten(2).transpose(1, 2)  # [B, 16, C]
            feat_slot = self.to_slot(feat_flat)  # [B, 16, slot_dim]

            # Slot attention
            k = self.k(feat_slot)
            v = self.v(feat_slot)
            for _ in range(3):
                q = self.q(slots)
                attn = torch.einsum('bkd,bnd->bkn', q, k) / (self.slot_dim ** 0.5)
                attn = F.softmax(attn, dim=1)
                updates = torch.einsum('bkn,bnd->bkd', attn, v)
                slots = self.gru(
                    updates.reshape(-1, self.slot_dim),
                    slots.reshape(-1, self.slot_dim)
                ).reshape(B, self.num_slots, self.slot_dim)

            # Per-slot motion
            motion = self.slot_to_motion(slots) * self.motion_scale  # [B, K, 2]

            # Per-slot mask at low res
            masks_lowres = self.slot_to_mask(slots)  # [B, K, C*16]
            masks_lowres = masks_lowres.reshape(B, self.num_slots, -1, 4, 4)
            masks_lowres = masks_lowres.mean(dim=2)  # [B, K, 4, 4]

            # Upsample masks
            masks = self.mask_upsample(masks_lowres)  # [B, K, 16, 16]
            masks = F.softmax(masks, dim=1)  # Compete

            # Apply per-slot motion
            warped = torch.zeros_like(curr)
            for k in range(self.num_slots):
                # Flow for this slot
                flow_k = motion[:, k].reshape(B, 2, 1, 1).expand(-1, -1, H, W)
                warped_k = spatial_transformer(curr, flow_k)
                warped = warped + masks[:, k:k+1] * warped_k

            # Residual
            res = self.residual(warped - curr) * 0.1
            pred = torch.clamp(warped + res, 0, 1)
            preds.append(pred)

        return torch.stack(preds, dim=1)


def train_and_eval(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        model.train()
        for frames, _ in train_loader:
            frames = frames.to(device)
            preds = model(frames)
            loss = F.mse_loss(preds, frames[:, 2:])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    total_mse = 0
    recovery_mse = 0
    n = 0
    nr = 0

    with torch.no_grad():
        for frames, masks in val_loader:
            frames = frames.to(device)
            masks = masks.to(device)
            preds = model(frames)
            targets = frames[:, 2:]

            total_mse += F.mse_loss(preds, targets).item()
            n += 1

            recovery = get_recovery_frames(masks)[:, 2:]
            if recovery.any():
                B, T, C, H, W = preds.shape
                r_exp = recovery.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,C,H,W)
                r_p = preds[r_exp].reshape(-1)
                r_t = targets[r_exp].reshape(-1)
                if len(r_p) > 0:
                    recovery_mse += F.mse_loss(r_p, r_t).item()
                    nr += 1

    return {
        'mse': total_mse / max(n, 1),
        'recovery_mse': recovery_mse / max(nr, 1),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train-size', type=int, default=500)
    parser.add_argument('--val-size', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_data = MicroDataset(num_sequences=args.train_size, seed=42)
    val_data = MicroDataset(num_sequences=args.val_size, seed=1000)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    print(f"Device: {args.device}\n")

    models = {
        'Motion': MotionPredictor(channels=32),
        'Slot+Motion': SlotMotionPredictor(channels=32, num_slots=4, slot_dim=32),
    }

    print("=" * 60)
    print("MOTION PREDICTION (Video Codec Insight)")
    print("=" * 60)

    # Reference from previous experiments
    print("\nReference (from efficiency_frontier.py):")
    print("  Baseline:    Recovery=0.0463")
    print("  Delta:       Recovery=0.0013 (+97.2%)")
    print("  Slot+Delta:  Recovery=0.0115 (+75.1%)")

    print("\nMotion-based models:")
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        metrics = train_and_eval(model, train_loader, val_loader, args.epochs, args.lr, args.device)
        improvement = (0.0463 - metrics['recovery_mse']) / 0.0463 * 100
        print(f"  {name}: Params={params:,} MSE={metrics['mse']:.4f} Recovery={metrics['recovery_mse']:.4f} ({improvement:+.1f}%)")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print("\nMotion prediction separates WHAT (appearance) from WHERE (motion).")
    print("If it beats delta prediction, the video codec insight transfers.")
    print("If not, direct delta prediction is better for prediction tasks.")
    print("=" * 60)


if __name__ == '__main__':
    main()
