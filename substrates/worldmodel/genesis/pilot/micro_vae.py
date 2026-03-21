"""Micro VAE: Minimal latent tokenizer for 16x16 frames.

Compresses 16x16 -> 4x4 latent (16x compression).
Target: ~20K params for fair comparison.
"""

import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F


class MicroVAE(nn.Module):
    """Minimal VAE for 16x16 grayscale images.

    Architecture:
    - Encoder: 16x16 -> 4x4 latent
    - Decoder: 4x4 latent -> 16x16
    - Latent dim: 8 channels at 4x4 = 128 tokens
    """

    def __init__(self, latent_channels=8):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: 16x16 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # 4x4
            nn.GELU(),
        )

        # Latent projection (mu and logvar)
        self.to_mu = nn.Conv2d(32, latent_channels, 1)
        self.to_logvar = nn.Conv2d(32, latent_channels, 1)

        # Decoder: 4x4 -> 16x16
        self.from_latent = nn.Conv2d(latent_channels, 32, 1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 8x8
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 16x16
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode to latent distribution parameters.

        Args:
            x: [B, 1, 16, 16]

        Returns:
            mu: [B, latent_channels, 4, 4]
            logvar: [B, latent_channels, 4, 4]
        """
        h = self.encoder(x)
        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample from latent distribution."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        """Decode from latent.

        Args:
            z: [B, latent_channels, 4, 4]

        Returns:
            x_recon: [B, 1, 16, 16]
        """
        h = self.from_latent(z)
        return self.decoder(h)

    def forward(self, x):
        """Full forward pass.

        Args:
            x: [B, 1, 16, 16]

        Returns:
            x_recon: [B, 1, 16, 16]
            mu: [B, latent_channels, 4, 4]
            logvar: [B, latent_channels, 4, 4]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def get_latent(self, x):
        """Get deterministic latent (for downstream use).

        Args:
            x: [B, 1, 16, 16]

        Returns:
            z: [B, latent_channels, 4, 4]
        """
        mu, _ = self.encode(x)
        return mu

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def vae_loss(x, x_recon, mu, logvar, beta=0.001):
    """VAE loss = reconstruction + KL divergence."""
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


if __name__ == '__main__':
    print("Testing MicroVAE...")

    model = MicroVAE(latent_channels=8)
    print(f"Parameters: {model.count_parameters():,}")

    # Test forward
    x = torch.randn(4, 1, 16, 16).sigmoid()
    x_recon, mu, logvar = model(x)

    print(f"Input: {x.shape}")
    print(f"Recon: {x_recon.shape}")
    print(f"Latent: {mu.shape}")

    # Test loss
    loss, recon, kl = vae_loss(x, x_recon, mu, logvar)
    print(f"Loss: {loss.item():.4f} (recon={recon.item():.4f}, kl={kl.item():.4f})")

    # Test latent extraction
    z = model.get_latent(x)
    print(f"Deterministic latent: {z.shape}")

    print("All tests passed!")
