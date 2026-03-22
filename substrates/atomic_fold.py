#!/usr/bin/env python3
"""
Atomic Fold — The unified equation.

All four separations collapsed:
  - Training IS inference (same softmax weights for readout and update)
  - Storage IS readout (reading the codebook writes it)
  - Memory IS generation (iterate without input)
  - Confidence IS the energy landscape (kappa shapes attention)

One method: step(). It reads, writes, classifies, and learns in one pass.
"""

import torch
import torch.nn.functional as F


class AtomicFold:
    def __init__(self, d, tau=0.1, lr=0.01, spawn_energy=None, merge_thresh=0.95):
        self.d = d
        self.tau = tau
        self.lr = lr
        self.merge_thresh = merge_thresh
        # Energy threshold for spawning. If None, derived from first N inputs.
        self.spawn_energy = spawn_energy
        self.V = torch.empty(0, d, device='cuda')       # prototypes on unit sphere
        self.labels = []                                  # one label per prototype
        self.kappa = torch.empty(0, device='cuda')       # per-prototype confidence
        self.n = 0

    def energy(self, r):
        """Hopfield energy: how well does V collectively explain r?"""
        if self.n == 0:
            return float('inf')
        sims = self.V @ r
        return -torch.logsumexp(self.kappa * sims / self.tau, dim=0).item()

    def step(self, r, label=None):
        """
        THE operation. Read IS write. Returns predicted label.

        With label: supervised update (attract same-class, repel different-class).
        Without label: unsupervised update (reduce reconstruction error).
        Either way, the attention weights that produce the output ARE the update signal.
        """
        r = F.normalize(r, dim=0)

        # Birth: empty codebook
        if self.n == 0:
            self._spawn(r, label)
            return label

        # --- THE READ (which is also THE WRITE) ---

        sims = self.V @ r                                      # (n,) cosine similarities
        weighted_sims = self.kappa * sims                      # confidence-weighted
        weights = F.softmax(weighted_sims / self.tau, dim=0)   # attention over codebook

        # Output: class vote weighted by attention AND confidence
        pred = self._vote(weights)

        # Energy: collective coverage check
        E = -torch.logsumexp(weighted_sims / self.tau, dim=0).item()

        # Spawn if energy too high (the field doesn't explain this input)
        if self.spawn_energy is not None and E > self.spawn_energy:
            self._spawn(r, label)
            return pred

        # --- THE WRITE (which was THE READ) ---

        if label is not None:
            # Supervised: target concentrates weight on same-class prototypes
            mask = torch.tensor([1.0 if l == label else 0.0 for l in self.labels],
                                device='cuda')
            target = weights * mask
            t_sum = target.sum()
            if t_sum > 1e-10:
                target = target / t_sum
            else:
                # No same-class prototypes exist yet — spawn
                self._spawn(r, label)
                return pred

            error = weights - target  # positive = wrong-class (repel), negative = right-class (attract)
        else:
            # Unsupervised: error from reconstruction
            output = weights @ self.V         # (d,) weighted reconstruction
            residual = r - output             # (d,) what V missed
            # Per-prototype error proportional to participation and alignment with residual
            error = -weights * (self.V @ residual)  # attract all toward reducing residual

        # Position update on tangent space of unit sphere
        perp = r.unsqueeze(0) - sims.unsqueeze(1) * self.V    # (n, d) perpendicular components
        self.V -= self.lr * error.unsqueeze(1) * perp          # gradient step
        self.V = F.normalize(self.V, dim=1)                    # back to sphere

        # Confidence update: kappa learns each prototype's reliability
        if label is not None:
            # Correct-class prototypes that fired: increase confidence
            # Wrong-class prototypes that fired: decrease confidence
            self.kappa -= self.lr * error * sims.abs()
            self.kappa = self.kappa.clamp(min=0.1)

        return pred

    def classify(self, r):
        """Read-only classification (for evaluation). Does NOT update."""
        if self.n == 0:
            return None
        r = F.normalize(r, dim=0)
        sims = self.V @ r
        weights = F.softmax(self.kappa * sims / self.tau, dim=0)
        return self._vote(weights)

    def generate(self, seed, steps=10):
        """Generation: iterate step() without input. r_{t+1} = reconstruct(r_t)."""
        r = F.normalize(seed, dim=0)
        trajectory = [r.clone()]
        for _ in range(steps):
            sims = self.V @ r
            weights = F.softmax(self.kappa * sims / self.tau, dim=0)
            r = F.normalize(weights @ self.V, dim=0)
            trajectory.append(r.clone())
        return trajectory

    def _spawn(self, r, label):
        """Birth: add a new prototype."""
        self.V = torch.cat([self.V, r.unsqueeze(0)])
        self.labels.append(label)
        self.kappa = torch.cat([self.kappa, torch.ones(1, device='cuda')])
        self.n += 1
        self._try_merge(self.n - 1)

    def _try_merge(self, idx):
        """Compression: merge if new prototype is redundant."""
        if idx == 0:
            return
        sims = self.V[idx] @ self.V[:idx].T
        best_i = sims.argmax().item()
        if sims[best_i].abs() > self.merge_thresh and self.labels[best_i] == self.labels[idx]:
            # Fuse: average positions, sum confidences
            self.V[best_i] = F.normalize(self.V[best_i] + self.V[idx], dim=0)
            self.kappa[best_i] = self.kappa[best_i] + self.kappa[idx]
            # Remove the new one
            self.V = torch.cat([self.V[:idx], self.V[idx+1:]])
            self.kappa = torch.cat([self.kappa[:idx], self.kappa[idx+1:]])
            del self.labels[idx]
            self.n -= 1

    def _vote(self, weights):
        """Soft class vote."""
        per_class = {}
        for i, l in enumerate(self.labels):
            per_class[l] = per_class.get(l, 0.0) + weights[i].item()
        return max(per_class, key=per_class.get)

    def stats(self):
        """Current state summary."""
        return {
            'n_prototypes': self.n,
            'kappa_mean': self.kappa.mean().item() if self.n > 0 else 0,
            'kappa_std': self.kappa.std().item() if self.n > 1 else 0,
            'kappa_min': self.kappa.min().item() if self.n > 0 else 0,
            'kappa_max': self.kappa.max().item() if self.n > 0 else 0,
        }
