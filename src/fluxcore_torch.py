#!/usr/bin/env python3
"""
FluxCore TorchCodebook — CUDA-accelerated codebook layer.

Same algorithm as NumpyCodebook (and the codebook layer of fluxcore_manytofew.py):
  - Spawn: when max cosine similarity < spawn_thresh, add normalize(r)
  - Update: additive rule — v_winner += lr * r, then normalize
  - Merge: at spawn time, fuse if |cos(v_new, v_best)| > merge_thresh

This module handles ONLY the codebook layer — no matrix RK dynamics.
It is the standard GPU path for all CIFAR-100 / large-scale experiments.

Device: automatically uses CUDA if available, else CPU.
"""

import torch
import torch.nn.functional as F


class TorchCodebook:
    """
    Fold codebook on GPU. All vectors stored as (N, d) float32 CUDA tensor.

    Parameters
    ----------
    d : int
        Input dimensionality.
    spawn_thresh : float
        Spawn threshold. Spawn if max_sim < spawn_thresh.
    merge_thresh : float
        Merge threshold. Fuse if |cos(v_new, v_best)| > merge_thresh at spawn time.
    lr : float
        Additive update rate.
    device : str or torch.device
        Target device. Default 'cuda' if available, else 'cpu'.
    """

    def __init__(self, d, spawn_thresh=0.95, merge_thresh=0.95, lr=0.015,
                 device=None):
        self.d            = d
        self.spawn_thresh = spawn_thresh
        self.merge_thresh = merge_thresh
        self.lr           = lr
        self.device       = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        # Codebook: (N, d) float32 on device. Grows via append.
        self.vectors  = torch.empty((0, d), dtype=torch.float32, device=self.device)
        self.labels   = []          # parallel list, CPU
        self.n_spawned = 0
        self.n_merged  = 0

    def step(self, r, label=None):
        """
        Process one input vector r (1-D tensor or array, length d).

        Parameters
        ----------
        r : torch.Tensor or array-like, shape (d,)
        label : any, optional

        Returns
        -------
        winner : int — index of the matched/spawned codebook vector
        """
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=self.device)
        else:
            r = r.to(self.device).float()

        r_norm = F.normalize(r.unsqueeze(0), dim=1).squeeze(0)  # unit vector

        if len(self.vectors) == 0:
            return self._spawn(r_norm, label)

        sims    = self.vectors @ r_norm          # (N,) cosines (vectors already unit)
        winner  = int(sims.argmax())
        max_sim = float(sims[winner])

        if max_sim < self.spawn_thresh:
            return self._spawn(r_norm, label)

        # Additive update, then renormalize
        v = self.vectors[winner] + self.lr * r_norm
        self.vectors[winner] = v / (v.norm() + 1e-15)
        return winner

    def _spawn(self, r_norm, label):
        """Add new unit vector; merge if too close to an existing one."""
        if len(self.vectors) > 0:
            abs_sims = (self.vectors @ r_norm).abs()  # (N,)
            best_i   = int(abs_sims.argmax())
            if float(abs_sims[best_i]) > self.merge_thresh:
                fused = self.vectors[best_i] + r_norm
                self.vectors[best_i] = fused / (fused.norm() + 1e-15)
                self.n_merged += 1
                return best_i

        self.vectors = torch.cat([self.vectors, r_norm.unsqueeze(0)], dim=0)
        self.labels.append(label)
        self.n_spawned += 1
        return len(self.vectors) - 1

    def classify_batch(self, X, k=1):
        """
        Classify a batch of inputs.

        Parameters
        ----------
        X : torch.Tensor or array-like, shape (n, d)
        k : int — k-NN majority vote (k=1 is nearest prototype)

        Returns
        -------
        preds : list of predicted labels, length n
        """
        if len(self.vectors) == 0:
            return [None] * (len(X) if hasattr(X, '__len__') else 1)

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(self.device).float()

        # X may not be unit-normalized — normalize for cosine sim
        X_n = F.normalize(X, dim=1)           # (n, d)
        sims = X_n @ self.vectors.T            # (n, N)

        if k == 1:
            winners = sims.argmax(dim=1).tolist()
            return [self.labels[w] for w in winners]

        # k-NN majority vote
        top_k = sims.topk(k, dim=1).indices   # (n, k)
        preds = []
        for i in range(len(X)):
            votes = [self.labels[j] for j in top_k[i].tolist()]
            preds.append(max(set(votes), key=votes.count))
        return preds

    def step_batch_train(self, X, Y):
        """
        Train on a batch sequentially (one sample at a time).
        Used for task training loops.

        Parameters
        ----------
        X : array-like, shape (n, d)
        Y : array-like, shape (n,)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, device='cpu')

        X_n = F.normalize(X, dim=1)
        for i in range(len(X_n)):
            self.step(X_n[i], label=int(Y[i]))

    def __len__(self):
        return len(self.vectors)

    def __repr__(self):
        return (f"TorchCodebook(d={self.d}, n={len(self)}, "
                f"spawn_thresh={self.spawn_thresh}, device={self.device})")
