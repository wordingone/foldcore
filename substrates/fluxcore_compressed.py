#!/usr/bin/env python3
"""
FluxCore Compressed — Unified FluxCore+RK Equation

Unifies FluxCore's perception mechanism with the Reflexive Kernel's
generation mechanism into one update rule.

State: n cells, each M_i in R^(k x k)

Phi(M) = tanh(alpha*M + beta*M^2/k)                    # eigenform drive
Psi(Mi,Mj) = tanh(alpha*(Mi+Mj)/2 + beta*Mi*Mj/k)     # cross-application

ef_dist_i = ||Phi(M_i) - M_i|| / max(||M_i||, 1)
a_i = exp(-ef_dist_i^2 / sigma^2)                       # autonomy
S_i = ||R - M_i|| / (||M_i|| + eps)                    # surprise
lr_i = lr_base * (1 + k_s * S_i)                       # adaptive learning rate
w_ij = softmax(cosine(M_i, M_j) / tau)                  # coupling weights (i!=j)

dM_i = a_i * (Phi(M_i) - M_i)                          # eigenform drive
      + (1-a_i) * lr_i * (R - M_i)                     # perception
      + (1-a_i) * sum_j(w_ij * (Psi(M_i,M_j) - M_i))  # coupling

M_i += dt * dM_i + noise(k, 0.01)
M_i = clip(M_i, max_norm=3.0)

When r=None, skip perception drive entirely.

Zero dependencies beyond rk.py math helpers.
"""

import math
import random
import sys

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import (mzero, madd, msub, mscale, mmul, mtanh, frob,
                mcosine, mclip, mrand, meye)


class CompressedKernel:
    """
    Compressed FluxCore+RK kernel.
    n cells, each M_i in R^(k x k).
    Unified perception + generation update rule.
    """

    def __init__(self, n=8, k=4, d=64, seed=42, proj_seed=999,
                 alpha=1.2, beta=0.8, lr_base=0.08, k_s=20,
                 sigma=0.3, tau=0.3, dt=0.03, noise_scale=0.01,
                 max_norm=3.0):
        self.n = n
        self.k = k
        self.d = d
        self.alpha = alpha
        self.beta = beta
        self.lr_base = lr_base
        self.k_s = k_s
        self.sigma = sigma
        self.tau = tau
        self.dt = dt
        self.noise_scale = noise_scale
        self.max_norm = max_norm

        # Initialize cells
        random.seed(seed)
        self.cells = [mrand(k, 0.8) for _ in range(n)]
        self.step_count = 0

        # Build fixed projection matrix P in R^(k^2 x d)
        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

    def project(self, r):
        """Project r in R^d -> R in R^(k x k) via P @ r reshaped."""
        k = self.k
        flat = [sum(self.P[i][j] * r[j] for j in range(self.d))
                for i in range(k * k)]
        return [flat[i * k:(i + 1) * k] for i in range(k)]

    def phi(self, M):
        """Phi(M) = tanh(alpha*M + beta*M^2/k)"""
        linear = mscale(M, self.alpha)
        quadratic = mscale(mmul(M, M), self.beta / self.k)
        return mtanh(madd(linear, quadratic))

    def psi(self, Mi, Mj):
        """Psi(Mi,Mj) = tanh(alpha*(Mi+Mj)/2 + beta*Mi*Mj/k)"""
        avg = mscale(madd(Mi, Mj), self.alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), self.beta / self.k)
        return mtanh(madd(avg, prod))

    def eigenform_distance(self, M):
        """||Phi(M) - M|| / max(||M||, 1)"""
        return frob(msub(self.phi(M), M)) / max(frob(M), 1.0)

    def autonomy(self, M):
        d = self.eigenform_distance(M)
        return math.exp(-(d * d) / (self.sigma * self.sigma))

    def surprise(self, R, M):
        """S = ||R - M|| / (||M|| + eps)"""
        return frob(msub(R, M)) / (frob(M) + 1e-15)

    def step(self, r=None):
        """
        One update step. r is a d-dimensional input vector or None.
        Returns list of dM matrices (before applying to cells).
        """
        n, k = self.n, self.k
        cells = self.cells

        # Project input if provided
        R = self.project(r) if r is not None else None

        # Compute autonomy for each cell
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        # Compute surprise for each cell (only if signal present)
        surprises = None
        if R is not None:
            surprises = [self.surprise(R, cells[i]) for i in range(n)]

        # Coupling weights: w_ij = softmax(cosine(M_i, M_j) / tau), i!=j
        weights = []
        for i in range(n):
            raw = []
            for j in range(n):
                if i == j:
                    raw.append(-1e10)
                else:
                    raw.append(mcosine(cells[i], cells[j]) / self.tau)
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # Compute dM for each cell
        dMs = []
        for i in range(n):
            ai = alphas[i]
            Mi = cells[i]

            # Eigenform drive
            phi_M = self.phi(Mi)
            ef_drive = msub(phi_M, Mi)

            # Coupling drive
            cp_drive = mzero(k)
            for j in range(n):
                if i == j:
                    continue
                w = weights[i][j]
                if w < 1e-8:
                    continue
                psi_ij = self.psi(Mi, cells[j])
                cp_drive = madd(cp_drive, mscale(msub(psi_ij, Mi), w))

            if R is not None:
                # Full update with perception
                lr_i = self.lr_base * (1 + self.k_s * surprises[i])
                perception = mscale(msub(R, Mi), (1 - ai) * lr_i)
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(madd(mscale(ef_drive, ai), perception), coupling)
            else:
                # No signal: skip perception drive
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(mscale(ef_drive, ai), coupling)

            dMs.append(dM)

        # Apply updates
        for i in range(n):
            noise = mrand(k, self.noise_scale)
            cells[i] = madd(cells[i], madd(mscale(dMs[i], self.dt), noise))
            cells[i] = mclip(cells[i], self.max_norm)

        self.step_count += 1
        return dMs

    # ── Observables ──────────────────────────────────────────

    def mean_ef_dist(self):
        return sum(self.eigenform_distance(c) for c in self.cells) / self.n

    def mean_autonomy(self):
        return sum(self.autonomy(c) for c in self.cells) / self.n

    def mean_energy(self, dMs):
        """Mean of frob(dM_i)^2 per cell."""
        return sum(frob(dM) ** 2 for dM in dMs) / self.n

    def mean_surprise(self, r):
        """Mean surprise across cells for a given input vector."""
        R = self.project(r)
        return sum(self.surprise(R, c) for c in self.cells) / self.n

    def composite(self):
        R = meye(self.k)
        for c in self.cells:
            R = mmul(R, c)
        return R

    def composite_alignment(self, r):
        """Cosine similarity between composite and projected input."""
        R = self.project(r)
        C = self.composite()
        return mcosine(C, R)
