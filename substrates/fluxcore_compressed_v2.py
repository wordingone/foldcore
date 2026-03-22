#!/usr/bin/env python3
"""
FluxCore Compressed v2 — Dynamic Cell Spawning

Extends CompressedKernel with:
- Dynamic cell spawning when input novelty exceeds threshold
- Pruning of dormant cells based on surprise contribution
- All original dynamics preserved when spawning=False

State: variable-size cell list, each M_i in R^(k x k)
"""

import math
import random
import sys

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import (mzero, madd, msub, mscale, mmul, mtanh, frob,
                mcosine, mclip, mrand, meye)


class CompressedKernel:
    """
    Compressed FluxCore+RK kernel with dynamic spawning.
    Variable number of cells, each M_i in R^(k x k).
    """

    def __init__(self, n=8, k=4, d=64, seed=42, proj_seed=999,
                 alpha=1.2, beta=0.8, lr_base=0.08, k_s=20,
                 sigma=0.3, tau=0.3, dt=0.03, noise_scale=0.01,
                 max_norm=3.0, max_cells=500, spawning=True, k_couple=5):
        self.n = n
        self.n_min = n  # never prune below starting count
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
        self.max_cells = max_cells
        self.spawning = spawning
        self.k_couple = k_couple  # top-k sparse coupling (None = all-to-all)

        # Initialize cells
        random.seed(seed)
        self.cells = [mrand(k, 0.8) for _ in range(n)]
        self.step_count = 0

        # Build fixed projection matrix P in R^(k^2 x d)
        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

        # Spawning statistics
        self.sim_mean = 0.5
        self.sim_var = 0.01
        self.spawn_alpha = 0.01

        # Per-cell tracking
        self.cell_surprise_ema = [0.1] * n
        self.cell_age = [0] * n
        self.total_spawned = 0

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

    def _copy_matrix(self, M):
        """Deep copy a k x k matrix."""
        return [row[:] for row in M]

    def step(self, r=None):
        """
        One update step. r is a d-dimensional input vector or None.
        Returns list of dM matrices (before applying to cells).
        """
        n = len(self.cells)
        k = self.k
        cells = self.cells

        # Project input if provided
        R = self.project(r) if r is not None else None

        # --- Spawning logic (before dynamics) ---
        if self.spawning and R is not None:
            # Compute max cosine similarity between R and all current cells
            max_sim = max(mcosine(cells[i], R) for i in range(n))

            # Update running statistics
            a = self.spawn_alpha
            self.sim_mean = a * max_sim + (1 - a) * self.sim_mean
            self.sim_var = a * (max_sim - self.sim_mean) ** 2 + (1 - a) * self.sim_var

            # Spawn threshold
            spawn_thresh = self.sim_mean - 2 * math.sqrt(self.sim_var + 1e-8)

            # Spawn if input is novel enough
            if max_sim < spawn_thresh and len(self.cells) < self.max_cells:
                new_cell = self._copy_matrix(R)
                self.cells.append(new_cell)
                self.cell_surprise_ema.append(0.1)
                self.cell_age.append(0)
                self.total_spawned += 1
                n = len(self.cells)
                cells = self.cells

        # Compute autonomy for each cell
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        # Compute surprise for each cell (only if signal present)
        surprises = None
        if R is not None:
            surprises = [self.surprise(R, cells[i]) for i in range(n)]

            # Update per-cell surprise EMA
            a = self.spawn_alpha
            for i in range(n):
                self.cell_surprise_ema[i] = a * surprises[i] + (1 - a) * self.cell_surprise_ema[i]

        # Coupling weights: softmax(cosine(M_i, M_j) / tau), sparse top-k
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
            # Sparse top-k: zero all but k_couple nearest neighbors
            if self.k_couple is not None and n > self.k_couple + 1:
                indexed = [(exps[j], j) for j in range(n) if j != i]
                indexed.sort(reverse=True)
                top_k_idx = set(idx for _, idx in indexed[:self.k_couple])
                exps = [exps[j] if j in top_k_idx else 0.0 for j in range(n)]
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

        # Age all cells
        for i in range(n):
            self.cell_age[i] += 1

        # --- Pruning logic ---
        if self.spawning and len(self.cells) > self.n_min:
            to_remove = []
            for i in range(n):
                if (self.cell_surprise_ema[i] < 1e-6
                        and self.cell_age[i] > 500
                        and len(self.cells) - len(to_remove) > self.n_min):
                    to_remove.append(i)

            # Remove in reverse order to preserve indices
            for i in sorted(to_remove, reverse=True):
                del self.cells[i]
                del self.cell_surprise_ema[i]
                del self.cell_age[i]
                if i < len(dMs):
                    del dMs[i]

        self.step_count += 1
        return dMs

    # -- Observables --

    def mean_ef_dist(self):
        n = len(self.cells)
        if n == 0:
            return 0.0
        return sum(self.eigenform_distance(c) for c in self.cells) / n

    def mean_autonomy(self):
        n = len(self.cells)
        if n == 0:
            return 0.0
        return sum(self.autonomy(c) for c in self.cells) / n

    def mean_energy(self, dMs):
        """Mean of frob(dM_i)^2 per cell."""
        n = len(dMs)
        if n == 0:
            return 0.0
        return sum(frob(dM) ** 2 for dM in dMs) / n

    def mean_surprise(self, r):
        """Mean surprise across cells for a given input vector."""
        n = len(self.cells)
        if n == 0:
            return 0.0
        R = self.project(r)
        return sum(self.surprise(R, c) for c in self.cells) / n

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
