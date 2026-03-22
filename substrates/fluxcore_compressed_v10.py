#!/usr/bin/env python3
"""
FluxCore Compressed v10 — Gaussian-Weighted Perception

Extends v2: perception learning rate scaled by Gaussian of cell-to-input distance.
lr_i_eff = lr_i * exp(-||M_i - R||_F^2 / (2 * sigma^2))
sigma = self-calibrating mean cell-to-input distance (updated per step).

Every cell still receives perception. Cells close to R: strong pull.
Cells far from R: weak but nonzero pull (maintain meaningful eigenform).
Coupling unchanged. Synthesizes lessons from Steps 44-47.

One-variable change for Step 48.
"""

import math
import random
import sys

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import (mzero, madd, msub, mscale, mmul, mtanh, frob,
                mcosine, mclip, mrand, meye)


class CompressedKernel:
    def __init__(self, n=8, k=4, d=64, seed=42, proj_seed=999,
                 alpha=1.2, beta=0.8, lr_base=0.08, k_s=20,
                 sigma=0.3, tau=0.3, dt=0.03, noise_scale=0.01,
                 max_norm=3.0, max_cells=500, spawning=True, k_couple=5,
                 perc_sigma=None):
        self.n = n
        self.n_min = n
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
        self.k_couple = k_couple
        # perc_sigma: fixed perception sigma, or None = self-calibrating (mean dist)
        self.perc_sigma = perc_sigma
        self._mean_dist_ema = None  # EMA of mean cell-to-input distance

        random.seed(seed)
        self.cells = [mrand(k, 0.8) for _ in range(n)]
        self.step_count = 0

        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

        self.sim_mean = 0.5
        self.sim_var = 0.01
        self.spawn_alpha = 0.01

        self.cell_surprise_ema = [0.1] * n
        self.cell_age = [0] * n
        self.total_spawned = 0

    def project(self, r):
        k = self.k
        flat = [sum(self.P[i][j] * r[j] for j in range(self.d))
                for i in range(k * k)]
        return [flat[i * k:(i + 1) * k] for i in range(k)]

    def phi(self, M):
        linear = mscale(M, self.alpha)
        quadratic = mscale(mmul(M, M), self.beta / self.k)
        return mtanh(madd(linear, quadratic))

    def psi(self, Mi, Mj):
        avg = mscale(madd(Mi, Mj), self.alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), self.beta / self.k)
        return mtanh(madd(avg, prod))

    def eigenform_distance(self, M):
        return frob(msub(self.phi(M), M)) / max(frob(M), 1.0)

    def autonomy(self, M):
        d = self.eigenform_distance(M)
        return math.exp(-(d * d) / (self.sigma * self.sigma))

    def surprise(self, R, M):
        return frob(msub(R, M)) / (frob(M) + 1e-15)

    def _copy_matrix(self, M):
        return [row[:] for row in M]

    def step(self, r=None):
        n = len(self.cells)
        k = self.k
        cells = self.cells

        R = self.project(r) if r is not None else None

        if self.spawning and R is not None:
            max_sim = max(mcosine(cells[i], R) for i in range(n))
            a = self.spawn_alpha
            self.sim_mean = a * max_sim + (1 - a) * self.sim_mean
            self.sim_var = a * (max_sim - self.sim_mean) ** 2 + (1 - a) * self.sim_var
            spawn_thresh = self.sim_mean - 2 * math.sqrt(self.sim_var + 1e-8)
            if max_sim < spawn_thresh and len(self.cells) < self.max_cells:
                new_cell = self._copy_matrix(R)
                self.cells.append(new_cell)
                self.cell_surprise_ema.append(0.1)
                self.cell_age.append(0)
                self.total_spawned += 1
                n = len(self.cells)
                cells = self.cells

        alphas = [self.autonomy(cells[i]) for i in range(n)]

        # Compute cell-to-input distances for Gaussian perception weighting
        dists = None
        perc_sigma_val = None
        if R is not None:
            dists = [frob(msub(cells[i], R)) for i in range(n)]
            mean_dist = sum(dists) / n if n > 0 else 1.0

            if self.perc_sigma is not None:
                perc_sigma_val = self.perc_sigma
            else:
                # Self-calibrating: EMA of mean dist
                if self._mean_dist_ema is None:
                    self._mean_dist_ema = mean_dist
                else:
                    self._mean_dist_ema = 0.05 * mean_dist + 0.95 * self._mean_dist_ema
                perc_sigma_val = max(self._mean_dist_ema, 1e-6)

        surprises = None
        if R is not None:
            surprises = [self.surprise(R, cells[i]) for i in range(n)]
            a = self.spawn_alpha
            for i in range(n):
                self.cell_surprise_ema[i] = a * surprises[i] + (1 - a) * self.cell_surprise_ema[i]

        # Coupling weights (unchanged from v2)
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
            if self.k_couple is not None and n > self.k_couple + 1:
                indexed = [(exps[j], j) for j in range(n) if j != i]
                indexed.sort(reverse=True)
                top_k_idx = set(idx for _, idx in indexed[:self.k_couple])
                exps = [exps[j] if j in top_k_idx else 0.0 for j in range(n)]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        dMs = []
        for i in range(n):
            ai = alphas[i]
            Mi = cells[i]
            phi_M = self.phi(Mi)
            ef_drive = msub(phi_M, Mi)
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
                lr_i = self.lr_base * (1 + self.k_s * surprises[i])
                # Gaussian distance weighting on perception
                gauss_w = math.exp(-dists[i] ** 2 / (2 * perc_sigma_val ** 2))
                lr_i_eff = lr_i * gauss_w
                perception = mscale(msub(R, Mi), (1 - ai) * lr_i_eff)
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(madd(mscale(ef_drive, ai), perception), coupling)
            else:
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(mscale(ef_drive, ai), coupling)
            dMs.append(dM)

        for i in range(n):
            noise = mrand(k, self.noise_scale)
            cells[i] = madd(cells[i], madd(mscale(dMs[i], self.dt), noise))
            cells[i] = mclip(cells[i], self.max_norm)

        for i in range(n):
            self.cell_age[i] += 1

        if self.spawning and len(self.cells) > self.n_min:
            to_remove = []
            for i in range(n):
                if (self.cell_surprise_ema[i] < 1e-6
                        and self.cell_age[i] > 500
                        and len(self.cells) - len(to_remove) > self.n_min):
                    to_remove.append(i)
            for i in sorted(to_remove, reverse=True):
                del self.cells[i]
                del self.cell_surprise_ema[i]
                del self.cell_age[i]
                if i < len(dMs):
                    del dMs[i]

        self.step_count += 1
        return dMs

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
        n = len(dMs)
        if n == 0:
            return 0.0
        return sum(frob(dM) ** 2 for dM in dMs) / n

    def mean_surprise(self, r):
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
        R = self.project(r)
        C = self.composite()
        return mcosine(C, R)
