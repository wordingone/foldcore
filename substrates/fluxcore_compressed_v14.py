#!/usr/bin/env python3
"""
FluxCore Compressed v14 — Dual-Representation, Data-Seeded Codebook Init

Each cell has TWO representations:
1. Vector v_i in S^(d-1): fold-style codebook for coverage/routing/spawning.
   Updated: v_winner <- normalize(v_winner + lr_cb * r). No coupling on vectors.
2. Matrix M_i in R^(k x k): RK-style dynamics for generation.
   Updated: eigenform + coupling (all cells) + perception (winner only).

Perception routing: winner-take-all by vector cosine.
  winner = argmax_i cos(v_i, r)
  Only winner gets matrix perception term. Others: eigenform + coupling only.

Spawning: mu-2sigma on codebook cosines (same as v2 but using vectors not matrices).
Coverage measured by codebook alignment to division centers.

Synthesis of all failure experiments (Steps 41-49):
- Coverage needs independent cells (fold-style) -> vector codebook
- Generation needs coupled cells (RK-style) -> matrix dynamics
- Decoupled by routing: codebook routes, matrix generates

One architectural change from v2 for Step 50.
"""

import math
import random
import sys

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import (mzero, madd, msub, mscale, mmul, mtanh, frob,
                mcosine, mclip, mrand, meye)


def _dot(a, b):
    return sum(a[i] * b[i] for i in range(len(a)))


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _normalize(v):
    n = _norm(v)
    if n < 1e-15:
        return v[:]
    return [x / n for x in v]


def _vec_cosine(a, b):
    na = _norm(a)
    nb = _norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return _dot(a, b) / (na * nb)


def _vec_add(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _vec_scale(v, s):
    return [x * s for x in v]


class CompressedKernel:
    def __init__(self, n=8, k=4, d=64, seed=42, proj_seed=999,
                 alpha=1.2, beta=0.8, lr_base=0.08, k_s=20,
                 sigma=0.3, tau=0.3, dt=0.03, noise_scale=0.01,
                 max_norm=3.0, max_cells=500, spawning=True, k_couple=5,
                 lr_codebook=0.1, init_codebook=None):
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
        self.lr_codebook = lr_codebook

        random.seed(seed)
        self.cells = [mrand(k, 0.8) for _ in range(n)]
        # Codebook: unit vectors in R^d — data-seeded if provided, else random
        # CHANGED from v12: accept init_codebook (list of n raw vectors) to seed from data
        if init_codebook is not None:
            self.codebook = [_normalize(v) for v in init_codebook[:n]]
        else:
            self.codebook = [_normalize([random.gauss(0, 1) for _ in range(d)])
                             for _ in range(n)]
        self.step_count = 0

        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

        # Spawning statistics (on codebook cosines)
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

        # Codebook similarities (used for spawning and routing)
        cb_sims = None
        winner = None
        if r is not None:
            cb_sims = [_vec_cosine(self.codebook[i], r) for i in range(n)]
            winner = max(range(n), key=lambda i: cb_sims[i])

            # Spawning: mu-2sigma on codebook cosines
            if self.spawning:
                max_sim = cb_sims[winner]
                a = self.spawn_alpha
                self.sim_mean = a * max_sim + (1 - a) * self.sim_mean
                self.sim_var = a * (max_sim - self.sim_mean) ** 2 + (1 - a) * self.sim_var
                spawn_thresh = self.sim_mean - 2 * math.sqrt(self.sim_var + 1e-8)
                if max_sim < spawn_thresh and len(self.cells) < self.max_cells:
                    # New cell: codebook vector = normalized r, matrix = projected R
                    new_v = _normalize(r)
                    new_M = self._copy_matrix(self.project(r))
                    self.cells.append(new_M)
                    self.codebook.append(new_v)
                    self.cell_surprise_ema.append(0.1)
                    self.cell_age.append(0)
                    self.total_spawned += 1
                    n = len(self.cells)
                    cells = self.cells
                    # Recompute after spawn
                    cb_sims = [_vec_cosine(self.codebook[i], r) for i in range(n)]
                    winner = max(range(n), key=lambda i: cb_sims[i])

            # Update winner codebook vector (fold-style)
            v_w = self.codebook[winner]
            self.codebook[winner] = _normalize(_vec_add(v_w, _vec_scale(r, self.lr_codebook)))

        # Project r for matrix perception (winner only)
        R = self.project(r) if r is not None else None

        # Autonomy
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        # Surprise (matrix-based, for winner tracking)
        surprises = None
        if R is not None:
            surprises = [self.surprise(R, cells[i]) for i in range(n)]
            a = self.spawn_alpha
            for i in range(n):
                self.cell_surprise_ema[i] = a * surprises[i] + (1 - a) * self.cell_surprise_ema[i]

        # Coupling weights (matrix cosines, unchanged from v2)
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

            if R is not None and i == winner:
                # Winner: eigenform + perception + coupling
                lr_i = self.lr_base * (1 + self.k_s * surprises[i])
                perception = mscale(msub(R, Mi), (1 - ai) * lr_i)
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(madd(mscale(ef_drive, ai), perception), coupling)
            else:
                # Non-winner (or generation mode): eigenform + coupling only
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
                del self.codebook[i]
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
