#!/usr/bin/env python3
"""
FluxCore Compressed v16 — Fold-Faithful Codebook + Matrix Dynamics

Three changes from v15 to transplant fold's proven memory system:

1. Fixed spawn threshold = 0.5 (replaces mu-2sigma).
   Fold hand-tuned (threshold=0.5) gets 33/33. Autothresh (mu-2sigma) gets 21/33.

2. Fold-style additive codebook update: v_w = normalize(v_w + 0.015 * r).
   Fold uses memLr=0.015. LVQ negative push for k_neg losers (kept from v15).

3. Merge: pairwise codebook merge when |cos(vi, vj)| > 0.95.
   Fold uses use-count weighted average. Here: equal weight (normalize(vi + vj)).
   Deletes redundant cells (codebook + matrix) to prevent bloat.

Matrix dynamics unchanged from v15: winner gets perception, all get eigenform + coupling.
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
                 lr_codebook=0.015, init_codebook=None,
                 neg_lr=0.01, k_neg=3,
                 spawn_thresh=0.5, merge_thresh=0.95):
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
        self.neg_lr = neg_lr
        self.k_neg = k_neg
        self.spawn_thresh = spawn_thresh  # fixed threshold (replaces mu-2sigma)
        self.merge_thresh = merge_thresh

        random.seed(seed)
        self.cells = [mrand(k, 0.8) for _ in range(n)]
        # Codebook: data-seeded if provided, else random
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

        self.spawn_alpha = 0.01  # for cell_surprise_ema only

        self.cell_surprise_ema = [0.1] * n
        self.cell_age = [0] * n
        self.total_spawned = 0
        self.total_merged = 0

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

    def _merge_codebook(self):
        """Pairwise merge: if |cos(vi, vj)| > merge_thresh, fuse into vi, delete vj."""
        to_delete = set()
        n = len(self.codebook)
        for i in range(n):
            if i in to_delete:
                continue
            for j in range(i + 1, n):
                if j in to_delete:
                    continue
                c = _vec_cosine(self.codebook[i], self.codebook[j])
                if abs(c) > self.merge_thresh:
                    # Merge j into i (equal weight)
                    self.codebook[i] = _normalize(_vec_add(self.codebook[i], self.codebook[j]))
                    to_delete.add(j)
                    self.total_merged += 1

        if to_delete:
            keep = [i for i in range(n) if i not in to_delete]
            self.codebook = [self.codebook[i] for i in keep]
            self.cells = [self.cells[i] for i in keep]
            self.cell_surprise_ema = [self.cell_surprise_ema[i] for i in keep]
            self.cell_age = [self.cell_age[i] for i in keep]

    def step(self, r=None):
        n = len(self.cells)
        k = self.k
        cells = self.cells

        # Codebook similarities
        cb_sims = None
        winner = None
        if r is not None:
            cb_sims = [_vec_cosine(self.codebook[i], r) for i in range(n)]
            winner = max(range(n), key=lambda i: cb_sims[i])

            # Fixed-threshold spawning (CHANGED from v15: mu-2sigma -> fixed 0.5)
            if self.spawning:
                max_sim = cb_sims[winner]
                if max_sim < self.spawn_thresh and len(self.cells) < self.max_cells:
                    new_v = _normalize(r)
                    new_M = self._copy_matrix(self.project(r))
                    self.cells.append(new_M)
                    self.codebook.append(new_v)
                    self.cell_surprise_ema.append(0.1)
                    self.cell_age.append(0)
                    self.total_spawned += 1
                    n = len(self.cells)
                    cells = self.cells
                    cb_sims = [_vec_cosine(self.codebook[i], r) for i in range(n)]
                    winner = max(range(n), key=lambda i: cb_sims[i])

            # Fold-style additive winner update (CHANGED from v15: LVQ+ -> additive lr=0.015)
            self.codebook[winner] = _normalize(
                _vec_add(self.codebook[winner], _vec_scale(r, self.lr_codebook))
            )
            # LVQ negative push for k_neg nearest losers (kept from v15)
            if self.k_neg > 0 and n > 1:
                losers_by_sim = sorted(
                    [i for i in range(n) if i != winner],
                    key=lambda i: cb_sims[i], reverse=True
                )
                for loser in losers_by_sim[:self.k_neg]:
                    sim_l = cb_sims[loser]
                    if sim_l > 0:
                        self.codebook[loser] = _normalize(
                            _vec_add(self.codebook[loser], _vec_scale(r, -self.neg_lr * sim_l))
                        )

        # Project r for matrix perception (winner only)
        R = self.project(r) if r is not None else None

        # Autonomy
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        # Surprise
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

            if R is not None and i == winner:
                lr_i = self.lr_base * (1 + self.k_s * surprises[i])
                perception = mscale(msub(R, Mi), (1 - ai) * lr_i)
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

        # Pruning (matrix-based, unchanged)
        if self.spawning and len(self.cells) > self.n_min:
            to_remove = []
            for i in range(len(self.cells)):
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

        # Codebook merge (ADDED: fold mechanism to prevent redundancy bloat)
        if r is not None and len(self.codebook) > self.n_min:
            self._merge_codebook()

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
