#!/usr/bin/env python3
"""
FluxCore Many-to-Few (v17) — Separated Codebook + Matrix Architecture

Architecture:
1. Codebook layer: Pure fold memory system. Unit vectors in R^d.
   - Spawn threshold: fixed 0.5
   - Update: additive normalize(v + 0.015 * r) on winner
   - Merge: incremental at spawn time — new vector vs all existing, |cos| > 0.95 -> fuse
   - Grows to 300+ vectors for coverage (cheap dot products, O(d) per lookup)

2. Matrix layer: Fixed n_matrix=8 cells. RK dynamics unchanged from v2.
   - Eigenform + coupling (all cells) + perception (winner cell only)
   - Fixed count — no O(n^2) matrix explosion

3. Routing: each codebook vector is assigned to a matrix cell at spawn time.
   - Assignment: nearest matrix cell by projected cosine similarity
   - When codebook winner fires, its assigned matrix cell gets perception
   - Many codebook vectors -> same matrix cell (many-to-few)

Compute profile:
- Codebook lookup: O(codebook_size * d) — cheap, just dot products
- Matrix dynamics: O(8^2 * k^2) — fixed small network
- No O(n^2) matrix operations
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


class ManyToFewKernel:
    def __init__(self, n_matrix=8, k=4, d=384, seed=42, proj_seed=999,
                 alpha=1.2, beta=0.8, lr_base=0.08, k_s=20,
                 sigma=0.3, tau=0.3, dt=0.03, noise_scale=0.01,
                 max_norm=3.0, k_couple=5,
                 spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015):
        self.n_matrix = n_matrix
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
        self.k_couple = k_couple
        self.spawn_thresh = spawn_thresh
        self.merge_thresh = merge_thresh
        self.lr_codebook = lr_codebook

        random.seed(seed)
        # Matrix layer: fixed n_matrix cells, RK dynamics
        self.cells = [mrand(k, 0.8) for _ in range(n_matrix)]

        # Codebook layer: starts empty, grows via fold-style spawning
        self.codebook = []       # list of unit vectors in R^d
        self.cb_assignment = []  # which matrix cell (0..n_matrix-1) each cb vec maps to

        self.step_count = 0
        self.total_spawned = 0
        self.total_merged = 0

        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

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

    def _try_merge_new(self, new_idx):
        """
        Incremental merge: compare new vector against all existing vectors.
        If |cos| > merge_thresh, fuse new into the most similar existing, delete new.
        Returns the winner index after potential merge.
        O(n * d) per call.
        """
        if new_idx == 0:
            return new_idx
        new_v = self.codebook[new_idx]
        best_i, best_abs = -1, 0.0
        for i in range(new_idx):
            c = _vec_cosine(new_v, self.codebook[i])
            a = abs(c)
            if a > best_abs:
                best_abs, best_i = a, i
        if best_abs > self.merge_thresh:
            # Fuse new into best_i (equal weight)
            self.codebook[best_i] = _normalize(_vec_add(self.codebook[best_i], new_v))
            del self.codebook[new_idx]
            del self.cb_assignment[new_idx]
            self.total_merged += 1
            return best_i
        return new_idx

    def step(self, r=None):
        n = self.n_matrix
        k = self.k
        cells = self.cells

        winner_cell = None
        R = None

        if r is not None:
            R = self.project(r)

            # --- Codebook layer (fold memory system) ---
            cb_winner = None
            max_sim = -1.0
            if self.codebook:
                sims = [_vec_cosine(v, r) for v in self.codebook]
                cb_winner = max(range(len(self.codebook)), key=lambda i: sims[i])
                max_sim = sims[cb_winner]

            # Spawn check (fold's fixed threshold)
            if cb_winner is None or max_sim < self.spawn_thresh:
                new_v = _normalize(r)
                # Assign to nearest matrix cell by projected cosine
                cell_sims = [mcosine(cells[j], R) for j in range(n)]
                # mcosine can return NaN for zero matrices — guard with identity check
                assignment = max(
                    range(n),
                    key=lambda j: cell_sims[j] if cell_sims[j] == cell_sims[j] else -1e10
                )
                self.codebook.append(new_v)
                self.cb_assignment.append(assignment)
                self.total_spawned += 1
                cb_winner = len(self.codebook) - 1
                # Incremental merge: new vector vs all existing
                cb_winner = self._try_merge_new(cb_winner)

            # Update winner codebook vector (fold's additive rule)
            self.codebook[cb_winner] = _normalize(
                _vec_add(self.codebook[cb_winner], _vec_scale(r, self.lr_codebook))
            )

            # Route to assigned matrix cell
            winner_cell = self.cb_assignment[cb_winner]

        # --- Matrix layer (fixed 8 cells, RK dynamics) ---
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        winner_surprise = None
        if R is not None and winner_cell is not None:
            winner_surprise = self.surprise(R, cells[winner_cell])

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

            if R is not None and i == winner_cell:
                lr_i = self.lr_base * (1 + self.k_s * winner_surprise)
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

        self.step_count += 1
        return dMs

    def mean_ef_dist(self):
        return sum(self.eigenform_distance(c) for c in self.cells) / self.n_matrix

    def mean_autonomy(self):
        return sum(self.autonomy(c) for c in self.cells) / self.n_matrix

    def mean_energy(self, dMs):
        n = len(dMs)
        if n == 0:
            return 0.0
        return sum(frob(dM) ** 2 for dM in dMs) / n

    def composite(self):
        R = meye(self.k)
        for c in self.cells:
            R = mmul(R, c)
        return R
