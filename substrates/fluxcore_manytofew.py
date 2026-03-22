#!/usr/bin/env python3
"""
FluxCore Many-to-Few Kernel — Canonical Implementation

Architecture: two decoupled layers.

1. Codebook layer (fold memory system)
   Unit vectors in R^d. Handles coverage, routing, and novelty detection.
   - Spawn: when max cosine similarity < spawn_thresh, add normalize(r) to codebook.
   - Update: additive rule — v_winner += lr_codebook * r, then normalize.
   - Merge: at spawn time, compare new vector against all existing; if |cos| > merge_thresh,
     fuse into the most similar existing vector (equal weight) and discard the new one.
   - Assignment: each codebook vector is permanently assigned to a matrix cell
     (nearest by projected cosine at spawn time).

2. Matrix layer (RK dynamics)
   Fixed n_matrix cells M_i in R^(k x k). Handles generation and temporal dynamics.
   - Eigenform drive: phi(M) - M, where phi(M) = tanh(alpha*M + beta*M^2/k).
   - Coupling: softmax-weighted psi(M_i, M_j) across k_couple nearest neighbors.
   - Perception: winner cell (routed from codebook layer) receives input projection.
   - All three terms are gated by autonomy a_i = exp(-ef_dist^2 / sigma^2).

Routing: codebook winner -> its assigned matrix cell gets perception.
Many codebook vectors -> same matrix cell (many-to-few).

Compute profile:
- Codebook lookup: O(codebook_size * d) — dot products only, no matrix ops.
- Matrix dynamics: O(n_matrix^2 * k^2) — fixed small network.

Proven configuration (CSI benchmark, 20 experiments):
- n_matrix=8, k=4, d=384, spawn_thresh=0.5, lr_codebook=0.015, merge_thresh=0.95
- Coverage: 33/33 CSI divisions. Generation energy: 0.081. Runtime: 25s/1920 records.
"""

import math
import random


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


from rk import (mzero, madd, msub, mscale, mmul, mtanh, frob,
                mcosine, mclip, mrand, meye)


class ManyToFewKernel:
    """
    Many-to-few FluxCore kernel. See module docstring for architecture details.

    Parameters
    ----------
    n_matrix : int
        Number of matrix cells (fixed). Default 8 (proven optimal).
    k : int
        Matrix dimension (k x k). Default 4.
    d : int
        Input dimensionality for codebook. Default 384 (CSI embedding size).
    seed : int
        RNG seed for matrix initialization.
    proj_seed : int
        RNG seed for random projection matrix P (maps R^d -> R^(k*k)).
    alpha, beta : float
        Eigenform nonlinearity coefficients. phi(M) = tanh(alpha*M + beta*M^2/k).
    lr_base : float
        Base learning rate for matrix perception.
    k_s : float
        Surprise scaling for adaptive perception rate.
    sigma : float
        Autonomy bandwidth: a_i = exp(-ef_dist^2 / sigma^2).
    tau : float
        Coupling temperature for softmax neighbor weights.
    dt : float
        Euler integration step size.
    noise_scale : float
        Magnitude of per-step matrix noise.
    max_norm : float
        Matrix element clipping bound.
    k_couple : int
        Number of coupling neighbors per cell.
    spawn_thresh : float
        Codebook spawn threshold. Spawn if max_sim < spawn_thresh. Default 0.5.
    merge_thresh : float
        Codebook merge threshold. Fuse if |cos(v_i, v_new)| > merge_thresh. Default 0.95.
    lr_codebook : float
        Codebook additive update rate. v += lr_codebook * r, then normalize. Default 0.015.
    """

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
        self.cells = [mrand(k, 0.8) for _ in range(n_matrix)]

        # Codebook: starts empty, grows via fold-style spawning
        self.codebook = []       # unit vectors in R^d
        self.cb_assignment = []  # matrix cell index for each codebook vector

        # Labeled codebook: parallel to self.codebook
        self.cb_labels = []      # label assigned at spawn time; None if no label provided

        self.step_count = 0
        self.total_spawned = 0
        self.total_merged = 0

        # Random projection: R^d -> R^(k*k), frozen after init
        random.seed(proj_seed)
        scale = 1.0 / math.sqrt(d)
        self.P = [[random.gauss(0, scale) for _ in range(d)]
                   for _ in range(k * k)]

    def project(self, r):
        """Project input r from R^d to R^(k x k) via frozen random matrix P."""
        k = self.k
        flat = [sum(self.P[i][j] * r[j] for j in range(self.d))
                for i in range(k * k)]
        return [flat[i * k:(i + 1) * k] for i in range(k)]

    def phi(self, M):
        """Eigenform map: phi(M) = tanh(alpha*M + beta*M^2/k)."""
        linear = mscale(M, self.alpha)
        quadratic = mscale(mmul(M, M), self.beta / self.k)
        return mtanh(madd(linear, quadratic))

    def psi(self, Mi, Mj):
        """Coupling interaction between cells i and j."""
        avg = mscale(madd(Mi, Mj), self.alpha / 2.0)
        prod = mscale(mmul(Mi, Mj), self.beta / self.k)
        return mtanh(madd(avg, prod))

    def eigenform_distance(self, M):
        """Distance from M to its eigenform phi(M), normalized by ||M||."""
        return frob(msub(self.phi(M), M)) / max(frob(M), 1.0)

    def autonomy(self, M):
        """Autonomy a_i = exp(-ef_dist^2 / sigma^2). High when near eigenform."""
        d = self.eigenform_distance(M)
        return math.exp(-(d * d) / (self.sigma * self.sigma))

    def surprise(self, R, M):
        """Surprise of cell M given projected input R: ||R - M|| / ||M||."""
        return frob(msub(R, M)) / (frob(M) + 1e-15)

    def _try_merge_new(self, new_idx):
        """
        Incremental merge at spawn time. O(n * d).

        Compares the new codebook vector (at new_idx) against all existing vectors.
        If any existing vector has |cos| > merge_thresh, fuse new into it (equal weight)
        and discard new. Returns winner index after potential merge.
        """
        if new_idx == 0:
            return new_idx
        new_v = self.codebook[new_idx]
        best_i, best_abs = -1, 0.0
        for i in range(new_idx):
            a = abs(_vec_cosine(new_v, self.codebook[i]))
            if a > best_abs:
                best_abs, best_i = a, i
        if best_abs > self.merge_thresh:
            self.codebook[best_i] = _normalize(_vec_add(self.codebook[best_i], new_v))
            del self.codebook[new_idx]
            del self.cb_assignment[new_idx]
            del self.cb_labels[new_idx]   # keep best_i's label (it has more history)
            self.total_merged += 1
            return best_i
        return new_idx

    def classify(self, r, k=1):
        """
        Return the label of the nearest codebook vector (k=1) or majority
        vote among the k nearest prototypes.

        Parameters
        ----------
        r : list[float]
            Input vector in R^d.
        k : int
            Number of nearest neighbors for voting. k=1 is 1-nearest prototype.

        Returns
        -------
        label : any or None
            Predicted label, or None if the codebook is empty.
        """
        if not self.codebook:
            return None
        sims = [(_vec_cosine(v, r), self.cb_labels[i])
                for i, v in enumerate(self.codebook)]
        sims.sort(reverse=True)
        top_k = [label for _, label in sims[:k]]
        return max(set(top_k), key=top_k.count)

    def step(self, r=None, label=None):
        """
        One update step.

        Parameters
        ----------
        r : list[float] or None
            Input vector in R^d. If None, runs generation mode (no perception).
        label : any or None
            Label for this input. Stored on the codebook vector at spawn time.
            Ignored if r is None or no spawn occurs.

        Returns
        -------
        dMs : list[matrix]
            Per-cell update vectors (useful for computing mean energy).
        """
        n = self.n_matrix
        k = self.k
        cells = self.cells
        winner_cell = None
        R = None

        if r is not None:
            R = self.project(r)

            # Codebook: find winner
            cb_winner = None
            max_sim = -1.0
            if self.codebook:
                sims = [_vec_cosine(v, r) for v in self.codebook]
                cb_winner = max(range(len(self.codebook)), key=lambda i: sims[i])
                max_sim = sims[cb_winner]

            # Spawn if no codebook or winner similarity below threshold
            if cb_winner is None or max_sim < self.spawn_thresh:
                new_v = _normalize(r)
                # Assign to nearest matrix cell by projected cosine
                cell_sims = [mcosine(cells[j], R) for j in range(n)]
                assignment = max(
                    range(n),
                    key=lambda j: cell_sims[j] if cell_sims[j] == cell_sims[j] else -1e10
                )
                self.codebook.append(new_v)
                self.cb_assignment.append(assignment)
                self.cb_labels.append(label)
                self.total_spawned += 1
                cb_winner = len(self.codebook) - 1
                cb_winner = self._try_merge_new(cb_winner)

            # Additive winner update (fold's proven rule)
            self.codebook[cb_winner] = _normalize(
                _vec_add(self.codebook[cb_winner], _vec_scale(r, self.lr_codebook))
            )

            # Route to assigned matrix cell
            winner_cell = self.cb_assignment[cb_winner]

        # Matrix dynamics (fixed n_matrix cells)
        alphas = [self.autonomy(cells[i]) for i in range(n)]

        winner_surprise = None
        if R is not None and winner_cell is not None:
            winner_surprise = self.surprise(R, cells[winner_cell])

        # Coupling weights: softmax over k_couple nearest neighbors
        weights = []
        for i in range(n):
            raw = [mcosine(cells[i], cells[j]) / self.tau if i != j else -1e10
                   for j in range(n)]
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            if self.k_couple is not None and n > self.k_couple + 1:
                indexed = sorted(
                    [(exps[j], j) for j in range(n) if j != i], reverse=True
                )
                top_k = {idx for _, idx in indexed[:self.k_couple]}
                exps = [exps[j] if j in top_k else 0.0 for j in range(n)]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # Compute per-cell updates
        dMs = []
        for i in range(n):
            ai = alphas[i]
            Mi = cells[i]
            ef_drive = msub(self.phi(Mi), Mi)
            cp_drive = mzero(k)
            for j in range(n):
                if i == j or weights[i][j] < 1e-8:
                    continue
                cp_drive = madd(cp_drive, mscale(msub(self.psi(Mi, cells[j]), Mi), weights[i][j]))

            if R is not None and i == winner_cell:
                lr_i = self.lr_base * (1 + self.k_s * winner_surprise)
                perception = mscale(msub(R, Mi), (1 - ai) * lr_i)
                coupling = mscale(cp_drive, 1 - ai)
                dM = madd(madd(mscale(ef_drive, ai), perception), coupling)
            else:
                dM = madd(mscale(ef_drive, ai), mscale(cp_drive, 1 - ai))
            dMs.append(dM)

        # Apply updates
        for i in range(n):
            cells[i] = mclip(madd(cells[i], madd(mscale(dMs[i], self.dt), mrand(k, self.noise_scale))), self.max_norm)

        self.step_count += 1
        return dMs

    def mean_ef_dist(self):
        """Mean eigenform distance across all matrix cells."""
        return sum(self.eigenform_distance(c) for c in self.cells) / self.n_matrix

    def mean_autonomy(self):
        """Mean autonomy across all matrix cells."""
        return sum(self.autonomy(c) for c in self.cells) / self.n_matrix

    def mean_energy(self, dMs):
        """Mean squared Frobenius norm of update vectors. Zero if no updates."""
        if not dMs:
            return 0.0
        return sum(frob(dM) ** 2 for dM in dMs) / len(dMs)

    def composite(self):
        """Product of all matrix cells (composite state)."""
        R = meye(self.k)
        for c in self.cells:
            R = mmul(R, c)
        return R
