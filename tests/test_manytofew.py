#!/usr/bin/env python3
"""
FluxCore ManyToFewKernel test suite — Step 62.

Tier 1: Unit tests (fast, no data).
Tier 2: Regression tests (seed=42, proj_seed=999, CSI data, deterministic).
Tier 3: Benchmark tests (@pytest.mark.slow).

Run all:   pytest tests/test_manytofew.py
Skip slow: pytest tests/test_manytofew.py -m "not slow"
"""

import os
import sys
import json
import math
import random

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fluxcore_manytofew import ManyToFewKernel, _vec_cosine
from rk import frob


# ─── Paths ────────────────────────────────────────────────────────────────────

_DATA = os.path.join(os.path.dirname(__file__), '..', 'data')
CSI_EMBEDDED = f'{_DATA}/csi_embedded.json'
CSI_CENTERS  = f'{_DATA}/csi_division_centers.json'


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _rand_unit(d, seed=None):
    if seed is not None:
        random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(d)]
    n = math.sqrt(sum(x * x for x in v) + 1e-15)
    return [x / n for x in v]


def _normalize(v):
    n = math.sqrt(sum(x * x for x in v) + 1e-15)
    return [x / n for x in v]


# ─── Tier 1: Unit tests ───────────────────────────────────────────────────────

def test_init_cells_and_empty_codebook():
    """Init creates correct number of matrix cells and empty codebook."""
    k = ManyToFewKernel(n_matrix=8, k=4, d=16, seed=42, proj_seed=999)
    assert len(k.cells) == 8
    assert len(k.codebook) == 0
    assert len(k.cb_assignment) == 0
    assert k.total_spawned == 0
    assert k.total_merged == 0


def test_step_spawns_codebook_vector():
    """Step with random input spawns a codebook vector."""
    k = ManyToFewKernel(n_matrix=4, k=2, d=8, seed=42, proj_seed=999)
    r = _rand_unit(8, seed=1)
    assert len(k.codebook) == 0
    k.step(r=r)
    assert len(k.codebook) == 1
    assert k.total_spawned == 1


def test_step_generation_no_crash():
    """Step with r=None (generation mode) returns dMs without crash."""
    k = ManyToFewKernel(n_matrix=4, k=2, d=8, seed=42, proj_seed=999)
    dMs = k.step(r=None)
    assert dMs is not None
    assert len(dMs) == 4


def test_merge_triggers_on_near_identical_inputs():
    """Merge triggers when two nearly-identical inputs are provided (|cos| > 0.95).

    Merge fires at spawn time only. Spawn fires when max_sim < spawn_thresh.
    Near-parallel r2 (cos > 0.5) would NOT spawn at all. The correct trigger is
    a nearly anti-parallel vector (cos ≈ -1): spawn fires because cos < 0.5,
    and |cos| > merge_thresh so the new vector is fused into the existing one.
    """
    d = 16
    k = ManyToFewKernel(n_matrix=4, k=2, d=d, seed=42, proj_seed=999,
                        merge_thresh=0.95, spawn_thresh=0.5)

    r1 = _rand_unit(d, seed=10)
    k.step(r=r1)
    assert k.total_spawned == 1

    # Nearly anti-parallel: r2 ≈ -r1 → cos < 0.5 (spawn fires) AND |cos| > 0.95 (merge fires)
    random.seed(11)
    r2 = _normalize([-r1[i] + random.gauss(0, 0.01) for i in range(d)])
    cos_val = _vec_cosine(r1, r2)
    assert cos_val < 0.5       # spawn will fire
    assert abs(cos_val) > 0.95  # merge will fire

    k.step(r=r2)
    assert k.total_spawned == 2   # spawn was attempted
    assert k.total_merged == 1    # but fused into existing
    assert len(k.codebook) == 1   # net codebook size unchanged


def test_spawn_does_not_trigger_when_matched():
    """Spawn does NOT trigger when input matches existing codebook vector (cos > 0.5)."""
    d = 16
    k = ManyToFewKernel(n_matrix=4, k=2, d=d, seed=42, proj_seed=999,
                        spawn_thresh=0.5)

    r = _rand_unit(d, seed=20)
    k.step(r=r)
    assert len(k.codebook) == 1

    # Repeat same vector — max_sim ≈ 1.0, above spawn_thresh, no new spawn
    for _ in range(10):
        k.step(r=r)
    assert len(k.codebook) == 1


# ─── Tier 2: Regression tests ─────────────────────────────────────────────────

@pytest.fixture(scope='module')
def csi_data():
    with open(CSI_EMBEDDED) as f:
        records = json.load(f)
    with open(CSI_CENTERS) as f:
        centers = json.load(f)
    return records, centers


@pytest.fixture(scope='module')
def trained_kernel(csi_data):
    """
    Run canonical CSI benchmark (1920 records) then 3000 generation steps.
    Returns (kernel, gen_energy). Shared across Tier 2 tests (module scope).
    """
    records, _ = csi_data
    records = sorted(records, key=lambda r: r['division'])

    k = ManyToFewKernel(
        n_matrix=8, k=4, d=384, seed=42, proj_seed=999,
        tau=0.3, k_couple=5,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015,
    )

    for rec in records:
        k.step(r=rec['vec'])

    dMs = []
    for _ in range(3000):
        dMs = k.step(r=None)

    gen_energy = sum(frob(dM) ** 2 for dM in dMs) / len(dMs)
    return k, gen_energy


def test_regression_spawned_and_merged(trained_kernel):
    """Exactly 359 codebook vectors spawned, 0 merges."""
    k, _ = trained_kernel
    assert k.total_spawned == 359
    assert k.total_merged == 0


def test_regression_coverage(trained_kernel, csi_data):
    """Coverage = 33/33 CSI divisions."""
    k, _ = trained_kernel
    _, centers = csi_data
    div_names = sorted(centers.keys())
    covered = set()
    for v in k.codebook:
        best_div, best_val = None, 0.0
        for dn in div_names:
            sim = _vec_cosine(v, centers[dn])
            if abs(sim) > abs(best_val):
                best_val, best_div = sim, dn
        if best_div and abs(best_val) > 0.3:
            covered.add(best_div)
    assert len(covered) == 33


def test_regression_n_matrix_8(trained_kernel):
    """n_matrix=8 — step returns exactly 8 dMs."""
    k, _ = trained_kernel
    assert len(k.cells) == 8


def test_regression_generation_energy(trained_kernel):
    """Generation energy at 3000 steps > 0.05."""
    _, gen_energy = trained_kernel
    assert gen_energy > 0.05


# ─── Tier 3: Benchmark tests (slow) ───────────────────────────────────────────

@pytest.mark.slow
def test_drift_first_spawn_at_step_1():
    """After distribution shift A→B, first spawn occurs at step 1."""
    d = 384
    NOISE = 0.04  # scaled for d=384: cos ≈ 0.62

    def make_unit(seed):
        random.seed(seed)
        v = [random.gauss(0, 1) for _ in range(d)]
        return _normalize(v)

    def sample(mu, seed=None):
        if seed is not None:
            random.seed(seed)
        x = [mu[i] + random.gauss(0, NOISE) for i in range(d)]
        return _normalize(x)

    mu_A = make_unit(1001)

    # Build mu_B nearly orthogonal to mu_A (cos ≈ 0.1)
    random.seed(2002)
    raw_B = [random.gauss(0, 1) for _ in range(d)]
    dot_AB = sum(mu_A[i] * raw_B[i] for i in range(d))
    raw_B = [raw_B[i] - dot_AB * mu_A[i] + 0.1 * mu_A[i] for i in range(d)]
    mu_B = _normalize(raw_B)

    k = ManyToFewKernel(
        n_matrix=8, k=4, d=d, seed=42, proj_seed=999,
        spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015,
    )

    random.seed(42)
    for _ in range(1000):
        k.step(r=sample(mu_A))

    spawns_before = k.total_spawned
    random.seed(43)
    k.step(r=sample(mu_B))
    assert k.total_spawned > spawns_before, (
        f"Expected spawn at step 1 of drift (before={spawns_before}, after={k.total_spawned})"
    )


@pytest.mark.slow
def test_chaotic_lorenz_more_vectors_than_mackeyglass():
    """Lorenz-63 spawns more codebook vectors than Mackey-Glass (attractor complexity)."""
    D = 64
    N = 2000

    def embed_scalar(x, d=D):
        r = [math.sin(x * math.pi * i / d) for i in range(d)]
        return _normalize(r)

    def embed_lorenz(xyz, d=D):
        x, y, z = xyz
        xn, yn, zn = x / 25.0, y / 25.0, (z - 25.0) / 25.0
        feats, per = [], d // 3
        for v in [xn, yn, zn]:
            for i in range(per):
                feats.append(math.sin(v * math.pi * (i + 1) / per))
        while len(feats) < d:
            feats.append(math.cos(feats[-1]))
        return _normalize(feats[:d])

    # Generate Mackey-Glass (tau=17, dt=0.1)
    buf = [0.9] * (171 + N + 1)
    for t in range(170, 170 + N):
        xt, xt_tau = buf[t], buf[t - 170]
        buf[t + 1] = xt + 0.1 * (0.2 * xt_tau / (1.0 + xt_tau ** 10) - 0.1 * xt)
    mg = [embed_scalar(buf[170 + i]) for i in range(N)]

    # Generate Lorenz-63
    x, y, z = 1.0, 1.0, 1.0
    lz = []
    for _ in range(N):
        dx = 10 * (y - x)
        dy = x * (28 - z) - y
        dz = x * y - (8 / 3) * z
        x += 0.01 * dx; y += 0.01 * dy; z += 0.01 * dz
        lz.append(embed_lorenz((x, y, z)))

    k_mg = ManyToFewKernel(n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
                            spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015)
    k_lz = ManyToFewKernel(n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
                            spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015)

    for r in mg:
        k_mg.step(r=r)
    for r in lz:
        k_lz.step(r=r)

    assert k_lz.total_spawned > k_mg.total_spawned, (
        f"Expected Lorenz ({k_lz.total_spawned}) > MG ({k_mg.total_spawned})"
    )
