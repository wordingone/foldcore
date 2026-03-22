#!/usr/bin/env python3
"""
Step 60: Chaotic time series benchmark with v17 ManyToFewKernel.

Protocol (same as original Phase 7b, adapted for v17):
- Signal 1: Mackey-Glass (tau=17): 2000 steps ingestion + 2000 gen
- Signal 2: Lorenz-63: same protocol
- Shift detection: feed alternate signal after settling, report first spawn event
- d=64 ([researcher]'s spec)

Key questions:
- How many codebook vectors per attractor?
- Does codebook growth track attractor complexity?
- Does generation survive with fixed 8 matrix cells?
- Shift detection via first spawn event timing?
"""

import sys
import math
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import frob
from fluxcore_compressed_v17 import ManyToFewKernel, _vec_cosine

D = 64
N_STEPS = 2000
WINDOW = 200
AUTOREG_STEPS = 2000
SHIFT_STEPS = 500


# --- Signal Generators ---

def gen_mackey_glass(n_steps=3000, dt=0.1, tau_steps=170, x0=0.9):
    buf_size = tau_steps + n_steps + 1
    x = [x0] * buf_size
    for t in range(tau_steps, tau_steps + n_steps):
        xt = x[t]
        xt_tau = x[t - tau_steps]
        dx = 0.2 * xt_tau / (1.0 + xt_tau ** 10) - 0.1 * xt
        x[t + 1] = xt + dt * dx
    return x[tau_steps: tau_steps + n_steps]


def gen_lorenz(n_steps=3000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0,
               x0=1.0, y0=1.0, z0=1.0):
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0
    for _ in range(n_steps - 1):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx; y += dt * dy; z += dt * dz
        xs.append(x); ys.append(y); zs.append(z)
    return list(zip(xs, ys, zs))


# --- Embedding (d=64) ---

def embed_scalar(x, d=D):
    r = [math.sin(x * math.pi * i / d) for i in range(d)]
    norm = math.sqrt(sum(v * v for v in r) + 1e-15)
    return [v / norm for v in r]


def embed_lorenz(xyz, d=D):
    x, y, z = xyz
    xn = x / 25.0; yn = y / 25.0; zn = (z - 25.0) / 25.0
    feats = []
    per_axis = d // 3
    for v in [xn, yn, zn]:
        for i in range(per_axis):
            feats.append(math.sin(v * math.pi * (i + 1) / per_axis))
    while len(feats) < d:
        feats.append(math.cos(feats[-1]))
    feats = feats[:d]
    norm = math.sqrt(sum(v * v for v in feats) + 1e-15)
    return [v / norm for v in feats]


# --- Metrics ---

def max_cb_sim(kernel, r):
    if not kernel.codebook:
        return 0.0
    return max(_vec_cosine(v, r) for v in kernel.codebook)


def mean_energy(dMs):
    if not dMs:
        return 0.0
    return sum(frob(dM) ** 2 for dM in dMs) / len(dMs)


# --- Benchmark Functions ---

def run_ingestion(kernel, series, n_steps, window, label):
    """Feed n_steps, report codebook growth and max_sim per window."""
    print(f"\n  [{label}] Ingestion ({n_steps} steps, window={window})...")
    print(f"  {'Step':>6}  {'MaxSim':>8}  {'CB':>5}  {'Spawned':>8}  {'Energy':>10}")

    last_dMs = []
    spawn_windows = []  # spawns per window
    prev_spawned = kernel.total_spawned

    for w_start in range(0, n_steps, window):
        w_end = min(w_start + window, n_steps)
        max_sims = []
        for i in range(w_start, w_end):
            r = series[i]
            last_dMs = kernel.step(r=r)
            max_sims.append(max_cb_sim(kernel, r))

        mean_sim = sum(max_sims) / len(max_sims)
        new_spawns = kernel.total_spawned - prev_spawned
        spawn_windows.append(new_spawns)
        prev_spawned = kernel.total_spawned
        e = mean_energy(last_dMs)
        print(f"  {w_end:>6}  {mean_sim:>8.4f}  {len(kernel.codebook):>5}  {kernel.total_spawned:>8}  {e:>10.6f}")

    settled_sim = sum(max_sims[-window//2:]) / (window//2)  # last half of last window
    return last_dMs, settled_sim, spawn_windows


def run_autoregressive(kernel, n_steps, label):
    """Disconnect input, run n_steps, track energy survival."""
    print(f"\n  [{label}] Autoregressive generation ({n_steps} steps)...")
    energy_trace = []
    dMs = []
    for i in range(n_steps):
        dMs = kernel.step(r=None)
        if (i + 1) % 400 == 0:
            e = mean_energy(dMs)
            energy_trace.append((i + 1, e))
            print(f"    step {i+1:>5}: energy={e:.6f}")
    final_e = energy_trace[-1][1] if energy_trace else 0.0
    survived = final_e > 0.001
    print(f"  [{label}] Final energy={final_e:.6f}  SURVIVAL: {'YES' if survived else 'NO'}")
    return energy_trace, final_e, survived


def run_shift_detection(kernel, switch_series, settled_sim, n_steps, label):
    """
    Feed alternate signal. Report:
    - First spawn event (first step where max_sim < 0.5 AND we spawn)
    - Max_sim trace
    """
    print(f"\n  [{label}] Shift detection ({n_steps} steps of alternate signal)...")
    print(f"  {'Step':>6}  {'MaxSim':>8}  {'CB':>5}  {'NewSpawns':>10}")

    spawned_at_start = kernel.total_spawned
    first_spawn_step = None
    max_sims = []

    for i in range(n_steps):
        r = switch_series[i]
        cb_before = kernel.total_spawned
        kernel.step(r=r)
        ms = max_cb_sim(kernel, r)
        max_sims.append(ms)

        if first_spawn_step is None and kernel.total_spawned > cb_before:
            first_spawn_step = i + 1

        if (i + 1) % 100 == 0:
            new_spawns = kernel.total_spawned - spawned_at_start
            print(f"  {i+1:>6}  {ms:>8.4f}  {len(kernel.codebook):>5}  {new_spawns:>10}")

    total_spawns_shift = kernel.total_spawned - spawned_at_start
    spike_sim = sum(max_sims[:50]) / min(50, len(max_sims))  # avg first 50 steps
    print(f"\n  [{label}] First spawn step: {first_spawn_step or 'NONE'}")
    print(f"  [{label}] Total spawns during shift: {total_spawns_shift}")
    print(f"  [{label}] Pre-shift settled sim: {settled_sim:.4f}, post-shift sim@50: {spike_sim:.4f}")
    print(f"  [{label}] Sim drop: {settled_sim - spike_sim:.4f} {'DETECTED' if spike_sim < settled_sim * 0.9 else 'WEAK'}")

    return first_spawn_step, total_spawns_shift, spike_sim


# --- Main ---

def main():
    print("=" * 70)
    print("  Step 60 -- Chaotic Time Series Benchmark (v17 ManyToFewKernel)")
    print(f"  d={D}, n_matrix=8, k=4, spawn=0.5, lr_cb=0.015")
    print("=" * 70)

    t0 = time.time()

    print("\nGenerating signals...")
    mg_raw = gen_mackey_glass(n_steps=N_STEPS + SHIFT_STEPS)
    lorenz_raw = gen_lorenz(n_steps=N_STEPS + SHIFT_STEPS)
    mg_embedded = [embed_scalar(x) for x in mg_raw]
    lorenz_embedded = [embed_lorenz(xyz) for xyz in lorenz_raw]

    print(f"  MG: {len(mg_raw)} steps, range=[{min(mg_raw):.3f}, {max(mg_raw):.3f}]")
    print(f"  LZ: {len(lorenz_raw)} steps, x=[{min(p[0] for p in lorenz_raw):.2f}, {max(p[0] for p in lorenz_raw):.2f}]")

    # =========================================================
    # Signal 1: Mackey-Glass
    # =========================================================
    print("\n" + "-" * 70)
    print("  SIGNAL 1: Mackey-Glass (tau=17)")
    print("-" * 70)

    kernel_mg = ManyToFewKernel(
        n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
        tau=0.3, k_couple=5, spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015
    )

    mg_last_dMs, mg_settled_sim, mg_spawn_windows = run_ingestion(
        kernel_mg, mg_embedded, N_STEPS, WINDOW, "MG")
    mg_cb_size = len(kernel_mg.codebook)
    mg_total_spawned = kernel_mg.total_spawned

    mg_energy_trace, mg_final_energy, mg_survived = run_autoregressive(
        kernel_mg, AUTOREG_STEPS, "MG")

    mg_first_spawn, mg_shift_spawns, mg_spike_sim = run_shift_detection(
        kernel_mg, lorenz_embedded[:SHIFT_STEPS], mg_settled_sim, SHIFT_STEPS, "MG->Lorenz")

    # =========================================================
    # Signal 2: Lorenz-63
    # =========================================================
    print("\n" + "-" * 70)
    print("  SIGNAL 2: Lorenz-63")
    print("-" * 70)

    kernel_lz = ManyToFewKernel(
        n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
        tau=0.3, k_couple=5, spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015
    )

    lz_last_dMs, lz_settled_sim, lz_spawn_windows = run_ingestion(
        kernel_lz, lorenz_embedded, N_STEPS, WINDOW, "LZ")
    lz_cb_size = len(kernel_lz.codebook)
    lz_total_spawned = kernel_lz.total_spawned

    lz_energy_trace, lz_final_energy, lz_survived = run_autoregressive(
        kernel_lz, AUTOREG_STEPS, "LZ")

    lz_first_spawn, lz_shift_spawns, lz_spike_sim = run_shift_detection(
        kernel_lz, mg_embedded[:SHIFT_STEPS], lz_settled_sim, SHIFT_STEPS, "LZ->MG")

    # =========================================================
    # Summary
    # =========================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  STEP 60 SUMMARY")
    print("=" * 70)

    print(f"\n  Codebook growth (attractor complexity):")
    print(f"    Mackey-Glass: {mg_cb_size} vectors  ({mg_total_spawned} spawns, {kernel_mg.total_merged} merges)")
    print(f"    Lorenz-63:    {lz_cb_size} vectors  ({lz_total_spawned} spawns, {kernel_lz.total_merged} merges)")
    more_complex = "Lorenz" if lz_cb_size > mg_cb_size else "MG" if mg_cb_size > lz_cb_size else "Equal"
    print(f"    Higher complexity (more vectors): {more_complex}")

    print(f"\n  Generation survival (2000 steps):")
    print(f"    Mackey-Glass: {'YES' if mg_survived else 'NO'} (energy={mg_final_energy:.6f})")
    print(f"    Lorenz-63:    {'YES' if lz_survived else 'NO'} (energy={lz_final_energy:.6f})")

    print(f"\n  Shift detection (first spawn event):")
    print(f"    MG->Lorenz: first spawn at step {mg_first_spawn or 'NONE'}, {mg_shift_spawns} total spawns")
    print(f"    LZ->MG:     first spawn at step {lz_first_spawn or 'NONE'}, {lz_shift_spawns} total spawns")
    print(f"    MG sim drop (shifted->lorenz): {mg_settled_sim:.4f} -> {mg_spike_sim:.4f}")
    print(f"    LZ sim drop (shifted->MG):     {lz_settled_sim:.4f} -> {lz_spike_sim:.4f}")

    checks = [
        ("MG codebook grows (>1 vector)",          mg_cb_size > 1),
        ("LZ codebook grows (>1 vector)",           lz_cb_size > 1),
        ("LZ more complex than MG (more vectors)", lz_cb_size >= mg_cb_size),
        ("MG generation survives 2000",            mg_survived),
        ("LZ generation survives 2000",            lz_survived),
        ("MG->LZ shift detected (spawn event)",    mg_first_spawn is not None),
        ("LZ->MG shift detected (spawn event)",    lz_first_spawn is not None),
    ]

    n_pass = 0
    print(f"\n  Checks:")
    for name, ok in checks:
        sym = "PASS" if ok else "FAIL"
        print(f"    [{sym}] {name}")
        if ok: n_pass += 1
    print(f"\n  {n_pass}/{len(checks)} passed.")
    print(f"  Runtime: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()
