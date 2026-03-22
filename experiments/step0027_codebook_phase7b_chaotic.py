#!/usr/bin/env python3
"""
Phase 7b -- Chaotic Time Series Benchmark

Tests FluxCore CompressedKernel v2 against:
  Signal 1: Mackey-Glass (tau=17) -- classic strange attractor
  Signal 2: Lorenz-63 -- 3D chaotic system

Metrics:
  - Adaptation curve (surprise vs step, per 100 steps)
  - Cell count (spawning response to attractor complexity)
  - Mean autonomy (generation quality)
  - Generation survival (2000 steps autoregressive, energy threshold)
  - Shift detection (surprise spike when switching signals)
"""

import sys
import math

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob, mzero
from fluxcore_compressed_v2 import CompressedKernel

# -------------------------------------------------------------
# Signal Generation
# -------------------------------------------------------------

def gen_mackey_glass(n_steps=3000, dt=0.1, tau_steps=170, x0=0.9):
    """Mackey-Glass delay differential equation, discretized."""
    # Buffer long enough for delay
    buf_size = tau_steps + n_steps + 1
    x = [x0] * buf_size

    # Run from index tau_steps onward
    for t in range(tau_steps, tau_steps + n_steps):
        xt = x[t]
        xt_tau = x[t - tau_steps]
        dx = 0.2 * xt_tau / (1.0 + xt_tau ** 10) - 0.1 * xt
        x[t + 1] = xt + dt * dx

    series = x[tau_steps: tau_steps + n_steps]
    return series


def gen_lorenz(n_steps=3000, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0,
               x0=1.0, y0=1.0, z0=1.0):
    """Lorenz-63 system, Euler integration."""
    xs, ys, zs = [x0], [y0], [z0]
    x, y, z = x0, y0, z0
    for _ in range(n_steps - 1):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x = x + dt * dx
        y = y + dt * dy
        z = z + dt * dz
        xs.append(x); ys.append(y); zs.append(z)
    return list(zip(xs, ys, zs))


# -------------------------------------------------------------
# Embedding Functions
# -------------------------------------------------------------

def embed_scalar(x, d=16):
    """Embed scalar into R^d via sin features, then normalize."""
    r = [math.sin(x * math.pi * i / d) for i in range(d)]
    norm = math.sqrt(sum(v * v for v in r) + 1e-15)
    return [v / norm for v in r]


def embed_lorenz(xyz, d=16):
    """Embed (x,y,z) into R^d via sin/cos features, then normalize."""
    x, y, z = xyz
    # Normalize each axis to roughly [-1, 1]
    # Lorenz: x,y in [-20,20], z in [0,50]
    xn = x / 25.0
    yn = y / 25.0
    zn = (z - 25.0) / 25.0

    feats = []
    vals = [xn, yn, zn]
    per_axis = d // len(vals)
    for v in vals:
        for i in range(per_axis):
            feats.append(math.sin(v * math.pi * (i + 1) / per_axis))
    # Pad to d if needed
    while len(feats) < d:
        feats.append(math.cos(feats[-1]))
    feats = feats[:d]

    norm = math.sqrt(sum(v * v for v in feats) + 1e-15)
    return [v / norm for v in feats]


# -------------------------------------------------------------
# Benchmark Helpers
# -------------------------------------------------------------

def compute_mean_surprise(kernel, r):
    """Mean surprise across all cells for vector r."""
    return kernel.mean_surprise(r)


def run_warmup_and_adapt(kernel, embedded_series, warmup=1500, adapt=1500,
                          window=100, label=""):
    """
    Feed warmup steps (no metrics), then collect adaptation metrics
    over 'adapt' steps in windows of 'window'.
    Returns: list of (step, mean_surprise, cell_count, mean_autonomy)
    """
    # Warmup
    print(f"  [{label}] Warming up ({warmup} steps)...", flush=True)
    for i in range(warmup):
        r = embedded_series[i]
        kernel.step(r=r)

    # Collect final warmup stats
    warmup_cells = len(kernel.cells)
    warmup_au = kernel.mean_autonomy()
    last_r = embedded_series[warmup - 1]
    warmup_surp = compute_mean_surprise(kernel, last_r)

    print(f"  [{label}] Warmup done: cells={warmup_cells}, "
          f"autonomy={warmup_au:.4f}, surprise={warmup_surp:.4f}")

    # Adaptation phase with windowed metrics
    print(f"  [{label}] Adaptation phase ({adapt} steps, window={window})...")
    adapt_curve = []
    for w_start in range(0, adapt, window):
        w_end = min(w_start + window, adapt)
        surprises = []
        for i in range(w_start, w_end):
            idx = warmup + i
            r = embedded_series[idx]
            kernel.step(r=r)
            surprises.append(compute_mean_surprise(kernel, r))
        mean_surp = sum(surprises) / len(surprises)
        cells_now = len(kernel.cells)
        au_now = kernel.mean_autonomy()
        adapt_curve.append((warmup + w_end, mean_surp, cells_now, au_now))
        print(f"    step {warmup + w_end:>5}: surprise={mean_surp:.4f}  "
              f"cells={cells_now}  autonomy={au_now:.4f}")

    return warmup_cells, warmup_au, warmup_surp, adapt_curve


def run_autoregressive(kernel, n_steps=2000, threshold=0.001, label=""):
    """
    Disconnect input, run n_steps autoregressive. Track energy survival.
    Returns list of (step, energy) sampled every 200 steps, plus final energy.
    """
    print(f"  [{label}] Autoregressive generation ({n_steps} steps)...", flush=True)
    energy_trace = []
    for i in range(n_steps):
        dMs = kernel.step(r=None)
        if (i + 1) % 200 == 0:
            e = kernel.mean_energy(dMs)
            energy_trace.append((i + 1, e))
            print(f"    step {i+1:>5}: energy={e:.6f}")

    final_energy = energy_trace[-1][1] if energy_trace else 0.0
    survived = final_energy > threshold
    print(f"  [{label}] Final energy={final_energy:.6f}, "
          f"threshold={threshold}, survived={'YES' if survived else 'NO'}")
    return energy_trace, final_energy, survived


def run_shift_detection(kernel, switch_series, settled_surprise,
                         n_steps=500, label=""):
    """
    Feed the OTHER signal for n_steps. Measure surprise spike.
    Returns: list of surprises, ratio vs settled.
    """
    print(f"  [{label}] Shift detection ({n_steps} steps of alt signal)...", flush=True)
    surprises = []
    for i in range(n_steps):
        r = switch_series[i]
        kernel.step(r=r)
        surprises.append(compute_mean_surprise(kernel, r))

    # Spike = mean of first 100 steps vs settled
    spike_window = surprises[:100]
    spike_mean = sum(spike_window) / len(spike_window)
    ratio = spike_mean / (settled_surprise + 1e-15)
    print(f"  [{label}] Shift spike (first 100): {spike_mean:.4f}, "
          f"ratio vs settled={ratio:.2f}x")
    return surprises, spike_mean, ratio


# -------------------------------------------------------------
# Main Benchmark
# -------------------------------------------------------------

def main():
    D = 16
    N_STEPS = 3000
    WARMUP = 1500
    ADAPT = 1500
    WINDOW = 100
    AUTOREG_STEPS = 2000
    ENERGY_THRESH = 0.001
    SHIFT_STEPS = 500

    print("=" * 70)
    print("  Phase 7b -- Chaotic Time Series Benchmark")
    print("  FluxCore CompressedKernel v2")
    print("=" * 70)

    # -- Generate signals --------------------------------------
    print("\nGenerating signals...")
    mg_raw = gen_mackey_glass(n_steps=N_STEPS + SHIFT_STEPS)
    lorenz_raw = gen_lorenz(n_steps=N_STEPS + SHIFT_STEPS)

    mg_embedded = [embed_scalar(x, d=D) for x in mg_raw]
    lorenz_embedded = [embed_lorenz(xyz, d=D) for xyz in lorenz_raw]

    print(f"  Mackey-Glass: {len(mg_raw)} steps, "
          f"range=[{min(mg_raw):.3f}, {max(mg_raw):.3f}]")
    print(f"  Lorenz-63:    {len(lorenz_raw)} steps, "
          f"x-range=[{min(p[0] for p in lorenz_raw):.2f}, {max(p[0] for p in lorenz_raw):.2f}]")

    # ---------------------------------------------------------
    # Signal 1: Mackey-Glass
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SIGNAL 1: Mackey-Glass (tau=17)")
    print("-" * 70)

    kernel_mg = CompressedKernel(n=8, k=4, d=D, spawning=True, tau=0.3, seed=42)

    mg_warmup_cells, mg_warmup_au, mg_warmup_surp, mg_adapt_curve = \
        run_warmup_and_adapt(kernel_mg, mg_embedded,
                             warmup=WARMUP, adapt=ADAPT, window=WINDOW,
                             label="MG")

    # Final settled surprise = last window mean surprise
    mg_settled_surp = mg_adapt_curve[-1][1]
    mg_final_cells = mg_adapt_curve[-1][2]
    mg_final_au = mg_adapt_curve[-1][3]

    # Autoregressive
    mg_energy_trace, mg_final_energy, mg_survived = \
        run_autoregressive(kernel_mg, n_steps=AUTOREG_STEPS,
                           threshold=ENERGY_THRESH, label="MG")

    # Shift: feed Lorenz signal
    mg_shift_surprises, mg_spike_mean, mg_shift_ratio = \
        run_shift_detection(kernel_mg, lorenz_embedded[:SHIFT_STEPS],
                            n_steps=SHIFT_STEPS,
                            settled_surprise=mg_settled_surp, label="MG->Lorenz")

    # ---------------------------------------------------------
    # Signal 2: Lorenz-63
    # ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("  SIGNAL 2: Lorenz-63")
    print("-" * 70)

    kernel_lz = CompressedKernel(n=8, k=4, d=D, spawning=True, tau=0.3, seed=42)

    lz_warmup_cells, lz_warmup_au, lz_warmup_surp, lz_adapt_curve = \
        run_warmup_and_adapt(kernel_lz, lorenz_embedded,
                             warmup=WARMUP, adapt=ADAPT, window=WINDOW,
                             label="LZ")

    lz_settled_surp = lz_adapt_curve[-1][1]
    lz_final_cells = lz_adapt_curve[-1][2]
    lz_final_au = lz_adapt_curve[-1][3]

    # Autoregressive
    lz_energy_trace, lz_final_energy, lz_survived = \
        run_autoregressive(kernel_lz, n_steps=AUTOREG_STEPS,
                           threshold=ENERGY_THRESH, label="LZ")

    # Shift: feed Mackey-Glass signal
    lz_shift_surprises, lz_spike_mean, lz_shift_ratio = \
        run_shift_detection(kernel_lz, mg_embedded[:SHIFT_STEPS],
                            n_steps=SHIFT_STEPS,
                            settled_surprise=lz_settled_surp, label="LZ->MG")

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    out("=" * 70)
    out("  Phase 7b -- Chaotic Time Series Benchmark")
    out("  FluxCore CompressedKernel v2")
    out("  n=8 start, k=4, spawning=True, tau=0.3, d=16")
    out("=" * 70)

    # Signal stats
    out(f"\nSignal stats:")
    out(f"  Mackey-Glass: {N_STEPS} steps, range=[{min(mg_raw):.3f}, {max(mg_raw):.3f}]")
    out(f"  Lorenz-63:    {N_STEPS} steps, "
        f"x=[{min(p[0] for p in lorenz_raw):.2f},{max(p[0] for p in lorenz_raw):.2f}] "
        f"y=[{min(p[1] for p in lorenz_raw):.2f},{max(p[1] for p in lorenz_raw):.2f}] "
        f"z=[{min(p[2] for p in lorenz_raw):.2f},{max(p[2] for p in lorenz_raw):.2f}]")

    # -- MG results --
    out("\n" + "-" * 70)
    out("  SIGNAL 1: Mackey-Glass (tau=17)")
    out("-" * 70)
    out(f"\n  Warmup ({WARMUP} steps):")
    out(f"    cells={mg_warmup_cells}  autonomy={mg_warmup_au:.4f}  surprise={mg_warmup_surp:.4f}")
    out(f"\n  Adaptation curve (step / mean_surprise / cells / autonomy):")
    for step, surp, cells, au in mg_adapt_curve:
        out(f"    step {step:>5}: surprise={surp:.4f}  cells={cells}  autonomy={au:.4f}")
    out(f"\n  Final state (step {WARMUP + ADAPT}):")
    out(f"    cells={mg_final_cells}  autonomy={mg_final_au:.4f}  settled_surprise={mg_settled_surp:.4f}")

    # Surprise trend
    first_surp = mg_adapt_curve[0][1]
    last_surp = mg_adapt_curve[-1][1]
    surp_trend = "DECREASING" if last_surp < first_surp else "STABLE/INCREASING"
    out(f"    surprise trend: {first_surp:.4f} -> {last_surp:.4f} ({surp_trend})")

    out(f"\n  Autoregressive generation ({AUTOREG_STEPS} steps):")
    for step, e in mg_energy_trace:
        out(f"    step {step:>5}: energy={e:.6f}")
    out(f"    Final energy={mg_final_energy:.6f}  threshold={ENERGY_THRESH}")
    out(f"    Survived: {'YES' if mg_survived else 'NO'}")

    out(f"\n  Shift detection (MG -> Lorenz, {SHIFT_STEPS} steps):")
    out(f"    Settled surprise (MG):   {mg_settled_surp:.4f}")
    out(f"    Spike surprise (first 100 steps Lorenz): {mg_spike_mean:.4f}")
    out(f"    Shift ratio: {mg_shift_ratio:.2f}x  "
        f"({'DETECTED' if mg_shift_ratio > 1.5 else 'WEAK'})")

    # -- Lorenz results --
    out("\n" + "-" * 70)
    out("  SIGNAL 2: Lorenz-63")
    out("-" * 70)
    out(f"\n  Warmup ({WARMUP} steps):")
    out(f"    cells={lz_warmup_cells}  autonomy={lz_warmup_au:.4f}  surprise={lz_warmup_surp:.4f}")
    out(f"\n  Adaptation curve (step / mean_surprise / cells / autonomy):")
    for step, surp, cells, au in lz_adapt_curve:
        out(f"    step {step:>5}: surprise={surp:.4f}  cells={cells}  autonomy={au:.4f}")
    out(f"\n  Final state (step {WARMUP + ADAPT}):")
    out(f"    cells={lz_final_cells}  autonomy={lz_final_au:.4f}  settled_surprise={lz_settled_surp:.4f}")

    first_surp_lz = lz_adapt_curve[0][1]
    last_surp_lz = lz_adapt_curve[-1][1]
    surp_trend_lz = "DECREASING" if last_surp_lz < first_surp_lz else "STABLE/INCREASING"
    out(f"    surprise trend: {first_surp_lz:.4f} -> {last_surp_lz:.4f} ({surp_trend_lz})")

    out(f"\n  Autoregressive generation ({AUTOREG_STEPS} steps):")
    for step, e in lz_energy_trace:
        out(f"    step {step:>5}: energy={e:.6f}")
    out(f"    Final energy={lz_final_energy:.6f}  threshold={ENERGY_THRESH}")
    out(f"    Survived: {'YES' if lz_survived else 'NO'}")

    out(f"\n  Shift detection (Lorenz -> MG, {SHIFT_STEPS} steps):")
    out(f"    Settled surprise (Lorenz): {lz_settled_surp:.4f}")
    out(f"    Spike surprise (first 100 steps MG): {lz_spike_mean:.4f}")
    out(f"    Shift ratio: {lz_shift_ratio:.2f}x  "
        f"({'DETECTED' if lz_shift_ratio > 1.5 else 'WEAK'})")

    # -- Cross-signal comparison --
    out("\n" + "-" * 70)
    out("  CROSS-SIGNAL COMPARISON")
    out("-" * 70)
    out(f"  Mackey-Glass: start_cells=8  final_cells={mg_final_cells}  "
        f"(+{mg_final_cells - 8} spawned)")
    out(f"  Lorenz-63:    start_cells=8  final_cells={lz_final_cells}  "
        f"(+{lz_final_cells - 8} spawned)")
    more_complex = "Lorenz" if lz_final_cells > mg_final_cells else \
                   "Mackey-Glass" if mg_final_cells > lz_final_cells else "Equal"
    out(f"  Higher attractor complexity (more cells): {more_complex}")

    out(f"\n  Generation survival:")
    out(f"    Mackey-Glass: {'YES' if mg_survived else 'NO'} (energy={mg_final_energy:.6f})")
    out(f"    Lorenz-63:    {'YES' if lz_survived else 'NO'} (energy={lz_final_energy:.6f})")

    out(f"\n  Shift detection (ratio > 1.5 = detected):")
    out(f"    MG->Lorenz: {mg_shift_ratio:.2f}x  ({'DETECTED' if mg_shift_ratio > 1.5 else 'WEAK'})")
    out(f"    LZ->MG:     {lz_shift_ratio:.2f}x  ({'DETECTED' if lz_shift_ratio > 1.5 else 'WEAK'})")

    # -- Summary verdict --
    out("\n" + "=" * 70)
    out("  VERDICT")
    out("=" * 70)

    checks = [
        ("MG surprise decreasing",        last_surp < first_surp),
        ("Lorenz surprise decreasing",     last_surp_lz < first_surp_lz),
        ("MG spawning active",             mg_final_cells > 8),
        ("Lorenz spawning active",         lz_final_cells > 8),
        ("MG generation survives 2000",    mg_survived),
        ("Lorenz generation survives 2000", lz_survived),
        ("MG shift detected (>1.5x)",      mg_shift_ratio > 1.5),
        ("LZ shift detected (>1.5x)",      lz_shift_ratio > 1.5),
    ]

    n_pass = 0
    for name, ok in checks:
        sym = "PASS" if ok else "FAIL"
        out(f"  [{sym}] {name}")
        if ok:
            n_pass += 1

    out(f"\n  {n_pass}/{len(checks)} criteria passed.")
    out("=" * 70)

    # Write results file
    results_path = "B:/M/avir/research/fluxcore/results/phase7b_chaotic.txt"
    with open(results_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults written to: {results_path}")


if __name__ == '__main__':
    main()
