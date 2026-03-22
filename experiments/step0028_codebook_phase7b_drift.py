#!/usr/bin/env python3
"""
Phase 7b - Streaming Concept Drift Benchmark

Tests FluxCore CompressedKernel v2 on concept drift detection and adaptation.
Compares spawning kernel vs fixed-n baseline.
"""

import sys
import math
import random

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import mcosine, frob, mzero
from fluxcore_compressed_v2 import CompressedKernel

# Reproducibility
SEED = 42
random.seed(SEED)

# Distribution setup (d=64)
D = 64

def make_unit_vector(seed):
    random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(D)]
    n = math.sqrt(sum(x * x for x in v) + 1e-15)
    return [x / n for x in v]

def sample_distribution(mu, noise=0.1):
    """Sample from N(mu, noise*I) then normalize to unit sphere."""
    x = [mu[i] + random.gauss(0, noise) for i in range(D)]
    n = math.sqrt(sum(v * v for v in x) + 1e-15)
    return [v / n for v in x]

# Build mu_A and mu_B such that cos(A, B) ~ 0.1
mu_A = make_unit_vector(seed=1001)

# Build mu_B nearly orthogonal to mu_A: project random vector, remove A component
random.seed(2002)
raw_B = [random.gauss(0, 1) for _ in range(D)]
dot_AB = sum(mu_A[i] * raw_B[i] for i in range(D))
# Remove A component
raw_B = [raw_B[i] - dot_AB * mu_A[i] for i in range(D)]
# Add small A component to get cos ~ 0.1
raw_B = [raw_B[i] + 0.1 * mu_A[i] for i in range(D)]
nb = math.sqrt(sum(v * v for v in raw_B) + 1e-15)
mu_B = [v / nb for v in raw_B]

cos_AB = sum(mu_A[i] * mu_B[i] for i in range(D))

# Surprise measurement

def measure_surprise(kernel, r):
    """Mean surprise across all cells for input r."""
    return kernel.mean_surprise(r)

# Run one full scenario

def run_scenario(label, spawning):
    kernel = CompressedKernel(
        n=8, k=4, d=D,
        seed=42, proj_seed=999,
        alpha=1.2, beta=0.8,
        lr_base=0.08, k_s=20,
        sigma=0.3, tau=0.3,
        dt=0.03, noise_scale=0.01,
        max_norm=3.0, max_cells=500,
        spawning=spawning
    )

    results = {}

    # Phase 1: Warmup on distribution A (1000 steps)
    print("")
    print("  [%s] Phase 1: Warmup on Distribution A (1000 steps)" % label)
    print("  %6s  %10s  %6s  %8s" % ("Step", "Surprise", "Cells", "Spawned"))

    phase1_surprises = []

    random.seed(SEED)
    for step in range(1, 1001):
        r = sample_distribution(mu_A)
        kernel.step(r=r)
        s = measure_surprise(kernel, r)
        phase1_surprises.append(s)

        if step % 100 == 0:
            window = phase1_surprises[-100:]
            mean_w = sum(window) / len(window)
            n_cells = len(kernel.cells)
            print("  %6d  %10.4f  %6d  %8d" % (step, mean_w, n_cells, kernel.total_spawned))

    # Settled surprise = mean over last 100 steps of A
    settled_surprise = sum(phase1_surprises[-100:]) / 100
    detection_threshold = 1.5 * settled_surprise
    results['settled_surprise'] = settled_surprise
    results['cells_after_warmup'] = len(kernel.cells)
    results['spawned_after_warmup'] = kernel.total_spawned

    print("")
    print("  Settled surprise (last 100 of A): %.4f" % settled_surprise)
    print("  Detection threshold (1.5x):        %.4f" % detection_threshold)
    print("  Cells at end of warmup:            %d" % len(kernel.cells))

    # Phase 2: Switch to B (500 steps) - detect drift
    print("")
    print("  [%s] Phase 2: Distribution B - Drift Detection (500 steps)" % label)
    print("  %6s  %10s  %6s  %8s  %s" % ("Step", "Surprise", "Cells", "Spawned", "Note"))

    detection_step = None
    phase2_surprises = []
    spawned_at_switch = kernel.total_spawned

    random.seed(SEED + 1)
    for step in range(1, 501):
        r = sample_distribution(mu_B)
        kernel.step(r=r)
        s = measure_surprise(kernel, r)
        phase2_surprises.append(s)

        if detection_step is None and s > detection_threshold:
            detection_step = step

        if step % 50 == 0:
            window = phase2_surprises[-50:]
            mean_w = sum(window) / len(window)
            note = ("DETECTED@%d" % detection_step) if detection_step and step >= detection_step else ""
            n_cells = len(kernel.cells)
            print("  %6d  %10.4f  %6d  %8d  %s" % (step, mean_w, n_cells, kernel.total_spawned, note))

    results['detection_step'] = detection_step
    results['spawned_during_drift'] = kernel.total_spawned - spawned_at_switch
    results['cells_during_drift'] = len(kernel.cells) - results['cells_after_warmup']
    results['phase2_final_surprise'] = sum(phase2_surprises[-50:]) / 50

    # Adaptation speed: steps for surprise to re-settle (return to <= 1.2 * settled)
    adapt_thresh = 1.2 * settled_surprise
    adaptation_step = None
    for i, sv in enumerate(phase2_surprises):
        if i > 50 and sv <= adapt_thresh:  # wait past initial spike
            adaptation_step = i + 1
            break
    results['adaptation_step'] = adaptation_step

    print("")
    print("  Detection step (first spike > 1.5x settled): %s" % (detection_step or 'NOT DETECTED'))
    print("  Adaptation step (re-settle on B):            %s" % (adaptation_step or 'NOT SETTLED'))
    print("  New spawns during drift:                     %d" % results['spawned_during_drift'])

    # Phase 3: Switch back to A (500 steps) - reacquisition
    print("")
    print("  [%s] Phase 3: Back to Distribution A - Reacquisition (500 steps)" % label)
    print("  %6s  %10s  %6s  %8s" % ("Step", "Surprise", "Cells", "Spawned"))

    phase3_surprises = []
    spawned_at_reacq = kernel.total_spawned

    reacq_thresh = 1.2 * settled_surprise
    reacquisition_step = None

    random.seed(SEED + 2)
    for step in range(1, 501):
        r = sample_distribution(mu_A)
        kernel.step(r=r)
        s = measure_surprise(kernel, r)
        phase3_surprises.append(s)

        if reacquisition_step is None and s <= reacq_thresh:
            reacquisition_step = step

        if step % 50 == 0:
            window = phase3_surprises[-50:]
            mean_w = sum(window) / len(window)
            n_cells = len(kernel.cells)
            print("  %6d  %10.4f  %6d  %8d" % (step, mean_w, n_cells, kernel.total_spawned))

    results['reacquisition_step'] = reacquisition_step
    results['spawned_during_reacq'] = kernel.total_spawned - spawned_at_reacq
    results['final_cells'] = len(kernel.cells)
    results['total_spawned'] = kernel.total_spawned

    print("")
    print("  Reacquisition step (return to ~settled):     %s" % (reacquisition_step or 'NOT REACQUIRED'))
    print("  New spawns during reacquisition:             %d" % results['spawned_during_reacq'])
    print("  Total cells at end:                          %d" % results['final_cells'])
    print("  Total spawns over all phases:                %d" % results['total_spawned'])

    # Adaptation curve (B phase, per 50 steps)
    adapt_curve = []
    for chunk_start in range(0, 500, 50):
        chunk = phase2_surprises[chunk_start:chunk_start + 50]
        adapt_curve.append(sum(chunk) / len(chunk))
    results['adapt_curve'] = adapt_curve

    # Reacquisition curve (A phase, per 50 steps)
    reacq_curve = []
    for chunk_start in range(0, 500, 50):
        chunk = phase3_surprises[chunk_start:chunk_start + 50]
        reacq_curve.append(sum(chunk) / len(chunk))
    results['reacq_curve'] = reacq_curve

    return results


# =============================================================================
# Main
# =============================================================================

W = 72

print("=" * W)
print("  FluxCore Phase 7b - Streaming Concept Drift Benchmark")
print("  d=%d, n=8, k=4, tau=0.3" % D)
print("  cos(mu_A, mu_B) = %.4f  (target ~0.1, nearly orthogonal)" % cos_AB)
print("=" * W)

print("")
print("-" * W)
print("  SPAWNING KERNEL  (spawning=True, dynamic cells)")
print("-" * W)
spawning_results = run_scenario("SPAWNING", spawning=True)

print("")
print("-" * W)
print("  FIXED KERNEL  (spawning=False, n=8 fixed)")
print("-" * W)
fixed_results = run_scenario("FIXED", spawning=False)


# =============================================================================
# Comparison Summary
# =============================================================================

print("")
print("=" * W)
print("  COMPARISON SUMMARY")
print("=" * W)

print("")
print("  Distribution separation: cos(A, B) = %.4f" % cos_AB)
print("  Settled surprise (A baseline): %.4f" % spawning_results['settled_surprise'])

print("")
print("  %-40s  %10s  %10s" % ("Metric", "Spawning", "Fixed"))
print("  " + "-" * 40 + "  " + "-" * 10 + "  " + "-" * 10)

s = spawning_results
f = fixed_results

rows = [
    ("Settled surprise (mean last 100 of A)", s['settled_surprise'], f['settled_surprise']),
    ("Detection step (first spike >1.5x)",    s['detection_step'] or 9999,  f['detection_step'] or 9999),
    ("Adaptation step (re-settle on B)",      s['adaptation_step'] or 9999, f['adaptation_step'] or 9999),
    ("Reacquisition step (return to A)",      s['reacquisition_step'] or 9999, f['reacquisition_step'] or 9999),
    ("Spawns during drift (Phase 2)",         s['spawned_during_drift'], f['spawned_during_drift']),
    ("Spawns during reacquisition (Phase 3)", s['spawned_during_reacq'], f['spawned_during_reacq']),
    ("Total spawns (all phases)",             s['total_spawned'], f['total_spawned']),
    ("Final cell count",                      s['final_cells'], f['final_cells']),
]

for lbl, sv, fv in rows:
    if isinstance(sv, float):
        print("  %-40s  %10.4f  %10.4f" % (lbl, sv, fv))
    else:
        print("  %-40s  %10d  %10d" % (lbl, sv, fv))

# Adaptation curve comparison
print("")
print("  Adaptation Curve (surprise per 50 steps after switch to B):")
print("  %8s  %10s  %10s" % ("Steps", "Spawning", "Fixed"))
print("  " + "-" * 8 + "  " + "-" * 10 + "  " + "-" * 10)
for i, (sv, fv) in enumerate(zip(s['adapt_curve'], f['adapt_curve'])):
    step_range = "%d-%d" % (i * 50 + 1, (i + 1) * 50)
    print("  %8s  %10.4f  %10.4f" % (step_range, sv, fv))

# Reacquisition curve comparison
print("")
print("  Reacquisition Curve (surprise per 50 steps after switch back to A):")
print("  %8s  %10s  %10s" % ("Steps", "Spawning", "Fixed"))
print("  " + "-" * 8 + "  " + "-" * 10 + "  " + "-" * 10)
for i, (sv, fv) in enumerate(zip(s['reacq_curve'], f['reacq_curve'])):
    step_range = "%d-%d" % (i * 50 + 1, (i + 1) * 50)
    print("  %8s  %10.4f  %10.4f" % (step_range, sv, fv))

# Verdict
print("")
print("  " + "-" * W)
print("  VERDICT")
print("  " + "-" * W)

det_s = s['detection_step'] or 9999
det_f = f['detection_step'] or 9999
ada_s = s['adaptation_step'] or 9999
ada_f = f['adaptation_step'] or 9999
reacq_s = s['reacquisition_step'] or 9999
reacq_f = f['reacquisition_step'] or 9999

det_winner = "SPAWNING" if det_s <= det_f else "FIXED"
ada_winner = "SPAWNING" if ada_s <= ada_f else "FIXED"
reacq_winner = "SPAWNING" if reacq_s <= reacq_f else "FIXED"

print("")
print("  Detection latency:   %s faster (%s vs %s steps)" % (
    det_winner,
    det_s if det_s < 9999 else "N/A",
    det_f if det_f < 9999 else "N/A"))
print("  Adaptation speed:    %s faster (%s vs %s steps)" % (
    ada_winner,
    ada_s if ada_s < 9999 else "N/A",
    ada_f if ada_f < 9999 else "N/A"))
print("  Reacquisition speed: %s faster (%s vs %s steps)" % (
    reacq_winner,
    reacq_s if reacq_s < 9999 else "N/A",
    reacq_f if reacq_f < 9999 else "N/A"))
print("  Spawn cost:          %d total spawns (spawning kernel)" % s['total_spawned'])

advantages_spawning = sum([
    det_s <= det_f,
    ada_s <= ada_f,
    reacq_s <= reacq_f,
])
print("")
print("  Spawning kernel wins %d/3 primary metrics." % advantages_spawning)
if advantages_spawning >= 2:
    print("  -> Dynamic spawning HELPS concept drift adaptation.")
else:
    print("  -> Fixed kernel is competitive; spawning adds overhead without proportional gain.")

print("")
print("=" * W)
print("  END Phase 7b Benchmark")
print("=" * W)
