#!/usr/bin/env python3
"""
Compressed FluxCore+RK - 5-Test Suite

Tests autoregression, signal tracking, reacquisition,
shift detection, and perturbation recovery.
"""

import sys
import math
import random
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from rk import frob, mcosine, msub
from fluxcore_compressed import CompressedKernel


# ==============================================================
# Signal generation - use separate RNG to avoid entangling with kernel
# ==============================================================

def make_cluster_center(seed, d=64):
    rng = random.Random(seed)
    center = [rng.gauss(0, 1) for _ in range(d)]
    n = math.sqrt(sum(x * x for x in center))
    return [x / n for x in center]

def make_sampler(center, std=0.3, seed=None):
    """Return a function that generates samples from a cluster.
    Uses its own RNG to avoid entangling with kernel internals."""
    rng = random.Random(seed)
    def sample():
        return [c + rng.gauss(0, std) for c in center]
    return sample

CENTER_A = make_cluster_center(100)
CENTER_B = make_cluster_center(200)

W = 78
results = {}

print("=" * W)
print("  FluxCore Compressed - Unified FluxCore+RK Test Suite")
print("  Phi(M) = tanh(alpha*M + beta*M^2/k), perception via projected input")
print("=" * W)


def measure_surprise(kernel, center):
    """Measure mean surprise using the exact center (no noise)."""
    R = kernel.project(center)
    total = 0
    for c in kernel.cells:
        total += frob(msub(R, c)) / (frob(c) + 1e-15)
    return total / kernel.n


def measure_cosine_surprise(kernel, center):
    """Measure mean (1 - |cosine|) between projected signal and cells.
    More sensitive to directional changes than norm-based surprise."""
    R = kernel.project(center)
    total = 0
    for c in kernel.cells:
        total += 1.0 - abs(mcosine(R, c))
    return total / kernel.n


# ==============================================================
# TEST 1: Autoregression (generation)
# ==============================================================
print(f"\n{'-'*W}")
print("  TEST 1: AUTOREGRESSION (generation)")
print(f"  Seed with signal A for 100 steps, then disconnect for 10,000 steps")
print(f"{'-'*W}\n")

kernel = CompressedKernel(n=8, k=4, d=64, seed=42)
sample_a = make_sampler(CENTER_A, seed=77)

# Seed phase: 100 steps with signal A
for _ in range(100):
    kernel.step(sample_a())

print(f"  After seeding: ef_dist={kernel.mean_ef_dist():.4f}, autonomy={kernel.mean_autonomy():.4f}")

# Autoregression: 10,000 steps with no signal
print(f"\n  {'step':>6}  {'mean_ef_dist':>12}  {'mean_energy':>12}  {'mean_autonomy':>14}")
print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*14}")

last_energy = 0
for step in range(1, 10001):
    dMs = kernel.step(r=None)
    if step % 500 == 0:
        energy = kernel.mean_energy(dMs)
        ef = kernel.mean_ef_dist()
        au = kernel.mean_autonomy()
        print(f"  {step:>6}  {ef:>12.6f}  {energy:>12.8f}  {au:>14.6f}")
        last_energy = energy

pass1 = last_energy > 0.001
results['T1_autoregression'] = pass1
print(f"\n  Final mean_energy: {last_energy:.8f}")
print(f"  -> {'PASS' if pass1 else 'FAIL'} (threshold: > 0.001)")


# ==============================================================
# TEST 2: Signal tracking
# ==============================================================
print(f"\n{'-'*W}")
print("  TEST 2: SIGNAL TRACKING")
print(f"  Self-org 2000 steps, then feed signal A for 500 steps")
print(f"{'-'*W}\n")

kernel2 = CompressedKernel(n=8, k=4, d=64, seed=42)
sample_a2 = make_sampler(CENTER_A, seed=88)

# Self-organize
for _ in range(2000):
    kernel2.step(r=None)

print(f"  After self-org: ef_dist={kernel2.mean_ef_dist():.4f}, autonomy={kernel2.mean_autonomy():.4f}")

# Feed signal A for 500 steps
print(f"\n  {'step':>6}  {'surprise':>14}  {'cos_surprise':>14}")
print(f"  {'-'*6}  {'-'*14}  {'-'*14}")

surprise_at_50 = None
surprise_at_500 = None

for step in range(1, 501):
    kernel2.step(sample_a2())
    if step % 50 == 0:
        s = measure_surprise(kernel2, CENTER_A)
        cs = measure_cosine_surprise(kernel2, CENTER_A)
        print(f"  {step:>6}  {s:>14.6f}  {cs:>14.6f}")
        if step == 50:
            surprise_at_50 = s
        if step == 500:
            surprise_at_500 = s

pass2 = surprise_at_500 < surprise_at_50
results['T2_signal_tracking'] = pass2
print(f"\n  Surprise at step 50:  {surprise_at_50:.6f}")
print(f"  Surprise at step 500: {surprise_at_500:.6f}")
print(f"  -> {'PASS' if pass2 else 'FAIL'} (surprise must decrease)")


# ==============================================================
# TEST 3: Reacquisition
# ==============================================================
print(f"\n{'-'*W}")
print("  TEST 3: REACQUISITION")
print(f"  Signal A -> Signal B -> Signal A again")
print(f"{'-'*W}\n")

kernel3 = CompressedKernel(n=8, k=4, d=64, seed=42)
sample_a3 = make_sampler(CENTER_A, seed=99)
sample_b3 = make_sampler(CENTER_B, seed=101)

# Self-organize
for _ in range(2000):
    kernel3.step(r=None)

# First A exposure: 500 steps, track surprise trajectory
surprises_a1 = []
for step in range(1, 501):
    kernel3.step(sample_a3())
    if step % 50 == 0:
        s = measure_surprise(kernel3, CENTER_A)
        surprises_a1.append((step, s))

print("  First A exposure:")
for step, s in surprises_a1:
    print(f"    step {step:>4}: surprise = {s:.6f}")

# B exposure: 500 steps
for step in range(1, 501):
    kernel3.step(sample_b3())

print(f"\n  After B exposure: ef_dist={kernel3.mean_ef_dist():.4f}")

# Second A exposure: 500 steps (fresh sampler)
sample_a3b = make_sampler(CENTER_A, seed=102)
surprises_a2 = []
for step in range(1, 501):
    kernel3.step(sample_a3b())
    if step % 50 == 0:
        s = measure_surprise(kernel3, CENTER_A)
        surprises_a2.append((step, s))

print("\n  Second A exposure:")
for step, s in surprises_a2:
    print(f"    step {step:>4}: surprise = {s:.6f}")

# Compute convergence rate
def convergence_rate(trajectory):
    if len(trajectory) < 2:
        return 0
    return (trajectory[0][1] - trajectory[-1][1]) / len(trajectory)

rate_a1 = convergence_rate(surprises_a1)
rate_a2 = convergence_rate(surprises_a2)
final_a1 = surprises_a1[-1][1] if surprises_a1 else float('inf')
final_a2 = surprises_a2[-1][1] if surprises_a2 else float('inf')

faster = rate_a2 > rate_a1 * 1.2  # 20% faster convergence
lower_final = final_a2 < final_a1

pass3 = faster or lower_final
results['T3_reacquisition'] = pass3
print(f"\n  Rate A1: {rate_a1:.6f}, Rate A2: {rate_a2:.6f}")
print(f"  Final surprise A1: {final_a1:.6f}, A2: {final_a2:.6f}")
print(f"  20% faster: {faster}, Lower final: {lower_final}")
print(f"  -> {'PASS' if pass3 else 'FAIL'}")


# ==============================================================
# TEST 4: Shift detection
# ==============================================================
print(f"\n{'-'*W}")
print("  TEST 4: SHIFT DETECTION")
print(f"  Signal A settles, then switch to B - surprise must spike")
print(f"{'-'*W}\n")

kernel4 = CompressedKernel(n=8, k=4, d=64, seed=42)
sample_a4 = make_sampler(CENTER_A, seed=111)
sample_b4 = make_sampler(CENTER_B, seed=112)

# Self-organize
for _ in range(2000):
    kernel4.step(r=None)

# Feed signal A for 500 steps to settle
for step in range(500):
    kernel4.step(sample_a4())

# Measure settled surprise using center A
settled_surprise = measure_surprise(kernel4, CENTER_A)
settled_cos = measure_cosine_surprise(kernel4, CENTER_A)
print(f"  Settled surprise (A): norm={settled_surprise:.6f}, cos={settled_cos:.6f}")

# Switch to B - measure surprise at switch point
switch_surprise = measure_surprise(kernel4, CENTER_B)
switch_cos = measure_cosine_surprise(kernel4, CENTER_B)
print(f"  Switch surprise (B):  norm={switch_surprise:.6f}, cos={switch_cos:.6f}")

# Run 50 steps with B
for step in range(50):
    kernel4.step(sample_b4())

after_surprise = measure_surprise(kernel4, CENTER_B)
after_cos = measure_cosine_surprise(kernel4, CENTER_B)
print(f"  After 50 steps (B):   norm={after_surprise:.6f}, cos={after_cos:.6f}")

spike_ratio = switch_surprise / (settled_surprise + 1e-15)
pass4 = switch_surprise > settled_surprise * 1.5
results['T4_shift_detection'] = pass4
print(f"\n  Spike ratio: {spike_ratio:.4f}x")
print(f"  -> {'PASS' if pass4 else 'FAIL'} (threshold: spike > 1.5x settled)")


# ==============================================================
# TEST 5: Autoregression + perturbation
# ==============================================================
print(f"\n{'-'*W}")
print("  TEST 5: AUTOREGRESSION + PERTURBATION")
print(f"  Autoregress -> perturb with B -> autoregress again")
print(f"{'-'*W}\n")

kernel5 = CompressedKernel(n=8, k=4, d=64, seed=42)
sample_b5 = make_sampler(CENTER_B, seed=222)

# Pre-perturbation autoregression: 2000 steps
for _ in range(2000):
    kernel5.step(r=None)

# Measure pre-perturbation composite alignment with B
pre_alignment = kernel5.composite_alignment(CENTER_B)
pre_energy_samples = []
for _ in range(100):
    dMs = kernel5.step(r=None)
    pre_energy_samples.append(kernel5.mean_energy(dMs))
pre_energy = sum(pre_energy_samples) / len(pre_energy_samples)

print(f"  Pre-perturbation:")
print(f"    Composite alignment with B: {pre_alignment:.6f}")
print(f"    Mean energy (100 steps):     {pre_energy:.8f}")

# Inject signal B briefly: 100 steps
for _ in range(100):
    kernel5.step(sample_b5())

immed_alignment = kernel5.composite_alignment(CENTER_B)
print(f"\n  After B injection (100 steps):")
print(f"    Composite alignment with B: {immed_alignment:.6f}")

# Post-perturbation autoregression: 2000 steps
for _ in range(2000):
    kernel5.step(r=None)

post_alignment = kernel5.composite_alignment(CENTER_B)
post_energy_samples = []
for _ in range(100):
    dMs = kernel5.step(r=None)
    post_energy_samples.append(kernel5.mean_energy(dMs))
post_energy = sum(post_energy_samples) / len(post_energy_samples)

print(f"\n  Post-perturbation (2000 steps later):")
print(f"    Composite alignment with B: {post_alignment:.6f}")
print(f"    Mean energy (100 steps):     {post_energy:.8f}")

pass5 = post_alignment > pre_alignment
results['T5_perturbation'] = pass5
print(f"\n  Alignment increased: {pre_alignment:.6f} -> {post_alignment:.6f}")
print(f"  Energy changed: {pre_energy:.8f} -> {post_energy:.8f}")
print(f"  -> {'PASS' if pass5 else 'FAIL'} (post-alignment > pre-alignment)")


# ==============================================================
# SUMMARY
# ==============================================================
print(f"\n{'='*W}")
print("  COMPRESSED FLUXCORE+RK - TEST RESULTS")
print(f"{'='*W}\n")

for name, passed in results.items():
    print(f"  {name:<25} {'PASS' if passed else 'FAIL'}")

total = len(results)
passed_count = sum(1 for v in results.values() if v)
print(f"\n  {passed_count}/{total} tests passed")
print(f"{'='*W}")
