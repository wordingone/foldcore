#!/usr/bin/env python3
"""
Step 59: Phase 7b concept drift benchmark with v17 ManyToFewKernel.

Protocol (same as original Phase 7b, adapted for v17):
- Phase 1: Warmup 1000 steps on distribution A
- Phase 2: Switch to B for 500 steps
- Phase 3: Switch back to A for 500 steps

d=384 to match CSI. Distributions A and B are nearly orthogonal (cos~0.1).

Key question: does the codebook's fold-style spawning detect concept drift?
Expected: max_sim drops below 0.5 -> spawn burst during drift. Spawn-rate IS the drift signal.

Metrics reported:
- Max codebook similarity per window (low = novel input)
- Spawn rate per window
- Generation energy before/during/after drift (50-step generation windows)
- Adaptation speed (spawn-rate returns to baseline)
- Reacquisition speed
"""

import sys
import math
import random
import time

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

from fluxcore_compressed_v17 import ManyToFewKernel, _vec_cosine, _norm, _normalize

SEED = 42
D = 384  # Match CSI embedding dimension

# Noise scaling: for d=384, noise=0.1 gives intra-distribution cos~0.2 (below spawn threshold).
# Scale to maintain cos~0.62 (same as original d=64, noise=0.1):
# cos ~ 1/(1 + sigma^2 * d) = 0.62 -> sigma = sqrt(0.38 / (0.62 * 384)) = 0.040
NOISE = 0.04


def make_unit_vector(seed):
    random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(D)]
    n = math.sqrt(sum(x * x for x in v) + 1e-15)
    return [x / n for x in v]


def sample_distribution(mu, noise=NOISE):
    """Sample from N(mu, noise*I) then normalize to unit sphere."""
    x = [mu[i] + random.gauss(0, noise) for i in range(D)]
    n = math.sqrt(sum(v * v for v in x) + 1e-15)
    return [v / n for v in x]


def max_codebook_sim(kernel, r):
    """Max cosine similarity between r and any codebook vector. Drift signal: low = novel."""
    if not kernel.codebook:
        return 0.0
    return max(_vec_cosine(v, r) for v in kernel.codebook)


def gen_energy(kernel, steps=50):
    """Run steps of generation (r=None), return mean energy of last dMs."""
    dMs = []
    for _ in range(steps):
        dMs = kernel.step(r=None)
    from rk import frob
    if not dMs:
        return 0.0
    return sum(frob(dM) ** 2 for dM in dMs) / len(dMs)


# Build distributions
mu_A = make_unit_vector(seed=1001)

# Build mu_B nearly orthogonal to mu_A (cos ~ 0.1)
random.seed(2002)
raw_B = [random.gauss(0, 1) for _ in range(D)]
dot_AB = sum(mu_A[i] * raw_B[i] for i in range(D))
raw_B = [raw_B[i] - dot_AB * mu_A[i] for i in range(D)]
raw_B = [raw_B[i] + 0.1 * mu_A[i] for i in range(D)]
nb = math.sqrt(sum(v * v for v in raw_B) + 1e-15)
mu_B = [v / nb for v in raw_B]

cos_AB = sum(mu_A[i] * mu_B[i] for i in range(D))

W = 72
print("=" * W)
print("  FluxCore Step 59 - Concept Drift Benchmark (v17 ManyToFewKernel)")
print(f"  d={D}, n_matrix=8, k=4, tau=0.3, spawn=0.5, lr_cb=0.015, noise={NOISE}")
print(f"  cos(mu_A, mu_B) = {cos_AB:.4f}  (target ~0.1, nearly orthogonal)")
print("=" * W)

kernel = ManyToFewKernel(
    n_matrix=8, k=4, d=D, seed=42, proj_seed=999,
    tau=0.3, k_couple=5,
    spawn_thresh=0.5, merge_thresh=0.95, lr_codebook=0.015
)

t0 = time.time()

# =============================================================================
# Phase 1: Warmup on Distribution A (1000 steps)
# =============================================================================
print("")
print("  Phase 1: Warmup on Distribution A (1000 steps)")
print(f"  {'Step':>6}  {'MaxSim':>8}  {'CB':>5}  {'Spawned':>8}  {'GenEnergy':>10}")

phase1_sims = []
random.seed(SEED)
for step in range(1, 1001):
    r = sample_distribution(mu_A)
    kernel.step(r=r)
    ms = max_codebook_sim(kernel, r)
    phase1_sims.append(ms)

    if step % 100 == 0:
        window = phase1_sims[-100:]
        mean_sim = sum(window) / len(window)
        cb_size = len(kernel.codebook)
        # Quick generation energy snapshot (10 steps only)
        from rk import frob
        dMs_snap = kernel.step(r=None)
        e_snap = sum(frob(dM) ** 2 for dM in dMs_snap) / len(dMs_snap) if dMs_snap else 0.0
        print(f"  {step:>6}  {mean_sim:>8.4f}  {cb_size:>5}  {kernel.total_spawned:>8}  {e_snap:>10.6f}")

settled_sim = sum(phase1_sims[-100:]) / 100
novelty_threshold = settled_sim * 0.85  # sim drops 15% = novel
spawn_baseline = kernel.total_spawned  # spawns during warmup
cb_after_warmup = len(kernel.codebook)

print("")
print(f"  Settled max_sim (last 100 of A): {settled_sim:.4f}")
print(f"  Novelty threshold (85% of settled): {novelty_threshold:.4f}")
print(f"  Codebook size after warmup: {cb_after_warmup}")
print(f"  Spawns during warmup: {spawn_baseline}")

# Baseline generation energy (after warmup, 50 steps)
e_before = gen_energy(kernel, steps=50)
print(f"  Generation energy (pre-drift, 50 steps): {e_before:.6f}")

# =============================================================================
# Phase 2: Switch to B (500 steps) — drift detection
# =============================================================================
print("")
print("  Phase 2: Distribution B — Drift Detection (500 steps)")
print(f"  {'Step':>6}  {'MaxSim':>8}  {'CB':>5}  {'NewSpawns':>10}  {'SpawnRate':>10}  {'Note':>14}")

phase2_sims = []
spawned_at_switch = kernel.total_spawned
spawn_burst_detected = None
adaptation_step = None  # step where sim re-settles near B's settled sim

random.seed(SEED + 1)
for step in range(1, 501):
    r = sample_distribution(mu_B)
    kernel.step(r=r)
    ms = max_codebook_sim(kernel, r)
    phase2_sims.append(ms)

    if step % 50 == 0:
        window = phase2_sims[-50:]
        mean_sim = sum(window) / len(window)
        new_spawns = kernel.total_spawned - spawned_at_switch
        spawn_rate = new_spawns / step  # spawns per step so far
        note = ""
        if mean_sim < novelty_threshold and spawn_burst_detected is None:
            spawn_burst_detected = step
            note = f"BURST@{step}"
        elif spawn_burst_detected:
            note = "adapting"
        cb_size = len(kernel.codebook)
        print(f"  {step:>6}  {mean_sim:>8.4f}  {cb_size:>5}  {new_spawns:>10}  {spawn_rate:>10.4f}  {note:>14}")

spawned_during_drift = kernel.total_spawned - spawned_at_switch
cb_growth_drift = len(kernel.codebook) - cb_after_warmup

# Adaptation: find when B sim re-settles (mean over 50-step window stabilizes > 0.5)
b_settled_sim = sum(phase2_sims[-100:]) / 100
for i in range(50, 500):
    window = phase2_sims[i-50:i]
    if sum(window) / len(window) > 0.5:
        adaptation_step = i - 50 + 1
        break

e_during = gen_energy(kernel, steps=50)
print("")
print(f"  Spawns during drift: {spawned_during_drift}  (CB grew by {cb_growth_drift})")
print(f"  B settled max_sim: {b_settled_sim:.4f}")
print(f"  Spawn burst detected at step: {spawn_burst_detected or 'none'}")
print(f"  Adaptation step (sim > 0.5): {adaptation_step or 'not settled'}")
print(f"  Generation energy (mid-drift, 50 steps): {e_during:.6f}")

# =============================================================================
# Phase 3: Back to A (500 steps) — reacquisition
# =============================================================================
print("")
print("  Phase 3: Back to Distribution A — Reacquisition (500 steps)")
print(f"  {'Step':>6}  {'MaxSim':>8}  {'CB':>5}  {'NewSpawns':>10}  {'SpawnRate':>10}")

phase3_sims = []
spawned_at_reacq = kernel.total_spawned
reacquisition_step = None

random.seed(SEED + 2)
for step in range(1, 501):
    r = sample_distribution(mu_A)
    kernel.step(r=r)
    ms = max_codebook_sim(kernel, r)
    phase3_sims.append(ms)

    if reacquisition_step is None and step > 20:
        window = phase3_sims[-20:]
        if sum(window) / len(window) >= settled_sim * 0.95:
            reacquisition_step = step

    if step % 50 == 0:
        window = phase3_sims[-50:]
        mean_sim = sum(window) / len(window)
        new_spawns = kernel.total_spawned - spawned_at_reacq
        spawn_rate = new_spawns / step
        cb_size = len(kernel.codebook)
        print(f"  {step:>6}  {mean_sim:>8.4f}  {cb_size:>5}  {new_spawns:>10}  {spawn_rate:>10.4f}")

spawned_during_reacq = kernel.total_spawned - spawned_at_reacq
e_after = gen_energy(kernel, steps=50)
a_settled_reacq = sum(phase3_sims[-100:]) / 100

print("")
print(f"  Reacquisition step (sim >= {settled_sim*0.95:.3f}): {reacquisition_step or 'not reacquired'}")
print(f"  Spawns during reacquisition: {spawned_during_reacq}")
print(f"  A re-settled max_sim: {a_settled_reacq:.4f}")
print(f"  Generation energy (post-drift, 50 steps): {e_after:.6f}")

# =============================================================================
# Summary
# =============================================================================
elapsed = time.time() - t0
print("")
print("=" * W)
print("  STEP 59 SUMMARY — v17 Concept Drift")
print("=" * W)

print(f"\n  Distribution separation: cos(A, B) = {cos_AB:.4f}")
print(f"  Settled max_sim on A: {settled_sim:.4f}")
print(f"  Settled max_sim on B: {b_settled_sim:.4f}")

print(f"\n  Codebook behavior:")
print(f"    After warmup (1000 steps A):  {cb_after_warmup} vectors, {spawn_baseline} spawns")
print(f"    After drift  (500 steps B):   {cb_after_warmup + cb_growth_drift} vectors (+{cb_growth_drift} spawns={spawned_during_drift})")
print(f"    After reacq  (500 steps A):   {len(kernel.codebook)} vectors (+{spawned_during_reacq} spawns)")
print(f"    Total: {kernel.total_spawned} spawns, {kernel.total_merged} merges")

print(f"\n  Generation energy:")
print(f"    Pre-drift (after warmup):  {e_before:.6f}")
print(f"    Mid-drift (after B phase): {e_during:.6f}")
print(f"    Post-drift (after reacq):  {e_after:.6f}")

print(f"\n  Adaptation timeline:")
print(f"    Spawn burst detected:  step {spawn_burst_detected or 'N/A'} of B phase")
print(f"    Adapted to B:          step {adaptation_step or 'N/A'} of B phase")
print(f"    Reacquired A:          step {reacquisition_step or 'N/A'} of A phase")

print(f"\n  Runtime: {elapsed:.1f}s")

# Verdict
drift_detected = spawn_burst_detected is not None
drift_fast = adaptation_step is not None and adaptation_step < 300
reacq_fast = reacquisition_step is not None and reacquisition_step < 200
spawn_surge = spawned_during_drift > spawn_baseline * 0.2  # drift spawned >20% of warmup total

print(f"\n  Codebook drift detection (spawn surge): {'YES' if spawn_surge else 'NO'} ({spawned_during_drift} spawns during drift)")
print(f"  Fast adaptation to B (<300 steps):     {'YES' if drift_fast else 'NO'}")
print(f"  Fast reacquisition of A (<200 steps):  {'YES' if reacq_fast else 'NO'}")

if drift_detected and drift_fast:
    print(f"\n  -> PASS: v17 codebook detects and adapts to concept drift via spawn-rate mechanism.")
elif drift_detected:
    print(f"\n  -> PARTIAL: Drift detected but slow adaptation.")
else:
    print(f"\n  -> CHECK: No spawn burst — codebook may have retained A coverage into B phase.")

print("")
print("=" * W)
print("  END Step 59")
print("=" * W)
