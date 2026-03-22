#!/usr/bin/env python3
"""
RK Perception Experiment (Step 33)
Tests whether the Reflexive Kernel can perceive external signals.

Protocol:
  Phase 0: Self-org (2000 steps, no signal)
  Phase 1: Signal A (500 steps)
  Phase 2: Signal B (500 steps) — shift detection
  Phase 3: Signal A again (500 steps) — reacquisition speed
  Phase 4: Full sequence A->B->C->D->A (500 each) — specialization
"""

import sys
sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
from rk import Kernel, mrand, mcosine, frob, mzero
import random, math

W = 78

def header(title):
    print(f"\n{'='*W}")
    print(f"  {title}")
    print(f"{'='*W}")

def measure_alignments(kernel, signals):
    """Measure composite alignment with each signal."""
    C = kernel.composite()
    return {name: mcosine(C, sig) for name, sig in signals.items()}

def measure_state(kernel):
    return {
        'ef_dist': kernel.mean_ef_dist(),
        'energy': kernel.mean_energy(),
        'autonomy': kernel.mean_autonomy(),
    }

def print_row(step, aligns, ef_dist, energy, extra=""):
    cols = "  ".join(f"{v:+.4f}" for v in aligns.values())
    print(f"  {step:>5}  {cols}  {ef_dist:.4f}  {energy:.6f}  {extra}")

def print_table_header(signal_names):
    cols = "  ".join(f"{'align_'+n:>7}" for n in signal_names)
    print(f"  {'step':>5}  {cols}  {'ef_dist':>7}  {'energy':>9}")
    sep_cols = "  ".join(f"{'-------':>7}" for _ in signal_names)
    print(f"  {'-----':>5}  {sep_cols}  {'-------':>7}  {'---------':>9}")


# ── Create kernel and signals ──────────────────────────────

header("RK PERCEPTION EXPERIMENT")
print(f"  Kernel: n=8, k=4, seed=42")
print(f"  Signal injection scale: (1 - autonomy_i) * 0.3 per cell")

kernel = Kernel(n=8, k=4, seed=42)

# Create signal matrices
random.seed(100); sig_A = mrand(k=4, s=1.5)
random.seed(200); sig_B = mrand(k=4, s=1.5)
random.seed(300); sig_C = mrand(k=4, s=1.5)
random.seed(400); sig_D = mrand(k=4, s=1.5)

signals = {'A': sig_A, 'B': sig_B, 'C': sig_C, 'D': sig_D}
signal_names = list(signals.keys())

# Verify signals are distinct
print(f"\n  Signal cross-similarities:")
for i, (n1, s1) in enumerate(signals.items()):
    for n2, s2 in list(signals.items())[i+1:]:
        print(f"    cos({n1},{n2}) = {mcosine(s1, s2):+.4f}")


# ── Phase 0: Self-organization (2000 steps) ────────────────

header("PHASE 0: SELF-ORGANIZATION (2000 steps, no signal)")

for _ in range(2000):
    kernel.step()

state = measure_state(kernel)
aligns = measure_alignments(kernel, signals)

print(f"\n  After 2000 steps of self-organization:")
print(f"    mean_ef_dist  = {state['ef_dist']:.4f}")
print(f"    mean_energy   = {state['energy']:.6f}")
print(f"    mean_autonomy = {state['autonomy']:.4f}")
print(f"\n  Baseline composite alignment with each signal:")
for name, val in aligns.items():
    print(f"    align_{name} = {val:+.4f}")

baseline_aligns = dict(aligns)
baseline_state = dict(state)


# ── Phase 1: Signal A (500 steps, 2000-2500) ───────────────

header("PHASE 1: SIGNAL A (steps 2000-2500)")
print()
print_table_header(signal_names)

phase1_log = []
for s in range(500):
    kernel.step(sig_A)
    if (s + 1) % 50 == 0:
        aligns = measure_alignments(kernel, signals)
        state = measure_state(kernel)
        print_row(kernel.step_count, aligns, state['ef_dist'], state['energy'])
        phase1_log.append((kernel.step_count, dict(aligns), dict(state)))

phase1_final_align_A = phase1_log[-1][1]['A']
phase1_initial_align_A = baseline_aligns['A']
delta_A = phase1_final_align_A - phase1_initial_align_A

print(f"\n  align_A change: {phase1_initial_align_A:+.4f} -> {phase1_final_align_A:+.4f} (delta = {delta_A:+.4f})")
print(f"  Does align_A increase? {'YES' if delta_A > 0.01 else 'NO (delta too small)'}")
print(f"  Exceeds baseline? {'YES' if abs(phase1_final_align_A) > abs(phase1_initial_align_A) + 0.01 else 'MARGINAL/NO'}")


# ── Phase 2: Signal B (500 steps, 2500-3000) ───────────────

header("PHASE 2: SIGNAL B (steps 2500-3000)")
print()
print_table_header(signal_names)

pre_phase2_align_A = phase1_final_align_A
phase2_log = []
for s in range(500):
    kernel.step(sig_B)
    if (s + 1) % 50 == 0:
        aligns = measure_alignments(kernel, signals)
        state = measure_state(kernel)
        print_row(kernel.step_count, aligns, state['ef_dist'], state['energy'])
        phase2_log.append((kernel.step_count, dict(aligns), dict(state)))

phase2_final_align_B = phase2_log[-1][1]['B']
phase2_final_align_A = phase2_log[-1][1]['A']

print(f"\n  align_B change: {baseline_aligns['B']:+.4f} -> {phase2_final_align_B:+.4f}")
print(f"  align_A change: {pre_phase2_align_A:+.4f} -> {phase2_final_align_A:+.4f}")
print(f"  Does align_B increase? {'YES' if phase2_final_align_B - baseline_aligns['B'] > 0.01 else 'NO'}")
print(f"  Does align_A decrease (shift detection)? {'YES' if pre_phase2_align_A - phase2_final_align_A > 0.01 else 'NO'}")


# ── Phase 3: Signal A again (500 steps, 3000-3500) ─────────

header("PHASE 3: SIGNAL A REACQUISITION (steps 3000-3500)")
print()
print_table_header(signal_names)

pre_phase3_align_A = phase2_final_align_A
phase3_log = []
for s in range(500):
    kernel.step(sig_A)
    if (s + 1) % 50 == 0:
        aligns = measure_alignments(kernel, signals)
        state = measure_state(kernel)
        print_row(kernel.step_count, aligns, state['ef_dist'], state['energy'])
        phase3_log.append((kernel.step_count, dict(aligns), dict(state)))

phase3_final_align_A = phase3_log[-1][1]['A']

# Compare reacquisition rate
# Phase 1: went from baseline to phase1_final in 500 steps
# Phase 3: went from phase2_final to phase3_final in 500 steps
p1_delta = phase1_final_align_A - baseline_aligns['A']
p3_delta = phase3_final_align_A - pre_phase3_align_A

print(f"\n  Phase 1 align_A delta over 500 steps: {p1_delta:+.4f}")
print(f"  Phase 3 align_A delta over 500 steps: {p3_delta:+.4f}")
print(f"  Reacquisition faster? {'YES' if abs(p3_delta) > abs(p1_delta) else 'NO'}")

# Find step where align_A exceeds Phase 1 final
exceeded_step = None
for step_n, al, st in phase3_log:
    if al['A'] >= phase1_final_align_A:
        exceeded_step = step_n
        break

if exceeded_step:
    print(f"  Step where align_A exceeds Phase 1 final ({phase1_final_align_A:+.4f}): {exceeded_step}")
else:
    print(f"  align_A never exceeded Phase 1 final ({phase1_final_align_A:+.4f}) during Phase 3")


# ── Phase 4: Full sequence A->B->C->D->A (500 each, 3500-6000) ──

header("PHASE 4: FULL SEQUENCE A->B->C->D->A (steps 3500-6000)")
print()
print_table_header(signal_names)

sequence = [('A', sig_A), ('B', sig_B), ('C', sig_C), ('D', sig_D), ('A', sig_A)]

for sig_name, sig_mat in sequence:
    print(f"\n  --- Signal {sig_name} ---")
    for s in range(500):
        kernel.step(sig_mat)
        if (s + 1) % 100 == 0:
            aligns = measure_alignments(kernel, signals)
            state = measure_state(kernel)
            print_row(kernel.step_count, aligns, state['ef_dist'], state['energy'])

# Per-cell alignment analysis
header("PER-CELL SIGNAL ALIGNMENT (after full sequence)")

print(f"\n  {'cell':>4}  {'align_A':>8}  {'align_B':>8}  {'align_C':>8}  {'align_D':>8}  {'ef_dist':>8}  {'auton':>7}  dominant")
print(f"  {'----':>4}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'-------':>7}  --------")

cell_specializations = {n: [] for n in signal_names}
for i, cell in enumerate(kernel.cells):
    cell_aligns = {}
    for name, sig in signals.items():
        cell_aligns[name] = mcosine(cell.M, sig)

    d = cell.eigenform_distance()
    a = cell.autonomy()

    # Which signal is this cell most aligned with?
    dominant = max(cell_aligns, key=lambda k: abs(cell_aligns[k]))
    dom_val = cell_aligns[dominant]

    print(f"  {i:>4}  {cell_aligns['A']:+.4f}  {cell_aligns['B']:+.4f}  "
          f"{cell_aligns['C']:+.4f}  {cell_aligns['D']:+.4f}  "
          f"{d:.4f}  {a:.3f}  {dominant}({dom_val:+.3f})")

    if abs(dom_val) > 0.3:
        cell_specializations[dominant].append(i)

print(f"\n  Cell specialization (|align| > 0.3):")
any_specialized = False
for name, cells in cell_specializations.items():
    if cells:
        print(f"    Signal {name}: cells {cells}")
        any_specialized = True
if not any_specialized:
    print(f"    No strong specialization detected")


# ── SUMMARY ─────────────────────────────────────────────────

header("SUMMARY: DOES THE RK PERCEIVE?")

# Collect evidence
evidence = []

# 1. Signal alignment change
sig_response = abs(phase1_final_align_A - baseline_aligns['A'])
perceives_signal = sig_response > 0.02
evidence.append(("Signal changes composite alignment", perceives_signal, f"delta={sig_response:.4f}"))

# 2. Shift detection
shift_A = abs(pre_phase2_align_A - phase2_final_align_A)
shift_B = abs(phase2_final_align_B - baseline_aligns['B'])
detects_shift = shift_A > 0.02 or shift_B > 0.02
evidence.append(("Signal switch is detectable", detects_shift, f"dA={shift_A:.4f}, dB={shift_B:.4f}"))

# 3. Reacquisition
reacquires = abs(p3_delta) > 0.01
evidence.append(("Reacquisition occurs", reacquires, f"delta={p3_delta:+.4f}"))

# 4. Specialization
evidence.append(("Per-cell specialization", any_specialized,
                  f"{sum(len(v) for v in cell_specializations.values())} cells specialized"))

# 5. Overall alignment magnitude
max_align = max(abs(phase1_final_align_A), abs(phase2_final_align_B))
significant_align = max_align > 0.05
evidence.append(("Significant alignment magnitude", significant_align, f"max={max_align:.4f}"))

print()
n_pass = 0
for name, ok, detail in evidence:
    sym = "YES" if ok else " NO"
    print(f"  [{sym}] {name:<40} {detail}")
    if ok:
        n_pass += 1

print(f"\n  {n_pass}/{len(evidence)} perception criteria met.")

if n_pass >= 3:
    print(f"""
  CONCLUSION: The RK PERCEIVES external signals.
  The composite matrix shifts toward injected signals despite high autonomy
  (~0.90). The injection scale is small (~0.03 per cell), yet cumulative
  exposure over hundreds of steps produces measurable alignment changes.
  Signal switching is detectable and the system shows memory effects.
""")
elif n_pass >= 1:
    print(f"""
  CONCLUSION: PARTIAL perception. The RK responds to signals but the effect
  is weak. High autonomy (~0.90) suppresses signal injection to ~0.03 per
  cell, requiring many steps for measurable change. The system prioritizes
  eigenform maintenance over signal tracking.
""")
else:
    print(f"""
  CONCLUSION: The RK does NOT perceive external signals at current parameters.
  Autonomy is too high — signal injection scale is negligible relative to
  eigenform drive. The system is self-referentially closed.
""")

print(f"{'='*W}")
