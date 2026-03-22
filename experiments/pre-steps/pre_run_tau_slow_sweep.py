#!/usr/bin/env python3
"""
tau_slow Binding Sweep — Stage 3 Forward Path.

Sweeps tau_slow values and measures alive_gap to determine whether
the slow-timescale I accumulator (mem_slow) is a binding constraint.

Protocol (canonical):
    tau_slow values: [0.0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1]
    Seeds:           [42, 137, 2024, 999, 7, 314, 1618, 2718, 1414, 577]
    K values:        [4, 6, 8, 10]
    n_perm:          8
    n_trials:        6

Canonical params (Session 21): w_lr=0.0003, gamma=3.0, tau=0.3, delta=1.0.
tau_slow=0.0 is the control — must reproduce baseline alive_gap.

Output:
    - Per-condition results (tau_slow, K, seed, alive_gap)
    - Summary table: tau_slow vs mean_alive_gap (across seeds x K), std, min, max
    - Monotonicity check across tau_slow values
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism


# ===============================================================
# Sweep parameters
# ===============================================================

TAU_SLOW_VALUES = [0.0, 0.001, 0.003, 0.005, 0.01, 0.03, 0.1]
SEEDS = [42, 137, 2024, 999, 7, 314, 1618, 2718, 1414, 577]
K_VALUES = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6

# Canonical params (Session 21)
BASE_PARAMS = {
    'w_lr':   0.0003,
    'tau':    0.3,
    'gamma':  3.0,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}


# ===============================================================
# run_sequence and measure_gap (mirrors run_anima_stage1.py)
# ===============================================================

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    """Run one sequence through org. Returns centroid."""
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    # Organism warm-up (no signal)
    for _ in range(n_org):
        xs = org.step(xs)

    # Signal sequence
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)

    # Final settle
    for _ in range(n_final):
        xs = org.step(xs)

    return org.centroid(xs)


def measure_gap(rule_params, signals, k, seed, n_perm=N_PERM, n_trials=N_TRIALS):
    """
    Measure alive_gap for an AnimaOrganism with given rule_params.

    Returns alive_gap (avg_within_cosine - avg_between_cosine).
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}

    org = AnimaOrganism(seed=42, alive=True, rule_params=rule_params)

    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c = run_sequence(org, perm, signals, seed, trial)
            trials.append(c)
        endpoints[pi] = trials

    within = []
    between = []
    pis = sorted(endpoints.keys())

    for pi in pis:
        cs = endpoints[pi]
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                within.append(vcosine(cs[i], cs[j]))

    for i in range(len(pis)):
        for j in range(i + 1, len(pis)):
            for c1 in endpoints[pis[i]]:
                for c2 in endpoints[pis[j]]:
                    between.append(vcosine(c1, c2))

    avg_w = sum(within) / max(len(within), 1)
    avg_b = sum(between) / max(len(between), 1)
    return avg_w - avg_b


# ===============================================================
# Main sweep
# ===============================================================

def run_tau_slow_sweep():
    print("=" * 72)
    print("  tau_slow BINDING SWEEP — Stage 3 Forward Path")
    print(f"  tau_slow values: {TAU_SLOW_VALUES}")
    print(f"  seeds ({len(SEEDS)}): {SEEDS}")
    print(f"  K values: {K_VALUES}")
    print(f"  n_perm={N_PERM}  n_trials={N_TRIALS}")
    print(f"  canonical params: {BASE_PARAMS}")
    print("=" * 72)
    print()

    # results[tau_slow] = list of alive_gap values (one per seed x K condition)
    results = {}
    # detailed: keyed by (tau_slow, seed, k)
    detailed = {}

    total = len(TAU_SLOW_VALUES) * len(SEEDS) * len(K_VALUES)
    done = 0

    for tau_slow in TAU_SLOW_VALUES:
        rule_params = dict(BASE_PARAMS, tau_slow=tau_slow)
        gaps = []

        for k in K_VALUES:
            sig_seed = 42 + k * 200
            sigs = make_signals(k, seed=sig_seed)

            for seed in SEEDS:
                done += 1
                print(f"  [{done:3d}/{total}] tau_slow={tau_slow:.4f}  K={k}  seed={seed} ...",
                      end=" ", flush=True)

                gap = measure_gap(rule_params, sigs, k, seed)
                gaps.append(gap)
                detailed[(tau_slow, seed, k)] = gap

                print(f"alive_gap={gap:+.4f}")

        results[tau_slow] = gaps
        mean_gap = sum(gaps) / len(gaps)
        print(f"  -> tau_slow={tau_slow:.4f}  mean_alive_gap={mean_gap:+.4f}  "
              f"(n={len(gaps)} conditions)")
        print()

    # ===============================================================
    # Summary table
    # ===============================================================

    print()
    print("=" * 72)
    print("  SUMMARY TABLE (mean across seeds x K)")
    print("=" * 72)
    print(f"  {'tau_slow':>10}  {'mean_gap':>10}  {'std':>8}  {'min':>10}  {'max':>10}  {'n':>4}")
    print("  " + "-" * 60)

    summary = {}
    for tau_slow in TAU_SLOW_VALUES:
        gaps = results[tau_slow]
        n = len(gaps)
        mean = sum(gaps) / n
        variance = sum((g - mean) ** 2 for g in gaps) / n
        std = math.sqrt(variance)
        gmin = min(gaps)
        gmax = max(gaps)
        summary[tau_slow] = {'mean': mean, 'std': std, 'min': gmin, 'max': gmax, 'n': n}
        print(f"  {tau_slow:>10.4f}  {mean:>+10.4f}  {std:>8.4f}  {gmin:>+10.4f}  {gmax:>+10.4f}  {n:>4}")

    # ===============================================================
    # Monotonicity check
    # ===============================================================

    print()
    print("=" * 72)
    print("  MONOTONICITY CHECK (mean_alive_gap across tau_slow values)")
    print("=" * 72)

    means = [summary[t]['mean'] for t in TAU_SLOW_VALUES]
    monotone_inc = all(means[i] <= means[i + 1] for i in range(len(means) - 1))
    monotone_dec = all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    if monotone_inc:
        verdict = "MONOTONE_INCREASING"
    elif monotone_dec:
        verdict = "MONOTONE_DECREASING"
    else:
        # Find peak
        peak_idx = means.index(max(means))
        peak_val = TAU_SLOW_VALUES[peak_idx]
        verdict = f"non-monotone (peak at tau_slow={peak_val})"

    # Show values
    vals_str = "  ".join(f"{m:+.4f}" for m in means)
    print(f"  [{vals_str}]")
    print(f"  Verdict: {verdict}")

    # Binding check: is any tau_slow > 0 better than tau_slow = 0 baseline?
    baseline = summary[0.0]['mean']
    best_nonzero_tau = max(TAU_SLOW_VALUES[1:], key=lambda t: summary[t]['mean'])
    best_nonzero_mean = summary[best_nonzero_tau]['mean']
    improvement = best_nonzero_mean - baseline
    baseline_std = summary[0.0]['std']

    print()
    print(f"  Baseline (tau_slow=0.0):  mean_gap={baseline:+.4f}  std={baseline_std:.4f}")
    print(f"  Best tau_slow > 0:        tau_slow={best_nonzero_tau}  mean_gap={best_nonzero_mean:+.4f}")
    print(f"  Improvement over baseline: {improvement:+.4f}")

    # Rough effect size (improvement / baseline_std)
    if baseline_std > 0:
        effect_size = improvement / baseline_std
        print(f"  Effect size (d = delta/std): {effect_size:+.3f}")
        if abs(effect_size) > 0.5:
            binding_verdict = "BINDING (effect > 0.5 SD)"
        elif abs(effect_size) > 0.2:
            binding_verdict = "WEAKLY_BINDING (0.2 < effect <= 0.5 SD)"
        else:
            binding_verdict = "NON-BINDING (effect <= 0.2 SD)"
    else:
        binding_verdict = "INDETERMINATE (baseline std = 0)"

    print()
    print(f"  Binding verdict: {binding_verdict}")
    print("=" * 72)

    return results, summary, detailed


if __name__ == '__main__':
    run_tau_slow_sweep()
