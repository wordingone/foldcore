#!/usr/bin/env python3
"""
ANIMA Extended Parameter Sweep — w_lr and gamma.

Extends the boundary-optimal results from Task 1 to find interior optima:
  - w_lr=0 → no W learning → I accumulates noise → gap should collapse
  - gamma=∞ → signal overwhelms dynamics → gap should collapse

If either parameter shows a clear interior optimum (rollover before boundary),
that is the Stage 2 adaptive target.

w_lr extended: [0.0001, 0.0003, 0.001, 0.003, 0.01]  (log-scale, low end)
gamma extended: [0.9, 1.2, 1.5, 2.0, 3.0]             (above canonical)

10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism
from run_anima_stage1 import run_sequence, measure_gap


CANONICAL = {
    'w_lr':   0.01,
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}

SEEDS = [42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111]
KS = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6
BIRTH_SEED = 42

SWEEPS = {
    'w_lr':  [0.0001, 0.0003, 0.001, 0.003, 0.01],
    'gamma': [0.9, 1.2, 1.5, 2.0, 3.0],
}


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m)**2 for v in vals) / max(len(vals) - 1, 1))
    return m, s


def measure_alive_gap_multi_k(rule_params):
    gaps = []
    for k in KS:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in SEEDS:
            org = AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=rule_params)
            g = measure_gap(org, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
            gaps.append(g)
    return gaps


def sweep_parameter(param_name, param_values):
    print(f"\n{'='*72}")
    print(f"  EXTENDED SWEEP: {param_name}")
    print(f"  Values: {param_values}")
    print(f"  Seeds: {len(SEEDS)}  K: {KS}  n_perm={N_PERM}  n_trials={N_TRIALS}")
    print(f"{'='*72}")

    results = []
    for val in param_values:
        rule_params = dict(CANONICAL)
        rule_params[param_name] = val

        gaps = measure_alive_gap_multi_k(rule_params)
        m, s = mean_std(gaps)
        results.append({'value': val, 'mean': m, 'std': s, 'n': len(gaps)})
        print(f"  {param_name}={val:.4f}:  alive_gap={m:+.4f}  std={s:.4f}  (n={len(gaps)})", flush=True)

    means = [r['mean'] for r in results]
    best_idx = means.index(max(means))
    is_boundary = (best_idx == 0 or best_idx == len(means) - 1)

    # Check for interior rollover: does gap decrease after the peak?
    has_rollover = False
    rollover_direction = None
    if best_idx > 0 and best_idx < len(means) - 1:
        # Interior peak
        has_rollover = True
    elif best_idx == 0:
        # Best at low end — check if it's clearly declining (rollover exists somewhere below range)
        rollover_direction = "below range"
    else:
        # Best at high end — rollover exists somewhere above range
        rollover_direction = "above range"

    # Monotonicity check
    diffs = [means[i+1] - means[i] for i in range(len(means)-1)]
    n_positive = sum(1 for d in diffs if d > 0)
    n_negative = sum(1 for d in diffs if d < 0)
    if n_positive > n_negative:
        monotone = "increasing"
    elif n_negative > n_positive:
        monotone = "decreasing"
    else:
        monotone = "non-monotone"

    print(f"\n  Best: {param_name}={param_values[best_idx]:.4f}  gap={max(means):+.4f}  (boundary={'YES' if is_boundary else 'NO — INTERIOR OPTIMUM'})")
    print(f"  Monotonicity: {monotone}  (diffs: {[f'{d:+.4f}' for d in diffs]})")
    if has_rollover:
        print(f"  *** INTERIOR OPTIMUM FOUND — Stage 2 adaptive target candidate ***")
    else:
        print(f"  Rollover expected: {rollover_direction}")

    verdict = "INTERIOR OPTIMUM" if has_rollover else f"BOUNDARY-OPTIMAL ({rollover_direction})"

    return {
        'param': param_name,
        'values': param_values,
        'results': results,
        'best_idx': best_idx,
        'best_value': param_values[best_idx],
        'best_mean': max(means),
        'is_boundary': is_boundary,
        'has_rollover': has_rollover,
        'monotone': monotone,
        'diffs': diffs,
        'verdict': verdict,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("  ANIMA Extended Parameter Sweep")
    print("  Goal: find interior optima for w_lr and gamma")
    print(f"  Canonical: {CANONICAL}")
    print("=" * 72)

    all_results = {}
    for param_name, param_values in SWEEPS.items():
        r = sweep_parameter(param_name, param_values)
        all_results[param_name] = r

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for name, r in all_results.items():
        print(f"  {name}:")
        print(f"    Best value: {r['best_value']}  gap={r['best_mean']:+.4f}")
        print(f"    Monotonicity: {r['monotone']}")
        print(f"    Verdict: {r['verdict']}")

    interior = [n for n, r in all_results.items() if r['has_rollover']]
    boundary = [n for n, r in all_results.items() if not r['has_rollover']]

    print()
    if interior:
        print(f"  INTERIOR OPTIMA FOUND: {interior}")
        print(f"  -> Stage 2 adaptive target(s): {interior}")
        print(f"  -> Proceed to Task 3 implementation")
    else:
        print(f"  No interior optima found: {boundary} remain boundary-optimal")
        print(f"  -> Calibrate to best values, prepare Stage 2 vacuous pass declaration")
    print("=" * 72)
