#!/usr/bin/env python3
"""
ANIMA Parameter Binding Diagnostic.

Tests whether w_lr, tau, gamma are binding constraints on MI gap (alive_gap).

For each parameter, vary it across a range while holding others at canonical values.
Use 5 seeds (paired), K=[4,6,8,10], n_perm=8, n_trials=6.

A parameter is binding if range(MI gaps) > 2 * within-condition std.
Boundary-optimal = calibration only (c035 lesson).

Canonical values:
    w_lr=0.01, tau=0.3, gamma=0.9

Parameter grids:
    w_lr:  [0.001, 0.005, 0.01, 0.02, 0.05]
    tau:   [0.1, 0.2, 0.3, 0.5, 0.7]
    gamma: [0.3, 0.5, 0.7, 0.9, 1.2]
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism
from run_anima_stage1 import run_sequence, measure_gap


# ═══════════════════════════════════════════════════════════════
# Canonical values and parameter grids
# ═══════════════════════════════════════════════════════════════

CANONICAL = {
    'w_lr':   0.01,
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}

PARAM_GRIDS = {
    'w_lr':  [0.001, 0.005, 0.01, 0.02, 0.05],
    'tau':   [0.1, 0.2, 0.3, 0.5, 0.7],
    'gamma': [0.3, 0.5, 0.7, 0.9, 1.2],
}

SEEDS = [42, 137, 2024, 7777, 99991]
KS = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6
BIRTH_SEED = 42


# ═══════════════════════════════════════════════════════════════
# Measure alive_gap for a given rule_params config
# ═══════════════════════════════════════════════════════════════

def measure_alive_gap_multi_k(rule_params, seeds, ks, n_perm, n_trials):
    """
    Measure alive_gap averaged over seeds x K.
    Returns list of per-(seed,K) gaps for statistics.
    """
    gaps = []
    for k in ks:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in seeds:
            org = AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=rule_params)
            g = measure_gap(org, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            gaps.append(g)
    return gaps


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m)**2 for v in vals) / len(vals))
    return m, s


# ═══════════════════════════════════════════════════════════════
# Binding diagnostic for one parameter
# ═══════════════════════════════════════════════════════════════

def diagnose_parameter(param_name, param_values, verbose=True):
    """
    For each value in param_values, measure alive_gap with all other params at canonical.
    Returns dict with per-value results and binding verdict.
    """
    if verbose:
        print(f"\n{'='*72}")
        print(f"  PARAMETER: {param_name}")
        print(f"  Values: {param_values}")
        print(f"{'='*72}")

    results = []
    for val in param_values:
        rule_params = dict(CANONICAL)
        rule_params[param_name] = val

        gaps = measure_alive_gap_multi_k(rule_params, SEEDS, KS, N_PERM, N_TRIALS)
        m, s = mean_std(gaps)
        results.append({
            'value': val,
            'mean': m,
            'std': s,
            'n': len(gaps),
            'gaps': gaps,
        })

        if verbose:
            print(f"  {param_name}={val:.4f}:  alive_gap={m:+.4f}  std={s:.4f}  (n={len(gaps)})")

    # Compute binding verdict
    means = [r['mean'] for r in results]
    stds = [r['std'] for r in results]
    gap_range = max(means) - min(means)
    avg_std = sum(stds) / len(stds)

    # Binding criterion: range > 2 * avg_within_std
    is_binding = gap_range > 2.0 * avg_std

    # Check for boundary-optimal (best at edge of range)
    best_idx = means.index(max(means))
    is_boundary_optimal = (best_idx == 0 or best_idx == len(means) - 1)

    verdict = "BINDING"
    if not is_binding:
        verdict = "NON-BINDING"
    elif is_boundary_optimal:
        verdict = "BINDING (boundary-optimal = calibration only)"

    if verbose:
        print(f"\n  Range: {gap_range:.4f}   Avg std: {avg_std:.4f}   Ratio: {gap_range/max(avg_std,1e-9):.2f}")
        print(f"  Best value: {param_values[best_idx]} (idx={best_idx}, boundary={'YES' if is_boundary_optimal else 'NO'})")
        print(f"  Verdict: {verdict}")

    return {
        'param': param_name,
        'values': param_values,
        'results': results,
        'gap_range': gap_range,
        'avg_std': avg_std,
        'ratio': gap_range / max(avg_std, 1e-9),
        'is_binding': is_binding,
        'is_boundary_optimal': is_boundary_optimal,
        'verdict': verdict,
        'best_value': param_values[best_idx],
        'best_mean': max(means),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  ANIMA Parameter Binding Diagnostic")
    print("  Parameters: w_lr, tau, gamma")
    print(f"  Seeds: {SEEDS}  (5 paired seeds)")
    print(f"  K: {KS}  n_perm={N_PERM}  n_trials={N_TRIALS}")
    print(f"  Canonical: {CANONICAL}")
    print("=" * 72)

    all_results = {}
    for param_name, param_values in PARAM_GRIDS.items():
        diag = diagnose_parameter(param_name, param_values, verbose=True)
        all_results[param_name] = diag

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  {'Parameter':<10}  {'Range':>8}  {'AvgStd':>8}  {'Ratio':>7}  {'Verdict'}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*35}")
    for name, r in all_results.items():
        print(f"  {name:<10}  {r['gap_range']:>8.4f}  {r['avg_std']:>8.4f}  {r['ratio']:>7.2f}  {r['verdict']}")

    print()
    print("  Binding = range > 2x avg_within_std")
    print("  Boundary-optimal = best at edge of range = calibration only (c035)")
    print()

    binding = [n for n, r in all_results.items() if r['is_binding'] and not r['is_boundary_optimal']]
    calibration = [n for n, r in all_results.items() if r['is_binding'] and r['is_boundary_optimal']]
    non_binding = [n for n, r in all_results.items() if not r['is_binding']]

    print(f"  Genuinely binding:    {binding if binding else '(none)'}")
    print(f"  Calibration-only:     {calibration if calibration else '(none)'}")
    print(f"  Non-binding:          {non_binding if non_binding else '(none)'}")
    print()

    # Best values for Stage 2 adaptive candidates
    print("  Best values per parameter (for Stage 2 adaptive seed):")
    for name, r in all_results.items():
        print(f"    {name} = {r['best_value']}  (alive_gap={r['best_mean']:+.4f})")

    print("=" * 72)
