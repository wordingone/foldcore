#!/usr/bin/env python3
"""
ANIMA Stage 1 Ground Truth Test.

Tests whether AnimaOrganism produces MI gap > 0:
    still_gap  = gap for alive=False (no I accumulation — control)
    alive_gap  = gap for alive=True  (I accumulates prediction error)

MI gap > 0 means final states are sequence-distinguishable.

Methodology:
    - 10 seeds (c019: 10+ seeds required)
    - K = [4, 6, 8, 10] (c027: multi-K averaging)
    - n_perm = 8, n_trials = 6
    - delta = 1.0 (Session 16 canonical)

Imports infrastructure from harness.py (make_signals, gen_perms,
run_sequence, measure_gap, vcosine) but uses AnimaOrganism.
"""

import math
import random
import sys
import os

# Make src importable
sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism


# ═══════════════════════════════════════════════════════════════
# Adapted run_sequence for AnimaOrganism
# (mirrors harness.run_sequence exactly, swaps Organism for AnimaOrganism)
# ═══════════════════════════════════════════════════════════════

def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    """Run one sequence through org. Returns (centroid, xs)."""
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

    return org.centroid(xs), xs


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
    """
    MI gap = avg_within_cosine - avg_between_cosine.
    Positive means same-order trials cluster together
    (distinguishable final states per sequence).
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
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


# ═══════════════════════════════════════════════════════════════
# Main comparison: STILL vs ALIVE across 10 seeds x K=[4,6,8,10]
# ═══════════════════════════════════════════════════════════════

def run_anima_comparison(rule_params=None,
                         ks=None,
                         seeds=None,
                         birth_seed=42,
                         n_perm=8,
                         n_trials=6,
                         verbose=True):
    """
    Compare STILL (alive=False) vs ALIVE (alive=True) AnimaOrganism.

    Returns dict with still_gap, alive_gap, gap_delta, ground_truth_pass.
    """
    if rule_params is None:
        rule_params = {}
    if ks is None:
        ks = [4, 6, 8, 10]
    if seeds is None:
        # 10 seeds as required by c019
        seeds = [42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111]

    still_gaps = []
    alive_gaps = []

    total = len(ks) * len(seeds)
    done = 0

    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)

        for s in seeds:
            done += 1
            if verbose:
                print(f"  [{done:2d}/{total}] K={k} seed={s} ...", flush=True)

            # STILL: alive=False, no I accumulation
            org_still = AnimaOrganism(seed=birth_seed, alive=False, rule_params=rule_params)
            g_still = measure_gap(org_still, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            still_gaps.append(g_still)

            # ALIVE: alive=True, I accumulates prediction error
            org_alive = AnimaOrganism(seed=birth_seed, alive=True, rule_params=rule_params)
            g_alive = measure_gap(org_alive, sigs, k, s, n_perm=n_perm, n_trials=n_trials)
            alive_gaps.append(g_alive)

            if verbose:
                print(f"         still={g_still:+.4f}  alive={g_alive:+.4f}  delta={g_alive - g_still:+.4f}")

    still_avg = sum(still_gaps) / len(still_gaps)
    alive_avg = sum(alive_gaps) / len(alive_gaps)
    gap_delta = alive_avg - still_avg

    still_std = math.sqrt(sum((g - still_avg)**2 for g in still_gaps) / len(still_gaps))
    alive_std = math.sqrt(sum((g - alive_avg)**2 for g in alive_gaps) / len(alive_gaps))

    # Ground truth: alive gap > 0 (sequence-distinguishable final states)
    ground_truth_pass = alive_avg > 0.0

    # Rough d-statistic (alive_avg - still_avg) / pooled_std
    pooled_std = math.sqrt((still_std**2 + alive_std**2) / 2.0 + 1e-15)
    d_stat = gap_delta / pooled_std

    return {
        'still_gap': still_avg,
        'still_std': still_std,
        'alive_gap': alive_avg,
        'alive_std': alive_std,
        'gap_delta': gap_delta,
        'd_stat': d_stat,
        'ground_truth_pass': ground_truth_pass,
        'n_seeds': len(seeds),
        'ks': ks,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("  ANIMA STAGE 1 — Ground Truth Test")
    print("  W + I dynamics: does I accumulation create MI gap > 0?")
    print("=" * 72)
    print()

    rule_params = {
        'w_lr':   0.0003,  # Session 19: interior optimum
        'tau':    0.3,
        'gamma':  3.0,     # Session 19: boundary-optimal, new canonical
        'w_clip': 2.0,
        'noise':  0.005,
        'delta':  1.0,     # Session 16 canonical
    }

    print(f"Rule params: {rule_params}")
    print()

    result = run_anima_comparison(
        rule_params=rule_params,
        ks=[4, 6, 8, 10],
        seeds=[42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111],
        birth_seed=42,
        n_perm=8,
        n_trials=6,
        verbose=True,
    )

    print()
    print("=" * 72)
    print("  RESULTS")
    print("=" * 72)
    print(f"  STILL  gap: {result['still_gap']:+.4f}  (std={result['still_std']:.4f})")
    print(f"  ALIVE  gap: {result['alive_gap']:+.4f}  (std={result['alive_std']:.4f})")
    print(f"  Delta:      {result['gap_delta']:+.4f}")
    print(f"  d-stat:     {result['d_stat']:+.4f}")
    print(f"  Seeds:      {result['n_seeds']}  K-values: {result['ks']}")
    print()
    print(f"  Ground truth (alive_gap > 0): {'PASS' if result['ground_truth_pass'] else 'FAIL'}")
    print("=" * 72)
