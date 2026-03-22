#!/usr/bin/env python3
"""
ANIMA Stage 2 Validation — Phase-dependent adaptive w_lr vs best fixed (0.0003).

Stage 2 exit criterion:
    1. Adaptive alive_gap > Fixed alive_gap (paired d > 0)
    2. Statistical significance: d > 0.5 or p < 0.05
    3. Ground truth: adaptive alive_gap > 0

FIXED baseline: w_lr=0.0003 (best fixed from extended sweep, alive_gap=+0.0948).
ADAPTIVE: AnimaOrganismAdaptive with phase-dependent w_lr_eff.

Tests multiple w_lr_base/err_scale combinations to find effective parameters.

10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6.
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism
from anima_organism_adaptive import AnimaOrganismAdaptive
from run_anima_stage1 import measure_gap


SEEDS = [42, 137, 256, 512, 1024, 2024, 7777, 31337, 99991, 111111]
KS = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6
BIRTH_SEED = 42

# Fixed baseline: best fixed w_lr from extended sweep
FIXED_PARAMS = {
    'w_lr':   0.0003,
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}

# Adaptive configurations to try
ADAPTIVE_CONFIGS = [
    {'w_lr_base': 0.001,  'err_scale': 3.0,   'label': 'base=0.001 scale=3'},
    {'w_lr_base': 0.001,  'err_scale': 10.0,  'label': 'base=0.001 scale=10'},
    {'w_lr_base': 0.001,  'err_scale': 30.0,  'label': 'base=0.001 scale=30'},
    {'w_lr_base': 0.0003, 'err_scale': 3.0,   'label': 'base=0.0003 scale=3'},
    {'w_lr_base': 0.0003, 'err_scale': 10.0,  'label': 'base=0.0003 scale=10'},
]

BASE_ADAPTIVE_PARAMS = {
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m)**2 for v in vals) / max(len(vals) - 1, 1))
    return m, s


def run_sequence_adaptive(org, order, signals, base_seed, trial,
                          n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for _ in range(n_org):
        xs = org.step(xs)
    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)
    for _ in range(n_final):
        xs = org.step(xs)
    return org.centroid(xs), xs


def measure_gap_adaptive(org, signals, k, seed, n_perm=8, n_trials=6):
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence_adaptive(org, perm, signals, seed, trial)
            trials.append(c)
        endpoints[pi] = trials
    within, between = [], []
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


def t_test_paired(diffs):
    n = len(diffs)
    if n < 2:
        return 0.0, 1.0
    m = sum(diffs) / n
    s = math.sqrt(sum((d - m)**2 for d in diffs) / (n - 1))
    if s < 1e-15:
        return (float('inf') if m > 0 else float('-inf')), 0.0
    t = m / (s / math.sqrt(n))
    p_two = 2 * 0.5 * math.erfc(abs(t) / math.sqrt(2))
    return t, min(p_two, 1.0)


def run_adaptive_config(cfg, fixed_gaps):
    """Run one adaptive config against the already-computed fixed_gaps."""
    params = dict(BASE_ADAPTIVE_PARAMS)
    params['w_lr_base'] = cfg['w_lr_base']
    params['err_scale'] = cfg['err_scale']

    adaptive_gaps = []
    w_lr_eff_means = []
    total = len(KS) * len(SEEDS)
    done = 0

    for k in KS:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in SEEDS:
            done += 1
            print(f"    [{done:2d}/{total}] K={k} seed={s} ...", end=' ', flush=True)
            org = AnimaOrganismAdaptive(seed=BIRTH_SEED, alive=True, rule_params=params)
            g = measure_gap_adaptive(org, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
            adaptive_gaps.append(g)
            stats = org.get_w_lr_eff_stats()
            w_lr_eff_means.append(stats['mean'])
            print(f"gap={g:+.4f}  w_lr_eff=[{stats['min']:.5f},{stats['max']:.5f}]",
                  flush=True)

    diffs = [a - f for a, f in zip(adaptive_gaps, fixed_gaps)]
    adap_mean, adap_std = mean_std(adaptive_gaps)
    fixed_mean, fixed_std = mean_std(fixed_gaps)
    diff_mean, diff_std = mean_std(diffs)
    pooled_std = math.sqrt((fixed_std**2 + adap_std**2) / 2.0 + 1e-15)
    d_stat = (adap_mean - fixed_mean) / pooled_std
    t_stat, p_val = t_test_paired(diffs)
    wlr_mean = sum(w_lr_eff_means) / len(w_lr_eff_means)

    c1 = adap_mean > fixed_mean
    c2 = d_stat > 0.5 or p_val < 0.05
    c3 = adap_mean > 0.0
    passed = c1 and c2 and c3

    print(f"    => adaptive={adap_mean:+.4f}  fixed={fixed_mean:+.4f}  "
          f"d={d_stat:+.3f}  p={p_val:.4f}  w_lr_eff_mean={wlr_mean:.5f}  "
          f"{'PASS' if passed else 'fail'}")

    return {
        'label': cfg['label'],
        'adaptive_mean': adap_mean,
        'adaptive_std': adap_std,
        'diff_mean': diff_mean,
        'd_stat': d_stat,
        't_stat': t_stat,
        'p_val': p_val,
        'w_lr_eff_mean': wlr_mean,
        'c1': c1, 'c2': c2, 'c3': c3,
        'passed': passed,
    }


if __name__ == '__main__':
    print("=" * 72)
    print("  ANIMA Stage 2 Validation — Phase-adaptive w_lr vs Fixed")
    print(f"  Fixed baseline: w_lr=0.0003 (best fixed from extended sweep)")
    print(f"  Seeds: {SEEDS}")
    print(f"  K: {KS}  n_perm={N_PERM}  n_trials={N_TRIALS}")
    print("=" * 72)

    # ── Compute fixed baseline once ───────────────────────────
    print("\n[Computing FIXED baseline w_lr=0.0003 ...]")
    fixed_gaps = []
    total = len(KS) * len(SEEDS)
    done = 0
    for k in KS:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in SEEDS:
            done += 1
            print(f"  [{done:2d}/{total}] K={k} seed={s} ...", end=' ', flush=True)
            org = AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=FIXED_PARAMS)
            g = measure_gap(org, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
            fixed_gaps.append(g)
            print(f"gap={g:+.4f}", flush=True)
    fixed_mean, fixed_std = mean_std(fixed_gaps)
    print(f"\n  FIXED mean={fixed_mean:+.4f}  std={fixed_std:.4f}\n")

    # ── Test each adaptive config ─────────────────────────────
    all_results = []
    for cfg in ADAPTIVE_CONFIGS:
        print(f"\n[Adaptive config: {cfg['label']}]")
        r = run_adaptive_config(cfg, fixed_gaps)
        all_results.append(r)

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  FIXED (w_lr=0.0003): {fixed_mean:+.4f}  (std={fixed_std:.4f})")
    print()
    print(f"  {'Config':<25}  {'Adaptive':>9}  {'d':>7}  {'p':>7}  {'w_lr_eff':>9}  Result")
    print(f"  {'-'*25}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*9}  ------")
    for r in all_results:
        status = 'PASS' if r['passed'] else 'fail'
        print(f"  {r['label']:<25}  {r['adaptive_mean']:>+9.4f}  {r['d_stat']:>+7.3f}  "
              f"{r['p_val']:>7.4f}  {r['w_lr_eff_mean']:>9.5f}  {status}")

    passed = [r for r in all_results if r['passed']]
    print()
    if passed:
        best = max(passed, key=lambda r: r['d_stat'])
        print(f"  STAGE 2: PASS — best config: {best['label']}")
        print(f"  d={best['d_stat']:+.3f}  p={best['p_val']:.4f}")
    else:
        best = max(all_results, key=lambda r: r['d_stat'])
        print(f"  STAGE 2: FAIL across all configs")
        print(f"  Best d={best['d_stat']:+.3f} ({best['label']})")
        print(f"  -> Signal does not beat best fixed w_lr at current parameterization")
    print("=" * 72)
