#!/usr/bin/env python3
"""
ANIMA Stage 2 Final Test + Calibrated Baseline.

Part 1 — Reactive w_lr vs Fixed 0.0003:
    FIXED:    AnimaOrganism w_lr=0.0003 (best fixed from Session 19 extended sweep)
    REACTIVE: AnimaOrganismAdaptive w_lr_base=0.001, err_scale=2.0
              w_lr_eff = 0.001 / (1 + 2.0 * prev_mean_abs_err)
              At typical err~0.35: w_lr_eff ≈ 0.001/1.7 ≈ 0.00059
              At high err~1.0:    w_lr_eff ≈ 0.001/3.0 ≈ 0.00033 (near optimum)
              At low err~0.05:    w_lr_eff ≈ 0.001/1.1 ≈ 0.00091

    If REACTIVE beats FIXED → Stage 2 passes.
    If not → Stage 2 vacuous (Amendment 1).

Part 2 — Calibrated Baseline:
    AnimaOrganism w_lr=0.0003, gamma=3.0
    10 seeds, K=[4,6,8,10]
    Establishes new canonical MI gap for Session 19 record.
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

FIXED_PARAMS = {
    'w_lr':   0.0003,
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}

REACTIVE_PARAMS = {
    'w_lr_base': 0.001,
    'err_scale': 2.0,
    'tau':    0.3,
    'gamma':  0.9,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}

CALIBRATED_PARAMS = {
    'w_lr':   0.0003,
    'tau':    0.3,
    'gamma':  3.0,
    'w_clip': 2.0,
    'noise':  0.005,
    'delta':  1.0,
}


def mean_std(vals):
    m = sum(vals) / len(vals)
    s = math.sqrt(sum((v - m)**2 for v in vals) / max(len(vals) - 1, 1))
    return m, s


def measure_gap_adaptive(org, signals, k, seed, n_perm=8, n_trials=6):
    """measure_gap for AnimaOrganismAdaptive (uses its own step/centroid)."""
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)

    def run_sequence(perm, trial):
        random.seed(seed)
        xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
        for _ in range(300):
            xs = org.step(xs)
        for idx, sid in enumerate(perm):
            random.seed(seed * 1000 + sid * 100 + idx * 10 + trial)
            sig = [signals[sid][k2] + random.gauss(0, 0.05) for k2 in range(D)]
            for _ in range(60):
                xs = org.step(xs, sig)
            for _ in range(30):
                xs = org.step(xs)
        for _ in range(60):
            xs = org.step(xs)
        return org.centroid(xs)

    endpoints = {}
    for pi, perm in enumerate(perms):
        endpoints[pi] = [run_sequence(perm, t) for t in range(n_trials)]

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


def run_condition(label, make_org_fn, use_adaptive=False):
    """Run all seeds/K for one condition. Returns list of gaps (len=len(KS)*len(SEEDS))."""
    gaps = []
    total = len(KS) * len(SEEDS)
    done = 0
    for k in KS:
        sig_seed = BIRTH_SEED + k * 200
        sigs = make_signals(k, seed=sig_seed)
        for s in SEEDS:
            done += 1
            print(f"  [{done:2d}/{total}] K={k} seed={s} ...", end=' ', flush=True)
            org = make_org_fn()
            if use_adaptive:
                g = measure_gap_adaptive(org, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
                stats = org.get_w_lr_eff_stats()
                print(f"gap={g:+.4f}  w_lr_eff=[{stats['min']:.5f},{stats['max']:.5f}]",
                      flush=True)
            else:
                g = measure_gap(org, sigs, k, s, n_perm=N_PERM, n_trials=N_TRIALS)
                print(f"gap={g:+.4f}", flush=True)
            gaps.append(g)
    return gaps


if __name__ == '__main__':
    print("=" * 72)
    print("  ANIMA Stage 2 Final Test + Calibrated Baseline")
    print("=" * 72)

    # ── Part 1: Fixed vs Reactive ────────────────────────────────
    print("\n[Part 1: FIXED w_lr=0.0003]")
    fixed_gaps = run_condition(
        "FIXED",
        lambda: AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=FIXED_PARAMS),
        use_adaptive=False
    )
    fixed_mean, fixed_std = mean_std(fixed_gaps)
    print(f"\n  FIXED mean={fixed_mean:+.4f}  std={fixed_std:.4f}")

    print("\n[Part 1: REACTIVE w_lr_base=0.001, err_scale=2.0]")
    reactive_gaps = run_condition(
        "REACTIVE",
        lambda: AnimaOrganismAdaptive(seed=BIRTH_SEED, alive=True, rule_params=REACTIVE_PARAMS),
        use_adaptive=True
    )
    reactive_mean, reactive_std = mean_std(reactive_gaps)
    print(f"\n  REACTIVE mean={reactive_mean:+.4f}  std={reactive_std:.4f}")

    diffs = [r - f for r, f in zip(reactive_gaps, fixed_gaps)]
    diff_mean, diff_std = mean_std(diffs)
    pooled_std = math.sqrt((fixed_std**2 + reactive_std**2) / 2.0 + 1e-15)
    d_stat = (reactive_mean - fixed_mean) / pooled_std
    t_stat, p_val = t_test_paired(diffs)

    c1 = reactive_mean > fixed_mean
    c2 = d_stat > 0.5 or p_val < 0.05
    c3 = reactive_mean > 0.0
    stage2_pass = c1 and c2 and c3

    print("\n" + "=" * 72)
    print("  STAGE 2 RESULT")
    print("=" * 72)
    print(f"  FIXED    mean={fixed_mean:+.4f}  std={fixed_std:.4f}")
    print(f"  REACTIVE mean={reactive_mean:+.4f}  std={reactive_std:.4f}")
    print(f"  diff     mean={diff_mean:+.4f}  std={diff_std:.4f}")
    print(f"  d={d_stat:+.3f}  t={t_stat:+.3f}  p={p_val:.4f}")
    print(f"  c1 (reactive>fixed): {c1}")
    print(f"  c2 (d>0.5 or p<0.05): {c2}")
    print(f"  c3 (reactive>0): {c3}")
    print(f"  => {'STAGE 2: PASS' if stage2_pass else 'STAGE 2: FAIL — vacuous pass criteria met'}")

    # ── Part 2: Calibrated Baseline ──────────────────────────────
    print("\n" + "=" * 72)
    print("  PART 2: Calibrated Baseline (w_lr=0.0003, gamma=3.0)")
    print("=" * 72)
    print("\n[Computing calibrated baseline...]")
    calibrated_gaps = run_condition(
        "CALIBRATED",
        lambda: AnimaOrganism(seed=BIRTH_SEED, alive=True, rule_params=CALIBRATED_PARAMS),
        use_adaptive=False
    )
    cal_mean, cal_std = mean_std(calibrated_gaps)

    print("\n" + "=" * 72)
    print("  CALIBRATED BASELINE")
    print("=" * 72)
    print(f"  w_lr=0.0003, gamma=3.0")
    print(f"  mean={cal_mean:+.4f}  std={cal_std:.4f}")
    print(f"  vs uncalibrated (w_lr=0.0003, gamma=0.9): +0.0948")
    print(f"  improvement: {cal_mean - 0.0948:+.4f}")
    print("=" * 72)
