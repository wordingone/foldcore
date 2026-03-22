#!/usr/bin/env python3
"""
Phase 1 Diagnostic: Characterize dual-timescale I signals at multiple w_lr values.

This is a DIAGNOSTIC script — it measures candidate signals for Stage 3 adaptation.
It does NOT implement the actual adaptation mechanism.

Signals measured per (w_lr, seed, K):
  1. I_fast_mean: mean |mem[i][k]| at cycle end, averaged over cycles, normalized per-step
  2. I_slow_mean: mean |mem_slow[i][k]| at cycle end, averaged over cycles, normalized per-step
  3. I_curvature: norm(mem_end - mem_start) / cycle_steps, averaged over cycles
  4. W_velocity:  norm(w_end - w_prev_end), averaged over consecutive pairs
  5. Ratio:       I_slow_mean / I_fast_mean
  6. alive_gap:   standard MI gap (alive=True organism)

A "cycle" = one full permutation: K signals x (n_per_sig=60 + n_settle=30) steps + n_final=60.
Cycle steps: K*(60+30)+60

I_slow uses tau_slow=0.005 (accumulates over ~200 steps, spanning multiple K cycles at K=4).
"""

import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from harness import make_signals, gen_perms, vcosine, D, NC
from anima_organism import AnimaOrganism, _tanh, _clip


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

TAU_SLOW = 0.005      # slow accumulator time constant (~200-step memory)
N_PER_SIG = 60        # steps per signal presentation
N_SETTLE = 30         # settle steps after each signal
N_FINAL = 60          # final settle steps at end of permutation
N_ORG = 300           # organism warm-up steps (no signal)
N_PERM = 8            # permutations per K


# ═══════════════════════════════════════════════════════════════
# InstrumentedAnimaOrganism
# Wraps AnimaOrganism logic, adds mem_slow and per-cycle tracking.
# Does NOT modify anima_organism.py.
# ═══════════════════════════════════════════════════════════════

class InstrumentedAnimaOrganism:
    """
    Mirrors AnimaOrganism step() logic exactly, adds:
      - mem_slow[NC][D]: slow accumulator with tau_slow=0.005
      - Per-step err tracking (reset per cycle)
      - Cycle boundary snapshots of mem, mem_slow, w
    """

    def __init__(self, seed=42, alive=True, rule_params=None, tau_slow=TAU_SLOW):
        if rule_params is None:
            rule_params = {}

        self.seed = seed
        self.alive = alive
        self.tau_slow = tau_slow

        # Mirror AnimaOrganism parameters
        self.w_lr = rule_params.get('w_lr', 0.0003)
        self.tau = rule_params.get('tau', 0.3)
        self.gamma = rule_params.get('gamma', 3.0)
        self.w_clip = rule_params.get('w_clip', 2.0)
        self.noise = rule_params.get('noise', 0.005)
        self.delta = rule_params.get('delta', 1.0)

        random.seed(seed)

        self.x = [
            [random.gauss(0, 0.5) for _ in range(D)]
            for _ in range(NC)
        ]
        self.w = [
            [random.gauss(0, 0.1) for _ in range(D)]
            for _ in range(NC)
        ]
        self.mem = [
            [0.0] * D for _ in range(NC)
        ]
        self.mem_slow = [
            [0.0] * D for _ in range(NC)
        ]

        # Instrumentation state
        self._step_errs = []          # |err| values within current cycle
        self._cycle_snapshots = []    # list of (mem_start, mem_end, mem_slow_end, w_end) per cycle

    def step(self, xs, signal=None):
        """
        Mirror of AnimaOrganism.step() with mem_slow tracking.
        Also accumulates per-step |err| for I_curvature.
        """
        new_xs = []

        if signal is not None:
            sig_mean = sum(signal) / D
        else:
            sig_mean = 0.0

        step_err_sum = 0.0
        step_err_count = 0

        for i in range(NC):
            new_row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D

                neighbor = xs[i][kp] * xs[i][km]
                w_pred = self.w[i][k] * neighbor

                if signal is not None:
                    s_kp = signal[kp]
                    s_km = signal[km]
                    neighbor_sig = (xs[i][kp] + self.gamma * s_kp) * (xs[i][km] + self.gamma * s_km)
                    actual = neighbor_sig
                    err = actual - w_pred
                else:
                    err = 0.0

                # Update W
                if signal is not None:
                    dw = self.w_lr * _tanh(err)
                    self.w[i][k] += dw
                    self.w[i][k] = _clip(self.w[i][k], -self.w_clip, self.w_clip)

                # Update I (fast mem)
                if self.alive and signal is not None:
                    self.mem[i][k] = (1.0 - self.tau) * self.mem[i][k] + self.tau * _tanh(err)

                # Update I_slow (slow accumulator — always tracks when alive)
                if self.alive and signal is not None:
                    self.mem_slow[i][k] = (1.0 - self.tau_slow) * self.mem_slow[i][k] + self.tau_slow * _tanh(err)

                # Track err magnitude for instrumentation
                if signal is not None:
                    step_err_sum += abs(err)
                    step_err_count += 1

                # State update (mirror of AnimaOrganism exactly)
                w_drive = self.w[i][k] * neighbor
                i_drive = self.mem[i][k] * xs[i][k]
                pre_act = xs[i][k] + w_drive + i_drive
                phi = _tanh(pre_act)
                v = (1.0 - self.delta) * xs[i][k] + self.delta * phi
                v += random.gauss(0, self.noise)
                new_row.append(v)
            new_xs.append(new_row)

        if step_err_count > 0:
            self._step_errs.append(step_err_sum / step_err_count)

        return new_xs

    def centroid(self, xs):
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]

    def snapshot_mem(self):
        """Deep copy of mem."""
        return [[self.mem[i][k] for k in range(D)] for i in range(NC)]

    def snapshot_mem_slow(self):
        return [[self.mem_slow[i][k] for k in range(D)] for i in range(NC)]

    def snapshot_w(self):
        return [[self.w[i][k] for k in range(D)] for i in range(NC)]

    def flat_norm(self, m):
        """Frobenius norm of NC x D matrix."""
        return math.sqrt(sum(m[i][k] ** 2 for i in range(NC) for k in range(D)) + 1e-15)

    def flat_diff_norm(self, m1, m2):
        """Norm of element-wise difference."""
        return math.sqrt(sum((m1[i][k] - m2[i][k]) ** 2
                             for i in range(NC) for k in range(D)) + 1e-15)

    def mean_abs(self, m):
        """Mean of |m[i][k]| across all elements."""
        total = sum(abs(m[i][k]) for i in range(NC) for k in range(D))
        return total / (NC * D)

    def reset_cycle_tracking(self):
        self._step_errs = []
        self._cycle_snapshots = []


# ═══════════════════════════════════════════════════════════════
# Cycle runner: runs permutations, collects per-cycle snapshots
# ═══════════════════════════════════════════════════════════════

def run_permutations_instrumented(org, signals, k, seed, n_perm=N_PERM, n_trials=1):
    """
    Run n_perm permutations with n_trials each, collecting per-cycle signal measurements.

    Returns:
        cycles: list of dicts with keys:
            mem_start, mem_end, mem_slow_end, w_start, w_end,
            steps, step_errs (list of per-step mean |err|)
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    cycle_steps = k * (N_PER_SIG + N_SETTLE) + N_FINAL

    cycles = []
    prev_w = None

    for pi, perm in enumerate(perms):
        for trial in range(n_trials):
            # Snapshot state at cycle start
            mem_start = org.snapshot_mem()
            w_start = org.snapshot_w()
            org._step_errs = []

            # Run the permutation
            random.seed(seed)
            xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

            # Warm-up only on first cycle
            if pi == 0 and trial == 0:
                for _ in range(N_ORG):
                    xs = org.step(xs)

            for idx, sid in enumerate(perm):
                random.seed(seed * 1000 + sid * 100 + idx * 10 + trial)
                sig = [signals[sid][k2] + random.gauss(0, 0.05) for k2 in range(D)]
                for _ in range(N_PER_SIG):
                    xs = org.step(xs, sig)
                for _ in range(N_SETTLE):
                    xs = org.step(xs)

            for _ in range(N_FINAL):
                xs = org.step(xs)

            mem_end = org.snapshot_mem()
            mem_slow_end = org.snapshot_mem_slow()
            w_end = org.snapshot_w()

            cycles.append({
                'mem_start': mem_start,
                'mem_end': mem_end,
                'mem_slow_end': mem_slow_end,
                'w_start': w_start,
                'w_end': w_end,
                'prev_w': prev_w,
                'steps': cycle_steps,
                'step_errs': list(org._step_errs),
            })
            prev_w = w_end

    return cycles


# ═══════════════════════════════════════════════════════════════
# Signal computation from cycle snapshots
# ═══════════════════════════════════════════════════════════════

def compute_signals_from_cycles(org, cycles, k):
    """
    Compute the 5 diagnostic signals from cycle snapshots.
    Returns dict of signal measurements.
    """
    cycle_steps = k * (N_PER_SIG + N_SETTLE) + N_FINAL

    i_fast_means = []
    i_slow_means = []
    i_curvatures = []
    w_velocities = []

    for cyc in cycles:
        # I_fast_mean: mean |mem_end[i][k]| — NOT normalized.
        # mem values saturate via tanh and do not grow with cycle length.
        # Normalizing by cycle_steps would introduce a reverse K-confound into the ratio.
        i_fast = org.mean_abs(cyc['mem_end'])
        i_fast_means.append(i_fast)

        # I_slow_mean: mean |mem_slow_end[i][k]|, normalized per step
        i_slow = org.mean_abs(cyc['mem_slow_end']) / cycle_steps
        i_slow_means.append(i_slow)

        # I_curvature: norm(mem_end - mem_start) / cycle_steps
        curv = org.flat_diff_norm(cyc['mem_end'], cyc['mem_start']) / cycle_steps
        i_curvatures.append(curv)

        # W_velocity: norm(w_end - prev_w_end), skip first cycle (no prev_w)
        if cyc['prev_w'] is not None:
            vel = org.flat_diff_norm(cyc['w_end'], cyc['prev_w'])
            w_velocities.append(vel)

    def safe_mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    i_fast_mean = safe_mean(i_fast_means)
    i_slow_mean = safe_mean(i_slow_means)
    i_curvature = safe_mean(i_curvatures)
    w_velocity = safe_mean(w_velocities)
    ratio = i_slow_mean / max(i_fast_mean, 1e-15)

    return {
        'I_fast_mean': i_fast_mean,
        'I_slow_mean': i_slow_mean,
        'I_curvature': i_curvature,
        'W_velocity': w_velocity,
        'ratio': ratio,
    }


# ═══════════════════════════════════════════════════════════════
# alive_gap measurement (uses n_trials=2 for speed)
# ═══════════════════════════════════════════════════════════════

def run_sequence_plain(org, order, signals, base_seed, trial):
    """Plain run_sequence using InstrumentedAnimaOrganism (no snapshot overhead)."""
    random.seed(base_seed)
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    for _ in range(N_ORG):
        xs = org.step(xs)

    for idx, sid in enumerate(order):
        random.seed(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = [signals[sid][k] + random.gauss(0, 0.05) for k in range(D)]
        for _ in range(N_PER_SIG):
            xs = org.step(xs, sig)
        for _ in range(N_SETTLE):
            xs = org.step(xs)

    for _ in range(N_FINAL):
        xs = org.step(xs)

    return org.centroid(xs)


def measure_alive_gap(rule_params, signals, k, seed, n_perm=N_PERM, n_trials=2):
    """
    Measure MI gap for alive=True organism.
    Uses n_trials=2 (reduced from 6 for Phase 1 speed — documented here).

    One organism is created for ALL permutations — W learns and I accumulates
    sequence history across the full sweep, matching run_anima_stage1.py protocol.
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    org = InstrumentedAnimaOrganism(seed=42, alive=True, rule_params=rule_params)
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c = run_sequence_plain(org, perm, signals, seed, trial)
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
# Main diagnostic loop
# ═══════════════════════════════════════════════════════════════

def run_diagnostic():
    w_lr_values = [0.0001, 0.0003, 0.001, 0.003, 0.01]
    seeds = [42, 137]
    ks = [4, 6, 8, 10]

    base_params = {
        'tau':    0.3,
        'gamma':  3.0,
        'w_clip': 2.0,
        'noise':  0.005,
        'delta':  1.0,
    }

    print("=" * 72)
    print("  PHASE 1 DIAGNOSTIC — Dual-Timescale I Signal Characterization")
    print(f"  tau_slow={TAU_SLOW}  n_perm={N_PERM}  n_trials=1 (signal), 2 (alive_gap)")
    print(f"  w_lr values: {w_lr_values}")
    print(f"  seeds: {seeds}   K: {ks}")
    print("=" * 72)
    print()

    # Results table: keyed by (w_lr, seed, k)
    results = {}

    total = len(w_lr_values) * len(seeds) * len(ks)
    done = 0

    for w_lr in w_lr_values:
        rule_params = dict(base_params, w_lr=w_lr)

        for seed in seeds:
            for k in ks:
                done += 1
                print(f"[{done:3d}/{total}] w_lr={w_lr:.4f}  seed={seed:3d}  K={k} ...", flush=True)

                sig_seed = 42 + k * 200
                sigs = make_signals(k, seed=sig_seed)

                # Signal measurement: 1 trial per perm
                org = InstrumentedAnimaOrganism(
                    seed=42, alive=True, rule_params=rule_params, tau_slow=TAU_SLOW
                )
                cycles = run_permutations_instrumented(org, sigs, k, seed, n_perm=N_PERM, n_trials=1)
                sigs_measured = compute_signals_from_cycles(org, cycles, k)

                # alive_gap measurement: fresh organism, 2 trials per perm
                alive_gap = measure_alive_gap(rule_params, sigs, k, seed, n_perm=N_PERM, n_trials=2)

                result = {**sigs_measured, 'alive_gap': alive_gap}
                results[(w_lr, seed, k)] = result

                print(f"         I_fast={result['I_fast_mean']:.4e}  I_slow={result['I_slow_mean']:.4e}  "
                      f"ratio={result['ratio']:.4f}")
                print(f"         I_curv={result['I_curvature']:.4e}  W_vel={result['W_velocity']:.4e}  "
                      f"alive_gap={result['alive_gap']:+.4f}")

    # ═══════════════════════════════════════════════════════════════
    # Summary table: averaged across seeds and K per w_lr
    # ═══════════════════════════════════════════════════════════════

    print()
    print("=" * 72)
    print("  SUMMARY (averaged across seeds and K)")
    print("=" * 72)
    print(f"{'w_lr':>8}  {'I_fast_mean':>12}  {'I_slow_mean':>12}  {'ratio':>8}  "
          f"{'I_curv':>12}  {'W_vel':>12}  {'alive_gap':>10}")
    print("-" * 72)

    for w_lr in w_lr_values:
        keys = [(w_lr, s, k) for s in seeds for k in ks]
        n = len(keys)

        def avg_field(field):
            return sum(results[kk][field] for kk in keys) / n

        i_fast = avg_field('I_fast_mean')
        i_slow = avg_field('I_slow_mean')
        ratio = avg_field('ratio')
        curv = avg_field('I_curvature')
        vel = avg_field('W_velocity')
        gap = avg_field('alive_gap')

        print(f"{w_lr:>8.4f}  {i_fast:>12.4e}  {i_slow:>12.4e}  {ratio:>8.4f}  "
              f"{curv:>12.4e}  {vel:>12.4e}  {gap:>+10.4f}")

    print()
    print("=" * 72)
    print("  SIGNAL MONOTONICITY CHECK (across w_lr, avg over seeds x K)")
    print("=" * 72)
    signal_names = ['I_fast_mean', 'I_slow_mean', 'ratio', 'I_curvature', 'W_velocity', 'alive_gap']
    for sig_name in signal_names:
        vals = []
        for w_lr in w_lr_values:
            keys = [(w_lr, s, k) for s in seeds for k in ks]
            n = len(keys)
            v = sum(results[kk][sig_name] for kk in keys) / n
            vals.append(v)
        # Check if monotonic increasing or decreasing
        inc = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        dec = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
        if inc:
            mono = "MONOTONE_INC"
        elif dec:
            mono = "MONOTONE_DEC"
        else:
            mono = "non-monotone"
        vals_str = "  ".join(f"{v:.3e}" for v in vals)
        print(f"  {sig_name:<16}: {mono:<14}  [{vals_str}]")

    print()
    print("  NOTE: alive_gap inverted-U expected. All other signals diagnostic.")
    print("  NOTE: alive_gap used n_trials=2 (reduced from canonical 6 for speed).")
    print("=" * 72)


if __name__ == '__main__':
    run_diagnostic()
