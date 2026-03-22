#!/usr/bin/env python3
"""
STAGE 3 CONTROLS

Control A: Heterogeneous eta without adaptation
  Does per-cell eta heterogeneity help even when not adaptive?
  Compare:
    - STILL (alpha frozen)
    - ALIVE-stage2 (fixed global eta=0.0003)
    - ALIVE-hetero-eta (per-cell fixed eta, not adaptive)

Control B: resp_z autocorrelation
  Is resp_z temporally correlated?
  If autocorrelation ~ 0, delta_rz signal would be noise.
"""

import math
import random
import time

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

D  = 12
NC = 6
W  = 72

# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def vcosine(a, b):
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na2 += ai * ai
        nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15)
    nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


# ═══════════════════════════════════════════════════════════════
# Organism with optional heterogeneous eta
# ═══════════════════════════════════════════════════════════════

class Organism:

    def __init__(self, seed=42, alive=False, eta=0.0003, hetero_eta=False):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.hetero_eta = hetero_eta
        self.total_alpha_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

        # Heterogeneous eta: per-cell per-dimension, fixed
        if hetero_eta:
            self.eta = [
                [random.uniform(0.0001, 0.001) for _ in range(D)]
                for _ in range(NC)
            ]
        else:
            # Scalar eta
            self.eta = eta

        # For Control B: record resp_z history
        self.resp_z_history = []

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        # ── BARE DYNAMICS ────────────────────────────────────
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D
                km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k]
                    + beta * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        # ── SIGNAL-MODULATED DYNAMICS ────────────────────────
        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D
                    km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta * (xs[i][kp] + gamma * signal[kp])
                               * (xs[i][km] + gamma * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

        # ── ONLINE PLASTICITY ────────────────────────────────
        if self.alive and signal:
            response = []
            for i in range(NC):
                response.append([abs(phi_sig[i][k] - phi_bare[i][k])
                                 for k in range(D)])

            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            overall_mean = sum(all_resp) / len(all_resp)
            overall_std = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            resp_z_step = []
            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std
                    resp_z_step.append(resp_z)

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    # Determine eta for this cell
                    if self.hetero_eta:
                        eta_ik = self.eta[i][k]
                    else:
                        eta_ik = self.eta

                    if abs(dev) < 0.01:
                        # at column mean: break symmetry
                        push = eta_ik * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        # above-average response: amplify diversity
                        direction = 1.0 if dev > 0 else -1.0
                        push = eta_ik * math.tanh(resp_z) * direction * 0.5
                    else:
                        # below-average response: gentle drift
                        push = eta_ik * 0.1 * random.gauss(0, 1.0)

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old)

            # For Control B: record resp_z values
            self.resp_z_history.append(resp_z_step)

        # ── ATTENTION ────────────────────────────────────────
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][k] * xs[j][k] for k in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # ── STATE UPDATE ─────────────────────────────────────
        new = []
        for i in range(NC):
            p = [v for v in phi_sig[i]]

            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            fp_d = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
            plast = math.exp(-(fp_d * fp_d) / 0.0225)

            if plast > 0.01 and self.eps > 0:
                pull = [0.0] * D
                for j in range(NC):
                    if i == j or weights[i][j] < 1e-8:
                        continue
                    for k in range(D):
                        pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
                p = [p[k] + plast * self.eps * pull[k] for k in range(D)]

            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p[k]
                v += random.gauss(0, self.noise)
                v = max(-self.clip, min(self.clip, v))
                nx.append(v)
            new.append(nx)
        return new

    def centroid(self, xs):
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]


# ═══════════════════════════════════════════════════════════════
# Signal generation
# ═══════════════════════════════════════════════════════════════

def make_signals(k, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(k, n_perm, seed):
    random.seed(seed)
    base = list(range(k))
    perms = []
    seen = set()
    perms.append(tuple(base))
    seen.add(tuple(base))
    perms.append(tuple(reversed(base)))
    seen.add(tuple(reversed(base)))
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms


def run_sequence(org, order, signals, base_seed, trial,
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


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
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
# CONTROL A: Heterogeneous fixed eta
# ═══════════════════════════════════════════════════════════════

def control_a():
    print("=" * W)
    print("  CONTROL A: HETEROGENEOUS ETA WITHOUT ADAPTATION")
    print("  Does eta heterogeneity help when NOT adaptive?")
    print("=" * W)

    t_start = time.time()

    SEED = 42
    test_ks = [4, 6, 8, 10]
    test_seeds = [42, 137, 2024]

    print(f"\n{'-'*W}")
    print(f"  Testing three conditions:")
    print(f"  1. STILL (alpha frozen, baseline)")
    print(f"  2. ALIVE-stage2 (global eta=0.0003)")
    print(f"  3. ALIVE-hetero-eta (per-cell fixed eta ~ U(0.0001, 0.001))")
    print(f"{'-'*W}\n")

    # Condition 1: STILL
    print("  --- STILL (alpha frozen) ---")
    still_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            still = Organism(seed=SEED, alive=False)
            g = measure_gap(still, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        still_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    still_overall = sum(still_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {still_overall:+.4f}\n")

    # Condition 2: ALIVE-stage2 (global eta)
    print("  --- ALIVE-stage2 (global eta=0.0003) ---")
    alive_global_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            alive = Organism(seed=SEED, alive=True, eta=0.0003, hetero_eta=False)
            g = measure_gap(alive, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        alive_global_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    alive_global_overall = sum(alive_global_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {alive_global_overall:+.4f} "
          f"delta vs STILL: {alive_global_overall - still_overall:+.4f}\n")

    # Condition 3: ALIVE-hetero-eta (per-cell fixed)
    print("  --- ALIVE-hetero-eta (per-cell fixed eta) ---")
    alive_hetero_results = {}
    for k in test_ks:
        sigs = make_signals(k, seed=SEED + k * 200)
        gaps = []
        for s in test_seeds:
            alive = Organism(seed=SEED, alive=True, hetero_eta=True)
            g = measure_gap(alive, sigs, k, s)
            gaps.append(g)
        avg = sum(gaps) / len(gaps)
        alive_hetero_results[k] = {'avg': avg, 'gaps': gaps}
        print(f"  K={k:>2}: avg={avg:+.4f} "
              f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)

    alive_hetero_overall = sum(alive_hetero_results[k]['avg'] for k in test_ks) / len(test_ks)
    print(f"  overall: {alive_hetero_overall:+.4f} "
          f"delta vs STILL: {alive_hetero_overall - still_overall:+.4f}\n")

    # Novel signal test
    print(f"\n{'-'*W}")
    print(f"  NOVEL SIGNAL TEST")
    print(f"  6 novel worlds, K=[6,8]")
    print(f"{'-'*W}\n")

    novel_still = []
    novel_global = []
    novel_hetero = []

    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=99999 + wi * 37 + k)
            ts = 77 + wi * 13 + k

            st = Organism(seed=SEED, alive=False)
            ag = Organism(seed=SEED, alive=True, eta=0.0003, hetero_eta=False)
            ah = Organism(seed=SEED, alive=True, hetero_eta=True)

            sg = measure_gap(st, nsigs, k, ts)
            gg = measure_gap(ag, nsigs, k, ts)
            hg = measure_gap(ah, nsigs, k, ts)

            novel_still.append(sg)
            novel_global.append(gg)
            novel_hetero.append(hg)

            print(f"  w={wi} K={k}: STILL={sg:+.4f} "
                  f"GLOBAL={gg:+.4f} HETERO={hg:+.4f}", flush=True)

    ns_avg = sum(novel_still) / len(novel_still)
    ng_avg = sum(novel_global) / len(novel_global)
    nh_avg = sum(novel_hetero) / len(novel_hetero)

    print(f"\n  Novel averages:")
    print(f"    STILL:  {ns_avg:+.4f}")
    print(f"    GLOBAL: {ng_avg:+.4f} (delta: {ng_avg - ns_avg:+.4f})")
    print(f"    HETERO: {nh_avg:+.4f} (delta: {nh_avg - ns_avg:+.4f})")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  CONTROL A VERDICT")
    print(f"{'='*W}\n")

    print(f"  Overall averages (K=[4,6,8,10]):")
    print(f"    STILL:        {still_overall:+.4f}")
    print(f"    ALIVE-global: {alive_global_overall:+.4f} "
          f"(delta: {alive_global_overall - still_overall:+.4f})")
    print(f"    ALIVE-hetero: {alive_hetero_overall:+.4f} "
          f"(delta: {alive_hetero_overall - still_overall:+.4f})")

    hetero_wins_global = alive_hetero_overall > alive_global_overall
    hetero_wins_still = alive_hetero_overall > still_overall

    print(f"\n  Hetero vs Global: {alive_hetero_overall - alive_global_overall:+.4f}")
    print(f"  Hetero vs STILL:  {alive_hetero_overall - still_overall:+.4f}")

    print(f"\n  Novel signal test:")
    print(f"    HETERO beats GLOBAL: {'YES' if nh_avg > ng_avg else 'NO'} "
          f"({nh_avg - ng_avg:+.4f})")
    print(f"    HETERO beats STILL:  {'YES' if nh_avg > ns_avg else 'NO'} "
          f"({nh_avg - ns_avg:+.4f})")

    if hetero_wins_global and hetero_wins_still:
        print(f"\n  RESULT: Heterogeneous eta HELPS even when fixed.")
        print(f"  The variance in learning rates provides benefit without adaptation.")
    elif hetero_wins_still:
        print(f"\n  RESULT: Heterogeneous eta helps vs STILL but not vs global.")
        print(f"  Effect is real but weaker than uniform eta=0.0003.")
    else:
        print(f"\n  RESULT: Heterogeneous eta does NOT help when fixed.")
        print(f"  The benefit requires adaptation, not just variance.")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}\n")

    return {
        'still_overall': still_overall,
        'alive_global_overall': alive_global_overall,
        'alive_hetero_overall': alive_hetero_overall,
        'novel_still': ns_avg,
        'novel_global': ng_avg,
        'novel_hetero': nh_avg
    }


# ═══════════════════════════════════════════════════════════════
# CONTROL B: resp_z autocorrelation
# ═══════════════════════════════════════════════════════════════

def control_b():
    print("=" * W)
    print("  CONTROL B: RESP_Z AUTOCORRELATION")
    print("  Is the resp_z signal temporally correlated?")
    print("=" * W)

    t_start = time.time()

    SEED = 42
    k = 8
    sigs = make_signals(k, seed=SEED + k * 200)

    print(f"\n{'-'*W}")
    print(f"  Running ALIVE organism (eta=0.0003) on K={k} signal sequence")
    print(f"  Recording resp_z values at each signal step")
    print(f"{'-'*W}\n")

    # Run organism and collect resp_z history
    org = Organism(seed=SEED, alive=True, eta=0.0003)
    order = list(range(k))
    _, _ = run_sequence(org, order, sigs, SEED, trial=0)

    print(f"  Signal steps recorded: {len(org.resp_z_history)}")
    print(f"  Cells per step: {NC * D}")

    # Compute lag-1 autocorrelation for each (i,k) cell
    if len(org.resp_z_history) < 2:
        print("\n  ERROR: Not enough signal steps to compute autocorrelation.")
        return

    # Reshape: resp_z_history is list of time steps, each containing NC*D values
    # We want NC*D time series, one per cell
    n_steps = len(org.resp_z_history)
    n_cells = NC * D

    # Extract time series for each cell
    cell_series = []
    for cell_idx in range(n_cells):
        series = [org.resp_z_history[t][cell_idx] for t in range(n_steps)]
        cell_series.append(series)

    # Compute lag-1 autocorrelation for each cell
    autocorrs = []
    for series in cell_series:
        if len(series) < 2:
            continue

        # Lag-1 autocorrelation: corr(x[0:n-1], x[1:n])
        x = series[:-1]
        y = series[1:]

        if len(x) == 0:
            continue

        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        if var_x < 1e-15 or var_y < 1e-15:
            autocorrs.append(0.0)
        else:
            autocorrs.append(cov / math.sqrt(var_x * var_y))

    # Statistics
    mean_ac = sum(autocorrs) / len(autocorrs)
    var_ac = sum((a - mean_ac) ** 2 for a in autocorrs) / len(autocorrs)
    std_ac = math.sqrt(var_ac)
    min_ac = min(autocorrs)
    max_ac = max(autocorrs)

    print(f"\n{'-'*W}")
    print(f"  LAG-1 AUTOCORRELATION STATISTICS")
    print(f"{'-'*W}\n")

    print(f"  Cells analyzed: {len(autocorrs)}")
    print(f"  Mean:  {mean_ac:+.4f}")
    print(f"  Std:   {std_ac:.4f}")
    print(f"  Min:   {min_ac:+.4f}")
    print(f"  Max:   {max_ac:+.4f}")

    # Distribution analysis
    near_zero = sum(1 for a in autocorrs if abs(a) < 0.1)
    positive = sum(1 for a in autocorrs if a > 0.1)
    negative = sum(1 for a in autocorrs if a < -0.1)

    print(f"\n  Distribution:")
    print(f"    Near zero (|ac| < 0.1): {near_zero}/{len(autocorrs)} "
          f"({100*near_zero/len(autocorrs):.1f}%)")
    print(f"    Positive (ac > 0.1):     {positive}/{len(autocorrs)} "
          f"({100*positive/len(autocorrs):.1f}%)")
    print(f"    Negative (ac < -0.1):    {negative}/{len(autocorrs)} "
          f"({100*negative/len(autocorrs):.1f}%)")

    # ═══════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*W}")
    print(f"  CONTROL B VERDICT")
    print(f"{'='*W}\n")

    if abs(mean_ac) < 0.05:
        print(f"  RESULT: resp_z is NOT autocorrelated (mean ~ 0).")
        print(f"  Delta_rz = resp_z[t] - resp_z[t-1] would be pure noise.")
        print(f"  Experiment B (delta_rz adaptation) is UNLIKELY to work.")
    elif mean_ac > 0.1:
        print(f"  RESULT: resp_z is POSITIVELY autocorrelated (mean = {mean_ac:+.4f}).")
        print(f"  Delta_rz signal contains information about changing sensitivity.")
        print(f"  Experiment B (delta_rz adaptation) is PLAUSIBLE.")
    elif mean_ac < -0.1:
        print(f"  RESULT: resp_z is NEGATIVELY autocorrelated (mean = {mean_ac:+.4f}).")
        print(f"  Oscillatory pattern. Delta_rz may capture phase information.")
        print(f"  Experiment B worth testing but unclear interpretation.")
    else:
        print(f"  RESULT: resp_z has WEAK autocorrelation (mean = {mean_ac:+.4f}).")
        print(f"  Delta_rz signal is noisy but may contain weak information.")
        print(f"  Experiment B is MARGINAL — low prior probability of success.")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}\n")

    return {
        'mean_ac': mean_ac,
        'std_ac': std_ac,
        'min_ac': min_ac,
        'max_ac': max_ac,
        'near_zero': near_zero,
        'positive': positive,
        'negative': negative,
        'n_cells': len(autocorrs)
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def run():
    print("=" * W)
    print("  STAGE 3 CONTROLS")
    print("  A: Does eta heterogeneity help when NOT adaptive?")
    print("  B: Is resp_z temporally correlated?")
    print("=" * W)
    print()

    t_start = time.time()

    # Control A
    results_a = control_a()

    # Control B
    results_b = control_b()

    # ═══════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════

    print("=" * W)
    print("  FINAL SUMMARY")
    print("=" * W)
    print()

    print("  CONTROL A:")
    print(f"    STILL:        {results_a['still_overall']:+.4f}")
    print(f"    ALIVE-global: {results_a['alive_global_overall']:+.4f} "
          f"(+{results_a['alive_global_overall'] - results_a['still_overall']:+.4f})")
    print(f"    ALIVE-hetero: {results_a['alive_hetero_overall']:+.4f} "
          f"(+{results_a['alive_hetero_overall'] - results_a['still_overall']:+.4f})")
    print(f"    Novel: HETERO={results_a['novel_hetero']:+.4f} "
          f"GLOBAL={results_a['novel_global']:+.4f} "
          f"STILL={results_a['novel_still']:+.4f}")

    hetero_helps = results_a['alive_hetero_overall'] > results_a['alive_global_overall']
    print(f"\n    Verdict: Heterogeneous eta {'HELPS' if hetero_helps else 'does NOT help'} "
          f"when fixed.")

    print()
    print("  CONTROL B:")
    print(f"    Mean autocorr: {results_b['mean_ac']:+.4f}")
    print(f"    Std:           {results_b['std_ac']:.4f}")
    print(f"    Near zero:     {results_b['near_zero']}/{results_b['n_cells']} "
          f"({100*results_b['near_zero']/results_b['n_cells']:.1f}%)")

    if abs(results_b['mean_ac']) < 0.05:
        exp_b_outlook = "UNLIKELY to succeed"
    elif abs(results_b['mean_ac']) > 0.1:
        exp_b_outlook = "PLAUSIBLE"
    else:
        exp_b_outlook = "MARGINAL"

    print(f"\n    Verdict: Experiment B (delta_rz) is {exp_b_outlook}.")

    print(f"\n  Total runtime: {time.time() - t_start:.0f}s")
    print(f"  {'='*W}")


if __name__ == '__main__':
    run()
