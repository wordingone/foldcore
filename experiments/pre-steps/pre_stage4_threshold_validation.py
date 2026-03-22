#!/usr/bin/env python3
"""
Stage 4 Phase 1 Validation: threshold=0.1 vs canonical=0.01

10-seed validation of the threshold sensitivity finding from stage4_diagnostic.py.
The diagnostic found threshold=0.1 gives +20% over canonical (3 seeds, may be noise).

Tests:
  - threshold=0.01 (canonical)
  - threshold=0.02
  - threshold=0.05
  - threshold=0.1

Protocol: 10 seeds x K=[4,6,8,10] x n_perm=8, n_trials=6 (c015, c017)
Statistics: mean, std, Cohen's d, Mann-Whitney U p-value vs canonical

Outcome determines:
  (a) threshold=0.1 genuinely better -> update canonical baseline
  (b) threshold=0.1 is noise -> canonical stays (but threshold still binding)
  (c) optimal threshold in [0.02, 0.1] -> reveals landscape shape
"""

import math
import random
import time


D  = 12
NC = 6


# ── Stats helpers ─────────────────────────────────────────────────────────────

def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def cohens_d(a, b):
    """Pooled Cohen's d: (mean_a - mean_b) / pooled_std."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    sa, sb = std(a), std(b)
    pooled = math.sqrt(((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2))
    return (mean(a) - mean(b)) / (pooled + 1e-15)


def mann_whitney_u_p(a, b):
    """
    Exact Mann-Whitney U p-value (two-sided) via normal approximation.
    Returns p-value. Works for n >= 3.
    """
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 1.0

    # Count U statistic
    u = sum(1 for x in a for y in b if x > y) + 0.5 * sum(1 for x in a for y in b if x == y)
    mu_u = na * nb / 2.0
    sigma_u = math.sqrt(na * nb * (na + nb + 1) / 12.0) + 1e-15
    z = (u - mu_u) / sigma_u
    # Two-sided p from normal CDF approximation
    p = 2.0 * (1.0 - _norm_cdf(abs(z)))
    return p


def _norm_cdf(x):
    """Standard normal CDF via math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


# ── Living Seed (copied from the_living_seed.py, threshold configurable) ──────

def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


class Organism:
    def __init__(self, seed=42, alive=False, eta=0.0003, threshold=0.01):
        self.beta      = 0.5
        self.gamma     = 0.9
        self.eps       = 0.15
        self.tau       = 0.3
        self.delta     = 0.35
        self.noise     = 0.005
        self.clip      = 4.0
        self.alive     = alive
        self.eta       = eta
        self.threshold = threshold

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

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
                row_out = row
                phi_sig.append(row_out)
        else:
            phi_sig = phi_bare

        if self.alive and signal:
            response = [[abs(phi_sig[i][k] - phi_bare[i][k]) for k in range(D)]
                        for i in range(NC)]
            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            overall_mean = sum(all_resp) / len(all_resp)
            overall_std  = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z   = (response[i][k] - overall_mean) / overall_std
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev      = self.alpha[i][k] - col_mean

                    if abs(dev) < self.threshold:
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)

                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))

        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][k] * xs[j][k] for k in range(D))
                    raw.append(d / (D * self.tau))
            mx   = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s    = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        new = []
        for i in range(NC):
            p        = list(phi_sig[i])
            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            fp_d     = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
            plast    = math.exp(-(fp_d * fp_d) / 0.0225)

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


def make_signals(k, seed):
    random.seed(seed)
    sigs = {}
    for i in range(k):
        s  = [random.gauss(0, 0.5) for _ in range(D)]
        nm = math.sqrt(sum(v * v for v in s) + 1e-15)
        sigs[i] = [v * 0.8 / nm for v in s]
    return sigs


def gen_perms(k, n_perm, seed):
    random.seed(seed)
    base  = list(range(k))
    perms = [tuple(base), tuple(reversed(base))]
    seen  = set(perms)
    att   = 0
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


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
    perms     = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            c, _ = run_sequence(org, perm, signals, seed, trial)
            trials.append(c)
        endpoints[pi] = trials

    within  = []
    between = []
    pis     = sorted(endpoints.keys())
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

    avg_w = sum(within)  / max(len(within),  1)
    avg_b = sum(between) / max(len(between), 1)
    return avg_w - avg_b


def measure_threshold(threshold, seeds, ks, sig_seed_base=42):
    """Measure mean MI gap for a threshold across all seeds and K values."""
    gaps = []
    for seed in seeds:
        seed_gaps = []
        for k in ks:
            sigs = make_signals(k, seed=sig_seed_base + k * 200)
            org  = Organism(seed=seed, alive=True, eta=0.0003, threshold=threshold)
            g    = measure_gap(org, sigs, k, seed)
            seed_gaps.append(g)
        gaps.append(mean(seed_gaps))  # per-seed mean across K values
    return gaps  # one value per seed (for stats)


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    t_start = time.time()

    SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
    KS    = [4, 6, 8, 10]   # c017: K=[4,6,8,10]

    # Thresholds: canonical + candidates from sweep
    CANONICAL   = 0.01
    THRESHOLDS  = [0.01, 0.02, 0.05, 0.1]

    print("=" * 72)
    print("  STAGE 4 THRESHOLD VALIDATION — 10-seed")
    print("  Validating diagnostic finding: threshold=0.1 +20% over canonical")
    print(f"  Seeds: {SEEDS}")
    print(f"  K: {KS}  n_perm=8  n_trials=6  (c015, c017)")
    print(f"  Thresholds: {THRESHOLDS}")
    print("=" * 72)

    results = {}

    for thresh in THRESHOLDS:
        tag = " [CANONICAL]" if thresh == CANONICAL else ""
        print(f"\n  Running threshold={thresh}{tag} ...", flush=True)
        t0   = time.time()
        gaps = measure_threshold(thresh, SEEDS, KS)
        elapsed = time.time() - t0

        m  = mean(gaps)
        s  = std(gaps)
        cv = abs(s / m) * 100 if m != 0 else 0.0

        results[thresh] = {'gaps': gaps, 'mean': m, 'std': s, 'cv': cv}
        print(f"    mean={m:+.4f}  std={s:.4f}  CV={cv:.1f}%  ({elapsed:.0f}s)",
              flush=True)
        print(f"    per-seed: {[f'{g:+.4f}' for g in gaps]}", flush=True)

    # ── Statistical comparison vs canonical ───────────────────────────────────
    print("\n" + "=" * 72)
    print("  STATISTICAL COMPARISON vs CANONICAL (threshold=0.01)")
    print("=" * 72)

    canonical_gaps = results[CANONICAL]['gaps']
    canonical_mean = results[CANONICAL]['mean']
    canonical_std  = results[CANONICAL]['std']

    print(f"\n  Canonical: mean={canonical_mean:+.4f}  std={canonical_std:.4f}  "
          f"CV={results[CANONICAL]['cv']:.1f}%")
    print()
    print(f"  {'Threshold':>10}  {'Mean':>8}  {'Std':>7}  {'Delta':>8}  "
          f"{'Delta%':>7}  {'d':>6}  {'p-val':>8}  {'Verdict'}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*20}")

    for thresh in THRESHOLDS:
        if thresh == CANONICAL:
            continue
        r      = results[thresh]
        delta  = r['mean'] - canonical_mean
        pct    = delta / abs(canonical_mean) * 100 if canonical_mean != 0 else 0.0
        d      = cohens_d(r['gaps'], canonical_gaps)
        p      = mann_whitney_u_p(r['gaps'], canonical_gaps)

        if p < 0.05 and delta > 0:
            verdict = "BETTER (p<0.05)"
        elif p < 0.05 and delta < 0:
            verdict = "WORSE (p<0.05)"
        elif p < 0.1 and delta > 0:
            verdict = "trend better (p<0.1)"
        elif p < 0.1 and delta < 0:
            verdict = "trend worse (p<0.1)"
        else:
            verdict = "no sig diff"

        print(f"  {thresh:>10.3f}  {r['mean']:>+8.4f}  {r['std']:>7.4f}  "
              f"{delta:>+8.4f}  {pct:>+6.1f}%  {d:>+6.3f}  {p:>8.4f}  {verdict}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  FULL RESULTS TABLE (all thresholds)")
    print("=" * 72)
    print(f"\n  {'Threshold':>10}  {'Mean Gap':>10}  {'Std':>8}  {'CV%':>6}  {'vs Canonical':>14}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*14}")
    for thresh in THRESHOLDS:
        r     = results[thresh]
        delta = r['mean'] - canonical_mean
        tag   = " <-- canonical" if thresh == CANONICAL else ""
        print(f"  {thresh:>10.3f}  {r['mean']:>+10.4f}  {r['std']:>8.4f}  "
              f"{r['cv']:>5.1f}%  {delta:>+14.4f}{tag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    all_means = [results[t]['mean'] for t in THRESHOLDS]
    max_mean  = max(all_means)
    best_thresh = THRESHOLDS[all_means.index(max_mean)]

    # Check if best threshold is significantly better than canonical
    if best_thresh != CANONICAL:
        best_gaps = results[best_thresh]['gaps']
        best_p    = mann_whitney_u_p(best_gaps, canonical_gaps)
        best_d    = cohens_d(best_gaps, canonical_gaps)
        best_delta = results[best_thresh]['mean'] - canonical_mean
        best_pct   = best_delta / abs(canonical_mean) * 100

        print(f"\n  Best threshold: {best_thresh}  (mean={max_mean:+.4f})")
        print(f"  vs canonical:   delta={best_delta:+.4f}  ({best_pct:+.1f}%)  "
              f"d={best_d:+.3f}  p={best_p:.4f}")

        if best_p < 0.05:
            print(f"\n  (a) CANONICAL BASELINE SHOULD BE UPDATED")
            print(f"      threshold={best_thresh} significantly outperforms canonical (p={best_p:.4f}).")
            print(f"      Update threshold in harness.py and re-run Stage 4 experiments with new baseline.")
        elif best_p < 0.1:
            print(f"\n  (b) WEAK TREND — threshold={best_thresh} trends better (p={best_p:.4f})")
            print(f"      Not significant at p<0.05. Threshold is still a binding parameter (30% range)")
            print(f"      but canonical does not need updating. Design adaptive threshold experiment.")
        else:
            print(f"\n  (b) NOISE — threshold=0.1 advantage from diagnostic was 3-seed noise.")
            print(f"      Canonical stays. Threshold still binding (30% range in diagnostic).")
            print(f"      Proceed to adaptive threshold experiment without updating canonical.")
    else:
        print(f"\n  Canonical (threshold=0.01) is optimal among tested values.")
        print(f"  Design adaptive threshold experiment starting from canonical.")

    # Check monotonicity
    vals = [results[t]['mean'] for t in THRESHOLDS]
    diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
    monotone_up   = all(d >= 0 for d in diffs)
    monotone_down = all(d <= 0 for d in diffs)
    if monotone_up:
        landscape = "MONOTONE INCREASING — higher threshold consistently better"
    elif monotone_down:
        landscape = "MONOTONE DECREASING — lower threshold consistently better"
    else:
        landscape = "NON-MONOTONE — threshold has complex effect on MI gap"

    print(f"\n  Landscape shape: {landscape}")
    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print("=" * 72)


if __name__ == '__main__':
    run()
