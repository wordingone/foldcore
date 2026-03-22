#!/usr/bin/env python3
"""
STAGE 3 SESSION 8: Delta_rz Eta Adaptation with Rigorous Validation

Repeats Session 6 Experiment B with MUCH more rigorous statistical validation:
- 10 seeds instead of 3
- Paired t-tests for significance
- Effect size reporting (Cohen's d)
- 95% confidence intervals

Using CANONICAL parameters as baseline (Candidate #2 failed validation).

Delta_rz mechanism: eta adapts based on change in responsiveness.
- delta_rz = resp_z(t) - resp_z(t-1)
- If delta_rz > 0: responsiveness improved → increase eta
- If delta_rz < 0: responsiveness degraded → decrease eta

Frozen frame: 7 -> 6 (eta scalar becomes per-cell adaptive)

Stage 3 exit criteria:
1. Adaptive eta beats fixed eta (p < 0.05)
2. Ground truth passes (gap > 0)
3. Eta rates converge to non-trivial values (not all equal, not all zero)
"""

import math
import random
import time
from scipy import stats

D = 12
NC = 6
W = 72

# 10-seed validation set
SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
KS = [4, 6, 8, 10]
BIRTH_SEED = 42
NOVEL_SEED_BASE = 99999


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


class Organism:
    """Stage 3: Delta_rz adaptive eta."""

    def __init__(self, seed=42, alive=False, eta=0.0003, adaptive_eta=False):
        # Canonical frozen parameters
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0

        self.seed = seed
        self.alive = alive
        self.adaptive_eta = adaptive_eta
        self.total_alpha_shift = 0.0
        self.total_eta_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

        if adaptive_eta:
            # Per-cell eta initialization
            self.eta = [
                [0.0001 + random.random() * 0.0009 for _ in range(D)]
                for _ in range(NC)
            ]
            # Previous resp_z for delta computation
            self.prev_resp_z = [
                [0.0 for _ in range(D)]
                for _ in range(NC)
            ]
            self.has_prev = False
        else:
            self.eta_scalar = eta

        self.eta_lo = 0.00005
        self.eta_hi = 0.003

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def eta_flat(self):
        if self.adaptive_eta:
            return [e for row in self.eta for e in row]
        return [self.eta_scalar] * (NC * D)

    def eta_stats(self):
        vals = self.eta_flat()
        mn = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mn)**2 for v in vals) / len(vals))
        return {
            'mean': mn, 'std': std,
            'min': min(vals), 'max': max(vals),
            'at_lo': sum(1 for v in vals if v <= self.eta_lo + 1e-10),
            'at_hi': sum(1 for v in vals if v >= self.eta_hi - 1e-10),
        }

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma

        # Bare dynamics
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

        # Signal-modulated dynamics
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

        # Online plasticity
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

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    if self.adaptive_eta:
                        eta_ik = self.eta[i][k]
                    else:
                        eta_ik = self.eta_scalar

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    # Alpha plasticity (canonical rule)
                    if abs(dev) < 0.01:
                        push = eta_ik * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = eta_ik * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = eta_ik * 0.1 * random.gauss(0, 1.0)

                    old_a = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old_a)

                    # Eta plasticity (delta_rz second-order)
                    if self.adaptive_eta and self.has_prev:
                        delta_rz = resp_z - self.prev_resp_z[i][k]
                        # Meta-rate = 0.1 (reuses drift multiplier)
                        push_e = 0.1 * eta_ik * math.tanh(delta_rz)

                        old_e = self.eta[i][k]
                        self.eta[i][k] += push_e
                        self.eta[i][k] = max(self.eta_lo,
                                             min(self.eta_hi, self.eta[i][k]))
                        self.total_eta_shift += abs(self.eta[i][k] - old_e)

                    if self.adaptive_eta:
                        self.prev_resp_z[i][k] = resp_z

            if self.adaptive_eta:
                self.has_prev = True

        # Attention
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][kk] * xs[j][kk] for kk in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # State update
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
                    for kk in range(D):
                        pull[kk] += weights[i][j] * (phi_bare[j][kk] - phi_bare[i][kk])
                p = [p[kk] + plast * self.eps * pull[kk] for kk in range(D)]

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


def per_seed_gap(alive, adaptive_eta, ks, seed, birth_seed):
    """Computes training gap for a single seed across all K values."""
    gaps = []
    for k in ks:
        sig_seed = birth_seed + k * 200
        sigs = make_signals(k, seed=sig_seed)
        org = Organism(seed=birth_seed, alive=alive, eta=0.0003, adaptive_eta=adaptive_eta)
        g = measure_gap(org, sigs, k, seed)
        gaps.append(g)
    return sum(gaps) / len(gaps)


def per_seed_novel_gap(alive, adaptive_eta, birth_seed, novel_seed_base):
    """Computes novel signal gap."""
    gaps = []
    for wi in range(6):
        for k in [6, 8]:
            nsigs = make_signals(k, seed=novel_seed_base + wi * 37 + k)
            ts = 77 + wi * 13 + k
            org = Organism(seed=birth_seed, alive=alive, eta=0.0003, adaptive_eta=adaptive_eta)
            g = measure_gap(org, nsigs, k, ts)
            gaps.append(g)
    return sum(gaps) / len(gaps)


def paired_ttest_summary(x, y, name_x, name_y):
    """Paired t-test with effect size."""
    diff = [yi - xi for xi, yi in zip(x, y)]
    mean_diff = sum(diff) / len(diff)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diff) / (len(diff) - 1))
    se = std_diff / math.sqrt(len(diff))
    t_crit = stats.t.ppf(0.975, len(diff) - 1)
    ci_95 = (mean_diff - t_crit * se, mean_diff + t_crit * se)
    t_stat, p_value = stats.ttest_rel(y, x)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'mean_diff': mean_diff,
        'ci_95': ci_95,
    }


def main():
    print("=" * 80)
    print("  STAGE 3 SESSION 8: DELTA_RZ ETA ADAPTATION (RIGOROUS VALIDATION)")
    print("=" * 80)
    print(f"\nBaseline: CANONICAL parameters")
    print(f"Seeds: {SEEDS}")
    print(f"K values: {KS}")
    print(f"\nFrozen frame: 7 -> 6 (eta scalar becomes per-cell adaptive)")
    print()

    t0 = time.time()

    # ── TRAINING GAPS ────────────────────────────────────────────
    print("Running training gap tests...")
    still_training = []
    fixed_training = []
    adaptive_training = []

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] seed={seed}", flush=True)

        g_still = per_seed_gap(alive=False, adaptive_eta=False, ks=KS,
                               seed=seed, birth_seed=BIRTH_SEED)
        still_training.append(g_still)

        g_fixed = per_seed_gap(alive=True, adaptive_eta=False, ks=KS,
                               seed=seed, birth_seed=BIRTH_SEED)
        fixed_training.append(g_fixed)

        g_adaptive = per_seed_gap(alive=True, adaptive_eta=True, ks=KS,
                                  seed=seed, birth_seed=BIRTH_SEED)
        adaptive_training.append(g_adaptive)

        print(f"      STILL: {g_still:+.4f}  |  fixed: {g_fixed:+.4f}  |  adaptive: {g_adaptive:+.4f}  |  delta: {g_adaptive - g_fixed:+.4f}")

    # Training summary
    print("\n" + "-" * 80)
    print("TRAINING GAP SUMMARY:")
    print("-" * 80)
    mean_still = sum(still_training) / len(still_training)
    mean_fixed = sum(fixed_training) / len(fixed_training)
    mean_adaptive = sum(adaptive_training) / len(adaptive_training)

    print(f"STILL:    mean={mean_still:+.4f}")
    print(f"Fixed:    mean={mean_fixed:+.4f} (delta vs STILL: {mean_fixed - mean_still:+.4f})")
    print(f"Adaptive: mean={mean_adaptive:+.4f} (delta vs STILL: {mean_adaptive - mean_still:+.4f})")

    training_stats = paired_ttest_summary(fixed_training, adaptive_training,
                                          "fixed", "adaptive")
    print(f"\nPaired t-test (adaptive vs fixed):")
    print(f"  t-statistic: {training_stats['t_stat']:.4f}")
    print(f"  p-value: {training_stats['p_value']:.6f}")
    print(f"  Mean difference: {training_stats['mean_diff']:+.4f}")
    print(f"  95% CI: ({training_stats['ci_95'][0]:+.4f}, {training_stats['ci_95'][1]:+.4f})")
    print(f"  Cohen's d: {training_stats['cohens_d']:.4f}")

    training_significant = training_stats['p_value'] < 0.05 and training_stats['mean_diff'] > 0
    print(f"\nTraining result: {'SIGNIFICANT IMPROVEMENT' if training_significant else 'NOT SIGNIFICANT'}")

    # ── NOVEL GAPS ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Running novel signal gap tests...")
    birth_seeds = [42, 137, 314, 2024, 7]

    still_novel = []
    fixed_novel = []
    adaptive_novel = []

    for bs in birth_seeds:
        print(f"  birth_seed={bs}...", flush=True)

        g_still = per_seed_novel_gap(alive=False, adaptive_eta=False,
                                     birth_seed=bs, novel_seed_base=NOVEL_SEED_BASE)
        still_novel.append(g_still)

        g_fixed = per_seed_novel_gap(alive=True, adaptive_eta=False,
                                     birth_seed=bs, novel_seed_base=NOVEL_SEED_BASE)
        fixed_novel.append(g_fixed)

        g_adaptive = per_seed_novel_gap(alive=True, adaptive_eta=True,
                                        birth_seed=bs, novel_seed_base=NOVEL_SEED_BASE)
        adaptive_novel.append(g_adaptive)

        print(f"      STILL: {g_still:+.4f}  |  fixed: {g_fixed:+.4f}  |  adaptive: {g_adaptive:+.4f}  |  delta: {g_adaptive - g_fixed:+.4f}")

    # Novel summary
    print("\n" + "-" * 80)
    print("NOVEL GAP SUMMARY:")
    print("-" * 80)
    mean_still_n = sum(still_novel) / len(still_novel)
    mean_fixed_n = sum(fixed_novel) / len(fixed_novel)
    mean_adaptive_n = sum(adaptive_novel) / len(adaptive_novel)

    print(f"STILL:    mean={mean_still_n:+.4f}")
    print(f"Fixed:    mean={mean_fixed_n:+.4f} (delta vs STILL: {mean_fixed_n - mean_still_n:+.4f})")
    print(f"Adaptive: mean={mean_adaptive_n:+.4f} (delta vs STILL: {mean_adaptive_n - mean_still_n:+.4f})")

    novel_stats = paired_ttest_summary(fixed_novel, adaptive_novel,
                                       "fixed", "adaptive")
    print(f"\nPaired t-test (adaptive vs fixed):")
    print(f"  t-statistic: {novel_stats['t_stat']:.4f}")
    print(f"  p-value: {novel_stats['p_value']:.6f}")
    print(f"  Mean difference: {novel_stats['mean_diff']:+.4f}")
    print(f"  95% CI: ({novel_stats['ci_95'][0]:+.4f}, {novel_stats['ci_95'][1]:+.4f})")
    print(f"  Cohen's d: {novel_stats['cohens_d']:.4f}")

    novel_significant = novel_stats['p_value'] < 0.05 and novel_stats['mean_diff'] > 0
    print(f"\nNovel result: {'SIGNIFICANT IMPROVEMENT' if novel_significant else 'NOT SIGNIFICANT'}")

    # ── ETA DISTRIBUTION ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Eta distribution analysis (post-training):")
    print("-" * 80)

    for s in [42, 137, 2024]:
        org = Organism(seed=BIRTH_SEED, alive=True, adaptive_eta=True)
        sigs = make_signals(8, seed=BIRTH_SEED + 1600)
        order = list(range(8))
        run_sequence(org, order, sigs, s, trial=0)
        es = org.eta_stats()
        print(f"  seed={s}: mean={es['mean']:.6f} std={es['std']:.6f} "
              f"min={es['min']:.6f} max={es['max']:.6f} "
              f"at_lo={es['at_lo']}/72 at_hi={es['at_hi']}/72")

    # ── STAGE 3 CRITERIA ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STAGE 3 EXIT CRITERIA:")
    print("=" * 80)

    ground_truth = mean_adaptive > 0.0
    print(f"1. Adaptive beats fixed (training): {'PASS' if training_significant else 'FAIL'} (p={training_stats['p_value']:.6f})")
    print(f"2. Adaptive beats fixed (novel):    {'PASS' if novel_significant else 'FAIL'} (p={novel_stats['p_value']:.6f})")
    print(f"3. Ground truth (gap > 0):           {'PASS' if ground_truth else 'FAIL'}")

    # Check eta convergence
    org_test = Organism(seed=BIRTH_SEED, alive=True, adaptive_eta=True)
    sigs_test = make_signals(8, seed=BIRTH_SEED + 1600)
    run_sequence(org_test, list(range(8)), sigs_test, 42, trial=0)
    es_test = org_test.eta_stats()

    non_trivial = (es_test['std'] > 1e-5 and
                   es_test['at_lo'] < 70 and
                   es_test['at_hi'] < 70)
    print(f"4. Eta rates non-trivial:            {'PASS' if non_trivial else 'FAIL'} (std={es_test['std']:.6f})")

    # ── FINAL VERDICT ────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FINAL VERDICT:")
    print("=" * 80)

    all_pass = training_significant and ground_truth and non_trivial

    if all_pass:
        print("\nSTAGE 3: PASS")
        print(f"Delta_rz eta adaptation beats fixed eta with statistical significance.")
        print(f"Frozen frame reduced: 7 -> 6")
        print(f"Training improvement: {training_stats['mean_diff']:+.4f} (d={training_stats['cohens_d']:.2f})")
        print(f"Novel improvement: {novel_stats['mean_diff']:+.4f} (d={novel_stats['cohens_d']:.2f})")
    elif training_significant and not ground_truth:
        print("\nSTAGE 3: FAIL (ground truth violation)")
    elif not training_significant:
        print("\nSTAGE 3: FAIL (no significant improvement)")
        print(f"Adaptive eta does NOT reliably beat fixed eta across seeds.")

    print(f"\nRuntime: {time.time() - t0:.0f}s")
    print("=" * 80)

    return {
        'training_significant': training_significant,
        'novel_significant': novel_significant,
        'ground_truth': ground_truth,
        'non_trivial': non_trivial,
        'training_stats': training_stats,
        'novel_stats': novel_stats,
        'stage3_pass': all_pass,
    }


if __name__ == '__main__':
    result = main()
