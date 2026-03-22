#!/usr/bin/env python3
"""
Stage 4 Clip Bounds Sensitivity Diagnostic (exact spec)

Tests alpha_clip_lo/hi configurations using PAIRED seeds.
Protocol: n_perm=8, n_trials=6, K=[4,6,8,10] per c015/c017.
Seeds: [42, 137, 2024, 999, 7] — same for ALL configs.

Configs:
  Canonical:       [0.3, 1.8]
  Narrow:          [0.5, 1.5]
  Wide:            [0.1, 2.5]
  Asymmetric-low:  [0.1, 1.8]
  Asymmetric-high: [0.3, 2.5]
  Very narrow:     [0.7, 1.3]  <-- strongest test: range=0.6 vs canonical 1.5
"""

import math
import random


D = 12
NC = 6
W = 72


def vcosine(a, b):
    dot = na2 = nb2 = 0.0
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
    def __init__(self, seed=42, alive=False, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.eta = 0.0003
        self.symmetry_break_mult = 0.3
        self.amplify_mult = 0.5
        self.drift_mult = 0.1
        self.threshold = 0.01
        self.alpha_clip_lo = clip_lo
        self.alpha_clip_hi = clip_hi
        self.seed = seed
        self.alive = alive

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]
        # Clamp birth alpha to clip bounds
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(self.alpha_clip_lo,
                                       min(self.alpha_clip_hi, self.alpha[i][k]))

        # Tracking for alpha distribution stats
        self.alpha_sum = 0.0
        self.alpha_sum_sq = 0.0
        self.alpha_lo_hits = 0
        self.alpha_hi_hits = 0
        self.alpha_steps = 0

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
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

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
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < self.threshold:
                        push = self.eta * self.symmetry_break_mult * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * self.amplify_mult
                    else:
                        push = self.eta * self.drift_mult * random.gauss(0, 1.0)

                    new_val = self.alpha[i][k] + push
                    clipped = max(self.alpha_clip_lo, min(self.alpha_clip_hi, new_val))

                    # Track distribution
                    self.alpha_steps += 1
                    self.alpha_sum += clipped
                    self.alpha_sum_sq += clipped * clipped
                    if new_val < self.alpha_clip_lo:
                        self.alpha_lo_hits += 1
                    elif new_val > self.alpha_clip_hi:
                        self.alpha_hi_hits += 1

                    self.alpha[i][k] = clipped

        # Attention
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


def make_signals(K, seed=42):
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(D)] for _ in range(K)]


def gen_perms(K, n_perm=8, seed=42):
    random.seed(seed)
    idxs = list(range(K))
    perms = []
    for _ in range(n_perm):
        p = idxs[:]
        random.shuffle(p)
        perms.append(p)
    return perms


def run_sequence(org, perm, sigs, rng_seed, trial=0, n_steps=50):
    rng = random.Random(rng_seed * 100 + trial)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for idx in perm:
        sig = sigs[idx]
        for _ in range(n_steps):
            xs = org.step(xs, signal=sig if org.alive else None)
    return xs


def measure_gap(org, perm_list, sigs, seed, n_trials=6):
    finals_per_perm = []
    for perm_i, perm in enumerate(perm_list):
        trial_finals = []
        for t in range(n_trials):
            final_xs = run_sequence(org, perm, sigs, seed + perm_i * 1000, trial=t)
            trial_finals.append([row[:] for row in final_xs])
        finals_per_perm.append(trial_finals)

    within = []
    between = []
    for i in range(len(perm_list)):
        fi_all = finals_per_perm[i]
        for a_idx in range(len(fi_all)):
            for b_idx in range(a_idx + 1, len(fi_all)):
                for ci in range(NC):
                    within.append(vcosine(fi_all[a_idx][ci], fi_all[b_idx][ci]))
        for j in range(i + 1, len(perm_list)):
            fj_all = finals_per_perm[j]
            for ta in fi_all:
                for tb in fj_all:
                    for ci in range(NC):
                        between.append(vcosine(ta[ci], tb[ci]))

    avg_within = sum(within) / len(within) if within else 0.0
    avg_between = sum(between) / len(between) if between else 0.0
    return avg_within - avg_between


def eval_config(clip_lo, clip_hi, seeds, ks, n_perm=8, n_trials=6):
    """Evaluate a clip config across seeds and K values. Returns per-seed gaps + alpha stats."""
    gaps = []
    alpha_means = []
    alpha_stds = []
    lo_pcts = []
    hi_pcts = []

    for seed in seeds:
        seed_gaps = []
        org = Organism(seed=seed, alive=True, clip_lo=clip_lo, clip_hi=clip_hi)

        for K in ks:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            gap = measure_gap(org, perms, sigs, seed, n_trials=n_trials)
            seed_gaps.append(gap)

        seed_gap = sum(seed_gaps) / len(seed_gaps)
        gaps.append(seed_gap)

        # Alpha distribution stats
        n = org.alpha_steps
        if n > 0:
            mean_a = org.alpha_sum / n
            var_a = org.alpha_sum_sq / n - mean_a ** 2
            std_a = math.sqrt(max(var_a, 0.0))
            lo_pct = org.alpha_lo_hits / n * 100
            hi_pct = org.alpha_hi_hits / n * 100
        else:
            mean_a = std_a = lo_pct = hi_pct = 0.0

        alpha_means.append(mean_a)
        alpha_stds.append(std_a)
        lo_pcts.append(lo_pct)
        hi_pcts.append(hi_pct)

    return gaps, alpha_means, alpha_stds, lo_pcts, hi_pcts


def mean(v): return sum(v) / len(v)
def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m)**2 for x in v) / len(v))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_pvalue(a, b):
    """Two-tailed paired t-test p-value."""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = mean(diffs)
    if n < 2:
        return 1.0
    var_d = sum((d - mean_d)**2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) + 1e-15
    t = mean_d / se
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return p


def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    mean_d = mean(diffs)
    std_d = math.sqrt(sum((d - mean_d)**2 for d in diffs) / max(len(diffs)-1, 1))
    return mean_d / (std_d + 1e-15)


if __name__ == '__main__':
    print("=" * W)
    print("  STAGE 4: CLIP BOUNDS SENSITIVITY DIAGNOSTIC")
    print("  Paired seeds design — testing structural role of alpha_clip bounds")
    print("=" * W)

    SEEDS = [42, 137, 2024, 999, 7]
    KS = [4, 6, 8, 10]
    N_PERM = 8
    N_TRIALS = 6

    print(f"\nProtocol: {len(SEEDS)} paired seeds, K={KS}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Seeds: {SEEDS}")

    configs = [
        ("Canonical",      0.3, 1.8),
        ("Narrow",         0.5, 1.5),
        ("Wide",           0.1, 2.5),
        ("Asymm-low",      0.1, 1.8),
        ("Asymm-high",     0.3, 2.5),
        ("Very narrow",    0.7, 1.3),
    ]

    print(f"\nRunning {len(configs)} configurations × {len(SEEDS)} seeds × {len(KS)} K values...")
    print(f"(~{len(configs) * len(SEEDS) * len(KS) * N_PERM * N_TRIALS} total permutation runs)\n")

    all_gaps = {}
    all_alpha_stats = {}

    for name, lo, hi in configs:
        print(f"  [{name}] clip=[{lo}, {hi}]... ", end='', flush=True)
        gaps, a_means, a_stds, lo_pcts, hi_pcts = eval_config(
            lo, hi, SEEDS, KS, n_perm=N_PERM, n_trials=N_TRIALS)
        all_gaps[name] = gaps
        all_alpha_stats[name] = (a_means, a_stds, lo_pcts, hi_pcts)
        m = mean(gaps)
        s = std(gaps)
        print(f"mean={m:+.4f}  std={s:.4f}")

    canonical_gaps = all_gaps["Canonical"]

    # ── Results table ──────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  MI GAP RESULTS (paired vs Canonical)")
    print("=" * W)
    print(f"\n  {'Config':<14} {'lo':>5} {'hi':>5} {'mean':>8} {'std':>7} {'CV%':>6} {'diff':>8} {'d':>7} {'p':>8}  per-seed")
    print(f"  {'-'*14} {'-'*5} {'-'*5} {'-'*8} {'-'*7} {'-'*6} {'-'*8} {'-'*7} {'-'*8}")

    can_mean = mean(canonical_gaps)
    can_std = std(canonical_gaps)
    can_cv = can_std / abs(can_mean) * 100

    for name, lo, hi in configs:
        gaps = all_gaps[name]
        m = mean(gaps)
        s = std(gaps)
        cv = s / abs(m) * 100 if abs(m) > 1e-6 else 999.0

        if name == "Canonical":
            diff_str = "(baseline)"
            d_str = "---"
            p_str = "---"
        else:
            diff = m - can_mean
            d = cohens_d_paired(gaps, canonical_gaps)
            p = paired_t_pvalue(gaps, canonical_gaps)
            diff_str = f"{diff:+.4f}"
            d_str = f"{d:+.3f}"
            p_str = f"{p:.3f}"

        per_seed = "  ".join(f"{g:+.4f}" for g in gaps)
        print(f"  {name:<14} {lo:>5.2f} {hi:>5.2f} {m:>+8.4f} {s:>7.4f} {cv:>5.1f}% "
              f"{diff_str:>8} {d_str:>7} {p_str:>8}  {per_seed}")

    # ── Alpha distribution table ───────────────────────────────────────────
    print("\n" + "=" * W)
    print("  ALPHA DISTRIBUTION STATS (averaged across seeds)")
    print("=" * W)
    print(f"\n  {'Config':<14} {'alpha_mean':>11} {'alpha_std':>10} {'lo_bound%':>10} {'hi_bound%':>10}  interpretation")
    print(f"  {'-'*14} {'-'*11} {'-'*10} {'-'*10} {'-'*10}")

    for name, lo, hi in configs:
        a_means, a_stds, lo_pcts, hi_pcts = all_alpha_stats[name]
        am = mean(a_means)
        as_ = mean(a_stds)
        lp = mean(lo_pcts)
        hp = mean(hi_pcts)
        total_sat = lp + hp
        if total_sat > 10:
            interp = "HIGH saturation — clip actively constrains"
        elif total_sat > 2:
            interp = "MODERATE saturation"
        else:
            interp = "low saturation — clip inert"
        print(f"  {name:<14} {am:>11.4f} {as_:>10.4f} {lp:>9.2f}% {hp:>9.2f}%  {interp}")

    # ── Alpha profile similarity ───────────────────────────────────────────
    print("\n" + "=" * W)
    print("  ALPHA PROFILE COMPARISON (qualitative)")
    print("=" * W)
    can_am = mean(all_alpha_stats["Canonical"][0])
    can_as = mean(all_alpha_stats["Canonical"][1])
    print(f"\n  Canonical alpha: mean={can_am:.4f}, std={can_as:.4f}")
    print(f"\n  {'Config':<14} {'mean shift':>11} {'std shift':>10}  qualitative")
    for name, lo, hi in configs:
        if name == "Canonical":
            continue
        a_means, a_stds, _, _ = all_alpha_stats[name]
        am = mean(a_means)
        as_ = mean(a_stds)
        ms = am - can_am
        ss = as_ - can_as
        if abs(ms) > 0.05 or abs(ss) > 0.02:
            qual = "DIFFERENT profile"
        else:
            qual = "similar profile"
        print(f"  {name:<14} {ms:>+11.4f} {ss:>+10.4f}  {qual}")

    # ── Verdict ───────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  VERDICT")
    print("=" * W)

    mi_range = (max(mean(all_gaps[n]) for n, _, _ in configs) -
                min(mean(all_gaps[n]) for n, _, _ in configs))
    mi_range_pct = mi_range / abs(can_mean) * 100

    sig_configs = []
    for name, lo, hi in configs:
        if name == "Canonical":
            continue
        p = paired_t_pvalue(all_gaps[name], canonical_gaps)
        if p < 0.05:
            sig_configs.append(name)

    print(f"\n  Canonical: mean={can_mean:+.4f}, std={can_std:.4f}, CV={can_cv:.1f}%")
    print(f"  MI gap range across configs: {mi_range_pct:.1f}% of canonical")

    if sig_configs:
        print(f"\n  VERDICT: BINDING")
        print(f"  Significant conditions: {sig_configs}")
        print(f"  Clip bounds are a viable Stage 4 target.")
        print(f"  Proceed to Phase 2 (10-seed validation).")
    else:
        print(f"\n  VERDICT: NON-BINDING (kill criterion met)")
        print(f"  No clip configuration produces p<0.05 vs canonical.")
        print(f"  Clip bounds are non-binding structural parameters.")
        print(f"\n  ARCHITECTURE REVIEW TRIGGER FIRES:")
        print(f"  Both Stage 4 structural candidates (threshold, clip bounds) are non-binding.")
        print(f"  Team-lead must rule on path forward:")
        print(f"    A. Declare Stage 4 vacuous (Amendment 1) — requires mechanism test")
        print(f"    B. Skip to Stage 6 (functional form) — constitutional ruling required")
        print(f"    C. Identify novel Stage 4 structural candidate")

    print("\n" + "=" * W)
