#!/usr/bin/env python3
"""
Stage 4 Clip Bounds Sensitivity Diagnostic

Phase 1 kill criterion for alpha_clip bounds as Stage 4 target.
Uses PAIRED seeds to fix c024 (independent 3-seed confound at CV=37%).

Canonical: alpha_clip_lo=0.3, alpha_clip_hi=1.8 (range=1.5)

Design:
- Same 5 seeds used for ALL clip conditions (paired design)
- Compare MI gap (variant_gap) across conditions
- Kill criterion: if all conditions produce similar MI gap → non-binding
- Proceed criterion: if >=1 condition differs by >10% from canonical → binding
- n_perm=8, n_trials=6 per seed (2× exposure protocol per c015)

Clip conditions tested:
  NARROW_TIGHT:   [0.5, 1.6]   range=1.1  (74% of canonical)
  NARROW:         [0.4, 1.6]   range=1.2  (80% of canonical)
  CANONICAL:      [0.3, 1.8]   range=1.5  (100% — baseline)
  WIDE:           [0.1, 2.2]   range=2.1  (140% of canonical)
  VERY_WIDE:      [0.05, 2.8]  range=2.75 (183% of canonical)
  ASYMM_LO:       [0.05, 1.8]  range=1.75 — only expand lower bound
  ASYMM_HI:       [0.3, 2.8]   range=2.5  — only expand upper bound

Adversarial note: Initial alpha distribution is ~[0.4, 1.8], so the LOWER
bound (0.3) may already be below the natural alpha range — it might not bind
at all. Testing ASYMM_LO vs ASYMM_HI reveals whether lower or upper bound is
the active constraint (or neither).
"""

import math
import random
import sys
import os

# ── Copy core infrastructure from harness.py ──────────────────────────────

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

        # Initialize alpha — SAME distribution regardless of clip bounds
        # This mirrors canonical: birth alpha ~ [0.4, 1.8]
        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]
        # Clamp birth alpha to clip bounds (matters for narrow conditions)
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(self.alpha_clip_lo,
                                       min(self.alpha_clip_hi, self.alpha[i][k]))

        # Track alpha range at end for diagnostics
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

                    old = self.alpha[i][k]
                    self.alpha[i][k] += push
                    new_val = max(self.alpha_clip_lo,
                                  min(self.alpha_clip_hi, self.alpha[i][k]))
                    self.alpha_steps += 1
                    if new_val != self.alpha[i][k]:
                        if self.alpha[i][k] < self.alpha_clip_lo:
                            self.alpha_lo_hits += 1
                        else:
                            self.alpha_hi_hits += 1
                    self.alpha[i][k] = new_val

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

        return new, phi_bare


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
            xs, _ = org.step(xs, signal=sig if org.alive else None)
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


def eval_clip(clip_lo, clip_hi, seeds, K=6, n_perm=8, n_trials=6):
    """Evaluate a single clip configuration across seeds. Returns per-seed gaps."""
    gaps = []
    for seed in seeds:
        sigs = make_signals(K, seed=seed + 500)
        perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
        org = Organism(seed=seed, alive=True, clip_lo=clip_lo, clip_hi=clip_hi)
        gap = measure_gap(org, perms, sigs, seed, n_trials=n_trials)
        gaps.append(gap)
    return gaps


def stats(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean)**2 for v in values) / n)
    return mean, std


def _norm_cdf(z):
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_pvalue(a, b):
    """One-sample t-test on differences d = a[i] - b[i]. Returns two-tailed p."""
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    mean_d = sum(diffs) / n
    if n < 2:
        return 1.0
    var_d = sum((d - mean_d)**2 for d in diffs) / (n - 1)
    se = math.sqrt(var_d / n) + 1e-15
    t = mean_d / se
    # Approximate p using normal for n>=5
    p = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return p


def cohens_d_paired(a, b):
    """Cohen's d for paired samples."""
    diffs = [a[i] - b[i] for i in range(len(a))]
    mean_d = sum(diffs) / len(diffs)
    std_d = math.sqrt(sum((d - mean_d)**2 for d in diffs) / max(len(diffs)-1, 1))
    return mean_d / (std_d + 1e-15)


# ── Clip saturation diagnostic ────────────────────────────────────────────

def measure_clip_saturation(clip_lo, clip_hi, seed=42, K=6, n_steps=50, n_perm=8, n_trials=6):
    """Run organism and report what fraction of alpha updates hit the clip bounds."""
    sigs = make_signals(K, seed=seed + 500)
    perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
    org = Organism(seed=seed, alive=True, clip_lo=clip_lo, clip_hi=clip_hi)
    measure_gap(org, perms, sigs, seed, n_trials=n_trials)
    lo_frac = org.alpha_lo_hits / max(org.alpha_steps, 1)
    hi_frac = org.alpha_hi_hits / max(org.alpha_steps, 1)
    total_frac = (org.alpha_lo_hits + org.alpha_hi_hits) / max(org.alpha_steps, 1)
    return lo_frac, hi_frac, total_frac, org.alpha_steps


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * W)
    print("  STAGE 4: CLIP BOUNDS SENSITIVITY DIAGNOSTIC")
    print("  Phase 1 kill criterion — paired seeds design")
    print("=" * W)

    # Protocol: 5 paired seeds, K=6, n_perm=8, n_trials=6
    SEEDS = [42, 137, 2024, 9999, 55555]
    K = 6
    N_PERM = 8
    N_TRIALS = 6

    print(f"\nProtocol: {len(SEEDS)} paired seeds, K={K}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Seeds: {SEEDS}")

    # ── Part 1: Clip saturation at canonical ──────────────────────────────
    print("\n" + "=" * W)
    print("  PART 1: Clip saturation analysis (canonical [0.3, 1.8])")
    print("=" * W)
    print(f"  {'Seed':<8} {'lo_hits%':<12} {'hi_hits%':<12} {'total%':<12} {'n_steps'}")
    total_lo = total_hi = total_steps = 0
    for s in SEEDS:
        lo_f, hi_f, tot_f, steps = measure_clip_saturation(0.3, 1.8, seed=s,
                                                             K=K, n_perm=N_PERM,
                                                             n_trials=N_TRIALS)
        print(f"  {s:<8} {lo_f*100:<12.3f} {hi_f*100:<12.3f} {tot_f*100:<12.3f} {steps}")
        total_lo += lo_f * steps
        total_hi += hi_f * steps
        total_steps += steps

    print(f"\n  Aggregate: lo_hits={total_lo/total_steps*100:.3f}%  "
          f"hi_hits={total_hi/total_steps*100:.3f}%  "
          f"total={( total_lo+total_hi)/total_steps*100:.3f}%")
    print(f"\n  Interpretation:")
    print(f"    - If total_hits < 1%: clips are structurally non-binding (likely non-binding)")
    print(f"    - If total_hits > 5%: clips are actively constraining alpha range (potentially binding)")
    print(f"    - Asymmetry lo vs hi reveals which bound is active")

    # ── Part 2: MI gap sweep across clip configurations ───────────────────
    print("\n" + "=" * W)
    print("  PART 2: MI gap across clip configurations (paired seeds)")
    print("=" * W)

    clip_conditions = [
        ("NARROW_TIGHT", 0.5, 1.6),   # range=1.1, 73% of canonical
        ("NARROW",       0.4, 1.6),   # range=1.2, 80% of canonical
        ("CANONICAL",    0.3, 1.8),   # range=1.5, baseline
        ("WIDE",         0.1, 2.2),   # range=2.1, 140% of canonical
        ("VERY_WIDE",    0.05, 2.8),  # range=2.75, 183% of canonical
        ("ASYMM_LO",     0.05, 1.8),  # expand lower only
        ("ASYMM_HI",     0.3, 2.8),   # expand upper only
    ]

    all_results = {}
    print(f"\n  {'Condition':<14} {'lo':>6} {'hi':>6} {'mean gap':>10} {'std':>8} {'CV%':>7}  per-seed gaps")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*10} {'-'*8} {'-'*7}")

    canonical_gaps = None
    for name, lo, hi in clip_conditions:
        gaps = eval_clip(lo, hi, SEEDS, K=K, n_perm=N_PERM, n_trials=N_TRIALS)
        mean_g, std_g = stats(gaps)
        cv = std_g / abs(mean_g) * 100 if abs(mean_g) > 1e-6 else 999.0
        gaps_str = "  ".join(f"{g:+.4f}" for g in gaps)
        print(f"  {name:<14} {lo:>6.3f} {hi:>6.3f} {mean_g:>+10.4f} {std_g:>8.4f} {cv:>6.1f}%  {gaps_str}")
        all_results[name] = gaps
        if name == "CANONICAL":
            canonical_gaps = gaps

    # ── Part 3: Paired statistical comparison vs canonical ────────────────
    print("\n" + "=" * W)
    print("  PART 3: Paired comparisons vs CANONICAL")
    print("=" * W)
    print(f"\n  {'Condition':<14} {'mean diff':>10} {'d':>8} {'p (paired-t)':>14}  verdict")
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*14}")

    any_significant = False
    significant_conditions = []
    for name, lo, hi in clip_conditions:
        if name == "CANONICAL":
            print(f"  {'CANONICAL':<14} {'(baseline)':>10} {'---':>8} {'---':>14}")
            continue
        gaps = all_results[name]
        mean_diff = sum(gaps) / len(gaps) - sum(canonical_gaps) / len(canonical_gaps)
        d = cohens_d_paired(gaps, canonical_gaps)
        p = paired_t_pvalue(gaps, canonical_gaps)
        sig = p < 0.05
        verdict = "SIG (BINDING)" if sig else "no sig"
        if sig:
            any_significant = True
            significant_conditions.append(name)
        print(f"  {name:<14} {mean_diff:>+10.4f} {d:>+8.3f} {p:>14.3f}  {verdict}")

    # ── Part 4: Verdict ───────────────────────────────────────────────────
    print("\n" + "=" * W)
    print("  VERDICT")
    print("=" * W)

    canonical_mean = sum(canonical_gaps) / len(canonical_gaps)
    canonical_std = stats(canonical_gaps)[1]
    canonical_cv = canonical_std / abs(canonical_mean) * 100

    print(f"\n  Canonical baseline: mean={canonical_mean:+.4f}, std={canonical_std:.4f}, "
          f"CV={canonical_cv:.1f}%")

    # Range of MI gaps across all conditions
    all_means = [sum(v)/len(v) for v in all_results.values()]
    mi_range = (max(all_means) - min(all_means)) / abs(canonical_mean) * 100
    print(f"  MI gap range across conditions: {mi_range:.1f}% of canonical")

    if any_significant:
        print(f"\n  VERDICT: BINDING")
        print(f"  Conditions with significant effect: {significant_conditions}")
        print(f"  Clip bounds are a binding structural parameter — viable Stage 4 target.")
        print(f"  Recommendation: Proceed to Phase 2 (10-seed validation of best candidate).")
    else:
        print(f"\n  VERDICT: NON-BINDING (kill criterion met)")
        print(f"  No clip condition produces significant MI difference vs canonical.")
        print(f"  Clip bounds do not constrain performance in the tested range.")
        print(f"  Recommendation: Declare clip bounds non-binding, advance to next Stage 4 candidate.")
        print(f"  Next candidates per plan: symmetry_break_mult, amplify_mult, drift_mult")

    print("\n" + "=" * W)
    print("  ADVERSARIAL NOTE")
    print("=" * W)
    print(f"""
  The initial alpha distribution is ~[0.4, 1.8] (birth distribution).
  The canonical lower clip (0.3) is BELOW the initial alpha floor (0.4).
  If plasticity mostly amplifies alpha toward higher values, the lower bound
  may be structurally irrelevant from birth. ASYMM_LO tests this directly.

  Similarly: if alpha diversification saturates at the upper clip (1.8),
  expanding it should increase performance. ASYMM_HI tests the upper bound.

  If NEITHER asymmetric condition shows an effect, the clip bounds are inert
  as a structural parameter — the system operates well within the allowed range.
  This is the same pattern as threshold: structurally real, but not binding.
""")
    print("=" * W)
