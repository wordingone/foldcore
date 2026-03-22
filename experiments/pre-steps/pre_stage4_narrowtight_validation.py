#!/usr/bin/env python3
"""
10-seed validation: NARROW_TIGHT [0.5, 1.6] vs Canonical [0.3, 1.8]

Task #18: Confirm whether NARROW_TIGHT improvement is real or CV artifact.
Protocol: 10 paired seeds, K=[4,6,8,10], n_perm=8, n_trials=6.

NOTE: Even if significant, this is a FIXED optimization finding, not an
adaptive target. The direction (narrower = better) implies an anti-signal
for adaptive clip bounds — expanding bounds when hitting them would be wrong.
"""

import math
import random

D = 12
NC = 6
W = 72


def vcosine(a, b):
    dot = na2 = nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi; na2 += ai * ai; nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15); nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10: return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


class Organism:
    def __init__(self, seed=42, alive=False, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9; self.eps = 0.15; self.tau = 0.3
        self.delta = 0.35; self.noise = 0.005; self.clip = 4.0; self.eta = 0.0003
        self.symmetry_break_mult = 0.3; self.amplify_mult = 0.5; self.drift_mult = 0.1
        self.threshold = 0.01; self.alpha_clip_lo = clip_lo; self.alpha_clip_hi = clip_hi
        self.alive = alive
        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(clip_lo, min(clip_hi, self.alpha[i][k]))

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D; km = (k - 1) % D
                row.append(math.tanh(
                    self.alpha[i][k] * xs[i][k] + beta * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)

        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D; km = (k - 1) % D
                    row.append(math.tanh(
                        self.alpha[i][k] * xs[i][k]
                        + beta * (xs[i][kp] + gamma * signal[kp])
                               * (xs[i][km] + gamma * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare

        if self.alive and signal:
            response = [
                [abs(phi_sig[i][k] - phi_bare[i][k]) for k in range(D)]
                for i in range(NC)
            ]
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
                    self.alpha[i][k] = max(self.alpha_clip_lo,
                                           min(self.alpha_clip_hi, self.alpha[i][k] + push))

        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    raw.append(sum(xs[i][k] * xs[j][k] for k in range(D)) / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        new = []
        for i in range(NC):
            p = phi_sig[i][:]
            bare_diff = [phi_bare[i][k] - xs[i][k] for k in range(D)]
            fp_d = vnorm(bare_diff) / max(vnorm(xs[i]), 1.0)
            plast = math.exp(-(fp_d * fp_d) / 0.0225)
            if plast > 0.01 and self.eps > 0:
                pull = [0.0] * D
                for j in range(NC):
                    if i == j or weights[i][j] < 1e-8: continue
                    for k in range(D):
                        pull[k] += weights[i][j] * (phi_bare[j][k] - phi_bare[i][k])
                p = [p[k] + plast * self.eps * pull[k] for k in range(D)]
            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p[k]
                v += random.gauss(0, self.noise)
                nx.append(max(-self.clip, min(self.clip, v)))
            new.append(nx)
        return new


def make_signals(K, seed=42):
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(D)] for _ in range(K)]


def gen_perms(K, n_perm=8, seed=42):
    random.seed(seed)
    idxs = list(range(K)); perms = []
    for _ in range(n_perm):
        p = idxs[:]; random.shuffle(p); perms.append(p)
    return perms


def run_sequence(org, perm, sigs, rng_seed, trial=0):
    rng = random.Random(rng_seed * 100 + trial)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for idx in perm:
        sig = sigs[idx]
        for _ in range(50):
            xs = org.step(xs, signal=sig if org.alive else None)
    return xs


def measure_gap(org, perm_list, sigs, seed, n_trials=6):
    finals = []
    for pi, perm in enumerate(perm_list):
        tf = []
        for t in range(n_trials):
            tf.append([r[:] for r in run_sequence(org, perm, sigs, seed + pi * 1000, t)])
        finals.append(tf)
    within = []; between = []
    for i in range(len(finals)):
        fi = finals[i]
        for a in range(len(fi)):
            for b in range(a + 1, len(fi)):
                for c in range(NC): within.append(vcosine(fi[a][c], fi[b][c]))
        for j in range(i + 1, len(finals)):
            fj = finals[j]
            for ta in fi:
                for tb in fj:
                    for c in range(NC): between.append(vcosine(ta[c], tb[c]))
    return (sum(within) / len(within) - sum(between) / len(between)) if within else 0.0


def eval_config(clip_lo, clip_hi, seeds, ks, n_perm=8, n_trials=6):
    gaps = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, clip_lo=clip_lo, clip_hi=clip_hi)
        sg = []
        for K in ks:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            sg.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
        gaps.append(sum(sg) / len(sg))
    return gaps


def mean(v): return sum(v) / len(v)
def std_pop(v): m = mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_p(a, b):
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = mean(diffs)
    if n < 2: return 1.0
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) + 1e-15
    t = md / se
    return 2.0 * (1.0 - _norm_cdf(abs(t)))


def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    return md / (sd + 1e-15)


if __name__ == '__main__':
    SEEDS = [42, 137, 2024, 999, 7, 314, 1618, 2718, 4242, 8888]
    KS = [4, 6, 8, 10]
    N_PERM = 8
    N_TRIALS = 6

    print("=" * W)
    print("  10-SEED VALIDATION: NARROW_TIGHT [0.5,1.6] vs CANONICAL [0.3,1.8]")
    print("=" * W)
    print(f"\nSeeds ({len(SEEDS)}): {SEEDS}")
    print(f"Protocol: K={KS}, n_perm={N_PERM}, n_trials={N_TRIALS}")

    print(f"\nRunning Canonical [0.3, 1.8]...", flush=True)
    can_gaps = eval_config(0.3, 1.8, SEEDS, KS, N_PERM, N_TRIALS)
    print(f"  mean={mean(can_gaps):+.4f}")

    print(f"Running NARROW_TIGHT [0.5, 1.6]...", flush=True)
    nt_gaps = eval_config(0.5, 1.6, SEEDS, KS, N_PERM, N_TRIALS)
    print(f"  mean={mean(nt_gaps):+.4f}")

    print("\n" + "=" * W)
    print("  PER-SEED RESULTS")
    print("=" * W)
    print(f"\n  {'Seed':<8} {'Canonical':>12} {'NARROW_TIGHT':>14} {'diff':>10}  dir")
    print(f"  {'-'*8} {'-'*12} {'-'*14} {'-'*10}")
    n_above = 0
    for i, s in enumerate(SEEDS):
        diff = nt_gaps[i] - can_gaps[i]
        dirn = "above" if diff > 0 else "below"
        if diff > 0: n_above += 1
        print(f"  {s:<8} {can_gaps[i]:>+12.4f} {nt_gaps[i]:>+14.4f} {diff:>+10.4f}  {dirn}")

    cm = mean(can_gaps); cs = std_pop(can_gaps)
    nm = mean(nt_gaps);  ns = std_pop(nt_gaps)
    diff = nm - cm
    d = cohens_d_paired(nt_gaps, can_gaps)
    p = paired_t_p(nt_gaps, can_gaps)
    cv_c = cs / abs(cm) * 100
    cv_n = ns / abs(nm) * 100

    print(f"\n  {'':12} {'mean':>10} {'std':>8} {'CV%':>7}")
    print(f"  {'Canonical':<12} {cm:>+10.4f} {cs:>8.4f} {cv_c:>6.1f}%")
    print(f"  {'NARROW_TIGHT':<12} {nm:>+10.4f} {ns:>8.4f} {cv_n:>6.1f}%")
    print(f"\n  Paired diff:    {diff:+.4f}  ({diff/abs(cm)*100:+.1f}% of canonical)")
    print(f"  Cohen's d:      {d:+.3f}")
    print(f"  p-value:        {p:.4f}")
    print(f"  Seeds above:    {n_above}/{len(SEEDS)}")

    print("\n" + "=" * W)
    print("  VERDICT")
    print("=" * W)

    if p < 0.05 and diff > 0:
        print(f"\n  SIGNIFICANT IMPROVEMENT (p={p:.4f}, d={d:+.3f})")
        print(f"  NARROW_TIGHT [0.5, 1.6] genuinely outperforms canonical [0.3, 1.8].")
        print(f"\n  CRITICAL IMPLICATION — ANTI-SIGNAL:")
        print(f"  Narrower bounds = better performance.")
        print(f"  The obvious adaptive strategy (expand bounds when alpha hits them)")
        print(f"  would INCREASE the clip range, which HURTS performance.")
        print(f"  This is an anti-signal pattern identical to c020 (delta_correlation)")
        print(f"  and c022 (error-push correlation) — adaptation in the intuitive")
        print(f"  direction actively degrades performance.")
        print(f"\n  SCIENTIFIC VALUE: Better fixed initialization found.")
        print(f"  RECOMMENDATION: Update canonical to [0.5, 1.6]. Do NOT attempt")
        print(f"  adaptive clip bounds using utilization/saturation signal.")
        print(f"  The Stage 4 adaptive target must be something other than clip bounds.")
    elif p < 0.05 and diff < 0:
        print(f"\n  SIGNIFICANT DEGRADATION (p={p:.4f}) — NARROW_TIGHT is worse.")
        print(f"  Clip tightening hurts. Proceed with canonical.")
    else:
        print(f"\n  NOT SIGNIFICANT (p={p:.4f})")
        print(f"  NARROW_TIGHT improvement does not survive 10-seed validation.")
        print(f"  The 5-seed result (p=0.096) was a CV artifact.")
        print(f"  Clip bounds confirmed non-binding. Architecture review applies.")

    print("=" * W)
