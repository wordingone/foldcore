#!/usr/bin/env python3
"""
Stage 4: Adaptive Tau — Phase 2 Validation

Phase 1 found: C_adapt_narrow (plast-driven tau [0.15,0.35]) beats:
  - canonical tau=0.3: d=+1.944, p=0.0000
  - fixed tau=0.2:     d=+0.615, p=0.169 (borderline, need more seeds)

Phase 2: 10 seeds to resolve adaptive vs best fixed.
If adaptive beats fixed tau=0.2 at p<0.05: Stage 4 confirmed.
If not: tau=0.2 is calibration, not adaptation.

Protocol: 10 paired seeds, K=[4,6,8,10], n_perm=4, n_trials=3
Only conditions B (fixed 0.2) and C (adapt narrow) to save compute.
"""

import math
import random
import sys

D = 12
NC = 6
W = 72

EPS_C = 0.15
DELTA_C = 0.35
NOISE_C = 0.005

K_VALUES = [4, 6, 8, 10]
SEEDS = [42, 137, 2024, 999, 7, 10, 11, 12, 13, 14]
N_PERM = 4
N_TRIALS = 3


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
    def __init__(self, seed=42, alive=False,
                 tau=0.3, eps=EPS_C, delta=DELTA_C,
                 clip_lo=0.3, clip_hi=1.8,
                 adaptive_tau=False, tau_min=0.15, tau_max=0.35):
        self.beta = 0.5; self.gamma = 0.9
        self.eps = eps; self.delta = delta
        self.noise = NOISE_C; self.clip = 4.0
        self.eta = 0.0003
        self.symmetry_break_mult = 0.3
        self.amplify_mult = 0.5
        self.drift_mult = 0.1
        self.threshold = 0.01
        self.alpha_clip_lo = clip_lo
        self.alpha_clip_hi = clip_hi
        self.alive = alive
        self.adaptive_tau = adaptive_tau
        self.tau_min = tau_min
        self.tau_max = tau_max

        if adaptive_tau:
            self.tau_cells = [[0.5 * (tau_min + tau_max)] * D for _ in range(NC)]
            self.tau = None
        else:
            self.tau = tau
            self.tau_cells = None

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
                    self.alpha[i][k] * xs[i][k]
                    + beta * xs[i][kp] * xs[i][km]))
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
                    if self.adaptive_tau:
                        tau_i = sum(self.tau_cells[i]) / D
                    else:
                        tau_i = self.tau
                    raw.append(sum(xs[i][k] * xs[j][k] for k in range(D)) / (D * tau_i))
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

            if self.adaptive_tau:
                for k in range(D):
                    self.tau_cells[i][k] = self.tau_min + (self.tau_max - self.tau_min) * plast
        return new


def make_signals(K, seed=42):
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(D)] for _ in range(K)]


def gen_perms(K, n_perm=4, seed=42):
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


def measure_gap(org, perm_list, sigs, seed, n_trials=3):
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


def eval_multik(seeds, k_values, n_perm, n_trials, **org_kwargs):
    gaps = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, **org_kwargs)
        k_gaps = []
        for K in k_values:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            k_gaps.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
        gaps.append(sum(k_gaps) / len(k_gaps))
    return gaps


def mean(v): return sum(v) / len(v) if v else 0.0
def std_pop(v): m = mean(v); return math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if v else 0.0

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def paired_t_p(a, b):
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = mean(diffs)
    if n < 2: return 1.0
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) + 1e-15
    return 2.0 * (1.0 - _norm_cdf(abs(md / se)))

def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    return md / (sd + 1e-15)


if __name__ == '__main__':
    print("=" * W)
    print("  STAGE 4: ADAPTIVE TAU — PHASE 2 VALIDATION")
    print("  C_adapt_narrow [0.15,0.35] vs B_fixed_0.2")
    print("  10 seeds — definitive test")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    sys.stdout.flush()

    # Also include canonical for reference
    print(f"\n  Running A_canon_0.3...", flush=True)
    canon_gaps = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS, tau=0.3)
    print(f"    mean={mean(canon_gaps):+.4f}, std={std_pop(canon_gaps):.4f}")
    for i, s in enumerate(SEEDS):
        print(f"      seed {s:>5}: {canon_gaps[i]:+.4f}")
    sys.stdout.flush()

    print(f"\n  Running B_fixed_0.2...", flush=True)
    fixed_gaps = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS, tau=0.2)
    print(f"    mean={mean(fixed_gaps):+.4f}, std={std_pop(fixed_gaps):.4f}")
    for i, s in enumerate(SEEDS):
        print(f"      seed {s:>5}: {fixed_gaps[i]:+.4f}")
    sys.stdout.flush()

    print(f"\n  Running C_adapt_narrow...", flush=True)
    adapt_gaps = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS,
                              adaptive_tau=True, tau_min=0.15, tau_max=0.35)
    print(f"    mean={mean(adapt_gaps):+.4f}, std={std_pop(adapt_gaps):.4f}")
    for i, s in enumerate(SEEDS):
        print(f"      seed {s:>5}: {adapt_gaps[i]:+.4f}")
    sys.stdout.flush()

    # Results
    print(f"\n{'='*W}")
    print(f"  RESULTS")
    print(f"{'='*W}\n")

    print(f"  {'Condition':<20} {'Mean':>8} {'Std':>8} {'CV%':>6}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*6}")
    for name, gaps in [('A_canon_0.3', canon_gaps), ('B_fixed_0.2', fixed_gaps), ('C_adapt_narrow', adapt_gaps)]:
        m = mean(gaps); s = std_pop(gaps); cv = s / abs(m) * 100
        print(f"  {name:<20} {m:>+8.4f} {s:>8.4f} {cv:>6.1f}")

    print(f"\n  Pairwise comparisons:")
    # Adapt vs Canon
    d_ac = cohens_d_paired(adapt_gaps, canon_gaps)
    p_ac = paired_t_p(adapt_gaps, canon_gaps)
    print(f"    C_adapt vs A_canon:  d={d_ac:+.3f}, p={p_ac:.4f}")

    # Fixed vs Canon
    d_fc = cohens_d_paired(fixed_gaps, canon_gaps)
    p_fc = paired_t_p(fixed_gaps, canon_gaps)
    print(f"    B_fixed vs A_canon:  d={d_fc:+.3f}, p={p_fc:.4f}")

    # CRITICAL: Adapt vs Fixed
    d_af = cohens_d_paired(adapt_gaps, fixed_gaps)
    p_af = paired_t_p(adapt_gaps, fixed_gaps)
    print(f"    C_adapt vs B_fixed:  d={d_af:+.3f}, p={p_af:.4f}  *** CRITICAL TEST ***")

    # Per-seed comparison: adapt vs fixed
    print(f"\n  Per-seed (adapt - fixed):")
    diffs = [adapt_gaps[i] - fixed_gaps[i] for i in range(len(SEEDS))]
    n_pos = sum(1 for d in diffs if d > 0)
    for i, s in enumerate(SEEDS):
        sign = "+" if diffs[i] > 0 else "-"
        print(f"    seed {s:>5}: adapt={adapt_gaps[i]:+.4f}, fixed={fixed_gaps[i]:+.4f}, diff={diffs[i]:+.5f} {sign}")
    print(f"    {n_pos}/{len(SEEDS)} seeds favor adaptive")

    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")
    if p_af < 0.05 and d_af > 0:
        print(f"""
  ADAPTIVE TAU BEATS BEST FIXED at p={p_af:.4f}, d={d_af:+.3f}.
  STAGE 4 CONFIRMED: tau (attention temperature) is now adaptive,
  driven by plasticity signal (Principle II compliant).

  Frozen frame shrinks: tau moves from 'frozen' to 'adaptive'.
  Current frozen frame: 5/8 (alpha, eta, tau adaptive).
""")
    elif d_af > 0 and p_af < 0.10:
        print(f"""
  BORDERLINE: d={d_af:+.3f}, p={p_af:.4f}. Suggestive but not definitive.
  Consider Phase 3 with 20 seeds or refined tau range.
""")
    elif d_af > 0:
        print(f"""
  TREND in correct direction (d={d_af:+.3f}) but NOT significant (p={p_af:.4f}).
  The adaptive rule does not clearly beat fixed tau=0.2.
  This is likely CALIBRATION: tau=0.2 is simply a better fixed value.
  Update canonical tau from 0.3 to 0.2 and move to other Stage 4 targets.
""")
    else:
        print(f"""
  ADAPTIVE WORSE THAN FIXED (d={d_af:+.3f}, p={p_af:.4f}).
  The plast-driven tau rule does not work. The tau=0.2 finding is pure calibration.
  Update canonical tau and move on.
""")
    print("=" * W)
