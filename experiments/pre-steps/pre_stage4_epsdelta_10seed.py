#!/usr/bin/env python3
"""
Stage 4: Eps/Delta 10-Seed Validation

Entry 046 found at 5 seeds:
  eps=0.05:  d=-3.092, p=0.000 (strong degradation vs canonical eps=0.15)
  delta=0.1: d=-3.122, p=0.000 (strong degradation vs canonical delta=0.35)
  eps=0.5:   d=+0.674, p=0.131 (borderline improvement)
  delta=0.7: d=+0.738, p=0.099 (borderline improvement)

BUT Entry 046 also found tau=0.2 d=+1.202, p=0.007 — which Entry 050 showed
is a 5-seed false positive (10-seed: d=+0.317, p=0.317).

This script validates ALL eps/delta conditions from Entry 046 at 10 seeds.
Key questions:
1. Do the degradation effects (eps=0.05, delta=0.1) survive? (probably yes — 3x larger)
2. Do the borderline improvements (eps=0.5, delta=0.7) survive?
3. Does ANY non-canonical value significantly beat canonical?

Protocol: 10 paired seeds, K=[4,6,8,10], n_perm=4, n_trials=3
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
                 clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9
        self.eps = eps; self.delta = delta
        self.noise = NOISE_C; self.clip = 4.0
        self.eta = 0.0003; self.tau = tau
        self.symmetry_break_mult = 0.3
        self.amplify_mult = 0.5
        self.drift_mult = 0.1
        self.threshold = 0.01
        self.alpha_clip_lo = clip_lo
        self.alpha_clip_hi = clip_hi
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


# ── Conditions ──────────────────────────────────────────────────────
# From Entry 046, these are the conditions that showed signal:
CONDITIONS = {
    # Canonical baseline
    'CANON':        {'eps': 0.15, 'delta': 0.35},
    # Eps conditions
    'eps=0.05':     {'eps': 0.05, 'delta': 0.35},   # d=-3.092 at 5 seeds
    'eps=0.0':      {'eps': 0.0,  'delta': 0.35},   # d=-0.630 (borderline)
    'eps=0.3':      {'eps': 0.3,  'delta': 0.35},   # d=+0.426
    'eps=0.5':      {'eps': 0.5,  'delta': 0.35},   # d=+0.674 (borderline improvement)
    # Delta conditions
    'delta=0.1':    {'eps': 0.15, 'delta': 0.1},    # d=-3.122 at 5 seeds
    'delta=0.2':    {'eps': 0.15, 'delta': 0.2},    # d=-1.384
    'delta=0.5':    {'eps': 0.15, 'delta': 0.5},    # d=+0.580 (borderline)
    'delta=0.7':    {'eps': 0.15, 'delta': 0.7},    # d=+0.738 (borderline improvement)
}


if __name__ == '__main__':
    print("=" * W)
    print("  STAGE 4: EPS/DELTA 10-SEED VALIDATION")
    print("  Testing all Entry 046 conditions at 10 seeds")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Conditions: {len(CONDITIONS)}")
    sys.stdout.flush()

    results = {}
    for name, params in CONDITIONS.items():
        print(f"\n  Running {name}...", flush=True)
        gaps = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS, **params)
        results[name] = gaps
        m = mean(gaps); s = std_pop(gaps)
        print(f"    mean={m:+.4f}, std={s:.4f}, CV={s/abs(m)*100:.1f}%")
        for i, seed in enumerate(SEEDS):
            print(f"      seed {seed:>5}: {gaps[i]:+.4f}")
        sys.stdout.flush()

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*W}\n")

    canon = results['CANON']
    cm = mean(canon)

    print(f"  {'Condition':<15} {'Mean':>8} {'Std':>8} {'d_vs_can':>9} {'p_vs_can':>9} {'%diff':>7}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*7}")

    for name, gaps in results.items():
        m = mean(gaps); s = std_pop(gaps)
        if name == 'CANON':
            print(f"  {name:<15} {m:>+8.4f} {s:>8.4f} {'BASE':>9} {'BASE':>9} {'BASE':>7}")
        else:
            d = cohens_d_paired(gaps, canon)
            p = paired_t_p(gaps, canon)
            pct = (m - cm) / abs(cm) * 100
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {name:<15} {m:>+8.4f} {s:>8.4f} {d:>+9.3f} {p:>9.4f} {pct:>+7.1f}% {sig}")

    # ── Key comparisons ──────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  KEY COMPARISONS (vs Entry 046 5-seed results)")
    print(f"{'='*W}\n")

    key_conditions = [
        ('eps=0.05', -3.092),
        ('eps=0.5', +0.674),
        ('delta=0.1', -3.122),
        ('delta=0.7', +0.738),
    ]

    for name, d5 in key_conditions:
        gaps = results[name]
        d10 = cohens_d_paired(gaps, canon)
        p10 = paired_t_p(gaps, canon)
        print(f"  {name}:  5-seed d={d5:+.3f}  →  10-seed d={d10:+.3f}, p={p10:.4f}")

    # ── Verdict ──────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")

    any_beats_canon = False
    for name, gaps in results.items():
        if name == 'CANON': continue
        d = cohens_d_paired(gaps, canon)
        p = paired_t_p(gaps, canon)
        if d > 0 and p < 0.05:
            any_beats_canon = True
            print(f"\n  {name} BEATS canonical: d={d:+.3f}, p={p:.4f}")

    if not any_beats_canon:
        print(f"""
  NO non-canonical eps or delta value significantly beats canonical.

  If degradation effects survive: canonical values are locally optimal.
  This is CALIBRATION — the canonical values are good fixed values.
  Neither eps nor delta is a viable Stage 4 adaptive target.

  Combined with tau (c032), threshold (c023), clip bounds (c025),
  eta (vacuously passed), and beta/gamma (impossible under Principle II):
  ALL structural parameters are either non-binding or impossible.

  Stage 4 assessment needed: vacuous passage under Amendment 1?
""")
    else:
        print(f"""
  At least one condition beats canonical. Further investigation needed
  to determine if this represents adaptive opportunity or better calibration.
""")
    print("=" * W)
