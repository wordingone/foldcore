#!/usr/bin/env python3
"""
Stage 4: Adaptive Tau — Phase 1 Experiment

MECHANISM: tau controls attention temperature. From the K-decomposition:
- K=4 (200 steps): tau=0.2 beats tau=0.3 (d=+1.336, p=0.003)
- K=8 (400 steps): tau=0.3 beats tau=0.2 (d=-0.266)

This means: optimal tau is LOW early (sharp attention, fast differentiation)
and HIGHER later (relaxed attention, stable integration).

ADAPTIVE RULE: Use plasticity signal (plast_i) as proxy for training stage.
  plast_i = exp(-fp_d_i^2 / 0.0225)  -- already computed per cell
  When plast_i is LOW (active learning) -> tau should be LOW (sharp)
  When plast_i is HIGH (settled) -> tau should be HIGHER (relaxed)

  tau_i = tau_min + (tau_max - tau_min) * plast_i

This is Principle II compliant: plast is computed by the same dynamics
that process input, not a separate evaluator.

CONTROLS:
  A: Canonical tau=0.3 (fixed)
  B: Optimal fixed tau=0.2
  C: Adaptive tau (plast-driven, tau_min=0.15, tau_max=0.35)
  D: Adaptive tau (wider range, tau_min=0.1, tau_max=0.5)

PROTOCOL: 5 paired seeds [42,137,2024,999,7], K=[4,6,8,10], n_perm=4, n_trials=3
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
SEEDS = [42, 137, 2024, 999, 7]
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
    """Organism with optional per-cell adaptive tau."""

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
            # Per-cell tau, initialized to midpoint
            self.tau_cells = [[0.5 * (tau_min + tau_max)] * D for _ in range(NC)]
            self.tau = None  # signal that per-cell is active
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

        # Compute attention weights — per-cell tau if adaptive
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    if self.adaptive_tau:
                        # Use cell i's tau (averaged over dimensions for scalar temp)
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

            # Update adaptive tau based on plasticity signal
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
    """Evaluate with multi-K averaging. Single organism per seed, reused across K."""
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
    print("  STAGE 4: ADAPTIVE TAU — PHASE 1 EXPERIMENT")
    print("  Plast-driven per-cell tau vs fixed tau baselines")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    sys.stdout.flush()

    conditions = {
        'A_canon_0.3': {'tau': 0.3},
        'B_fixed_0.2': {'tau': 0.2},
        'C_adapt_narrow': {'adaptive_tau': True, 'tau_min': 0.15, 'tau_max': 0.35},
        'D_adapt_wide':   {'adaptive_tau': True, 'tau_min': 0.1,  'tau_max': 0.5},
    }

    all_gaps = {}
    for name, kwargs in conditions.items():
        print(f"\n  Running {name}...", flush=True)
        gaps = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS, **kwargs)
        all_gaps[name] = gaps
        m = mean(gaps); s = std_pop(gaps)
        cv = s / abs(m) * 100 if m != 0 else float('inf')
        print(f"    mean={m:+.4f}, std={s:.4f}, CV={cv:.1f}%")
        print(f"    per-seed: {' '.join(f'{g:+.4f}' for g in gaps)}")
        sys.stdout.flush()

    # Pairwise comparisons against canonical
    baseline = all_gaps['A_canon_0.3']
    print(f"\n{'='*W}")
    print(f"  PAIRWISE COMPARISONS (vs A_canon_0.3 baseline)")
    print(f"{'='*W}\n")
    print(f"  {'condition':<20} {'mean':>8} {'diff':>8} {'d':>8} {'p':>8}  verdict")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, gaps in all_gaps.items():
        m = mean(gaps)
        if name == 'A_canon_0.3':
            print(f"  {name:<20} {m:>+8.4f} {'---':>8} {'---':>8} {'---':>8}  baseline")
            continue
        diff = m - mean(baseline)
        d = cohens_d_paired(gaps, baseline)
        p = paired_t_p(gaps, baseline)
        verdict = ""
        if p < 0.01: verdict = "*** STRONG"
        elif p < 0.05: verdict = "** SIG"
        elif abs(d) > 0.5: verdict = "* borderline"
        print(f"  {name:<20} {m:>+8.4f} {diff:>+8.4f} {d:>+8.3f} {p:>8.4f}  {verdict}")
    sys.stdout.flush()

    # Also compare adaptive vs best fixed
    print(f"\n{'='*W}")
    print(f"  ADAPTIVE vs BEST FIXED (B_fixed_0.2)")
    print(f"{'='*W}\n")
    best_fixed = all_gaps['B_fixed_0.2']
    for name in ['C_adapt_narrow', 'D_adapt_wide']:
        gaps = all_gaps[name]
        diff = mean(gaps) - mean(best_fixed)
        d = cohens_d_paired(gaps, best_fixed)
        p = paired_t_p(gaps, best_fixed)
        print(f"  {name} vs B_fixed_0.2: diff={diff:+.4f}, d={d:+.3f}, p={p:.4f}")

    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")
    # Check if any adaptive condition beats both fixed
    best_adaptive = None
    best_d = 0
    for name in ['C_adapt_narrow', 'D_adapt_wide']:
        d_canon = cohens_d_paired(all_gaps[name], baseline)
        d_fixed = cohens_d_paired(all_gaps[name], best_fixed)
        if d_canon > 0 and d_fixed > 0:
            if d_canon > best_d:
                best_d = d_canon
                best_adaptive = name
    if best_adaptive:
        d_c = cohens_d_paired(all_gaps[best_adaptive], baseline)
        p_c = paired_t_p(all_gaps[best_adaptive], baseline)
        d_f = cohens_d_paired(all_gaps[best_adaptive], best_fixed)
        p_f = paired_t_p(all_gaps[best_adaptive], best_fixed)
        print(f"\n  BEST ADAPTIVE: {best_adaptive}")
        print(f"    vs canonical: d={d_c:+.3f}, p={p_c:.4f}")
        print(f"    vs best fixed: d={d_f:+.3f}, p={p_f:.4f}")
        if p_c < 0.05 and d_f > 0:
            print(f"    ADAPTIVE TAU BEATS CANONICAL. Stage 4 candidate CONFIRMED.")
        elif p_c < 0.05 and d_f <= 0:
            print(f"    Beats canonical but NOT best fixed. This is CALIBRATION, not adaptation.")
        else:
            print(f"    Does not reach significance against canonical.")
    else:
        print(f"\n  No adaptive condition beats both fixed baselines.")
        print(f"  The tau=0.2 finding is likely CALIBRATION (better fixed value), not adaptation.")
    print()
    print("=" * W)
