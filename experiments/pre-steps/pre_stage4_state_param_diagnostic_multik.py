#!/usr/bin/env python3
"""
Stage 4: State Update Parameter Sensitivity Diagnostic (Multi-K)

METHODOLOGY: K=[4,6,8,10] averaging per c027.
K=6-only diagnostic had CV≈14.8% — insufficient for reliable binding detection.
Multi-K averaging achieves CV≈5.1%.

Tests tau, eps, delta — the state update mechanism parameters.
These have NEVER been varied. If binding, they are Stage 4 targets
outside the plasticity rule entirely.

Sweep (hold others at canonical):
  tau:   [0.1, 0.2, 0.3, 0.5, 0.8]   canonical=0.3  (attention temperature)
  eps:   [0.0, 0.05, 0.15, 0.3, 0.5]  canonical=0.15 (attention pull strength)
  delta: [0.1, 0.2, 0.35, 0.5, 0.7]   canonical=0.35 (state mixing rate)

Protocol: 5 paired seeds [42, 137, 2024, 999, 7], K=[4,6,8,10], n_perm=8, n_trials=6
Per seed: MI gap = average of measure_gap at K=4, K=6, K=8, K=10.
"""

import math
import random

D = 12
NC = 6
W = 72

# Canonical values
TAU_C   = 0.3
EPS_C   = 0.15
DELTA_C = 0.35
NOISE_C = 0.005

# Multi-K protocol (c027)
K_VALUES = [4, 6, 8, 10]


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
                 tau=TAU_C, eps=EPS_C, delta=DELTA_C,
                 clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9
        self.eps = eps; self.tau = tau; self.delta = delta
        self.noise = NOISE_C; self.clip = 4.0
        self.eta = 0.0003
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

        # Attention using self.tau
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

        # State update using self.eps, self.delta
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


def eval_params_multik(seeds, k_values, n_perm, n_trials,
                       tau=TAU_C, eps=EPS_C, delta=DELTA_C):
    """
    Multi-K protocol (c027): for each seed, average MI gap across k_values.
    This reduces CV from ~14.8% (K=6 only) to ~5.1%.
    """
    gaps = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, tau=tau, eps=eps, delta=delta)
        k_gaps = []
        for K in k_values:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            k_gaps.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
        gaps.append(sum(k_gaps) / len(k_gaps))
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
    return 2.0 * (1.0 - _norm_cdf(abs(mean(diffs) / se)))


def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    return md / (sd + 1e-15)


def run_sweep(param_name, param_values, canonical_val, canon_gaps, seeds, k_values, n_perm, n_trials):
    print(f"\n{'='*W}")
    print(f"  {param_name.upper()} SWEEP (canonical={canonical_val})")
    print(f"{'='*W}")
    print(f"\n  {param_name:>8} {'mean':>10} {'std':>8} {'CV%':>6} {'diff':>9} {'d':>8} {'p':>8}  verdict")
    print(f"  {'-'*8} {'-'*10} {'-'*8} {'-'*6} {'-'*9} {'-'*8} {'-'*8}")

    results = {}
    for val in param_values:
        kwargs = {param_name: val}
        gaps = eval_params_multik(seeds, k_values, n_perm, n_trials, **kwargs)
        m = mean(gaps); s = std_pop(gaps)
        cv = s / abs(m) * 100 if m != 0 else float('inf')
        if val == canonical_val:
            diff_str = "(canonical)"; d_str = "---"; p_str = "---"; verdict = ""
        else:
            diff = m - mean(canon_gaps)
            d = cohens_d_paired(gaps, canon_gaps)
            p = paired_t_p(gaps, canon_gaps)
            diff_str = f"{diff:+.4f}"
            d_str = f"{d:+.3f}"
            p_str = f"{p:.3f}"
            verdict = "BINDING" if p < 0.05 else ("watch" if abs(d) > 0.5 else "")
        results[val] = gaps
        print(f"  {val:>8.3f} {m:>+10.4f} {s:>8.4f} {cv:>6.1f} {diff_str:>9} {d_str:>8} {p_str:>8}  {verdict}")
    return results


if __name__ == '__main__':
    SEEDS = [42, 137, 2024, 999, 7]
    N_PERM = 8
    N_TRIALS = 6

    print("=" * W)
    print("  STAGE 4: STATE UPDATE PARAMETER SENSITIVITY DIAGNOSTIC (MULTI-K)")
    print("  tau, eps, delta — multi-K protocol per c027")
    print("  K=[4,6,8,10] averaging: CV ~5.1% vs K=6-only ~14.8%")
    print("=" * W)
    print(f"\nProtocol: {len(SEEDS)} paired seeds, K={K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Seeds: {SEEDS}")

    # Canonical baseline (multi-K)
    print(f"\nRunning canonical baseline (multi-K)...", flush=True)
    can_gaps = eval_params_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS,
                                  tau=TAU_C, eps=EPS_C, delta=DELTA_C)
    can_mean = mean(can_gaps)
    can_std = std_pop(can_gaps)
    can_cv = can_std / abs(can_mean) * 100
    print(f"  Canonical: mean={can_mean:+.4f}, std={can_std:.4f}, CV={can_cv:.1f}%")
    print(f"  Per-seed: {' '.join(f'{g:+.4f}' for g in can_gaps)}")

    # TAU SWEEP
    tau_values = [0.1, 0.2, 0.3, 0.5, 0.8]
    tau_results = run_sweep('tau', tau_values, TAU_C, can_gaps,
                            SEEDS, K_VALUES, N_PERM, N_TRIALS)

    # EPS SWEEP
    eps_values = [0.0, 0.05, 0.15, 0.3, 0.5]
    eps_results = run_sweep('eps', eps_values, EPS_C, can_gaps,
                            SEEDS, K_VALUES, N_PERM, N_TRIALS)

    # DELTA SWEEP
    delta_values = [0.1, 0.2, 0.35, 0.5, 0.7]
    delta_results = run_sweep('delta', delta_values, DELTA_C, can_gaps,
                              SEEDS, K_VALUES, N_PERM, N_TRIALS)

    # SUMMARY
    print(f"\n{'='*W}")
    print(f"  SUMMARY: BINDING CANDIDATES (MULTI-K, p<0.05 or |d|>0.5)")
    print(f"{'='*W}")

    all_flags = []

    for tau, gaps in tau_results.items():
        if tau == TAU_C: continue
        d = cohens_d_paired(gaps, can_gaps); p = paired_t_p(gaps, can_gaps)
        if p < 0.05 or abs(d) > 0.5:
            all_flags.append(('tau', tau, mean(gaps) - can_mean, d, p))

    for eps, gaps in eps_results.items():
        if eps == EPS_C: continue
        d = cohens_d_paired(gaps, can_gaps); p = paired_t_p(gaps, can_gaps)
        if p < 0.05 or abs(d) > 0.5:
            all_flags.append(('eps', eps, mean(gaps) - can_mean, d, p))

    for delta, gaps in delta_results.items():
        if delta == DELTA_C: continue
        d = cohens_d_paired(gaps, can_gaps); p = paired_t_p(gaps, can_gaps)
        if p < 0.05 or abs(d) > 0.5:
            all_flags.append(('delta', delta, mean(gaps) - can_mean, d, p))

    if all_flags:
        print(f"\n  {'param':<8} {'value':>8} {'diff':>9} {'d':>8} {'p':>8}  status")
        print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")
        for param, val, diff, d, p in sorted(all_flags, key=lambda x: x[4]):
            status = "BINDING (p<0.05)" if p < 0.05 else "borderline (|d|>0.5)"
            print(f"  {param:<8} {val:>8.3f} {diff:>+9.4f} {d:>+8.3f} {p:>8.3f}  {status}")
        print(f"\n  BINDING PARAMETERS FOUND at multi-K resolution.")
        print(f"  These govern STATE EVOLUTION, not alpha adaptation.")
        print(f"  Verified under c027 methodology: K={K_VALUES} averaging.")
    else:
        print(f"\n  No parameters showed |d|>0.5 or p<0.05 at multi-K resolution.")
        print(f"  State update parameters are non-binding.")
        print(f"  The frozen frame floor is confirmed very high for this architecture.")
        print(f"  Stage 4 is vacuous (Amendment 1 applies — requires mechanism test).")

    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")
    binding_params = set(f[0] for f in all_flags if f[4] < 0.05)
    borderline_params = set(f[0] for f in all_flags if f[4] >= 0.05)
    print(f"\n  Canonical CV at multi-K: {can_cv:.1f}%")
    if binding_params:
        print(f"  BINDING (p<0.05): {', '.join(sorted(binding_params))}")
    if borderline_params:
        print(f"  BORDERLINE (|d|>0.5, p>=0.05): {', '.join(sorted(borderline_params))}")
    if not binding_params and not borderline_params:
        print(f"  ALL NON-BINDING: Stage 4 is vacuous.")
    print()
    print("=" * W)
