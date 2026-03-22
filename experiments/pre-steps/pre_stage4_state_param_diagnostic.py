#!/usr/bin/env python3
"""
Stage 4: State Update Parameter Sensitivity Diagnostic

Tests tau, eps, delta — the state update mechanism parameters.
These have NEVER been varied. If binding, they are Stage 4 targets
outside the plasticity rule entirely.

Sweep (hold others at canonical):
  tau:   [0.1, 0.2, 0.3, 0.5, 0.8]   canonical=0.3  (attention temperature)
  eps:   [0.0, 0.05, 0.15, 0.3, 0.5]  canonical=0.15 (attention pull strength)
  delta: [0.1, 0.2, 0.35, 0.5, 0.7]   canonical=0.35 (state mixing rate)

Protocol: 5 paired seeds [42, 137, 2024, 999, 7], K=6, n_perm=8, n_trials=6
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


def eval_params(seeds, K, n_perm, n_trials, tau=TAU_C, eps=EPS_C, delta=DELTA_C):
    gaps = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, tau=tau, eps=eps, delta=delta)
        sigs = make_signals(K, seed=seed + 500)
        perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
        gaps.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
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


if __name__ == '__main__':
    SEEDS = [42, 137, 2024, 999, 7]
    K = 6
    N_PERM = 8
    N_TRIALS = 6

    print("=" * W)
    print("  STAGE 4: STATE UPDATE PARAMETER SENSITIVITY DIAGNOSTIC")
    print("  tau, eps, delta — never previously varied")
    print("=" * W)
    print(f"\nProtocol: {len(SEEDS)} paired seeds, K={K}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Seeds: {SEEDS}")

    # Canonical baseline (run once, reuse for all comparisons)
    print(f"\nRunning canonical baseline...", flush=True)
    can_gaps = eval_params(SEEDS, K, N_PERM, N_TRIALS,
                           tau=TAU_C, eps=EPS_C, delta=DELTA_C)
    can_mean = mean(can_gaps)
    can_std = std_pop(can_gaps)
    can_cv = can_std / abs(can_mean) * 100
    print(f"  Canonical: mean={can_mean:+.4f}, std={can_std:.4f}, CV={can_cv:.1f}%")
    print(f"  Per-seed: {' '.join(f'{g:+.4f}' for g in can_gaps)}")

    # ── TAU SWEEP ──────────────────────────────────────────────────────────
    tau_values = [0.1, 0.2, 0.3, 0.5, 0.8]
    print(f"\n{'='*W}")
    print(f"  TAU SWEEP (attention temperature, canonical={TAU_C})")
    print(f"{'='*W}")
    print(f"\n  {'tau':>6} {'mean':>10} {'std':>8} {'diff':>9} {'d':>8} {'p':>8}  verdict")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")

    tau_results = {}
    for tau in tau_values:
        gaps = eval_params(SEEDS, K, N_PERM, N_TRIALS, tau=tau, eps=EPS_C, delta=DELTA_C)
        m = mean(gaps); s = std_pop(gaps)
        if tau == TAU_C:
            diff_str = "(canonical)"; d_str = "---"; p_str = "---"; verdict = ""
        else:
            diff = m - can_mean
            d = cohens_d_paired(gaps, can_gaps)
            p = paired_t_p(gaps, can_gaps)
            diff_str = f"{diff:+.4f}"
            d_str = f"{d:+.3f}"
            p_str = f"{p:.3f}"
            verdict = "FLAG" if p < 0.05 else ("watch" if abs(d) > 0.5 else "")
        tau_results[tau] = gaps
        print(f"  {tau:>6.2f} {m:>+10.4f} {s:>8.4f} {diff_str:>9} {d_str:>8} {p_str:>8}  {verdict}")

    # ── EPS SWEEP ──────────────────────────────────────────────────────────
    eps_values = [0.0, 0.05, 0.15, 0.3, 0.5]
    print(f"\n{'='*W}")
    print(f"  EPS SWEEP (attention pull strength, canonical={EPS_C})")
    print(f"{'='*W}")
    print(f"\n  {'eps':>6} {'mean':>10} {'std':>8} {'diff':>9} {'d':>8} {'p':>8}  verdict")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")

    eps_results = {}
    for eps in eps_values:
        gaps = eval_params(SEEDS, K, N_PERM, N_TRIALS, tau=TAU_C, eps=eps, delta=DELTA_C)
        m = mean(gaps); s = std_pop(gaps)
        if eps == EPS_C:
            diff_str = "(canonical)"; d_str = "---"; p_str = "---"; verdict = ""
        else:
            diff = m - can_mean
            d = cohens_d_paired(gaps, can_gaps)
            p = paired_t_p(gaps, can_gaps)
            diff_str = f"{diff:+.4f}"
            d_str = f"{d:+.3f}"
            p_str = f"{p:.3f}"
            verdict = "FLAG" if p < 0.05 else ("watch" if abs(d) > 0.5 else "")
        eps_results[eps] = gaps
        print(f"  {eps:>6.3f} {m:>+10.4f} {s:>8.4f} {diff_str:>9} {d_str:>8} {p_str:>8}  {verdict}")

    # ── DELTA SWEEP ────────────────────────────────────────────────────────
    delta_values = [0.1, 0.2, 0.35, 0.5, 0.7]
    print(f"\n{'='*W}")
    print(f"  DELTA SWEEP (state mixing rate, canonical={DELTA_C})")
    print(f"{'='*W}")
    print(f"\n  {'delta':>6} {'mean':>10} {'std':>8} {'diff':>9} {'d':>8} {'p':>8}  verdict")
    print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*9} {'-'*8} {'-'*8}")

    delta_results = {}
    for delta in delta_values:
        gaps = eval_params(SEEDS, K, N_PERM, N_TRIALS, tau=TAU_C, eps=EPS_C, delta=delta)
        m = mean(gaps); s = std_pop(gaps)
        if delta == DELTA_C:
            diff_str = "(canonical)"; d_str = "---"; p_str = "---"; verdict = ""
        else:
            diff = m - can_mean
            d = cohens_d_paired(gaps, can_gaps)
            p = paired_t_p(gaps, can_gaps)
            diff_str = f"{diff:+.4f}"
            d_str = f"{d:+.3f}"
            p_str = f"{p:.3f}"
            verdict = "FLAG" if p < 0.05 else ("watch" if abs(d) > 0.5 else "")
        delta_results[delta] = gaps
        print(f"  {delta:>6.3f} {m:>+10.4f} {s:>8.4f} {diff_str:>9} {d_str:>8} {p_str:>8}  {verdict}")

    # ── SUMMARY ───────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  SUMMARY: BINDING CANDIDATES")
    print(f"{'='*W}")

    all_flags = []
    # Re-check all with p<0.05 or |d|>0.5
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
        for param, val, diff, d, p in sorted(all_flags, key=lambda x: x[4]):
            status = "SIGNIFICANT" if p < 0.05 else "borderline"
            print(f"  {param:<8} {val:>8.3f} {diff:>+9.4f} {d:>+8.3f} {p:>8.3f}  {status}")
        print(f"\n  BINDING PARAMETERS FOUND — viable Stage 4 targets outside plasticity rule.")
        print(f"  These govern STATE EVOLUTION, not alpha adaptation.")
        print(f"  Proceeding with these as adaptive targets would genuinely change what the")
        print(f"  system computes — not just scaling, but the attentional geometry.")
    else:
        print(f"\n  No parameters showed |d|>0.5 or p<0.05.")
        print(f"  State update parameters are also non-binding.")
        print(f"  The frozen frame floor appears very high for this architecture.")

    print(f"\n{'='*W}")
    print(f"  ADVERSARIAL NOTE")
    print(f"{'='*W}")
    print(f"""
  This diagnostic uses K=6 only and 5 seeds. The clip bounds diagnostic
  showed that single-K (K=6) can miss effects visible at K=[4,6,8,10].
  CV at 5 seeds K=6 ~ 14-15%. Detects |d|~1.0+ reliably; misses d~0.5.

  If any parameter shows |d| > 0.3 with consistent direction (4+/5 seeds),
  it warrants a 10-seed K=[4,6,8,10] follow-up before concluding non-binding.

  Tau=0.1 (very sharp attention) and eps=0.0 (no attention pull) are the
  most structurally disruptive values — watch these specifically.
  Delta extremes (0.1: slow mixing, 0.7: fast mixing) change the timescale
  of state evolution — may show qualitatively different behavior.
""")
    print("=" * W)
