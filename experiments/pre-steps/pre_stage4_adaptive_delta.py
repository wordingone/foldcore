#!/usr/bin/env python3
"""
Stage 4: Adaptive Delta Phase 2

Question: Can an adaptive delta mechanism discover delta=1.0?

Background:
  - delta=1.0 (pure replacement) is optimal (d=+1.289, p=0.000, +6% vs 0.35)
  - Improvement is monotonic to boundary: calibration, not adaptive opportunity
  - BUT: can a Principle-II-compliant adaptive rule converge to the optimum?
    If yes: adaptive delta is functional even if not needed for calibration.
    If no: confirms delta=1.0 as purely calibration (not adaptive).

Adaptive mechanism (Principle II compliant):
  - Signal: state-output divergence ||p[k] - xs[k]|| per cell
    When new computation differs from old state, the cell is in transition.
    High divergence -> favor replacement (push delta toward 1.0)
    Low divergence -> favor blending (push delta toward 0.0)
  - This signal arises from the same dynamics that compute the state update.
    It cannot be removed without removing the computation.

Four conditions:
  1. fixed_0.35:       Old canonical delta=0.35 (baseline)
  2. fixed_1.0:        New canonical delta=1.0 (calibration result, upper bound)
  3. adaptive_full:    Adaptive delta [0.1, 1.0], starts at 0.35
  4. adaptive_reversed: Adaptive delta [0.1, 1.0], starts at 1.0

Protocol: 10 seeds, K=[4,6,8,10], n_perm=4, n_trials=3
"""

import math
import random
import sys
import time

D = 12
NC = 6
W = 72

EPS_C = 0.15
NOISE_C = 0.005

K_VALUES = [4, 6, 8, 10]
SEEDS = [42, 137, 2024, 999, 7, 10, 11, 12, 13, 14]
N_PERM = 4
N_TRIALS = 3

# Adaptive delta hyperparameters
DELTA_LR = 0.01       # learning rate for delta adaptation
DELTA_LO = 0.1        # lower bound
DELTA_HI = 1.0        # upper bound


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
    """
    Organism with optional adaptive delta.

    adaptive_delta=True: delta adapts based on state-output divergence.
      The signal: for each cell i, compute ||p[i] - xs[i]|| (divergence
      between new computation p and current state xs). High divergence means
      the computation wants to move far from current state — favor replacement
      (high delta). Low divergence means computation and state agree — blending
      is fine either way.

      Update: delta += lr * (mean_divergence - delta)
      This is a leaky integrator toward the current divergence signal.
      The signal is computed from p and xs, both products of the current step.
    """

    def __init__(self, seed=42, alive=False,
                 delta=0.35,
                 adaptive_delta=False,
                 delta_lr=DELTA_LR,
                 clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = EPS_C
        self.delta = delta
        self.noise = NOISE_C
        self.clip = 4.0
        self.eta = 0.0003
        self.tau = 0.3
        self.symmetry_break_mult = 0.3
        self.amplify_mult = 0.5
        self.drift_mult = 0.1
        self.threshold = 0.01
        self.alpha_clip_lo = clip_lo
        self.alpha_clip_hi = clip_hi
        self.alive = alive
        self.adaptive_delta = adaptive_delta
        self.delta_lr = delta_lr

        # Track delta trajectory for diagnostics
        self.delta_history = [delta]

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

        # ── ALPHA PLASTICITY ──────────────────────────────────
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

        # ── ATTENTION ─────────────────────────────────────────
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

        # ── STATE COMPUTATION (p = candidate next state) ──────
        p_all = []
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
            p_all.append(p)

        # ── ADAPTIVE DELTA UPDATE ─────────────────────────────
        # Signal: mean state-output divergence ||p[i] - xs[i]|| across cells.
        # Arises from p_all (computed above) and xs (current state).
        # Cannot be removed without removing state update computation.
        if self.adaptive_delta:
            total_div = 0.0
            for i in range(NC):
                diff = [p_all[i][k] - xs[i][k] for k in range(D)]
                total_div += vnorm(diff) / max(vnorm(xs[i]), 1.0)
            mean_div = total_div / NC

            # Normalize divergence to [0,1] range (tanh squash)
            # mean_div typically in [0, 3] range; tanh maps this to [0, ~1]
            target_delta = math.tanh(mean_div)
            target_delta = max(DELTA_LO, min(DELTA_HI, target_delta))

            # Leaky integrator: delta moves toward target at rate delta_lr
            self.delta = self.delta + self.delta_lr * (target_delta - self.delta)
            self.delta = max(DELTA_LO, min(DELTA_HI, self.delta))
            self.delta_history.append(self.delta)

        # ── STATE UPDATE ──────────────────────────────────────
        new = []
        for i in range(NC):
            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p_all[i][k]
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


def eval_multik(seeds, k_values, n_perm, n_trials, condition_kwargs):
    """
    Evaluate a condition across seeds and K values.
    Returns per-seed gaps (averaged across K) AND the final delta values.
    condition_kwargs: dict of kwargs to pass to Organism constructor.
    """
    gaps = []
    final_deltas = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, **condition_kwargs)
        k_gaps = []
        for K in k_values:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            k_gaps.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
        gaps.append(sum(k_gaps) / len(k_gaps))
        final_deltas.append(org.delta)
    return gaps, final_deltas


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


# ── CONDITIONS ────────────────────────────────────────────────
CONDITIONS = [
    ('fixed_0.35',       dict(delta=0.35, adaptive_delta=False)),
    ('fixed_1.0',        dict(delta=1.0,  adaptive_delta=False)),
    ('adaptive_full',    dict(delta=0.35, adaptive_delta=True,  delta_lr=DELTA_LR)),
    ('adaptive_reversed',dict(delta=1.0,  adaptive_delta=True,  delta_lr=DELTA_LR)),
]


if __name__ == '__main__':
    t_start = time.time()

    print("=" * W)
    print("  STAGE 4: ADAPTIVE DELTA PHASE 2")
    print("  Can adaptive delta discover delta=1.0?")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Delta LR: {DELTA_LR}, bounds: [{DELTA_LO}, {DELTA_HI}]")
    print(f"\nConditions: {[c[0] for c in CONDITIONS]}")
    sys.stdout.flush()

    results = {}
    delta_finals = {}
    for label, kwargs in CONDITIONS:
        print(f"\n  Running {label}...", flush=True)
        gaps, final_d = eval_multik(SEEDS, K_VALUES, N_PERM, N_TRIALS, kwargs)
        results[label] = gaps
        delta_finals[label] = final_d
        m = mean(gaps); s = std_pop(gaps)
        if kwargs.get('adaptive_delta', False):
            fd_mean = mean(final_d); fd_std = std_pop(final_d)
            print(f"    mean={m:+.4f}, std={s:.4f}, CV={s/abs(m)*100:.1f}%  "
                  f"final_delta={fd_mean:.4f}+/-{fd_std:.4f}")
        else:
            print(f"    mean={m:+.4f}, std={s:.4f}, CV={s/abs(m)*100:.1f}%")
        sys.stdout.flush()

    canon = results['fixed_0.35']
    fixed_1 = results['fixed_1.0']
    cm = mean(canon)

    print(f"\n{'='*W}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*W}\n")

    print(f"  {'Condition':<20} {'Mean':>8} {'Std':>8} {'d_vs_0.35':>10} {'p_vs_0.35':>10} {'%diff':>7} {'FinalDelta':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*7} {'-'*12}")

    for label, gaps in results.items():
        m = mean(gaps); s = std_pop(gaps)
        fd_str = f"{mean(delta_finals[label]):.4f}" if delta_finals[label][0] != delta_finals[label][-1] or label.startswith('adaptive') else "fixed"
        if label == 'fixed_0.35':
            print(f"  {label:<20} {m:>+8.4f} {s:>8.4f} {'BASE':>10} {'BASE':>10} {'BASE':>7} {'fixed':>12}")
        else:
            d = cohens_d_paired(gaps, canon)
            p = paired_t_p(gaps, canon)
            pct = (m - cm) / abs(cm) * 100
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"  {label:<20} {m:>+8.4f} {s:>8.4f} {d:>+10.3f} {p:>10.4f} {pct:>+7.1f}% {fd_str:>12} {sig}")

    # ── ADAPTIVE vs FIXED COMPARISONS ──────────────────────────
    print(f"\n  ADAPTIVE vs FIXED_1.0 COMPARISONS")
    print(f"  {'Condition':<20} {'d_vs_fixed1.0':>14} {'p_vs_fixed1.0':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")
    for label in ['adaptive_full', 'adaptive_reversed']:
        gaps = results[label]
        d = cohens_d_paired(gaps, fixed_1)
        p = paired_t_p(gaps, fixed_1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {label:<20} {d:>+14.3f} {p:>14.4f} {sig}")

    # ── DELTA CONVERGENCE ANALYSIS ──────────────────────────────
    print(f"\n  DELTA CONVERGENCE (adaptive conditions)")
    for label in ['adaptive_full', 'adaptive_reversed']:
        kwargs_dict = dict(CONDITIONS)[label] if False else None
        # get from CONDITIONS list
        for lbl, kw in CONDITIONS:
            if lbl == label:
                init_delta = kw['delta']
                break
        fd = delta_finals[label]
        print(f"  {label}: init={init_delta:.2f} -> final mean={mean(fd):.4f} +/- {std_pop(fd):.4f}")
        print(f"    per-seed finals: {[f'{x:.3f}' for x in fd]}")

    # ── VERDICT ─────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}\n")

    # Key questions:
    # 1. Does fixed_1.0 beat fixed_0.35? (Should be yes — calibration check)
    # 2. Does adaptive_full converge toward 1.0?
    # 3. Does adaptive_full match fixed_1.0? (Not significantly worse)
    # 4. Does adaptive_reversed stay near 1.0?

    fixed1_beats_canon = mean(fixed_1) > mean(canon)
    p_fixed1_vs_canon = paired_t_p(fixed_1, canon)

    adaptive_full_fd = mean(delta_finals['adaptive_full'])
    adaptive_rev_fd = mean(delta_finals['adaptive_reversed'])

    adaptive_full_gaps = results['adaptive_full']
    adaptive_rev_gaps = results['adaptive_reversed']

    # Adaptive vs fixed_1.0 — does adaptive match the calibrated optimum?
    p_adap_full_vs_fixed1 = paired_t_p(adaptive_full_gaps, fixed_1)
    p_adap_rev_vs_fixed1 = paired_t_p(adaptive_rev_gaps, fixed_1)

    converges_up = adaptive_full_fd > 0.7      # started at 0.35, converges toward 1.0?
    stays_high = adaptive_rev_fd > 0.7         # started at 1.0, stays there?
    matches_fixed1_full = p_adap_full_vs_fixed1 > 0.05
    matches_fixed1_rev = p_adap_rev_vs_fixed1 > 0.05

    print(f"  [1] fixed_1.0 beats fixed_0.35: {'YES' if fixed1_beats_canon else 'NO'} "
          f"(p={p_fixed1_vs_canon:.4f}, calibration confirmed)")
    print(f"  [2] adaptive_full converges toward 1.0: {'YES' if converges_up else 'NO'} "
          f"(final mean={adaptive_full_fd:.4f}, started at 0.35)")
    print(f"  [3] adaptive_reversed stays near 1.0: {'YES' if stays_high else 'NO'} "
          f"(final mean={adaptive_rev_fd:.4f}, started at 1.0)")
    print(f"  [4] adaptive_full matches fixed_1.0: {'YES' if matches_fixed1_full else 'NO'} "
          f"(p={p_adap_full_vs_fixed1:.4f}, no significant difference)")
    print(f"  [5] adaptive_reversed matches fixed_1.0: {'YES' if matches_fixed1_rev else 'NO'} "
          f"(p={p_adap_rev_vs_fixed1:.4f})")

    if fixed1_beats_canon and converges_up and matches_fixed1_full:
        print(f"""
  ADAPTIVE DELTA IS FUNCTIONAL:
  The mechanism discovers the optimum (delta=1.0) from below.
  Performance matches the hand-tuned calibration.
  This does NOT change the calibration decision (delta=1.0 is still optimal),
  but demonstrates a Principle-II-compliant adaptive mechanism for delta.
  The frozen frame does not shrink — delta=1.0 is still the answer
  regardless of initialization — but the mechanism is non-vacuous.
""")
    elif fixed1_beats_canon and not converges_up:
        print(f"""
  ADAPTIVE DELTA DOES NOT DISCOVER THE OPTIMUM:
  fixed_1.0 beats fixed_0.35 (calibration confirmed),
  but adaptive_full does not converge to 1.0 (final={adaptive_full_fd:.4f}).
  The divergence signal is not a reliable proxy for the optimal delta.
  Conclusion: delta=1.0 is a calibration choice, not an adaptive target.
  Amendment 1 (vacuous passage) applies if mechanism is non-degenerate.
""")
    elif not fixed1_beats_canon:
        print(f"""
  CALIBRATION FAILED TO REPLICATE:
  fixed_1.0 does not beat fixed_0.35 at this protocol.
  Check if Entry 052 results used different eval parameters.
  This is unexpected and requires investigation.
""")
    else:
        print(f"""
  MIXED RESULTS: See table above for details.
""")

    print(f"  Runtime: {time.time() - t_start:.1f}s")
    print(f"  {'='*W}")
