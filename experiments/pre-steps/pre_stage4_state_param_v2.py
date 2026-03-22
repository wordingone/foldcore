#!/usr/bin/env python3
"""
Stage 4: State Update Parameter Sensitivity — Version 2 (Correct Architecture)

Version 1 (stage4_state_param_diagnostic.py) used wrong core equation:
  - Neighbors within D (kp=(k+1)%D) instead of cell neighbors (kp=(k+1)%NC)
  - Different signal structure (W=72 vs D=12)
  - Different plasticity rule

This version copies harness.py EXACTLY, then adds tau/eps/delta to rule_params.
Uses the actual measure_gap protocol (within=same permutation repeated trials,
between=different permutations).

Protocol: 10 paired seeds, K=[4,6,8,10], n_perm=8, n_trials=6
"""

import math
import random

D = 12
NC = 6


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
    """Copied exactly from harness.py, with tau/eps/delta exposed via rule_params."""

    def __init__(self, seed=42, alive=False, rule_params=None):
        if rule_params is None:
            rule_params = {}

        # Core dynamics parameters — now exposed
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = rule_params.get('eps', 0.15)
        self.tau = rule_params.get('tau', 0.3)
        self.delta = rule_params.get('delta', 0.35)
        self.noise = 0.005
        self.clip = 4.0

        self.seed = seed
        self.alive = alive

        # Plasticity rule parameters (canonical)
        self.eta = rule_params.get('eta', 0.0003)
        self.symmetry_break_mult = rule_params.get('symmetry_break_mult', 0.3)
        self.amplify_mult = rule_params.get('amplify_mult', 0.5)
        self.drift_mult = rule_params.get('drift_mult', 0.1)
        self.threshold = rule_params.get('threshold', 0.01)
        self.alpha_clip_lo = rule_params.get('alpha_clip_lo', 0.3)
        self.alpha_clip_hi = rule_params.get('alpha_clip_hi', 1.8)

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

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
                response.append([abs(phi_sig[i][k] - phi_bare[i][k]) for k in range(D)])

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

                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(self.alpha_clip_lo,
                                           min(self.alpha_clip_hi, self.alpha[i][k]))

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
    perms = [tuple(base)]
    if k > 1:
        perms.append(tuple(reversed(base)))
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        att += 1
        random.shuffle(base)
        t = tuple(base)
        if t not in perms:
            perms.append(t)
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


def measure_gap(org, signals, k, seed, n_perm=8, n_trials=6):
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


def eval_params(seeds, K_list, n_perm, n_trials, rule_params=None):
    if rule_params is None:
        rule_params = {}
    gaps = []
    for seed in seeds:
        seed_gaps = []
        for K in K_list:
            org = Organism(seed=42, alive=True, rule_params=rule_params)
            sig_seed = 42 + K * 200
            sigs = make_signals(K, seed=sig_seed)
            g = measure_gap(org, sigs, K, seed, n_perm=n_perm, n_trials=n_trials)
            seed_gaps.append(g)
        gaps.append(sum(seed_gaps) / len(seed_gaps))
    return gaps


def mean(lst): return sum(lst) / len(lst)


def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_t_p(a, b):
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = mean(diffs)
    if n < 2: return 1.0
    var = sum((d - md)**2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) + 1e-15
    return 2.0 * (1.0 - _norm_cdf(abs(md / se)))


def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md)**2 for d in diffs) / max(len(diffs)-1, 1))
    return md / (sd + 1e-15)


SEEDS = [42, 137, 2024, 999, 7, 314, 1618, 2718, 4242, 8888]
K_LIST = [4, 6, 8, 10]
N_PERM = 8
N_TRIALS = 6

print("=" * 72)
print("  STAGE 4: STATE PARAM SENSITIVITY — V2 (Correct Harness Architecture)")
print("  tau, eps, delta — canonical harness, multi-K protocol")
print("=" * 72)
print(f"\nProtocol: {len(SEEDS)} paired seeds, K={K_LIST}, n_perm={N_PERM}, n_trials={N_TRIALS}")

print("\nRunning canonical baseline...")
canonical_gaps = eval_params(SEEDS, K_LIST, N_PERM, N_TRIALS, rule_params={})
can_mean = mean(canonical_gaps)
can_std = math.sqrt(sum((x - can_mean)**2 for x in canonical_gaps) / (len(canonical_gaps)-1))
can_cv = can_std / abs(can_mean) * 100 if can_mean != 0 else 999
print(f"  Canonical: mean={can_mean:+.4f}, std={can_std:.4f}, CV={can_cv:.1f}%")
print(f"  Per-seed: " + " ".join(f"{v:+.4f}" for v in canonical_gaps))

# ── DELTA SWEEP ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DELTA SWEEP (state mixing rate, canonical=0.35)")
print("=" * 72)
print(f"\n{'delta':>8} {'mean':>10} {'std':>8} {'diff':>10} {'d':>8} {'p':>8}  verdict")
print("  " + "-" * 60)

delta_vals = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9]
for dv in delta_vals:
    if dv == 0.35:
        print(f"  {dv:>6.2f}    {can_mean:+.4f}   {can_std:.4f} (canonical)      ---      ---")
        continue
    gaps = eval_params(SEEDS, K_LIST, N_PERM, N_TRIALS, rule_params={'delta': dv})
    gm = mean(gaps)
    gs = math.sqrt(sum((x - gm)**2 for x in gaps) / (len(gaps)-1))
    diff = gm - can_mean
    d = cohens_d_paired(gaps, canonical_gaps)
    p = paired_t_p(gaps, canonical_gaps)
    above = sum(1 for i in range(len(SEEDS)) if gaps[i] > canonical_gaps[i])
    flag = "FLAG" if p < 0.05 else ("watch" if p < 0.15 else "")
    print(f"  {dv:>6.2f}    {gm:+.4f}   {gs:.4f}   {diff:+.4f}   {d:+.3f}   {p:.3f}  {flag}  [{above}/{len(SEEDS)}]")

# ── TAU SWEEP ────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  TAU SWEEP (attention temperature, canonical=0.3)")
print("=" * 72)
print(f"\n{'tau':>8} {'mean':>10} {'std':>8} {'diff':>10} {'d':>8} {'p':>8}  verdict")
print("  " + "-" * 60)

tau_vals = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
for tv in tau_vals:
    if tv == 0.3:
        print(f"  {tv:>6.2f}    {can_mean:+.4f}   {can_std:.4f} (canonical)      ---      ---")
        continue
    gaps = eval_params(SEEDS, K_LIST, N_PERM, N_TRIALS, rule_params={'tau': tv})
    gm = mean(gaps)
    gs = math.sqrt(sum((x - gm)**2 for x in gaps) / (len(gaps)-1))
    diff = gm - can_mean
    d = cohens_d_paired(gaps, canonical_gaps)
    p = paired_t_p(gaps, canonical_gaps)
    flag = "FLAG" if p < 0.05 else ("watch" if p < 0.15 else "")
    print(f"  {tv:>6.2f}    {gm:+.4f}   {gs:.4f}   {diff:+.4f}   {d:+.3f}   {p:.3f}  {flag}")

# ── EPS SWEEP ────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  EPS SWEEP (attention pull strength, canonical=0.15)")
print("=" * 72)
print(f"\n{'eps':>8} {'mean':>10} {'std':>8} {'diff':>10} {'d':>8} {'p':>8}  verdict")
print("  " + "-" * 60)

eps_vals = [0.0, 0.05, 0.15, 0.3, 0.5, 1.0]
for ev in eps_vals:
    if ev == 0.15:
        print(f"  {ev:>6.3f}    {can_mean:+.4f}   {can_std:.4f} (canonical)      ---      ---")
        continue
    gaps = eval_params(SEEDS, K_LIST, N_PERM, N_TRIALS, rule_params={'eps': ev})
    gm = mean(gaps)
    gs = math.sqrt(sum((x - gm)**2 for x in gaps) / (len(gaps)-1))
    diff = gm - can_mean
    d = cohens_d_paired(gaps, canonical_gaps)
    p = paired_t_p(gaps, canonical_gaps)
    flag = "FLAG" if p < 0.05 else ("watch" if p < 0.15 else "")
    print(f"  {ev:>6.3f}    {gm:+.4f}   {gs:.4f}   {diff:+.4f}   {d:+.3f}   {p:.3f}  {flag}")

print("\n" + "=" * 72)
print("  DONE")
print("=" * 72)
