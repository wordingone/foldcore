#!/usr/bin/env python3
"""
Stage 4: Tau Effect Decomposition by K

The original finding (d=+1.202, p=0.007) used multi-K averaging.
The diagnostic (stage4_tau_metasignal.py) found ZERO difference at 200 steps.

Hypothesis: The tau effect lives in higher-K runs (K=8, K=10 = 400-500 steps),
not in short runs. This script decomposes the original finding by K value.

Also measures attention entropy — the hypothesized adaptive signal.

Protocol: Exact replication of stage4_state_param_multik_fast.py
- 5 seeds [42, 137, 2024, 999, 7]
- Single organism per seed reused across K values (matching original)
- n_perm=4, n_trials=3
"""

import math
import random
import sys

D = 12
NC = 6
W = 72

TAU_C = 0.3
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
        self._last_weights = None  # store attention weights

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
        self._last_weights = weights

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


def attention_entropy(weights):
    """Mean entropy of attention distributions across cells."""
    entropies = []
    for i in range(NC):
        h = 0.0
        for j in range(NC):
            if i == j: continue
            w = weights[i][j]
            if w > 1e-15:
                h -= w * math.log(w)
        entropies.append(h)
    return sum(entropies) / len(entropies)


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
    print("  TAU EFFECT DECOMPOSITION BY K VALUE")
    print("  Original finding: tau=0.2 d=+1.202, p=0.007 (multi-K)")
    print("  Question: Which K values drive the effect?")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"K values: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    sys.stdout.flush()

    # ── Phase 1: Replicate original finding exactly ──
    print(f"\n{'='*W}")
    print(f"  PHASE 1: EXACT REPLICATION (single org per seed, reused across K)")
    print(f"{'='*W}\n")
    sys.stdout.flush()

    canon_gaps = []  # tau=0.3
    test_gaps = []   # tau=0.2

    for seed in SEEDS:
        # Canonical: tau=0.3
        org_c = Organism(seed=seed, alive=True, tau=0.3)
        k_gaps_c = []
        for K in K_VALUES:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
            k_gaps_c.append(measure_gap(org_c, perms, sigs, seed, n_trials=N_TRIALS))
        canon_gaps.append(sum(k_gaps_c) / len(k_gaps_c))

        # Test: tau=0.2
        org_t = Organism(seed=seed, alive=True, tau=0.2)
        k_gaps_t = []
        for K in K_VALUES:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
            k_gaps_t.append(measure_gap(org_t, perms, sigs, seed, n_trials=N_TRIALS))
        test_gaps.append(sum(k_gaps_t) / len(k_gaps_t))

        print(f"  Seed {seed}: canon={canon_gaps[-1]:+.4f}, test={test_gaps[-1]:+.4f}, "
              f"diff={test_gaps[-1]-canon_gaps[-1]:+.4f}", flush=True)

    d_overall = cohens_d_paired(test_gaps, canon_gaps)
    p_overall = paired_t_p(test_gaps, canon_gaps)
    print(f"\n  REPLICATION: d={d_overall:+.3f}, p={p_overall:.4f}")
    print(f"  Mean canon={mean(canon_gaps):+.4f}, test={mean(test_gaps):+.4f}")
    sys.stdout.flush()

    # ── Phase 2: Decompose by K ──
    print(f"\n{'='*W}")
    print(f"  PHASE 2: PER-K DECOMPOSITION")
    print(f"{'='*W}\n")
    sys.stdout.flush()

    print(f"  {'K':>4} | {'tau=0.2':>10} {'tau=0.3':>10} | {'diff':>10} {'d':>8} {'p':>8}  note")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*8}-{'-'*8}")

    for ki, K in enumerate(K_VALUES):
        per_k_canon = []
        per_k_test = []
        for seed in SEEDS:
            # Must re-run per K independently to isolate K's contribution
            org_c = Organism(seed=seed, alive=True, tau=0.3)
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
            per_k_canon.append(measure_gap(org_c, perms, sigs, seed, n_trials=N_TRIALS))

            org_t = Organism(seed=seed, alive=True, tau=0.2)
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
            per_k_test.append(measure_gap(org_t, perms, sigs, seed, n_trials=N_TRIALS))

        m_c = mean(per_k_canon); m_t = mean(per_k_test)
        d_k = cohens_d_paired(per_k_test, per_k_canon)
        p_k = paired_t_p(per_k_test, per_k_canon)
        steps = K * 50
        note = f"({steps} steps)"
        if p_k < 0.05: note += " *** SIGNIFICANT"
        elif abs(d_k) > 0.5: note += " * borderline"
        print(f"  {K:>4} | {m_t:>+10.4f} {m_c:>+10.4f} | {m_t-m_c:>+10.4f} {d_k:>+8.3f} {p_k:>8.3f}  {note}")
        sys.stdout.flush()

    # ── Phase 3: Attention entropy at tau=0.2 vs 0.3 ──
    print(f"\n{'='*W}")
    print(f"  PHASE 3: ATTENTION ENTROPY (hypothesized adaptive signal)")
    print(f"{'='*W}\n")
    sys.stdout.flush()

    # Run a single K=6 trajectory and measure attention entropy periodically
    entropy_02 = {s: [] for s in range(0, 301, 50)}
    entropy_03 = {s: [] for s in range(0, 301, 50)}

    for seed in SEEDS:
        for tau, store in [(0.2, entropy_02), (0.3, entropy_03)]:
            org = Organism(seed=seed, alive=True, tau=tau)
            sigs = make_signals(6, seed=seed + 500)
            sig = sigs[0]
            rng = random.Random(seed * 100)
            xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

            for s in range(301):
                if s in store:
                    # Force a step to get weights, then record entropy
                    if s > 0:
                        xs = org.step(xs, signal=sig)
                    else:
                        # At step 0, run one step to get initial weights
                        xs_temp = org.step(xs, signal=sig)
                    if org._last_weights:
                        store[s].append(attention_entropy(org._last_weights))
                    else:
                        store[s].append(float('nan'))
                elif s > 0:
                    xs = org.step(xs, signal=sig)

    print(f"  {'step':>6} | {'H(att) tau=0.2':>14} {'H(att) tau=0.3':>14} | {'diff':>10}")
    print(f"  {'-'*6}-+-{'-'*14}-{'-'*14}-+-{'-'*10}")
    for s in sorted(entropy_02.keys()):
        h02 = mean(entropy_02[s])
        h03 = mean(entropy_03[s])
        diff = h02 - h03
        print(f"  {s:>6} | {h02:>14.4f} {h03:>14.4f} | {diff:>+10.4f}")
    sys.stdout.flush()

    # Max entropy for softmax over NC-1=5 neighbors
    max_h = -5 * (1/5) * math.log(1/5)
    print(f"\n  Max possible entropy (uniform over 5 neighbors): {max_h:.4f}")
    print(f"  Interpretation: Lower entropy = sharper attention")

    print(f"\n{'='*W}")
    print(f"  VERDICT")
    print(f"{'='*W}")
    print(f"\n  Overall replication: d={d_overall:+.3f}, p={p_overall:.4f}")
    if p_overall < 0.05:
        print(f"  Original finding REPLICATES.")
    elif abs(d_overall) > 0.5:
        print(f"  Borderline replication — effect present but weakened.")
    else:
        print(f"  FAILED TO REPLICATE. d=+1.202 may have been a false positive.")
    print()
    print("=" * W)
