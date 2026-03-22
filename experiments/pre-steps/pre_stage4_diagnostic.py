#!/usr/bin/env python3
"""
Stage 4 Phase 1 Kill-Criterion: Threshold Sensitivity Diagnostic

Tests whether `threshold` (the abs(dev) cutoff for symmetry-breaking branch
in the plasticity rule) is a binding constraint or a vacuous one like eta.

Canonical value: threshold = 0.01 (hardcoded in the_living_seed.py line 176)

Two measurements:
  1. Activation rate: what fraction of (cell, timestep) pairs trigger the
     symmetry-breaking branch? Low rate -> threshold rarely fires -> non-binding.
  2. MI gap sweep: does MI gap vary meaningfully across threshold values?
     Flat -> non-binding (vacuous like eta). Variable -> binding (Stage 4 target).

Kill criterion:
  - If MI gap flat across [0.001, 0.1] range -> threshold is vacuous, skip it.
  - If MI gap varies > ~10% across range -> threshold is binding, pursue.

Protocol: n_perm=8, n_trials=6 per c015. 3 seeds minimum per condition.
"""

import math
import random
import time


# ── System constants (match the_living_seed.py) ──────────────────────────────
D  = 12
NC = 6
W  = 72


def vcosine(a, b):
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a) + 1e-15)
    nb = math.sqrt(sum(bi * bi for bi in b) + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


# ── Organism with configurable threshold ─────────────────────────────────────

class Organism:
    """Living Seed organism with configurable plasticity threshold.

    threshold: abs(dev) cutoff for symmetry-breaking branch.
               Canonical value = 0.01 (the_living_seed.py line 176).
    track_activations: if True, record per-step branch activation counts.
    """

    def __init__(self, seed=42, alive=False, eta=0.0003, threshold=0.01,
                 track_activations=False):
        self.beta      = 0.5
        self.gamma     = 0.9
        self.eps       = 0.15
        self.tau       = 0.3
        self.delta     = 0.35
        self.noise     = 0.005
        self.clip      = 4.0
        self.seed      = seed
        self.alive     = alive
        self.eta       = eta
        self.threshold = threshold
        self.track_activations = track_activations

        # Activation tracking
        self.n_symmetry_break = 0   # abs(dev) < threshold
        self.n_amplify        = 0   # resp_z > 0 (above-average response)
        self.n_drift          = 0   # below-average response
        self.n_total_steps    = 0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

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
            overall_std  = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    if abs(dev) < self.threshold:
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                        if self.track_activations:
                            self.n_symmetry_break += 1
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(resp_z) * direction * 0.5
                        if self.track_activations:
                            self.n_amplify += 1
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)
                        if self.track_activations:
                            self.n_drift += 1

                    if self.track_activations:
                        self.n_total_steps += 1

                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))

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
        return new

    def centroid(self, xs):
        return [sum(xs[i][k] for i in range(NC)) / NC for k in range(D)]

    def activation_fractions(self):
        total = max(self.n_total_steps, 1)
        return {
            'symmetry_break': self.n_symmetry_break / total,
            'amplify':        self.n_amplify / total,
            'drift':          self.n_drift / total,
            'total_steps':    self.n_total_steps,
        }


# ── Protocol helpers (match harness.py / the_living_seed.py) ─────────────────

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
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        random.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
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
    """c015: n_perm=8, n_trials=6 standard protocol."""
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


# ── Diagnostic routines ───────────────────────────────────────────────────────

def measure_activation_rates(seeds, threshold, ks=(6, 8), sig_seed_base=42):
    """Measure branch activation fractions across seeds and K values."""
    all_fracs = {'symmetry_break': [], 'amplify': [], 'drift': []}

    for seed in seeds:
        for k in ks:
            sigs = make_signals(k, seed=sig_seed_base + k * 200)
            org = Organism(seed=seed, alive=True, eta=0.0003,
                           threshold=threshold, track_activations=True)
            # Run a full sequence to accumulate activation statistics
            perms = gen_perms(k, 4, seed=seed * 10 + k)
            for perm in perms[:2]:  # 2 permutations is enough for rate stats
                for trial in range(3):
                    run_sequence(org, perm, sigs, seed, trial)
            fracs = org.activation_fractions()
            for key in all_fracs:
                all_fracs[key].append(fracs[key])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    return {k: avg(v) for k, v in all_fracs.items()}


def measure_gap_for_threshold(threshold, seeds, ks=(6, 8), sig_seed_base=42):
    """Measure mean MI gap for a given threshold across seeds and K values."""
    gaps = []
    for seed in seeds:
        for k in ks:
            sigs = make_signals(k, seed=sig_seed_base + k * 200)
            org = Organism(seed=seed, alive=True, eta=0.0003, threshold=threshold)
            g = measure_gap(org, sigs, k, seed)
            gaps.append(g)
    return sum(gaps) / len(gaps), gaps


# ── Main diagnostic ───────────────────────────────────────────────────────────

def run():
    t_start = time.time()

    print("=" * 72)
    print("  STAGE 4 PHASE 1: THRESHOLD SENSITIVITY DIAGNOSTIC")
    print("  Kill criterion: is threshold a binding constraint or vacuous?")
    print("  Canonical threshold = 0.01 (the_living_seed.py line 176)")
    print("=" * 72)

    SEEDS     = [42, 137, 2024]        # 3-seed Phase 1 (quick)
    KS        = (6, 8)                 # Per task spec
    CANONICAL = 0.01
    THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

    # ── PART 1: Activation Rate Analysis ─────────────────────────────────────
    print("\n" + "-" * 72)
    print("  PART 1: Branch Activation Rate at Canonical Threshold")
    print("  How often does abs(dev) < 0.01 actually trigger?")
    print("-" * 72)
    print(f"\n  Measuring across {len(SEEDS)} seeds x K={list(KS)} ...")

    fracs = measure_activation_rates(SEEDS, threshold=CANONICAL, ks=KS)

    print(f"\n  Branch activation fractions (canonical threshold={CANONICAL}):")
    print(f"    symmetry_break (|dev| < threshold): {fracs['symmetry_break']:.3f} "
          f"({fracs['symmetry_break']*100:.1f}%)")
    print(f"    amplify        (resp_z > 0):        {fracs['amplify']:.3f} "
          f"({fracs['amplify']*100:.1f}%)")
    print(f"    drift          (resp_z <= 0):       {fracs['drift']:.3f} "
          f"({fracs['drift']*100:.1f}%)")

    sb_pct = fracs['symmetry_break'] * 100
    if sb_pct < 5.0:
        activation_verdict = "RARE (<5%) — symmetry-breaking branch rarely fires. Threshold likely non-binding."
    elif sb_pct > 20.0:
        activation_verdict = "FREQUENT (>20%) — symmetry-breaking branch fires often. Threshold likely binding."
    else:
        activation_verdict = f"MODERATE ({sb_pct:.1f}%) — ambiguous. Check MI gap sweep."

    print(f"\n  Activation verdict: {activation_verdict}")

    # ── PART 2: MI Gap Sweep ──────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  PART 2: MI Gap Sweep Across Threshold Values")
    print("  Protocol: n_perm=8, n_trials=6 (c015). 3 seeds x K=[6,8].")
    print(f"  Thresholds: {THRESHOLDS}")
    print("-" * 72)

    results = {}
    canonical_gap = None

    for thresh in THRESHOLDS:
        tag = "[CANONICAL]" if thresh == CANONICAL else ""
        print(f"\n  threshold={thresh} {tag}", flush=True)
        mean_gap, per_gap = measure_gap_for_threshold(thresh, SEEDS, ks=KS)
        results[thresh] = {'mean': mean_gap, 'per': per_gap}
        if thresh == CANONICAL:
            canonical_gap = mean_gap

        gap_strs = [f"{g:+.4f}" for g in per_gap]
        print(f"    mean={mean_gap:+.4f}  per=[{', '.join(gap_strs)}]", flush=True)

    # ── PART 3: Analysis ──────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  PART 3: Threshold Sensitivity Analysis")
    print("-" * 72)

    all_means = [results[t]['mean'] for t in THRESHOLDS]
    min_gap   = min(all_means)
    max_gap   = max(all_means)
    range_gap = max_gap - min_gap

    print(f"\n  MI gap across threshold sweep:")
    print(f"  {'Threshold':>12}  {'Mean Gap':>10}  {'vs Canonical':>12}  {'Notes'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*12}  {'-'*30}")
    for thresh in THRESHOLDS:
        mean = results[thresh]['mean']
        delta = mean - canonical_gap
        tag = "<-- canonical" if thresh == CANONICAL else ""
        print(f"  {thresh:>12.4f}  {mean:>+10.4f}  {delta:>+12.4f}  {tag}")

    print(f"\n  Summary statistics:")
    print(f"    Canonical gap:    {canonical_gap:+.4f}")
    print(f"    Min gap (sweep):  {min_gap:+.4f}  at threshold={THRESHOLDS[all_means.index(min_gap)]}")
    print(f"    Max gap (sweep):  {max_gap:+.4f}  at threshold={THRESHOLDS[all_means.index(max_gap)]}")
    print(f"    Range:            {range_gap:+.4f}  ({range_gap/abs(canonical_gap)*100:.1f}% of canonical)")

    # ── PART 4: Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    pct_range = range_gap / max(abs(canonical_gap), 1e-6) * 100

    # Kill criterion thresholds
    FLAT_THRESHOLD   = 5.0   # <5% range -> declare non-binding
    STRONG_THRESHOLD = 15.0  # >15% range -> declare binding

    print(f"\n  Activation rate: {fracs['symmetry_break']*100:.1f}% of steps trigger "
          f"symmetry-break branch")
    print(f"  MI gap range across sweep: {pct_range:.1f}% of canonical")

    if pct_range < FLAT_THRESHOLD:
        verdict = "NON-BINDING (VACUOUS)"
        detail  = (f"MI gap varies only {pct_range:.1f}% across 100x threshold range. "
                   f"Threshold is not a binding constraint. "
                   f"Mark as vacuous (like eta) and move to next candidate.")
        recommend = "Skip threshold. Investigate beta/gamma adaptation via new mechanism."
    elif pct_range > STRONG_THRESHOLD:
        verdict = "BINDING — GENUINE STAGE 4 TARGET"
        detail  = (f"MI gap varies {pct_range:.1f}% across threshold range. "
                   f"Threshold meaningfully controls system behavior. "
                   f"Proceed to Phase 2: self-generated threshold adaptation.")
        recommend = "Design threshold adaptation mechanism driven by self-generated signal."
    else:
        verdict = "AMBIGUOUS — WEAK SIGNAL"
        detail  = (f"MI gap varies {pct_range:.1f}% (between flat and binding thresholds). "
                   f"Effect exists but may be within noise floor (c014: CV=29%). "
                   f"10-seed validation required before committing to Phase 2.")
        recommend = "Run 10-seed validation before deciding. Effect may be noise."

    print(f"\n  THRESHOLD STATUS: {verdict}")
    print(f"  {detail}")
    print(f"\n  RECOMMENDATION: {recommend}")

    print(f"\n  Runtime: {time.time() - t_start:.0f}s")
    print("=" * 72)


if __name__ == '__main__':
    run()
