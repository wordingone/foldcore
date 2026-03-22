#!/usr/bin/env python3
"""
Stage 4: Tau Meta-Signal Diagnostic

WHERE does tau matter mechanistically? Compares tau=0.2 vs tau=0.3 across
three diagnostic lenses:

1. Alpha distribution trajectory — how alpha[i][k] values evolve over time
2. Per-cell MI trajectory — how MI gap builds up over 50-step blocks
3. Cell-cell state correlation — how differentiated cells become (pairwise cosine sim)

Paired-seed block: [42, 137, 2024], Multi-K: [4,6,8,10], n_perm=4, n_trials=3
Self-contained — copies Organism class from stage4_state_param_multik_fast.py.
"""

import math
import random
import sys

D = 12
NC = 6
W = 72

TAU_A = 0.2
TAU_B = 0.3
EPS_C = 0.15
DELTA_C = 0.35
NOISE_C = 0.005

K_VALUES = [4, 6, 8, 10]
SEEDS = [42, 137, 2024]
N_PERM = 4
N_TRIALS = 3

SNAPSHOT_STEPS = [0, 50, 100, 150, 200]


# ---------------------------------------------------------------------------
# Organism class (verbatim from stage4_state_param_multik_fast.py)
# ---------------------------------------------------------------------------

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
                 tau=TAU_B, eps=EPS_C, delta=DELTA_C,
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


# ---------------------------------------------------------------------------
# Signal / permutation helpers (verbatim)
# ---------------------------------------------------------------------------

def make_signals(K, seed=42):
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(D)] for _ in range(K)]


def gen_perms(K, n_perm=4, seed=42):
    random.seed(seed)
    idxs = list(range(K)); perms = []
    for _ in range(n_perm):
        p = idxs[:]; random.shuffle(p); perms.append(p)
    return perms


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def mean(v):
    return sum(v) / len(v) if v else 0.0

def std_pop(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / len(v)) if v else 0.0

def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def paired_t_p(a, b):
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = mean(diffs)
    if n < 2:
        return 1.0
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n) + 1e-15
    return 2.0 * (1.0 - _norm_cdf(abs(md / se)))

def cohens_d_paired(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    return md / (sd + 1e-15)


# ---------------------------------------------------------------------------
# DIAGNOSTIC 1: Alpha distribution trajectory
#
# Run step-by-step with K=6 single signal. Snapshot alpha stats at steps
# [0, 50, 100, 150, 200].
# ---------------------------------------------------------------------------

def alpha_snapshot(org):
    """Return (mean, std, range) of all alpha[i][k] values."""
    flat = [org.alpha[i][k] for i in range(NC) for k in range(D)]
    m = mean(flat)
    s = std_pop(flat)
    r = max(flat) - min(flat)
    return m, s, r


def run_alpha_trajectory(seed, tau):
    """Run organism step-by-step, snapshot alpha at SNAPSHOT_STEPS."""
    org = Organism(seed=seed, alive=True, tau=tau)
    K = 6
    sigs = make_signals(K, seed=seed + 500)
    # Use first signal for the full trajectory
    sig = sigs[0]

    rng = random.Random(seed * 100)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    snapshots = {}
    step_count = 0
    for target in SNAPSHOT_STEPS:
        while step_count < target:
            xs = org.step(xs, signal=sig)
            step_count += 1
        snapshots[target] = alpha_snapshot(org)

    return snapshots


def diagnostic_alpha_trajectory():
    print(f"\n{'='*W}")
    print(f"  DIAGNOSTIC 1: ALPHA DISTRIBUTION TRAJECTORY")
    print(f"  K=6, single signal, steps={SNAPSHOT_STEPS}")
    print(f"{'='*W}")
    sys.stdout.flush()

    # Collect per-seed snapshots for each tau
    # For each (step, metric), we get a list over seeds
    metrics = ['mean', 'std', 'range']
    results = {tau: {step: {m: [] for m in metrics} for step in SNAPSHOT_STEPS}
               for tau in [TAU_A, TAU_B]}

    for seed in SEEDS:
        for tau in [TAU_A, TAU_B]:
            snaps = run_alpha_trajectory(seed, tau)
            for step in SNAPSHOT_STEPS:
                m, s, r = snaps[step]
                results[tau][step]['mean'].append(m)
                results[tau][step]['std'].append(s)
                results[tau][step]['range'].append(r)

    print(f"\n  Step-by-step alpha statistics (averaged over {len(SEEDS)} seeds):")
    print(f"\n  {'step':>6} | {'tau=0.2 mean':>12} {'tau=0.3 mean':>12} | {'tau=0.2 std':>11} {'tau=0.3 std':>11} | {'tau=0.2 range':>13} {'tau=0.3 range':>13}")
    print(f"  {'-'*6}-+-{'-'*12}-{'-'*12}-+-{'-'*11}-{'-'*11}-+-{'-'*13}-{'-'*13}")

    for step in SNAPSHOT_STEPS:
        a_mean = mean(results[TAU_A][step]['mean'])
        b_mean = mean(results[TAU_B][step]['mean'])
        a_std = mean(results[TAU_A][step]['std'])
        b_std = mean(results[TAU_B][step]['std'])
        a_range = mean(results[TAU_A][step]['range'])
        b_range = mean(results[TAU_B][step]['range'])
        print(f"  {step:>6} | {a_mean:>12.4f} {b_mean:>12.4f} | {a_std:>11.4f} {b_std:>11.4f} | {a_range:>13.4f} {b_range:>13.4f}")
    sys.stdout.flush()

    # Paired stats at final step (200)
    print(f"\n  Paired statistics at step 200 (tau=0.2 vs tau=0.3):")
    for m_name in metrics:
        vals_a = results[TAU_A][200][m_name]
        vals_b = results[TAU_B][200][m_name]
        d = cohens_d_paired(vals_a, vals_b)
        p = paired_t_p(vals_a, vals_b)
        diff = mean(vals_a) - mean(vals_b)
        print(f"    alpha_{m_name:>5}: diff={diff:+.5f}, d={d:+.3f}, p={p:.3f}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# DIAGNOSTIC 2: Per-step MI trajectory
#
# Run full multi-K protocol but measure MI gap at intermediate points
# (after each 50-step block).
# ---------------------------------------------------------------------------

def run_sequence_with_checkpoints(org, perm, sigs, rng_seed, trial=0):
    """Like run_sequence but returns final states after each 50-step block."""
    rng = random.Random(rng_seed * 100 + trial)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    checkpoints = []
    for idx in perm:
        sig = sigs[idx]
        for _ in range(50):
            xs = org.step(xs, signal=sig if org.alive else None)
        checkpoints.append([r[:] for r in xs])
    return checkpoints


def measure_gap_at_block(org, perm_list, sigs, seed, n_trials, block_idx):
    """Measure MI gap using only states up to block_idx (0-indexed)."""
    finals = []
    for pi, perm in enumerate(perm_list):
        tf = []
        for t in range(n_trials):
            cps = run_sequence_with_checkpoints(org, perm, sigs, seed + pi * 1000, t)
            tf.append(cps[block_idx])
        finals.append(tf)

    within = []; between = []
    for i in range(len(finals)):
        fi = finals[i]
        for a in range(len(fi)):
            for b in range(a + 1, len(fi)):
                for c in range(NC):
                    within.append(vcosine(fi[a][c], fi[b][c]))
        for j in range(i + 1, len(finals)):
            fj = finals[j]
            for ta in fi:
                for tb in fj:
                    for c in range(NC):
                        between.append(vcosine(ta[c], tb[c]))
    return (sum(within) / len(within) - sum(between) / len(between)) if within else 0.0


def diagnostic_mi_trajectory():
    print(f"\n{'='*W}")
    print(f"  DIAGNOSTIC 2: MI GAP TRAJECTORY OVER 50-STEP BLOCKS")
    print(f"  Multi-K={K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"{'='*W}")
    sys.stdout.flush()

    # We use K=4 as reference K to keep the number of blocks manageable
    # For each K, the number of blocks = K (one per signal in the permutation)
    # We measure at each block boundary.
    # To get a consistent "step" axis, we use blocks 1..4 (steps 50..200)
    # which is the minimum across all K values.
    n_blocks = min(K_VALUES)  # 4 blocks = steps 50, 100, 150, 200

    results = {tau: {blk: [] for blk in range(n_blocks)} for tau in [TAU_A, TAU_B]}

    for seed in SEEDS:
        for tau in [TAU_A, TAU_B]:
            k_block_gaps = {blk: [] for blk in range(n_blocks)}
            for K in K_VALUES:
                org = Organism(seed=seed, alive=True, tau=tau)
                sigs = make_signals(K, seed=seed + 500)
                perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
                for blk in range(n_blocks):
                    gap = measure_gap_at_block(org, perms, sigs, seed, N_TRIALS, blk)
                    k_block_gaps[blk].append(gap)
            for blk in range(n_blocks):
                results[tau][blk].append(mean(k_block_gaps[blk]))
        print(f"  Seed {seed} done.", flush=True)

    print(f"\n  MI gap trajectory (averaged over {len(SEEDS)} seeds, multi-K):")
    print(f"\n  {'block':>6} {'step':>6} | {'tau=0.2':>10} {'tau=0.3':>10} | {'diff':>10} {'d':>8} {'p':>8}")
    print(f"  {'-'*6} {'-'*6}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*8}-{'-'*8}")

    for blk in range(n_blocks):
        step = (blk + 1) * 50
        vals_a = results[TAU_A][blk]
        vals_b = results[TAU_B][blk]
        m_a = mean(vals_a)
        m_b = mean(vals_b)
        diff = m_a - m_b
        d = cohens_d_paired(vals_a, vals_b)
        p = paired_t_p(vals_a, vals_b)
        print(f"  {blk+1:>6} {step:>6} | {m_a:>+10.4f} {m_b:>+10.4f} | {diff:>+10.4f} {d:>+8.3f} {p:>8.3f}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# DIAGNOSTIC 3: Cell-cell state correlation
#
# After full run_sequence, compute pairwise cosine similarity between all
# cell state vectors. Lower mean = more differentiated cells.
# ---------------------------------------------------------------------------

def run_sequence_standard(org, perm, sigs, rng_seed, trial=0):
    """Standard run_sequence returning final state."""
    rng = random.Random(rng_seed * 100 + trial)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for idx in perm:
        sig = sigs[idx]
        for _ in range(50):
            xs = org.step(xs, signal=sig if org.alive else None)
    return xs


def cell_pairwise_cosine(xs):
    """Compute mean pairwise cosine similarity between all cell state vectors."""
    sims = []
    for i in range(NC):
        for j in range(i + 1, NC):
            sims.append(vcosine(xs[i], xs[j]))
    return mean(sims)


def diagnostic_cell_correlation():
    print(f"\n{'='*W}")
    print(f"  DIAGNOSTIC 3: CELL-CELL STATE CORRELATION")
    print(f"  Pairwise cosine similarity at end of run_sequence")
    print(f"  Multi-K={K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"{'='*W}")
    sys.stdout.flush()

    # For each seed, collect mean pairwise cosine over all (K, perm, trial) runs
    results = {tau: [] for tau in [TAU_A, TAU_B]}

    for seed in SEEDS:
        for tau in [TAU_A, TAU_B]:
            all_cosines = []
            for K in K_VALUES:
                org = Organism(seed=seed, alive=True, tau=tau)
                sigs = make_signals(K, seed=seed + 500)
                perms = gen_perms(K, n_perm=N_PERM, seed=seed + 300)
                for perm in perms:
                    for t in range(N_TRIALS):
                        xs = run_sequence_standard(org, perm, sigs, seed + perms.index(perm) * 1000, t)
                        all_cosines.append(cell_pairwise_cosine(xs))
            results[tau].append(mean(all_cosines))
        print(f"  Seed {seed} done.", flush=True)

    vals_a = results[TAU_A]
    vals_b = results[TAU_B]
    m_a = mean(vals_a)
    m_b = mean(vals_b)
    diff = m_a - m_b
    d = cohens_d_paired(vals_a, vals_b)
    p = paired_t_p(vals_a, vals_b)

    print(f"\n  Mean pairwise cosine similarity (averaged over {len(SEEDS)} seeds):")
    print(f"    tau=0.2: {m_a:+.4f}  (per-seed: {' '.join(f'{v:+.4f}' for v in vals_a)})")
    print(f"    tau=0.3: {m_b:+.4f}  (per-seed: {' '.join(f'{v:+.4f}' for v in vals_b)})")
    print(f"\n  Paired comparison (tau=0.2 vs tau=0.3):")
    print(f"    diff={diff:+.5f}, d={d:+.3f}, p={p:.3f}")
    print(f"\n  Interpretation: Lower cosine = more differentiated cells.")
    if diff < 0:
        print(f"    tau=0.2 produces MORE differentiated cells (lower cosine).")
    elif diff > 0:
        print(f"    tau=0.2 produces LESS differentiated cells (higher cosine).")
    else:
        print(f"    No difference in cell differentiation.")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * W)
    print("  STAGE 4: TAU META-SIGNAL DIAGNOSTIC")
    print("  Comparing tau=0.2 vs tau=0.3 — WHERE does tau matter?")
    print("=" * W)
    print(f"\nSeeds: {SEEDS}")
    print(f"Multi-K: {K_VALUES}, n_perm={N_PERM}, n_trials={N_TRIALS}")
    print(f"Tau values: {TAU_A} vs {TAU_B}")
    sys.stdout.flush()

    diagnostic_alpha_trajectory()
    diagnostic_mi_trajectory()
    diagnostic_cell_correlation()

    print(f"\n{'='*W}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*W}")
    print(f"""
  Tau controls the attention temperature in the cell-cell coupling weights.
  Lower tau = sharper attention (cells attend to fewer neighbors).
  Higher tau = softer attention (cells attend more uniformly).

  Diagnostic 1 (alpha trajectory) reveals whether tau changes HOW alpha
  values evolve — i.e., does the plasticity mechanism behave differently?

  Diagnostic 2 (MI trajectory) reveals whether tau changes WHEN the MI
  gap forms — earlier blocks vs later blocks.

  Diagnostic 3 (cell correlation) reveals whether tau changes the OUTCOME
  of state dynamics — more or less cell differentiation.

  If tau affects correlation but NOT alpha, the mechanism is purely in
  the state dynamics (coupling), not in the plasticity rule.
""")
    print("=" * W)
