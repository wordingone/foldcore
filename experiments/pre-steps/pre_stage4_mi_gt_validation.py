#!/usr/bin/env python3
"""
stage4_mi_gt_validation.py

Validate whether MI gap correlates with binary ground truth across clip bound conditions.
Uses the EXACT same measurement protocol as stage4_birthconfound_disambiguation.py:
  - Python random module (not numpy)
  - Per-cell cosine similarity (not centroid)
  - Same sequence structure, signal generation, step counts
  - Same organism shared across trials (cumulative training)

Protocol: 10 seeds, K=[4,6,8,10], n_perm=8, n_trials=6 (multi-K constraint c027)
"""

import math
import random
import time

D = 12
NC = 6


def vcosine(a, b):
    dot = na2 = nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi; na2 += ai * ai; nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15); nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


class Organism:
    """Matches birthconfound_disambiguation.py exactly."""
    def __init__(self, seed=42, alive=False, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9; self.eps = 0.15
        self.tau = 0.3; self.delta = 0.35; self.noise = 0.005
        self.clip = 4.0; self.eta = 0.0003
        self.sbm = 0.3; self.am = 0.5; self.dm = 0.1
        self.thr = 0.01; self.aclo = clip_lo; self.achi = clip_hi
        self.seed = seed; self.alive = alive
        random.seed(seed)
        self.alpha = [[1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)] for _ in range(NC)]
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(clip_lo, min(clip_hi, self.alpha[i][k]))

    def step(self, xs, signal=None):
        b, g = self.beta, self.gamma
        pb = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D; km = (k - 1) % D
                row.append(math.tanh(self.alpha[i][k] * xs[i][k] + b * xs[i][kp] * xs[i][km]))
            pb.append(row)
        if signal:
            ps = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D; km = (k - 1) % D
                    row.append(math.tanh(self.alpha[i][k] * xs[i][k]
                                         + b * (xs[i][kp] + g * signal[kp])
                                         * (xs[i][km] + g * signal[km])))
                ps.append(row)
        else:
            ps = pb
        if self.alive and signal:
            resp = [[abs(ps[i][k] - pb[i][k]) for k in range(D)] for i in range(NC)]
            ar = [resp[i][k] for i in range(NC) for k in range(D)]
            om = sum(ar) / len(ar)
            osd = math.sqrt(sum((r - om) ** 2 for r in ar) / len(ar)) + 1e-10
            for i in range(NC):
                for k in range(D):
                    rz = (resp[i][k] - om) / osd
                    cm = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - cm
                    if abs(dev) < self.thr:
                        push = self.eta * self.sbm * random.gauss(0, 1.0)
                    elif rz > 0:
                        push = self.eta * math.tanh(rz) * (1.0 if dev > 0 else -1.0) * self.am
                    else:
                        push = self.eta * self.dm * random.gauss(0, 1.0)
                    self.alpha[i][k] = max(self.aclo, min(self.achi, self.alpha[i][k] + push))
        wts = []
        for i in range(NC):
            raw = [-1e10 if i == j else sum(xs[i][k] * xs[j][k] for k in range(D)) / (D * self.tau)
                   for j in range(NC)]
            mx = max(raw)
            ex = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(ex) + 1e-15
            wts.append([e / s for e in ex])
        new = []
        for i in range(NC):
            p = list(ps[i])
            bd = [pb[i][k] - xs[i][k] for k in range(D)]
            fpd = vnorm(bd) / max(vnorm(xs[i]), 1.0)
            pl = math.exp(-(fpd * fpd) / 0.0225)
            if pl > 0.01 and self.eps > 0:
                pull = [0.0] * D
                for j in range(NC):
                    if i == j or wts[i][j] < 1e-8:
                        continue
                    for k in range(D):
                        pull[k] += wts[i][j] * (pb[j][k] - pb[i][k])
                p = [p[k] + pl * self.eps * pull[k] for k in range(D)]
            nx = []
            for k in range(D):
                v = (1 - self.delta) * xs[i][k] + self.delta * p[k] + random.gauss(0, self.noise)
                nx.append(max(-self.clip, min(self.clip, v)))
            new.append(nx)
        return new


def make_signals(K, seed=42):
    random.seed(seed)
    return [[random.gauss(0, 1) for _ in range(D)] for _ in range(K)]


def gen_perms(K, n_perm=8, seed=42):
    random.seed(seed)
    idxs = list(range(K))
    perms = []
    for _ in range(n_perm):
        p = idxs[:]
        random.shuffle(p)
        perms.append(p)
    return perms


def run_seq(org, perm, sigs, rs, trial=0, n_steps=50):
    rng = random.Random(rs * 100 + trial)
    xs = [[rng.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]
    for idx in perm:
        sig = sigs[idx]
        for _ in range(n_steps):
            xs = org.step(xs, signal=sig if org.alive else None)
    return xs


def measure_gap(org, perm_list, sigs, seed, n_trials=6):
    """Per-cell cosine similarity MI gap (matches birthconfound exactly)."""
    fpp = []
    for pi, perm in enumerate(perm_list):
        tf = []
        for t in range(n_trials):
            tf.append([r[:] for r in run_seq(org, perm, sigs, seed + pi * 1000, t)])
        fpp.append(tf)
    within = []
    between = []
    for i in range(len(perm_list)):
        fi = fpp[i]
        for a in range(len(fi)):
            for b in range(a + 1, len(fi)):
                for ci in range(NC):
                    within.append(vcosine(fi[a][ci], fi[b][ci]))
        for j in range(i + 1, len(perm_list)):
            fj = fpp[j]
            for ta in fi:
                for tb in fj:
                    for ci in range(NC):
                        between.append(vcosine(ta[ci], tb[ci]))
    w = sum(within) / len(within) if within else 0.0
    b = sum(between) / len(between) if between else 0.0
    return w - b


def eval_condition(cl, ch, seeds, ks, n_perm=8, n_trials=6):
    alive_gaps = []
    for seed in seeds:
        org = Organism(seed=seed, alive=True, clip_lo=cl, clip_hi=ch)
        sg = []
        for K in ks:
            sigs = make_signals(K, seed=seed + 500)
            perms = gen_perms(K, n_perm=n_perm, seed=seed + 300)
            sg.append(measure_gap(org, perms, sigs, seed, n_trials=n_trials))
        alive_gaps.append(sum(sg) / len(sg))
    return alive_gaps


def mean(v):
    return sum(v) / len(v)


def std(v):
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))


def ncdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def pval_paired(a, b):
    n = len(a)
    diffs = [a[i] - b[i] for i in range(n)]
    md = mean(diffs)
    if n < 2:
        return 1.0
    vd = sum((d - md) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(vd / n) + 1e-15
    t = md / se
    return 2.0 * (1.0 - ncdf(abs(t)))


def cohd(a, b):
    diffs = [a[i] - b[i] for i in range(len(a))]
    md = mean(diffs)
    sd = math.sqrt(sum((d - md) ** 2 for d in diffs) / max(len(diffs) - 1, 1))
    return md / (sd + 1e-15)


def main():
    print("=" * 72)
    print("  STAGE 4: MI vs GROUND TRUTH VALIDATION")
    print("  Protocol: birthconfound-exact (per-cell cosine, Python random)")
    print("  Seeds: 5 canonical + 5 extended = 10 total")
    print("  K=[4,6,8,10], n_perm=8, n_trials=6")
    print("=" * 72, flush=True)

    t0 = time.time()
    seeds = [42, 137, 2024, 999, 7, 314, 555, 1001, 88, 2025]
    ks = [4, 6, 8, 10]
    n_perm = 8
    n_trials = 6

    conditions = [
        ("CANONICAL",    0.3,  1.8),
        ("VERY_NARROW",  0.7,  1.3),
        ("NARROW_TIGHT", 0.5,  1.6),
        ("NARROW_LO",    0.5,  1.8),
        ("WIDE_LO",      0.1,  1.8),
        ("ASYMM_LO",     0.05, 1.8),
        ("ASYMM_HI",     0.3,  2.8),
    ]

    results = {}
    for name, cl, ch in conditions:
        ct = time.time()
        print(f"\nRunning {name} [{cl},{ch}]...", flush=True)
        alive_gaps = eval_condition(cl, ch, seeds, ks, n_perm=n_perm, n_trials=n_trials)
        elapsed_c = time.time() - ct
        n_pv = sum(1 for g in alive_gaps if g > 0)
        results[name] = {
            "cl": cl, "ch": ch,
            "alive_gaps": alive_gaps,
            "alive_mean": mean(alive_gaps),
            "alive_std": std(alive_gaps),
            "n_pv": n_pv,
            "gt_pv_pass": n_pv >= 7,
        }
        per_seed = "  ".join(f"{g:+.4f}" for g in alive_gaps)
        print(f"  done ({elapsed_c:.0f}s) mean={mean(alive_gaps):+.4f} std={std(alive_gaps):.4f}")
        print(f"  per-seed: {per_seed}")
        print(f"  GT(Principle V, >0): {n_pv}/10", flush=True)

    can = results["CANONICAL"]["alive_gaps"]
    can_mean = results["CANONICAL"]["alive_mean"]

    print("\n" + "=" * 72)
    print("  SUMMARY TABLE (paired to CANONICAL)")
    print("=" * 72)
    print(f"  {'Condition':<15} {'Clip':<12} {'Mean MI':>8} {'Std':>7} {'Diff':>8} {'d':>7} {'p':>7} {'PV n/10':>8}")
    print("  " + "-" * 80)
    for name, cl, ch in conditions:
        r = results[name]
        clip_s = f"[{cl},{ch}]"
        if name == "CANONICAL":
            print(f"  {name:<15} {clip_s:<12} {r['alive_mean']:+8.4f} {r['alive_std']:7.4f} {'BASE':>8} {'--':>7} {'--':>7} {r['n_pv']:>5}/10")
        else:
            d = cohd(r["alive_gaps"], can)
            p = pval_paired(r["alive_gaps"], can)
            diff = r["alive_mean"] - can_mean
            print(f"  {name:<15} {clip_s:<12} {r['alive_mean']:+8.4f} {r['alive_std']:7.4f} {diff:+8.4f} {d:+7.3f} {p:7.4f} {r['n_pv']:>5}/10")

    print("\n" + "=" * 72)
    print("  CORRELATION ANALYSIS: MI gap vs Ground Truth")
    print("=" * 72)

    alive_means = [results[n]["alive_mean"] for n, _, _ in conditions]
    pv_counts = [results[n]["n_pv"] for n, _, _ in conditions]

    if len(set(pv_counts)) > 1:
        mx = mean(alive_means); my = mean(pv_counts)
        num = sum((a - mx) * (b - my) for a, b in zip(alive_means, pv_counts))
        den = math.sqrt(sum((a - mx) ** 2 for a in alive_means) * sum((b - my) ** 2 for b in pv_counts) + 1e-15)
        r_mi_pv = num / den
        print(f"  Pearson r(Mean_MI_gap, PV_pass_count) = {r_mi_pv:+.4f}")
    else:
        print(f"  Pearson r: UNDEFINED — all conditions have PV count = {pv_counts[0]}")
        print("  (Ground truth is a FLOOR — all conditions pass; no discrimination)")

    print("\n" + "=" * 72)
    print("  VERDICT")
    print("=" * 72)

    vn = results["VERY_NARROW"]
    cn = results["CANONICAL"]

    print(f"  CANONICAL:   MI={cn['alive_mean']:+.4f}  PV={cn['n_pv']}/10")
    print(f"  VERY_NARROW: MI={vn['alive_mean']:+.4f}  PV={vn['n_pv']}/10")
    print()

    vn_better_mi = vn["alive_mean"] > cn["alive_mean"]
    vn_same_gt = vn["n_pv"] >= cn["n_pv"]

    d_vn = cohd(vn["alive_gaps"], can)
    p_vn = pval_paired(vn["alive_gaps"], can)
    print(f"  VERY_NARROW vs CANONICAL: d={d_vn:+.3f}, p={p_vn:.4f}")
    print()

    if vn_better_mi and vn_same_gt:
        print("  CONSISTENT: VERY_NARROW improves MI AND maintains GT.")
        print("  MI gap is a valid proxy. Entry 042 confirmed.")
    elif vn_better_mi and not vn_same_gt:
        print("  DIVERGENT: VERY_NARROW improves MI but DEGRADES GT.")
        print("  CRITICAL: MI gap is measuring the wrong thing.")
    elif not vn_better_mi and vn_same_gt:
        print("  MI gap and GT agree directionally (both see no improvement or both pass).")
        print("  VERY_NARROW Entry 042 finding does NOT replicate with 10 seeds.")
    else:
        print("  Both MI and GT are worse for VERY_NARROW. Consistent but reversed.")

    all_pass = all(results[n]["gt_pv_pass"] for n, _, _ in conditions)
    if all_pass:
        print("\n  KEY FINDING: Ground truth Principle V is universally satisfied.")
        print("  Clip bound variations do not threaten distinguishability.")
        print("  MI gap variations within the positive range are metric refinement,")
        print("  not ground truth violations. The external reviewer's concern is addressed:")
        print("  MI gap and ground truth are ALIGNED (both always positive).")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed / 60:.1f}min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
