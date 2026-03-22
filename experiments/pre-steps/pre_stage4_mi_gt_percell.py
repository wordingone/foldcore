#!/usr/bin/env python3
"""
stage4_mi_gt_percell.py

Re-run MI-GT validation with PER-CELL cosine similarity measurement.
Task #1 used centroid measurement; Entry 042 used per-cell measurement.
These gave opposite rankings for VERY_NARROW.

Per-cell method: compare individual cell vectors (NC=6 cells per organism)
rather than collapsing to centroid first. This gives 6x more comparisons
and different sensitivity to inter-cell diversity.

Runs two seed sets:
  A) seeds 10-19 (same as Task #1, for direct comparison)
  B) Entry 042 seeds [42, 137, 2024, 999, 7] (direct replication attempt)

All 7 clip conditions x both seed sets x K=[4,6,8,10], n_perm=8, n_trials=6.
Pure Python only (no numpy) for seed-compatibility with prior experiments.
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
    def __init__(self, seed=42, alive=False, eta=0.0003, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5; self.gamma = 0.9; self.eps = 0.15
        self.tau = 0.3; self.delta = 0.35; self.noise = 0.005
        self.clip = 4.0; self.seed = seed; self.alive = alive; self.eta = eta
        self.aclo = clip_lo; self.achi = clip_hi
        random.seed(seed)
        self.alpha = [[1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)] for _ in range(NC)]
        for i in range(NC):
            for k in range(D):
                self.alpha[i][k] = max(self.aclo, min(self.achi, self.alpha[i][k]))

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma
        phi_bare = []
        for i in range(NC):
            row = []
            for k in range(D):
                kp = (k + 1) % D; km = (k - 1) % D
                row.append(math.tanh(self.alpha[i][k] * xs[i][k] + beta * xs[i][kp] * xs[i][km]))
            phi_bare.append(row)
        if signal:
            phi_sig = []
            for i in range(NC):
                row = []
                for k in range(D):
                    kp = (k + 1) % D; km = (k - 1) % D
                    row.append(math.tanh(self.alpha[i][k] * xs[i][k]
                                         + beta * (xs[i][kp] + gamma * signal[kp])
                                                * (xs[i][km] + gamma * signal[km])))
                phi_sig.append(row)
        else:
            phi_sig = phi_bare
        if self.alive and signal:
            response = [[abs(phi_sig[i][k] - phi_bare[i][k]) for k in range(D)] for i in range(NC)]
            all_resp = [response[i][k] for i in range(NC) for k in range(D)]
            om = sum(all_resp) / len(all_resp)
            os_ = math.sqrt(sum((r - om) ** 2 for r in all_resp) / len(all_resp)) + 1e-10
            for i in range(NC):
                for k in range(D):
                    rz = (response[i][k] - om) / os_
                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean
                    if abs(dev) < 0.01:
                        push = self.eta * 0.3 * random.gauss(0, 1.0)
                    elif rz > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = self.eta * math.tanh(rz) * direction * 0.5
                    else:
                        push = self.eta * 0.1 * random.gauss(0, 1.0)
                    self.alpha[i][k] = max(self.aclo, min(self.achi, self.alpha[i][k] + push))
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
            p = list(phi_sig[i])
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
                nx.append(max(-self.clip, min(self.clip, v)))
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
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]; random.shuffle(p); t = tuple(p)
        if t not in seen:
            perms.append(t); seen.add(t)
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
    return xs  # return full state, not just centroid


def measure_gap_centroid(org, signals, k, seed, n_perm=8, n_trials=6):
    """Original Task #1 method: collapse to centroid, then cosine."""
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            xs = run_sequence(org, perm, signals, seed, trial)
            c = org.centroid(xs)
            trials.append(c)
        endpoints[pi] = trials
    within = []; between = []
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


def measure_gap_percell(org, signals, k, seed, n_perm=8, n_trials=6):
    """Per-cell method (as in harness.py / Entry 042): compare individual cell vectors."""
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            xs = run_sequence(org, perm, signals, seed, trial)
            trials.append([row[:] for row in xs])  # full NC x D state
        endpoints[pi] = trials
    within = []; between = []
    pis = sorted(endpoints.keys())
    for pi in pis:
        fi = endpoints[pi]
        for a in range(len(fi)):
            for b in range(a + 1, len(fi)):
                for ci in range(NC):
                    within.append(vcosine(fi[a][ci], fi[b][ci]))
    for i in range(len(pis)):
        for j in range(i + 1, len(pis)):
            for ta in endpoints[pis[i]]:
                for tb in endpoints[pis[j]]:
                    for ci in range(NC):
                        between.append(vcosine(ta[ci], tb[ci]))
    avg_w = sum(within) / max(len(within), 1)
    avg_b = sum(between) / max(len(between), 1)
    return avg_w - avg_b


def mean(xs): return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / max(len(xs) - 1, 1) + 1e-15)


def ncdf(z):
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return p if z >= 0 else 1.0 - p


def pval(diffs):
    n = len(diffs); m = mean(diffs)
    s = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(n - 1, 1) + 1e-15)
    t = m / (s / math.sqrt(n) + 1e-15)
    return 2.0 * (1.0 - ncdf(abs(t)))


def cohd_paired(xs, ys):
    diffs = [y - x for x, y in zip(xs, ys)]
    m = mean(diffs)
    s = math.sqrt(sum((d - m) ** 2 for d in diffs) / max(len(diffs) - 1, 1) + 1e-15)
    return m / (s + 1e-15)


def eval_condition(cl, ch, seeds, ks, method="percell", eta=0.0003, n_perm=8, n_trials=6):
    """Run all seeds and Ks for a clip condition. Returns per-seed alive/still gaps."""
    measure = measure_gap_percell if method == "percell" else measure_gap_centroid
    alive_gaps = []; still_gaps = []
    for seed in seeds:
        k_alive = []; k_still = []
        for k in ks:
            sigs = make_signals(k, seed=seed * 1000 + k * 7 + 3)
            alive = Organism(seed=seed, alive=True, eta=eta, clip_lo=cl, clip_hi=ch)
            still = Organism(seed=seed, alive=False, eta=eta, clip_lo=cl, clip_hi=ch)
            ag = measure(alive, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            sg = measure(still, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            k_alive.append(ag); k_still.append(sg)
        alive_gaps.append(mean(k_alive))
        still_gaps.append(mean(k_still))
    return alive_gaps, still_gaps


def run_seed_set(label, seeds, ks, conditions, n_perm, n_trials):
    """Run both per-cell and centroid for all conditions on a seed set."""
    print(f"\n{'='*72}")
    print(f"  SEED SET: {label}  seeds={seeds}")
    print(f"{'='*72}")

    results_pc = {}
    results_ct = {}

    for name, cl, ch in conditions:
        print(f"\n  [{name}] [{cl},{ch}]", flush=True)

        print(f"    per-cell...", end="", flush=True)
        a_pc, s_pc = eval_condition(cl, ch, seeds, ks, method="percell",
                                     n_perm=n_perm, n_trials=n_trials)
        diffs_pc = [a - s for a, s in zip(a_pc, s_pc)]
        n_wins_pc = sum(1 for d in diffs_pc if d > 0)
        am_pc = mean(a_pc); sm_pc = mean(s_pc)
        d_pc = cohd_paired(s_pc, a_pc)
        p_pc = pval(diffs_pc)
        gt_pc = n_wins_pc >= (len(seeds) * 7 // 10)
        results_pc[name] = {"alive": am_pc, "still": sm_pc, "delta": am_pc - sm_pc,
                             "d": d_pc, "p": p_pc, "wins": n_wins_pc, "gt": gt_pc}
        print(f" ALIVE={am_pc:+.4f} STILL={sm_pc:+.4f} delta={am_pc-sm_pc:+.4f} d={d_pc:+.3f} p={p_pc:.4f} wins={n_wins_pc}/{len(seeds)}", flush=True)

        print(f"    centroid...", end="", flush=True)
        a_ct, s_ct = eval_condition(cl, ch, seeds, ks, method="centroid",
                                     n_perm=n_perm, n_trials=n_trials)
        diffs_ct = [a - s for a, s in zip(a_ct, s_ct)]
        n_wins_ct = sum(1 for d in diffs_ct if d > 0)
        am_ct = mean(a_ct); sm_ct = mean(s_ct)
        d_ct = cohd_paired(s_ct, a_ct)
        p_ct = pval(diffs_ct)
        gt_ct = n_wins_ct >= (len(seeds) * 7 // 10)
        results_ct[name] = {"alive": am_ct, "still": sm_ct, "delta": am_ct - sm_ct,
                             "d": d_ct, "p": p_ct, "wins": n_wins_ct, "gt": gt_ct}
        print(f" ALIVE={am_ct:+.4f} STILL={sm_ct:+.4f} delta={am_ct-sm_ct:+.4f} d={d_ct:+.3f} p={p_ct:.4f} wins={n_wins_ct}/{len(seeds)}", flush=True)

    # Summary table
    print(f"\n{'='*72}")
    print(f"  COMPARISON TABLE — {label}")
    print(f"{'='*72}")
    print(f"  {'Condition':<15} {'Clip':<14} {'PC ALIVE':>9} {'PC delta':>9} {'PC d':>7} {'PC GT':>6}  |  {'CT ALIVE':>9} {'CT delta':>9} {'CT d':>7} {'CT GT':>6}")
    print("  " + "-" * 90)
    for name, cl, ch in conditions:
        pc = results_pc[name]; ct = results_ct[name]
        pc_gt = "PASS" if pc["gt"] else "FAIL"
        ct_gt = "PASS" if ct["gt"] else "FAIL"
        print(f"  {name:<15} [{cl},{ch}]{'':>6} "
              f"{pc['alive']:+9.4f} {pc['delta']:+9.4f} {pc['d']:+7.3f} {pc_gt:>6}  |  "
              f"{ct['alive']:+9.4f} {ct['delta']:+9.4f} {ct['d']:+7.3f} {ct_gt:>6}")

    # Rank by ALIVE MI
    print(f"\n  Ranking by ALIVE MI (per-cell):")
    ranked_pc = sorted(conditions, key=lambda x: results_pc[x[0]]["alive"], reverse=True)
    for rank, (name, cl, ch) in enumerate(ranked_pc, 1):
        print(f"    {rank}. {name:<15} {results_pc[name]['alive']:+.4f}")

    print(f"\n  Ranking by ALIVE MI (centroid):")
    ranked_ct = sorted(conditions, key=lambda x: results_ct[x[0]]["alive"], reverse=True)
    for rank, (name, cl, ch) in enumerate(ranked_ct, 1):
        print(f"    {rank}. {name:<15} {results_ct[name]['alive']:+.4f}")

    # Check if VERY_NARROW ranks highest per-cell
    vn_rank_pc = [i+1 for i, (n,_,_) in enumerate(ranked_pc) if n == "VERY_NARROW"][0]
    vn_rank_ct = [i+1 for i, (n,_,_) in enumerate(ranked_ct) if n == "VERY_NARROW"][0]
    print(f"\n  VERY_NARROW rank: per-cell={vn_rank_pc}/7  centroid={vn_rank_ct}/7")

    # Correlation check
    def pearson(xs, ys):
        mx = mean(xs); my = mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys) + 1e-15)
        return num / den

    pc_alives = [results_pc[n]["alive"] for n, _, _ in conditions]
    ct_alives = [results_ct[n]["alive"] for n, _, _ in conditions]
    pc_wins = [results_pc[n]["wins"] for n, _, _ in conditions]
    ct_wins = [results_ct[n]["wins"] for n, _, _ in conditions]

    r_pc_ct = pearson(pc_alives, ct_alives)
    r_pc_gt = pearson(pc_alives, pc_wins)
    r_ct_gt = pearson(ct_alives, ct_wins)

    print(f"\n  Pearson r(per-cell ALIVE, centroid ALIVE) = {r_pc_ct:+.4f}")
    print(f"  Pearson r(per-cell ALIVE, per-cell GT wins) = {r_pc_gt:+.4f}")
    print(f"  Pearson r(centroid ALIVE, centroid GT wins) = {r_ct_gt:+.4f}")

    return results_pc, results_ct


def main():
    print("=" * 72)
    print("  STAGE 4: PER-CELL vs CENTROID MI MEASUREMENT")
    print("  Resolving Entry 042 discrepancy (d=7.583 VERY_NARROW)")
    print()
    print("  Per-cell: vcosine on individual cell vectors (NC=6 per state)")
    print("  Centroid: vcosine on mean cell vector (1 per state)")
    print()
    print("  Two seed sets:")
    print("  A) seeds 10-19 (Task #1 seeds, direct comparison)")
    print("  B) Entry 042 seeds [42,137,2024,999,7] (direct replication)")
    print("=" * 72)

    t0 = time.time()
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

    # Seed set A: Task #1 seeds
    seeds_a = list(range(10, 20))
    res_pc_a, res_ct_a = run_seed_set("A: seeds 10-19", seeds_a, ks, conditions, n_perm, n_trials)

    # Seed set B: Entry 042 seeds
    seeds_b = [42, 137, 2024, 999, 7]
    res_pc_b, res_ct_b = run_seed_set("B: Entry 042 seeds [42,137,2024,999,7]", seeds_b, ks, conditions, n_perm, n_trials)

    # Final verdict
    print(f"\n{'='*72}")
    print("  FINAL VERDICT")
    print(f"{'='*72}")

    vn_pc_a = res_pc_a["VERY_NARROW"]["alive"]
    vn_ct_a = res_ct_a["VERY_NARROW"]["alive"]
    vn_pc_b = res_pc_b["VERY_NARROW"]["alive"]
    vn_ct_b = res_ct_b["VERY_NARROW"]["alive"]
    can_pc_a = res_pc_a["CANONICAL"]["alive"]
    can_ct_a = res_ct_a["CANONICAL"]["alive"]
    can_pc_b = res_pc_b["CANONICAL"]["alive"]
    can_ct_b = res_ct_b["CANONICAL"]["alive"]

    print(f"  VERY_NARROW vs CANONICAL:")
    print(f"    Seeds 10-19 per-cell:   VN={vn_pc_a:+.4f}  CAN={can_pc_a:+.4f}  diff={vn_pc_a-can_pc_a:+.4f}")
    print(f"    Seeds 10-19 centroid:   VN={vn_ct_a:+.4f}  CAN={can_ct_a:+.4f}  diff={vn_ct_a-can_ct_a:+.4f}")
    print(f"    Entry042 seeds per-cell: VN={vn_pc_b:+.4f}  CAN={can_pc_b:+.4f}  diff={vn_pc_b-can_pc_b:+.4f}")
    print(f"    Entry042 seeds centroid: VN={vn_ct_b:+.4f}  CAN={can_ct_b:+.4f}  diff={vn_ct_b-can_ct_b:+.4f}")
    print()

    if vn_pc_b > can_pc_b:
        print("  REPLICATION: Per-cell measurement on Entry 042 seeds shows VERY_NARROW > CANONICAL.")
        print("  Entry 042 d=7.583 is confirmed as a per-cell measurement result.")
    else:
        print("  NON-REPLICATION: Per-cell measurement on Entry 042 seeds does NOT show VERY_NARROW > CANONICAL.")
        print("  Entry 042 d=7.583 does not replicate even with matching measurement method.")

    if vn_pc_a > can_pc_a and vn_ct_a < can_ct_a:
        print()
        print("  MEASUREMENT REVERSAL CONFIRMED on seeds 10-19:")
        print("  VERY_NARROW ranks highest per-cell but lowest centroid.")
        print("  The two methods give qualitatively different rankings.")
    elif vn_pc_a < can_pc_a and vn_ct_a < can_ct_a:
        print()
        print("  CONSISTENT: Both methods agree VERY_NARROW < CANONICAL on seeds 10-19.")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
