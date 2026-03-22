#!/usr/bin/env python3
"""
stage4_percell_vs_centroid.py

Compare per-cell vs centroid MI gap measurement on same organisms/trials.
Uses the cumulative-training protocol from stage4_mi_gt_validation.py.

Key question: Does VERY_NARROW rank highest with per-cell measurement
(replicating Entry 042's d=7.583) while ranking lowest with centroid?

Runs on TWO seed sets:
  Set A: [10-19]  (used in MI-GT validation)
  Set B: [42, 137, 2024, 999, 7]  (used in Entry 042)
"""

import math
import time
import numpy as np

D = 12
NC = 6


class Organism:
    def __init__(self, seed=42, alive=False, eta=0.0003, clip_lo=0.3, clip_hi=1.8):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.eta = eta
        self.aclo = clip_lo
        self.achi = clip_hi
        rng = np.random.RandomState(seed)
        self.alpha = np.clip(1.1 + 0.7 * (rng.random((NC, D)) * 2 - 1), clip_lo, clip_hi)
        self._rng = rng

    def step(self, xs, signal=None):
        beta, gamma = self.beta, self.gamma
        xs_kp = np.roll(xs, -1, axis=1)
        xs_km = np.roll(xs, 1, axis=1)

        phi_bare = np.tanh(self.alpha * xs + beta * xs_kp * xs_km)

        if signal is not None:
            sig_kp = np.roll(signal, -1)
            sig_km = np.roll(signal, 1)
            phi_sig = np.tanh(self.alpha * xs + beta * (xs_kp + gamma * sig_kp) * (xs_km + gamma * sig_km))
        else:
            phi_sig = phi_bare

        if self.alive and signal is not None:
            response = np.abs(phi_sig - phi_bare)
            om = response.mean()
            os_ = response.std() + 1e-10
            rz = (response - om) / os_

            col_mean = self.alpha.mean(axis=0, keepdims=True)
            dev = self.alpha - col_mean

            noise_push = self.eta * 0.3 * self._rng.randn(NC, D)
            explore_push = self.eta * 0.1 * self._rng.randn(NC, D)
            direction = np.sign(dev)
            amplify_push = self.eta * np.tanh(rz) * direction * 0.5

            near_zero = np.abs(dev) < 0.01
            pos_rz = rz > 0
            push = np.where(near_zero, noise_push,
                            np.where(pos_rz, amplify_push, explore_push))
            self.alpha = np.clip(self.alpha + push, self.aclo, self.achi)

        dots = np.einsum('id,jd->ij', xs, xs) / (D * self.tau)
        np.fill_diagonal(dots, -1e10)
        dots -= dots.max(axis=1, keepdims=True)
        np.clip(dots, -50, 50, out=dots)
        exps = np.exp(dots)
        weights = exps / (exps.sum(axis=1, keepdims=True) + 1e-15)

        bare_diff = phi_bare - xs
        fp_d = np.linalg.norm(bare_diff, axis=1) / np.maximum(np.linalg.norm(xs, axis=1), 1.0)
        plast = np.exp(-(fp_d * fp_d) / 0.0225)

        w_masked = weights.copy()
        w_masked[weights < 1e-8] = 0.0
        np.fill_diagonal(w_masked, 0.0)
        pull = w_masked @ phi_bare - (w_masked.sum(axis=1, keepdims=True)) * phi_bare

        mask = (plast > 0.01) & (self.eps > 0)
        p = phi_sig.copy()
        p[mask] += (plast[mask, None] * self.eps) * pull[mask]

        new = (1 - self.delta) * xs + self.delta * p
        new += self._rng.randn(NC, D) * self.noise
        np.clip(new, -self.clip, self.clip, out=new)
        return new


def make_signals(k, seed):
    rng = np.random.RandomState(seed)
    sigs = {}
    for i in range(k):
        s = rng.randn(D) * 0.5
        s = s * (0.8 / (np.linalg.norm(s) + 1e-15))
        sigs[i] = s
    return sigs


def gen_perms(k, n_perm, seed):
    rng = np.random.RandomState(seed)
    base = list(range(k))
    perms = [tuple(base), tuple(reversed(base))]
    seen = set(perms)
    att = 0
    while len(perms) < n_perm and att < n_perm * 50:
        p = base[:]
        rng.shuffle(p)
        t = tuple(p)
        if t not in seen:
            perms.append(t)
            seen.add(t)
        att += 1
    return perms


def run_sequence(org, order, signals, base_seed, trial,
                 n_org=300, n_per_sig=60, n_settle=30, n_final=60):
    rng = np.random.RandomState(base_seed)
    xs = rng.randn(NC, D) * 0.5

    for _ in range(n_org):
        xs = org.step(xs)
    for idx, sid in enumerate(order):
        sig_rng = np.random.RandomState(base_seed * 1000 + sid * 100 + idx * 10 + trial)
        sig = signals[sid] + sig_rng.randn(D) * 0.05
        for _ in range(n_per_sig):
            xs = org.step(xs, sig)
        for _ in range(n_settle):
            xs = org.step(xs)
    for _ in range(n_final):
        xs = org.step(xs)
    return xs  # Return full (NC, D) array


def vcosine(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.clip(dot / (na * nb), -1.0, 1.0))


def measure_gap_both(org, signals, k, seed, n_perm=8, n_trials=6):
    """
    Compute BOTH centroid and per-cell MI gap on the same organism/trials.
    Uses cumulative training (shared organism across all trials).
    Returns (centroid_gap, percell_gap).
    """
    perms = gen_perms(k, n_perm, seed=seed * 10 + k)
    endpoints = {}  # perm_idx -> list of (NC, D) arrays
    for pi, perm in enumerate(perms):
        trials = []
        for trial in range(n_trials):
            xs = run_sequence(org, perm, signals, seed, trial)
            trials.append(xs)
        endpoints[pi] = trials

    # Centroid MI gap
    centroid_within = []
    centroid_between = []
    # Per-cell MI gap
    percell_within = []
    percell_between = []

    pis = sorted(endpoints.keys())
    for pi in pis:
        xs_list = endpoints[pi]
        for a in range(len(xs_list)):
            for b in range(a + 1, len(xs_list)):
                # Centroid
                ca = xs_list[a].mean(axis=0)
                cb = xs_list[b].mean(axis=0)
                centroid_within.append(vcosine(ca, cb))
                # Per-cell
                for ci in range(NC):
                    percell_within.append(vcosine(xs_list[a][ci], xs_list[b][ci]))

    for i in range(len(pis)):
        for j in range(i + 1, len(pis)):
            for xa in endpoints[pis[i]]:
                for xb in endpoints[pis[j]]:
                    # Centroid
                    ca = xa.mean(axis=0)
                    cb = xb.mean(axis=0)
                    centroid_between.append(vcosine(ca, cb))
                    # Per-cell
                    for ci in range(NC):
                        percell_between.append(vcosine(xa[ci], xb[ci]))

    centroid_gap = float(np.mean(centroid_within) - np.mean(centroid_between))
    percell_gap = float(np.mean(percell_within) - np.mean(percell_between))
    return centroid_gap, percell_gap


def eval_condition(cl, ch, seeds, ks, eta=0.0003, n_perm=8, n_trials=6):
    centroid_gaps = []
    percell_gaps = []
    for si, seed in enumerate(seeds):
        k_centroid = []
        k_percell = []
        for k in ks:
            sigs = make_signals(k, seed=seed * 1000 + k * 7 + 3)
            org = Organism(seed=seed, alive=True, eta=eta, clip_lo=cl, clip_hi=ch)
            cg, pg = measure_gap_both(org, sigs, k, seed, n_perm=n_perm, n_trials=n_trials)
            k_centroid.append(cg)
            k_percell.append(pg)
        centroid_gaps.append(np.mean(k_centroid))
        percell_gaps.append(np.mean(k_percell))
        print(f"    seed {seed}: centroid={np.mean(k_centroid):+.4f} percell={np.mean(k_percell):+.4f}",
              flush=True)
    return centroid_gaps, percell_gaps


def ncdf(z):
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    p = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return p if z >= 0 else 1.0 - p


def pval(diffs):
    n = len(diffs)
    m = np.mean(diffs)
    s = np.std(diffs, ddof=1)
    t = m / (s / math.sqrt(n) + 1e-15)
    return 2.0 * (1.0 - ncdf(abs(float(t))))


def cohd_paired(xs, ys):
    diffs = [y - x for x, y in zip(xs, ys)]
    m = np.mean(diffs)
    s = np.std(diffs, ddof=1)
    return float(m / (s + 1e-15))


def main():
    print("=" * 72)
    print("  PER-CELL vs CENTROID MI GAP COMPARISON")
    print()
    print("  Protocol: Multi-K [4,6,8,10], n_perm=8, n_trials=6")
    print("  Same organism measured BOTH ways in each run")
    print("=" * 72, flush=True)

    t0 = time.time()

    seed_sets = {
        "SetA_10-19": list(range(10, 20)),
        "SetB_Entry042": [42, 137, 2024, 999, 7],
    }

    ks = [4, 6, 8, 10]
    conditions = [
        ("CANONICAL",    0.3,  1.8),
        ("VERY_NARROW",  0.7,  1.3),
        ("NARROW_TIGHT", 0.5,  1.6),
        ("NARROW_LO",    0.5,  1.8),
        ("WIDE_LO",      0.1,  1.8),
        ("ASYMM_LO",     0.05, 1.8),
        ("ASYMM_HI",     0.3,  2.8),
    ]

    all_results = {}

    for set_name, seeds in seed_sets.items():
        print(f"\n{'=' * 72}")
        print(f"  SEED SET: {set_name} ({seeds})")
        print(f"{'=' * 72}", flush=True)

        results = {}
        for name, cl, ch in conditions:
            ct = time.time()
            print(f"\n  {name} [{cl},{ch}]...", flush=True)
            centroid_gaps, percell_gaps = eval_condition(cl, ch, seeds, ks)
            elapsed_c = time.time() - ct

            cm = np.mean(centroid_gaps)
            pm = np.mean(percell_gaps)

            results[name] = {
                "cl": cl, "ch": ch,
                "centroid_gaps": centroid_gaps,
                "percell_gaps": percell_gaps,
                "centroid_mean": float(cm),
                "percell_mean": float(pm),
            }
            print(f"    done ({elapsed_c:.0f}s)  centroid={cm:+.4f}  percell={pm:+.4f}", flush=True)

        all_results[set_name] = results

        # Summary for this seed set
        print(f"\n  {'Condition':<15} {'Clip':<12} {'Centroid':>10} {'Per-cell':>10} {'Ratio':>7}")
        print("  " + "-" * 60)
        for name, cl, ch in conditions:
            r = results[name]
            ratio = r["percell_mean"] / (r["centroid_mean"] + 1e-10)
            print(f"  {name:<15} [{cl},{ch}]{'':>4} {r['centroid_mean']:+10.4f} {r['percell_mean']:+10.4f} {ratio:7.2f}x")

        # Ranking comparison
        centroid_ranked = sorted(conditions, key=lambda c: results[c[0]]["centroid_mean"], reverse=True)
        percell_ranked = sorted(conditions, key=lambda c: results[c[0]]["percell_mean"], reverse=True)

        print(f"\n  RANKING (highest MI gap first):")
        print(f"  {'Rank':<6} {'Centroid':<18} {'Per-cell':<18}")
        for i in range(len(conditions)):
            cn = centroid_ranked[i][0]
            pn = percell_ranked[i][0]
            marker = " **" if cn != pn else ""
            print(f"  {i+1:<6} {cn:<18} {pn:<18}{marker}")

        # VERY_NARROW vs CANONICAL comparison
        can_c = results["CANONICAL"]["centroid_gaps"]
        can_p = results["CANONICAL"]["percell_gaps"]
        vn_c = results["VERY_NARROW"]["centroid_gaps"]
        vn_p = results["VERY_NARROW"]["percell_gaps"]

        d_centroid = cohd_paired(can_c, vn_c)
        p_centroid = pval([v - c for v, c in zip(vn_c, can_c)])
        d_percell = cohd_paired(can_p, vn_p)
        p_percell = pval([v - c for v, c in zip(vn_p, can_p)])

        print(f"\n  VERY_NARROW vs CANONICAL:")
        print(f"    Centroid: d={d_centroid:+.3f}, p={p_centroid:.4f}")
        print(f"    Per-cell: d={d_percell:+.3f}, p={p_percell:.4f}")

        vn_centroid_rank = [c[0] for c in centroid_ranked].index("VERY_NARROW") + 1
        vn_percell_rank = [c[0] for c in percell_ranked].index("VERY_NARROW") + 1
        print(f"    VERY_NARROW centroid rank: {vn_centroid_rank}/7")
        print(f"    VERY_NARROW per-cell rank: {vn_percell_rank}/7")

    # === CROSS-SEED-SET COMPARISON ===
    print(f"\n{'=' * 72}")
    print("  CROSS-SEED-SET COMPARISON")
    print(f"{'=' * 72}")
    print(f"\n  {'Condition':<15} {'SetA cent':>10} {'SetA pc':>10} {'SetB cent':>10} {'SetB pc':>10}")
    print("  " + "-" * 60)
    for name, cl, ch in conditions:
        ra = all_results["SetA_10-19"][name]
        rb = all_results["SetB_Entry042"][name]
        print(f"  {name:<15} {ra['centroid_mean']:+10.4f} {ra['percell_mean']:+10.4f} "
              f"{rb['centroid_mean']:+10.4f} {rb['percell_mean']:+10.4f}")

    # === VERDICT ===
    print(f"\n{'=' * 72}")
    print("  VERDICT")
    print(f"{'=' * 72}")

    for set_name, seeds in seed_sets.items():
        results = all_results[set_name]
        centroid_ranked = sorted(conditions, key=lambda c: results[c[0]]["centroid_mean"], reverse=True)
        percell_ranked = sorted(conditions, key=lambda c: results[c[0]]["percell_mean"], reverse=True)

        vn_cr = [c[0] for c in centroid_ranked].index("VERY_NARROW") + 1
        vn_pr = [c[0] for c in percell_ranked].index("VERY_NARROW") + 1

        print(f"\n  {set_name}:")
        print(f"    VERY_NARROW centroid rank: {vn_cr}/7  per-cell rank: {vn_pr}/7")
        print(f"    Centroid best: {centroid_ranked[0][0]}  Per-cell best: {percell_ranked[0][0]}")

        if vn_pr <= 2 and vn_cr >= 5:
            print(f"    => RANK REVERSAL CONFIRMED: per-cell and centroid give opposite conclusions")
        elif vn_pr <= 3 and vn_cr >= 4:
            print(f"    => PARTIAL REVERSAL: per-cell favors VERY_NARROW, centroid does not")
        else:
            print(f"    => CONSISTENT: both methods agree on VERY_NARROW ranking")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
