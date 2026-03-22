#!/usr/bin/env python3
"""
STAGE 3 EXPERIMENT B: Second-Order Delta_rz Eta Adaptation

Eta adapts based on delta_rz = resp_z(t) - resp_z(t-1).
If delta_rz > 0, responsiveness IMPROVED after last alpha update,
so increase eta (the last learning rate was good).
If delta_rz < 0, responsiveness degraded, so decrease eta.

Control B showed resp_z autocorrelation = 0.98 — delta_rz is
a clean, informative signal about the CHANGE in responsiveness.

Frozen frame accounting:
  Removed: eta as global scalar
  Added:   0 (reuses drift multiplier 0.1 as meta_rate)
  Net:     7 -> 6 = -1
"""

import math
import random
import time

D  = 12
NC = 6
W  = 72


def vcosine(a, b):
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for ai, bi in zip(a, b):
        dot += ai * bi
        na2 += ai * ai
        nb2 += bi * bi
    na = math.sqrt(na2 + 1e-15)
    nb = math.sqrt(nb2 + 1e-15)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def vnorm(v):
    return math.sqrt(sum(vi * vi for vi in v) + 1e-15)


class Organism:
    """Stage 3-B: Delta_rz adaptive eta."""

    def __init__(self, seed=42, alive=False, eta=0.0003, adaptive_eta=False):
        self.beta = 0.5
        self.gamma = 0.9
        self.eps = 0.15
        self.tau = 0.3
        self.delta = 0.35
        self.noise = 0.005
        self.clip = 4.0
        self.seed = seed
        self.alive = alive
        self.adaptive_eta = adaptive_eta
        self.total_alpha_shift = 0.0
        self.total_eta_shift = 0.0

        random.seed(seed)
        self.alpha = [
            [1.1 + 0.7 * (random.random() * 2 - 1) for _ in range(D)]
            for _ in range(NC)
        ]

        if adaptive_eta:
            self.eta = [
                [0.0001 + random.random() * 0.0009 for _ in range(D)]
                for _ in range(NC)
            ]
            # Previous resp_z for delta computation
            self.prev_resp_z = [
                [0.0 for _ in range(D)]
                for _ in range(NC)
            ]
            self.has_prev = False
        else:
            self.eta_scalar = eta

        self.eta_lo = 0.00005
        self.eta_hi = 0.003

    def alpha_flat(self):
        return [a for row in self.alpha for a in row]

    def eta_flat(self):
        if self.adaptive_eta:
            return [e for row in self.eta for e in row]
        return [self.eta_scalar] * (NC * D)

    def eta_stats(self):
        vals = self.eta_flat()
        mn = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mn)**2 for v in vals) / len(vals))
        return {
            'mean': mn, 'std': std,
            'min': min(vals), 'max': max(vals),
            'at_lo': sum(1 for v in vals if v <= self.eta_lo + 1e-10),
            'at_hi': sum(1 for v in vals if v >= self.eta_hi - 1e-10),
        }

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
            overall_std = math.sqrt(
                sum((r - overall_mean) ** 2 for r in all_resp) / len(all_resp)
            ) + 1e-10

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    if self.adaptive_eta:
                        eta_ik = self.eta[i][k]
                    else:
                        eta_ik = self.eta_scalar

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    # ── Alpha plasticity (same as canonical) ────
                    if abs(dev) < 0.01:
                        push = eta_ik * 0.3 * random.gauss(0, 1.0)
                    elif resp_z > 0:
                        direction = 1.0 if dev > 0 else -1.0
                        push = eta_ik * math.tanh(resp_z) * direction * 0.5
                    else:
                        push = eta_ik * 0.1 * random.gauss(0, 1.0)

                    old_a = self.alpha[i][k]
                    self.alpha[i][k] += push
                    self.alpha[i][k] = max(0.3, min(1.8, self.alpha[i][k]))
                    self.total_alpha_shift += abs(self.alpha[i][k] - old_a)

                    # ── Eta plasticity (delta_rz second-order) ──
                    if self.adaptive_eta and self.has_prev:
                        delta_rz = resp_z - self.prev_resp_z[i][k]
                        # delta_rz > 0: responsiveness improved -> increase eta
                        # delta_rz < 0: responsiveness degraded -> decrease eta
                        # Reuse existing drift multiplier (0.1) as meta_rate
                        push_e = 0.1 * eta_ik * math.tanh(delta_rz)

                        old_e = self.eta[i][k]
                        self.eta[i][k] += push_e
                        self.eta[i][k] = max(self.eta_lo,
                                             min(self.eta_hi, self.eta[i][k]))
                        self.total_eta_shift += abs(self.eta[i][k] - old_e)

                    if self.adaptive_eta:
                        self.prev_resp_z[i][k] = resp_z

            if self.adaptive_eta:
                self.has_prev = True

        # ── ATTENTION ────────────────────────────────────────
        weights = []
        for i in range(NC):
            raw = []
            for j in range(NC):
                if i == j:
                    raw.append(-1e10)
                else:
                    d = sum(xs[i][kk] * xs[j][kk] for kk in range(D))
                    raw.append(d / (D * self.tau))
            mx = max(raw)
            exps = [math.exp(min(v - mx, 50)) for v in raw]
            s = sum(exps) + 1e-15
            weights.append([e / s for e in exps])

        # ── STATE UPDATE ─────────────────────────────────────
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
                    for kk in range(D):
                        pull[kk] += weights[i][j] * (phi_bare[j][kk] - phi_bare[i][kk])
                p = [p[kk] + plast * self.eps * pull[kk] for kk in range(D)]

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
    perms = []
    seen = set()
    perms.append(tuple(base))
    seen.add(tuple(base))
    perms.append(tuple(reversed(base)))
    seen.add(tuple(reversed(base)))
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


def measure_gap(org, signals, k, seed, n_perm=4, n_trials=3):
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


def run():
    print("=" * W)
    print("  STAGE 3 EXPERIMENT B: DELTA_RZ ETA ADAPTATION")
    print("  eta adapts via second-order signal: delta_rz = rz(t) - rz(t-1)")
    print("  Frozen frame: 7 -> 6 (net -1)")
    print("=" * W)

    t0 = time.time()
    SEED = 42
    test_ks = [4, 6, 8, 10]
    test_seeds = [42, 137, 2024]

    conditions = {
        'STILL':      dict(alive=False, eta=0.0003, adaptive_eta=False),
        'ALIVE-s2':   dict(alive=True,  eta=0.0003, adaptive_eta=False),
        'ALIVE-s3-B': dict(alive=True,  eta=0.0003, adaptive_eta=True),
    }

    # ── Main benchmark ────────────────────────────────────────
    results = {}
    for cname, ckwargs in conditions.items():
        print(f"\n  --- {cname} ---")
        cond_results = {}
        for k in test_ks:
            sigs = make_signals(k, seed=SEED + k * 200)
            gaps = []
            for s in test_seeds:
                org = Organism(seed=SEED, **ckwargs)
                g = measure_gap(org, sigs, k, s)
                gaps.append(g)
            avg = sum(gaps) / len(gaps)
            cond_results[k] = {'avg': avg, 'gaps': gaps}
            print(f"  K={k:>2}: avg={avg:+.4f} "
                  f"[{min(gaps):+.3f}..{max(gaps):+.3f}]", flush=True)
        overall = sum(cond_results[k]['avg'] for k in test_ks) / len(test_ks)
        results[cname] = {'per_k': cond_results, 'overall': overall}
        print(f"  overall: {overall:+.4f}")

    # ── Novel signals ─────────────────────────────────────────
    print(f"\n  --- Novel Signal Test ---")
    novel = {}
    for cname, ckwargs in conditions.items():
        gaps = []
        for wi in range(6):
            for k in [6, 8]:
                nsigs = make_signals(k, seed=99999 + wi * 37 + k)
                ts = 77 + wi * 13 + k
                org = Organism(seed=SEED, **ckwargs)
                g = measure_gap(org, nsigs, k, ts)
                gaps.append(g)
        avg = sum(gaps) / len(gaps)
        novel[cname] = avg
        print(f"  {cname:>12}: novel avg={avg:+.4f}", flush=True)

    # ── Eta distribution analysis ─────────────────────────────
    print(f"\n  --- Eta Distribution (post-run) ---")
    for s in test_seeds:
        org = Organism(seed=SEED, alive=True, adaptive_eta=True)
        sigs = make_signals(8, seed=SEED + 1600)
        order = list(range(8))
        run_sequence(org, order, sigs, s, trial=0)
        es = org.eta_stats()
        print(f"  seed={s}: mean={es['mean']:.6f} std={es['std']:.6f} "
              f"min={es['min']:.6f} max={es['max']:.6f} "
              f"at_lo={es['at_lo']} at_hi={es['at_hi']}")

    # ── Verdict ───────────────────────────────────────────────
    print(f"\n{'=' * W}")
    print(f"  EXPERIMENT B VERDICT")
    print(f"{'=' * W}\n")

    s3 = results['ALIVE-s3-B']['overall']
    s2 = results['ALIVE-s2']['overall']
    st = results['STILL']['overall']

    print(f"  STILL:      {st:+.4f}")
    print(f"  ALIVE-s2:   {s2:+.4f} (delta vs STILL: {s2 - st:+.4f})")
    print(f"  ALIVE-s3-B: {s3:+.4f} (delta vs STILL: {s3 - st:+.4f})")
    print(f"  s3 vs s2:   {s3 - s2:+.4f}")

    print(f"\n  Novel:")
    print(f"  STILL:      {novel['STILL']:+.4f}")
    print(f"  ALIVE-s2:   {novel['ALIVE-s2']:+.4f}")
    print(f"  ALIVE-s3-B: {novel['ALIVE-s3-B']:+.4f}")
    print(f"  s3 vs s2:   {novel['ALIVE-s3-B'] - novel['ALIVE-s2']:+.4f}")

    s3_beats_s2 = s3 > s2
    s3_beats_still = s3 > st
    s2_beats_still = s2 > st
    novel_s3_beats = novel['ALIVE-s3-B'] > novel['ALIVE-s2']
    ground_truth = s3_beats_still

    print(f"\n  Stage 3 criteria:")
    print(f"    s3 > s2 (main):    {'PASS' if s3_beats_s2 else 'FAIL'}")
    print(f"    s3 > s2 (novel):   {'PASS' if novel_s3_beats else 'FAIL'}")
    print(f"    s2 > STILL:        {'PASS' if s2_beats_still else 'FAIL'}")
    print(f"    Ground truth:      {'PASS' if ground_truth else 'FAIL'}")

    if s3_beats_s2 and ground_truth:
        print(f"\n  STAGE 3-B: PASS")
        print(f"  Delta_rz eta beats fixed eta. Frozen frame reduced 7->6.")
    elif s3_beats_still and not s3_beats_s2:
        print(f"\n  STAGE 3-B: PARTIAL")
        print(f"  Delta_rz eta beats STILL but not fixed eta.")
    else:
        print(f"\n  STAGE 3-B: FAIL")
        print(f"  Delta_rz eta does not improve over fixed eta.")

    print(f"\n  Runtime: {time.time() - t0:.0f}s")
    print(f"  {'=' * W}")


if __name__ == '__main__':
    run()
