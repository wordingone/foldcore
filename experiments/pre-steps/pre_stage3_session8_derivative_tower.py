#!/usr/bin/env python3
"""
STAGE 3 SESSION 8: Delta_rz + Derivative Tower Autocorrelation Analysis

Adds derivative tower autocorrelation measurement to understand signal structure:
- resp_z autocorrelation (order 0)
- delta_rz autocorrelation (order 1: 1st derivative)
- delta_delta_rz autocorrelation (order 2: 2nd derivative)

Question: Does the derivative tower collapse? This tells us how many stages
of adaptation we can drive with the resp_z signal family.
"""

import math
import random
import time
from scipy import stats

D = 12
NC = 6
W = 72

# Validation set
SEEDS = [42, 137, 2024, 7, 314, 1618, 2718, 3141, 9999, 31337]
KS = [4, 6, 8, 10]
BIRTH_SEED = 42
NOVEL_SEED_BASE = 99999


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
    """Stage 3: Delta_rz adaptive eta with derivative tower tracking."""

    def __init__(self, seed=42, alive=False, eta=0.0003, adaptive_eta=False,
                 track_derivatives=False):
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
        self.track_derivatives = track_derivatives
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
            self.prev_resp_z = [[0.0] * D for _ in range(NC)]
            self.prev_delta_rz = [[0.0] * D for _ in range(NC)]
            self.has_prev = False
            self.has_prev_delta = False
        else:
            self.eta_scalar = eta

        self.eta_lo = 0.00005
        self.eta_hi = 0.003

        # Derivative tower tracking
        if track_derivatives:
            self.resp_z_history = []  # [(t, [(i, k, value)])]
            self.delta_rz_history = []
            self.delta_delta_rz_history = []

    def compute_autocorrelations(self):
        """Compute lag-1 autocorrelation for each derivative order."""
        if not self.track_derivatives:
            return None

        def lag1_autocorr(history):
            """Compute per-cell lag-1 autocorrelation from history."""
            if len(history) < 2:
                return None

            # Organize by cell
            cell_series = {}
            for t, cells in history:
                for i, k, val in cells:
                    key = (i, k)
                    if key not in cell_series:
                        cell_series[key] = []
                    cell_series[key].append(val)

            # Compute lag-1 autocorrelation for each cell
            autocorrs = []
            for key, series in cell_series.items():
                if len(series) < 2:
                    continue
                # Pearson correlation between series[:-1] and series[1:]
                x = series[:-1]
                y = series[1:]
                n = len(x)
                if n < 2:
                    continue
                mx = sum(x) / n
                my = sum(y) / n
                sx = math.sqrt(sum((xi - mx)**2 for xi in x) / n)
                sy = math.sqrt(sum((yi - my)**2 for yi in y) / n)
                if sx < 1e-10 or sy < 1e-10:
                    continue
                r = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n * sx * sy)
                autocorrs.append(r)

            if not autocorrs:
                return None

            return {
                'mean': sum(autocorrs) / len(autocorrs),
                'std': math.sqrt(sum((r - sum(autocorrs)/len(autocorrs))**2 for r in autocorrs) / len(autocorrs)),
                'min': min(autocorrs),
                'max': max(autocorrs),
                'n_cells': len(autocorrs),
            }

        return {
            'resp_z': lag1_autocorr(self.resp_z_history),
            'delta_rz': lag1_autocorr(self.delta_rz_history),
            'delta_delta_rz': lag1_autocorr(self.delta_delta_rz_history),
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

            # Track resp_z
            resp_z_cells = []
            delta_rz_cells = []
            delta_delta_rz_cells = []

            for i in range(NC):
                for k in range(D):
                    resp_z = (response[i][k] - overall_mean) / overall_std

                    if self.adaptive_eta:
                        eta_ik = self.eta[i][k]
                    else:
                        eta_ik = self.eta_scalar

                    col_mean = sum(self.alpha[j][k] for j in range(NC)) / NC
                    dev = self.alpha[i][k] - col_mean

                    # Alpha plasticity
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

                    # Eta plasticity + derivative tracking
                    if self.adaptive_eta:
                        if self.has_prev:
                            delta_rz = resp_z - self.prev_resp_z[i][k]
                            push_e = 0.1 * eta_ik * math.tanh(delta_rz)

                            old_e = self.eta[i][k]
                            self.eta[i][k] += push_e
                            self.eta[i][k] = max(self.eta_lo,
                                                 min(self.eta_hi, self.eta[i][k]))
                            self.total_eta_shift += abs(self.eta[i][k] - old_e)

                            if self.track_derivatives:
                                delta_rz_cells.append((i, k, delta_rz))

                            # Second derivative
                            if self.has_prev_delta:
                                delta_delta_rz = delta_rz - self.prev_delta_rz[i][k]
                                if self.track_derivatives:
                                    delta_delta_rz_cells.append((i, k, delta_delta_rz))

                            self.prev_delta_rz[i][k] = delta_rz

                        self.prev_resp_z[i][k] = resp_z

                        if self.track_derivatives:
                            resp_z_cells.append((i, k, resp_z))

            if self.adaptive_eta:
                if not self.has_prev:
                    self.has_prev = True
                elif not self.has_prev_delta:
                    self.has_prev_delta = True

            if self.track_derivatives and resp_z_cells:
                t = len(self.resp_z_history)
                self.resp_z_history.append((t, resp_z_cells))
                if delta_rz_cells:
                    self.delta_rz_history.append((t, delta_rz_cells))
                if delta_delta_rz_cells:
                    self.delta_delta_rz_history.append((t, delta_delta_rz_cells))

        # Attention
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


def main():
    print("=" * 80)
    print("  DERIVATIVE TOWER AUTOCORRELATION ANALYSIS")
    print("=" * 80)
    print("\nRunning delta_rz adaptive eta with derivative tracking...")
    print("Seeds: [42, 137, 2024]")
    print()

    t0 = time.time()

    # Run with derivative tracking on 3 seeds
    test_seeds = [42, 137, 2024]
    autocorr_results = []

    for seed in test_seeds:
        print(f"\n--- Seed {seed} ---")
        org = Organism(seed=BIRTH_SEED, alive=True, eta=0.0003,
                       adaptive_eta=True, track_derivatives=True)

        # Run a full sequence
        sigs = make_signals(8, seed=BIRTH_SEED + 1600)
        order = list(range(8))
        run_sequence(org, order, sigs, seed, trial=0)

        # Compute autocorrelations
        ac = org.compute_autocorrelations()
        autocorr_results.append(ac)

        print(f"\nAutocorrelation results:")
        if ac['resp_z']:
            print(f"  resp_z (order 0):")
            print(f"    mean={ac['resp_z']['mean']:+.4f} std={ac['resp_z']['std']:.4f}")
            print(f"    min={ac['resp_z']['min']:+.4f} max={ac['resp_z']['max']:+.4f}")
            print(f"    n_cells={ac['resp_z']['n_cells']}")

        if ac['delta_rz']:
            print(f"  delta_rz (order 1 - 1st derivative):")
            print(f"    mean={ac['delta_rz']['mean']:+.4f} std={ac['delta_rz']['std']:.4f}")
            print(f"    min={ac['delta_rz']['min']:+.4f} max={ac['delta_rz']['max']:+.4f}")
            print(f"    n_cells={ac['delta_rz']['n_cells']}")

        if ac['delta_delta_rz']:
            print(f"  delta_delta_rz (order 2 - 2nd derivative):")
            print(f"    mean={ac['delta_delta_rz']['mean']:+.4f} std={ac['delta_delta_rz']['std']:.4f}")
            print(f"    min={ac['delta_delta_rz']['min']:+.4f} max={ac['delta_delta_rz']['max']:+.4f}")
            print(f"    n_cells={ac['delta_delta_rz']['n_cells']}")

    # Aggregate across seeds
    print("\n" + "=" * 80)
    print("AGGREGATE RESULTS (across seeds):")
    print("=" * 80)

    for order_name in ['resp_z', 'delta_rz', 'delta_delta_rz']:
        means = [ac[order_name]['mean'] for ac in autocorr_results if ac[order_name]]
        if means:
            avg_mean = sum(means) / len(means)
            print(f"\n{order_name}:")
            print(f"  Mean autocorrelation: {avg_mean:+.4f}")
            print(f"  Range: [{min(means):+.4f}, {max(means):+.4f}]")

            if order_name == 'resp_z':
                print(f"  Status: {'Confirmed ~0.98' if avg_mean > 0.95 else 'UNEXPECTED - should be ~0.98'}")
            elif order_name == 'delta_rz':
                if avg_mean > 0.5:
                    print(f"  Status: STRONG - usable for Stage 3 adaptation")
                elif avg_mean > 0.3:
                    print(f"  Status: MODERATE - borderline usable")
                else:
                    print(f"  Status: WEAK - tower degrading")
            elif order_name == 'delta_delta_rz':
                if avg_mean > 0.3:
                    print(f"  Status: STRUCTURED - hope for Stages 4+")
                elif avg_mean > 0.1:
                    print(f"  Status: WEAK STRUCTURE - marginal")
                else:
                    print(f"  Status: COLLAPSED - tower ends at order 1")

    print(f"\n" + "=" * 80)
    print("STRATEGIC IMPLICATIONS:")
    print("=" * 80)

    delta_rz_mean = sum(ac['delta_rz']['mean'] for ac in autocorr_results if ac['delta_rz']) / len([ac for ac in autocorr_results if ac['delta_rz']])
    delta2_rz_mean = sum(ac['delta_delta_rz']['mean'] for ac in autocorr_results if ac['delta_delta_rz']) / len([ac for ac in autocorr_results if ac['delta_delta_rz']])

    print(f"\ndelta_rz autocorr = {delta_rz_mean:+.4f}")
    print(f"delta_delta_rz autocorr = {delta2_rz_mean:+.4f}")

    if delta_rz_mean > 0.5:
        print("\n✓ delta_rz has strong temporal structure (>0.5)")
        print("  Stage 3 (eta adaptation via delta_rz) is theoretically viable")
        print("  BUT: Task #2 showed it doesn't improve performance (p=0.999)")
        print("  Conclusion: Signal exists but doesn't translate to gains")
    else:
        print("\n✗ delta_rz has weak temporal structure (<0.5)")
        print("  Stage 3 failure may be due to signal degradation")

    if delta2_rz_mean > 0.3:
        print("\n✓ delta_delta_rz has structure (>0.3)")
        print("  Derivative tower extends to order 2")
        print("  Stages 4+ may be reachable via 2nd derivative signals")
    else:
        print("\n✗ delta_delta_rz near zero (<0.3)")
        print("  Derivative tower collapses at order 2")
        print("  resp_z family can only drive 2 stages of adaptation")
        print("  Need fundamentally different signal for Stages 4+")

    print(f"\nRuntime: {time.time() - t0:.0f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
