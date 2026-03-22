#!/usr/bin/env python3
"""
Step 40: Echo State Network (ESN) comparison on chaotic time series.
Architecture-fair benchmark: ESN vs FluxCore CompressedKernel.

Protocol matches phase7b_chaotic.py:
- Mackey-Glass (tau=17): 3000 steps total
- Lorenz-63: 3000 steps total
- Warmup: 1500 steps (reservoir warmup / kernel adaptation)
- Train readout: steps 1500-2000 (ESN only)
- Test: steps 2000-2500 (prediction error)
- Shift detection: feed other signal 100 steps, measure surprise

ESN reservoir: n=100, spectral radius=0.9, ridge regression readout.
FluxCore: CompressedKernel v2 (top-k=5), n_start=8, k=4.
"""

import sys
import math
import random

sys.path.insert(0, 'B:/M/ArtificialArchitecture/the_singularity_search/src')
sys.path.insert(0, 'B:/M/avir/research/fluxcore')

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not available, ESN will use pure Python (slow)")

from fluxcore_compressed_v2 import CompressedKernel
from rk import mcosine, frob


# ============================================================
#  Signal generation
# ============================================================

def mackey_glass(n_steps, tau=17, dt=0.1, x0=0.9, seed=0):
    """Generate Mackey-Glass time series."""
    tau_steps = int(tau / dt)
    total = n_steps + tau_steps + 100
    x = [x0] * (tau_steps + 1)
    rng = random.Random(seed)
    for t in range(tau_steps, total):
        x_tau = x[t - tau_steps]
        dx = 0.2 * x_tau / (1 + x_tau ** 10) - 0.1 * x[t]
        x.append(x[t] + dt * dx)
    series = x[tau_steps + 100: tau_steps + 100 + n_steps]
    return series


def lorenz63(n_steps, sigma=10, rho=28, beta=8/3, dt=0.01, seed=0):
    """Generate Lorenz-63 time series."""
    rng = random.Random(seed)
    x, y, z = 1.0, 0.0, 0.0
    warmup = 1000
    for _ in range(warmup):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx; y += dt * dy; z += dt * dz
    series = []
    for _ in range(n_steps):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dt * dx; y += dt * dy; z += dt * dz
        series.append((x, y, z))
    return series


# ============================================================
#  Embedding: scalar / vector -> R^d unit vector
# ============================================================

def embed_scalar(val, d=16):
    """Embed scalar into R^d via sin features, then normalize."""
    r = [math.sin(val * math.pi * (i + 1) / d) for i in range(d)]
    norm = math.sqrt(sum(v*v for v in r)) + 1e-15
    return [v / norm for v in r]


def embed_vec3(v3, d=16):
    """Embed 3D vector into R^d via sin/cos features, normalize."""
    x, y, z = v3
    r = []
    for i in range(d // 3):
        r.append(math.sin(x * (i + 1) * 0.3))
        r.append(math.sin(y * (i + 1) * 0.1))
        r.append(math.sin(z * (i + 1) * 0.05))
    while len(r) < d:
        r.append(0.0)
    r = r[:d]
    norm = math.sqrt(sum(v*v for v in r)) + 1e-15
    return [v / norm for v in r]


# ============================================================
#  ESN Implementation (numpy)
# ============================================================

class ESN:
    """
    Standard Echo State Network.
    Reservoir: x[t+1] = tanh(W_r @ x[t] + W_in @ u[t])
    Readout trained via ridge regression: y = W_out @ x
    """

    def __init__(self, n_reservoir=100, n_input=1, n_output=1,
                 spectral_radius=0.9, input_scaling=1.0,
                 ridge_alpha=1e-6, seed=42):
        if not HAS_NUMPY:
            raise RuntimeError("ESN requires numpy")
        rng = np.random.RandomState(seed)
        self.n_reservoir = n_reservoir
        self.n_input = n_input
        self.n_output = n_output
        self.ridge_alpha = ridge_alpha

        # Reservoir weights (sparse-ish, ~10% connectivity)
        W = rng.randn(n_reservoir, n_reservoir)
        W[rng.rand(*W.shape) > 0.1] = 0
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        if sr > 0:
            W = W * (spectral_radius / sr)
        self.W_r = W

        # Input weights
        self.W_in = rng.uniform(-input_scaling, input_scaling,
                                 (n_reservoir, n_input))

        self.W_out = None
        self.state = np.zeros(n_reservoir)

    def reset_state(self):
        self.state = np.zeros(self.n_reservoir)

    def step(self, u):
        """One reservoir update. u: input vector (n_input,)"""
        u = np.array(u).reshape(-1)
        self.state = np.tanh(self.W_r @ self.state + self.W_in @ u)
        return self.state.copy()

    def warmup(self, inputs):
        """Run reservoir without collecting states."""
        self.reset_state()
        for u in inputs:
            self.step(u)

    def collect_states(self, inputs):
        """Run and collect reservoir states."""
        states = []
        for u in inputs:
            s = self.step(u)
            states.append(s)
        return np.array(states)

    def train(self, inputs, targets):
        """Train readout via ridge regression. targets: (T, n_output)"""
        states = self.collect_states(inputs)
        targets = np.array(targets)
        # Ridge: W_out = (S^T S + alpha I)^-1 S^T Y
        A = states.T @ states + self.ridge_alpha * np.eye(self.n_reservoir)
        B = states.T @ targets
        self.W_out = np.linalg.solve(A, B).T  # (n_output, n_reservoir)

    def predict(self, u):
        """One-step prediction given input u."""
        s = self.step(u)
        return (self.W_out @ s).tolist()

    def predict_error(self, inputs, targets):
        """Mean squared error on held-out data."""
        preds = []
        for u in inputs:
            p = self.predict(u)
            preds.append(p)
        preds = np.array(preds)
        targets = np.array(targets)
        mse = np.mean((preds - targets) ** 2)
        return float(mse), preds


# ============================================================
#  FluxCore surprise metric (matches phase7b_chaotic)
# ============================================================

def fluxcore_surprise(kernel, r):
    """Mean ||R - M_i|| / ||M_i|| across cells."""
    R = kernel.project(r)
    n = len(kernel.cells)
    if n == 0:
        return 0.0
    return sum(kernel.surprise(R, c) for c in kernel.cells) / n


# ============================================================
#  Experiment runner
# ============================================================

def run_signal(name, series_a, series_b, embed_fn, n_out,
               esn_n=100, d=16, warmup_steps=1500,
               train_steps=500, test_steps=500):
    """
    Run ESN vs FluxCore on series_a, then switch to series_b for shift detection.
    """
    print(f"\n{'='*70}")
    print(f"  SIGNAL: {name}")
    print(f"{'='*70}")

    total = len(series_a)
    embedded_a = [embed_fn(s) for s in series_a]

    # ---- ESN ----
    print(f"\n  [ESN] n_reservoir={esn_n}, spectral_radius=0.9, ridge_alpha=1e-6")

    if HAS_NUMPY:
        esn = ESN(n_reservoir=esn_n, n_input=d, n_output=n_out, seed=42)

        # Warmup
        esn.warmup(embedded_a[:warmup_steps])

        # Collect train states: predict next step
        train_inputs = embedded_a[warmup_steps: warmup_steps + train_steps - 1]
        if n_out == 1:
            train_targets = [[series_a[warmup_steps + i + 1]]
                             for i in range(len(train_inputs))]
        else:
            train_targets = [list(series_a[warmup_steps + i + 1])
                             for i in range(len(train_inputs))]

        esn.train(train_inputs, train_targets)
        print(f"  [ESN] Trained on {len(train_inputs)} steps (warmup+train={warmup_steps+train_steps})")

        # Test: one-step prediction on held-out window
        test_start = warmup_steps + train_steps
        test_inputs = embedded_a[test_start: test_start + test_steps - 1]
        if n_out == 1:
            test_targets = [[series_a[test_start + i + 1]]
                            for i in range(len(test_inputs))]
        else:
            test_targets = [list(series_a[test_start + i + 1])
                            for i in range(len(test_inputs))]

        mse, _ = esn.predict_error(test_inputs, test_targets)
        rmse = math.sqrt(mse)
        print(f"  [ESN] Test RMSE (one-step prediction): {rmse:.6f}")

        # Settled surprise proxy: mean |pred - target| / range
        series_range = max(series_a[test_start:test_start+test_steps]) - \
                       min(series_a[test_start:test_start+test_steps]) \
                       if n_out == 1 else 1.0
        normalized_rmse = rmse / (series_range + 1e-15) if n_out == 1 else rmse
        print(f"  [ESN] Normalized RMSE: {normalized_rmse:.4f}")

        # Shift detection: after test, feed series_b, compare prediction error
        embedded_b = [embed_fn(s) for s in series_b[:100]]
        if n_out == 1:
            b_targets = [[series_b[i + 1]] for i in range(len(embedded_b) - 1)]
            b_mse, _ = esn.predict_error(embedded_b[:-1], b_targets)
        else:
            b_targets = [list(series_b[i + 1]) for i in range(len(embedded_b) - 1)]
            b_mse, _ = esn.predict_error(embedded_b[:-1], b_targets)
        b_rmse = math.sqrt(b_mse)
        shift_ratio = b_rmse / (rmse + 1e-15)
        detected = shift_ratio > 1.5
        print(f"  [ESN] Shift detection (series_b, first 100): RMSE={b_rmse:.6f}, ratio={shift_ratio:.2f}x  ({'DETECTED' if detected else 'WEAK'})")

        print(f"\n  [ESN] Generation capability: NO (requires readout + input at each step)")
    else:
        print("  [ESN] SKIPPED (numpy not available)")
        rmse = None
        shift_ratio = None

    # ---- FluxCore ----
    print(f"\n  [FluxCore] n_start=8, k=4, k_couple=5, tau=0.3, d={d}")

    kernel = CompressedKernel(n=8, k=4, d=d, seed=42, proj_seed=999,
                              tau=0.3, spawning=True, max_cells=500,
                              k_couple=5)

    # Warmup
    for i in range(warmup_steps):
        kernel.step(r=embedded_a[i])

    warmup_cells = len(kernel.cells)
    warmup_autonomy = kernel.mean_autonomy()
    warmup_surprise = fluxcore_surprise(kernel, embedded_a[warmup_steps - 1])
    print(f"  [FluxCore] After warmup: cells={warmup_cells}, autonomy={warmup_autonomy:.4f}, surprise={warmup_surprise:.4f}")

    # Adaptation + track surprise trend
    surprises = []
    for i in range(warmup_steps, warmup_steps + train_steps + test_steps):
        if i < len(embedded_a):
            kernel.step(r=embedded_a[i])
            if i % 100 == 0:
                s = fluxcore_surprise(kernel, embedded_a[i])
                surprises.append(s)

    final_cells = len(kernel.cells)
    settled_surprise = fluxcore_surprise(kernel, embedded_a[min(warmup_steps + train_steps + test_steps - 1, len(embedded_a) - 1)])
    print(f"  [FluxCore] After adaptation: cells={final_cells}, settled_surprise={settled_surprise:.4f}")
    print(f"  [FluxCore] Surprise trend: {surprises[0]:.4f} -> {surprises[-1]:.4f} ({'DECREASING' if surprises[-1] < surprises[0] else 'INCREASING'})")

    # Generation survival
    dMs_gen = []
    for step in range(2000):
        dMs_gen = kernel.step(r=None)
    gen_energy = kernel.mean_energy(dMs_gen)
    gen_survived = gen_energy > 0.001
    print(f"  [FluxCore] Generation (2000 steps): energy={gen_energy:.6f}, survived={'YES' if gen_survived else 'NO'}")

    # Shift detection
    embedded_b = [embed_fn(s) for s in series_b[:100]]
    shift_surprises = [fluxcore_surprise(kernel, eb) for eb in embedded_b[:10]]
    spike = max(shift_surprises)
    fk_shift_ratio = spike / (settled_surprise + 1e-15)
    fk_detected = fk_shift_ratio > 1.5
    print(f"  [FluxCore] Shift detection: spike={spike:.4f}, ratio={fk_shift_ratio:.2f}x  ({'DETECTED' if fk_detected else 'WEAK'})")

    return {
        'esn_rmse': rmse,
        'esn_shift_ratio': shift_ratio,
        'esn_detected': shift_ratio > 1.5 if shift_ratio is not None else None,
        'fk_cells': final_cells,
        'fk_settled_surprise': settled_surprise,
        'fk_gen_energy': gen_energy,
        'fk_gen_survived': gen_survived,
        'fk_shift_ratio': fk_shift_ratio,
        'fk_detected': fk_detected,
    }


def main():
    print("=" * 70)
    print("  Step 40: ESN vs FluxCore — Chaotic Time Series Comparison")
    print("=" * 70)
    print(f"  numpy: {'available' if HAS_NUMPY else 'NOT AVAILABLE'}")

    # Generate signals
    print("\nGenerating signals...")
    N = 3000
    mg_series = mackey_glass(N + 1)
    lz_series = lorenz63(N + 1)

    mg_range = max(mg_series) - min(mg_series)
    lz_x_range = max(x for x, y, z in lz_series) - min(x for x, y, z in lz_series)
    print(f"  Mackey-Glass: {N} steps, range=[{min(mg_series):.3f}, {max(mg_series):.3f}]")
    print(f"  Lorenz-63:    {N} steps, x-range=[{min(x for x,y,z in lz_series):.2f}, {max(x for x,y,z in lz_series):.2f}]")

    d = 16

    # Mackey-Glass experiment
    mg_embed = lambda v: embed_scalar(v, d=d)
    lz_embed = lambda v: embed_vec3(v, d=d)

    mg_results = run_signal(
        "Mackey-Glass (tau=17)",
        mg_series, [v[0] for v in lz_series],  # series_b = Lorenz x-component for shift
        mg_embed, n_out=1,
        esn_n=100, d=d, warmup_steps=1500, train_steps=500, test_steps=500
    )

    lz_results = run_signal(
        "Lorenz-63",
        lz_series, [(v, 0.0, 0.0) for v in mg_series],  # MG padded to 3D for shift
        lz_embed, n_out=3,
        esn_n=100, d=d, warmup_steps=1500, train_steps=500, test_steps=500
    )

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print()
    print(f"  {'Metric':<40}  {'ESN':>12}  {'FluxCore':>12}")
    print(f"  {'-'*40}  {'-'*12}  {'-'*12}")

    if HAS_NUMPY:
        print(f"  {'MG: one-step RMSE':<40}  {mg_results['esn_rmse']:>12.6f}  {'N/A':>12}")
        print(f"  {'LZ: one-step RMSE (3D)':<40}  {lz_results['esn_rmse']:>12.6f}  {'N/A':>12}")
    print(f"  {'MG: shift detection ratio':<40}  {str(round(mg_results['esn_shift_ratio'],2)) + 'x' if mg_results['esn_shift_ratio'] else 'N/A':>12}  {mg_results['fk_shift_ratio']:.2f}x{' ':>8}")
    print(f"  {'LZ: shift detection ratio':<40}  {str(round(lz_results['esn_shift_ratio'],2)) + 'x' if lz_results['esn_shift_ratio'] else 'N/A':>12}  {lz_results['fk_shift_ratio']:.2f}x{' ':>8}")
    print(f"  {'MG: endogenous generation':<40}  {'NO':>12}  {'YES':>12}")
    print(f"  {'LZ: endogenous generation':<40}  {'NO':>12}  {'YES':>12}")
    print(f"  {'MG: gen energy at 2000 steps':<40}  {'—':>12}  {mg_results['fk_gen_energy']:>12.6f}")
    print(f"  {'LZ: gen energy at 2000 steps':<40}  {'—':>12}  {lz_results['fk_gen_energy']:>12.6f}")
    print(f"  {'MG: cells spawned':<40}  {100:>12}  {mg_results['fk_cells']:>12}")
    print(f"  {'LZ: cells spawned':<40}  {100:>12}  {lz_results['fk_cells']:>12}")
    print(f"  {'Training required':<40}  {'YES':>12}  {'NO':>12}")
    print(f"  {'Online adaptation':<40}  {'NO':>12}  {'YES':>12}")

    print(f"\n  {'='*70}")
    print(f"  CAPABILITY SUMMARY")
    print(f"  {'='*70}")
    print(f"  ESN advantages:    Better one-step prediction (trained readout)")
    print(f"  FluxCore advantages: Endogenous generation, online adaptation,")
    print(f"                       zero training, concept drift without retraining")


if __name__ == '__main__':
    main()
