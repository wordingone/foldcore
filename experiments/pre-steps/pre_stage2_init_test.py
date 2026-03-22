#!/usr/bin/env python3
"""
Task #19: Randomized Initialization Test

Tests whether beta/gamma convergence to 0.50/0.90 is a real attractor
or just initialization stickiness.

5 test points:
1. (0.1, 0.1) - low corner
2. (1.0, 0.5) - mid range
3. (0.3, 1.5) - grid-search optimum
4. (1.5, 0.1) - high beta, low gamma
5. (0.05, 2.0) - boundary extremes

Plus fine grid search around 0.50/0.90.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from living_seed_stage2_gradient import OrganismStage2, make_signals, measure_mi
import random
import math
import statistics

def run_init_test(beta_init, gamma_init, seed=42, k=8):
    """Run Stage 2 experiment from given beta/gamma initialization."""
    random.seed(seed)

    # Generate signal set
    signals = make_signals(k, seed)

    # Create organism with custom initialization
    org = OrganismStage2(seed=seed, alive=True, stage2=True, eta=0.0003)
    org.beta = beta_init
    org.gamma = gamma_init

    # Initialize state matrix (NC x D)
    from living_seed_stage2_gradient import NC, D
    xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

    # Warm-up phase - bare dynamics
    for _ in range(200):
        xs = org.step(xs)

    # Training phase - signal presentation
    for _ in range(600):
        sig_idx = random.choice(list(signals.keys()))
        sig = [signals[sig_idx][k] + random.gauss(0, 0.05) for k in range(D)]
        xs = org.step(xs, sig)

    # Record final converged values
    final_beta = org.beta
    final_gamma = org.gamma
    beta_shift = org.total_beta_shift
    gamma_shift = org.total_gamma_shift

    # Measure MI (reduced trials for speed)
    final_mi = measure_mi(org, signals, k, seed, n_trials=10)

    return {
        'beta_init': beta_init,
        'gamma_init': gamma_init,
        'beta_final': final_beta,
        'gamma_final': final_gamma,
        'beta_shift': beta_shift,
        'gamma_shift': gamma_shift,
        'mi_novel': final_mi
    }


def fine_grid_search(center_beta=0.5, center_gamma=0.9, radius=0.2, grid_size=20, k=8):
    """Fine grid search around suspected attractor."""
    from living_seed_stage2_gradient import NC, D

    print(f"\n{'='*70}")
    print(f"FINE GRID SEARCH: {grid_size}x{grid_size} around ({center_beta:.2f}, {center_gamma:.2f})")
    print(f"{'='*70}\n")

    beta_min = center_beta - radius
    beta_max = center_beta + radius
    gamma_min = center_gamma - radius
    gamma_max = center_gamma + radius

    beta_step = (beta_max - beta_min) / (grid_size - 1)
    gamma_step = (gamma_max - gamma_min) / (grid_size - 1)

    best_mi = -999.0
    best_beta = 0.0
    best_gamma = 0.0

    grid_results = []

    # Generate fixed signal set
    signals = make_signals(k, seed=42)

    for i in range(grid_size):
        beta = beta_min + i * beta_step
        for j in range(grid_size):
            gamma = gamma_min + j * gamma_step

            # Run with FROZEN beta/gamma (stage2=False)
            org = OrganismStage2(seed=42, alive=True, stage2=False, eta=0.0003)
            org.beta = beta
            org.gamma = gamma

            # Initialize state
            xs = [[random.gauss(0, 0.5) for _ in range(D)] for _ in range(NC)]

            # Warm-up
            for _ in range(200):
                xs = org.step(xs)

            # Training
            for _ in range(600):
                sig_idx = random.choice(list(signals.keys()))
                sig = [signals[sig_idx][kk] + random.gauss(0, 0.05) for kk in range(D)]
                xs = org.step(xs, sig)

            mi = measure_mi(org, signals, k, seed=42, n_trials=10)
            grid_results.append((beta, gamma, mi))

            if mi > best_mi:
                best_mi = mi
                best_beta = beta
                best_gamma = gamma

    print(f"Grid search complete: {len(grid_results)} configurations tested")
    print(f"Best static configuration: beta={best_beta:.4f}, gamma={best_gamma:.4f}, MI={best_mi:.4f}")

    return best_beta, best_gamma, best_mi, grid_results


def main():
    print("="*70)
    print("TASK #19: RANDOMIZED INITIALIZATION TEST")
    print("="*70)
    print("Testing whether beta/gamma convergence is a real attractor")
    print("or initialization stickiness.\n")

    # Define 5 test initialization points
    test_points = [
        (0.1, 0.1, "low corner"),
        (1.0, 0.5, "mid range"),
        (0.3, 1.5, "grid-search optimum from previous run"),
        (1.5, 0.1, "high beta, low gamma"),
        (0.05, 2.0, "boundary extremes")
    ]

    results = []

    # Run each initialization across multiple seeds
    NUM_SEEDS = 3  # Reduced from 5 for faster execution

    for beta_init, gamma_init, description in test_points:
        print(f"\n{'-'*70}")
        print(f"INITIALIZATION: beta={beta_init:.2f}, gamma={gamma_init:.2f} ({description})")
        print(f"{'-'*70}\n")

        seed_results = []

        for seed_idx in range(NUM_SEEDS):
            seed = 100 + seed_idx * 10
            result = run_init_test(beta_init, gamma_init, seed=seed)
            seed_results.append(result)

            print(f"Seed {seed}: beta {beta_init:.2f} -> {result['beta_final']:.4f} "
                  f"(shift={result['beta_shift']:+.4f}), "
                  f"gamma {gamma_init:.2f} -> {result['gamma_final']:.4f} "
                  f"(shift={result['gamma_shift']:+.4f}), "
                  f"MI={result['mi_novel']:.4f}")

        # Statistics across seeds
        final_betas = [r['beta_final'] for r in seed_results]
        final_gammas = [r['gamma_final'] for r in seed_results]
        final_mis = [r['mi_novel'] for r in seed_results]

        mean_beta = statistics.mean(final_betas)
        std_beta = statistics.stdev(final_betas) if len(final_betas) > 1 else 0.0
        mean_gamma = statistics.mean(final_gammas)
        std_gamma = statistics.stdev(final_gammas) if len(final_gammas) > 1 else 0.0
        mean_mi = statistics.mean(final_mis)

        print(f"\nCONVERGED TO: beta={mean_beta:.4f}±{std_beta:.4f}, "
              f"gamma={mean_gamma:.4f}±{std_gamma:.4f}, MI={mean_mi:.4f}")

        results.append({
            'init': (beta_init, gamma_init),
            'description': description,
            'final_beta': mean_beta,
            'final_gamma': mean_gamma,
            'std_beta': std_beta,
            'std_gamma': std_gamma,
            'mean_mi': mean_mi
        })

    # Summary
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Init (beta, gamma)':<30} {'Converged (beta, gamma)':<30} {'MI':<10}")
    print(f"{'-'*70}")

    for r in results:
        init_str = f"({r['init'][0]:.2f}, {r['init'][1]:.2f})"
        final_str = f"({r['final_beta']:.4f}±{r['std_beta']:.4f}, {r['final_gamma']:.4f}±{r['std_gamma']:.4f})"
        print(f"{init_str:<30} {final_str:<30} {r['mean_mi']:.4f}")

    # Check if all converged to same attractor
    all_final_betas = [r['final_beta'] for r in results]
    all_final_gammas = [r['final_gamma'] for r in results]

    beta_range = max(all_final_betas) - min(all_final_betas)
    gamma_range = max(all_final_gammas) - min(all_final_gammas)

    print(f"\nRange across all initializations:")
    print(f"  Beta range:  {beta_range:.4f}")
    print(f"  Gamma range: {gamma_range:.4f}")

    CONVERGENCE_THRESHOLD = 0.05

    if beta_range < CONVERGENCE_THRESHOLD and gamma_range < CONVERGENCE_THRESHOLD:
        print(f"\n*** REAL ATTRACTOR DETECTED ***")
        print(f"All initializations converged to same region (range < {CONVERGENCE_THRESHOLD})")
        print(f"Frozen frame floor appears REAL at beta≈{statistics.mean(all_final_betas):.4f}, "
              f"gamma≈{statistics.mean(all_final_gammas):.4f}")
    else:
        print(f"\n*** INITIALIZATION STICKINESS DETECTED ***")
        print(f"Different initializations converged to different values (range >= {CONVERGENCE_THRESHOLD})")
        print(f"Floor may be artifact of local minima or weak gradients.")

    # Fine grid search (reduced size for speed)
    best_beta, best_gamma, best_mi, grid_results = fine_grid_search(
        center_beta=0.5,
        center_gamma=0.9,
        radius=0.2,
        grid_size=10  # 10x10 instead of 20x20 for faster execution
    )

    # Compare adaptive vs best static
    adaptive_mi = statistics.mean([r['mean_mi'] for r in results])

    print(f"\n{'='*70}")
    print("ADAPTIVE vs BEST STATIC COMPARISON")
    print(f"{'='*70}\n")
    print(f"Best static (fine grid):  beta={best_beta:.4f}, gamma={best_gamma:.4f}, MI={best_mi:.4f}")
    print(f"Adaptive (mean across inits): MI={adaptive_mi:.4f}")
    print(f"Advantage: {adaptive_mi - best_mi:+.4f}")

    if adaptive_mi > best_mi:
        print("\n*** ADAPTIVE WINS ***")
        print("Gradient-based adaptation discovers better configuration than fine grid search.")
    else:
        print("\n*** STATIC WINS ***")
        print("Fine grid search found better configuration than adaptive.")

    print(f"\n{'='*70}")
    print("TASK #19 COMPLETE")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    main()
