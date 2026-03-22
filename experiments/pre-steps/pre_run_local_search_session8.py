#!/usr/bin/env python3
"""
Local Evolutionary Search Around Candidate #2 (Session 8)

Candidate #2 from Session 7:
- Training gap: +0.176 (vs canonical +0.168 = +7.6%)
- Novel gap: +0.238 (vs canonical +0.173 = +37.6%)

This script runs a focused local search around this breakthrough rule.
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from evolve import run_evolution

# Candidate #2 center (Session 7 breakthrough)
CANDIDATE_2 = {
    'eta': 0.000526,
    'symmetry_break_mult': 0.53345,
    'amplify_mult': 0.387234,
    'drift_mult': 0.157323,
    'threshold': 0.001777,
    'alpha_clip_lo': 0.358291,
    'alpha_clip_hi': 1.909935,
}

# Local search configuration
CONFIG = {
    'population_size': 12,
    'generations': 5,
    'top_k': 4,
    'mutation_sigma': 0.1,  # NARROW — local search only
    'crossover_prob': 0.5,
    'seeds': [42, 137, 2024],
    'k_values': [4, 6, 8, 10],
    'novel_seed_base': 99999,
    'birth_seed': 42,
    'early_stop_patience': 3,
    'checkpoint_every': 1,
    'eval_mode': 'full',  # FULL eval — Session 7 proved medium is unreliable
    'seed_rule': CANDIDATE_2,  # Seed around this rule
    'seed_sigma': 0.1,  # Perturbation scale for initial population
}


def main():
    print("=" * 72)
    print("  LOCAL EVOLUTIONARY SEARCH AROUND CANDIDATE #2")
    print("=" * 72)
    print()
    print("Center rule (Candidate #2 from Session 7):")
    for param, val in CANDIDATE_2.items():
        print(f"  {param:20} = {val:.6f}")
    print()
    print("Search configuration:")
    print(f"  Population: {CONFIG['population_size']}")
    print(f"  Generations: {CONFIG['generations']}")
    print(f"  Mutation sigma: {CONFIG['mutation_sigma']} (LOCAL)")
    print(f"  Eval mode: {CONFIG['eval_mode']}")
    print(f"  Seeds: {CONFIG['seeds']}")
    print(f"  K values: {CONFIG['k_values']}")
    print()
    print("Strategy:")
    print("  1. Seed initial population with Candidate #2 + 11 perturbations")
    print("  2. Run 5 generations of local evolution (sigma=0.1)")
    print("  3. Use FULL eval (Session 7: medium eval is unreliable)")
    print("  4. Early stop if no improvement for 3 generations")
    print()
    print("=" * 72)
    print()

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    # Set checkpoint path with session identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = results_dir / f"local_search_session8_{timestamp}.json"

    # Run evolution
    results = run_evolution(config=CONFIG, checkpoint_path=checkpoint_path, verbose=True)

    # Print top 3 results
    print()
    print("=" * 72)
    print("  TOP 3 RESULTS FOR VALIDATION")
    print("=" * 72)
    print()
    for i, res in enumerate(results['final_population'][:3]):
        print(f"Rank {i+1}:")
        print(f"  Fitness: {res['fitness']:+.4f}")
        print(f"  Training gap: {res['variant_gap']:+.4f}")
        print(f"  Novel gap: {res['novel_gap']:+.4f}")
        print(f"  Rule parameters:")
        for param, val in res['rule'].items():
            print(f"    {param:20} = {val:.6f}")
        print()

    print("=" * 72)
    print(f"Results saved to: {checkpoint_path.parent / f'final_{checkpoint_path.name}'}")
    print("=" * 72)

    return results


if __name__ == '__main__':
    main()
