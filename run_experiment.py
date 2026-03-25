"""
run_experiment.py — Universal PRISM harness.

Usage:
  python run_experiment.py --step 1007 --substrate experiments/sub1007_mymech.py
  python run_experiment.py --step 1006 --substrate experiments/sub1006_994.py --seeds 10 --steps 10000

The substrate file defines ONLY the substrate class (no main, no chain setup).
This script IS the harness. Game order, result saving, chain kill — all enforced here.

Substrate file must define one of:
  1. SUBSTRATE_CLASS = MyClass  (explicit, preferred)
  2. A single class with process() + on_level_transition() methods

Jun directive 2026-03-24: the harness is the CONSTANT. Substrates are the VARIABLE.
"""
import argparse
import importlib.util
import inspect
import sys
import os
import time
import json
import shutil

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'experiments'))

from substrates.chain import (ChainRunner, ArcGameWrapper, make_prism_mode,
                              make_prism_random, compute_chain_kill)


def _load_substrate_class(path: str):
    """Load substrate class from a Python file.

    Looks for SUBSTRATE_CLASS module var first, then searches for
    any class with process() and on_level_transition() methods.

    Returns (substrate_cls, module) tuple.
    """
    spec = importlib.util.spec_from_file_location("_substrate_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Explicit declaration wins
    if hasattr(mod, 'SUBSTRATE_CLASS'):
        return mod.SUBSTRATE_CLASS, mod

    # Auto-discover: find class with process + on_level_transition
    candidates = []
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if (obj.__module__ == '_substrate_module'
                and hasattr(obj, 'process')
                and hasattr(obj, 'on_level_transition')):
            candidates.append(obj)

    if len(candidates) == 1:
        return candidates[0], mod
    elif len(candidates) > 1:
        names = [c.__name__ for c in candidates]
        raise ValueError(
            f"Multiple substrate classes found in {path}: {names}. "
            f"Set SUBSTRATE_CLASS = YourClass at module level."
        )
    else:
        raise ValueError(
            f"No substrate class found in {path}. "
            f"Class must have process() and on_level_transition() methods."
        )


def _validate_aggregated(aggregated: dict, expected_phases: list, n_seeds: int):
    """Validate aggregated results before saving. Raises if incomplete."""
    errors = []

    # Check all phases present
    missing = [name for name, _ in expected_phases if name not in aggregated]
    if missing:
        errors.append(f"Missing phases: {missing}")

    # Check each phase has n_seeds results
    for name, data in aggregated.items():
        if isinstance(data, dict) and 'seeds' in data:
            if len(data['seeds']) < n_seeds:
                errors.append(f"{name}: only {len(data['seeds'])}/{n_seeds} seeds")

    if errors:
        raise ValueError("Chain incomplete — cannot save results:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description="PRISM experiment runner")
    parser.add_argument('--step', type=int, required=True, help='Step number (e.g. 1007)')
    parser.add_argument('--substrate', required=True, help='Path to substrate file')
    parser.add_argument('--steps', type=int, default=10_000, help='Steps per game (default 10000)')
    parser.add_argument('--seeds', type=int, default=10, help='Seeds (default 10, min enforced)')
    parser.add_argument('--baseline', default='B:/M/the-search/chain_results/baseline_994.json',
                        help='Baseline JSON path for chain kill verdict')
    parser.add_argument('--save-as-baseline', action='store_true',
                        help='Copy result to chain_results/baseline_994.json (use for step 1006)')
    parser.add_argument('--random-games', type=int, default=3,
                        help='Randomly select N games from API pool (default=3, Jun 2026-03-25)')
    parser.add_argument('--game-seed', type=int, default=None,
                        help='Seed for random game selection (deterministic if set)')
    args = parser.parse_args()

    # Load substrate
    substrate_path = os.path.abspath(args.substrate)
    if not os.path.exists(substrate_path):
        print(f"ERROR: substrate file not found: {substrate_path}")
        sys.exit(1)

    substrate_cls, substrate_mod = _load_substrate_class(substrate_path)
    substrate_name = substrate_cls.__name__
    print("=" * 70)
    print(f"STEP {args.step} — {substrate_name}")
    print(f"Substrate: {substrate_path}")
    print("=" * 70)

    t0 = time.time()

    if args.random_games > 0:
        # Random game selection from full API pool
        chain = make_prism_random(
            n_games=args.random_games,
            game_seed=args.game_seed,
            n_steps=args.steps,
        )
        randomize = True
    else:
        chain, randomize = make_prism_mode("C", n_steps=args.steps)

    # Print game version hashes for whatever games are in the chain
    for name, wrapper in chain:
        if isinstance(wrapper, ArcGameWrapper):
            env_dir = name.lower()
            try:
                h = next(d for d in os.listdir(
                    f'B:/M/the-search/environment_files/{env_dir}') if len(d) >= 8)
            except (StopIteration, FileNotFoundError):
                h = '?'
            print(f"  {name}={h}", end="")
    print()
    print(f"Budget: {args.steps} steps/game, {args.seeds} seeds, randomized order")
    print()

    runner = ChainRunner(chain=chain, n_seeds=args.seeds, randomize_order=randomize, verbose=True)
    aggregated = runner.run(substrate_cls, substrate_kwargs={})

    # Validate completeness before saving
    _validate_aggregated(aggregated, chain, args.seeds)

    # Chain kill verdict
    chain_kill = compute_chain_kill(aggregated, baseline_path=args.baseline)

    # Save results (enforced — cannot be bypassed when using this runner)
    out_path = runner.save_results(
        aggregated=aggregated,
        substrate_name=substrate_name,
        step=args.step,
        config=getattr(substrate_mod, 'CONFIG', {}),
        chain_kill=chain_kill,
    )

    # Optionally promote to canonical baseline
    if args.save_as_baseline:
        shutil.copy(out_path, args.baseline)
        print(f"Saved as canonical baseline: {args.baseline}")

    # Final summary
    print()
    print("=" * 70)
    print(f"STEP {args.step} RESULTS:")
    for name, data in aggregated.items():
        if isinstance(data, dict) and 'l1_rate' in data:
            print(f"  {name}: L1={data['l1_rate']:.0%}  avg_t={data['mean_elapsed']:.1f}s")
    cs = chain_kill.get('chain_score', {})
    print(f"  Chain score: {cs.get('phases_passed', '?')}/{cs.get('phases_total', '?')}")
    print(f"  Chain kill verdict: {chain_kill.get('verdict', 'NO_BASELINE')}")
    if 'per_game_delta' in chain_kill:
        for game, delta in chain_kill['per_game_delta'].items():
            if delta.get('delta') is not None:
                sign = '+' if delta['delta'] >= 0 else ''
                print(f"    {game}: {sign}{delta['delta']:+.0%} ({delta['baseline']:.0%} → {delta['current']:.0%})")
    print(f"  Results: {out_path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print(f"STEP {args.step} DONE")


if __name__ == "__main__":
    main()
