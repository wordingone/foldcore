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
                        help='Number of games to randomly select from API pool (default=3). '
                             'Jun 2026-03-25: preview games dead, all runs use random pool.')
    parser.add_argument('--game-seed', type=int, default=None,
                        help='Seed for random game selection (deterministic if set)')
    parser.add_argument('--blind', action='store_true', default=True,
                        help='ALWAYS ON (Jun 2026-03-25). Game names hidden in output.')
    parser.add_argument('--no-blind', action='store_true',
                        help='Jun-only override to disable blind mode')
    args = parser.parse_args()

    # === ENFORCED RULES (Jun 2026-03-25) — NOT CIRCUMVENTABLE ===

    # Rule 1: Blind mode is ALWAYS on unless Jun explicitly overrides
    if args.no_blind:
        print("WARNING: --no-blind is a Jun-only override. Blind mode disabled.")
        args.blind = False
    else:
        args.blind = True

    # Rule 2: Random games MUST be from full pool (no zero = no preview-only)
    if args.random_games <= 0:
        print("ERROR: --random-games must be > 0. All experiments use random pool. (Jun 2026-03-25)")
        sys.exit(1)

    # Rule 3: Track used game-seeds to prevent reuse
    _seed_log = 'B:/M/the-search/chain_results/.used_seeds.txt'
    if args.game_seed is not None:
        used = set()
        if os.path.exists(_seed_log):
            with open(_seed_log) as f:
                used = {int(line.strip()) for line in f if line.strip().isdigit()}
        if args.game_seed in used:
            print(f"ERROR: game-seed {args.game_seed} already used. Fresh seeds only. (Jun 2026-03-25)")
            print(f"Used seeds: {sorted(used)}")
            sys.exit(1)
        with open(_seed_log, 'a') as f:
            f.write(f"{args.game_seed}\n")
    else:
        # No seed specified = random each time (always fresh)
        pass

    # === END ENFORCED RULES ===

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

    # Jun 2026-03-25: ALL runs use random game selection from full API pool.
    # Preview games (FT09/LS20/VC33) are dead. No more hardcoded game configs.
    if args.random_games < 1:
        print("WARNING: --random-games must be >= 1 (Jun directive 2026-03-25). Using 3.")
        args.random_games = 3
    chain = make_prism_random(
        n_games=args.random_games,
        game_seed=args.game_seed,
        n_steps=args.steps,
    )
    randomize = True

    # Print game version hashes for whatever games are in the chain
    # --blind: mask game names so neither side can design for specific games
    _game_labels = {}
    for i, (name, wrapper) in enumerate(chain):
        label = f"GAME_{i+1}" if args.blind else name
        _game_labels[name] = label
        if isinstance(wrapper, ArcGameWrapper):
            env_dir = name.lower()
            try:
                h = next(d for d in os.listdir(
                    f'B:/M/the-search/environment_files/{env_dir}') if len(d) >= 8)
            except (StopIteration, FileNotFoundError):
                h = '?'
            print(f"  {label}={'***' if args.blind else h}", end="")
    print()
    print(f"Budget: {args.steps} steps/game, {args.seeds} seeds, randomized order")
    print()

    # --blind: suppress verbose game-by-game output to prevent name leaks
    runner = ChainRunner(chain=chain, n_seeds=args.seeds, randomize_order=randomize,
                         verbose=(not args.blind))
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
    any_zero = False
    for name, data in aggregated.items():
        if isinstance(data, dict) and 'l1_rate' in data:
            label = _game_labels.get(name, name)
            print(f"  {label}: L1={data['l1_rate']:.0%}  avg_t={data['mean_elapsed']:.1f}s")
            if data['l1_rate'] == 0:
                any_zero = True
    cs = chain_kill.get('chain_score', {})
    print(f"  Chain score: {cs.get('phases_passed', '?')}/{cs.get('phases_total', '?')}")
    if any_zero:
        print(f"  ** DEBATE FAIL: one or more games at 0% L1 (Jun directive 2026-03-25) **")
    print(f"  Chain kill verdict: {chain_kill.get('verdict', 'NO_BASELINE')}")
    if 'per_game_delta' in chain_kill:
        for game, delta in chain_kill['per_game_delta'].items():
            if delta.get('delta') is not None:
                label = _game_labels.get(game, game)
                sign = '+' if delta['delta'] >= 0 else ''
                print(f"    {label}: {sign}{delta['delta']:+.0%} ({delta['baseline']:.0%} → {delta['current']:.0%})")
    print(f"  Results: {'[blind]' if args.blind else out_path}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    print(f"STEP {args.step} DONE")


if __name__ == "__main__":
    main()
