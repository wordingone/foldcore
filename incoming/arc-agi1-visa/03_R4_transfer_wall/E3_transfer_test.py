"""
E3 Transfer Test — Leo #11709 DECIDER (2026-05-30).

Takes the MDL-positive holed ops formed in E3_density_curve step 3
and tests whether they reduce BFS search cost on the ORIGINAL
(un-augmented) held-out tasks.

CLIMATE:
  TRANSFER_GENUINE (cost drops significantly):
    augmentation surfaced latent structure that generalizes to real held-out
    → 12-op self-compilation bootstraps → continue this direction.
  HOLLOW_CLIMB (no cost reduction on original held-out):
    augmentation injected its own symmetry; ARC's real structure still below R*
    → code-synthesis pivot STANDS.

Read from: E3_density_curve_result.json (requires step 3 with MDL-positive ops)
Write to: E3_transfer_test_result.json

The transfer test:
  - Extract MDL-positive holed ops from step 3 result.
  - For each holed op, generate all 12 concrete instantiations
    (substitute each of the 12 seed ops as the hole filler).
  - Build a compound library from these instantiations.
  - Run BFS WITH this library on each of the 200 original held-out tasks.
  - Compare to baseline BFS (no library).
  - Report: mean_cost_baseline, mean_cost_with_library, delta_pct, solved counts.
  Threshold: same as R* (delta < -5%) = TRANSFER_GENUINE. Otherwise HOLLOW_CLIMB.
"""

import numpy as np
import json
import itertools
import sys
import os
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
CURVE_RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_density_curve_result.json"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_transfer_test_result.json"

BUDGET = 10000
MAX_LENGTH = 5
SPLIT_SEED = 42
N_HELD = 200
TRANSFER_MARGIN_PCT = 5.0  # same as R* margin

# ── Seed DSL ─────────────────────────────────────────────────────────────────

def _bg(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(x):
    nz = np.argwhere(x != _bg(x))
    if len(nz) == 0:
        return x
    return x[nz.min(0)[0]:nz.max(0)[0]+1, nz.min(0)[1]:nz.max(0)[1]+1]

SEED_OPS = {
    'id':    lambda x: x,
    'fh':    lambda x: x[:, ::-1],
    'fv':    lambda x: x[::-1],
    'tr':    lambda x: x.T,
    'rot':   lambda x: np.rot90(x),
    'crop':  _crop,
    'dup_h': lambda x: np.hstack([x, x]),
    'dup_v': lambda x: np.vstack([x, x]),
    'mir_h': lambda x: np.hstack([x, x[:, ::-1]]),
    'mir_v': lambda x: np.vstack([x, x[::-1]]),
    'up2':   lambda x: np.repeat(np.repeat(x, 2, 0), 2, 1),
    'down2': lambda x: x[::2, ::2],
}

SORTED_SEED_OPS = sorted(SEED_OPS.keys())


# ── Grid operations ───────────────────────────────────────────────────────────

def apply_op(name, grid, library=None):
    try:
        import numpy as np
        g = np.array(grid, dtype=np.int64)
        if library and name in library:
            for sub in library[name]:
                result = apply_op(sub, g, library)
                if result is None:
                    return None
                g = result
            return g
        return SEED_OPS[name](g)
    except Exception:
        return None


def apply_program(ops, grid, library=None):
    g = np.array(grid, dtype=np.int64)
    for name in ops:
        result = apply_op(name, g, library)
        if result is None:
            return None
        g = result
    return g


def check_program(ops, task_io, library=None):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is None or result.shape != out.shape or not np.array_equal(result, out):
            return False
    return True


# ── BFS with compound library ──────────────────────────────────────────────────

def search_with_library(task_io, library, max_length=MAX_LENGTH, budget=BUDGET):
    """BFS using seed ops + compound library ops as primitives."""
    all_ops = SORTED_SEED_OPS + sorted(library.keys())
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(all_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io, library):
                return list(ops), nodes
    return None, nodes


def search_baseline(task_io, max_length=MAX_LENGTH, budget=BUDGET):
    """BFS without library (baseline)."""
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(SORTED_SEED_OPS, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io):
                return list(ops), nodes
    return None, nodes


# ── Holed op library builder ───────────────────────────────────────────────────

def build_holed_library(holed_ops_list):
    """
    For each MDL-positive holed op, generate all 12 concrete instantiations
    by substituting each seed op as the hole filler.
    Returns a dict: compound_name -> [sub_op_names...]
    """
    library = {}
    for op_entry in holed_ops_list:
        name = op_entry['name']
        # Parse: prefix___HOLE___suffix
        parts = name.split('___HOLE___')
        if len(parts) != 2:
            print(f"  WARNING: cannot parse holed op name '{name}', skipping.")
            continue
        prefix_str, suffix_str = parts
        prefix = [p for p in prefix_str.split('__') if p and p != 'eps']
        suffix = [s for s in suffix_str.split('__') if s and s != 'eps']

        for filler in SORTED_SEED_OPS:
            compound_name = f"__hld_{prefix_str}_{filler}_{suffix_str}"
            compound_ops = prefix + [filler] + suffix
            # Only add if the compound fits in seed vocabulary
            if all(op in SEED_OPS for op in compound_ops):
                library[compound_name] = compound_ops

    return library


# ── Load tasks ────────────────────────────────────────────────────────────────

def load_all_tasks():
    tasks = []
    counts = {}
    for subdir in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, subdir)
        if not os.path.isdir(d):
            counts[subdir] = 0
            continue
        n = 0
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(d, fname)) as f:
                data = json.load(f)
            pairs = [{'input': p['input'], 'output': p['output']}
                     for p in data.get('train', [])
                     if len(p['input']) > 0 and len(p['output']) > 0]
            if pairs:
                tasks.append({'id': fname.replace('.json', ''), 'io': pairs,
                               'split_dir': subdir})
                n += 1
        counts[subdir] = n
    return tasks, counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E3 TRANSFER TEST — Leo #11709 DECIDER")
    print("Does the formed holed library reduce cost on ORIGINAL held-out?")
    print("=" * 70)
    sys.stdout.flush()

    # Load curve result
    if not os.path.exists(CURVE_RESULT_PATH):
        print(f"ERROR: {CURVE_RESULT_PATH} not found. Run E3_density_curve.py first.")
        sys.exit(1)

    with open(CURVE_RESULT_PATH) as f:
        curve_result = json.load(f)

    # Verify step 3 has MDL-positive ops
    steps = curve_result.get('steps', [])
    if len(steps) < 3:
        print(f"ERROR: fewer than 3 steps in curve result ({len(steps)} found).")
        sys.exit(1)

    step3 = steps[2]
    mdl_positive_ops = step3.get('mdl_positive_ops', [])
    if not mdl_positive_ops:
        print("No MDL-positive ops in step 3 — HOLLOW_CLIMB confirmed, no transfer to test.")
        sys.exit(0)

    print(f"MDL-positive ops from step 3 ({len(mdl_positive_ops)}):")
    for op in mdl_positive_ops:
        print(f"  {op['name']}: count={op['count']}, gain={op['gain']}")

    # Build library
    print("\nBuilding compound library from holed ops...")
    library = build_holed_library(mdl_positive_ops)
    print(f"  Library size: {len(library)} compound ops")
    for name, ops in list(library.items())[:10]:
        print(f"    {name}: {ops}")
    sys.stdout.flush()

    if not library:
        print("ERROR: no valid compound ops generated from holed ops.")
        sys.exit(1)

    # Reconstruct held-out tasks from same split as density curve
    print("\nReconstructing held-out tasks (same split as E3_density_curve)...")
    all_tasks, dir_counts = load_all_tasks()
    training_tasks = [t for t in all_tasks if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(training_tasks))
    held_tasks = [training_tasks[i] for i in indices[:N_HELD]]
    print(f"  Held-out: {len(held_tasks)} tasks (training, split_seed={SPLIT_SEED})")

    # Cross-check: if gated result has held_task_ids, verify match
    if 'held_task_ids' in curve_result:
        curve_held_ids = set(curve_result['held_task_ids'])
        our_held_ids = {t['id'] for t in held_tasks}
        if curve_held_ids == our_held_ids:
            print("  Held-out ID match: CONFIRMED (same 200 tasks)")
        else:
            print(f"  WARNING: held-out ID mismatch! "
                  f"Expected {len(curve_held_ids)}, got {len(our_held_ids)}, "
                  f"intersection={len(curve_held_ids & our_held_ids)}")

    # Run baseline BFS on held-out
    print("\nRunning BASELINE BFS on held-out (no library)...")
    sys.stdout.flush()
    baseline_costs = []
    baseline_solved = 0
    for task in held_tasks:
        prog, cost = search_baseline(task['io'])
        baseline_costs.append(cost)
        if prog:
            baseline_solved += 1
    baseline_mean = float(np.mean(baseline_costs))
    print(f"  Baseline: mean_cost={baseline_mean:.1f}, solved={baseline_solved}/{N_HELD}")
    sys.stdout.flush()

    # Run BFS WITH holed library on held-out
    print("\nRunning BFS WITH holed library on held-out...")
    sys.stdout.flush()
    library_costs = []
    library_solved = 0
    library_new_solves = 0
    for i, task in enumerate(held_tasks):
        prog, cost = search_with_library(task['io'], library)
        library_costs.append(cost)
        if prog:
            library_solved += 1
            # Check if this is a new solve (not solved by baseline)
            baseline_prog, _ = search_baseline(task['io'])
            if not baseline_prog:
                library_new_solves += 1
    library_mean = float(np.mean(library_costs))
    delta_pct = (library_mean - baseline_mean) / baseline_mean * 100
    print(f"  With library: mean_cost={library_mean:.1f}, solved={library_solved}/{N_HELD}, "
          f"new_solves={library_new_solves}")
    print(f"  Delta: {delta_pct:+.1f}%")
    sys.stdout.flush()

    # Verdict
    if delta_pct < -TRANSFER_MARGIN_PCT:
        verdict = f"TRANSFER_GENUINE: delta={delta_pct:+.1f}% < -{TRANSFER_MARGIN_PCT}%. " \
                  f"Augmentation surfaced latent structure that generalizes. 12-op continues."
    else:
        verdict = f"HOLLOW_CLIMB: delta={delta_pct:+.1f}% >= -{TRANSFER_MARGIN_PCT}%. " \
                  f"Holed ops do NOT lower cost on original held-out. " \
                  f"Augmentation injected its own symmetry. Code-synthesis pivot STANDS."

    print(f"\nVERDICT: {verdict}")

    # Breakdown by aug transform origin
    aug_origin_analysis = {}
    for op in mdl_positive_ops:
        name = op['name']
        # Identify which D4 transforms compose the op
        parts = name.split('___HOLE___')
        if len(parts) == 2:
            prefix_str, suffix_str = parts
            aug_origin_analysis[name] = {
                'prefix': prefix_str,
                'suffix': suffix_str,
                'count': op['count'],
                'gain': op['gain'],
                'aug_overlap_note': (
                    'OVERLAP: op contains D4 aug transforms (mir_h, mir_v, rot, fh, fv, tr)'
                    if any(t in name for t in ['mir_h', 'mir_v', 'rot', 'fh', 'fv', 'tr'])
                    else 'No D4 aug overlap in op name'
                ),
            }

    result = {
        'experiment': 'E3-transfer-test',
        'note': (
            'Leo #11709 DECIDER: do holed ops from step-3 (augmented source) '
            'transfer to original (un-augmented) held-out? '
            'Transfer genuine = structure generalizes; hollow = aug artifact.'
        ),
        'config': {
            'budget': BUDGET,
            'max_length': MAX_LENGTH,
            'split_seed': SPLIT_SEED,
            'n_held': N_HELD,
            'transfer_margin_pct': TRANSFER_MARGIN_PCT,
        },
        'source_ops': mdl_positive_ops,
        'library_size': len(library),
        'baseline': {
            'mean_cost': baseline_mean,
            'solved': baseline_solved,
            'n_held': N_HELD,
        },
        'with_library': {
            'mean_cost': library_mean,
            'solved': library_solved,
            'new_solves': library_new_solves,
            'n_held': N_HELD,
        },
        'delta_pct': delta_pct,
        'verdict': verdict,
        'aug_origin_analysis': aug_origin_analysis,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {RESULT_PATH}")


if __name__ == '__main__':
    main()
