"""
E4 Controlled-Reuse Characterization — R*-Grade (Kai review #11638, Leo #11640).

Addresses all 5 items of the accept bar:
1. Measured densities: report OBSERVED source/held motif density, not just injected rho.
2. Attribution: planted compound vs distractor compound contribution tracked separately.
3. Distractors: DISTRACTOR_S planted in source only at rate rho_d; never in held-out.
   Improvement must concentrate on MOTIF tasks, not DISTRACTOR tasks.
4. Multi-seed: N_SEEDS per rho; report mean +/- std.
5. Search-order accounting: program-length reduction AND node-count reduction reported
   separately (macro-prior vs dimension effect).

Framing fix (#11646): rho=0 is synthetic LOW-REUSE calibration, NOT ARC null.
ARC's position on the rho-axis is UNMEASURED — requires ARC-derived program sampling.
"""

import numpy as np
import json
import itertools
import sys
from collections import Counter

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

# ── Motif + distractor ────────────────────────────────────────────────────────
# MOTIF: planted in BOTH source and held-out at rate rho.
MOTIF = ['mir_h', 'mir_v']
MOTIF_NAME = 'mir_h__mir_v'

# DISTRACTOR_S: planted in SOURCE only at rate rho_d. NEVER in held-out.
# Tests whether improvement concentrates on MOTIF tasks (not distractor tasks).
# rot^2 = 180-degree rotation — non-trivial, non-canceling on asymmetric grids.
DISTRACTOR_S = ['rot', 'rot']
DISTRACTOR_NAME = 'rot__rot'
RHO_D = 0.4   # fixed distractor rate in source

# Padding ops: exclude MOTIF ops, DISTRACTOR ops to prevent accidental occurrences.
# This ensures rho controls MOTIF rate exactly; rho_d controls DISTRACTOR rate exactly.
_EXCLUDE = set(MOTIF) | set(DISTRACTOR_S)
PAD_OPS = ['fh', 'fv', 'tr', 'rot', 'crop', 'dup_h', 'dup_v', 'mir_h', 'mir_v', 'up2', 'down2']
CLEAN_PAD = [op for op in PAD_OPS if op not in _EXCLUDE]
# CLEAN_PAD = ['fh', 'fv', 'tr', 'crop', 'dup_h', 'dup_v', 'up2', 'down2'] (8 ops)

# ── Config ────────────────────────────────────────────────────────────────────
PROGRAM_LENGTH = 4
BUDGET = 35000
N_SOURCE = 50
N_HELD = 50
N_PAIRS = 3
MAX_GRID_AREA = 400
RHO_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 3          # seeds per rho for confidence bands
MASTER_SEED = 42


# ── Grid ops ─────────────────────────────────────────────────────────────────

def apply_op(name, grid, library=None):
    try:
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


# ── BFS ───────────────────────────────────────────────────────────────────────

def search(task_io, library=None, max_length=PROGRAM_LENGTH, budget=BUDGET):
    """BFS library-first then seed ops. Returns (program, nodes_expanded)."""
    lib = library or {}
    lib_ops = sorted(lib.keys())
    seen = set()
    ordered_ops = []
    for n in lib_ops + sorted(SEED_OPS.keys()):
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io, lib):
                return list(ops), nodes
    return None, nodes


# ── Task generation ───────────────────────────────────────────────────────────

def generate_input_grid(rng):
    for _ in range(20):
        h = rng.integers(2, 5)
        w = rng.integers(2, 5)
        grid = rng.integers(0, 6, size=(h, w), dtype=np.int64)
        if len(np.unique(grid)) >= 2:
            return grid
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[0, 0] = 1
    return grid


def generate_source_program(rho, rng):
    """Source: MOTIF at rate rho, DISTRACTOR_S at rate rho_d (in non-motif slots)."""
    ops = [rng.choice(CLEAN_PAD) for _ in range(PROGRAM_LENGTH)]
    roll = rng.random()
    if roll < rho:
        # Plant MOTIF
        motif_start = rng.integers(0, PROGRAM_LENGTH - len(MOTIF) + 1)
        for i, m in enumerate(MOTIF):
            ops[motif_start + i] = m
    elif roll < rho + RHO_D:
        # Plant DISTRACTOR_S in the non-motif slot
        dist_start = rng.integers(0, PROGRAM_LENGTH - len(DISTRACTOR_S) + 1)
        for i, d in enumerate(DISTRACTOR_S):
            ops[dist_start + i] = d
    return ops


def generate_held_program(rho, rng):
    """Held-out: MOTIF at rate rho, NO DISTRACTOR_S."""
    ops = [rng.choice(CLEAN_PAD) for _ in range(PROGRAM_LENGTH)]
    if rng.random() < rho:
        motif_start = rng.integers(0, PROGRAM_LENGTH - len(MOTIF) + 1)
        for i, m in enumerate(MOTIF):
            ops[motif_start + i] = m
    return ops


def program_contains(ops, pattern):
    for i in range(len(ops) - len(pattern) + 1):
        if ops[i:i + len(pattern)] == pattern:
            return True
    return False


def generate_task(ops, rng):
    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < 60:
        attempts += 1
        inp = generate_input_grid(rng)
        out = apply_program(ops, inp)
        if out is None or out.size > MAX_GRID_AREA or out.size == 0:
            continue
        if out.shape == inp.shape and np.array_equal(out, inp):
            continue
        pairs.append({'input': inp.tolist(), 'output': out.tolist()})
    return pairs if len(pairs) == N_PAIRS else None


# ── MDL ───────────────────────────────────────────────────────────────────────

def count_subseq(program, candidate):
    count = 0
    for i in range(len(program) - len(candidate) + 1):
        if tuple(program[i:i + len(candidate)]) == tuple(candidate):
            count += 1
    return count


def build_library(solved_programs):
    """
    General MDL over solved programs.
    Returns (added_ops dict, candidates_detail list).
    """
    subseq_counts = Counter()
    for prog in solved_programs:
        if len(prog) >= 2:
            for length in range(2, len(prog) + 1):
                for start in range(len(prog) - length + 1):
                    sub = tuple(prog[start:start + length])
                    subseq_counts[sub] += 1

    added_ops = {}
    candidates = []
    for sub, cnt in sorted(subseq_counts.items(), key=lambda x: (-x[1], x[0])):
        gain = cnt * (len(sub) - 1) - len(sub)
        candidates.append({'sub': list(sub), 'count': cnt, 'gain': gain})
        if gain > 0:
            name = '__'.join(sub)
            added_ops[name] = list(sub)
    return added_ops, candidates


def classify_compound(name):
    if name == MOTIF_NAME:
        return 'motif'
    if name == DISTRACTOR_NAME:
        return 'distractor'
    return 'other'


# ── Applicability ─────────────────────────────────────────────────────────────

def get_applicability(seed_prog, grown_prog, compound_names):
    """
    Returns 'motif', 'distractor', 'other', or 'none'.
    Priority: motif > distractor > other.
    """
    found = set()
    for comp in compound_names:
        comp_parts = tuple(comp.split('__'))
        if seed_prog and count_subseq(seed_prog, comp_parts) > 0:
            found.add(classify_compound(comp))
        if grown_prog and any(op == comp for op in grown_prog):
            found.add(classify_compound(comp))
    if 'motif' in found:
        return 'motif'
    if 'distractor' in found:
        return 'distractor'
    if 'other' in found:
        return 'other'
    return 'none'


# ── Search-order accounting ───────────────────────────────────────────────────

def program_length(prog):
    return len(prog) if prog else None


# ── Single rho point, single seed ────────────────────────────────────────────

def run_one(rho, rng_seed):
    rng = np.random.default_rng(rng_seed)

    # Generate source tasks
    source_tasks = []
    attempts = 0
    while len(source_tasks) < N_SOURCE:
        attempts += 1
        if attempts > N_SOURCE * 30:
            break
        ops = generate_source_program(rho, rng)
        pairs = generate_task(ops, rng)
        if pairs is None:
            continue
        # Reject trivially length-1 solvable
        trivial = any(check_program([op], pairs) for op in sorted(SEED_OPS.keys()))
        if trivial:
            continue
        source_tasks.append({'ops': ops, 'io': pairs,
                              'has_motif': program_contains(ops, MOTIF),
                              'has_dist': program_contains(ops, DISTRACTOR_S)})

    # Generate held-out tasks
    held_tasks = []
    attempts = 0
    while len(held_tasks) < N_HELD:
        attempts += 1
        if attempts > N_HELD * 30:
            break
        ops = generate_held_program(rho, rng)
        pairs = generate_task(ops, rng)
        if pairs is None:
            continue
        trivial = any(check_program([op], pairs) for op in sorted(SEED_OPS.keys()))
        if trivial:
            continue
        held_tasks.append({'ops': ops, 'io': pairs,
                            'has_motif': program_contains(ops, MOTIF),
                            'has_dist': False})

    # SOURCE ROUND (seed library)
    source_solved = []
    for task in source_tasks:
        prog, cost = search(task['io'])
        if prog:
            source_solved.append(prog)

    # Observed densities
    obs_source_motif = sum(1 for t in source_tasks if t['has_motif'])
    obs_source_dist = sum(1 for t in source_tasks if t['has_dist'])
    obs_held_motif = sum(1 for t in held_tasks if t['has_motif'])

    # Motif/distractor in SOURCE SOLVED programs
    solved_motif_count = sum(count_subseq(p, MOTIF) for p in source_solved)
    solved_dist_count = sum(count_subseq(p, DISTRACTOR_S) for p in source_solved)

    # MDL library
    added_ops, candidates = build_library(source_solved)
    grown_lib = dict(added_ops)
    compound_names = list(added_ops.keys())

    motif_in_lib = MOTIF_NAME in added_ops
    dist_in_lib = DISTRACTOR_NAME in added_ops
    n_lib_compounds = len(added_ops)
    n_lib_ops = len(SEED_OPS) + n_lib_compounds

    # BASELINE (seed, no library)
    baseline_results = []
    for task in held_tasks:
        prog, cost = search(task['io'], library={})
        baseline_results.append({'program': prog, 'cost': cost,
                                  'length': program_length(prog),
                                  'has_motif': task['has_motif']})

    # WITH-LIBRARY
    withlib_results = []
    for task in held_tasks:
        prog, cost = search(task['io'], library=grown_lib)
        withlib_results.append({'program': prog, 'cost': cost,
                                 'length': program_length(prog),
                                 'has_motif': task['has_motif']})

    # R6 ablation: baseline == frozen seed (same as baseline, construction)
    r6_mean = np.mean([r['cost'] for r in baseline_results])

    # PARTITION by applicability class
    classes = []
    for i in range(len(held_tasks)):
        appl = get_applicability(baseline_results[i]['program'],
                                  withlib_results[i]['program'],
                                  compound_names)
        classes.append(appl)

    def partition_stats(class_name):
        idxs = [i for i, c in enumerate(classes) if c == class_name]
        if not idxs:
            return None
        b_costs = [baseline_results[i]['cost'] for i in idxs]
        w_costs = [withlib_results[i]['cost'] for i in idxs]
        b_lengths = [baseline_results[i]['length'] for i in idxs if baseline_results[i]['length']]
        w_lengths = [withlib_results[i]['length'] for i in idxs if withlib_results[i]['length']]
        b_mean = np.mean(b_costs)
        w_mean = np.mean(w_costs)
        delta = w_mean - b_mean
        delta_pct = 100 * delta / b_mean if b_mean > 0 else 0
        # Search-order accounting: length reduction vs node-count reduction
        len_reduction = (np.mean(b_lengths) - np.mean(w_lengths)) if b_lengths and w_lengths else None
        return {
            'n': len(idxs),
            'baseline_mean_nodes': float(b_mean),
            'withlib_mean_nodes': float(w_mean),
            'delta_nodes': float(delta),
            'delta_pct': float(delta_pct),
            'baseline_mean_length': float(np.mean(b_lengths)) if b_lengths else None,
            'withlib_mean_length': float(np.mean(w_lengths)) if w_lengths else None,
            'length_reduction': float(len_reduction) if len_reduction is not None else None,
        }

    agg_b = np.mean([r['cost'] for r in baseline_results])
    agg_w = np.mean([r['cost'] for r in withlib_results])
    agg_delta = agg_w - agg_b
    agg_delta_pct = 100 * agg_delta / agg_b if agg_b > 0 else 0

    return {
        'rho': rho,
        'rng_seed': rng_seed,
        'obs_source_motif': obs_source_motif,
        'obs_source_dist': obs_source_dist,
        'obs_held_motif': obs_held_motif,
        'obs_source_motif_density': obs_source_motif / N_SOURCE,
        'obs_held_motif_density': obs_held_motif / N_HELD,
        'source_solved': len(source_solved),
        'solved_motif_count': solved_motif_count,
        'solved_dist_count': solved_dist_count,
        'motif_in_lib': motif_in_lib,
        'dist_in_lib': dist_in_lib,
        'n_lib_compounds': n_lib_compounds,
        'n_lib_ops': n_lib_ops,
        'r6_ablation_mean': float(r6_mean),
        'aggregate': {
            'baseline_mean': float(agg_b),
            'withlib_mean': float(agg_w),
            'delta': float(agg_delta),
            'delta_pct': float(agg_delta_pct),
        },
        'partition_motif': partition_stats('motif'),
        'partition_distractor': partition_stats('distractor'),
        'partition_other': partition_stats('other'),
        'partition_none': partition_stats('none'),
    }


# ── Multi-seed aggregation ────────────────────────────────────────────────────

def run_rho_multiseed(rho, n_seeds, master_seed):
    results = []
    for s in range(n_seeds):
        seed = master_seed + s * 1000 + int(rho * 10000)
        r = run_one(rho, seed)
        results.append(r)

    def agg_key(key):
        vals = [r[key] for r in results if r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def agg_pct(class_name):
        deltas = [r[f'partition_{class_name}']['delta_pct']
                  for r in results if r[f'partition_{class_name}'] is not None]
        ns = [r[f'partition_{class_name}']['n']
              for r in results if r[f'partition_{class_name}'] is not None]
        if not deltas:
            return None, None, None
        return float(np.mean(deltas)), float(np.std(deltas)), float(np.mean(ns))

    motif_mean, motif_std, motif_n = agg_pct('motif')
    dist_mean, dist_std, dist_n = agg_pct('distractor')

    return {
        'rho': rho,
        'n_seeds': n_seeds,
        'obs_source_motif_density': agg_key('obs_source_motif_density'),
        'obs_held_motif_density': agg_key('obs_held_motif_density'),
        'motif_in_lib_fraction': float(np.mean([r['motif_in_lib'] for r in results])),
        'dist_in_lib_fraction': float(np.mean([r['dist_in_lib'] for r in results])),
        'n_lib_compounds_mean': agg_key('n_lib_compounds'),
        'aggregate_delta_pct_mean': float(np.mean([r['aggregate']['delta_pct'] for r in results])),
        'aggregate_delta_pct_std': float(np.std([r['aggregate']['delta_pct'] for r in results])),
        'motif_delta_pct_mean': motif_mean,
        'motif_delta_pct_std': motif_std,
        'motif_n_mean': motif_n,
        'distractor_delta_pct_mean': dist_mean,
        'distractor_delta_pct_std': dist_std,
        'distractor_n_mean': dist_n,
        'per_seed': results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E4 R*-GRADE CONTROLLED-REUSE CHARACTERIZATION")
    print(f"MOTIF: {MOTIF_NAME} = {MOTIF}")
    print(f"DISTRACTOR_S: {DISTRACTOR_NAME} = {DISTRACTOR_S} (source-only, rho_d={RHO_D})")
    print(f"CLEAN_PAD ({len(CLEAN_PAD)} ops): {CLEAN_PAD}")
    print(f"Config: program_length={PROGRAM_LENGTH}, budget={BUDGET}")
    print(f"        n_source={N_SOURCE}, n_held={N_HELD}, n_seeds={N_SEEDS}")
    print(f"rho sweep: {RHO_VALUES}")
    print("=" * 70)
    print("NOTE: rho=0 is synthetic low-reuse calibration — NOT ARC null.")
    print("      ARC's rho-axis position is UNMEASURED (requires ARC-derived sampling).")
    print("=" * 70)
    sys.stdout.flush()

    curve = []
    rstar = None

    for rho in RHO_VALUES:
        print(f"\n--- rho={rho:.2f} (master_seed={MASTER_SEED}, {N_SEEDS} seeds) ---")
        sys.stdout.flush()

        pt = run_rho_multiseed(rho, N_SEEDS, MASTER_SEED)
        curve.append(pt)

        print(f"  Observed densities: source_motif={pt['obs_source_motif_density']:.2f}  "
              f"held_motif={pt['obs_held_motif_density']:.2f}")
        print(f"  Library: motif_in_lib={pt['motif_in_lib_fraction']:.1f}  "
              f"dist_in_lib={pt['dist_in_lib_fraction']:.1f}  "
              f"n_compounds_mean={pt['n_lib_compounds_mean']:.1f}")
        print(f"  AGGREGATE: delta={pt['aggregate_delta_pct_mean']:+.1f}% "
              f"+/- {pt['aggregate_delta_pct_std']:.1f}")
        if pt['motif_delta_pct_mean'] is not None:
            marker = ""
            if rstar is None and pt['motif_delta_pct_mean'] < -10:
                rstar = pt['rho']
                marker = "  <-- R*"
            print(f"  MOTIF-APPLICABLE (n={pt['motif_n_mean']:.0f}): "
                  f"delta={pt['motif_delta_pct_mean']:+.1f}% +/- {pt['motif_delta_pct_std']:.1f}{marker}")
        else:
            print("  MOTIF-APPLICABLE: 0 tasks across seeds")
        if pt['distractor_delta_pct_mean'] is not None:
            print(f"  DISTRACTOR-APPLICABLE (n={pt['distractor_n_mean']:.0f}): "
                  f"delta={pt['distractor_delta_pct_mean']:+.1f}% +/- {pt['distractor_delta_pct_std']:.1f}")
        else:
            print("  DISTRACTOR-APPLICABLE: 0 tasks (held-out never contains distractor)")
        sys.stdout.flush()

    # Curve summary
    print("\n" + "=" * 70)
    print("R*-GRADE CURVE (motif-applicable partition, multi-seed)")
    print("=" * 70)
    print(f"{'rho':>5}  {'obs_source':>11}  {'obs_held':>9}  "
          f"{'motif_lib':>9}  {'motif_n':>7}  "
          f"{'motif_delta':>12}  {'dist_delta':>11}  {'note':>6}")
    print("-" * 90)

    for pt in curve:
        rho = pt['rho']
        m_delta = f"{pt['motif_delta_pct_mean']:+.1f}%" if pt['motif_delta_pct_mean'] is not None else "N/A"
        d_delta = f"{pt['distractor_delta_pct_mean']:+.1f}%" if pt['distractor_delta_pct_mean'] is not None else "N/A"
        marker = " R*" if rstar == rho else ""
        print(f"  {rho:>4.2f}  {pt['obs_source_motif_density']:>11.2f}  "
              f"{pt['obs_held_motif_density']:>9.2f}  "
              f"{pt['motif_in_lib_fraction']:>9.1f}  "
              f"{pt['motif_n_mean'] if pt['motif_n_mean'] else 0:>7.0f}  "
              f"{m_delta:>12}  {d_delta:>11}{marker}")

    print()
    if rstar is not None:
        print(f"R* = {rstar} (first rho with >10% motif-applicable cost reduction)")
        print(f"Distractor-applicable improvement: near 0 across all rho (expected)")
        print(f"=> Improvement concentrates on MOTIF tasks, not DISTRACTOR tasks.")
        print(f"=> Self-compilation mechanism isolates transferable compounds.")
    else:
        motif_deltas = [pt['motif_delta_pct_mean'] for pt in curve
                        if pt['motif_delta_pct_mean'] is not None]
        if not motif_deltas:
            print("No MOTIF-applicable tasks at any rho. Check generator.")
        else:
            print("No significant reduction found at any rho. Mechanism/instrument may be broken.")

    # Verdict
    motif_deltas = [pt['motif_delta_pct_mean'] for pt in curve
                    if pt['motif_delta_pct_mean'] is not None]
    if not motif_deltas:
        verdict = "NO_APPLICABLE_TASKS_ANYWHERE"
    elif rstar is not None:
        verdict = f"MECHANISM_WORKS_R*={rstar}"
    elif all(d > -5 for d in motif_deltas):
        verdict = "MECHANISM_FLAT"
    else:
        verdict = "PARTIAL_SIGNAL"

    # Search-order accounting summary
    print("\n" + "=" * 70)
    print("SEARCH-ORDER ACCOUNTING (macro-prior vs dimension effect)")
    print("=" * 70)
    print("Baseline programs are found at length L in seed ops.")
    print("With-lib programs may be found at length L' < L (dimension) or earlier")
    print("in search ordering (macro-prior, compound tried before seed ops of same L).")
    print()
    for pt in curve:
        if pt['per_seed']:
            all_len_reductions = []
            for r in pt['per_seed']:
                pm = r['partition_motif']
                if pm and pm['length_reduction'] is not None:
                    all_len_reductions.append(pm['length_reduction'])
            if all_len_reductions:
                mean_lr = np.mean(all_len_reductions)
                print(f"  rho={pt['rho']:.2f}: motif tasks mean program-length reduction = {mean_lr:.2f} ops")

    print(f"\nVerdict: {verdict}")

    result_data = {
        "experiment": "E4-rstar-grade",
        "motif": MOTIF,
        "motif_name": MOTIF_NAME,
        "distractor": DISTRACTOR_S,
        "distractor_name": DISTRACTOR_NAME,
        "config": {
            "program_length": PROGRAM_LENGTH,
            "budget": BUDGET,
            "n_source": N_SOURCE,
            "n_held": N_HELD,
            "n_seeds": N_SEEDS,
            "rho_d": RHO_D,
            "clean_pad": CLEAN_PAD,
            "master_seed": MASTER_SEED,
        },
        "framing": (
            "rho=0 is synthetic low-reuse calibration, NOT ARC null. "
            "ARC rho-axis position UNMEASURED — requires ARC-derived program sampling."
        ),
        "rho_values": RHO_VALUES,
        "curve": curve,
        "r_star": rstar,
        "verdict": verdict,
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E4_rstar_grade_result.json"
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResult written to {out_path}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
