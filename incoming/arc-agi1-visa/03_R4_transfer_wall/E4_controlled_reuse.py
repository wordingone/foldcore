"""
E4 Controlled-Reuse Characterization — Leo #11627.

Synthetic generator over the 12-op seed DSL. Plants a MOTIF sub-program
(length-2 composition) with probability rho. Sweeps rho from ~0 (ARC-like)
to high. At each rho: source/held-out split, MDL library, held-out search-cost
WITH vs WITHOUT + R6 ablation + compound-applicability partition.

Output: held-out search-cost-reduction curve vs rho. R* = rho where
significant reduction begins. ARC's location on axis: rho~0.

Consistency check: rho=0 point should replicate ARC null (no reduction,
no MDL-positive compounds). Run FIRST.

Pre-registered outcomes:
- monotone reduction-vs-rho: mechanism validated, R* located.
- flat even at high rho: mechanism/instrument broken (debug).
- threshold curve: R* characterized.
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

# Ops that won't cancel or produce identity easily; used for random padding.
PAD_OPS = ['fh', 'fv', 'tr', 'rot', 'crop', 'dup_h', 'dup_v', 'mir_h', 'mir_v', 'up2', 'down2']
# Exclude 'id' from padding (creates trivially shorter equivalents).

# ── Primary MOTIF ─────────────────────────────────────────────────────────────
MOTIF = ['mir_h', 'mir_v']   # length-2; MDL_gain=1 on ARC at count=3
MOTIF_NAME = 'mir_h__mir_v'

# ── Config ────────────────────────────────────────────────────────────────────
PROGRAM_LENGTH = 4     # ground-truth program length
BUDGET = 35000         # covers all len-4 programs for both 12- and 13-op libraries
N_SOURCE = 100
N_HELD = 100
N_PAIRS = 3
MAX_GRID_AREA = 400    # reject tasks where output area exceeds this
RHO_VALUES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MASTER_SEED = 42


# ── Grid and task generation ──────────────────────────────────────────────────

def apply_op(name, grid, library):
    try:
        g = np.array(grid, dtype=np.int64)
        if name in library:
            for sub in library[name]:
                result = apply_op(sub, g, library)
                if result is None:
                    return None
                g = result
            return g
        return SEED_OPS[name](g)
    except Exception:
        return None


def apply_program(ops, grid, library):
    g = np.array(grid, dtype=np.int64)
    for name in ops:
        result = apply_op(name, g, library)
        if result is None:
            return None
        g = result
    return g


def check_program(ops, task_io, library):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is None or result.shape != out.shape or not np.array_equal(result, out):
            return False
    return True


def generate_input_grid(rng):
    """Random grid 2-4 x 2-4, values 0-5 with at least 2 distinct values."""
    for _ in range(20):
        h = rng.integers(2, 5)
        w = rng.integers(2, 5)
        grid = rng.integers(0, 6, size=(h, w), dtype=np.int64)
        if len(np.unique(grid)) >= 2:
            return grid
    # Fallback: guaranteed 2 values
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[0, 0] = 1
    return grid


def generate_program_ops(rho, rng):
    """
    Generate a length-PROGRAM_LENGTH program.
    With prob rho: include MOTIF at a random position.
    Otherwise: all random PAD_OPS.
    Exclude 'id' to avoid trivial short equivalents.
    """
    ops = [rng.choice(PAD_OPS) for _ in range(PROGRAM_LENGTH)]
    if rng.random() < rho:
        motif_start = rng.integers(0, PROGRAM_LENGTH - len(MOTIF) + 1)
        for i, m in enumerate(MOTIF):
            ops[motif_start + i] = m
    return ops


def program_contains_motif(ops):
    for i in range(len(ops) - len(MOTIF) + 1):
        if ops[i:i + len(MOTIF)] == MOTIF:
            return True
    return False


def generate_task(ops, rng):
    """
    Generate N_PAIRS I/O pairs for program `ops`.
    Returns list of {'input': ..., 'output': ...} or None if invalid.
    """
    lib = {}
    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < 50:
        attempts += 1
        inp = generate_input_grid(rng)
        out = apply_program(ops, inp, lib)
        if out is None:
            continue
        if out.size > MAX_GRID_AREA:
            continue
        if out.shape == inp.shape and np.array_equal(out, inp):
            continue  # identity transform — uninformative
        if out.size == 0:
            continue
        pairs.append({'input': inp.tolist(), 'output': out.tolist()})
    return pairs if len(pairs) == N_PAIRS else None


def search(task_io, library, max_length=PROGRAM_LENGTH, budget=BUDGET):
    """BFS: library compounds first, then seed ops. Returns (program, nodes)."""
    lib_ops = sorted(library.keys())
    seed_names = sorted(SEED_OPS.keys())
    seen = set()
    ordered_ops = []
    for n in lib_ops + seed_names:
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io, library):
                return list(ops), nodes
    return None, nodes


def count_subseq(program, candidate):
    count = 0
    for i in range(len(program) - len(candidate) + 1):
        if tuple(program[i:i + len(candidate)]) == tuple(candidate):
            count += 1
    return count


def mdl_gain(candidate, solved_programs):
    occ = sum(count_subseq(prog, candidate) for prog in solved_programs)
    return occ, occ * (len(candidate) - 1) - len(candidate)


def is_applicable(seed_prog, grown_prog, compound_names):
    for comp in compound_names:
        comp_parts = tuple(comp.split('__'))
        if seed_prog and count_subseq(seed_prog, comp_parts) > 0:
            return True
        if grown_prog and any(op == comp for op in grown_prog):
            return True
    return False


# ── Single rho point ──────────────────────────────────────────────────────────

def run_rho_point(rho, rng_seed):
    rng = np.random.default_rng(rng_seed)

    # ── Generate tasks ─────────────────────────────────────────────────────
    all_tasks = []
    n_attempts = 0
    while len(all_tasks) < N_SOURCE + N_HELD:
        n_attempts += 1
        if n_attempts > (N_SOURCE + N_HELD) * 20:
            break
        ops = generate_program_ops(rho, rng)
        pairs = generate_task(ops, rng)
        if pairs is None:
            continue
        # Quick filter: not trivially solvable at length 1
        trivial = False
        for length1_op in sorted(SEED_OPS.keys()):
            if check_program([length1_op], pairs, {}):
                trivial = True
                break
        if trivial:
            continue
        all_tasks.append({'ops': ops, 'io': pairs, 'has_motif': program_contains_motif(ops)})

    source_tasks = all_tasks[:N_SOURCE]
    held_tasks = all_tasks[N_SOURCE:N_SOURCE + N_HELD]
    n_source_motif = sum(1 for t in source_tasks if t['has_motif'])
    n_held_motif = sum(1 for t in held_tasks if t['has_motif'])

    # ── SOURCE ROUND (seed library) ───────────────────────────────────────
    seed_lib = {}
    source_results = []
    for task in source_tasks:
        prog, cost = search(task['io'], seed_lib)
        source_results.append({'ops': task['ops'], 'program': prog, 'cost': cost,
                                'has_motif': task['has_motif']})

    source_solved = [r for r in source_results if r['program']]
    solved_programs = [r['program'] for r in source_solved if r['program']]

    # ── ABSTRACTION: MDL ──────────────────────────────────────────────────
    subseq_counts = Counter()
    for prog in solved_programs:
        if len(prog) >= 2:
            for length in range(2, len(prog) + 1):
                for start in range(len(prog) - length + 1):
                    sub = tuple(prog[start:start + length])
                    subseq_counts[sub] += 1

    added_ops = {}
    candidates_detail = []
    for sub, cnt in sorted(subseq_counts.items(), key=lambda x: (-x[1], x[0])):
        _, gain = mdl_gain(sub, solved_programs)
        candidates_detail.append({'sub': list(sub), 'count': cnt, 'gain': gain})
        if gain > 0:
            name = '__'.join(sub)
            added_ops[name] = list(sub)

    grown_lib = dict(added_ops)
    compound_names = list(added_ops.keys())

    # ── BASELINE (seed, no compounds) ─────────────────────────────────────
    baseline_results = []
    for task in held_tasks:
        prog, cost = search(task['io'], {})
        baseline_results.append({'program': prog, 'cost': cost,
                                  'has_motif': task['has_motif']})

    # ── WITH-LIBRARY ──────────────────────────────────────────────────────
    withlib_results = []
    for task in held_tasks:
        prog, cost = search(task['io'], grown_lib)
        withlib_results.append({'program': prog, 'cost': cost,
                                 'has_motif': task['has_motif']})

    # ── R6 (frozen=seed) ──────────────────────────────────────────────────
    r6_mean = np.mean([r['cost'] for r in baseline_results])  # frozen=seed=baseline

    # ── PARTITION ─────────────────────────────────────────────────────────
    applicable_ids = set()
    for i in range(len(held_tasks)):
        seed_prog = baseline_results[i]['program']
        grown_prog = withlib_results[i]['program']
        if is_applicable(seed_prog, grown_prog, compound_names):
            applicable_ids.add(i)

    appl_base = [baseline_results[i] for i in range(len(held_tasks)) if i in applicable_ids]
    appl_with = [withlib_results[i] for i in range(len(held_tasks)) if i in applicable_ids]
    nonappl_base = [baseline_results[i] for i in range(len(held_tasks)) if i not in applicable_ids]
    nonappl_with = [withlib_results[i] for i in range(len(held_tasks)) if i not in applicable_ids]

    agg_base_mean = np.mean([r['cost'] for r in baseline_results])
    agg_with_mean = np.mean([r['cost'] for r in withlib_results])
    agg_delta = agg_with_mean - agg_base_mean
    agg_delta_pct = 100 * agg_delta / agg_base_mean if agg_base_mean > 0 else 0

    appl_result = None
    if appl_base:
        appl_base_mean = np.mean([r['cost'] for r in appl_base])
        appl_with_mean = np.mean([r['cost'] for r in appl_with])
        appl_delta = appl_with_mean - appl_base_mean
        appl_delta_pct = 100 * appl_delta / appl_base_mean if appl_base_mean > 0 else 0
        appl_result = {
            'n': len(appl_base),
            'baseline_mean': float(appl_base_mean),
            'withlib_mean': float(appl_with_mean),
            'delta': float(appl_delta),
            'delta_pct': float(appl_delta_pct),
        }

    nonappl_result = None
    if nonappl_base:
        nonappl_base_mean = np.mean([r['cost'] for r in nonappl_base])
        nonappl_with_mean = np.mean([r['cost'] for r in nonappl_with])
        nonappl_delta = nonappl_with_mean - nonappl_base_mean
        nonappl_delta_pct = 100 * nonappl_delta / nonappl_base_mean if nonappl_base_mean > 0 else 0
        nonappl_result = {
            'n': len(nonappl_base),
            'baseline_mean': float(nonappl_base_mean),
            'withlib_mean': float(nonappl_with_mean),
            'delta': float(nonappl_delta),
            'delta_pct': float(nonappl_delta_pct),
        }

    return {
        'rho': rho,
        'n_source': len(source_tasks),
        'n_held': len(held_tasks),
        'n_source_motif': n_source_motif,
        'n_held_motif': n_held_motif,
        'source_solved': len(source_solved),
        'added_ops': list(added_ops.keys()),
        'candidates': candidates_detail[:10],
        'aggregate': {
            'baseline_mean': float(agg_base_mean),
            'withlib_mean': float(agg_with_mean),
            'r6_mean': float(r6_mean),
            'delta': float(agg_delta),
            'delta_pct': float(agg_delta_pct),
        },
        'applicable': appl_result,
        'non_applicable': nonappl_result,
        'n_applicable': len(applicable_ids),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E4 CONTROLLED-REUSE CHARACTERIZATION")
    print(f"Motif: {MOTIF_NAME} = {MOTIF}")
    print(f"Config: program_length={PROGRAM_LENGTH}, budget={BUDGET}")
    print(f"        n_source={N_SOURCE}, n_held={N_HELD}, n_pairs={N_PAIRS}")
    print(f"rho sweep: {RHO_VALUES}")
    print("=" * 70)
    sys.stdout.flush()

    print("\n[CONSISTENCY CHECK] Running rho=0.0 first...")
    sys.stdout.flush()

    curve = []
    rho_seed_offset = 0

    for rho in RHO_VALUES:
        rng_seed = MASTER_SEED + rho_seed_offset
        rho_seed_offset += 1000

        print(f"\n--- rho={rho:.1f} (seed={rng_seed}) ---")
        sys.stdout.flush()

        result = run_rho_point(rho, rng_seed)
        curve.append(result)

        print(f"  Source: {result['n_source_motif']}/{result['n_source']} tasks have motif")
        print(f"  Held:   {result['n_held_motif']}/{result['n_held']} tasks have motif (ground truth)")
        print(f"  Source solved: {result['source_solved']}/{result['n_source']}")
        print(f"  Added ops: {result['added_ops']}")
        print(f"  Applicable (BFS-confirmed): {result['n_applicable']}/{result['n_held']}")

        agg = result['aggregate']
        print(f"  AGGREGATE: baseline={agg['baseline_mean']:.1f}  with-lib={agg['withlib_mean']:.1f}"
              f"  delta={agg['delta']:+.1f} ({agg['delta_pct']:+.1f}%)")

        if result['applicable']:
            a = result['applicable']
            print(f"  APPLICABLE ({a['n']} tasks): baseline={a['baseline_mean']:.1f}"
                  f"  with-lib={a['withlib_mean']:.1f}  delta={a['delta']:+.1f} ({a['delta_pct']:+.1f}%)")
        else:
            print(f"  APPLICABLE: 0 tasks")

        if result['non_applicable']:
            na = result['non_applicable']
            print(f"  NON-APPLICABLE ({na['n']} tasks): baseline={na['baseline_mean']:.1f}"
                  f"  with-lib={na['withlib_mean']:.1f}  delta={na['delta']:+.1f} ({na['delta_pct']:+.1f}%)")

        if rho == 0.0:
            # Consistency check
            if result['n_applicable'] == 0 and not result['added_ops']:
                print("  [CONSISTENCY CHECK PASS] rho=0 -> NO_APPLICABLE_TASKS, no library formed.")
                print("  Calibration confirmed: axis includes ARC regime.")
            else:
                print(f"  [CONSISTENCY CHECK PARTIAL] rho=0: {result['n_applicable']} applicable,"
                      f" {len(result['added_ops'])} compounds added. Check generator.")

        sys.stdout.flush()

    # ── Curve summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("REDUCTION-VS-RHO CURVE (applicable partition)")
    print("=" * 70)
    print(f"{'rho':>6}  {'applicable':>10}  {'appl_baseline':>14}  {'appl_withlib':>13}  {'delta%':>8}  {'compounds':>10}")
    print("-" * 70)

    rstar = None
    for pt in curve:
        rho = pt['rho']
        n_appl = pt['n_applicable']
        compounds = len(pt['added_ops'])
        if pt['applicable']:
            a = pt['applicable']
            delta_pct = a['delta_pct']
            marker = " <-- R*" if rstar is None and delta_pct < -10 else ""
            if rstar is None and delta_pct < -10:
                rstar = rho
            print(f"  {rho:>4.1f}  {n_appl:>10}  {a['baseline_mean']:>14.1f}  "
                  f"{a['withlib_mean']:>13.1f}  {delta_pct:>8.1f}%  {compounds:>10}{marker}")
        else:
            print(f"  {rho:>4.1f}  {n_appl:>10}  {'N/A':>14}  {'N/A':>13}  {'N/A':>8}  {compounds:>10}")

    print()
    if rstar is not None:
        print(f"R* (first rho with >10% cost reduction) = {rstar}")
        print(f"ARC location: rho~0 (mir_h__mir_v appears 3/400 tasks ~ 0.75%)")
        print(f"ARC << R* => self-compilation requires significantly higher reuse density than ARC provides.")
    else:
        if all(pt['n_applicable'] == 0 for pt in curve):
            print("No applicable tasks at any rho. Check generator: motifs not appearing in BFS solutions.")
        else:
            print("No significant reduction found at any rho. Mechanism/instrument may be broken.")

    # ── Verdict ────────────────────────────────────────────────────────────
    applicable_deltas = [pt['applicable']['delta_pct'] for pt in curve if pt['applicable']]
    if not applicable_deltas:
        verdict = "NO_APPLICABLE_TASKS_ANYWHERE"
    elif rstar is not None:
        verdict = f"MECHANISM_WORKS_R*={rstar}"
    elif all(d > -5 for d in applicable_deltas):
        verdict = "MECHANISM_FLAT"
    else:
        verdict = "PARTIAL_SIGNAL"

    print(f"\nVerdict: {verdict}")

    result_data = {
        "experiment": "E4-controlled-reuse",
        "motif": MOTIF,
        "motif_name": MOTIF_NAME,
        "config": {
            "program_length": PROGRAM_LENGTH,
            "budget": BUDGET,
            "n_source": N_SOURCE,
            "n_held": N_HELD,
            "n_pairs": N_PAIRS,
            "master_seed": MASTER_SEED,
        },
        "rho_values": RHO_VALUES,
        "curve": curve,
        "r_star": rstar,
        "verdict": verdict,
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E4_controlled_reuse_result.json"
    with open(out_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == '__main__':
    main()
