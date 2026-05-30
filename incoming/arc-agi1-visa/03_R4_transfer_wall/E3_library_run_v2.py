"""
E3 library run v2 — budget=10000, max_length=5.
ONE variable changed from v1 (budget 3000→10000).
Same 200/200 split (seed=42). Compound-applicability partition (Leo #11621).

Pre-registered outcomes:
- library forms + applicable cheaper → MECHANISM_WORKS
- library forms + applicable NOT cheaper → MECHANISM_BROKEN
- no applicable tasks → NO_APPLICABLE_TASKS (then max_length is next lever)
"""

import numpy as np
import json
import itertools
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Optional

ARC_TRAIN = "incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/training"

BUDGET = 10000
MAX_LENGTH = 5
N_SOURCE = 200
SPLIT_SEED = 42


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


def apply_op(name, grid, library):
    try:
        g = np.array(grid, dtype=np.int64)
        if name in library:
            for sub_name in library[name]:
                g = apply_op(sub_name, g, library)
            return g
        return SEED_OPS[name](g)
    except Exception:
        return None


def apply_program(ops, grid, library):
    try:
        g = np.array(grid, dtype=np.int64)
        for name in ops:
            g = apply_op(name, g, library)
            if g is None:
                return None
        return g
    except Exception:
        return None


def check_program(ops, task_io, library):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is None:
            return False
        if result.shape != out.shape:
            return False
        if not np.array_equal(result, out):
            return False
    return True


def search(task_io, library, max_length=MAX_LENGTH, budget=BUDGET):
    op_names = sorted(library.keys()) + sorted(SEED_OPS.keys())
    # deduplicate, preserve order: library compounds first
    seen = set()
    ordered_ops = []
    for n in op_names:
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
    if not program or len(candidate) > len(program):
        return 0
    count = 0
    for i in range(len(program) - len(candidate) + 1):
        if tuple(program[i:i + len(candidate)]) == tuple(candidate):
            count += 1
    return count


def mdl_gain(candidate, solved_programs):
    occ = sum(count_subseq(prog, candidate) for prog in solved_programs)
    gain = occ * (len(candidate) - 1) - len(candidate)
    return occ, gain


def is_applicable(seed_prog, grown_prog, compound_names):
    for comp in compound_names:
        comp_parts = tuple(comp.split('__'))
        if seed_prog and count_subseq(seed_prog, comp_parts) > 0:
            return True
        if grown_prog and any(op == comp for op in grown_prog):
            return True
    return False


def load_tasks():
    tasks = []
    for fname in sorted(os.listdir(ARC_TRAIN)):
        if not fname.endswith('.json'):
            continue
        path = os.path.join(ARC_TRAIN, fname)
        with open(path) as f:
            d = json.load(f)
        task_io = [{'input': p['input'], 'output': p['output']} for p in d['train']]
        tasks.append({'id': fname[:-5], 'io': task_io})
    return tasks


def main():
    print(f"E3 LIBRARY RUN v2 — ARC-AGI-1 with compound-applicability partition")
    print(f"Config: 12-op DSL, max_length={MAX_LENGTH}, budget={BUDGET}, split_seed={SPLIT_SEED}")
    print(f"Partition: held-out by compound-applicability (Leo #11621 pre-registered)")
    print("=" * 70)
    sys.stdout.flush()

    all_tasks = load_tasks()
    print(f"Loaded {len(all_tasks)} tasks")

    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(all_tasks))
    source_idx = sorted(indices[:N_SOURCE])
    held_idx = sorted(indices[N_SOURCE:])

    source_tasks = [all_tasks[i] for i in source_idx]
    held_tasks = [all_tasks[i] for i in held_idx]
    print(f"Split: {len(source_tasks)} source / {len(held_tasks)} held-out")
    sys.stdout.flush()

    # ── SOURCE ROUND (seed library) ──────────────────────────────────────────
    print(f"\nSOURCE ROUND ({len(source_tasks)} tasks, seed library)...")
    sys.stdout.flush()

    seed_lib = {}  # no compounds yet
    source_results = []
    for i, task in enumerate(source_tasks):
        prog, cost = search(task['io'], seed_lib)
        outcome = 'solved' if prog is not None else 'failed'
        source_results.append({'id': task['id'], 'outcome': outcome, 'program': prog, 'cost': cost})
        if prog is not None:
            print(f"  {task['id']}: {prog}  cost={cost}")
            sys.stdout.flush()

    source_solved = [r for r in source_results if r['outcome'] == 'solved']
    print(f"  Solved: {len(source_solved)}/{len(source_tasks)}")
    sys.stdout.flush()

    # ── ABSTRACTION: MDL anti-unification ────────────────────────────────────
    print(f"\nABSTRACTION: MDL over {len(source_solved)} solved source traces...")
    sys.stdout.flush()

    solved_programs = [r['program'] for r in source_solved if r['program']]

    subseq_counts = Counter()
    for prog in solved_programs:
        if len(prog) >= 2:
            for length in range(2, len(prog) + 1):
                for start in range(len(prog) - length + 1):
                    sub = tuple(prog[start:start + length])
                    subseq_counts[sub] += 1

    candidates = [(sub, cnt) for sub, cnt in subseq_counts.items()]
    candidates.sort(key=lambda x: (-x[1], x[0]))

    print(f"  Candidates ({len(candidates)}):")
    added_ops = {}
    for sub, cnt in candidates:
        occ, gain = mdl_gain(sub, solved_programs)
        print(f"    {'__'.join(sub)}  count={cnt}  MDL_gain={gain}")
        if gain > 0:
            name = '__'.join(sub)
            added_ops[name] = list(sub)
    sys.stdout.flush()

    grown_lib = dict(added_ops)
    print(f"  Added ops ({len(added_ops)}): {list(added_ops.keys())}")
    print(f"  Library: 12 -> {12 + len(added_ops)} ops")
    sys.stdout.flush()

    compound_names = list(added_ops.keys())

    # ── BASELINE (seed library, 12 ops) ──────────────────────────────────────
    print(f"\nBASELINE (seed, 12 ops): {len(held_tasks)} held-out tasks...")
    sys.stdout.flush()

    baseline_results = []
    for i, task in enumerate(held_tasks):
        prog, cost = search(task['io'], {})
        outcome = 'solved' if prog is not None else 'failed'
        baseline_results.append({'id': task['id'], 'outcome': outcome, 'program': prog, 'cost': cost})
        if prog is not None or (i % 50 == 0):
            suffix = f"  prog={prog}  cost={cost}" if prog else f"  cost={cost}"
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: {outcome}{suffix}")
            sys.stdout.flush()

    baseline_solved = [r for r in baseline_results if r['outcome'] == 'solved']
    baseline_costs = [r['cost'] for r in baseline_results]
    baseline_mean = np.mean(baseline_costs)
    print(f"  Baseline: {len(baseline_solved)}/{len(held_tasks)} solved, mean_cost={baseline_mean:.1f}")
    sys.stdout.flush()

    # ── WITH-LIBRARY (grown library) ─────────────────────────────────────────
    print(f"\nWITH-LIBRARY (grown, {12 + len(added_ops)} ops): {len(held_tasks)} held-out tasks...")
    sys.stdout.flush()

    withlib_results = []
    for i, task in enumerate(held_tasks):
        prog, cost = search(task['io'], grown_lib)
        outcome = 'solved' if prog is not None else 'failed'
        withlib_results.append({'id': task['id'], 'outcome': outcome, 'program': prog, 'cost': cost})
        if prog is not None or (i % 50 == 0):
            suffix = f"  prog={prog}  cost={cost}" if prog else f"  cost={cost}"
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: {outcome}{suffix}")
            sys.stdout.flush()

    withlib_solved = [r for r in withlib_results if r['outcome'] == 'solved']
    withlib_costs = [r['cost'] for r in withlib_results]
    withlib_mean = np.mean(withlib_costs)
    print(f"  With-lib: {len(withlib_solved)}/{len(held_tasks)} solved, mean_cost={withlib_mean:.1f}")
    sys.stdout.flush()

    # ── COMPOUND-APPLICABILITY PARTITION ─────────────────────────────────────
    applicable_ids = set()
    for i, task in enumerate(held_tasks):
        seed_prog = baseline_results[i]['program']
        grown_prog = withlib_results[i]['program']
        if is_applicable(seed_prog, grown_prog, compound_names):
            applicable_ids.add(task['id'])

    applicable_baseline = [r for r in baseline_results if r['id'] in applicable_ids]
    applicable_withlib = [r for r in withlib_results if r['id'] in applicable_ids]
    nonappl_baseline = [r for r in baseline_results if r['id'] not in applicable_ids]
    nonappl_withlib = [r for r in withlib_results if r['id'] not in applicable_ids]

    # ── R6 ABLATION (frozen=seed, same as baseline) ───────────────────────────
    # Frozen library = seed = baseline (no compounds added = 0 ops added)
    r6_mean = baseline_mean  # frozen = seed always

    # ── RESULT ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("E3 LIBRARY RUN v2 RESULT")
    print("=" * 70)
    print(f"Seed ops (12): {sorted(SEED_OPS.keys())}")
    print(f"Added ops ({len(added_ops)}): {list(added_ops.keys())}")
    print()

    agg_delta = withlib_mean - baseline_mean
    agg_delta_pct = 100 * agg_delta / baseline_mean if baseline_mean > 0 else 0

    print("AGGREGATE (held-out tasks):")
    print(f"  BASELINE (seed):   mean_cost={baseline_mean:.1f}  solved={len(baseline_solved)}/{len(held_tasks)}")
    print(f"  WITH-LIB (grown):  mean_cost={withlib_mean:.1f}  solved={len(withlib_solved)}/{len(held_tasks)}")
    print(f"  R6 (frozen=seed):  mean_cost={r6_mean:.1f}  (= baseline by construction)")
    print(f"  Aggregate delta:   {agg_delta:+.1f} nodes ({agg_delta_pct:+.1f}%)")
    print()

    if len(applicable_ids) == 0:
        verdict = "NO_APPLICABLE_TASKS"
        print(f"COMPOUND-APPLICABLE: 0 tasks — compounds too narrow")
    else:
        appl_base_mean = np.mean([r['cost'] for r in applicable_baseline])
        appl_with_mean = np.mean([r['cost'] for r in applicable_withlib])
        appl_delta = appl_with_mean - appl_base_mean
        appl_solved_base = sum(1 for r in applicable_baseline if r['outcome'] == 'solved')
        appl_solved_with = sum(1 for r in applicable_withlib if r['outcome'] == 'solved')

        print(f"COMPOUND-APPLICABLE ({len(applicable_ids)} tasks):")
        print(f"  Seed:    mean_cost={appl_base_mean:.1f}  solved={appl_solved_base}")
        print(f"  With-lib: mean_cost={appl_with_mean:.1f}  solved={appl_solved_with}")
        print(f"  Delta:   {appl_delta:+.1f}")

        if appl_delta < -10:
            verdict = "MECHANISM_WORKS"
        else:
            verdict = "MECHANISM_BROKEN"

    if len(nonappl_baseline) > 0:
        nonappl_base_mean = np.mean([r['cost'] for r in nonappl_baseline])
        nonappl_with_mean = np.mean([r['cost'] for r in nonappl_withlib])
        nonappl_delta = nonappl_with_mean - nonappl_base_mean
        nonappl_solved_base = sum(1 for r in nonappl_baseline if r['outcome'] == 'solved')
        nonappl_solved_with = sum(1 for r in nonappl_withlib if r['outcome'] == 'solved')
        print(f"\nNON-APPLICABLE ({len(nonappl_baseline)} tasks):")
        print(f"  Seed:    mean_cost={nonappl_base_mean:.1f}  solved={nonappl_solved_base}")
        print(f"  With-lib: mean_cost={nonappl_with_mean:.1f}  solved={nonappl_solved_with}")
        print(f"  Delta:   {nonappl_delta:+.1f} ({100*nonappl_delta/nonappl_base_mean:+.1f}%)")

    print(f"\nVerdict: {verdict}")
    sys.stdout.flush()

    result = {
        "experiment": "E3-library-run-v2",
        "config": {"max_length": MAX_LENGTH, "budget": BUDGET, "split_seed": SPLIT_SEED,
                   "n_source": N_SOURCE},
        "seed_ops": sorted(SEED_OPS.keys()),
        "added_ops": list(added_ops.keys()),
        "added_ops_detail": added_ops,
        "source_solved": len(source_solved),
        "source_total": len(source_tasks),
        "mdl_candidates": [
            {"sub": list(sub), "count": cnt,
             "mdl_gain": cnt * (len(sub) - 1) - len(sub)}
            for sub, cnt in candidates
        ],
        "aggregate": {
            "baseline_mean_cost": float(baseline_mean),
            "withlib_mean_cost": float(withlib_mean),
            "r6_mean_cost": float(r6_mean),
            "baseline_solved": len(baseline_solved),
            "withlib_solved": len(withlib_solved),
            "delta": float(agg_delta),
            "delta_pct": float(agg_delta_pct),
        },
        "compound_applicable_count": len(applicable_ids),
        "verdict": verdict,
    }

    if len(applicable_ids) > 0:
        result["applicable"] = {
            "baseline_mean_cost": float(appl_base_mean),
            "withlib_mean_cost": float(appl_with_mean),
            "delta": float(appl_delta),
            "baseline_solved": appl_solved_base,
            "withlib_solved": appl_solved_with,
        }
        result["non_applicable"] = {
            "baseline_mean_cost": float(nonappl_base_mean),
            "withlib_mean_cost": float(nonappl_with_mean),
            "delta": float(nonappl_delta),
            "baseline_solved": nonappl_solved_base,
            "withlib_solved": nonappl_solved_with,
        }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_library_run_v2_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == '__main__':
    main()
