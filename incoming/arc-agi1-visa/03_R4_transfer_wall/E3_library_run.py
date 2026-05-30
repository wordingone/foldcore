"""
E3 Library Run — real ARC-AGI-1 tasks with compound-applicability partition.
Leo directive #11619, 2026-05-30.

Same seed/budget as baseline (one variable): 12-op DSL, max_length=5, budget=3000.
Random 200/200 source/held-out split (split_seed=42).

Partition: held-out tasks split by compound-applicability.
  compound-applicable = seed program OR grown program contains a learned compound
                        as a sub-sequence (or uses a compound op directly).
  Interpretation (pre-registered):
    applicable cheaper + others flat  -> MECHANISM_WORKS; next = raise yield.
    applicable NOT cheaper            -> MECHANISM_BROKEN; debug.
    no applicable tasks               -> NO_APPLICABLE_TASKS; compounds too narrow; next = raise yield.
"""

import numpy as np
import json
import itertools
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

ARC_TRAIN = "incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/training"

# ── 12-op seed DSL (identical to E3_seed_baseline.py) ──────────────────────────

def _bg(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(x):
    nz = np.argwhere(x != _bg(x))
    if len(nz) == 0:
        return x
    r0, r1 = nz[:, 0].min(), nz[:, 0].max()
    c0, c1 = nz[:, 1].min(), nz[:, 1].max()
    return x[r0:r1+1, c0:c1+1]

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

# ── Search ──────────────────────────────────────────────────────────────────────

def apply_program(ops, grid, library):
    try:
        g = np.array(grid, dtype=np.int64)
        for name in ops:
            g = library[name](g)
        return g
    except Exception:
        return None

def check_program(ops, task_io, library):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is None or result.shape != out.shape or not np.array_equal(result, out):
            return False
    return True

def search(task_io, library, max_length=5, budget=3000):
    op_names = sorted(library.keys())
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(op_names, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io, library):
                return list(ops), nodes
    return None, nodes

# ── Trace ───────────────────────────────────────────────────────────────────────

@dataclass
class Trace:
    task_id: str
    program: Optional[List[str]]
    outcome: str  # 'solved' | 'failed'
    cost: int

def solve_and_trace(task_id, task_io, library, max_length=5, budget=3000):
    prog, cost = search(task_io, library, max_length, budget)
    return Trace(task_id, prog, 'solved' if prog is not None else 'failed', cost)

# ── MDL / abstraction ───────────────────────────────────────────────────────────

def count_subseq(seq, subseq):
    n, m = len(seq), len(subseq)
    count, i = 0, 0
    while i <= n - m:
        if tuple(seq[i:i+m]) == subseq:
            count += 1; i += m
        else:
            i += 1
    return count

def extract_candidates(traces, min_count=2, min_length=2):
    freq = defaultdict(int)
    solved = [t for t in traces if t.outcome == 'solved' and t.program]
    for trace in solved:
        seen = set()
        for length in range(min_length, len(trace.program) + 1):
            for i in range(len(trace.program) - length + 1):
                sub = tuple(trace.program[i:i+length])
                if sub not in seen:
                    freq[sub] += 1
                    seen.add(sub)
    return [(sub, cnt) for sub, cnt in freq.items() if cnt >= min_count]

def mdl_gain(candidate, traces):
    solved = [t for t in traces if t.outcome == 'solved' and t.program]
    occ = sum(count_subseq(t.program, candidate) for t in solved)
    return occ * (len(candidate) - 1) - len(candidate)

def abstract_library(traces, library, min_count=2):
    candidates = extract_candidates(traces, min_count=min_count)
    new_lib = dict(library)
    added = []
    for subseq, _ in sorted(candidates, key=lambda x: (-x[1], -len(x[0]))):
        gain = mdl_gain(subseq, traces)
        if gain <= 0:
            continue
        op_name = '__'.join(subseq)
        if op_name in new_lib:
            continue
        lib_snap = dict(new_lib)
        ops_seq = list(subseq)
        def make_fn(seq, lib):
            def fn(g):
                for op in seq:
                    g = lib[op](g)
                return g
            return fn
        new_lib[op_name] = make_fn(ops_seq, lib_snap)
        added.append(op_name)
    return new_lib, added, candidates

# ── Compound-applicability ──────────────────────────────────────────────────────

def is_applicable(seed_prog, grown_prog, compound_names):
    """True if either program contains a learned compound as a sub-sequence."""
    for comp in compound_names:
        comp_parts = tuple(comp.split('__'))
        if seed_prog and count_subseq(seed_prog, comp_parts) > 0:
            return True
        if grown_prog and any(op == comp for op in grown_prog):
            return True
    return False

# ── Data loading ────────────────────────────────────────────────────────────────

def load_tasks():
    tasks = []
    for fname in sorted(os.listdir(ARC_TRAIN)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(ARC_TRAIN, fname)) as f:
            d = json.load(f)
        task_io = [{'input': p['input'], 'output': p['output']} for p in d['train']]
        tasks.append({'id': fname[:-5], 'io': task_io})
    return tasks

# ── Partition stats ─────────────────────────────────────────────────────────────

def partition_stats(ids, seed_tr, grown_tr):
    if not ids:
        return None
    s_costs = [seed_tr[i].cost for i in ids]
    g_costs = [grown_tr[i].cost for i in ids]
    s_mean = float(np.mean(s_costs))
    g_mean = float(np.mean(g_costs))
    delta = s_mean - g_mean
    delta_pct = (delta / s_mean * 100) if s_mean > 0 else 0.0
    return {
        'n': len(ids),
        'seed_mean_cost': s_mean,
        'grown_mean_cost': g_mean,
        'delta_cost': delta,
        'delta_pct': delta_pct,
        'seed_solved': sum(1 for i in ids if seed_tr[i].outcome == 'solved'),
        'grown_solved': sum(1 for i in ids if grown_tr[i].outcome == 'solved'),
    }

# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    MAX_LENGTH = 5
    BUDGET = 3000
    MIN_COUNT = 2
    SPLIT_SEED = 42
    N_SOURCE = 200

    print("E3 LIBRARY RUN — ARC-AGI-1 with compound-applicability partition")
    print(f"Config: 12-op DSL, max_length={MAX_LENGTH}, budget={BUDGET}, split_seed={SPLIT_SEED}")
    print("Partition: held-out by compound-applicability (Leo #11619 pre-registered)")
    print("=" * 70)
    sys.stdout.flush()

    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")

    rng = random.Random(SPLIT_SEED)
    indices = list(range(len(tasks)))
    rng.shuffle(indices)
    source_tasks = [tasks[i] for i in indices[:N_SOURCE]]
    held_tasks   = [tasks[i] for i in indices[N_SOURCE:]]
    print(f"Split: {len(source_tasks)} source / {len(held_tasks)} held-out")
    sys.stdout.flush()

    seed_library = dict(SEED_OPS)

    # ── SOURCE ROUND ──────────────────────────────────────────────────────────
    print(f"\nSOURCE ROUND ({len(source_tasks)} tasks, seed library)...")
    sys.stdout.flush()
    source_traces = []
    for task in source_tasks:
        t = solve_and_trace(task['id'], task['io'], seed_library, MAX_LENGTH, BUDGET)
        source_traces.append(t)
    source_solved = [t for t in source_traces if t.outcome == 'solved']
    print(f"  Solved: {len(source_solved)}/{len(source_tasks)}")
    for t in source_solved:
        print(f"    {t.task_id}: {t.program}  cost={t.cost}")
    sys.stdout.flush()

    # ── ABSTRACTION ───────────────────────────────────────────────────────────
    print(f"\nABSTRACTION: MDL over {len(source_solved)} solved source traces...")
    sys.stdout.flush()
    grown_library, added_ops, candidates = abstract_library(
        source_traces, seed_library, min_count=MIN_COUNT)
    print(f"  Candidates ({len(candidates)}):")
    for sub, cnt in candidates[:15]:
        gain = mdl_gain(sub, source_traces)
        print(f"    {'__'.join(sub)}  count={cnt}  MDL_gain={gain}")
    print(f"  Added ops ({len(added_ops)}): {added_ops}")
    print(f"  Library: {len(seed_library)} -> {len(grown_library)} ops")
    sys.stdout.flush()

    # ── BASELINE: held-out with seed ──────────────────────────────────────────
    print(f"\nBASELINE (seed, {len(seed_library)} ops): {len(held_tasks)} held-out tasks...")
    sys.stdout.flush()
    seed_traces = {}
    for i, task in enumerate(held_tasks):
        t = solve_and_trace(task['id'], task['io'], seed_library, MAX_LENGTH, BUDGET)
        seed_traces[task['id']] = t
        if t.outcome == 'solved' or (i % 50 == 0):
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: {t.outcome}  cost={t.cost}"
                  + (f"  prog={t.program}" if t.outcome == 'solved' else ""))
            sys.stdout.flush()
    seed_solved = sum(1 for t in seed_traces.values() if t.outcome == 'solved')
    seed_cost_mean = float(np.mean([t.cost for t in seed_traces.values()]))
    print(f"  Baseline: {seed_solved}/{len(held_tasks)} solved, mean_cost={seed_cost_mean:.1f}")
    sys.stdout.flush()

    # ── WITH-LIBRARY: held-out with grown ─────────────────────────────────────
    print(f"\nWITH-LIBRARY (grown, {len(grown_library)} ops): {len(held_tasks)} held-out tasks...")
    sys.stdout.flush()
    grown_traces = {}
    for i, task in enumerate(held_tasks):
        t = solve_and_trace(task['id'], task['io'], grown_library, MAX_LENGTH, BUDGET)
        grown_traces[task['id']] = t
        if t.outcome == 'solved' or (i % 50 == 0):
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: {t.outcome}  cost={t.cost}"
                  + (f"  prog={t.program}" if t.outcome == 'solved' else ""))
            sys.stdout.flush()
    grown_solved = sum(1 for t in grown_traces.values() if t.outcome == 'solved')
    grown_cost_mean = float(np.mean([t.cost for t in grown_traces.values()]))
    print(f"  With-lib: {grown_solved}/{len(held_tasks)} solved, mean_cost={grown_cost_mean:.1f}")
    sys.stdout.flush()

    # ── PARTITION ─────────────────────────────────────────────────────────────
    compound_names = added_ops
    applicable_ids, non_applicable_ids = [], []
    for task in held_tasks:
        tid = task['id']
        sp = seed_traces[tid].program
        gp = grown_traces[tid].program
        if compound_names and is_applicable(sp, gp, compound_names):
            applicable_ids.append(tid)
        else:
            non_applicable_ids.append(tid)

    app_stats = partition_stats(applicable_ids, seed_traces, grown_traces)
    non_app_stats = partition_stats(non_applicable_ids, seed_traces, grown_traces)

    # ── R6 (baseline = frozen seed) ───────────────────────────────────────────
    r6_cost_mean = seed_cost_mean

    # ── AGGREGATE ─────────────────────────────────────────────────────────────
    agg_delta = seed_cost_mean - grown_cost_mean
    agg_delta_pct = (agg_delta / seed_cost_mean * 100) if seed_cost_mean > 0 else 0.0

    print("\n" + "=" * 70)
    print("E3 LIBRARY RUN RESULT")
    print("=" * 70)
    print(f"Seed ops ({len(seed_library)}): {sorted(seed_library.keys())}")
    print(f"Added ops ({len(added_ops)}): {added_ops}")
    print()
    print(f"AGGREGATE ({len(held_tasks)} held-out tasks):")
    print(f"  BASELINE (seed):   mean_cost={seed_cost_mean:.1f}  solved={seed_solved}/{len(held_tasks)}")
    print(f"  WITH-LIB (grown):  mean_cost={grown_cost_mean:.1f}  solved={grown_solved}/{len(held_tasks)}")
    print(f"  R6 (frozen=seed):  mean_cost={r6_cost_mean:.1f}  (= baseline by construction)")
    print(f"  Aggregate delta:   {agg_delta:+.1f} nodes ({agg_delta_pct:+.1f}%)")
    print()

    if app_stats:
        print(f"COMPOUND-APPLICABLE ({app_stats['n']} tasks):")
        print(f"  Seed:    mean_cost={app_stats['seed_mean_cost']:.1f}  solved={app_stats['seed_solved']}")
        print(f"  With-lib: mean_cost={app_stats['grown_mean_cost']:.1f}  solved={app_stats['grown_solved']}")
        print(f"  Delta:   {app_stats['delta_cost']:+.1f} ({app_stats['delta_pct']:+.1f}%)")
    else:
        print("COMPOUND-APPLICABLE: 0 tasks — compounds too narrow")

    print()
    if non_app_stats:
        print(f"NON-APPLICABLE ({non_app_stats['n']} tasks):")
        print(f"  Seed:    mean_cost={non_app_stats['seed_mean_cost']:.1f}  solved={non_app_stats['seed_solved']}")
        print(f"  With-lib: mean_cost={non_app_stats['grown_mean_cost']:.1f}  solved={non_app_stats['grown_solved']}")
        print(f"  Delta:   {non_app_stats['delta_cost']:+.1f} ({non_app_stats['delta_pct']:+.1f}%)")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    if len(applicable_ids) == 0:
        verdict = "NO_APPLICABLE_TASKS — compounds too narrow; next: raise trace yield (budget/length)"
    elif app_stats and app_stats['delta_cost'] > 0:
        non_delta = non_app_stats['delta_cost'] if non_app_stats else 0.0
        if abs(non_delta) < app_stats['delta_cost']:
            verdict = ("MECHANISM_WORKS — applicable cheaper + others flat; "
                       "ceiling = compound-reuse frequency; next: raise yield")
        else:
            verdict = (f"MIXED — applicable_delta={app_stats['delta_cost']:+.1f} "
                       f"non_applicable_delta={non_delta:+.1f}; investigate")
    else:
        verdict = "MECHANISM_BROKEN — applicable NOT cheaper; debug instrument/mechanism"

    print(f"\nVerdict: {verdict}")

    # ── OUTPUT ────────────────────────────────────────────────────────────────
    result = {
        "experiment": "E3-library-run",
        "config": {
            "n_tasks": len(tasks),
            "n_source": len(source_tasks),
            "n_held": len(held_tasks),
            "max_length": MAX_LENGTH,
            "budget": BUDGET,
            "min_count": MIN_COUNT,
            "split_seed": SPLIT_SEED,
        },
        "seed_ops": sorted(seed_library.keys()),
        "added_ops": added_ops,
        "grown_lib_size": len(grown_library),
        "source_solved": len(source_solved),
        "source_solved_programs": [
            {"id": t.task_id, "program": t.program, "cost": t.cost}
            for t in source_solved
        ],
        "candidates": [
            {"subseq": list(s), "count": c, "mdl_gain": mdl_gain(s, source_traces)}
            for s, c in candidates
        ],
        "aggregate": {
            "seed_mean_cost": seed_cost_mean,
            "grown_mean_cost": grown_cost_mean,
            "r6_mean_cost": r6_cost_mean,
            "delta_cost": agg_delta,
            "delta_pct": agg_delta_pct,
            "seed_solved": seed_solved,
            "grown_solved": grown_solved,
        },
        "partition": {
            "compound_applicable": app_stats,
            "applicable_task_ids": applicable_ids,
            "non_applicable": non_app_stats,
            "non_applicable_task_ids": non_applicable_ids,
        },
        "verdict": verdict,
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_library_run_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == '__main__':
    main()
