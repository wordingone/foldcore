"""
E3 seed baseline — 12-op fixed seed DSL on all 400 ARC-AGI-1 training tasks.
Runs BEFORE the library experiment to confirm the seed generates traces.
Reports: solve-rate + median search-cost.

Leo directive #11617: mail seed baseline FIRST.
"""

import numpy as np
import json
import itertools
import os
import sys

ARC_TRAIN = "incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/training"

# ── Seed DSL: 12 fixed global ops (no data-fitted transforms) ─────────────────

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

OP_NAMES = sorted(SEED_OPS.keys())


def apply_program(ops, grid):
    try:
        g = np.array(grid, dtype=np.int64)
        for name in ops:
            g = SEED_OPS[name](g)
        return g
    except Exception:
        return None


def check_program(ops, task_io):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp)
        if result is None:
            return False
        if result.shape != out.shape:
            return False
        if not np.array_equal(result, out):
            return False
    return True


def search(task_io, max_length=5, budget=3000):
    """Returns (program_ops or None, nodes_expanded)."""
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(OP_NAMES, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io):
                return list(ops), nodes
    return None, nodes


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
    print("E3 SEED BASELINE — 12-op fixed DSL on 400 ARC-AGI-1 training tasks")
    print(f"Seed ops ({len(OP_NAMES)}): {OP_NAMES}")
    print(f"Search: max_length=5, budget=3000 nodes/task")
    print("=" * 70)
    sys.stdout.flush()

    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks from {ARC_TRAIN}")
    sys.stdout.flush()

    results = []
    for i, task in enumerate(tasks):
        prog, cost = search(task['io'], max_length=5, budget=3000)
        outcome = 'solved' if prog is not None else 'failed'
        results.append({
            'id': task['id'],
            'outcome': outcome,
            'program': prog,
            'cost': cost,
        })
        if prog is not None or (i % 50 == 0):
            print(f"  [{i+1:3d}/400] {task['id']}: {outcome}"
                  + (f"  prog={prog}  cost={cost}" if prog else f"  cost={cost}"))
            sys.stdout.flush()

    solved = [r for r in results if r['outcome'] == 'solved']
    failed = [r for r in results if r['outcome'] == 'failed']
    costs_all = [r['cost'] for r in results]
    costs_solved = [r['cost'] for r in solved]

    print("\n" + "=" * 70)
    print("SEED BASELINE RESULT")
    print("=" * 70)
    print(f"Total tasks:     {len(tasks)}")
    print(f"Solved:          {len(solved)}/{len(tasks)} ({100*len(solved)/len(tasks):.1f}%)")
    print(f"Failed:          {len(failed)}/{len(tasks)}")
    print(f"Median cost (all):    {np.median(costs_all):.0f} nodes")
    print(f"Median cost (solved): {np.median(costs_solved):.0f} nodes" if solved else "Median cost (solved): N/A")
    print()
    print("Solved tasks:")
    for r in solved:
        print(f"  {r['id']}: {r['program']}  cost={r['cost']}")

    # Structured result
    result = {
        "experiment": "E3-seed-baseline",
        "seed_ops": OP_NAMES,
        "config": {"max_length": 5, "budget": 3000},
        "n_tasks": len(tasks),
        "n_solved": len(solved),
        "solve_rate_pct": 100 * len(solved) / len(tasks),
        "median_cost_all": float(np.median(costs_all)),
        "median_cost_solved": float(np.median(costs_solved)) if solved else None,
        "solved_tasks": [r['id'] for r in solved],
        "solved_programs": [r['program'] for r in solved],
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_seed_baseline_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")


if __name__ == '__main__':
    main()
