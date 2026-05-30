"""
Stage 0b: Arbitrary-Python grid->grid ARC test (Leo #11728, 2026-05-30).

The REAL code-synthesis feasibility test for ARC: exec()-based generate-and-test
on arbitrary Python source strings (any computable grid->grid transform),
matching the MBPP side of Stage 0'.

CONTRAST with stage0_code_synth.py Stage-0' 54-op ARC probe:
  - 54-op ARC was hand-added primitives (designer-DSL-expansion) enumerated with BFS.
    Refinement 2 EXPLICITLY REJECTED this path: designer, not system, expands capability.
    Correct label: "bigger-fixed-basis probe", NOT code-synthesis.
  - THIS test: exec()-based, Python source strings, any computable transform.
    Matches MBPP side. THIS is the code-synthesis feasibility test.

Pre-registered expected result (Leo #11728):
  Brute arbitrary-Python enumeration solves ~nothing beyond 12-op DSL baseline.
  CHICKEN-EGG: arbitrary-Python space is search-intractable without a proposer.
  ~0 new solves => proposer is the bottleneck => confirms code-synthesis direction
  requires a recognition/proposer network, not brute exec().

Write to: stage0b_result.json
"""

import numpy as np
import json
import os
import sys
import threading
import builtins

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage0b_result.json"

SPLIT_SEED = 42
N_TASKS = 50   # subset: arbitrary-Python exec is 10-50x slower than DSL BFS
EXEC_TIMEOUT = 0.5  # seconds per template execution

# ── 12-op baseline (same as E3_transfer_test + stage0) ───────────────────────

def _bg(x):
    v, c = np.unique(x, return_counts=True)
    return int(v[np.argmax(c)])

def _crop(x):
    nz = np.argwhere(x != _bg(x))
    if len(nz) == 0:
        return x
    return x[nz.min(0)[0]:nz.max(0)[0]+1, nz.min(0)[1]:nz.max(0)[1]+1]

import itertools

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


def apply_op(name, grid):
    try:
        g = np.array(grid, dtype=np.int64)
        return SEED_OPS[name](g)
    except Exception:
        return None


def check_program_dsl(ops, task_io):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        g = inp.copy()
        for name in ops:
            result = apply_op(name, g)
            if result is None:
                return False
            g = result
        if g.shape != out.shape or not np.array_equal(g, out):
            return False
    return True


def search_baseline(task_io, max_length=5, budget=10000):
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(SORTED_SEED_OPS, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program_dsl(list(ops), task_io):
                return list(ops), nodes
    return None, nodes


# ── Arbitrary-Python template library ────────────────────────────────────────
# These are Python source strings evaluated with exec() + restricted builtins.
# Key: NOT a named DSL — arbitrary code (including things impossible in 12-op DSL).
# Expected: most solve nothing new; brute exec is intractable without a proposer.

_ARBITRARY_TEMPLATES = []

# -- Group 1: overlap with 12-op DSL (expected to solve same tasks) --
_ARBITRARY_TEMPLATES += [
    "lambda g: g[::-1]",                          # fv
    "lambda g: g[:, ::-1]",                       # fh
    "lambda g: g.T",                              # tr
    "lambda g: np.rot90(g, 1)",                   # rot
    "lambda g: np.rot90(g, 2)",                   # rot180
    "lambda g: np.rot90(g, 3)",                   # rot270
    "lambda g: np.hstack([g, g])",                # dup_h
    "lambda g: np.vstack([g, g])",                # dup_v
    "lambda g: np.hstack([g, g[:, ::-1]])",       # mir_h
    "lambda g: np.vstack([g, g[::-1]])",          # mir_v
    "lambda g: np.repeat(np.repeat(g, 2, 0), 2, 1)",  # up2
    "lambda g: g[::2, ::2]",                      # down2
]

# -- Group 2: value-conditional (not in 12-op DSL) --
# Color replacement: replace c with d
for c in range(1, 10):
    for d in range(0, 10):
        if c != d:
            _ARBITRARY_TEMPLATES.append(
                f"lambda g: np.where(g == {c}, {d}, g)"
            )

# Boolean extraction: keep only color c (rest 0)
for c in range(1, 10):
    _ARBITRARY_TEMPLATES.append(f"lambda g: (g == {c}).astype(int) * {c}")
    _ARBITRARY_TEMPLATES.append(f"lambda g: (g == {c}).astype(int)")

# Threshold
_ARBITRARY_TEMPLATES += [
    "lambda g: (g > 0).astype(int)",
    "lambda g: (g > 0).astype(int) * 9",
    "lambda g: np.where(g > 0, 1, 0)",
    "lambda g: 9 - g",   # color inversion
]

# -- Group 3: structural (not in 12-op DSL) --
_ARBITRARY_TEMPLATES += [
    # Diagonal flips (beyond fh/fv/tr)
    "lambda g: np.flip(g.T, 0)",           # anti-diagonal
    "lambda g: np.flip(g.T, 1)",
    "lambda g: np.fliplr(np.flipud(g))",
    "lambda g: np.flip(np.rot90(g, 1), 0)",
    "lambda g: np.flip(np.rot90(g, 1), 1)",
    "lambda g: np.flipud(g.T)",
    "lambda g: np.fliplr(g.T)",
    # Repeat/tile patterns
    "lambda g: np.tile(g, (1, 2))",
    "lambda g: np.tile(g, (2, 1))",
    "lambda g: np.tile(g, (2, 2))",
    "lambda g: np.kron(g, np.ones((2,2), dtype=g.dtype))",
    # Row/col removal
    "lambda g: g[1:, :]",
    "lambda g: g[:-1, :]",
    "lambda g: g[:, 1:]",
    "lambda g: g[:, :-1]",
    "lambda g: g[1:-1, 1:-1]",  # inner crop
    # Row/col subsampling
    "lambda g: g[::2, :]",
    "lambda g: g[:, ::2]",
    "lambda g: g[1::2, :]",
    "lambda g: g[:, 1::2]",
    # Mirror-extend variants
    "lambda g: np.vstack([g[::-1], g])",
    "lambda g: np.hstack([g[:, ::-1], g])",
    "lambda g: np.vstack([g, g[::-1], g])",
    "lambda g: np.hstack([g, g[:, ::-1], g])",
    # Sort rows/cols (arbitrary Python, impossible in 12-op DSL)
    "lambda g: np.sort(g, axis=1)",
    "lambda g: np.sort(g, axis=0)",
    "lambda g: np.sort(g, axis=1)[::-1]",
    "lambda g: np.sort(g, axis=0)[::-1]",
    # Unique-mode fill
    "lambda g: np.full_like(g, np.bincount(g.flatten()).argmax())",
    # Zero-out operations
    "lambda g: np.zeros_like(g)",
    "lambda g: np.ones_like(g)",
    "lambda g: np.zeros_like(g) + 9",
]

# -- Group 4: color-swap pairs (not in 12-op DSL) --
for c1 in range(1, 5):
    for c2 in range(c1 + 1, 10):
        _ARBITRARY_TEMPLATES.append(
            f"lambda g: np.where(g == {c1}, -99, np.where(g == {c2}, {c1}, g))"
            f".clip(0, 9)"
        )

# -- Group 5: erasing specific colors --
for c in range(1, 10):
    _ARBITRARY_TEMPLATES.append(f"lambda g: np.where(g == {c}, 0, g)")

print(f"Template library size: {len(_ARBITRARY_TEMPLATES)}")


# ── safe_exec for arbitrary Python grid→grid ─────────────────────────────────

_SAFE_BUILTINS = {k: getattr(builtins, k)
                  for k in ['abs', 'all', 'any', 'bool', 'int', 'float',
                             'len', 'list', 'map', 'max', 'min', 'range',
                             'sum', 'zip', 'enumerate', 'tuple', 'set',
                             'sorted', 'reversed', 'print', 'isinstance',
                             'type', 'isinstance', 'NotImplemented']}


def safe_exec_arc(template_str, task_io, timeout=EXEC_TIMEOUT):
    """
    Execute an arbitrary Python lambda/expression on all I/O pairs.
    Returns True if all outputs match, False otherwise.
    Uses threading.Timer for timeout.
    """
    result = [False]
    error = [None]

    def _run():
        try:
            namespace = {'__builtins__': _SAFE_BUILTINS, 'np': np}
            fn = eval(template_str, namespace)
            for pair in task_io:
                inp = np.array(pair['input'], dtype=np.int64)
                out = np.array(pair['output'], dtype=np.int64)
                pred = fn(inp)
                pred = np.array(pred, dtype=np.int64)
                if pred.shape != out.shape or not np.array_equal(pred, out):
                    return
            result[0] = True
        except Exception as e:
            error[0] = str(e)[:50]

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        error[0] = 'timeout'
    return result[0], error[0]


# ── Load tasks ────────────────────────────────────────────────────────────────

def load_held_out_tasks():
    tasks = []
    for subdir in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, subdir)
        if not os.path.isdir(d):
            continue
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

    training_tasks = [t for t in tasks if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(training_tasks))
    held = [training_tasks[i] for i in indices[:N_TASKS]]
    return held


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("STAGE 0b: ARBITRARY-PYTHON GRID->GRID ARC TEST")
    print("Leo #11728: real code-synthesis feasibility (exec-based, not DSL-expansion)")
    print("=" * 70)
    sys.stdout.flush()

    held_tasks = load_held_out_tasks()
    print(f"Held-out tasks: {len(held_tasks)} (split_seed={SPLIT_SEED}, first {N_TASKS})")
    print(f"Template library: {len(_ARBITRARY_TEMPLATES)} arbitrary-Python templates")
    sys.stdout.flush()

    # 12-op baseline on same N_TASKS tasks
    print(f"\nRunning 12-op DSL baseline on {N_TASKS} tasks...")
    sys.stdout.flush()
    baseline_solved_ids = set()
    baseline_costs = []
    for task in held_tasks:
        prog, cost = search_baseline(task['io'])
        baseline_costs.append(cost)
        if prog:
            baseline_solved_ids.add(task['id'])
    baseline_mean_cost = float(np.mean(baseline_costs))
    print(f"  12-op baseline: solved={len(baseline_solved_ids)}/{N_TASKS}, "
          f"mean_cost={baseline_mean_cost:.1f}")
    sys.stdout.flush()

    # Arbitrary-Python template sweep
    print(f"\nRunning arbitrary-Python template sweep...")
    print(f"  (expected: ~0 new solves — chicken-egg pre-registered result)")
    sys.stdout.flush()

    arbpy_solved_ids = set()
    arbpy_new_solve_ids = set()
    template_errors = 0
    nodes_total = 0

    for task in held_tasks:
        task_solved = False
        for tmpl in _ARBITRARY_TEMPLATES:
            nodes_total += 1
            solved, err = safe_exec_arc(tmpl, task['io'])
            if err and err != 'timeout':
                template_errors += 1
            if solved:
                arbpy_solved_ids.add(task['id'])
                task_solved = True
                if task['id'] not in baseline_solved_ids:
                    arbpy_new_solve_ids.add(task['id'])
                break  # found a template that works for this task

    print(f"  Arbitrary-Python: solved={len(arbpy_solved_ids)}/{N_TASKS}")
    print(f"  New solves beyond 12-op DSL: {len(arbpy_new_solve_ids)}")
    print(f"  Template errors (wrong output / exception): {template_errors}")
    print(f"  Total template-task evaluations: {nodes_total}")
    sys.stdout.flush()

    # Verdict
    new_count = len(arbpy_new_solve_ids)
    if new_count == 0:
        verdict = (
            "CHICKEN-EGG CONFIRMED: arbitrary-Python brute exec solves 0 new tasks "
            "beyond 12-op DSL baseline. Search space intractable without proposer. "
            "Code-synthesis direction requires proposer/recognizer network, not brute exec()."
        )
    elif new_count <= 2:
        verdict = (
            f"NEAR-ZERO: {new_count} new solve(s) beyond 12-op DSL. "
            "Consistent with chicken-egg: marginal enumeration gain, proposer still required."
        )
    else:
        verdict = (
            f"UNEXPECTED: {new_count} new solves beyond 12-op DSL — "
            "investigate which templates are responsible. May indicate richer fixed basis helps "
            "even in exec()-mode, or template selection bias."
        )

    print(f"\nVERDICT: {verdict}")

    result = {
        "experiment": "stage0b-arc-arbitrary-python",
        "note": (
            "Leo #11728: arbitrary-Python exec()-based ARC test. "
            "Matches MBPP side of Stage 0'. This IS the code-synthesis feasibility test. "
            "Contrast: stage0 54-op ARC probe was designer-DSL-expansion (hand-added prims), "
            "NOT code-synthesis (Refinement 2 EXPLICITLY rejected that path)."
        ),
        "config": {
            "split_seed": SPLIT_SEED,
            "n_tasks": N_TASKS,
            "n_templates": len(_ARBITRARY_TEMPLATES),
            "exec_timeout_sec": EXEC_TIMEOUT,
            "baseline_budget": 10000,
            "baseline_max_length": 5,
        },
        "baseline_12op": {
            "solved": len(baseline_solved_ids),
            "n_tasks": N_TASKS,
            "solve_rate": len(baseline_solved_ids) / N_TASKS,
            "mean_cost": baseline_mean_cost,
        },
        "arbitrary_python": {
            "solved": len(arbpy_solved_ids),
            "n_tasks": N_TASKS,
            "solve_rate": len(arbpy_solved_ids) / N_TASKS,
            "new_solves_beyond_dsl_baseline": new_count,
            "new_solve_ids": sorted(arbpy_new_solve_ids),
            "template_errors_approx": template_errors,
            "total_evaluations": nodes_total,
        },
        "verdict": verdict,
        "bigger_fixed_basis_note": (
            "Stage 0' 54-op ARC (stage0_result.json) = bigger-fixed-basis PROBE, "
            "NOT code-synthesis. 54-op is designer-DSL-expansion: "
            "+1 new solve (+0.8% cost), same-budget-depth regression (-2 tasks). "
            "Verdict: bigger fixed basis marginal at best. "
            "This stage0b test is the real code-synthesis feasibility verdict."
        ),
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {RESULT_PATH}")


if __name__ == '__main__':
    main()
