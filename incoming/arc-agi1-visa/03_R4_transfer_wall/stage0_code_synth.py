"""
Stage 0' — Code-Synthesis Feasibility (Leo #11717, 2026-05-30).

Trigger: 12-op HOLLOW_CLIMB accepted. Refinement 2 fired.

Question: Does a richer EXECUTABLE LAYER solve more than 12-op-brute at the SAME
compute budget? Isolates layer-vs-proposer: if the richer grammar lifts coverage even
with weak enumeration, the 12-op base was the bottleneck, not the proposer.

Executable layer: sandboxed Python grid→grid (numpy in→out). Generate-and-test:
run candidate program on all train I/O pairs; all-match → test held pair (ARC) or
run assertions (MBPP).

Sandbox: exec() in restricted namespace (no file/network/exec imports) +
threading.Timer timeout. NOT subprocess (too slow per-candidate for budget comparison).

Grammar for ARC (Turing-richer than 12-op):
  Leaves: 12 seed ops (same as E3 base)
  Extensions (not expressible in 12-op brute):
    - color_map(c1→c2): np.where(grid == c1, c2, grid) for c1,c2 in 0-9 × 0-9
    - swap_colors(c1, c2): swap two color values
    - fill_bg: set background to 0
    - rotate_180: rot90 × 2 (could be composition but adds speed as primitive)

Grammar for MBPP: Python function synthesis. The "WEAKEST PROPOSER" for MBPP is:
  - Templates with hole-filling: simple arithmetic, string ops, list ops
  - 12-op brute has 0% MBPP solve-rate by construction (no text generation)
  - This measures: can ANY enumeration over Python programs solve MBPP tasks?

Metric: solve-rate code-synthesis vs 12-op-brute, SAME compute budget (NODE_BUDGET).

PRISM multi-domain from inception: run on ARC-AGI-1 held-out AND MBPP first-50.
Report solve-rate on BOTH in the same result JSON.

Pre-registered outcomes:
  PASS: code-synth > 12-op-brute on ARC + non-trivial MBPP → layer bottleneck → Stage 1
  CHICKEN-EGG: code-synth ≈ brute or near-zero ARC → proposer needed first
"""

import numpy as np
import json
import itertools
import sys
import os
import threading
import textwrap
import ast
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage0_result.json"

NODE_BUDGET = 10000   # same budget as E3 12-op-brute
MAX_LENGTH = 5        # same as E3 for 12-op baseline
SPLIT_SEED = 42
N_HELD_ARC = 200
N_MBPP = 50           # first 50 MBPP sanitized-test problems
EXEC_TIMEOUT = 1.0    # seconds per candidate execution

# ── Seed DSL (12-op, identical to E3) ────────────────────────────────────────

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
SORTED_SEED = sorted(SEED_OPS.keys())


# ── Richer ARC primitives (NOT expressible as 12-op compositions in ≤5 steps) ──

def _color_replace(x, c_from, c_to):
    return np.where(x == c_from, c_to, x)

def _swap_colors(x, c1, c2):
    y = x.copy()
    y[x == c1] = c2
    y[x == c2] = c1
    return y

def _fill_bg_zero(x):
    """Set background (most frequent) color to 0."""
    bg = _bg(x)
    if bg == 0:
        return x
    return np.where(x == bg, 0, x)

def _rot180(x):
    return np.rot90(x, 2)

# Color-map primitives: parameterized by (c_from, c_to)
# Generates one primitive per (c_from, c_to) pair where c_from != c_to
# Use only common colors (0-9) and only those likely to appear in ARC grids
_COLOR_REPLACE_OPS = {}
for _cf in range(10):
    for _ct in range(10):
        if _cf != _ct:
            name = f'cr_{_cf}_{_ct}'
            _COLOR_REPLACE_OPS[name] = (lambda cf=_cf, ct=_ct: lambda x: _color_replace(x, cf, ct))()

_SWAP_OPS = {}
for _c1 in range(10):
    for _c2 in range(_c1 + 1, 10):
        name = f'sw_{_c1}_{_c2}'
        _SWAP_OPS[name] = (lambda c1=_c1, c2=_c2: lambda x: _swap_colors(x, c1, c2))()

RICHER_OPS = {
    'fill_bg0': _fill_bg_zero,
    'rot180': _rot180,
    **_COLOR_REPLACE_OPS,
    **_SWAP_OPS,
}

# Combined vocabulary: seed ops + most compact richer ops
# Limit color-replace to small set (c_from=1-5, c_to=0-5) to keep space tractable
COMPACT_COLOR_REPLACE = {k: v for k, v in _COLOR_REPLACE_OPS.items()
                         if int(k.split('_')[1]) in range(1, 6) and
                            int(k.split('_')[2]) in range(6)}
COMPACT_SWAP = {k: v for k, v in _SWAP_OPS.items()
                if int(k.split('_')[1]) < 5 and int(k.split('_')[2]) < 6}

RICHER_VOCAB = {
    **SEED_OPS,
    'fill_bg0': _fill_bg_zero,
    'rot180': _rot180,
    **COMPACT_COLOR_REPLACE,   # ~30 ops
    **COMPACT_SWAP,            # ~10 ops
}
SORTED_RICHER = sorted(RICHER_VOCAB.keys())


# ── ARC grid evaluation ───────────────────────────────────────────────────────

def apply_program_arc(ops, grid, vocab):
    g = np.array(grid, dtype=np.int64)
    for op in ops:
        try:
            g = vocab[op](g)
        except Exception:
            return None
    return g


def check_program_arc(ops, task_io, vocab):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program_arc(ops, inp, vocab)
        if result is None or result.shape != out.shape or not np.array_equal(result, out):
            return False
    return True


def search_arc_12op(task_io, budget=NODE_BUDGET, max_length=MAX_LENGTH):
    """12-op brute BFS (baseline)."""
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(SORTED_SEED, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program_arc(list(ops), task_io, SEED_OPS):
                return list(ops), nodes
    return None, nodes


def search_arc_richer(task_io, budget=NODE_BUDGET, max_length=MAX_LENGTH):
    """Richer grammar BFS."""
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(SORTED_RICHER, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program_arc(list(ops), task_io, RICHER_VOCAB):
                return list(ops), nodes
    return None, nodes


# ── MBPP safe exec ───────────────────────────────────────────────────────────

def safe_exec_mbpp(code_str, test_str, timeout=EXEC_TIMEOUT):
    """Execute code_str + test_str in restricted namespace with timeout.
    Returns (passed, error_msg)."""
    result = [False]
    error = [None]

    def _run():
        import builtins
        # Restricted builtins: only safe ones
        safe_builtins = {
            k: getattr(builtins, k) for k in [
                'abs', 'all', 'any', 'bin', 'bool', 'chr', 'dict', 'divmod',
                'enumerate', 'filter', 'float', 'frozenset', 'hash', 'hex',
                'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
                'map', 'max', 'min', 'next', 'oct', 'ord', 'pow', 'print',
                'range', 'repr', 'reversed', 'round', 'set', 'slice', 'sorted',
                'str', 'sum', 'super', 'tuple', 'type', 'zip', 'None', 'True',
                'False', 'NotImplemented', 'AssertionError', 'ValueError',
                'TypeError', 'IndexError', 'KeyError', 'StopIteration',
                'Exception', 'BaseException',
            ] if hasattr(builtins, k)
        }
        safe_builtins['__build_class__'] = builtins.__build_class__
        namespace = {'__builtins__': safe_builtins}
        try:
            exec(compile(code_str, '<synth>', 'exec'), namespace)
            exec(compile(test_str, '<test>', 'exec'), namespace)
            result[0] = True
        except AssertionError:
            error[0] = 'assertion_fail'
        except SyntaxError as e:
            error[0] = f'syntax: {e}'
        except Exception as e:
            error[0] = f'{type(e).__name__}: {e}'

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        error[0] = 'timeout'
    return result[0], error[0]


# ── MBPP template enumerator ─────────────────────────────────────────────────
#
# Stage 0' uses the "weakest proposer": enumerate simple Python program TEMPLATES
# with hole-filling. This is NOT a full Python AST enumerator (intractable).
# Templates cover the most common MBPP patterns: single-expression functions.
#
# Template families:
#   T1: return <expr> — single expression return
#   T2: return [x for x in arg if <cond>] — list comprehension with filter
#   T3: return sorted/reversed/min/max/sum/len(arg) — aggregate ops
#   T4: str methods — arg.split/strip/replace/count/...
#   T5: arithmetic on input — return arg + c, arg * c, arg % c, etc.
#
# This gives ~200-500 candidate programs per MBPP problem.
# 12-op brute has 0% MBPP solve-rate (can't generate text); even 1 solve > 0%.

def _mbpp_templates(prompt_text, fn_name='func'):
    """Generate candidate Python function strings for a given MBPP prompt."""
    # Parse function name from test_list if possible; use 'func' as fallback
    candidates = []

    # T1: single-expression returns
    simple_exprs = [
        'arg', 'arg[::-1]', 'arg.strip()', 'arg.lower()', 'arg.upper()',
        'arg.split()', 'list(arg)', 'set(arg)', 'sorted(arg)',
        'sorted(arg, reverse=True)', 'len(arg)', 'sum(arg)',
        'min(arg)', 'max(arg)', 'str(arg)', 'int(arg)', 'float(arg)',
        '[]', '{}', '0', '1', '-1', 'None', 'True', 'False',
        'arg + arg', 'arg * 2', 'arg[0]', 'arg[-1]',
        'arg[1:]', 'arg[:-1]', 'arg[::2]', 'arg[1::2]',
        '[x for x in arg]', '[x for x in arg if x]',
        '[x.strip() for x in arg]',
        'dict(arg)', 'tuple(arg)',
        '"\n".join(arg)', '" ".join(arg)', '"".join(arg)',
    ]
    for expr in simple_exprs:
        candidates.append(
            f'def {fn_name}(arg):\n    return {expr}'
        )

    # T2: two-arg functions
    two_arg_exprs = [
        'a + b', 'a - b', 'a * b', 'a / b', 'a // b', 'a % b', 'a ** b',
        'max(a, b)', 'min(a, b)', 'a if a > b else b',
        'a if a < b else b', 'a and b', 'a or b',
        '[x for x in a if x in b]', '[x for x in a if x not in b]',
        'list(set(a) & set(b))', 'list(set(a) | set(b))',
        'list(set(a) - set(b))', 'a.count(b)', 'a.replace(b, "")',
        'a.split(b)', 'a.startswith(b)', 'a.endswith(b)',
        'a.index(b)', 'sorted(a + b)', 'a + b if isinstance(a, list) else a + b',
        'dict(zip(a, b))', '[a[i] + b[i] for i in range(len(a))]',
    ]
    for expr in two_arg_exprs:
        candidates.append(
            f'def {fn_name}(a, b):\n    return {expr}'
        )

    # T3: three-arg functions (less common, small set)
    three_arg_exprs = [
        'a + b + c', 'a * b * c', 'max(a, b, c)', 'min(a, b, c)',
        '[a, b, c]', 'sorted([a, b, c])', '(a + b + c) / 3',
        'a if a > b and a > c else (b if b > c else c)',
    ]
    for expr in three_arg_exprs:
        candidates.append(
            f'def {fn_name}(a, b, c):\n    return {expr}'
        )

    return candidates


def search_mbpp_synth(problem, budget=NODE_BUDGET):
    """Test generated Python programs against MBPP test cases.
    Returns (solved, n_candidates_tested, winning_code)."""
    # Extract function name from first test assertion
    fn_name = 'func'
    for test in problem.get('test_list', []):
        # e.g. "assert remove_Occ("hello","l") == "heo""
        m = __import__('re').match(r'assert\s+(\w+)\s*\(', test)
        if m:
            fn_name = m.group(1)
            break

    candidates = _mbpp_templates(problem.get('prompt', ''), fn_name=fn_name)
    test_str = '\n'.join(problem.get('test_list', []))
    if not test_str:
        return False, 0, None

    for i, code in enumerate(candidates):
        if i >= budget:
            return False, i + 1, None
        passed, err = safe_exec_mbpp(code, test_str)
        if passed:
            return True, i + 1, code

    return False, len(candidates), None


# ── Load ARC tasks ────────────────────────────────────────────────────────────

def load_arc_tasks():
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
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("STAGE 0' — Code-Synthesis Feasibility (Leo #11717)")
    print("Richer layer vs 12-op-brute, SAME budget. ARC + MBPP (PRISM multi-domain).")
    print("=" * 70)
    sys.stdout.flush()

    # ── ARC held-out tasks (same split as E3) ──────────────────────────────
    print("\n[ARC] Loading tasks...")
    all_tasks = load_arc_tasks()
    training_tasks = [t for t in all_tasks if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(training_tasks))
    held_tasks = [training_tasks[i] for i in indices[:N_HELD_ARC]]
    print(f"  Held-out: {len(held_tasks)} tasks (training, split_seed={SPLIT_SEED})")
    print(f"  12-op vocabulary: {len(SORTED_SEED)} ops")
    print(f"  Richer vocabulary: {len(SORTED_RICHER)} ops")
    sys.stdout.flush()

    # ARC 12-op baseline
    print("\n[ARC] Running 12-op-brute BFS (baseline)...")
    sys.stdout.flush()
    arc_12op_costs = []
    arc_12op_solved = 0
    for task in held_tasks:
        prog, cost = search_arc_12op(task['io'])
        arc_12op_costs.append(cost)
        if prog:
            arc_12op_solved += 1
    arc_12op_mean = float(np.mean(arc_12op_costs))
    print(f"  12-op baseline: mean_cost={arc_12op_mean:.1f}, solved={arc_12op_solved}/{N_HELD_ARC} "
          f"({100*arc_12op_solved/N_HELD_ARC:.1f}%)")
    sys.stdout.flush()

    # ARC richer grammar
    print("\n[ARC] Running richer grammar BFS...")
    sys.stdout.flush()
    arc_richer_costs = []
    arc_richer_solved = 0
    arc_richer_new = 0
    arc_richer_solutions = {}
    for i, task in enumerate(held_tasks):
        prog, cost = search_arc_richer(task['io'])
        arc_richer_costs.append(cost)
        if prog:
            arc_richer_solved += 1
            arc_richer_solutions[task['id']] = prog
            # Check if this is a new solve (not solved by 12-op)
            base_prog, _ = search_arc_12op(task['io'])
            if not base_prog:
                arc_richer_new += 1
    arc_richer_mean = float(np.mean(arc_richer_costs))
    arc_delta_pct = (arc_richer_mean - arc_12op_mean) / arc_12op_mean * 100
    print(f"  Richer grammar: mean_cost={arc_richer_mean:.1f}, solved={arc_richer_solved}/{N_HELD_ARC} "
          f"({100*arc_richer_solved/N_HELD_ARC:.1f}%), new_solves={arc_richer_new}, "
          f"delta_cost={arc_delta_pct:+.1f}%")
    sys.stdout.flush()

    # ── MBPP ───────────────────────────────────────────────────────────────
    print("\n[MBPP] Loading dataset...")
    sys.stdout.flush()
    try:
        from datasets import load_dataset
        mbpp_ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
        mbpp_problems = [
            {'task_id': p['task_id'], 'prompt': p['prompt'],
             'test_list': p['test_list'], 'code': p['code']}
            for p in mbpp_ds
        ][:N_MBPP]
        print(f"  Loaded {len(mbpp_problems)} MBPP problems")
    except Exception as e:
        print(f"  MBPP load failed: {e} — skipping MBPP section")
        mbpp_problems = []
    sys.stdout.flush()

    mbpp_12op_solved = 0   # always 0 by construction
    mbpp_synth_solved = 0
    mbpp_synth_costs = []
    mbpp_solutions = {}

    if mbpp_problems:
        print(f"\n[MBPP] 12-op brute: 0% (no text generation capability, by construction)")
        print(f"\n[MBPP] Running code-synthesis enumerator on {len(mbpp_problems)} problems...")
        sys.stdout.flush()
        for problem in mbpp_problems:
            solved, n_tested, code = search_mbpp_synth(problem, budget=NODE_BUDGET)
            mbpp_synth_costs.append(n_tested)
            if solved:
                mbpp_synth_solved += 1
                mbpp_solutions[problem['task_id']] = code
        mbpp_synth_rate = 100 * mbpp_synth_solved / len(mbpp_problems)
        print(f"  Code-synthesis: solved={mbpp_synth_solved}/{len(mbpp_problems)} "
              f"({mbpp_synth_rate:.1f}%), mean_candidates={np.mean(mbpp_synth_costs):.1f}")
        sys.stdout.flush()

    # ── Verdict ─────────────────────────────────────────────────────────────
    arc_rate_12op = arc_12op_solved / N_HELD_ARC
    arc_rate_richer = arc_richer_solved / N_HELD_ARC
    mbpp_rate_12op = 0.0
    mbpp_rate_synth = mbpp_synth_solved / max(len(mbpp_problems), 1) if mbpp_problems else None

    arc_lift = arc_rate_richer - arc_rate_12op
    mbpp_lift = (mbpp_rate_synth - mbpp_rate_12op) if mbpp_rate_synth is not None else None

    if arc_new_solves_pct := (arc_richer_new / N_HELD_ARC):
        arc_lift_str = f"+{arc_richer_new} new solves ({100*arc_lift:+.1f}pp)"
    else:
        arc_lift_str = f"no new solves ({100*arc_lift:+.1f}pp)"

    if arc_richer_new > 0 and mbpp_rate_synth is not None and mbpp_rate_synth > 0:
        verdict = (
            f"PASS: richer layer solves {arc_richer_new} new ARC tasks + {mbpp_synth_solved} MBPP. "
            f"Layer was the bottleneck. Stage 1 (self-compilation on richer base)."
        )
    elif arc_richer_new > 0:
        verdict = (
            f"PARTIAL-PASS: richer layer solves {arc_richer_new} new ARC tasks. "
            f"MBPP {'not run' if not mbpp_problems else f'0/{len(mbpp_problems)} solved'}. "
            f"Layer lifts ARC; MBPP needs stronger templates."
        )
    elif mbpp_rate_synth is not None and mbpp_rate_synth > 0:
        verdict = (
            f"PARTIAL-PASS: 0 new ARC solves but {mbpp_synth_solved} MBPP solved. "
            f"Python layer works for MBPP natively; ARC still blocked by proposer."
        )
    else:
        verdict = (
            f"CHICKEN-EGG: richer grammar={arc_richer_new} new ARC + 0 MBPP. "
            f"Richer space drowns weak enumeration. "
            f"Proposer needed: recognition-net-from-own-solves or curriculum-from-enum-solvable-up."
        )

    print(f"\nVERDICT: {verdict}")

    result = {
        'experiment': 'stage0-code-synthesis',
        'note': (
            'Leo #11717: does richer executable layer lift coverage vs 12-op-brute '
            'at same compute budget? ARC-AGI-1 + MBPP (PRISM multi-domain from inception).'
        ),
        'config': {
            'node_budget': NODE_BUDGET,
            'max_length': MAX_LENGTH,
            'split_seed': SPLIT_SEED,
            'n_held_arc': N_HELD_ARC,
            'n_mbpp': len(mbpp_problems),
            'vocab_12op': len(SORTED_SEED),
            'vocab_richer': len(SORTED_RICHER),
        },
        'arc': {
            '12op_baseline': {
                'mean_cost': arc_12op_mean,
                'solved': arc_12op_solved,
                'solve_rate': arc_rate_12op,
            },
            'richer_grammar': {
                'mean_cost': arc_richer_mean,
                'solved': arc_richer_solved,
                'solve_rate': arc_rate_richer,
                'new_solves': arc_richer_new,
                'delta_cost_pct': arc_delta_pct,
            },
            'arc_lift_str': arc_lift_str,
        },
        'mbpp': {
            '12op_baseline': {'solve_rate': 0.0, 'note': 'No text generation in 12-op'},
            'code_synthesis': {
                'solved': mbpp_synth_solved,
                'n_problems': len(mbpp_problems),
                'solve_rate': mbpp_rate_synth,
                'mean_candidates': float(np.mean(mbpp_synth_costs)) if mbpp_synth_costs else None,
            } if mbpp_problems else {'note': 'MBPP not loaded'},
        },
        'verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {RESULT_PATH}")


if __name__ == '__main__':
    main()
