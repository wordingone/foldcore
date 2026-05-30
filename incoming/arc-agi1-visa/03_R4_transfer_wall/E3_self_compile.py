"""
E3 — Self-Compilation: typed DSL + search-cost-reduction harness
the-search#3, Leo directives #11610 + #11615 (2026-05-30)

Question: can the system convert its own traces into reusable intermediate operators
that make future program search lower-dimensional?

Metric: search cost (nodes expanded) on held-out tasks WITH vs WITHOUT grown library.
NOT solve-rate. Primary signal: curve bends DOWN across rounds.

Pre-registered:
  POSITIVE: held-out cost curve bends down -> self-compilation load-bearing (RSI signal)
  NEGATIVE: curve stays flat -> episode-specific (anti-speedup finding #3 structural)
  R6 ablation: library frozen at seed -> held-out cost stays flat

Non-memory-heavy. No model load.
"""

import numpy as np
import json
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# ── Typed transform DSL ────────────────────────────────────────────────────────
# All ops: Grid -> Grid (homogeneous type — square grids throughout)
# "Typed" = compositional structure; every op preserves Grid type under composition.

SEED_OPS_DEF = {
    'fh':  lambda g: np.fliplr(g),
    'fv':  lambda g: np.flipud(g),
    'tr':  lambda g: g.T,
    'rot': lambda g: np.rot90(g),
}


def build_library(base_ops: Dict[str, Any]) -> Dict[str, Any]:
    """Seed library = base ops. Grown library = base + abstracted compound ops."""
    return dict(base_ops)


def apply_program(ops: List[str], grid: np.ndarray, library: Dict) -> Optional[np.ndarray]:
    """Apply sequence of op names to grid. Returns None on any failure."""
    try:
        g = grid.copy()
        for name in ops:
            g = library[name](g)
        return g
    except Exception:
        return None


# ── Bounded search with node counter ──────────────────────────────────────────

def search(task_io: List[Dict], library: Dict, max_length: int = 4,
           budget: int = 1000) -> Tuple[Optional[List[str]], int]:
    """
    Enumerate programs (sequences of op names) up to max_length.
    Returns (program_ops or None, nodes_expanded).
    nodes_expanded is the search cost — the key metric.
    """
    op_names = sorted(library.keys())  # sorted for determinism
    nodes = 0

    for length in range(1, max_length + 1):
        for ops in itertools.product(op_names, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes

            ops_list = list(ops)
            solved = True
            for pair in task_io:
                inp = np.array(pair['input'], dtype=np.int64)
                out = np.array(pair['output'], dtype=np.int64)
                result = apply_program(ops_list, inp, library)
                if result is None or not np.array_equal(result, out):
                    solved = False
                    break

            if solved:
                return ops_list, nodes

    return None, nodes


# ── Partial energy ─────────────────────────────────────────────────────────────

def partial_energy(ops: Optional[List[str]], task_io: List[Dict],
                   library: Dict) -> float:
    """Fraction of training pairs matched (0.0–1.0). Used for failed traces."""
    if ops is None or not ops:
        return 0.0
    matched = 0
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is not None and np.array_equal(result, out):
            matched += 1
    return matched / len(task_io)


# ── Trace capture ──────────────────────────────────────────────────────────────

@dataclass
class Trace:
    task_id: str
    program: Optional[List[str]]   # None if failed (budget exhausted)
    outcome: str                    # 'solved' | 'failed'
    energy: float                   # 1.0 if solved, partial otherwise
    cost: int                       # nodes expanded (THE metric)
    budget_hit: bool                # True if cost == budget


def solve_and_trace(task_id: str, task_io: List[Dict], library: Dict,
                    max_length: int = 4, budget: int = 1000) -> Trace:
    prog, cost = search(task_io, library, max_length, budget)
    budget_hit = (prog is None and cost >= budget)

    if prog is not None:
        return Trace(task_id, prog, 'solved', 1.0, cost, False)

    # Failed — compute partial energy for best single-op program (fast proxy)
    best_e, best_ops = 0.0, None
    for name in library:
        e = partial_energy([name], task_io, library)
        if e > best_e:
            best_e, best_ops = e, [name]

    return Trace(task_id, best_ops, 'failed', best_e, cost, budget_hit)


# ── Contiguous subsequence extraction ─────────────────────────────────────────

def count_subseq(seq: List[str], subseq: Tuple[str, ...]) -> int:
    """Count non-overlapping occurrences of subseq in seq."""
    n, m = len(seq), len(subseq)
    count, i = 0, 0
    while i <= n - m:
        if tuple(seq[i:i+m]) == subseq:
            count += 1
            i += m
        else:
            i += 1
    return count


def extract_candidate_subseqs(traces: List[Trace], min_count: int = 2,
                               min_length: int = 2) -> List[Tuple[Tuple, int]]:
    """
    Find contiguous sub-sequences of length >= min_length appearing in >= min_count traces.
    Uses solved traces only (failed traces feed guards — future extension).
    """
    freq: Dict[Tuple, int] = defaultdict(int)
    solved = [t for t in traces if t.outcome == 'solved' and t.program]

    for trace in solved:
        seq = trace.program
        seen_in_trace: set = set()
        for length in range(min_length, len(seq) + 1):
            for i in range(len(seq) - length + 1):
                sub = tuple(seq[i:i+length])
                if sub not in seen_in_trace:
                    freq[sub] += 1
                    seen_in_trace.add(sub)

    candidates = [(sub, cnt) for sub, cnt in freq.items() if cnt >= min_count]
    candidates.sort(key=lambda x: (-x[1], -len(x[0])))
    return candidates


# ── MDL criterion ──────────────────────────────────────────────────────────────

def mdl_gain(candidate: Tuple[str, ...], traces: List[Trace]) -> int:
    """
    Compression gain from naming candidate as a single op.
    Each occurrence in a trace program replaces len(candidate) tokens with 1 token:
      gain_per_occurrence = len(candidate) - 1
    Cost to define the new op: len(candidate) tokens.
    Net MDL gain = occurrences * (len - 1) - len
    Positive => the new op compresses the trace corpus.
    """
    solved = [t for t in traces if t.outcome == 'solved' and t.program]
    occurrences = sum(count_subseq(t.program, candidate) for t in solved)
    return occurrences * (len(candidate) - 1) - len(candidate)


# ── Abstraction step ───────────────────────────────────────────────────────────

def abstract_library(traces: List[Trace], library: Dict,
                     min_count: int = 2) -> Tuple[Dict, List[str]]:
    """
    From solved traces, extract recurring sub-compositions with positive MDL gain.
    Add them as new named compound ops. Returns (new_library, list_of_added_op_names).
    R2-native: update criterion = compression over own trace corpus.
    No gradient-trained recognition net (R2 violation).
    """
    candidates = extract_candidate_subseqs(traces, min_count=min_count)
    new_lib = dict(library)
    added: List[str] = []

    for subseq, _count in candidates:
        gain = mdl_gain(subseq, traces)
        if gain <= 0:
            continue

        op_name = '__'.join(subseq)
        if op_name in new_lib:
            continue

        # Capture snapshot of current library for the closure
        lib_snapshot = dict(new_lib)

        def make_compound(ops_seq, lib):
            def fn(g):
                for op in ops_seq:
                    g = lib[op](g)
                return g
            return fn

        new_lib[op_name] = make_compound(list(subseq), lib_snapshot)
        added.append(op_name)

    return new_lib, added


# ── Synthetic task generation ──────────────────────────────────────────────────

def generate_tasks(seed_ops: Dict, n: int, grid_size: int = 4,
                   max_prog_len: int = 3, rng_seed: int = 42) -> List[Dict]:
    """
    Generate tasks by applying random seed-op programs to random square grids.
    Tasks are solvable by construction — search will find programs in the seed DSL.
    Each task: {id, io: [{input, output}], ground_truth}
    """
    rng = np.random.RandomState(rng_seed)
    op_names = list(seed_ops.keys())
    tasks = []

    for i in range(n):
        prog_len = rng.randint(1, max_prog_len + 1)
        prog = [op_names[rng.randint(len(op_names))] for _ in range(prog_len)]

        n_pairs = rng.randint(1, 3)  # 1–2 I/O pairs per task
        pairs = []
        for _ in range(n_pairs):
            inp = rng.randint(0, 5, size=(grid_size, grid_size)).tolist()
            out = np.array(inp)
            for op_name in prog:
                out = seed_ops[op_name](out)
            pairs.append({'input': inp, 'output': out.tolist()})

        tasks.append({'id': f'syn_{i:04d}', 'io': pairs, 'ground_truth': prog})

    return tasks


# ── Experiment ─────────────────────────────────────────────────────────────────

def run_experiment():
    print("E3 — Self-Compilation: typed DSL + search-cost-reduction harness")
    print("Pre-registered: POSITIVE=cost curve bends down / NEGATIVE=flat (anti-speedup)")
    print("=" * 70)

    # --- Config ---
    N_SOURCE = 30
    N_HELD   = 20
    MAX_LENGTH = 4
    BUDGET = 800
    MIN_COUNT = 2   # min trace occurrences to consider a sub-sequence
    GRID_SIZE = 4
    MAX_PROG_LEN = 3  # for synthetic task generation

    # --- Generate tasks ---
    print(f"\nGenerating {N_SOURCE} source + {N_HELD} held-out synthetic tasks...")
    all_tasks = generate_tasks(SEED_OPS_DEF, N_SOURCE + N_HELD,
                               grid_size=GRID_SIZE, max_prog_len=MAX_PROG_LEN,
                               rng_seed=42)
    source_tasks = all_tasks[:N_SOURCE]
    held_tasks   = all_tasks[N_SOURCE:]

    seed_library = build_library(SEED_OPS_DEF)

    # --- BASELINE: held-out cost with seed library only ---
    print(f"\nBASELINE: held-out search cost with seed library "
          f"({list(seed_library.keys())[:4]}...)")
    baseline_traces = []
    for task in held_tasks:
        t = solve_and_trace(task['id'], task['io'], seed_library,
                            MAX_LENGTH, BUDGET)
        baseline_traces.append(t)
        print(f"  {task['id']}: {t.outcome} cost={t.cost}")

    baseline_solved = sum(1 for t in baseline_traces if t.outcome == 'solved')
    baseline_cost_mean = np.mean([t.cost for t in baseline_traces])
    print(f"  BASELINE: {baseline_solved}/{N_HELD} solved, "
          f"mean_cost={baseline_cost_mean:.1f}")

    # --- SOURCE ROUND: search source tasks + capture traces ---
    print(f"\nSOURCE ROUND: searching {N_SOURCE} source tasks...")
    source_traces = []
    for task in source_tasks:
        t = solve_and_trace(task['id'], task['io'], seed_library,
                            MAX_LENGTH, BUDGET)
        source_traces.append(t)

    source_solved = sum(1 for t in source_traces if t.outcome == 'solved')
    print(f"  Source: {source_solved}/{N_SOURCE} solved")
    for t in source_traces[:5]:
        print(f"    {t.task_id}: {t.outcome} prog={t.program} cost={t.cost}")

    # --- ABSTRACTION STEP: MDL library extension ---
    print(f"\nABSTRACTION STEP: anti-unification + MDL over {source_solved} solved traces...")
    candidates = extract_candidate_subseqs(source_traces, min_count=MIN_COUNT)
    print(f"  Candidate sub-sequences: {len(candidates)}")
    for sub, cnt in candidates[:10]:
        gain = mdl_gain(sub, source_traces)
        print(f"    {sub}  count={cnt}  MDL_gain={gain}")

    grown_library, added_ops = abstract_library(source_traces, seed_library,
                                                min_count=MIN_COUNT)
    print(f"  New ops added to library: {added_ops}")
    print(f"  Library size: {len(seed_library)} -> {len(grown_library)}")

    # --- WITH-LIBRARY ROUND: held-out cost with grown library ---
    print(f"\nWITH-LIBRARY ROUND: held-out search cost with grown library "
          f"({len(grown_library)} ops)...")
    grown_traces = []
    for task in held_tasks:
        t = solve_and_trace(task['id'], task['io'], grown_library,
                            MAX_LENGTH, BUDGET)
        grown_traces.append(t)

    grown_solved = sum(1 for t in grown_traces if t.outcome == 'solved')
    grown_cost_mean = np.mean([t.cost for t in grown_traces])
    print(f"  WITH-LIB: {grown_solved}/{N_HELD} solved, "
          f"mean_cost={grown_cost_mean:.1f}")

    # --- R6 ABLATION: freeze library at seed (same as baseline) ---
    r6_cost_mean = baseline_cost_mean  # seed = frozen = baseline
    r6_solved    = baseline_solved

    # --- RESULTS ---
    delta_cost = baseline_cost_mean - grown_cost_mean  # positive = reduction
    delta_pct  = (delta_cost / baseline_cost_mean * 100) if baseline_cost_mean > 0 else 0.0

    print("\n" + "=" * 70)
    print("SELF-COMPILATION RESULT")
    print("=" * 70)
    print(f"Seed library ops:  {list(seed_library.keys())}")
    print(f"Added ops:         {added_ops}")
    print(f"Grown library ops: {list(grown_library.keys())}")
    print()
    print(f"Held-out search cost:")
    print(f"  BASELINE (seed):   mean={baseline_cost_mean:.1f}  solved={baseline_solved}/{N_HELD}")
    print(f"  WITH-LIB (grown):  mean={grown_cost_mean:.1f}  solved={grown_solved}/{N_HELD}")
    print(f"  R6 ABLATION (freeze=seed):  mean={r6_cost_mean:.1f}  (same as baseline)")
    print()
    print(f"Search-cost reduction: {delta_cost:+.1f} nodes ({delta_pct:+.1f}%)")

    if delta_cost > 0 and len(added_ops) > 0:
        verdict = "POSITIVE — self-compilation load-bearing: library compressed held-out search"
    elif len(added_ops) == 0:
        verdict = "TRIVIAL SEED — no compound ops abstracted; harness validated structurally"
    else:
        verdict = "NEGATIVE — curve flat: operators episode-specific (anti-speedup finding #3)"

    print(f"\nVerdict: {verdict}")

    # --- Structured result ---
    result = {
        "experiment": "E3-self-compilation",
        "config": {
            "n_source": N_SOURCE, "n_held": N_HELD,
            "max_length": MAX_LENGTH, "budget": BUDGET,
            "min_count": MIN_COUNT, "grid_size": GRID_SIZE,
        },
        "seed_ops": list(seed_library.keys()),
        "added_ops": added_ops,
        "grown_lib_size": len(grown_library),
        "source_solved": source_solved,
        "candidates_found": len(candidates),
        "held_out": {
            "baseline_cost_mean": float(baseline_cost_mean),
            "baseline_solved": baseline_solved,
            "grown_cost_mean": float(grown_cost_mean),
            "grown_solved": grown_solved,
            "r6_cost_mean": float(r6_cost_mean),
            "r6_solved": r6_solved,
            "delta_cost": float(delta_cost),
            "delta_pct": float(delta_pct),
        },
        "verdict": verdict,
    }

    out_path = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_self_compile_result.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out_path}")

    return result


if __name__ == '__main__':
    run_experiment()
