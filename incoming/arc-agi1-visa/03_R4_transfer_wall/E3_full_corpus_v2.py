"""
E3 Full-Corpus Abstraction Engine v2 — Executable Holed Operators (Leo #11651).

v1 produced informational-only LCS patterns — flat result was a FALSE NEGATIVE
because holed operators were never made executable. v2 fixes the core:

EXECUTABLE HOLED OPERATORS:
  From programs [A, X, B] and [A, Y, B], anti-unification yields [A, ?, B]
  with a typed single-op hole. Executable means:
    (a) library stores (prefix, suffix) as a HoledOp entry;
    (b) during BFS, selecting HoledOp commits prefix and suffix, then
        fills the hole with a bounded inner search (12 seed ops = 12 nodes);
    (c) dimension reduction is measured on the outer BFS — solution in 1 outer
        node + 12 inner nodes, vs N outer nodes for standard BFS.

VERDICT FRAMING (pre-registration):
  VALID KILL: richer operators (holed + contiguous) executable + in library,
    held-out reduction still flat, R6 ablation flat.
  FORWARD: holed operators yield held-out reduction, R6 load-bearing.
  UNINFORMATIVE (NOT A KILL): holed ops logged but not executable (v1 failure).

R6 ABLATION: search with only contiguous library (no holes) vs full library
  (contiguous + holes). Isolates holed-op contribution.
"""

import numpy as np
import json
import itertools
import sys
import os
from collections import Counter, defaultdict

# ── Load data ─────────────────────────────────────────────────────────────────

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_full_corpus_v2_result.json"

BUDGET = 10000
MAX_LENGTH = 5
SPLIT_SEED = 42
N_SOURCE = 200
N_HELD = 200
N_PAIRS_MIN = 1

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


# ── Grid application ──────────────────────────────────────────────────────────

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


def count_pairs_matched(ops, task_io, library=None):
    count = 0
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = apply_program(ops, inp, library)
        if result is not None and result.shape == out.shape and np.array_equal(result, out):
            count += 1
    return count


# ── Executable holed operator ─────────────────────────────────────────────────

class HoledOp:
    """
    A single-op-hole operator: (prefix) ◦ [?] ◦ (suffix).
    Executable: applies prefix, then tries each seed op for the hole,
    then applies suffix. Returns filled program on match.
    """
    def __init__(self, prefix, suffix):
        self.prefix = list(prefix)
        self.suffix = list(suffix)
        p_str = '__'.join(prefix) if prefix else 'eps'
        s_str = '__'.join(suffix) if suffix else 'eps'
        self.name = f"{p_str}___HOLE___{s_str}"

    def try_fill(self, task_io, library=None):
        """
        Try each seed op for the hole. Returns (filled_ops, found_op, inner_nodes).
        inner_nodes = number of seed ops tried (max 12).
        """
        lib = library or {}
        n_pairs = len(task_io)

        # Precompute: apply prefix to each input
        prefix_states = []
        for pair in task_io:
            inp = np.array(pair['input'], dtype=np.int64)
            mid = apply_program(self.prefix, inp, lib)
            if mid is None:
                # Prefix fails on this task
                return None, None, len(SORTED_SEED_OPS)
            prefix_states.append((mid, np.array(pair['output'], dtype=np.int64)))

        # Try each seed op for the hole
        for i, hole_op in enumerate(SORTED_SEED_OPS):
            all_match = True
            for (mid, out) in prefix_states:
                mid2 = apply_op(hole_op, mid, lib)
                if mid2 is None:
                    all_match = False
                    break
                result = apply_program(self.suffix, mid2, lib)
                if (result is None or result.shape != out.shape or
                        not np.array_equal(result, out)):
                    all_match = False
                    break
            if all_match:
                filled = self.prefix + [hole_op] + self.suffix
                return filled, hole_op, i + 1

        return None, None, len(SORTED_SEED_OPS)

    @property
    def total_length(self):
        return len(self.prefix) + 1 + len(self.suffix)

    def mdl_gain(self, count):
        """
        MDL gain: each occurrence saves (total_length - 2) ops by replacing with
        1 outer op + 1 inner op. Cost of storing holed op = total_length.
        Gain = count * (total_length - 2) - total_length.
        """
        return count * (self.total_length - 2) - self.total_length

    def __repr__(self):
        return f"HoledOp({self.prefix} ◦ [?] ◦ {self.suffix})"


# ── BFS with partial tracking (unchanged from v1) ─────────────────────────────

def search_with_partials(task_io, library=None, max_length=MAX_LENGTH, budget=BUDGET):
    """
    BFS with partial-match tracking. Returns (solution, nodes, best_partial, best_partial_score).
    """
    lib = library or {}
    lib_ops = sorted(lib.keys())
    seen = set()
    ordered_ops = []
    for n in lib_ops + SORTED_SEED_OPS:
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    n_pairs = len(task_io)
    nodes = 0
    best_partial = None
    best_partial_score = 0.0

    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes, best_partial, best_partial_score
            matched = count_pairs_matched(list(ops), task_io, lib)
            score = matched / n_pairs
            if score > best_partial_score:
                best_partial_score = score
                best_partial = list(ops)
            if matched == n_pairs:
                return list(ops), nodes, list(ops), 1.0
    return None, nodes, best_partial, best_partial_score


# ── BFS with executable holed operators ───────────────────────────────────────

def search_with_holed_ops(task_io, contiguous_lib=None, holed_ops=None,
                           max_length=MAX_LENGTH, budget=BUDGET):
    """
    BFS that treats holed operators as single-node outer ops.
    Phase 1: Try all holed operators (each = 1 outer node + up to 12 inner nodes).
    Phase 2: Standard BFS with contiguous library only.

    The outer BFS position of holed ops: they're treated as length-1 ops in the
    outer BFS (equivalent to a compound op). This is the correct placement —
    they compete fairly with single seed ops at length 1.

    Returns (program, nodes_expanded).
    """
    lib = contiguous_lib or {}
    holed = holed_ops or []
    n_pairs = len(task_io)

    lib_ops = sorted(lib.keys())
    seen = set()
    ordered_ops = []
    for n in lib_ops + SORTED_SEED_OPS:
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    nodes = 0

    # PHASE 1: Try each holed operator at outer-length-1
    for holed_op in holed:
        nodes += 1
        if nodes > budget:
            return None, nodes, None
        filled, _, inner_nodes = holed_op.try_fill(task_io, lib)
        nodes += inner_nodes
        if nodes > budget:
            return filled, nodes, holed_op.name if filled else None
        if filled is not None:
            return filled, nodes, holed_op.name  # SELECTED: holed_op.name

    # PHASE 2: Standard BFS with contiguous library
    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes, None
            if check_program(list(ops), task_io, lib):
                return list(ops), nodes, None
    return None, nodes, None


# ── Holed operator extraction ─────────────────────────────────────────────────

def extract_holed_ops(programs):
    """
    Single-op-hole extraction via prefix-suffix alignment.
    For programs p1 = [prefix..., X, suffix...] and p2 = [prefix..., Y, suffix...]
    with X != Y and len(p1) == len(p2): extract HoledOp(prefix, suffix).

    MDL filter: holed op must have gain > 0 on the corpus.
    """
    # Count: for each (prefix, suffix) how many programs have prefix + ANY_OP + suffix
    pattern_programs = defaultdict(list)  # (prefix_t, suffix_t) -> [programs]

    for prog in programs:
        L = len(prog)
        if L < 2:
            continue
        for hole_pos in range(L):
            prefix = tuple(prog[:hole_pos])
            suffix = tuple(prog[hole_pos + 1:])
            pattern_programs[(prefix, suffix)].append(prog)

    # FORMATION FUNNEL (Leo #11669: discriminates starvation vs density vs bug)
    total_pairs_checked = len(pattern_programs)
    candidates_2plus = []   # (prefix, suffix) with >=2 distinct hole variants
    candidates_mdl_pos = [] # those that pass MDL
    gain_distribution = []  # gains for ALL 2plus candidates (including negatives)

    seen_patterns = set()
    holed_ops = []

    for (prefix, suffix), progs in pattern_programs.items():
        if (prefix, suffix) in seen_patterns:
            continue
        seen_patterns.add((prefix, suffix))

        # Get distinct hole ops
        hole_ops = set()
        for prog in progs:
            hole_pos = len(prefix)
            if len(prog) == len(prefix) + 1 + len(suffix):
                hole_ops.add(prog[hole_pos])

        if len(hole_ops) < 2:
            # Single variant — not anti-unification
            continue

        candidates_2plus.append((prefix, suffix))
        count = len(progs)
        hop = HoledOp(prefix, suffix)
        gain = hop.mdl_gain(count)
        gain_distribution.append(gain)

        if gain <= 0:
            continue

        candidates_mdl_pos.append((prefix, suffix))
        holed_ops.append({
            'op': hop,
            'count': count,
            'gain': gain,
            'hole_variants': sorted(hole_ops),
            'mdl_positive': True,
        })

    holed_ops.sort(key=lambda x: -x['gain'])

    funnel = {
        'total_pairs_checked': total_pairs_checked,
        'candidates_2plus_variants': len(candidates_2plus),
        'candidates_passed_MDL': len(candidates_mdl_pos),
        'gain_distribution': {
            'n': len(gain_distribution),
            'min': float(min(gain_distribution)) if gain_distribution else None,
            'max': float(max(gain_distribution)) if gain_distribution else None,
            'mean': float(np.mean(gain_distribution)) if gain_distribution else None,
            'n_positive': sum(1 for g in gain_distribution if g > 0),
        },
        'diagnosis': (
            'NO_SHARED_SKELETONS' if total_pairs_checked == 0 else
            'STRUCTURAL_STARVATION' if len(candidates_2plus) == 0 else
            'DENSITY_THRESHOLD' if len(candidates_mdl_pos) == 0 else
            'PASSED'
        ),
    }
    return holed_ops, funnel


# ── Contiguous MDL (unchanged from v1) ───────────────────────────────────────

def count_subseq(program, candidate):
    count = 0
    for i in range(len(program) - len(candidate) + 1):
        if tuple(program[i:i + len(candidate)]) == tuple(candidate):
            count += 1
    return count


def run_contiguous_mdl(programs, label=""):
    subseq_counts = Counter()
    for prog in programs:
        if len(prog) >= 2:
            for length in range(2, len(prog) + 1):
                for start in range(len(prog) - length + 1):
                    sub = tuple(prog[start:start + length])
                    subseq_counts[sub] += 1

    added_ops = {}
    candidates = []
    for sub, cnt in sorted(subseq_counts.items(), key=lambda x: (-x[1], len(x[0]))):
        gain = cnt * (len(sub) - 1) - len(sub)
        candidates.append({'sub': list(sub), 'count': cnt, 'gain': gain})
        if gain > 0:
            name = '__'.join(sub)
            added_ops[name] = list(sub)

    print(f"  [{label}] Contiguous MDL over {len(programs)} programs:")
    print(f"    Distinct candidates: {len(candidates)}")
    mdl_pos = [c for c in candidates if c['gain'] > 0]
    print(f"    MDL-positive: {len(mdl_pos)}")
    for c in mdl_pos[:5]:
        print(f"      {'__'.join(c['sub'])}: count={c['count']}, gain={c['gain']}")
    if len(mdl_pos) > 5:
        print(f"      ... ({len(mdl_pos) - 5} more)")

    return added_ops, candidates


# ── Load tasks ────────────────────────────────────────────────────────────────

def load_tasks(training_only=True):
    """
    Load ARC tasks. training_only=True (default) loads only the training/ subdir (400 tasks).
    Source and held-out are guaranteed DISJOINT (random split from the same pool).
    v1 was training-only (400 tasks). training_only=True ensures comparability with v1
    and prevents abstract-from-eval-then-test-on-eval leakage.
    """
    tasks = []
    counts_by_dir = {}
    subdirs = ['training'] if training_only else ['training', 'evaluation']
    for subdir in subdirs:
        d = os.path.join(DATA_PATH, subdir)
        if not os.path.isdir(d):
            counts_by_dir[subdir] = 0
            continue
        n = 0
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(d, fname)) as f:
                data = json.load(f)
            pairs = [{'input': p['input'], 'output': p['output']}
                     for p in data.get('train', []) if
                     len(p['input']) > 0 and len(p['output']) > 0]
            if len(pairs) >= N_PAIRS_MIN:
                tasks.append({'id': fname.replace('.json', ''), 'io': pairs,
                               'split_dir': subdir})
                n += 1
        counts_by_dir[subdir] = n
    return tasks, counts_by_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E3 FULL-CORPUS ABSTRACTION ENGINE v2 — EXECUTABLE HOLED OPERATORS")
    print(f"Config: budget={BUDGET}, max_length={MAX_LENGTH}, split_seed={SPLIT_SEED}")
    seed_ops_sorted = sorted(SEED_OPS.keys())
    print(f"Seed ops ({len(seed_ops_sorted)}): {seed_ops_sorted}")
    print("=" * 70)
    sys.stdout.flush()

    # Load and split tasks — TRAINING ONLY (matches v1; prevents eval leakage)
    all_tasks, dir_counts = load_tasks(training_only=True)
    print(f"Corpus metadata:")
    print(f"  Training-only mode: YES (matches v1, prevents abstract-from-eval leakage)")
    for d, n in dir_counts.items():
        print(f"  {d}/: {n} tasks loaded")
    print(f"  Total loaded: {len(all_tasks)} tasks")
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(all_tasks))
    source_tasks = [all_tasks[i] for i in indices[:N_SOURCE]]
    held_tasks = [all_tasks[i] for i in indices[N_SOURCE:N_SOURCE + N_HELD]]
    source_ids = {t['id'] for t in source_tasks}
    held_ids = {t['id'] for t in held_tasks}
    overlap = source_ids & held_ids
    print(f"  Source: {len(source_tasks)} tasks | Held-out: {len(held_tasks)} tasks")
    print(f"  Disjoint: {'YES' if not overlap else 'NO — OVERLAP: ' + str(overlap)}")
    print()
    sys.stdout.flush()

    # ── SOURCE ROUND ─────────────────────────────────────────────────────────

    print("SOURCE ROUND — BFS with partial-match tracking...")
    source_solved = []
    source_partials = []

    for i, task in enumerate(source_tasks):
        sol, nodes, best_partial, best_score = search_with_partials(task['io'])
        if sol:
            source_solved.append(sol)
            print(f"  {task['id']}: solved  cost={nodes}  prog={sol}")
        elif best_score >= 1 / max(len(task['io']), 1):
            source_partials.append(best_partial)

    print(f"\n  Solved: {len(source_solved)}/{N_SOURCE}")
    print(f"  Best-partial programs (score >= 1/n_pairs): {len(source_partials)}")

    partial_struct_gate = len(source_partials) > 0
    print(f"\n  PARTIAL STRUCTURE GATE:")
    print(f"    Failed tasks with partial match: {len(source_partials)}/{N_SOURCE - len(source_solved)}")
    if partial_struct_gate:
        print(f"    INFORMATIVE: {len(source_partials)} tasks show partial structure.")
    else:
        print(f"    INERT: no partial structure in failed traces. Full corpus = solved only.")
    sys.stdout.flush()

    full_corpus = source_solved + source_partials
    print(f"\n  Corpus sizes:")
    print(f"    Solved-only corpus: {len(source_solved)} programs")
    print(f"    Full corpus (solved + partials): {len(full_corpus)} programs")

    # ── MDL ANALYSIS ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("MDL ANALYSIS")
    print("=" * 70)

    # Contiguous MDL
    contiguous_lib_solved, cand_solved = run_contiguous_mdl(source_solved, "SOLVED-ONLY CONTIGUOUS")
    print()
    contiguous_lib_full, cand_full = run_contiguous_mdl(full_corpus, "FULL-CORPUS CONTIGUOUS")
    contiguous_lib = contiguous_lib_full  # use full-corpus library

    # Holed operator extraction (v2 — EXECUTABLE) with FORMATION FUNNEL
    print()
    print("  [FULL-CORPUS HOLED — EXECUTABLE] Anti-unification with single-op holes:")
    all_holed, funnel = extract_holed_ops(full_corpus)
    mdl_pos_holed = all_holed  # extract_holed_ops now returns only MDL-positive
    print(f"  FORMATION FUNNEL (Leo #11669 — discriminates starvation vs density vs bug):")
    print(f"    total_pairs_checked (all prefix/suffix combos): {funnel['total_pairs_checked']}")
    print(f"    candidates_2plus_variants (>=2 distinct hole ops): {funnel['candidates_2plus_variants']}")
    print(f"    candidates_passed_MDL (gain > 0): {funnel['candidates_passed_MDL']}")
    gd = funnel['gain_distribution']
    if gd['n'] > 0:
        print(f"    gain_distribution: n={gd['n']} min={gd['min']:.1f} "
              f"max={gd['max']:.1f} mean={gd['mean']:.1f} n_positive={gd['n_positive']}")
    print(f"    DIAGNOSIS: {funnel['diagnosis']}")
    print(f"    (a) NO_SHARED_SKELETONS = no candidates at all -> structural")
    print(f"    (b) STRUCTURAL_STARVATION = some pairs but <2 variants -> traces too homogeneous")
    print(f"    (c) DENSITY_THRESHOLD = candidates exist but gain<=0 -> count too low for MDL")
    print(f"    (d) PASSED = MDL-positive holed ops formed")
    print(f"    Total MDL-positive: {len(mdl_pos_holed)}")
    for h in mdl_pos_holed[:8]:
        variants_str = '/'.join(h['hole_variants'][:4])
        if len(h['hole_variants']) > 4:
            variants_str += '...'
        print(f"      {h['op'].name}: count={h['count']}, gain={h['gain']:.1f}, "
              f"variants=[{variants_str}]")
    if len(mdl_pos_holed) > 8:
        print(f"      ... ({len(mdl_pos_holed) - 8} more MDL-positive)")

    # Build library of executable holed ops
    executable_holed = [h['op'] for h in mdl_pos_holed]

    print(f"\n  LIBRARY SUMMARY:")
    print(f"    Contiguous compounds: {len(contiguous_lib)}")
    print(f"    Executable holed operators: {len(executable_holed)}")
    print(f"    Total library entries: {len(contiguous_lib) + len(executable_holed)}")
    sys.stdout.flush()

    # ── HELD-OUT EVALUATION ───────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("HELD-OUT EVALUATION")
    print("=" * 70)

    # BASELINE (seed ops only)
    print(f"\nBASELINE (seed, {len(seed_ops_sorted)} ops): {len(held_tasks)} held-out tasks...")
    baseline = []
    for i, task in enumerate(held_tasks):
        prog, cost, _ = search_with_holed_ops(task['io'], contiguous_lib={}, holed_ops=[],
                                               budget=BUDGET, max_length=MAX_LENGTH)
        baseline.append({'task_id': task['id'], 'prog': prog, 'cost': cost,
                          'solved': prog is not None})
        if prog:
            print(f"  [{i+1:3d}/{len(held_tasks)}] {task['id']}: solved  prog={prog}  cost={cost}")

    b_solved = sum(1 for r in baseline if r['solved'])
    b_mean = np.mean([r['cost'] for r in baseline])
    print(f"  Baseline: {b_solved}/{len(held_tasks)} solved, mean_cost={b_mean:.1f}")
    sys.stdout.flush()

    # R6 ABLATION: contiguous library only (no holed ops)
    print(f"\nR6-ABLATION (contiguous library, no holed ops): "
          f"{len(contiguous_lib)} compounds + {len(seed_ops_sorted)} seed ops...")
    r6_results = []
    for i, task in enumerate(held_tasks):
        prog, cost, _ = search_with_holed_ops(task['io'], contiguous_lib=contiguous_lib,
                                               holed_ops=[], budget=BUDGET, max_length=MAX_LENGTH)
        r6_results.append({'task_id': task['id'], 'prog': prog, 'cost': cost,
                            'solved': prog is not None})

    r6_solved = sum(1 for r in r6_results if r['solved'])
    r6_mean = np.mean([r['cost'] for r in r6_results])
    print(f"  R6-ablation: {r6_solved}/{len(held_tasks)} solved, mean_cost={r6_mean:.1f}")
    r6_delta = r6_mean - b_mean
    r6_delta_pct = 100 * r6_delta / b_mean if b_mean > 0 else 0
    print(f"  R6 vs baseline: {r6_delta:+.1f} ({r6_delta_pct:+.1f}%)")
    sys.stdout.flush()

    # WITH-HOLED-OPS: contiguous + executable holed
    print(f"\nWITH-HOLED-OPS: {len(contiguous_lib)} contiguous + "
          f"{len(executable_holed)} holed operators + {len(seed_ops_sorted)} seed ops...")
    holed_results = []
    holed_contributed = []  # tasks where a holed op SELECTED (not heuristic — exact)
    selected_holed_ops = Counter()
    for i, task in enumerate(held_tasks):
        prog, cost, sel = search_with_holed_ops(task['io'], contiguous_lib=contiguous_lib,
                                                 holed_ops=executable_holed,
                                                 budget=BUDGET, max_length=MAX_LENGTH)
        selected_by_holed = sel is not None
        if selected_by_holed:
            holed_contributed.append(task['id'])
            selected_holed_ops[sel] += 1
        holed_results.append({'task_id': task['id'], 'prog': prog, 'cost': cost,
                               'solved': prog is not None, 'holed_contributed': selected_by_holed,
                               'selected_holed': sel})

    h_solved = sum(1 for r in holed_results if r['solved'])
    h_mean = np.mean([r['cost'] for r in holed_results])
    print(f"  With-holed: {h_solved}/{len(held_tasks)} solved, mean_cost={h_mean:.1f}")
    h_delta = h_mean - b_mean
    h_delta_pct = 100 * h_delta / b_mean if b_mean > 0 else 0
    print(f"  With-holed vs baseline: {h_delta:+.1f} ({h_delta_pct:+.1f}%)")

    holed_only_delta = h_mean - r6_mean
    holed_only_pct = 100 * holed_only_delta / r6_mean if r6_mean > 0 else 0
    print(f"  Holed-only contribution (vs R6): {holed_only_delta:+.1f} ({holed_only_pct:+.1f}%)")

    # Tasks solved NEW by holed ops (not solved by seed-only or R6)
    baseline_ids = {r['task_id'] for r in baseline if r['solved']}
    holed_ids = {r['task_id'] for r in holed_results if r['solved']}
    new_solves = holed_ids - baseline_ids
    print(f"\n  HOLED-OP SELECTION (exact, not heuristic): {len(holed_contributed)} tasks")
    print(f"  Selected holed ops: {dict(selected_holed_ops.most_common(5))}")
    print(f"\n  NEW tasks solved by holed ops (not in baseline): {len(new_solves)}")
    for tid in sorted(new_solves)[:5]:
        matching = [r for r in holed_results if r['task_id'] == tid]
        if matching:
            print(f"    {tid}: prog={matching[0]['prog']}  cost={matching[0]['cost']}")
    sys.stdout.flush()

    # ── COMPOUND-APPLICABLE CHECK ─────────────────────────────────────────────

    print("\n  COMPOUND-APPLICABLE ANALYSIS:")
    # For contiguous compounds: count held-out tasks solved with these compound programs
    applicable_contiguous = 0
    applicable_holed = 0
    for r in holed_results:
        if r['prog'] is None:
            continue
        # Check if any compound from library is in the program
        for cname, cseq in contiguous_lib.items():
            if count_subseq(r['prog'], cseq) > 0:
                applicable_contiguous += 1
                break
        # Check if program was solved by holed op (uses a holed op compound structure)
        if r['holed_contributed']:
            applicable_holed += 1

    print(f"    Contiguous-compound-applicable held tasks: {applicable_contiguous}")
    print(f"    Holed-op-applicable held tasks: {applicable_holed}")

    # ── RESULTS ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("E3 FULL-CORPUS v2 RESULT")
    print("=" * 70)
    print(f"Source solved-corpus: {len(source_solved)} programs")
    print(f"Source partial-corpus: {len(source_partials)} programs (inert={not partial_struct_gate})")
    print(f"Contiguous MDL-positive compounds: {len(contiguous_lib)}")
    print(f"Executable holed operators (MDL-positive): {len(executable_holed)}")
    print()
    print(f"HELD-OUT EVALUATION:")
    print(f"  BASELINE (seed only):     mean={b_mean:.1f}  solved={b_solved}/{len(held_tasks)}")
    print(f"  R6 (contiguous only):     mean={r6_mean:.1f}  "
          f"delta={r6_delta:+.1f} ({r6_delta_pct:+.1f}%)  solved={r6_solved}/{len(held_tasks)}")
    print(f"  WITH-HOLED:               mean={h_mean:.1f}  "
          f"delta={h_delta:+.1f} ({h_delta_pct:+.1f}%)  solved={h_solved}/{len(held_tasks)}")
    print(f"  Holed-only contribution:  {holed_only_delta:+.1f} ({holed_only_pct:+.1f}%)")
    print(f"  NEW solves from holed ops: {len(new_solves)}")
    print()

    # Verdict
    THRESHOLD_NEW = 1
    THRESHOLD_REDUCTION = -5.0  # 5% reduction is "signal"

    if len(executable_holed) == 0:
        verdict = "NO_EXECUTABLE_HOLED_OPS_FORMED"
        print("VERDICT: NO_EXECUTABLE_HOLED_OPS_FORMED")
        print("  Anti-unification found no MDL-positive holed patterns.")
        print("  Corpus too sparse for holes. Try larger budget / more tasks.")
    elif len(new_solves) >= THRESHOLD_NEW:
        verdict = "FORWARD_HOLED_OPS_HELP"
        print(f"VERDICT: FORWARD — holed ops solve {len(new_solves)} new held-out tasks.")
        print(f"  Self-compilation with richer operators IS load-bearing.")
    elif h_delta_pct < THRESHOLD_REDUCTION:
        verdict = "FORWARD_HOLED_OPS_REDUCE_COST"
        print(f"VERDICT: FORWARD — holed ops reduce held-out mean cost by {abs(h_delta_pct):.1f}%.")
        print(f"  Cost reduction exceeds {abs(THRESHOLD_REDUCTION)}% threshold.")
    elif r6_delta_pct > -1.0 and h_delta_pct > -1.0:
        verdict = "KILL_RICHER_OPS_FLAT"
        print("VERDICT: VALID KILL — richer operators (contiguous + holed) are executable")
        print("  and available to search. Held-out reduction is still flat.")
        print("  R6 ablation: flat. Holed-op contribution: flat.")
        print("  ARC held-out does not benefit from richer self-compilation.")
    else:
        verdict = "PARTIAL_SIGNAL"
        print(f"VERDICT: PARTIAL_SIGNAL — contiguous {r6_delta_pct:+.1f}%, "
              f"holed contribution {holed_only_pct:+.1f}%")

    sys.stdout.flush()

    result = {
        "experiment": "E3-full-corpus-v2-executable-holes-with-funnel",
        "note": (
            "v2: holed operators EXECUTABLE. Formation funnel instruments all 3 failure causes. "
            "Training-only corpus (400 tasks). Source/held-out DISJOINT."
        ),
        "corpus_metadata": {
            "total_loaded": len(all_tasks),
            "training_only": True,
            "dir_counts": dir_counts,
            "n_source": len(source_tasks),
            "n_held": len(held_tasks),
            "source_held_disjoint": not overlap,
        },
        "config": {
            "budget": BUDGET, "max_length": MAX_LENGTH, "split_seed": SPLIT_SEED,
        },
        "source_solved": len(source_solved),
        "source_partials": len(source_partials),
        "partial_struct_gate": "INFORMATIVE" if partial_struct_gate else "INERT",
        "contiguous_compounds": len(contiguous_lib),
        "executable_holed_ops": len(executable_holed),
        "formation_funnel": funnel,
        "holed_ops_detail": [
            {"name": h['op'].name, "prefix": h['op'].prefix, "suffix": h['op'].suffix,
             "count": h['count'], "gain": float(h['gain']), "hole_variants": h['hole_variants']}
            for h in mdl_pos_holed[:20]
        ],
        "held_out": {
            "baseline": {"mean_cost": float(b_mean), "solved": b_solved},
            "r6_contiguous": {
                "mean_cost": float(r6_mean), "solved": r6_solved,
                "delta": float(r6_delta), "delta_pct": float(r6_delta_pct),
            },
            "with_holed": {
                "mean_cost": float(h_mean), "solved": h_solved,
                "delta": float(h_delta), "delta_pct": float(h_delta_pct),
            },
            "holed_only_contribution": float(holed_only_pct),
            "holed_selection": {
                "tasks_selected_holed": len(holed_contributed),
                "selected_ops": dict(selected_holed_ops.most_common(10)),
            },
            "new_solves_from_holed": len(new_solves),
            "new_solve_ids": sorted(new_solves),
        },
        "verdict": verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {RESULT_PATH}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
