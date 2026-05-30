"""
E4 Holed-Operators Characterization -- Aggregate-Net R*_holed (Leo #11665 + #11672).

Conditions (Leo #11672 exact spec):
  A = seed-only (baseline)
  B = seed + ALL MDL-learned contiguous compounds -- distractor CAN enter here
  C = seed + all-MDL-learned contiguous + holed ops
  planted_only = seed + ONLY planted motif op (attribution control -- NOT a condition for R*)

R* definition (Leo #11672 exact):
  aggregate_delta(X) = mean_cost_X(all_held) - mean_cost_A(all_held)
  rstar_X = min rho where aggregate_delta(X) < -MARGIN AND variance band excludes 0
  R* is AGGREGATE-NET, not motif-subset. Overhead from non-transferable compounds COUNTS.

Kai fixes (Leo #11665 + #11660):
  #1: Distractor roll INDEPENDENT of rho (separate draw).
  #2: planted_only as attribution control; B uses general MDL (all-learned).
  Distractor choice: dup_h__fv -- no single-op equivalent, persists in solved programs.

Attribution: report motif vs distractor fraction of B's aggregate improvement.
Holed selection: count tasks where BFS actually SELECTED a holed op (presence != selection).
"""

import numpy as np
import json
import itertools
import sys
from collections import defaultdict, Counter

# ── DSL ──────────────────────────────────────────────────────────────────────

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
SORTED_SEEDS = sorted(SEED_OPS.keys())

# ── Motif + distractor ────────────────────────────────────────────────────────

MOTIF = ['mir_h', 'mir_v']
MOTIF_NAME = 'mir_h__mir_v'

# dup_h__fv = hstack-duplicate then flip-vertical. No single-op equivalent.
# BFS finds it at length 2, not compressed. Distractor WILL enter MDL library.
DISTRACTOR_S = ['dup_h', 'fv']
DISTRACTOR_NAME = 'dup_h__fv'

RHO_D = 0.4   # independent of rho (Kai fix #1)

_EXCLUDE = set(MOTIF) | set(DISTRACTOR_S)
CLEAN_PAD = [op for op in sorted(SEED_OPS.keys()) if op not in _EXCLUDE]
# CLEAN_PAD = ['crop', 'down2', 'dup_v', 'fh', 'id', 'rot', 'tr', 'up2'] (8 ops)

# ── Config ────────────────────────────────────────────────────────────────────

PROGRAM_LENGTH = 4
BUDGET = 35000
N_SOURCE = 100
N_HELD = 100
N_PAIRS = 3
MAX_GRID_AREA = 400
RHO_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
N_SEEDS = 5           # more seeds for variance band (Leo #11665)
MASTER_SEED = 42
AGGREGATE_MARGIN = 5.0  # % aggregate net reduction required for R*

# ── Grid ops ─────────────────────────────────────────────────────────────────

def apply_op(name, grid, library=None):
    try:
        g = np.array(grid, dtype=np.int64)
        if library and name in library:
            for sub in library[name]:
                r = apply_op(sub, g, library)
                if r is None:
                    return None
                g = r
            return g
        return SEED_OPS[name](g)
    except Exception:
        return None


def apply_program(ops, grid, library=None):
    g = np.array(grid, dtype=np.int64)
    for name in ops:
        r = apply_op(name, g, library)
        if r is None:
            return None
        g = r
    return g


def check_program(ops, task_io, library=None):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        r = apply_program(ops, inp, library)
        if r is None or r.shape != out.shape or not np.array_equal(r, out):
            return False
    return True


# ── HoledOp ──────────────────────────────────────────────────────────────────

class HoledOp:
    """Single-op-hole: (prefix) o [?] o (suffix). Executable via try_fill()."""
    __slots__ = ('prefix', 'suffix', 'name', '_total_length')

    def __init__(self, prefix, suffix):
        self.prefix = list(prefix)
        self.suffix = list(suffix)
        p = '__'.join(prefix) if prefix else 'eps'
        s = '__'.join(suffix) if suffix else 'eps'
        self.name = f"{p}___H___{s}"
        self._total_length = len(prefix) + 1 + len(suffix)

    def try_fill(self, task_io, library=None):
        """Returns (filled_program, inner_nodes_tried). None if no fill found."""
        lib = library or {}
        prefix_states = []
        for pair in task_io:
            inp = np.array(pair['input'], dtype=np.int64)
            mid = apply_program(self.prefix, inp, lib)
            if mid is None:
                return None, len(SORTED_SEEDS)
            prefix_states.append((mid, np.array(pair['output'], dtype=np.int64)))

        for i, hole_op in enumerate(SORTED_SEEDS):
            all_match = True
            for (mid, out) in prefix_states:
                mid2 = apply_op(hole_op, mid, lib)
                if mid2 is None:
                    all_match = False
                    break
                result = apply_program(self.suffix, mid2, lib)
                if result is None or result.shape != out.shape or not np.array_equal(result, out):
                    all_match = False
                    break
            if all_match:
                return self.prefix + [hole_op] + self.suffix, i + 1
        return None, len(SORTED_SEEDS)

    def mdl_gain(self, count):
        return count * (self._total_length - 2) - self._total_length


# ── BFS (with holed-op tracking) ─────────────────────────────────────────────

def search(task_io, contiguous_lib=None, holed_ops=None,
           max_length=PROGRAM_LENGTH, budget=BUDGET):
    """Standard BFS. Holed ops tried first (outer-length-1 position)."""
    lib = contiguous_lib or {}
    holed = holed_ops or []

    lib_ops = sorted(lib.keys())
    seen = set()
    ordered_ops = []
    for n in lib_ops + SORTED_SEEDS:
        if n not in seen:
            seen.add(n)
            ordered_ops.append(n)

    nodes = 0
    selected_holed = None

    for hop in holed:
        nodes += 1
        if nodes > budget:
            return None, nodes, None
        filled, inner = hop.try_fill(task_io, lib)
        nodes += inner
        if nodes > budget:
            return filled, nodes, hop.name if filled else None
        if filled is not None:
            return filled, nodes, hop.name

    for length in range(1, max_length + 1):
        for ops in itertools.product(ordered_ops, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes, None
            if check_program(list(ops), task_io, lib):
                return list(ops), nodes, None
    return None, nodes, None


# ── MDL library ──────────────────────────────────────────────────────────────

def count_subseq(prog, candidate):
    count = 0
    for i in range(len(prog) - len(candidate) + 1):
        if tuple(prog[i:i + len(candidate)]) == tuple(candidate):
            count += 1
    return count


def build_general_mdl_lib(source_solved):
    """General MDL over source solved programs. Distractor CAN enter."""
    candidates = defaultdict(int)
    for prog in source_solved:
        for L in range(2, len(prog) + 1):
            for i in range(len(prog) - L + 1):
                sub = tuple(prog[i:i + L])
                candidates[sub] += count_subseq(prog, list(sub))

    lib = {}
    for sub, total_count in candidates.items():
        gain = total_count * (len(sub) - 1) - len(sub)
        if gain > 0:
            name = '__'.join(sub)
            lib[name] = list(sub)
    return lib


def build_motif_only_lib(source_solved):
    """Attribution control: ONLY planted motif compound if MDL-positive."""
    motif_tuple = tuple(MOTIF)
    count = sum(count_subseq(prog, MOTIF) for prog in source_solved)
    gain = count * (len(MOTIF) - 1) - len(MOTIF)
    if gain > 0:
        return {MOTIF_NAME: MOTIF}, count, gain
    return {}, count, gain


def classify_compound(name):
    """Classify library compound: motif / distractor / other."""
    if name == MOTIF_NAME:
        return 'motif'
    if name == DISTRACTOR_NAME:
        return 'distractor'
    return 'other'


def extract_holed_ops(programs):
    """Single-op-hole anti-unification. MDL filter: gain > 0."""
    pattern_programs = defaultdict(list)
    for prog in programs:
        L = len(prog)
        for hole_pos in range(L):
            prefix = tuple(prog[:hole_pos])
            suffix = tuple(prog[hole_pos + 1:])
            pattern_programs[(prefix, suffix)].append(prog)

    holed_ops = []
    for (prefix, suffix), progs in pattern_programs.items():
        hole_ops = set()
        for prog in progs:
            hole_pos = len(prefix)
            if len(prog) == len(prefix) + 1 + len(suffix):
                hole_ops.add(prog[hole_pos])
        if len(hole_ops) < 2:
            continue
        hop = HoledOp(prefix, suffix)
        gain = hop.mdl_gain(len(progs))
        if gain > 0:
            holed_ops.append({'op': hop, 'count': len(progs),
                              'gain': gain, 'variants': sorted(hole_ops)})
    holed_ops.sort(key=lambda x: -x['gain'])
    return holed_ops


# ── Task generation ───────────────────────────────────────────────────────────

def gen_input_grid(rng):
    for _ in range(20):
        h = rng.integers(2, 5)
        w = rng.integers(2, 5)
        grid = rng.integers(0, 6, size=(h, w), dtype=np.int64)
        if len(np.unique(grid)) >= 2:
            return grid
    g = np.zeros((3, 3), dtype=np.int64)
    g[0, 0] = 1
    return g


def _plant(ops, seq, rng):
    start = rng.integers(0, len(ops) - len(seq) + 1)
    for i, s in enumerate(seq):
        ops[start + i] = s
    return ops


def gen_source_program(rho, rng):
    """Kai fix #1: INDEPENDENT rolls for motif and distractor."""
    ops = [rng.choice(CLEAN_PAD) for _ in range(PROGRAM_LENGTH)]
    if rng.random() < rho:
        ops = _plant(ops, MOTIF, rng)
    if rng.random() < RHO_D:
        # Plant distractor at a random start position (may overlap motif region)
        dist_start = rng.integers(0, PROGRAM_LENGTH - len(DISTRACTOR_S) + 1)
        for i, d in enumerate(DISTRACTOR_S):
            ops[dist_start + i] = d
    return ops


def gen_held_program(rho, rng):
    """Held-out: only MOTIF at rate rho, no distractor."""
    ops = [rng.choice(CLEAN_PAD) for _ in range(PROGRAM_LENGTH)]
    if rng.random() < rho:
        ops = _plant(ops, MOTIF, rng)
    return ops


def program_contains(ops, pattern):
    pt = tuple(pattern)
    for i in range(len(ops) - len(pt) + 1):
        if tuple(ops[i:i + len(pt)]) == pt:
            return True
    return False


def gen_task(ops, rng):
    pairs = []
    attempts = 0
    while len(pairs) < N_PAIRS and attempts < 60:
        attempts += 1
        inp = gen_input_grid(rng)
        out = apply_program(ops, inp)
        if out is None or out.size > MAX_GRID_AREA or out.size == 0:
            continue
        if out.shape == inp.shape and np.array_equal(out, inp):
            continue
        pairs.append({'input': inp.tolist(), 'output': out.tolist()})
    return pairs if len(pairs) == N_PAIRS else None


# ── Single rho, single seed run ───────────────────────────────────────────────

def run_one(rho, rng_seed):
    rng = np.random.default_rng(rng_seed)

    # Generate source tasks
    source_tasks = []
    att = 0
    while len(source_tasks) < N_SOURCE and att < N_SOURCE * 30:
        att += 1
        ops = gen_source_program(rho, rng)
        pairs = gen_task(ops, rng)
        if pairs is None:
            continue
        if any(check_program([op], pairs) for op in SORTED_SEEDS):
            continue
        source_tasks.append({'ops': ops, 'io': pairs,
                             'has_motif': program_contains(ops, MOTIF),
                             'has_dist': program_contains(ops, DISTRACTOR_S)})

    # Generate held tasks (motif-only, no distractor)
    held_tasks = []
    att = 0
    while len(held_tasks) < N_HELD and att < N_HELD * 30:
        att += 1
        ops = gen_held_program(rho, rng)
        pairs = gen_task(ops, rng)
        if pairs is None:
            continue
        if any(check_program([op], pairs) for op in SORTED_SEEDS):
            continue
        held_tasks.append({'ops': ops, 'io': pairs,
                           'has_motif': program_contains(ops, MOTIF)})

    # BFS source (seed-only, collect solved programs)
    source_solved = []
    for task in source_tasks:
        prog, _, _ = search(task['io'])
        if prog:
            source_solved.append(prog)

    # Build libraries
    lib_B = build_general_mdl_lib(source_solved)  # all-MDL, distractor can enter
    lib_planted, motif_count, motif_gain = build_motif_only_lib(source_solved)
    holed_data = extract_holed_ops(source_solved)
    executable_holed = [h['op'] for h in holed_data]

    # Classify library compounds in B
    motif_in_lib = MOTIF_NAME in lib_B
    distractor_in_lib = DISTRACTOR_NAME in lib_B
    n_lib_total = len(lib_B)
    n_lib_motif = sum(1 for k in lib_B if classify_compound(k) == 'motif')
    n_lib_dist = sum(1 for k in lib_B if classify_compound(k) == 'distractor')
    n_lib_other = sum(1 for k in lib_B if classify_compound(k) == 'other')

    # Evaluate ALL conditions per held task in ONE pass
    costs_A, costs_B, costs_C, costs_P = [], [], [], []
    selected_holed_by_C = []
    holed_solved_new = 0

    for task in held_tasks:
        prog_A, cost_A, _ = search(task['io'], {}, [])
        prog_B, cost_B, _ = search(task['io'], lib_B, [])
        prog_C, cost_C, sel_hop = search(task['io'], lib_B, executable_holed)
        prog_P, cost_P, _ = search(task['io'], lib_planted, [])
        costs_A.append(cost_A)
        costs_B.append(cost_B)
        costs_C.append(cost_C)
        costs_P.append(cost_P)
        if sel_hop:
            selected_holed_by_C.append(sel_hop)
        if sel_hop and prog_A is None:
            holed_solved_new += 1

    mean_A = float(np.mean(costs_A))
    mean_B = float(np.mean(costs_B))
    mean_C = float(np.mean(costs_C))
    mean_P = float(np.mean(costs_P))

    def pct(a, b):
        return 100 * (b - a) / a if a > 0 else 0.0

    # Aggregate deltas (ALL held tasks, overhead included -- this is R*)
    agg_delta_B = float(pct(mean_A, mean_B))
    agg_delta_C = float(pct(mean_A, mean_C))
    agg_delta_P = float(pct(mean_A, mean_P))
    holed_contrib = float(pct(mean_B, mean_C))

    # Attribution: compare B vs planted_only
    # B improvement = agg_delta_B; planted_only improvement = agg_delta_P
    # Distractor contribution = B improvement - planted_only improvement (if distractor in lib)
    motif_fraction = None
    distractor_fraction = None
    if agg_delta_B != 0 and agg_delta_P != 0:
        motif_fraction = float(agg_delta_P / agg_delta_B) if agg_delta_B != 0 else None

    # Motif-applicable partition (diagnostic only, NOT R*)
    motif_idxs = [i for i, t in enumerate(held_tasks) if t['has_motif']]
    motif_partition_B, motif_partition_C = None, None
    if motif_idxs:
        ma_B = np.mean([costs_A[i] for i in motif_idxs])
        mb_B = np.mean([costs_B[i] for i in motif_idxs])
        mc_C = np.mean([costs_C[i] for i in motif_idxs])
        motif_partition_B = float(pct(ma_B, mb_B))
        motif_partition_C = float(pct(ma_B, mc_C))

    holed_selection_counts = Counter(selected_holed_by_C)

    return {
        'rho': rho,
        'rng_seed': rng_seed,
        'n_source': len(source_tasks),
        'n_held': len(held_tasks),
        'obs_source_motif': sum(1 for t in source_tasks if t['has_motif']),
        'obs_source_dist': sum(1 for t in source_tasks if t['has_dist']),
        'obs_held_motif': sum(1 for t in held_tasks if t['has_motif']),
        'source_solved': len(source_solved),
        # Library B (general MDL)
        'n_lib_total': n_lib_total,
        'motif_in_lib': motif_in_lib,
        'distractor_in_lib': distractor_in_lib,
        'n_lib_motif': n_lib_motif,
        'n_lib_dist': n_lib_dist,
        'n_lib_other': n_lib_other,
        # Attribution control (planted only)
        'motif_count_in_solved': motif_count,
        'motif_gain': motif_gain,
        # Holed ops
        'n_holed_ops': len(executable_holed),
        'holed_top3': [
            {'name': h['op'].name, 'count': h['count'], 'gain': float(h['gain'])}
            for h in holed_data[:3]
        ],
        # AGGREGATE deltas (R* source -- all held tasks, overhead included)
        'agg_delta_B': agg_delta_B,
        'agg_delta_C': agg_delta_C,
        'agg_delta_P': agg_delta_P,
        'holed_contribution_pct': holed_contrib,
        # Attribution
        'motif_fraction_of_B': motif_fraction,
        # Holed selection (evidence)
        'selected_holed_count': len(selected_holed_by_C),
        'selected_holed_ops': dict(holed_selection_counts.most_common(5)),
        'holed_new_solves': holed_solved_new,
        # Motif-applicable partition (diagnostic only, NOT R*)
        'motif_partition_B_pct': motif_partition_B,
        'motif_partition_C_pct': motif_partition_C,
        'n_motif_applicable': len(motif_idxs),
    }


# ── Multi-seed aggregation ────────────────────────────────────────────────────

def run_rho_multiseed(rho, n_seeds, master_seed):
    results = []
    for s in range(n_seeds):
        seed = master_seed + s * 1000 + int(rho * 10000)
        r = run_one(rho, seed)
        results.append(r)

    def m(key): return float(np.mean([r[key] for r in results]))
    def sd(key): return float(np.std([r[key] for r in results]))

    agg_B_vals = [r['agg_delta_B'] for r in results]
    agg_C_vals = [r['agg_delta_C'] for r in results]
    agg_P_vals = [r['agg_delta_P'] for r in results]

    return {
        'rho': rho,
        'n_seeds': n_seeds,
        # Aggregate R* metrics (all held, overhead included)
        'agg_delta_B_mean': float(np.mean(agg_B_vals)),
        'agg_delta_B_std': float(np.std(agg_B_vals)),
        'agg_delta_C_mean': float(np.mean(agg_C_vals)),
        'agg_delta_C_std': float(np.std(agg_C_vals)),
        'agg_delta_P_mean': float(np.mean(agg_P_vals)),
        'agg_delta_P_std': float(np.std(agg_P_vals)),
        'holed_contribution_mean': m('holed_contribution_pct'),
        # Library state
        'motif_in_lib_fraction': float(np.mean([r['motif_in_lib'] for r in results])),
        'distractor_in_lib_fraction': float(np.mean([r['distractor_in_lib'] for r in results])),
        'n_lib_total_mean': m('n_lib_total'),
        'n_lib_motif_mean': m('n_lib_motif'),
        'n_lib_dist_mean': m('n_lib_dist'),
        # Holed ops
        'n_holed_ops_mean': m('n_holed_ops'),
        'selected_holed_count_mean': m('selected_holed_count'),
        # Observation densities
        'obs_source_motif_density': m('obs_source_motif') / N_SOURCE,
        'obs_source_dist_density': m('obs_source_dist') / N_SOURCE,
        'obs_held_motif_density': m('obs_held_motif') / N_HELD,
        # Motif-applicable partition (diagnostic, NOT R*)
        'motif_partition_B_mean': float(np.mean([r['motif_partition_B_pct']
                                                  for r in results if r['motif_partition_B_pct'] is not None]) or 0),
        'motif_partition_C_mean': float(np.mean([r['motif_partition_C_pct']
                                                  for r in results if r['motif_partition_C_pct'] is not None]) or 0),
        'n_motif_applicable_mean': m('n_motif_applicable'),
        # Attribution
        'motif_fraction_mean': float(np.mean([r['motif_fraction_of_B']
                                               for r in results if r['motif_fraction_of_B'] is not None]) or 0),
        'per_seed': results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E4 HOLED-OPERATORS -- AGGREGATE-NET R*_holed (Leo #11672 exact spec)")
    print(f"MOTIF: {MOTIF_NAME}={MOTIF}")
    print(f"DISTRACTOR: {DISTRACTOR_NAME}={DISTRACTOR_S} (INDEPENDENT roll, rho_d={RHO_D})")
    print(f"CLEAN_PAD ({len(CLEAN_PAD)}): {sorted(CLEAN_PAD)}")
    print(f"Config: prog_len={PROGRAM_LENGTH}, budget={BUDGET}, "
          f"n_source={N_SOURCE}, n_held={N_HELD}, n_seeds={N_SEEDS}")
    print(f"R* definition: min rho where aggregate_delta < -{AGGREGATE_MARGIN}% "
          f"AND variance band excludes 0 (all held tasks, overhead included)")
    print("NOTE: rho=0 = synthetic low-reuse calibration. ARC rho-axis UNMEASURED.")
    print("CONDITIONS: A=seed | B=seed+all-MDL | C=seed+all-MDL+holed | P=seed+motif-only")
    print("=" * 72)
    sys.stdout.flush()

    curve = []
    rstar_contiguous = None  # from aggregate_delta(B)
    rstar_holed = None       # from aggregate_delta(C)

    for rho in RHO_VALUES:
        print(f"\n--- rho={rho:.2f} ({N_SEEDS} seeds) ---")
        sys.stdout.flush()
        pt = run_rho_multiseed(rho, N_SEEDS, MASTER_SEED)
        curve.append(pt)

        # Print aggregate R* metrics
        bm, bs = pt['agg_delta_B_mean'], pt['agg_delta_B_std']
        cm, cs = pt['agg_delta_C_mean'], pt['agg_delta_C_std']
        pm, ps = pt['agg_delta_P_mean'], pt['agg_delta_P_std']

        print(f"  obs: src_motif={pt['obs_source_motif_density']:.2f} "
              f"src_dist={pt['obs_source_dist_density']:.2f} "
              f"held_motif={pt['obs_held_motif_density']:.2f}")
        print(f"  lib B: {pt['n_lib_total_mean']:.0f} compounds "
              f"(motif={pt['motif_in_lib_fraction']:.2f} dist={pt['distractor_in_lib_fraction']:.2f})")
        print(f"  holed_ops={pt['n_holed_ops_mean']:.1f} "
              f"selected={pt['selected_holed_count_mean']:.1f}")

        b_marker = ""
        if rstar_contiguous is None and bm < -AGGREGATE_MARGIN and (bm + bs) < 0:
            rstar_contiguous = rho
            b_marker = " <-- R*_contiguous"
        print(f"  AGGREGATE B (all-MDL contiguous): {bm:+.1f}% +/-{bs:.1f}{b_marker}")

        c_marker = ""
        if rstar_holed is None and cm < -AGGREGATE_MARGIN and (cm + cs) < 0:
            rstar_holed = rho
            c_marker = " <-- R*_holed"
        print(f"  AGGREGATE C (+holed ops):         {cm:+.1f}% +/-{cs:.1f}{c_marker}")
        print(f"  PLANTED-ONLY P (attribution ctrl): {pm:+.1f}% +/-{ps:.1f}")
        print(f"  Holed contribution (B->C):         {pt['holed_contribution_mean']:+.1f}%")

        # Diagnostic: motif-applicable subset (NOT R*)
        if pt['n_motif_applicable_mean'] > 0:
            print(f"  [diagnostic] motif-applicable(n={pt['n_motif_applicable_mean']:.0f}): "
                  f"B={pt['motif_partition_B_mean']:+.1f}% C={pt['motif_partition_C_mean']:+.1f}%")
        sys.stdout.flush()

    # Summary table
    print("\n" + "=" * 72)
    print("AGGREGATE-NET R* CURVE (all held tasks, overhead included)")
    print("=" * 72)
    print(f"{'rho':>5}  {'src_d':>5}  {'lib_B':>5}  {'dist_in':>7}  "
          f"{'delta_B':>8}  {'delta_C':>8}  {'delta_P':>8}  {'n_holed':>7}  {'sel_h':>5}")
    print("-" * 74)
    for pt in curve:
        r = pt['rho']
        bm = f"{pt['agg_delta_B_mean']:+.1f}"
        cm = f"{pt['agg_delta_C_mean']:+.1f}"
        pm = f"{pt['agg_delta_P_mean']:+.1f}"
        b_tag = " R*c" if rstar_contiguous == r else ""
        c_tag = " R*h" if rstar_holed == r else ""
        print(f"  {r:.2f}  {pt['obs_source_motif_density']:.2f}  "
              f"{pt['n_lib_total_mean']:5.0f}  {pt['distractor_in_lib_fraction']:.2f}     "
              f"{bm:>8}  {cm:>8}  {pm:>8}  "
              f"{pt['n_holed_ops_mean']:7.1f}  {pt['selected_holed_count_mean']:5.1f}"
              f"{b_tag}{c_tag}")

    print()
    rsc = rstar_contiguous
    rsh = rstar_holed
    if rsc:
        print(f"R*_contiguous = {rsc} (aggregate B < -{AGGREGATE_MARGIN}%, band excludes 0)")
    else:
        print("R*_contiguous: NOT FOUND (aggregate B never crosses threshold)")
    if rsh:
        print(f"R*_holed = {rsh} (aggregate C < -{AGGREGATE_MARGIN}%, band excludes 0)")
        if rsc and rsh < rsc:
            print(f"=> Holed ops lower R*: {rsh} vs {rsc}")
    else:
        print("R*_holed: NOT FOUND")

    # Distractor concentration verdict
    print("\nDISTRACTOR CONCENTRATION:")
    for pt in curve:
        dist_f = pt['distractor_in_lib_fraction']
        if dist_f > 0:
            mf = pt['motif_fraction_mean']
            print(f"  rho={pt['rho']:.2f}: distractor_in_lib={dist_f:.2f}, "
                  f"motif_fraction_of_B_improvement={mf:.2f}")

    verdict = "UNDETERMINED"
    if rsh and rsc:
        if rsh < rsc:
            verdict = f"HOLED_LOWERS_RSTAR_{rsc}_to_{rsh}"
        else:
            verdict = f"HOLED_SAME_OR_HIGHER_RSTAR"
    elif rsh:
        verdict = f"RSTAR_HOLED={rsh}"
    elif rsc:
        verdict = f"RSTAR_CONTIGUOUS={rsc}_HOLED_NOT_FOUND"
    elif all(pt['agg_delta_B_mean'] >= 0 for pt in curve):
        verdict = "OVERHEAD_DOMINATES_NO_AGGREGATE_NET_GAIN"

    print(f"\nVerdict: {verdict}")
    sys.stdout.flush()

    result = {
        "experiment": "E4-holed-operators-v2",
        "spec": "Leo #11672 aggregate-net R*",
        "motif": MOTIF, "motif_name": MOTIF_NAME,
        "distractor": DISTRACTOR_S, "distractor_name": DISTRACTOR_NAME,
        "kai_fix_1": "distractor roll independent of rho",
        "kai_fix_2": "B = all-MDL (distractor can enter); planted_only = attribution control",
        "r_star_definition": f"min rho where aggregate_delta < -{AGGREGATE_MARGIN}% AND variance band excludes 0",
        "framing": "rho=0 = synthetic calibration. ARC rho-axis UNMEASURED.",
        "conditions": {
            "A": "seed-only baseline",
            "B": "seed + all-MDL-learned contiguous (distractor can enter)",
            "C": "seed + all-MDL-learned contiguous + executable holed ops",
            "P": "seed + planted-motif-only (attribution control)",
        },
        "config": {
            "program_length": PROGRAM_LENGTH, "budget": BUDGET,
            "n_source": N_SOURCE, "n_held": N_HELD, "n_seeds": N_SEEDS,
            "rho_d": RHO_D, "aggregate_margin_pct": AGGREGATE_MARGIN,
            "clean_pad": sorted(CLEAN_PAD), "master_seed": MASTER_SEED,
        },
        "rho_values": RHO_VALUES,
        "curve": curve,
        "r_star_contiguous": rsc,
        "r_star_holed": rsh,
        "verdict": verdict,
    }

    out = "incoming/arc-agi1-visa/03_R4_transfer_wall/E4_holed_result.json"
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {out}")
    sys.stdout.flush()


if __name__ == '__main__':
    main()
