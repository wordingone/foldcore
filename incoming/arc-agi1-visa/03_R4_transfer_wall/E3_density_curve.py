"""
E3 Trace-Density Amplification Curve (Leo #11683, 2026-05-30).

12-op self-compilation closed as FORMATION-DENSITY NEGATIVE.
Attributed cause: trace-starvation (28 programs, 4 skeletons, 0 MDL-positive).
This script amplifies the source trace corpus in steps and reports the holed
formation funnel at each step -> candidates_passed_MDL vs trace-count CURVE.

DISCRIMINATES:
  (A) trace-count insufficiency: curve climbs with density -> amplification
      bootstraps self-compilation on 12-op -> continue.
  (B) representational-base insufficiency: curve stays 0 at max density ->
      12-op programs genuinely structure-poor -> base is the ceiling.

STEPS:
  1. real-200: 200 source tasks from training-only (matches E3 v2 v3, reference point).
  2. full-real: all training+eval MINUS disjoint held-out 200 (~600 source tasks).
  3. +aug: step-2 source + D4 symmetry augmentation (SOURCE-ONLY, 6 transforms).
     Held-out stays original 200 DISJOINT tasks across ALL steps.

AUGMENTATION (step 3 only, SOURCE-ONLY):
  D4 symmetry: rot90, rot180, rot270, fh, fv, tr applied to all pairs.
  Each augmented copy is a new source task; BFS finds its program.
  NO augmented copies in held-out (zero leakage — enforced by ID-overlap AND content-hash).

KAI REVIEWER GATES (wired per #11689, before running):
  Gate 1: Result JSON includes held-out task IDs + per-step source IDs + counts.
  Gate 2: Source construction named explicitly per step, incl. split method.
  Gate 3 [LOAD-BEARING]: Content-hash augmented-leakage check — not just ID-overlap.
           source IDs exclude held IDs before augmentation AND content/payload-hash
           check that no augmented source copy matches any held task content.
           ARC has symmetric/near-duplicate tasks; hash check is the only real guard.
  Gate 4: Full curve table per step (trace count, program count, full funnel).
"""

import numpy as np
import json
import itertools
import sys
import os
from collections import Counter, defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/E3_density_curve_result.json"

BUDGET = 10000
MAX_LENGTH = 5
SPLIT_SEED = 42
N_HELD = 200

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

# D4 symmetry augmentation transforms (6 non-identity D4 elements available in DSL)
# rot90=rot, rot180=rot+rot, rot270=rot+rot+rot, fh, fv, transpose=tr
AUG_TRANSFORMS = {
    'rot90':  ['rot'],
    'rot180': ['rot', 'rot'],
    'rot270': ['rot', 'rot', 'rot'],
    'fh':     ['fh'],
    'fv':     ['fv'],
    'tr':     ['tr'],
}


# ── Grid operations ───────────────────────────────────────────────────────────

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


# ── Augmentation ──────────────────────────────────────────────────────────────

def augment_task(task, transform_ops, aug_name):
    """Apply a D4 transform to all (input, output) pairs of a task."""
    aug_pairs = []
    for pair in task['io']:
        try:
            inp = np.array(pair['input'], dtype=np.int64)
            out = np.array(pair['output'], dtype=np.int64)
            aug_inp = apply_program(transform_ops, inp)
            aug_out = apply_program(transform_ops, out)
            if aug_inp is None or aug_out is None:
                return None
            aug_pairs.append({'input': aug_inp.tolist(), 'output': aug_out.tolist()})
        except Exception:
            return None
    if not aug_pairs:
        return None
    return {
        'id': task['id'] + f'_aug_{aug_name}',
        'io': aug_pairs,
        'split_dir': task.get('split_dir', 'aug'),
        'is_augmented': True,
        'aug_type': aug_name,
        'source_id': task['id'],
    }


def augment_source_tasks(source_tasks):
    """Apply all D4 transforms to source tasks. Returns augmented copies only."""
    augmented = []
    for task in source_tasks:
        for aug_name, transform_ops in AUG_TRANSFORMS.items():
            aug = augment_task(task, transform_ops, aug_name)
            if aug is not None:
                augmented.append(aug)
    return augmented


# ── Gate 3: Content-hash leakage check ────────────────────────────────────────

def task_content_hash(task):
    """Canonical hash of a task's (input, output) pair content. Order-invariant."""
    pair_hashes = []
    for pair in task['io']:
        inp = tuple(tuple(int(c) for c in row) for row in pair['input'])
        out = tuple(tuple(int(c) for c in row) for row in pair['output'])
        pair_hashes.append((inp, out))
    return frozenset(pair_hashes)


def content_hash_leakage_check(aug_tasks, held_tasks):
    """
    Gate 3 (load-bearing): checks that no augmented source task's content
    matches any held task's content.

    ARC-AGI-1 has symmetric/near-duplicate tasks across train/eval — a held task
    could be a D4 image of a source task. ID-overlap is trivially safe by
    construction; only content-hash catches the real leakage.

    Returns a dict with method, hits count, and leakage details (must be 0).
    """
    print("  [Gate 3] Building content hashes for held tasks...")
    sys.stdout.flush()
    held_hash_map = {}  # hash -> held task id
    for t in held_tasks:
        h = task_content_hash(t)
        held_hash_map[h] = t['id']

    print(f"  [Gate 3] Checking {len(aug_tasks)} augmented tasks against {len(held_hash_map)} held hashes...")
    sys.stdout.flush()
    leakage_hits = []
    for aug_task in aug_tasks:
        h = task_content_hash(aug_task)
        if h in held_hash_map:
            leakage_hits.append({
                'aug_task_id': aug_task['id'],
                'matched_held_id': held_hash_map[h],
                'aug_type': aug_task.get('aug_type', '?'),
                'source_id': aug_task.get('source_id', '?'),
            })

    result = {
        'method': (
            'content_hash: frozenset of (input_grid_tuples, output_grid_tuples) '
            'per pair, order-invariant. Source IDs excluded held IDs before '
            'augmentation (ID gate). Content gate checks post-augmentation payload.'
        ),
        'n_augmented_checked': len(aug_tasks),
        'n_held_in_check': len(held_tasks),
        'hits': len(leakage_hits),
        'verdict': 'CLEAN' if not leakage_hits else f'LEAKAGE_DETECTED: {len(leakage_hits)} hits',
        'leakage_detail': leakage_hits[:20],
    }
    print(f"  [Gate 3] Content-hash leakage: {result['verdict']}")
    sys.stdout.flush()
    return result


# ── BFS ───────────────────────────────────────────────────────────────────────

def search_simple(task_io, max_length=MAX_LENGTH, budget=BUDGET):
    """Simple BFS (no library, no partials). Returns (program, nodes)."""
    nodes = 0
    for length in range(1, max_length + 1):
        for ops in itertools.product(SORTED_SEED_OPS, repeat=length):
            nodes += 1
            if nodes > budget:
                return None, nodes
            if check_program(list(ops), task_io):
                return list(ops), nodes
    return None, nodes


# ── Holed operators ───────────────────────────────────────────────────────────

class HoledOp:
    def __init__(self, prefix, suffix):
        self.prefix = list(prefix)
        self.suffix = list(suffix)
        p_str = '__'.join(prefix) if prefix else 'eps'
        s_str = '__'.join(suffix) if suffix else 'eps'
        self.name = f"{p_str}___HOLE___{s_str}"

    def mdl_gain(self, count):
        L = len(self.prefix) + 1 + len(self.suffix)
        return count * (L - 2) - L


def extract_holed_ops_with_funnel(programs):
    """Extract holed operators with formation funnel. Returns (mdl_positive_ops, funnel)."""
    pattern_programs = defaultdict(list)
    for prog in programs:
        L = len(prog)
        if L < 2:
            continue
        for hole_pos in range(L):
            prefix = tuple(prog[:hole_pos])
            suffix = tuple(prog[hole_pos + 1:])
            pattern_programs[(prefix, suffix)].append(prog)

    total_pairs_checked = len(pattern_programs)
    candidates_2plus = []
    candidates_mdl_pos = []
    gain_distribution = []
    seen_patterns = set()
    mdl_positive_ops = []

    for (prefix, suffix), progs in pattern_programs.items():
        if (prefix, suffix) in seen_patterns:
            continue
        seen_patterns.add((prefix, suffix))
        hole_ops = set()
        for prog in progs:
            hole_pos = len(prefix)
            if len(prog) == len(prefix) + 1 + len(suffix):
                hole_ops.add(prog[hole_pos])
        if len(hole_ops) < 2:
            continue
        candidates_2plus.append((prefix, suffix))
        count = len(progs)
        hop = HoledOp(prefix, suffix)
        gain = hop.mdl_gain(count)
        gain_distribution.append(gain)
        if gain > 0:
            candidates_mdl_pos.append((prefix, suffix))
            mdl_positive_ops.append({
                'name': hop.name,
                'count': count,
                'gain': gain,
                'hole_variants': sorted(hole_ops),
            })

    mdl_positive_ops.sort(key=lambda x: -x['gain'])

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
    return mdl_positive_ops, funnel


# ── Load tasks ────────────────────────────────────────────────────────────────

def load_all_tasks():
    """Load all tasks from training/ and evaluation/."""
    tasks = []
    counts = {}
    for subdir in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, subdir)
        if not os.path.isdir(d):
            counts[subdir] = 0
            continue
        n = 0
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
                n += 1
        counts[subdir] = n
    return tasks, counts


def solve_source_tasks(source_tasks, label=""):
    """BFS-solve source tasks. Returns solved programs."""
    solved = []
    n_solved = 0
    for task in source_tasks:
        prog, cost = search_simple(task['io'])
        if prog:
            solved.append(prog)
            n_solved += 1
    print(f"  {label}: {n_solved}/{len(source_tasks)} solved -> {len(solved)} programs")
    sys.stdout.flush()
    return solved


# ── Run one density step ───────────────────────────────────────────────────────

def run_density_step(step_name, source_tasks, held_tasks, step_num, source_construction):
    """
    Gate 1: includes held IDs + source IDs in result.
    Gate 2: includes source_construction description.
    Gate 4: full funnel in result.
    """
    # Source IDs for audit (gate 1)
    # For augmented tasks, use their synthetic IDs (they encode source_id + aug_type)
    source_ids = [t['id'] for t in source_tasks]
    held_ids_list = [t['id'] for t in held_tasks]

    # Disjoint check by ID (baseline)
    source_id_set = {t['id'] for t in source_tasks if not t.get('is_augmented', False)}
    held_id_set = set(held_ids_list)
    id_overlap = source_id_set & held_id_set

    print(f"\n{'='*70}")
    print(f"STEP {step_num}: {step_name}")
    print(f"  Source tasks: {len(source_tasks)}")
    print(f"  Held-out tasks: {len(held_tasks)}")
    print(f"  Source construction: {source_construction}")
    print(f"  ID disjoint (base source vs held): {'PASS' if not id_overlap else 'FAIL: ' + str(list(id_overlap)[:5])}")
    print(f"{'='*70}")
    sys.stdout.flush()

    # Solve source tasks
    print("Solving source tasks...")
    programs = solve_source_tasks(source_tasks, label="Source")

    if len(programs) < 2:
        print("  SKIP: fewer than 2 solved programs, formation impossible.")
        return {
            'step': step_num,
            'step_name': step_name,
            'source_construction': source_construction,
            'n_source_tasks': len(source_tasks),
            'n_source_base_tasks': sum(1 for t in source_tasks if not t.get('is_augmented', False)),
            'n_source_aug_tasks': sum(1 for t in source_tasks if t.get('is_augmented', False)),
            'n_programs': len(programs),
            'held_task_ids': held_ids_list,
            'source_base_task_ids': [t['id'] for t in source_tasks if not t.get('is_augmented', False)],
            'id_disjoint': not id_overlap,
            'funnel': {
                'total_pairs_checked': 0,
                'candidates_2plus_variants': 0,
                'candidates_passed_MDL': 0,
                'gain_distribution': {'n': 0, 'min': None, 'max': None, 'mean': None, 'n_positive': 0},
                'diagnosis': 'NO_PROGRAMS',
            },
            'mdl_positive_ops': [],
        }

    # Formation funnel
    print(f"Running formation funnel on {len(programs)} programs...")
    mdl_pos_ops, funnel = extract_holed_ops_with_funnel(programs)

    print(f"  Formation funnel:")
    print(f"    total_pairs_checked: {funnel['total_pairs_checked']}")
    print(f"    candidates_2plus_variants: {funnel['candidates_2plus_variants']}")
    print(f"    candidates_passed_MDL: {funnel['candidates_passed_MDL']}")
    gd = funnel['gain_distribution']
    if gd['n'] > 0:
        print(f"    gain_distribution: n={gd['n']}, min={gd['min']:.1f}, "
              f"max={gd['max']:.1f}, mean={gd['mean']:.1f}, n_positive={gd['n_positive']}")
    print(f"    DIAGNOSIS: {funnel['diagnosis']}")
    for op in mdl_pos_ops[:5]:
        print(f"      {op['name']}: count={op['count']}, gain={op['gain']:.1f}")
    sys.stdout.flush()

    return {
        'step': step_num,
        'step_name': step_name,
        'source_construction': source_construction,
        'n_source_tasks': len(source_tasks),
        'n_source_base_tasks': sum(1 for t in source_tasks if not t.get('is_augmented', False)),
        'n_source_aug_tasks': sum(1 for t in source_tasks if t.get('is_augmented', False)),
        'n_programs': len(programs),
        'held_task_ids': held_ids_list,
        'source_base_task_ids': [t['id'] for t in source_tasks if not t.get('is_augmented', False)],
        'id_disjoint': not id_overlap,
        'funnel': funnel,
        'mdl_positive_ops': [
            {'name': o['name'], 'count': o['count'], 'gain': o['gain'],
             'hole_variants': list(o['hole_variants'])}
            for o in mdl_pos_ops[:20]
        ],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("E3 TRACE-DENSITY AMPLIFICATION CURVE (Leo #11683)")
    print("Kai reviewer gates: 1=IDs-in-JSON, 2=construction-named, "
          "3=content-hash-leakage, 4=full-funnel")
    print(f"Config: budget={BUDGET}, max_length={MAX_LENGTH}, split_seed={SPLIT_SEED}")
    print(f"Held-out: {N_HELD} tasks (fixed, disjoint across all steps)")
    print("=" * 70)
    sys.stdout.flush()

    # Load ALL tasks
    all_tasks, dir_counts = load_all_tasks()
    print(f"\nLoaded: training={dir_counts.get('training', 0)}, "
          f"evaluation={dir_counts.get('evaluation', 0)}, "
          f"total={len(all_tasks)}")

    # Establish fixed held-out from training-only (same as E3 v2 v3 split)
    training_tasks = [t for t in all_tasks if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    indices = rng.permutation(len(training_tasks))
    held_tasks = [training_tasks[i] for i in indices[:N_HELD]]
    held_ids = {t['id'] for t in held_tasks}
    held_ids_list = sorted(t['id'] for t in held_tasks)

    # All tasks MINUS held-out = full source pool (step 2)
    full_source_pool = [t for t in all_tasks if t['id'] not in held_ids]
    training_source_pool = [t for t in training_tasks if t['id'] not in held_ids]

    print(f"Held-out: {len(held_tasks)} tasks (training, split_seed={SPLIT_SEED})")
    print(f"Training-only source pool: {len(training_source_pool)} tasks")
    print(f"Full source pool (step 2, train+eval minus held-out): {len(full_source_pool)} tasks")

    # ID disjoint check for step 2
    full_source_ids = {t['id'] for t in full_source_pool}
    id_overlap = held_ids & full_source_ids
    print(f"ID disjoint check (step 2): {'PASS' if not id_overlap else 'FAIL: ' + str(list(id_overlap)[:5])}")
    print()
    sys.stdout.flush()

    steps = []

    # STEP 1: 200 source from training-only
    step1_source = [training_tasks[i] for i in indices[N_HELD:N_HELD + 200]]
    step1_construction = (
        f"200 training tasks (indices {N_HELD}-{N_HELD+199} of training permutation, "
        f"split_seed={SPLIT_SEED}). Training-only. Disjoint from held-out by construction "
        f"(held = indices 0-{N_HELD-1} of same permutation)."
    )
    r1 = run_density_step(
        "real-200 (training-only, matches E3 v2 v3)",
        step1_source, held_tasks, 1,
        step1_construction,
    )
    steps.append(r1)

    # STEP 2: full real corpus (train+eval minus held-out, ~600 tasks)
    step2_construction = (
        f"ALL tasks from training ({dir_counts.get('training', 0)}) + "
        f"evaluation ({dir_counts.get('evaluation', 0)}) MINUS held-out ({N_HELD} tasks). "
        f"Total = {len(full_source_pool)} tasks. "
        f"Source-density stress test: goes beyond training-only to include eval tasks."
    )
    r2 = run_density_step(
        "full-real (train+eval minus held-out)",
        full_source_pool, held_tasks, 2,
        step2_construction,
    )
    steps.append(r2)

    # STEP 3: step-2 + D4 symmetry augmentation (SOURCE-ONLY)
    print("\nBuilding D4 augmentation for step 3 source tasks...")
    sys.stdout.flush()

    # Gate 3 precondition: source must exclude held BEFORE augmentation
    assert not id_overlap, f"Gate 3 precondition FAILED: full_source_pool overlaps held_ids"

    aug_tasks = augment_source_tasks(full_source_pool)
    step3_source = full_source_pool + aug_tasks
    print(f"  Original: {len(full_source_pool)} | Augmented copies: {len(aug_tasks)} | "
          f"Total step-3 source: {len(step3_source)}")

    # Gate 3 ID check
    aug_ids = {t['id'] for t in aug_tasks}
    aug_id_overlap = aug_ids & held_ids
    print(f"  ID-overlap check (aug vs held): {'CLEAN' if not aug_id_overlap else 'FAIL: ' + str(list(aug_id_overlap)[:5])}")

    # Gate 3 CONTENT-HASH check (the real leakage guard)
    leakage_check = content_hash_leakage_check(aug_tasks, held_tasks)
    if leakage_check['hits'] > 0:
        print(f"  WARNING: Content-hash leakage detected! {leakage_check['hits']} hits.")
        print(f"  First hit: {leakage_check['leakage_detail'][0]}")

    step3_construction = (
        f"Step 2 source ({len(full_source_pool)} tasks) + D4 augmentation "
        f"({len(AUG_TRANSFORMS)} transforms × {len(full_source_pool)} = {len(aug_tasks)} copies). "
        f"Total = {len(step3_source)} tasks. SOURCE-ONLY (held-out unchanged). "
        f"ID-overlap: {'CLEAN' if not aug_id_overlap else 'LEAKAGE'}. "
        f"Content-hash leakage: {leakage_check['verdict']}."
    )

    r3 = run_density_step(
        "+aug (full-real + D4 symmetry augmentation)",
        step3_source, held_tasks, 3,
        step3_construction,
    )
    r3['aug_leakage_check'] = leakage_check
    steps.append(r3)

    # ── CURVE SUMMARY ─────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("DENSITY AMPLIFICATION CURVE (Gate 4 — full table)")
    print("=" * 70)
    print(f"{'Step':<5} {'Source':>10} {'Programs':>10} {'2plus':>8} {'MDL-pos':>9} {'Diagnosis':<25}")
    for r in steps:
        f = r['funnel']
        print(f"  {r['step']:<3} {r['n_source_tasks']:>10} {r['n_programs']:>10} "
              f"{f['candidates_2plus_variants']:>8} {f['candidates_passed_MDL']:>9} "
              f"{f['diagnosis']:<25}")

    # Verdict
    max_mdl_pos = max(r['funnel']['candidates_passed_MDL'] for r in steps)
    if max_mdl_pos > 0:
        verdict = "CURVE_CLIMBS: trace-count insufficiency. Amplification bootstraps formation."
    else:
        verdict = (
            "CURVE_FLAT: base is the ceiling "
            "(12-op programs structure-poor at ALL tested densities). "
            "Code-synthesis pivot (Refinement 2)."
        )
    print(f"\nVERDICT: {verdict}")

    # ── Write result (Gates 1+2+3+4 embedded) ─────────────────────────────────

    curve_table = []
    for r in steps:
        f = r['funnel']
        curve_table.append({
            'step': r['step'],
            'step_name': r['step_name'],
            'n_source_tasks': r['n_source_tasks'],
            'n_programs': r['n_programs'],
            'candidates_2plus_variants': f['candidates_2plus_variants'],
            'candidates_passed_MDL': f['candidates_passed_MDL'],
            'gain_distribution': f['gain_distribution'],
            'diagnosis': f['diagnosis'],
        })

    result = {
        'experiment': 'E3-density-amplification-curve',
        'note': (
            'Formation funnel at 3 source-density steps. '
            'Discriminates trace-count insufficiency vs base insufficiency. '
            'Leo #11683. Kai gates 1-4 wired (#11689).'
        ),
        'config': {
            'budget': BUDGET,
            'max_length': MAX_LENGTH,
            'split_seed': SPLIT_SEED,
            'n_held': N_HELD,
            'aug_transforms': list(AUG_TRANSFORMS.keys()),
        },
        'corpus_metadata': {
            'total_loaded': len(all_tasks),
            'training': dir_counts.get('training', 0),
            'evaluation': dir_counts.get('evaluation', 0),
            'n_held': len(held_tasks),
            'n_full_source_pool_step2': len(full_source_pool),
            'id_disjoint_step2': not id_overlap,
        },
        # Gate 1: held-out task IDs for independent audit
        'held_task_ids': held_ids_list,
        'held_task_count': len(held_tasks),
        'steps': steps,
        'curve_table': curve_table,
        'curve_verdict': verdict,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {RESULT_PATH}")
    print("Gates: 1=held_task_ids+source_base_task_ids present, "
          "2=source_construction named per step, "
          "3=content-hash leakage check embedded in step 3 result, "
          "4=curve_table with full funnel.")


if __name__ == '__main__':
    main()
