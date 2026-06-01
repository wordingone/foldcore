"""
Stage F3 — Self-Modifying Grammar via Partial-Match Chunking

ONE VARIABLE: Grammar self-modification ON vs OFF.
  Baseline grammar: 1370-leaf set (identical to Stage 1k). space_hash=4978b6739bf55beb.
  Modification: depth-2 partial-match sub-programs promoted as new leaves, K=3 per task.
  Cross-task carry: chunks accumulate in _det_seed order. Min-occ=2 across seen tasks.

Hypothesis (R4): Recurring partial-match sub-programs, added as leaves, reduce eval cost
  on already-solved tasks (second-exposure speedup) and expand the reachable depth-5
  space for unsolved tasks.

Decision tree (written BEFORE running):
  Part A gate — median eval-cost reduction > seed-swing on >=6/10 Stage-1k-solved tasks:
    PASS -> carry accumulated grammar to Part B (190 unsolved tasks).
    FAIL -> direction killed; chunking not useful for speedup.
  Part B gate — D-lift >= +2 vs D=10/200 baseline:
    PASS -> self-modification escapes the D=10 plateau.
    FAIL -> direction dead.

PRE-REGISTERED spec:
  https://github.com/wordingone/the-search/issues/22
  Leo GO: mail #12288 (2026-06-01)
  Tracking issue: #22 (wordingone/the-search)

Frozen (ONE-VARIABLE spec check asserted at gate):
  - 1370-leaf grammar (stage1g/1h/1i/1j/1k leaf SET), depth-3 BFS
  - space_hash=4978b6739bf55beb (baseline; chunks added after hash assertion)
  - BUDGET_B=300000, TIME_PER_TASK_D=600.0 (same as Stage 1k)
  - ARM_D_SEED=0, task order: _det_seed(task_id) sort

Usage:
  python stage_f3_chunking.py --seed 42 --part-a          # Part A: 10 solved IDs
  python stage_f3_chunking.py --seed 42 --part-a --smoke   # 3 tasks, TEMP path
  python stage_f3_chunking.py --test-gate                  # gate injection tests
"""

import numpy as np
import json, os, sys, time, collections, hashlib, random, argparse, gc
from collections import deque, Counter

# -- Paths -------------------------------------------------------------------
DATA_PATH   = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
STAGE_DIR   = "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall"
S1D_PATHS   = {
    1:  f"{STAGE_DIR}/stage1d_seed1_b30000_result.json",
    42: f"{STAGE_DIR}/stage1d_premise_truth_b30000_minimality_result.json",
}
S1K_PATHS   = {
    1:  f"{STAGE_DIR}/stage1k_time_wall_seed1_result.json",
    42: f"{STAGE_DIR}/stage1k_time_wall_seed42_result.json",
}
TEMP_PATH   = f"{STAGE_DIR}/stage_f3_chunking_TEMP.json"
RESULT_TMPL = f"{STAGE_DIR}/stage_f3_chunking_seed{{seed}}_part{{part}}_result.json"

# -- Constants ----------------------------------------------------------------
N_HELD             = 200
BUDGET_B           = 300_000       # unchanged from Stage 1k
TIME_PER_TASK_D    = 600.0         # unchanged from Stage 1k
ARM_D_SEED         = 0             # default; matches Stage 1k seed42 canonical run
MAX_DEPTH          = 3             # unchanged
PREV_SPACE_HASH    = "4978b6739bf55beb"
PREV_N_LEAVES      = 1370

# Chunking parameters (pre-registered in issue #22)
K_CHUNKS           = 3             # chunks promoted per task (top-K from occ-sorted list)
MIN_OCC            = 2             # min-occ across seen tasks for promotion
N_CANDIDATES       = 5000          # random depth-2 candidate pairs sampled for partial-match
CANDIDATE_SEED     = 42            # deterministic, fixed for reproducibility

# Stage-1k-solved IDs (seed42 canonical run) — Part A ground truth
STAGE1K_SOLVED_SEED42 = [
    '1cf80156', '3618c87e', '3c9b0459', '6455b5f5', '67a3c6ac',
    '68b16354', '7468f01a', '74dd1130', 'a1570a43', 'a79310a0',
]
# Stage-1k eval counts for seed42 (baseline for eval-cost reduction measurement)
STAGE1K_EVALS_SEED42 = {
    '1cf80156': 493,    '3618c87e': 182349, '3c9b0459': 10442,  '6455b5f5': 243025,
    '67a3c6ac': 1711,   '68b16354': 2304,   '7468f01a': 255922, '74dd1130': 5471,
    'a1570a43': 262881, 'a79310a0': 58177,
}
# Stage-1k eval counts for seed1 (for seed-swing calculation, tasks solved by both)
STAGE1K_EVALS_SEED1 = {
    '67a3c6ac': 414, '7468f01a': 9765, '74dd1130': 1988, 'a1570a43': 129121, 'a79310a0': 25190,
}
# Stage-1k prog_tuples seed42 — used to seed candidate depth-2 sub-programs
STAGE1K_PROG_TUPLES_SEED42 = {
    '68b16354': [1059, 3, 705, 641], '67a3c6ac': [868, 815, 199, 2],
    '3c9b0459': [5, 806, 1287, 806],  '7468f01a': [868, 0, 2, 761],
    '6455b5f5': [1358, 863, 548, 1321], '74dd1130': [529, 8, 989, 464],
    '3618c87e': [559, 61, 158, 886],   '1cf80156': [865, 1151, 574, 0],
    'a79310a0': [1281, 68, 1342, 630], 'a1570a43': [665, 407, 66, 442],
}


# -- Deterministic seed helper (PYTHONHASHSEED-independent) ------------------
def _det_seed(*args):
    return int.from_bytes(
        hashlib.sha256("|".join(str(a) for a in args).encode()).digest()[:8], "big"
    )


# -- Object extraction (verbatim from stage1k) --------------------------------
_OBJ_CACHE = {}

def get_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])

def extract_objects(grid):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if not visited[r, c]:
                color = int(grid[r, c])
                cells = []
                q = deque([(r, c)])
                visited[r, c] = True
                while q:
                    rr, cc = q.popleft()
                    cells.append((rr, cc))
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = rr+dr, cc+dc
                        if (0 <= nr < h and 0 <= nc < w
                                and not visited[nr, nc]
                                and int(grid[nr, nc]) == color):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                objects.append({'color': color, 'cells': cells, 'area': len(cells)})
    return objects

def _extract_cached(grid):
    key = grid.tobytes()
    if key not in _OBJ_CACHE:
        _OBJ_CACHE[key] = extract_objects(grid)
    return _OBJ_CACHE[key]


# -- Predicates (verbatim from stage1k) ---------------------------------------
def _pred_largest(objs):
    if not objs: return []
    mx = max(o['area'] for o in objs)
    return [o for o in objs if o['area'] == mx]

def _pred_smallest(objs):
    if not objs: return []
    mn = min(o['area'] for o in objs)
    return [o for o in objs if o['area'] == mn]

PREDICATES = {
    'largest':       lambda g, objs: _pred_largest(objs),
    'smallest':      lambda g, objs: _pred_smallest(objs),
    'non_bg':        lambda g, objs: [o for o in objs if o['color'] != get_bg(g)],
    'bg_obj':        lambda g, objs: [o for o in objs if o['color'] == get_bg(g)],
    'unique_color':  lambda g, objs: [o for o in objs if collections.Counter(
                         x['color'] for x in objs)[o['color']] == 1],
    'most_common_c': lambda g, objs: ([o for o in objs if o['color'] ==
                         collections.Counter(x['color'] for x in objs).most_common(1)[0][0]]
                         if objs else []),
    'all':           lambda g, objs: objs,
}
for _c in range(1, 10):
    PREDICATES[f'color_{_c}'] = (lambda c: lambda g, objs: [o for o in objs if o['color'] == c])(_c)
PRED_NAMES = sorted(PREDICATES.keys())


# -- Transforms (verbatim from stage1k) ---------------------------------------
def _recolor(new_c):
    def fn(grid, sel):
        g = grid.copy()
        for obj in sel:
            for r, c in obj['cells']:
                g[r, c] = new_c
        return g
    return fn

def _delete(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        for r, c in obj['cells']: g[r, c] = bg
    return g

def _keep_only(grid, sel):
    g = np.full_like(grid, get_bg(grid))
    for obj in sel:
        for r, c in obj['cells']: g[r, c] = obj['color']
    return g

def _translate(dy, dx):
    def fn(grid, sel):
        h, w = grid.shape; g = grid.copy(); bg = get_bg(grid)
        for obj in sel:
            for r, c in obj['cells']: g[r, c] = bg
        for obj in sel:
            for r, c in obj['cells']:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w: g[nr, nc] = obj['color']
        return g
    return fn

def _geom_flip_h(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        cols = [c for r, c in cells]; c_min, c_max = min(cols), max(cols)
        for r, c in cells: g[r, c] = bg
        for r, c in cells:
            nc = c_min + (c_max - c)
            if 0 <= r < g.shape[0] and 0 <= nc < g.shape[1]: g[r, nc] = obj['color']
    return g

def _geom_flip_v(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]; r_min, r_max = min(rows), max(rows)
        for r, c in cells: g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r)
            if 0 <= nr < g.shape[0] and 0 <= c < g.shape[1]: g[nr, c] = obj['color']
    return g

def _geom_rot_90(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]; cols = [c for r, c in cells]
        r_min, c_min = min(rows), min(cols); c_max = max(cols); w = c_max - c_min + 1
        for r, c in cells: g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min; c_rel = c - c_min
            nr = r_min + (w - 1 - c_rel); nc = c_min + r_rel
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]: g[nr, nc] = obj['color']
    return g

def _geom_rot_180(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]; cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows); c_min, c_max = min(cols), max(cols)
        for r, c in cells: g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r); nc = c_min + (c_max - c)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]: g[nr, nc] = obj['color']
    return g

def _geom_rot_270(grid, sel):
    g = grid.copy(); bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]; cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows); c_min = min(cols); h = r_max - r_min + 1
        for r, c in cells: g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min; c_rel = c - c_min
            nr = r_min + c_rel; nc = c_min + (h - 1 - r_rel)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]: g[nr, nc] = obj['color']
    return g

RELATE_MODES = ['adjacent_down', 'adjacent_left', 'adjacent_right', 'adjacent_up']

def _largest_cc(selected_objects):
    if not selected_objects: return None
    return sorted(selected_objects,
                  key=lambda o: (-o['area'],
                                 min(r for r, c in o['cells']),
                                 min(c for r, c in o['cells'])))[0]

def _map_relate(grid, src_sel, anchor_sel, mode):
    anchor_obj = _largest_cc(anchor_sel)
    if anchor_obj is None: return grid.copy()
    anc_cells = anchor_obj['cells']
    anc_r0 = min(r for r, c in anc_cells); anc_r1 = max(r for r, c in anc_cells)
    anc_c0 = min(c for r, c in anc_cells); anc_c1 = max(c for r, c in anc_cells)
    h, w = grid.shape; g = grid.copy(); bg = get_bg(grid)
    for obj in src_sel:
        for r, c in obj['cells']: g[r, c] = bg
    for obj in src_sel:
        cells = obj['cells']
        if not cells: continue
        obj_r0 = min(r for r, c in cells); obj_r1 = max(r for r, c in cells)
        obj_c0 = min(c for r, c in cells); obj_c1 = max(c for r, c in cells)
        if mode == 'adjacent_left':
            dr = anc_r0 - obj_r0; dc = (anc_c0 - 1) - obj_c1
        elif mode == 'adjacent_right':
            dr = anc_r0 - obj_r0; dc = (anc_c1 + 1) - obj_c0
        elif mode == 'adjacent_up':
            dr = (anc_r0 - 1) - obj_r1; dc = anc_c0 - obj_c0
        elif mode == 'adjacent_down':
            dr = (anc_r1 + 1) - obj_r0; dc = anc_c0 - obj_c0
        else:
            dr, dc = 0, 0
        for r, c in cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w: g[nr, nc] = obj['color']
    return g

TRANSFORMS = {
    'delete': _delete, 'keep_only': _keep_only,
    'geom_flip_h': _geom_flip_h, 'geom_flip_v': _geom_flip_v,
    'geom_rot_90': _geom_rot_90, 'geom_rot_180': _geom_rot_180, 'geom_rot_270': _geom_rot_270,
}
for _c in range(10):
    TRANSFORMS[f'recolor_{_c}'] = (lambda c: lambda g, sel: _recolor(c)(g, sel))(_c)
for _dy, _dx in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
    TRANSFORMS[f'translate_{_dy:+d}_{_dx:+d}'] = (
        lambda dy, dx: lambda g, sel: _translate(dy, dx)(g, sel))(_dy, _dx)
TRANSFORM_NAMES = sorted(TRANSFORMS.keys())

def _crop_bg(g):
    bg = get_bg(g); nz = np.argwhere(g != bg)
    if len(nz) == 0: return g
    return g[nz[:,0].min():nz[:,0].max()+1, nz[:,1].min():nz[:,1].max()+1]

GRID_PRIMS = {
    'flip_h': lambda g: g[:, ::-1], 'flip_v': lambda g: g[::-1],
    'rot_90': lambda g: np.rot90(g, 1), 'rot_180': lambda g: np.rot90(g, 2),
    'rot_270': lambda g: np.rot90(g, 3), 'tr': lambda g: g.T,
    'up2': lambda g: np.repeat(np.repeat(g, 2, 0), 2, 1),
    'down2': lambda g: g[::2, ::2], 'id': lambda g: g, 'crop': _crop_bg,
}
PRIM_NAMES = sorted(GRID_PRIMS.keys())

_RECOLOR_TRANSFORMS   = sorted(t for t in TRANSFORM_NAMES if t.startswith('recolor_'))
_TRANSLATE_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith('translate_'))
_DELETE_TRANSFORMS    = ['delete']
_KEEPONLY_TRANSFORMS  = ['keep_only']
_GEOM_TRANSFORMS      = sorted(t for t in TRANSFORM_NAMES if t.startswith('geom_'))

MAP_TRANSFORM_FAMILIES = {
    'MAP_RECOLOR': _RECOLOR_TRANSFORMS, 'MAP_DELETE': _DELETE_TRANSFORMS,
    'MAP_KEEPONLY': _KEEPONLY_TRANSFORMS, 'MAP_TRANSLATE': _TRANSLATE_TRANSFORMS,
    'MAP_GEOM': _GEOM_TRANSFORMS,
}
PRIM_SKELETON_NAMES = [f'PRIM_{n.upper()}' for n in PRIM_NAMES]
SKELETON_ORDER = (
    PRIM_SKELETON_NAMES +
    ['MAP_DELETE', 'MAP_KEEPONLY', 'MAP_TRANSLATE', 'MAP_GEOM', 'MAP_RELATE', 'MAP_RECOLOR']
)

def skeleton_fills(sk_name):
    if sk_name.startswith('PRIM_'):
        prim_n = sk_name[5:].lower()
        return [('prim', prim_n)] if prim_n in GRID_PRIMS else []
    if sk_name == 'MAP_RELATE':
        return [
            ('map_relate', src_pred, anchor_pred, mode)
            for src_pred in PRED_NAMES
            for anchor_pred in PRED_NAMES
            for mode in RELATE_MODES
            if src_pred != anchor_pred
        ]
    if sk_name in MAP_TRANSFORM_FAMILIES:
        trs = MAP_TRANSFORM_FAMILIES[sk_name]
        return [('map_apply', pred, tr) for pred in PRED_NAMES for tr in trs]
    return []

def all_leaves_expanded():
    result = []
    for sk in SKELETON_ORDER:
        result.extend(skeleton_fills(sk))
    return result

def eval_program(prog, grid, lib={}):
    try:
        return _eval(prog, np.array(grid, dtype=np.int64), lib)
    except Exception:
        return None

def _eval(prog, g, lib):
    t = prog[0]
    if t == 'prim':   return GRID_PRIMS[prog[1]](g)
    if t == 'map_apply':
        try:
            objs = _extract_cached(g)
            selected = PREDICATES[prog[1]](g, objs)
            if not selected: return g.copy()
            return TRANSFORMS[prog[2]](g, selected)
        except Exception: return None
    if t == 'map_relate':
        try:
            src_pred_name, anchor_pred_name, mode = prog[1], prog[2], prog[3]
            objs = _extract_cached(g)
            src_sel    = PREDICATES[src_pred_name](g, objs)
            anchor_sel = PREDICATES[anchor_pred_name](g, objs)
            if not src_sel or not anchor_sel: return g.copy()
            return _map_relate(g, src_sel, anchor_sel, mode)
        except Exception: return None
    if t == 'compose':
        r = _eval(prog[2], g, lib)
        if r is None: return None
        return _eval(prog[1], r, lib)
    return None

def check_prog(prog, task_io):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        r = eval_program(prog, inp)
        if r is None or r.shape != out.shape or not np.array_equal(r, out): return False
    return True

def tuple_to_prog(tup, leaves):
    if len(tup) == 1: return leaves[tup[0]]
    return ('compose', leaves[tup[0]], tuple_to_prog(tup[1:], leaves))

def check_tuple(tup, task_io, leaves):
    return check_prog(tuple_to_prog(tup, leaves), task_io)

def compute_space_hash(leaves):
    leaf_strs = [str(l) for l in leaves]
    space_descriptor = json.dumps({
        'n_leaves': len(leaves),
        'max_depth': MAX_DEPTH,
        'canonical_form': 'left_linear',
        'leaf_ordering': 'SKELETON_ORDER',
        'leaf_repr_sample': leaf_strs[:5] + ['...'] + leaf_strs[-3:],
    }, sort_keys=True)
    return hashlib.sha256(space_descriptor.encode()).hexdigest()[:16]

def prog_hash(prog):
    return hashlib.sha256(str(prog).encode()).hexdigest()[:12]


# -- Chunking mechanism -------------------------------------------------------

def partial_match_score(prog, task_io):
    """Max cell-agreement across training pairs (0=no match, 1=full solve).
    Returns 0 if program errors or shape mismatches on all pairs.
    """
    best = 0.0
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        result = eval_program(prog, inp)
        if result is None or result.shape != out.shape:
            continue
        agreement = float(np.sum(result == out)) / out.size
        if agreement > best:
            best = agreement
    return best

def build_candidates(baseline_leaves, rng_seed=CANDIDATE_SEED, n_random=N_CANDIDATES):
    """Build the fixed candidate set of depth-2 programs for partial-match scanning.

    Candidates = random pairs of baseline leaf indices + known stage1k depth-2 sub-programs.
    The random sample is fixed (deterministic) so formation counts are reproducible.
    """
    n = len(baseline_leaves)
    rng = random.Random(rng_seed)
    # Random pairs: (i, j) -> compose(leaves[i], leaves[j])
    random_pairs = [(rng.randrange(n), rng.randrange(n)) for _ in range(n_random)]

    # Known depth-2 sub-programs from stage1k solutions: innermost pair (k, l) from each tuple
    known_pairs = []
    for tid, tup in STAGE1K_PROG_TUPLES_SEED42.items():
        known_pairs.append((tup[2], tup[3]))  # innermost composition

    # Deduplicate while preserving order: random first, then known
    seen = set()
    candidates = []
    for pair in random_pairs + known_pairs:
        if pair not in seen:
            seen.add(pair)
            candidates.append(pair)
    return candidates  # list of (i, j) index pairs

def scan_partial_matches(task_io, candidates, baseline_leaves):
    """Return dict: candidate_pair -> partial_match_score for this task.
    Only includes pairs with score > 0 (some partial match).
    """
    hits = {}
    for (i, j) in candidates:
        prog = ('compose', baseline_leaves[i], baseline_leaves[j])
        score = partial_match_score(prog, task_io)
        if score > 0.0:
            hits[(i, j)] = score
    return hits

def update_chunks(occ_counter, seen_task_partial_matches, candidates, baseline_leaves,
                  existing_chunk_pairs):
    """Update the occurrence counter and return the chunk set after promotion.

    occ_counter: Counter mapping (i,j) pair -> # tasks where it partial-matched
    Returns: updated existing_chunk_pairs (set of (i,j) pairs in chunk set)
    """
    # occ_counter is updated externally (caller increments for this task's matches)
    # Promote top-K candidates with occ >= MIN_OCC not already in chunk set
    eligible = [
        (pair, occ_counter[pair])
        for pair in occ_counter
        if occ_counter[pair] >= MIN_OCC and pair not in existing_chunk_pairs
    ]
    # Sort by occ descending, then by pair for determinism
    eligible.sort(key=lambda x: (-x[1], x[0]))
    new_promotions = [pair for pair, _ in eligible[:K_CHUNKS]]
    new_chunk_pairs = existing_chunk_pairs | set(new_promotions)
    return new_chunk_pairs, new_promotions

def chunks_to_leaves(chunk_pairs, baseline_leaves):
    """Convert (i,j) pair set to list of program leaves."""
    return [('compose', baseline_leaves[i], baseline_leaves[j]) for (i, j) in sorted(chunk_pairs)]


# -- Fail-closed gate ---------------------------------------------------------

def check_gate_violations(artifact, stage1d_held_id_set, actual_n_baseline_leaves,
                           baseline_space_hash):
    """Returns list of violation strings. Empty = gate passes."""
    violations = []

    # ONE-VARIABLE: baseline grammar must match stage1k exactly
    if artifact.get('n_baseline_leaves') != actual_n_baseline_leaves:
        violations.append(
            f"n_baseline_leaves={artifact.get('n_baseline_leaves')} != "
            f"actual {actual_n_baseline_leaves}")
    if artifact.get('baseline_space_hash') != PREV_SPACE_HASH:
        violations.append(
            f"baseline_space_hash={artifact.get('baseline_space_hash')} != "
            f"{PREV_SPACE_HASH} (baseline must match Stage 1k)")
    if artifact.get('prev_space_hash') != PREV_SPACE_HASH:
        violations.append(f"prev_space_hash mismatch")

    # Chunking parameters must match pre-reg
    if artifact.get('k_chunks') != K_CHUNKS:
        violations.append(f"k_chunks={artifact.get('k_chunks')} != {K_CHUNKS}")
    if artifact.get('min_occ') != MIN_OCC:
        violations.append(f"min_occ={artifact.get('min_occ')} != {MIN_OCC}")

    # Budget and time wall: unchanged from Stage 1k
    if artifact.get('budget_b') != BUDGET_B:
        violations.append(f"budget_b={artifact.get('budget_b')} != {BUDGET_B}")
    if artifact.get('time_limit_s') != TIME_PER_TASK_D:
        violations.append(f"time_limit_s={artifact.get('time_limit_s')} != {TIME_PER_TASK_D}")

    # Part A: task IDs must match stage1k solved set
    part = artifact.get('part')
    if part == 'A':
        artifact_task_ids = set(artifact.get('task_ids', []))
        expected_set = set(STAGE1K_SOLVED_SEED42)
        if artifact_task_ids != expected_set:
            extra   = artifact_task_ids - expected_set
            missing = expected_set - artifact_task_ids
            violations.append(
                f"Part A task_ids mismatch: extra={extra} missing={missing}")
        if artifact.get('n_tasks') != len(STAGE1K_SOLVED_SEED42):
            violations.append(
                f"Part A n_tasks={artifact.get('n_tasks')} != {len(STAGE1K_SOLVED_SEED42)}")

    # Part B: task IDs must match full held set
    elif part == 'B':
        artifact_held_ids = artifact.get('task_ids', [])
        if len(artifact_held_ids) != N_HELD:
            violations.append(
                f"Part B task_ids count={len(artifact_held_ids)} != {N_HELD}")
        if set(artifact_held_ids) != stage1d_held_id_set:
            violations.append("Part B task_ids set mismatch vs stage1d")

    # Per-task: no budget overruns
    for tid, rec in artifact.get('per_task', {}).items():
        if rec.get('n_evals', 0) > BUDGET_B:
            violations.append(f"per_task[{tid}]: n_evals={rec.get('n_evals')} > BUDGET_B")
            break

    # Baseline space_hash: must be PREV_SPACE_HASH (computed at runtime on baseline leaves)
    if baseline_space_hash != PREV_SPACE_HASH:
        violations.append(
            f"RUNTIME space_hash={baseline_space_hash} != {PREV_SPACE_HASH} -- "
            "baseline grammar changed (ONE-VARIABLE violation)")

    return violations

def check_gate_advisory(artifact):
    advisories = []
    n_formation_zero = sum(
        1 for rec in artifact.get('per_task', {}).values()
        if rec.get('formation_count', 0) == 0
    )
    if n_formation_zero == artifact.get('n_tasks', 0):
        advisories.append("ALL tasks have formation_count=0: partial-match scanning found nothing")
    n_promoted_zero = sum(
        1 for rec in artifact.get('per_task', {}).values()
        if rec.get('promoted_count', 0) == 0
    )
    if n_promoted_zero == artifact.get('n_tasks', 0):
        advisories.append("ALL tasks have promoted_count=0: no chunks ever reached MIN_OCC")
    return advisories


# -- Gate violation injection tests -------------------------------------------

def run_gate_tests(actual_n_baseline_leaves, stage1d_held_id_set, baseline_space_hash):
    dummy_per_task = {tid: {'n_evals': BUDGET_B, 'solved': False,
                            'formation_count': 0, 'promoted_count': 0}
                      for tid in STAGE1K_SOLVED_SEED42}
    good = {
        'part': 'A', 'n_tasks': 10,
        'task_ids': STAGE1K_SOLVED_SEED42,
        'n_baseline_leaves': actual_n_baseline_leaves,
        'baseline_space_hash': PREV_SPACE_HASH,
        'prev_space_hash': PREV_SPACE_HASH,
        'k_chunks': K_CHUNKS, 'min_occ': MIN_OCC,
        'budget_b': BUDGET_B, 'time_limit_s': TIME_PER_TASK_D,
        'per_task': dummy_per_task,
    }
    base_viols = check_gate_violations(good, stage1d_held_id_set,
                                       actual_n_baseline_leaves, baseline_space_hash)
    if base_viols:
        print(f"  SMOKE GATE TEST FAIL: good artifact has violations: {base_viols}")
        return False
    print("  [OK] Good artifact passes gate")

    tests = [
        ("baseline_space_hash wrong", {**good, 'baseline_space_hash': 'badhash12345678'}),
        ("budget_b wrong (Stage 1i value 30000)", {**good, 'budget_b': 30000}),
        ("time_limit_s wrong (Stage 1j value 120.0)", {**good, 'time_limit_s': 120.0}),
        ("k_chunks wrong (1)", {**good, 'k_chunks': 1}),
        ("min_occ wrong (1)", {**good, 'min_occ': 1}),
        ("n_baseline_leaves wrong (410)", {**good, 'n_baseline_leaves': 410}),
        ("Part A task_ids wrong (missing one ID)",
         {**good, 'task_ids': STAGE1K_SOLVED_SEED42[:-1]}),
        ("per_task budget overrun",
         {**good, 'per_task': {**dummy_per_task,
             STAGE1K_SOLVED_SEED42[0]: {'n_evals': BUDGET_B + 1, 'solved': False,
                                        'formation_count': 0, 'promoted_count': 0}}}),
    ]

    all_caught = True
    for label, bad_artifact in tests:
        viols = check_gate_violations(bad_artifact, stage1d_held_id_set,
                                      actual_n_baseline_leaves, baseline_space_hash)
        if viols:
            print(f"  [OK] Gate caught '{label}': {viols[0][:80]}")
        else:
            print(f"  FAIL: Gate MISSED '{label}'")
            all_caught = False
    return all_caught


# -- Part A metric computation ------------------------------------------------

def compute_part_a_metrics(per_task):
    """Compute eval-cost reduction vs stage1k baseline and seed-swing estimate."""
    # Eval-cost reduction: (baseline - enhanced) / baseline
    reductions = {}
    for tid in STAGE1K_SOLVED_SEED42:
        baseline = STAGE1K_EVALS_SEED42[tid]
        enhanced = per_task[tid].get('n_evals', BUDGET_B)
        if baseline > 0:
            reductions[tid] = (baseline - enhanced) / baseline

    # Seed-swing: median |log(seed42/seed1)| for tasks solved by both seeds
    import math
    swings = []
    for tid, seed1_evals in STAGE1K_EVALS_SEED1.items():
        seed42_evals = STAGE1K_EVALS_SEED42[tid]
        swings.append(abs(math.log(seed42_evals / seed1_evals)))
    swings.sort()
    seed_swing_log = swings[len(swings) // 2] if swings else 0.0
    # Convert to fractional reduction: 1 - 1/e^swing
    seed_swing_frac = 1.0 - (1.0 / (2.718281828 ** seed_swing_log)) if seed_swing_log > 0 else 0.0

    sorted_reductions = sorted(reductions.values())
    median_reduction = sorted_reductions[len(sorted_reductions) // 2] if sorted_reductions else 0.0

    tasks_exceeding_swing = sum(1 for r in reductions.values() if r > seed_swing_frac)
    part_a_gate_pass = (median_reduction > seed_swing_frac and
                        tasks_exceeding_swing >= 6)

    return {
        'reductions_per_task': reductions,
        'median_reduction': median_reduction,
        'seed_swing_log': seed_swing_log,
        'seed_swing_frac': seed_swing_frac,
        'tasks_exceeding_swing': tasks_exceeding_swing,
        'part_a_gate_pass': part_a_gate_pass,
        'verdict': 'PASS' if part_a_gate_pass else 'FAIL',
    }


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[1, 42], default=42)
    parser.add_argument('--part-a', action='store_true', help='Run Part A (10 solved IDs)')
    parser.add_argument('--part-b', action='store_true', help='Run Part B (190 unsolved + carry grammar)')
    parser.add_argument('--smoke', action='store_true', help='3 tasks only, TEMP path')
    parser.add_argument('--test-gate', action='store_true', help='Inject gate violations, check caught')
    args = parser.parse_args()

    if not args.part_a and not args.part_b and not args.test_gate:
        parser.error("Specify --part-a, --part-b, or --test-gate")

    # Load all tasks
    all_tasks = []
    for split in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, split)
        if not os.path.exists(d): continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                with open(os.path.join(d, fn)) as f:
                    dat = json.load(f)
                task_id = fn[:-5]
                all_tasks.append({'id': task_id, 'io': dat.get('train', [])})

    # Load stage1d held IDs (for Part B gate)
    s1d_path = S1D_PATHS[args.seed]
    with open(s1d_path) as f:
        s1d_result = json.load(f)
    stage1d_held_id_set = set(s1d_result['held_task_ids'])

    # Build and verify baseline grammar
    baseline_leaves = all_leaves_expanded()
    actual_n_baseline_leaves = len(baseline_leaves)
    baseline_space_hash = compute_space_hash(baseline_leaves)

    print(f"\nBaseline grammar: {actual_n_baseline_leaves} leaves (Stage 1k leaf set)")
    print(f"  baseline_space_hash = {baseline_space_hash}")
    print(f"  prev_space_hash (Stage 1k) = {PREV_SPACE_HASH}")
    assert baseline_space_hash == PREV_SPACE_HASH, (
        f"FATAL: baseline_space_hash {baseline_space_hash} != {PREV_SPACE_HASH} "
        "-- grammar or depth changed (ONE-VARIABLE violation)")
    print(f"  MATCH: baseline grammar unchanged from Stage 1k")
    print(f"\nChunking params (pre-reg #22): K={K_CHUNKS}, min_occ={MIN_OCC}, "
          f"n_candidates={N_CANDIDATES}")

    # Gate tests (always run if --test-gate, also on --smoke)
    if args.test_gate or args.smoke:
        print("\n=== Gate violation injection tests ===")
        all_caught = run_gate_tests(actual_n_baseline_leaves, stage1d_held_id_set,
                                    baseline_space_hash)
        if not all_caught:
            print("GATE TESTS FAILED -- fix before canonical run")
            sys.exit(1)
        print("All gate tests passed\n")
        if args.test_gate and not args.part_a and not args.part_b:
            return

    # Build candidate depth-2 program pairs (fixed, deterministic)
    candidates = build_candidates(baseline_leaves)
    print(f"\nCandidate depth-2 programs: {len(candidates)} "
          f"({N_CANDIDATES} random + {len(candidates)-N_CANDIDATES} known sub-programs)")

    if args.part_a:
        run_part("A", args, baseline_leaves, baseline_space_hash, stage1d_held_id_set,
                 candidates, all_tasks, chunk_carry=None)

    if args.part_b:
        # Part B requires Part A to have completed to get the carry grammar
        part_a_result_path = RESULT_TMPL.format(seed=args.seed, part='A')
        if not os.path.exists(part_a_result_path):
            print(f"FATAL: Part A result not found at {part_a_result_path}")
            print("  Run --part-a first. Part B requires the accumulated grammar from Part A.")
            sys.exit(1)
        with open(part_a_result_path) as f:
            part_a_result = json.load(f)
        if not part_a_result.get('part_a_metrics', {}).get('part_a_gate_pass', False):
            print("FATAL: Part A gate FAILED -- Part B is not authorized per pre-reg #22")
            sys.exit(1)
        # Recover carry grammar from Part A
        chunk_pairs_list = part_a_result.get('final_chunk_pairs', [])
        chunk_carry = set(tuple(p) for p in chunk_pairs_list)
        print(f"\nCarry grammar from Part A: {len(chunk_carry)} chunks accumulated")
        run_part("B", args, baseline_leaves, baseline_space_hash, stage1d_held_id_set,
                 candidates, all_tasks, chunk_carry=chunk_carry)


def run_part(part, args, baseline_leaves, baseline_space_hash, stage1d_held_id_set,
             candidates, all_tasks, chunk_carry):
    """Execute Part A or Part B search with chunk accumulation."""
    if part == 'A':
        # 10 Stage-1k-solved IDs, sorted by _det_seed(task_id)
        part_task_ids = sorted(STAGE1K_SOLVED_SEED42, key=lambda tid: _det_seed(tid))
        part_label = "Part A (10 Stage-1k-solved IDs)"
        result_path = TEMP_PATH if args.smoke else RESULT_TMPL.format(seed=args.seed, part='A')
        n_tasks = 3 if args.smoke else len(part_task_ids)
    else:
        # Part B: ALL 200 held tasks in _det_seed order, excluding stage1k solved IDs
        all_held = [t for t in all_tasks if t['id'] in stage1d_held_id_set]
        assert len(all_held) == N_HELD, f"Expected {N_HELD} held tasks, got {len(all_held)}"
        # Exclude the 10 Part-A tasks (carry grammar covers those)
        part_task_ids_set = set(STAGE1K_SOLVED_SEED42)
        part_b_ids = sorted(
            [t['id'] for t in all_held if t['id'] not in part_task_ids_set],
            key=lambda tid: _det_seed(tid)
        )
        part_task_ids = part_b_ids
        part_label = "Part B (190 unsolved held tasks)"
        result_path = TEMP_PATH if args.smoke else RESULT_TMPL.format(seed=args.seed, part='B')
        n_tasks = 3 if args.smoke else len(part_task_ids)

    task_id_to_task = {t['id']: t for t in all_tasks}
    run_tasks = [task_id_to_task[tid] for tid in part_task_ids[:n_tasks] if tid in task_id_to_task]

    print(f"\n{'SMOKE MODE' if args.smoke else 'CANONICAL MODE'}: {part_label}")
    print(f"  {len(run_tasks)} tasks, budget={BUDGET_B}, time_wall={TIME_PER_TASK_D:.0f}s")
    print(f"  result -> {result_path}")

    # Chunk state
    chunk_pairs = set(chunk_carry) if chunk_carry else set()
    occ_counter = Counter()  # (i,j) -> # tasks where partial-matched

    per_task = {}
    t_total = time.time()

    for task_num, task in enumerate(run_tasks):
        tid = task['id']
        # Grammar for this task: baseline + accumulated chunks
        chunk_leaves = chunks_to_leaves(chunk_pairs, baseline_leaves)
        extended_leaves = baseline_leaves + chunk_leaves
        n_extended = len(extended_leaves)

        # Scan partial matches for this task (against baseline candidates)
        t_scan = time.time()
        task_hits = scan_partial_matches(task['io'], candidates, baseline_leaves)
        scan_ms = (time.time() - t_scan) * 1000

        # Update occurrence counter
        for pair in task_hits:
            occ_counter[pair] += 1

        formation_count = len(task_hits)

        # Promote new chunks (candidates with occ >= MIN_OCC, not yet in chunk set)
        chunk_pairs, new_promotions = update_chunks(
            occ_counter, task_hits, candidates, baseline_leaves, chunk_pairs)
        promoted_count = len(new_promotions)

        # Search with extended grammar
        rng = random.Random(_det_seed(ARM_D_SEED, tid))
        t_task = time.time()
        n_evals = 0; solved = False; budget_exhausted = False
        used_chunk = False

        while time.time() - t_task < TIME_PER_TASK_D:
            i = rng.randrange(n_extended)
            j = rng.randrange(n_extended)
            k = rng.randrange(n_extended)
            l = rng.randrange(n_extended)
            n_evals += 1
            tup = (i, j, k, l)
            if check_tuple(tup, task['io'], extended_leaves):
                prog = tuple_to_prog(tup, extended_leaves)
                used_chunk = any(idx >= len(baseline_leaves) for idx in tup)
                per_task[tid] = {
                    'solved': True, 'n_evals': n_evals,
                    'time_limit_hit': False, 'budget_exhausted': False,
                    'prog_hash': prog_hash(prog), 'used_chunk': used_chunk,
                    'chunks_available': len(chunk_leaves),
                    'formation_count': formation_count,
                    'promoted_count': promoted_count,
                    'scan_ms': round(scan_ms, 1),
                    'baseline_evals': STAGE1K_EVALS_SEED42.get(tid),
                }
                solved = True
                break
            if n_evals >= BUDGET_B:
                budget_exhausted = True
                break

        if not solved:
            time_hit = (time.time() - t_task) >= TIME_PER_TASK_D
            per_task[tid] = {
                'solved': False, 'n_evals': n_evals,
                'time_limit_hit': time_hit, 'budget_exhausted': budget_exhausted,
                'chunks_available': len(chunk_leaves),
                'formation_count': formation_count,
                'promoted_count': promoted_count,
                'scan_ms': round(scan_ms, 1),
                'baseline_evals': STAGE1K_EVALS_SEED42.get(tid),
            }

        _OBJ_CACHE.clear(); gc.collect()

        # Progress
        elapsed = time.time() - t_total
        n_solved_so_far = sum(1 for r in per_task.values() if r.get('solved'))
        baseline_ev = STAGE1K_EVALS_SEED42.get(tid, '?')
        print(f"  [{task_num+1}/{len(run_tasks)}] {tid}: "
              f"evals={n_evals} solved={solved} chunks_avail={len(chunk_leaves)} "
              f"formation={formation_count} promoted={promoted_count} "
              f"baseline={baseline_ev} scan={scan_ms:.0f}ms elapsed={elapsed:.0f}s")

    # Aggregate stats
    elapsed_total = time.time() - t_total
    n_solved = sum(1 for r in per_task.values() if r.get('solved'))
    total_formation = sum(r.get('formation_count', 0) for r in per_task.values())
    total_promoted  = sum(r.get('promoted_count', 0) for r in per_task.values())

    print(f"\n{part_label} done: {n_solved}/{len(run_tasks)} solved in {elapsed_total:.0f}s")
    print(f"  Total partial-match programs found: {total_formation}")
    print(f"  Total chunks promoted: {total_promoted}")
    print(f"  Final chunk set size: {len(chunk_pairs)}")

    # Part A: compute eval-cost reduction metric
    part_a_metrics = None
    if part == 'A' and not args.smoke:
        part_a_metrics = compute_part_a_metrics(per_task)
        print(f"\nPart A metrics:")
        print(f"  Median eval-cost reduction: {part_a_metrics['median_reduction']:.3f}")
        print(f"  Seed-swing (log): {part_a_metrics['seed_swing_log']:.3f} "
              f"(frac: {part_a_metrics['seed_swing_frac']:.3f})")
        print(f"  Tasks exceeding swing: {part_a_metrics['tasks_exceeding_swing']}/10")
        print(f"  Part A gate: {part_a_metrics['verdict']}")
        for tid in sorted(STAGE1K_SOLVED_SEED42, key=lambda t: _det_seed(t)):
            r = part_a_metrics['reductions_per_task'].get(tid, float('nan'))
            print(f"    {tid}: reduction={r:.3f}")

    # Full declared task ID set for gate (smoke uses subset but declares full set)
    if part == 'A':
        full_task_ids = sorted(STAGE1K_SOLVED_SEED42, key=lambda tid: _det_seed(tid))
    else:
        full_task_ids = [t['id'] for t in run_tasks]  # Part B runs the full list

    artifact = {
        'experiment': 'stage_f3_chunking',
        'spec': 'https://github.com/wordingone/the-search/issues/22',
        'part': part,
        'smoke': args.smoke,
        'seed': args.seed,
        'n_tasks': len(full_task_ids),       # declared (full set, even in smoke mode)
        'n_tasks_run': len(run_tasks),        # actual tasks run (3 in smoke)
        'n_solved': n_solved,
        # Grammar provenance
        'n_baseline_leaves': len(baseline_leaves),
        'baseline_space_hash': baseline_space_hash,
        'prev_space_hash': PREV_SPACE_HASH,
        # Chunking params (pre-registered)
        'k_chunks': K_CHUNKS,
        'min_occ': MIN_OCC,
        'n_candidates': len(candidates),
        'candidate_seed': CANDIDATE_SEED,
        # Search params (unchanged from Stage 1k)
        'budget_b': BUDGET_B,
        'time_limit_s': TIME_PER_TASK_D,
        'arm_d_seed': ARM_D_SEED,
        'task_order': '_det_seed(task_id)',
        # Results
        'task_ids': full_task_ids,            # full declared set
        'task_ids_run': [t['id'] for t in run_tasks],  # actually run
        'per_task': per_task,
        'total_formation_count': total_formation,
        'total_promoted_count': total_promoted,
        'final_chunk_pairs': [list(p) for p in sorted(chunk_pairs)],
        'final_chunk_count': len(chunk_pairs),
        'elapsed_s': elapsed_total,
    }
    if part_a_metrics:
        artifact['part_a_metrics'] = part_a_metrics

    # Fail-closed gate
    advisories = check_gate_advisory(artifact)
    for adv in advisories:
        artifact.setdefault('gate_advisories', []).append(adv)
        print(f"  ADVISORY: {adv}")

    violations = check_gate_violations(artifact, stage1d_held_id_set,
                                       len(baseline_leaves), baseline_space_hash)
    if violations:
        print("\n=== FAIL-CLOSED PRE-WRITE GATE: VIOLATIONS -- ARTIFACT NOT WRITTEN ===")
        for v in violations:
            print(f"  VIOLATION: {v}")
        sys.exit(1)
    print("\nFail-closed gate: PASSED")

    with open(result_path, 'w') as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact written: {result_path}")


if __name__ == '__main__':
    main()
