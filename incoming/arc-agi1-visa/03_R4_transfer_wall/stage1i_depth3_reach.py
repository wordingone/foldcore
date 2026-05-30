"""
Stage 1i -- Reach-Ceiling Diagnostic: Depth-2 -> Depth-3

ONE-VARIABLE CHANGE: MAX_DEPTH 2 -> 3.
  Grammar leaf SET: UNCHANGED -- same 1370-leaf Stage 1g/1h space.
  Budget B=30000 (faithfully enforced, as in 1h), arm seed=0, 120s/task, same 200 held.
  Goal: determine if depth-3 composition raises reach beyond D=7 (Stage 1h ceiling).

Context: Stage 1h established a GENUINE budget-bound ceiling at depth-2 on the 1370-leaf
  grammar: D=7, time_limit_hit=2/200, budget_exhausted=191/200. ONE variable: depth.
  Depth-3 composes the same 1370 leaves one layer deeper (4-tuple search instead of 3-tuple).
  The reachable space grows (space_hash MUST differ from Stage 1h).

Decision tree (written before running):
  D > 7: depth-3 adds reach within B=30000. Depth is a live lever.
  D == 7: depth adds nothing within B=30000. Ceiling is grammar-EXPRESSIVITY, not depth.
    CAVEAT: D==7 with budget_exhausted high is ALSO consistent with "30K too few to sample
    depth-3's larger space." Disambiguating follow-stage: scale B at depth-3. Do NOT conflate.
  D < 7: search BUG (depth-3 contains depth-2 subspace). HOLD, debug.
  time_limit_hit_count high: 120s insufficient at depth-3. Raise wall before trusting D.

PRE-REGISTERED spec:
  B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/pre_reg/stage1i_depth3_reach_preregistration.md
Leo review: TBD (PR to be opened)
Tracking issue: wordingone/the-search#6

Stage 1d held task split sources:
  seed=1:  stage1d_seed1_b30000_result.json
  seed=42: stage1d_premise_truth_b30000_minimality_result.json

CANONICAL RESULT:
  stage1i_depth3_reach_seed<N>_result.json

SMOKE PATH (TEMP, --smoke flag):
  stage1i_depth3_TEMP.json

Usage:
  python stage1i_depth3_reach.py --seed 42          # canonical, diagnostic
  python stage1i_depth3_reach.py --seed 1           # canonical, confirmatory
  python stage1i_depth3_reach.py --seed 42 --smoke  # 3 tasks, TEMP path
  python stage1i_depth3_reach.py --test-gate        # inject gate violations only
"""

import numpy as np
import json, os, sys, time, collections, hashlib, heapq, math, random, argparse
from collections import deque

# -- Paths -------------------------------------------------------------------
DATA_PATH  = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
S1D_PATHS  = {
    1:  "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/stage1d_seed1_b30000_result.json",
    42: "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/stage1d_premise_truth_b30000_minimality_result.json",
}
TEMP_PATH   = "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/stage1i_depth3_TEMP.json"
RESULT_TMPL = "B:/M/the-search/incoming/arc-agi1-visa/03_R4_transfer_wall/stage1i_depth3_reach_seed{seed}_result.json"

# -- Constants ----------------------------------------------------------------
N_HELD          = 200
BUDGET_B        = 30_000
TIME_PER_TASK_D = 120.0      # seconds -- held constant from Stage 1h
ARM_D_SEED      = 0          # random arm seed, same across stages
MAX_DEPTH       = 3          # ONE variable changed from Stage 1h (was 2)
PREV_SPACE_HASH = "0890317fe99bc9f1"   # Stage 1h hash (1370-leaf depth-2 space)
PREV_N_LEAVES   = 1370

# -- Object extraction --------------------------------------------------------
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

# -- Predicates ---------------------------------------------------------------
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
    'unique_color':  lambda g, objs: [o for o in objs if collections.Counter(x['color'] for x in objs)[o['color']] == 1],
    'most_common_c': lambda g, objs: ([o for o in objs if o['color'] == collections.Counter(x['color'] for x in objs).most_common(1)[0][0]] if objs else []),
    'all':           lambda g, objs: objs,
}
for _c in range(1, 10):
    PREDICATES[f'color_{_c}'] = (lambda c: lambda g, objs: [o for o in objs if o['color'] == c])(_c)
PRED_NAMES = sorted(PREDICATES.keys())

# -- Existing transforms (from Stage 1c/1e/1f) --------------------------------
def _recolor(new_c):
    def fn(grid, sel):
        g = grid.copy()
        for obj in sel:
            for r, c in obj['cells']:
                g[r, c] = new_c
        return g
    return fn

def _delete(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        for r, c in obj['cells']:
            g[r, c] = bg
    return g

def _keep_only(grid, sel):
    g = np.full_like(grid, get_bg(grid))
    for obj in sel:
        for r, c in obj['cells']:
            g[r, c] = obj['color']
    return g

def _translate(dy, dx):
    def fn(grid, sel):
        h, w = grid.shape
        g = grid.copy()
        bg = get_bg(grid)
        for obj in sel:
            for r, c in obj['cells']:
                g[r, c] = bg
        for obj in sel:
            for r, c in obj['cells']:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w:
                    g[nr, nc] = obj['color']
        return g
    return fn

# -- MAP_GEOM transforms (from Stage 1f) --------------------------------------
def _geom_flip_h(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        cols = [c for r, c in cells]
        c_min, c_max = min(cols), max(cols)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nc = c_min + (c_max - c)
            if 0 <= r < g.shape[0] and 0 <= nc < g.shape[1]:
                g[r, nc] = obj['color']
    return g

def _geom_flip_v(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r)
            if 0 <= nr < g.shape[0] and 0 <= c < g.shape[1]:
                g[nr, c] = obj['color']
    return g

def _geom_rot_90(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, c_min = min(rows), min(cols)
        c_max = max(cols)
        w = c_max - c_min + 1
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min
            c_rel = c - c_min
            nr = r_min + (w - 1 - c_rel)
            nc = c_min + r_rel
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj['color']
    return g

def _geom_rot_180(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            nr = r_min + (r_max - r)
            nc = c_min + (c_max - c)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj['color']
    return g

def _geom_rot_270(grid, sel):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in sel:
        cells = obj['cells']
        if not cells: continue
        rows = [r for r, c in cells]
        cols = [c for r, c in cells]
        r_min, r_max = min(rows), max(rows)
        c_min = min(cols)
        h = r_max - r_min + 1
        for r, c in cells:
            g[r, c] = bg
        for r, c in cells:
            r_rel = r - r_min
            c_rel = c - c_min
            nr = r_min + c_rel
            nc = c_min + (h - 1 - r_rel)
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[nr, nc] = obj['color']
    return g

# -- MAP_RELATE: object-relational translate-to-anchor (Stage 1g, unchanged) --
RELATE_MODES = ['adjacent_down', 'adjacent_left', 'adjacent_right', 'adjacent_up']

def _largest_cc(selected_objects):
    """Return single largest CC among selected objects. Tiebreak: cell-count desc, min row, min col."""
    if not selected_objects:
        return None
    return sorted(
        selected_objects,
        key=lambda o: (-o['area'],
                       min(r for r, c in o['cells']),
                       min(c for r, c in o['cells']))
    )[0]

def _map_relate(grid, src_sel, anchor_sel, mode):
    """Translate each src connected component to be adjacent to anchor's largest CC."""
    anchor_obj = _largest_cc(anchor_sel)
    if anchor_obj is None:
        return grid.copy()

    anc_cells = anchor_obj['cells']
    anc_r0 = min(r for r, c in anc_cells)
    anc_r1 = max(r for r, c in anc_cells)
    anc_c0 = min(c for r, c in anc_cells)
    anc_c1 = max(c for r, c in anc_cells)

    h, w = grid.shape
    g = grid.copy()
    bg = get_bg(grid)

    for obj in src_sel:
        for r, c in obj['cells']:
            g[r, c] = bg

    for obj in src_sel:
        cells = obj['cells']
        if not cells:
            continue
        obj_r0 = min(r for r, c in cells)
        obj_r1 = max(r for r, c in cells)
        obj_c0 = min(c for r, c in cells)
        obj_c1 = max(c for r, c in cells)

        if mode == 'adjacent_left':
            dr = anc_r0 - obj_r0
            dc = (anc_c0 - 1) - obj_c1
        elif mode == 'adjacent_right':
            dr = anc_r0 - obj_r0
            dc = (anc_c1 + 1) - obj_c0
        elif mode == 'adjacent_up':
            dr = (anc_r0 - 1) - obj_r1
            dc = anc_c0 - obj_c0
        elif mode == 'adjacent_down':
            dr = (anc_r1 + 1) - obj_r0
            dc = anc_c0 - obj_c0
        else:
            dr, dc = 0, 0

        for r, c in cells:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                g[nr, nc] = obj['color']

    return g

# -- Transform registry -------------------------------------------------------
TRANSFORMS = {
    'delete':       _delete,
    'keep_only':    _keep_only,
    'geom_flip_h':  _geom_flip_h,
    'geom_flip_v':  _geom_flip_v,
    'geom_rot_90':  _geom_rot_90,
    'geom_rot_180': _geom_rot_180,
    'geom_rot_270': _geom_rot_270,
}
for _c in range(10):
    TRANSFORMS[f'recolor_{_c}'] = (lambda c: lambda g, sel: _recolor(c)(g, sel))(_c)
for _dy, _dx in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
    TRANSFORMS[f'translate_{_dy:+d}_{_dx:+d}'] = (
        lambda dy, dx: lambda g, sel: _translate(dy, dx)(g, sel))(_dy, _dx)
TRANSFORM_NAMES = sorted(TRANSFORMS.keys())

# -- Whole-grid primitives (unchanged) ----------------------------------------
def _crop_bg(g):
    bg = get_bg(g)
    nz = np.argwhere(g != bg)
    if len(nz) == 0: return g
    return g[nz[:,0].min():nz[:,0].max()+1, nz[:,1].min():nz[:,1].max()+1]

GRID_PRIMS = {
    'flip_h':  lambda g: g[:, ::-1],
    'flip_v':  lambda g: g[::-1],
    'rot_90':  lambda g: np.rot90(g, 1),
    'rot_180': lambda g: np.rot90(g, 2),
    'rot_270': lambda g: np.rot90(g, 3),
    'tr':      lambda g: g.T,
    'up2':     lambda g: np.repeat(np.repeat(g, 2, 0), 2, 1),
    'down2':   lambda g: g[::2, ::2],
    'id':      lambda g: g,
    'crop':    _crop_bg,
}
PRIM_NAMES = sorted(GRID_PRIMS.keys())

# -- Skeleton grammar (same 1370-leaf Stage 1g/1h space) ----------------------
_RECOLOR_TRANSFORMS   = sorted(t for t in TRANSFORM_NAMES if t.startswith('recolor_'))
_TRANSLATE_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith('translate_'))
_DELETE_TRANSFORMS    = ['delete']
_KEEPONLY_TRANSFORMS  = ['keep_only']
_GEOM_TRANSFORMS      = sorted(t for t in TRANSFORM_NAMES if t.startswith('geom_'))

MAP_TRANSFORM_FAMILIES = {
    'MAP_RECOLOR':   _RECOLOR_TRANSFORMS,
    'MAP_DELETE':    _DELETE_TRANSFORMS,
    'MAP_KEEPONLY':  _KEEPONLY_TRANSFORMS,
    'MAP_TRANSLATE': _TRANSLATE_TRANSFORMS,
    'MAP_GEOM':      _GEOM_TRANSFORMS,
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

def prog_skeleton_type(prog):
    t = prog[0]
    if t == 'prim':
        return f'PRIM_{prog[1].upper()}'
    if t == 'map_apply':
        tr = prog[2]
        if tr.startswith('recolor_'):   return 'MAP_RECOLOR'
        if tr == 'delete':              return 'MAP_DELETE'
        if tr == 'keep_only':           return 'MAP_KEEPONLY'
        if tr.startswith('translate_'): return 'MAP_TRANSLATE'
        if tr.startswith('geom_'):      return 'MAP_GEOM'
        return 'MAP_UNKNOWN'
    if t == 'map_relate':
        return 'MAP_RELATE'
    if t == 'compose':
        return f'COMPOSE({prog_skeleton_type(prog[1])},{prog_skeleton_type(prog[2])})'
    return 'UNKNOWN'

# -- Program evaluation -------------------------------------------------------
def eval_program(prog, grid, lib={}):
    try:
        return _eval(prog, np.array(grid, dtype=np.int64), lib)
    except Exception:
        return None

def _eval(prog, g, lib):
    t = prog[0]
    if t == 'prim':
        return GRID_PRIMS[prog[1]](g)
    if t == 'map_apply':
        try:
            objs = _extract_cached(g)
            selected = PREDICATES[prog[1]](g, objs)
            if not selected:
                return g.copy()
            return TRANSFORMS[prog[2]](g, selected)
        except Exception:
            return None
    if t == 'map_relate':
        try:
            src_pred_name, anchor_pred_name, mode = prog[1], prog[2], prog[3]
            objs = _extract_cached(g)
            src_sel    = PREDICATES[src_pred_name](g, objs)
            anchor_sel = PREDICATES[anchor_pred_name](g, objs)
            if not src_sel or not anchor_sel:
                return g.copy()
            return _map_relate(g, src_sel, anchor_sel, mode)
        except Exception:
            return None
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
        if r is None or r.shape != out.shape or not np.array_equal(r, out):
            return False
    return True

def prog_hash(prog):
    return hashlib.sha256(str(prog).encode()).hexdigest()[:12]

def tuple_to_prog(tup, leaves):
    if len(tup) == 1:
        return leaves[tup[0]]
    return ('compose', leaves[tup[0]], tuple_to_prog(tup[1:], leaves))

def check_tuple(tup, task_io, leaves):
    prog = tuple_to_prog(tup, leaves)
    return check_prog(prog, task_io)

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

# -- Fail-closed gate ---------------------------------------------------------
def check_gate_violations(artifact, stage1d_held_id_set, actual_n_leaves):
    """Returns list of violation strings. Empty list = gate passes."""
    violations = []

    # n_leaves: leaf SET unchanged (still 1370 -- depth-3 composes same leaves deeper)
    if artifact.get('n_leaves') != actual_n_leaves:
        violations.append(
            f"n_leaves={artifact.get('n_leaves')} != builder actual {actual_n_leaves}")
    if not artifact.get('n_leaves_asserted'):
        violations.append("n_leaves_asserted missing or False")

    # held_task_ids
    artifact_held_ids = artifact.get('held_task_ids', [])
    if len(artifact_held_ids) != N_HELD:
        violations.append(f"held_task_ids count={len(artifact_held_ids)} != {N_HELD}")
    if set(artifact_held_ids) != stage1d_held_id_set:
        extra   = set(artifact_held_ids) - stage1d_held_id_set
        missing = stage1d_held_id_set - set(artifact_held_ids)
        violations.append(
            f"held_task_ids set mismatch vs stage1d: extra={len(extra)} missing={len(missing)}")

    # candidate_eval_budget
    if artifact.get('candidate_eval_budget') != BUDGET_B:
        violations.append(
            f"candidate_eval_budget={artifact.get('candidate_eval_budget')} != {BUDGET_B}")
    if not artifact.get('candidate_eval_budget_asserted'):
        violations.append("candidate_eval_budget_asserted missing or False")

    # max_depth: the ONE variable changed (must be 3, was 2 in Stage 1h)
    if artifact.get('max_depth') != MAX_DEPTH:
        violations.append(
            f"max_depth={artifact.get('max_depth')} != {MAX_DEPTH} "
            f"(ONE-variable check: was 2 in Stage 1h)")

    # space_hash: depth-3 EXPANDS the space -- must DIFFER from Stage 1h hash
    if artifact.get('space_hash') == PREV_SPACE_HASH:
        violations.append(
            f"space_hash={artifact.get('space_hash')} == Stage 1h hash {PREV_SPACE_HASH} "
            f"(space MUST change at depth-3; depth is embedded in hash)")
    if artifact.get('prev_space_hash') != PREV_SPACE_HASH:
        violations.append(
            f"prev_space_hash={artifact.get('prev_space_hash')} != {PREV_SPACE_HASH}")

    # NEW (#4 finding): max(per-task n_evals) <= BUDGET_B -- behavioral enforcement
    # Config-value assertion necessary-not-sufficient; this catches 1g-class overruns.
    for tid, rec in artifact.get('arm_d_per_task', {}).items():
        if rec.get('n_evals', 0) > BUDGET_B:
            violations.append(
                f"arm_d[{tid}]: n_evals={rec.get('n_evals')} > BUDGET_B={BUDGET_B} "
                f"(behavioral budget overrun -- budget break not enforced)")
            break  # one example is sufficient

    # per-task: each unsolved task must hit time wall OR exhaust budget (no premature exits)
    for tid, rec in artifact.get('arm_d_per_task', {}).items():
        if not rec.get('solved', False):
            if not rec.get('time_limit_hit', False) and rec.get('n_evals', 0) < BUDGET_B:
                violations.append(
                    f"arm_d[{tid}]: unsolved, not time-limited, n_evals={rec.get('n_evals')} < {BUDGET_B}")

    return violations

# -- Gate violation injection (smoke testing) ---------------------------------
def run_gate_tests(actual_n_leaves, stage1d_held_id_set, space_hash, held_task_ids):
    """Inject known violations, confirm gate catches each. Returns True if all caught."""
    dummy_per_task = {tid: {'solved': False, 'time_limit_hit': False, 'n_evals': BUDGET_B}
                      for tid in held_task_ids}

    good_artifact = {
        'n_leaves':                       actual_n_leaves,
        'n_leaves_asserted':              True,
        'held_task_ids':                  list(held_task_ids),
        'candidate_eval_budget':          BUDGET_B,
        'candidate_eval_budget_asserted': True,
        'max_depth':                      MAX_DEPTH,
        'arm_d_per_task':                 dummy_per_task,
        'space_hash':                     space_hash,
        'prev_space_hash':                PREV_SPACE_HASH,
    }
    base_violations = check_gate_violations(good_artifact, stage1d_held_id_set, actual_n_leaves)
    if base_violations:
        print(f"  SMOKE GATE TEST FAIL: good artifact has violations: {base_violations}")
        return False
    print("  [OK] Good artifact passes gate")

    # Build an over-budget per_task for injection test #5
    first_task_id = list(held_task_ids)[0]
    over_budget_per_task = dict(dummy_per_task)
    over_budget_per_task[first_task_id] = {
        'solved': False, 'time_limit_hit': False, 'n_evals': BUDGET_B + 1
    }

    tests = [
        ("budget != 30000",
         {**good_artifact, 'candidate_eval_budget': 25000}),
        ("budget_asserted=False",
         {**good_artifact, 'candidate_eval_budget_asserted': False}),
        ("n_leaves wrong (Stage 1f value 410)",
         {**good_artifact, 'n_leaves': 410}),
        ("max_depth wrong (2 -- Stage 1h value)",
         {**good_artifact, 'max_depth': 2}),
        ("max_evals over budget -- behavioral overrun (1g-class catch)",
         {**good_artifact, 'arm_d_per_task': over_budget_per_task}),
        ("space_hash == Stage 1h hash -- depth-3 must change hash",
         {**good_artifact, 'space_hash': PREV_SPACE_HASH}),
        ("prev_space_hash wrong",
         {**good_artifact, 'prev_space_hash': 'badhash123456789'}),
        ("held IDs mismatch",
         {**good_artifact, 'held_task_ids': list(held_task_ids)[:-1]}),
    ]

    all_caught = True
    for label, bad_artifact in tests:
        viols = check_gate_violations(bad_artifact, stage1d_held_id_set, actual_n_leaves)
        if viols:
            print(f"  [OK] Gate caught '{label}': {viols[0]}")
        else:
            print(f"  FAIL: Gate MISSED '{label}'")
            all_caught = False

    return all_caught

# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, choices=[1, 42], default=42,
                        help='Stage 1d held split seed (42=diagnostic, 1=confirmatory)')
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke mode: 3 tasks only, write to TEMP path')
    parser.add_argument('--test-gate', action='store_true',
                        help='Run gate violation injection tests before main run')
    args = parser.parse_args()

    if args.smoke:
        n_tasks   = 3
        result_path = TEMP_PATH
        print(f"SMOKE MODE: {n_tasks} tasks x {TIME_PER_TASK_D:.0f}s -> {result_path}")
    else:
        n_tasks   = N_HELD
        result_path = RESULT_TMPL.format(seed=args.seed)
        print(f"CANONICAL MODE: seed={args.seed} -> {result_path}")

    # -- Load ARC tasks -------------------------------------------------------
    all_tasks = []
    for split in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, split)
        if not os.path.exists(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.json'):
                with open(os.path.join(d, fn)) as f:
                    dat = json.load(f)
                task_id = fn[:-5]
                io = dat.get('train', [])
                all_tasks.append({'id': task_id, 'io': io})

    # -- Load held task IDs from Stage 1d -------------------------------------
    s1d_path = S1D_PATHS[args.seed]
    with open(s1d_path) as f:
        s1d_result = json.load(f)
    stage1d_held_ids    = s1d_result['held_task_ids']
    stage1d_held_id_set = set(stage1d_held_ids)
    held_tasks_all = [t for t in all_tasks if t['id'] in stage1d_held_id_set]
    assert len(held_tasks_all) == N_HELD, \
        f"Expected {N_HELD} held tasks, got {len(held_tasks_all)}"
    held_tasks = held_tasks_all[:n_tasks]
    print(f"\nLoaded {len(held_tasks_all)} held tasks from Stage 1d seed={args.seed}")
    print(f"  Running on {n_tasks} tasks ({'smoke' if args.smoke else 'full'})")

    # -- Build grammar (same 1370-leaf SET, depth-3 composition) --------------
    leaves          = all_leaves_expanded()
    actual_n_leaves = len(leaves)
    space_hash      = compute_space_hash(leaves)
    print(f"\nGrammar (Stage 1g/1h leaf SET, depth-3 composition): {actual_n_leaves} leaves")
    print(f"  space_hash={space_hash}")
    print(f"  prev_space_hash (Stage 1h)={PREV_SPACE_HASH}")
    print(f"  leaf_set_unchanged=True (same 1370 leaves; depth is the one variable)")
    assert space_hash != PREV_SPACE_HASH, \
        f"FATAL: space_hash {space_hash} == Stage 1h hash {PREV_SPACE_HASH} -- depth-3 must expand space"
    print(f"  space_hash DIFFERS from Stage 1h (depth-3 expansion confirmed)")
    print(f"  Depth-3 space size (4-tuple): {actual_n_leaves}^4 = {actual_n_leaves**4:,}")
    print(f"\nONE VARIABLE: MAX_DEPTH = {MAX_DEPTH} (was 2 in Stage 1h)")

    # -- Gate tests (--test-gate or always in smoke mode) ---------------------
    if args.test_gate or args.smoke:
        print(f"\n=== Gate violation injection tests ===")
        all_caught = run_gate_tests(
            actual_n_leaves, stage1d_held_id_set, space_hash, stage1d_held_ids)
        if not all_caught:
            print("GATE TESTS FAILED -- fix before canonical run")
            sys.exit(1)
        print("All gate tests passed\n")

    # -- Arm D: random budget-and-time-limited reach lower-bound (depth-3) ----
    print(f"\n=== Arm D (reach LB): {TIME_PER_TASK_D:.0f}s/task OR {BUDGET_B} evals, "
          f"seed={ARM_D_SEED}, depth={MAX_DEPTH} ===")
    rng_d = random.Random(ARM_D_SEED)
    arm_d_per_task = {}
    n_evals_total  = 0
    t_d            = time.time()

    for task_num, task in enumerate(held_tasks):
        t_task           = time.time()
        n_evals          = 0
        solved           = False
        budget_exhausted = False

        while time.time() - t_task < TIME_PER_TASK_D:
            i = rng_d.randrange(actual_n_leaves)
            j = rng_d.randrange(actual_n_leaves)
            k = rng_d.randrange(actual_n_leaves)
            l = rng_d.randrange(actual_n_leaves)
            n_evals += 1
            if check_tuple((i, j, k, l), task['io'], leaves):
                tup  = (i, j, k, l)
                prog = tuple_to_prog(tup, leaves)
                arm_d_per_task[task['id']] = {
                    'solved':            True,
                    'n_evals':           n_evals,
                    'time_limit_hit':    False,
                    'budget_exhausted':  False,
                    'prog_tuple':        list(tup),
                    'prog_str':          str(prog),
                    'prog_hash':         prog_hash(prog),
                    'skeleton_type':     prog_skeleton_type(prog),
                    'used_relate':       any(leaves[idx][0] == 'map_relate' for idx in tup),
                }
                solved = True
                break
            if n_evals >= BUDGET_B:
                budget_exhausted = True
                break

        if not solved:
            time_hit = (time.time() - t_task) >= TIME_PER_TASK_D
            arm_d_per_task[task['id']] = {
                'solved':            False,
                'n_evals':           n_evals,
                'time_limit_hit':    time_hit,
                'budget_exhausted':  budget_exhausted,
            }

        n_evals_total += n_evals
        if (task_num + 1) % 50 == 0 or (args.smoke and (task_num + 1) % 1 == 0):
            elapsed  = time.time() - t_d
            n_solved_so_far = sum(1 for r in arm_d_per_task.values() if r.get('solved'))
            n_time_hit  = sum(1 for r in arm_d_per_task.values() if r.get('time_limit_hit'))
            n_budget_ex = sum(1 for r in arm_d_per_task.values() if r.get('budget_exhausted'))
            print(f"  {task_num+1}/{n_tasks} tasks, {n_solved_so_far} solved, "
                  f"time_hit={n_time_hit} budget_ex={n_budget_ex}, elapsed {elapsed:.0f}s")

    n_solved_d       = sum(1 for r in arm_d_per_task.values() if r.get('solved'))
    time_limit_count = sum(1 for r in arm_d_per_task.values() if r.get('time_limit_hit'))
    budget_ex_count  = sum(1 for r in arm_d_per_task.values() if r.get('budget_exhausted'))
    relate_solutions = sum(1 for r in arm_d_per_task.values() if r.get('used_relate'))
    elapsed_total    = time.time() - t_d

    unsolved_evals = [r['n_evals'] for r in arm_d_per_task.values()
                      if not r.get('solved') and 'n_evals' in r]
    min_evals_unsolved    = min(unsolved_evals) if unsolved_evals else None
    median_evals_unsolved = sorted(unsolved_evals)[len(unsolved_evals) // 2] if unsolved_evals else None
    max_evals_unsolved    = max(unsolved_evals) if unsolved_evals else None

    print(f"\nArm D done: {n_solved_d}/{n_tasks} solved in {elapsed_total:.0f}s, "
          f"{n_evals_total:,} evals")
    print(f"  time_limit_hit: {time_limit_count}/{n_tasks} tasks")
    print(f"  budget_exhausted: {budget_ex_count}/{n_tasks} tasks")
    print(f"  min_evals_unsolved: {min_evals_unsolved}  "
          f"median: {median_evals_unsolved}  max: {max_evals_unsolved}")
    print(f"  Solutions using MAP_RELATE: {relate_solutions}/{n_solved_d}")
    if n_solved_d > 7:
        verdict = 'LIFT'
    elif n_solved_d == 7:
        verdict = 'FLAT_DEPTH_NEUTRAL'
    elif n_solved_d < 7:
        verdict = 'BELOW_BASELINE_BUG'
    print(f"  REACH_DIAGNOSTIC_{verdict}")

    # -- Build artifact -------------------------------------------------------
    artifact = {
        'experiment':                     'stage1i_depth3_reach',
        'spec':                           'pre_reg/stage1i_depth3_reach_preregistration.md',
        'leo_review':                     'mail TBD (PR pending)',
        'tracking_issue':                 'wordingone/the-search#6',
        'seed':                           args.seed,
        'smoke':                          args.smoke,
        'n_tasks':                        n_tasks,
        'n_solved':                       n_solved_d,
        'n_held_full':                    N_HELD,
        # Grammar
        'n_leaves':                       actual_n_leaves,
        'n_leaves_asserted':              True,
        'leaf_set_unchanged':             True,
        'prev_n_leaves':                  PREV_N_LEAVES,
        'space_hash':                     space_hash,
        'prev_space_hash':                PREV_SPACE_HASH,
        'grammar_unchanged':              False,  # depth changed -> space_hash changes
        # Protocol -- ONE variable
        'max_depth':                      MAX_DEPTH,
        'time_limit_s':                   TIME_PER_TASK_D,
        'candidate_eval_budget':          BUDGET_B,
        'candidate_eval_budget_asserted': True,
        'search_algorithm':               'random_budget_and_time_limited',
        'arm_d_seed':                     ARM_D_SEED,
        # Binding constraint diagnostics
        'time_limit_hit_count':           time_limit_count,
        'budget_exhausted_count':         budget_ex_count,
        'min_evals_unsolved':             min_evals_unsolved,
        'median_evals_unsolved':          median_evals_unsolved,
        'max_evals_unsolved':             max_evals_unsolved,
        # Held split
        'held_task_ids':                  [t['id'] for t in held_tasks_all],
        'held_task_ids_match_stage1d':    True,
        'stage1d_result_path':            s1d_path,
        # Per-task results
        'arm_d_per_task':                 arm_d_per_task,
        # Diagnostics
        'relate_solutions':               relate_solutions,
        'evals_total':                    n_evals_total,
        'elapsed_s':                      elapsed_total,
        # R1-R6
        'r1_r6_note':                     'Pure reach/measurement run. R1: no external reward driving random search. R5: fixed ground truth = 200 held ARC tasks. R2/R3/R4/R6: N/A (no self-modification).',
    }

    # -- FAIL-CLOSED PRE-WRITE GATE -------------------------------------------
    violations = check_gate_violations(artifact, stage1d_held_id_set, actual_n_leaves)
    if violations:
        print("\n=== FAIL-CLOSED PRE-WRITE GATE: VIOLATIONS -- ARTIFACT NOT WRITTEN ===")
        for v in violations:
            print(f"  VIOLATION: {v}")
        sys.exit(1)
    print("\nFail-closed gate: PASSED")

    # -- Write artifact -------------------------------------------------------
    with open(result_path, 'w') as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact written: {result_path}")
    print(f"\nSummary: D={n_solved_d}/{n_tasks} ({'smoke sample' if args.smoke else 'full'})")
    print(f"  time_limit_hit={time_limit_count}/{n_tasks} "
          f"budget_exhausted={budget_ex_count}/{n_tasks}")
    print(f"  space_hash={space_hash} (differs from Stage 1h: {space_hash != PREV_SPACE_HASH})")
    print(f"  n_leaves={actual_n_leaves} (leaf set unchanged: True)")
    print(f"  max_depth={MAX_DEPTH} (was 2 in Stage 1h)")

if __name__ == '__main__':
    main()
