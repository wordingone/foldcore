"""
Stage 1d — Premise-Truth / Subprogram Containment (Leo #11818, 2026-05-30).

CONSOLIDATED SPEC (#11818 supersedes #11809/#11813).

PREMISE QUESTION: Do the Stage 1c ACCEPTED macros appear as subprograms of deeper
held solutions? NOT "does a new library form at higher budget" — that was the wrong
question. The FIXED Stage 1c library (1 macro: COMPOSE(PRIM_CROP,MAP_KEEPONLY))
is tested for reuse in held solutions across a budget coverage curve.

Components:
  1. base_artifact: stage1c_holed_result.json — fixed macro library (1 macro).
  2. Coverage curve: enumerate HELD tasks WITHOUT library at budgets 3000/6000/10000.
  3. Per held task solved: check if Stage 1c macro appears as EXACT subprogram.
     Near-miss (semantic): exhaustively check all concrete programs with macro skeleton.
  4. PREMISE_* classification (not HOLED_*):
     PREMISE_INSTRUMENTATION_INCOMPLETE: insufficient held coverage.
     PREMISE_SELECTION_BOTTLENECK: macros appear in held but transfer didn't select them.
     PREMISE_FALSE_EXACT: adequate coverage, macros absent from held solutions.
     PREMISE_NEAR_MISS_SEMANTIC: no exact hits but semantic near-misses present.

[Original Stage 1c-HOLED header follows for grammar/schema reference]

Stage 1c-HOLED: Holed Skeleton Grammar BFS (Leo #11766, 2026-05-30).

TWO-VARIABLE CHANGE (explicit, both deliberate per Leo #11766):
  1. Representation: concrete-330 → holed skeleton grammar (~14 skeleton types).
     Root cause fix for SINGLE_ARM_N200_DEPTH_STARVATION_SIGNAL (Stage 1c).
     Branching-bound: flat-330 depth-1=330²=108,900 @2.45%; flat depth-2≈36M intractable.
     With 14 skeletons: depth-1=196 skeleton pairs, depth-2=2744. Tractable search.
  2. Depth tractability: BFS ordered by skeleton pair (not lex concrete order).
     Within same BUDGET=3000, covers DIFFERENT programs: small-fill skeleton pairs first.
     Abstraction funnel: anti-unification at skeleton level.

Holed skeleton grammar (~14 types):
  MAP_RECOLOR: map_apply(pred, recolor_X) — 16 preds × 10 colors = 160 fills
  MAP_DELETE:  map_apply(pred, delete) — 16 fills
  MAP_KEEPONLY: map_apply(pred, keep_only) — 16 fills
  MAP_TRANSLATE: map_apply(pred, translate_*) — 16 preds × 8 dirs = 128 fills
  PRIM_*: 10 whole-grid primitives, 1 fill each

Key hypothesis: MAP_RECOLOR(non_bg→5) and MAP_RECOLOR(color_7→5) both map to skeleton
MAP_RECOLOR → anti-unification yields occ=2 at skeleton level. For compose skeletons:
body_size=3, savings=occ*2-3 > 0 for occ>=2 → MACRO FORMS at skeleton level.

Kai #11748 gate schema (DEPTH_STARVATION | FORMATION_NEGATIVE_DEPTH2 | HOLLOW_DEPTH2 |
  TRANSFER_GENUINE_DEPTH2) — applied at skeleton level.

Claim scope: explicitly notes two-variable change, skeleton-level grouping, concrete BFS
  coverage same ~2.45% depth-1 but with skeleton-pair ordering.
"""

import numpy as np
import json, os, sys, time, collections, hashlib
from collections import deque

DATA_PATH       = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
S1C_RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1c_holed_result.json"
RESULT_PATH     = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1d_premise_truth_b30000_minimality_result.json"
BASE_ARTIFACT   = "stage1d_premise_truth_result.json"  # original artifact; do NOT overwrite

BUDGET_POINTS   = [3000, 6000, 10000, 30000]  # Leo #11818 curve + #11832 confirmatory point
N_HELD          = 200
SPLIT_SEED      = 42
COVERAGE_THRESHOLD = 5  # min held solves at top budget for test to be conclusive

# Retained for helper function default args (dream_sleep, transfer_test, etc.)
N_DREAMS        = 30
MDL_MIN_OCC     = 2
TRANSFER_MARGIN = 5.0
N_MBPP          = 50

# ── Object extraction ─────────────────────────────────────────────────────────
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

# ── Object predicates (PRESERVED from Stage 1b/1c) ───────────────────────────
def _pred_largest(objs):
    if not objs: return []
    mx = max(o['area'] for o in objs)
    return [o for o in objs if o['area'] == mx]

def _pred_smallest(objs):
    if not objs: return []
    mn = min(o['area'] for o in objs)
    return [o for o in objs if o['area'] == mn]

def _pred_non_bg(grid, objs):
    bg = get_bg(grid)
    return [o for o in objs if o['color'] != bg]

def _pred_bg(grid, objs):
    bg = get_bg(grid)
    return [o for o in objs if o['color'] == bg]

def _pred_unique_color(objs):
    cc = collections.Counter(o['color'] for o in objs)
    return [o for o in objs if cc[o['color']] == 1]

def _pred_most_common_color(objs):
    if not objs: return []
    most = collections.Counter(o['color'] for o in objs).most_common(1)[0][0]
    return [o for o in objs if o['color'] == most]

def _pred_all(objs):
    return objs

PREDICATES = {
    'largest':       lambda g, objs: _pred_largest(objs),
    'smallest':      lambda g, objs: _pred_smallest(objs),
    'non_bg':        lambda g, objs: _pred_non_bg(g, objs),
    'bg_obj':        lambda g, objs: _pred_bg(g, objs),
    'unique_color':  lambda g, objs: _pred_unique_color(objs),
    'most_common_c': lambda g, objs: _pred_most_common_color(objs),
    'all':           lambda g, objs: _pred_all(objs),
}
for _c in range(1, 10):
    PREDICATES[f'color_{_c}'] = (lambda c: lambda g, objs: [o for o in objs if o['color'] == c])(_c)
PRED_NAMES = sorted(PREDICATES.keys())

# ── Object transforms (PRESERVED from Stage 1b/1c) ───────────────────────────
def _recolor(new_c):
    def fn(grid, selected):
        g = grid.copy()
        for obj in selected:
            for r, c in obj['cells']:
                g[r, c] = new_c
        return g
    return fn

def _delete(grid, selected):
    g = grid.copy()
    bg = get_bg(grid)
    for obj in selected:
        for r, c in obj['cells']:
            g[r, c] = bg
    return g

def _keep_only(grid, selected):
    g = np.full_like(grid, get_bg(grid))
    for obj in selected:
        for r, c in obj['cells']:
            g[r, c] = obj['color']
    return g

def _translate(dy, dx):
    def fn(grid, selected):
        h, w = grid.shape
        g = grid.copy()
        bg = get_bg(grid)
        for obj in selected:
            for r, c in obj['cells']:
                g[r, c] = bg
        for obj in selected:
            for r, c in obj['cells']:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w:
                    g[nr, nc] = obj['color']
        return g
    return fn

TRANSFORMS = {
    'delete':    lambda g, sel: _delete(g, sel),
    'keep_only': lambda g, sel: _keep_only(g, sel),
}
for _c in range(10):
    TRANSFORMS[f'recolor_{_c}'] = (lambda c: lambda g, sel: _recolor(c)(g, sel))(_c)
for _dy, _dx in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
    TRANSFORMS[f'translate_{_dy:+d}_{_dx:+d}'] = (
        lambda dy, dx: lambda g, sel: _translate(dy, dx)(g, sel))(_dy, _dx)
TRANSFORM_NAMES = sorted(TRANSFORMS.keys())

# ── Whole-grid primitives (PRESERVED from Stage 1b/1c) ───────────────────────
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

# ── Skeleton grammar (~14 types) ──────────────────────────────────────────────
# Transform families for skeleton grouping
_RECOLOR_TRANSFORMS  = sorted(t for t in TRANSFORM_NAMES if t.startswith('recolor_'))
_TRANSLATE_TRANSFORMS = sorted(t for t in TRANSFORM_NAMES if t.startswith('translate_'))
_DELETE_TRANSFORMS   = ['delete']
_KEEPONLY_TRANSFORMS = ['keep_only']

MAP_TRANSFORM_FAMILIES = {
    'MAP_RECOLOR':   _RECOLOR_TRANSFORMS,   # 10 fills per pred → 160 total
    'MAP_DELETE':    _DELETE_TRANSFORMS,     # 1 fill per pred  → 16 total
    'MAP_KEEPONLY':  _KEEPONLY_TRANSFORMS,   # 1 fill per pred  → 16 total
    'MAP_TRANSLATE': _TRANSLATE_TRANSFORMS,  # 8 fills per pred → 128 total
}
# PRIM skeletons: one per primitive name
PRIM_SKELETON_NAMES = [f'PRIM_{n.upper()}' for n in PRIM_NAMES]

# Full skeleton name list (defines BFS ordering — small fills first)
SKELETON_ORDER = (
    PRIM_SKELETON_NAMES +           # 10 skeletons, 1 fill each = 10 total
    ['MAP_DELETE', 'MAP_KEEPONLY',  # 16 fills each
     'MAP_TRANSLATE',               # 128 fills
     'MAP_RECOLOR']                 # 160 fills (last: most expensive)
)

def skeleton_fills(sk_name, lib=None):
    """All concrete programs of a given skeleton type (depth-0 leaves)."""
    if sk_name.startswith('PRIM_'):
        prim_n = sk_name[5:].lower()
        return [('prim', prim_n)] if prim_n in GRID_PRIMS else []
    if sk_name in MAP_TRANSFORM_FAMILIES:
        trs = MAP_TRANSFORM_FAMILIES[sk_name]
        return [('map_apply', pred, tr) for pred in PRED_NAMES for tr in trs]
    return []

def all_leaves_holed(lib=None):
    """All 330 concrete programs ordered by skeleton type (BFS search order)."""
    result = []
    if lib:
        for name in sorted(lib.keys()):
            result.append(('macro', name))
    for sk in SKELETON_ORDER:
        result.extend(skeleton_fills(sk, lib))
    return result

def prog_skeleton_type(prog):
    """Returns skeleton type string for abstraction funnel grouping."""
    t = prog[0]
    if t == 'prim':
        return f'PRIM_{prog[1].upper()}'
    if t == 'map_apply':
        tr = prog[2]
        if tr.startswith('recolor_'):   return 'MAP_RECOLOR'
        if tr == 'delete':              return 'MAP_DELETE'
        if tr == 'keep_only':           return 'MAP_KEEPONLY'
        if tr.startswith('translate_'): return 'MAP_TRANSLATE'
        return 'MAP_UNKNOWN'
    if t == 'compose':
        sk1 = prog_skeleton_type(prog[1])
        sk2 = prog_skeleton_type(prog[2])
        return f'COMPOSE({sk1},{sk2})'
    if t == 'macro':
        return f'MACRO({prog[1]})'
    return 'UNKNOWN'

def prog_fills(prog):
    """Extract structured concrete fill values from a program (not just prog_str)."""
    t = prog[0]
    if t == 'prim':
        return {'prim_name': prog[1]}
    if t == 'map_apply':
        return {'predicate': prog[1], 'transform': prog[2]}
    if t == 'compose':
        return {'left_fills': prog_fills(prog[1]), 'right_fills': prog_fills(prog[2])}
    if t == 'macro':
        return {'macro_name': prog[1]}
    return {'raw': str(prog)[:80]}

def prog_macro_names(prog):
    """Return list of all ('macro', name) macro names appearing anywhere in prog."""
    t = prog[0]
    if t == 'macro':
        return [prog[1]]
    if t == 'compose':
        return prog_macro_names(prog[1]) + prog_macro_names(prog[2])
    return []

# ── Program evaluation (PRESERVED from Stage 1c) ──────────────────────────────
def eval_program(prog, grid, lib):
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
    if t == 'compose':
        r = _eval(prog[2], g, lib)
        if r is None: return None
        return _eval(prog[1], r, lib)
    if t == 'macro':
        m = lib.get(prog[1])
        if m is None: return None
        return _eval(m['body'], g, lib)
    return None

def check_prog(prog, task_io, lib):
    for pair in task_io:
        inp = np.array(pair['input'], dtype=np.int64)
        out = np.array(pair['output'], dtype=np.int64)
        r = eval_program(prog, inp, lib)
        if r is None or r.shape != out.shape or not np.array_equal(r, out):
            return False
    return True

def prog_hash(prog):
    return hashlib.sha256(str(prog).encode()).hexdigest()[:12]

def _prog_size(prog):
    t = prog[0]
    if t == 'compose': return 1 + _prog_size(prog[1]) + _prog_size(prog[2])
    return 1

def prog_type(prog):
    t = prog[0]
    if t == 'prim': return 'prim'
    if t == 'map_apply': return 'map_apply'
    if t == 'macro': return 'macro'
    if t == 'compose':
        has_map = any(p[0] == 'map_apply' for p in (prog[1], prog[2]))
        return 'compose_map_apply' if has_map else 'compose_prim'
    return 'unknown'

# ── BFS search (holed: skeleton-pair ordering) ────────────────────────────────
def search_task_holed(task_io, lib, budget=3000, dl_weights=None):
    """BFS ordered by skeleton pair priority. Returns (prog, nodes, depth, skel_type)."""
    nodes = 0

    # Depth-0: iterate skeletons in order, all fills
    leaves = all_leaves_holed(lib)
    for prog in leaves:
        nodes += 1
        if nodes > budget: return None, nodes, -1, None
        if check_prog(prog, task_io, lib):
            return prog, nodes, 0, prog_skeleton_type(prog)

    # Depth-1: iterate skeleton pairs in fill-size priority order (small pairs first)
    # This ensures PRIM×PRIM (100 checks), MAP_DELETE×PRIM (160), etc. get full coverage
    sk_fills = {sk: skeleton_fills(sk, lib) for sk in SKELETON_ORDER}
    # Add macro fills
    if lib:
        sk_fills['MACRO'] = [('macro', name) for name in sorted(lib.keys())]
        depth1_skel_order = ['MACRO'] + SKELETON_ORDER
    else:
        depth1_skel_order = SKELETON_ORDER

    for sk1 in depth1_skel_order:
        progs1 = sk_fills.get(sk1, skeleton_fills(sk1, lib))
        for sk2 in depth1_skel_order:
            progs2 = sk_fills.get(sk2, skeleton_fills(sk2, lib))
            for p1 in progs1:
                for p2 in progs2:
                    nodes += 1
                    if nodes > budget: return None, nodes, -1, None
                    prog = ('compose', p1, p2)
                    if check_prog(prog, task_io, lib):
                        return prog, nodes, 1, prog_skeleton_type(prog)

    # Depth-2 (rarely reached within budget)
    for sk1 in depth1_skel_order:
        progs1 = sk_fills.get(sk1, skeleton_fills(sk1, lib))
        for sk2 in depth1_skel_order:
            progs2 = sk_fills.get(sk2, skeleton_fills(sk2, lib))
            for sk3 in depth1_skel_order:
                progs3 = sk_fills.get(sk3, skeleton_fills(sk3, lib))
                for p1 in progs1:
                    for p2 in progs2:
                        for p3 in progs3:
                            nodes += 1
                            if nodes > budget: return None, nodes, -1, None
                            for prog in [('compose', p1, ('compose', p2, p3)),
                                         ('compose', ('compose', p1, p2), p3)]:
                                nodes += 1
                                if nodes > budget: return None, nodes, -1, None
                                if check_prog(prog, task_io, lib):
                                    return prog, nodes, 2, prog_skeleton_type(prog)

    return None, nodes, -1, None

# ── Skeleton-level abstraction funnel (anti-unification + non-tautology MDL) ──
# Fill-cost bit model: log2(n_fills) bits to specify a fill within a skeleton.
# concrete_baseline_bits = log2(n_leaves) per program (cost to store concrete).
import math as _math

def _fill_bits(skel_type):
    """Bits needed to specify the concrete fill(s) within a skeleton type."""
    if skel_type.startswith('PRIM_'):       return 0.0  # no holes (1 fill)
    if skel_type == 'MAP_DELETE':           return _math.log2(16)   # 16 preds
    if skel_type == 'MAP_KEEPONLY':         return _math.log2(16)
    if skel_type == 'MAP_TRANSLATE':        return _math.log2(128)  # 16*8
    if skel_type == 'MAP_RECOLOR':          return _math.log2(160)  # 16*10
    if skel_type.startswith('COMPOSE('):
        # Extract inner skeleton types (best-effort)
        inner = skel_type[8:-1]
        mid   = inner.find(',')
        if mid > 0:
            sk1, sk2 = inner[:mid], inner[mid+1:]
            return _fill_bits(sk1) + _fill_bits(sk2)
        return _math.log2(330) * 2   # fallback
    return _math.log2(330)   # unknown: charge max

_N_LEAVES = 330  # total concrete programs at depth-0
_CONCRETE_BITS = _math.log2(_N_LEAVES)  # bits to specify one concrete program

def abstraction_sleep_holed(all_solved_records, library):
    """Anti-unification: group by skeleton type with NON-TAUTOLOGY MDL.

    Non-tautology guard (Kai #11768): skeleton macro forms ONLY if
    cost(skeleton) + sum_task(fill_bits) < cost(concrete programs stored separately).
    Without this: any two MAP_RECOLOR tasks trivially share the skeleton.

    Bit model:
      concrete_baseline_bits (per task) = log2(n_leaves) = log2(330)
      skeleton_bits = log2(n_skeleton_types) = log2(14)
      fill_bits_per_task = log2(n_fills_for_this_skeleton)
      net_gain = occ * concrete_baseline_bits - (skeleton_bits + occ * fill_bits_per_task)
    """
    new_lib = dict(library)
    n_skeletons = len(SKELETON_ORDER)
    skeleton_bits = _math.log2(max(n_skeletons, 1))

    # Group by (skeleton_type, unique_task_id)
    skel_task_set = collections.defaultdict(set)
    skel_recs     = collections.defaultdict(list)
    for rec in all_solved_records:
        sk = rec.get('skel_type') or prog_skeleton_type(rec['prog'])
        skel_task_set[sk].add(rec['task_id'])
        skel_recs[sk].append(rec)

    # Concrete-level grouping for comparison
    concrete_counts = collections.Counter(rec['prog_str'] for rec in all_solved_records)

    funnel_candidates = []
    accepted = 0

    for sk in sorted(skel_task_set.keys()):
        task_ids = skel_task_set[sk]
        occ = len(task_ids)
        is_compose = sk.startswith('COMPOSE')

        # Fill-cost MDL (non-tautology guard)
        fill_b = _fill_bits(sk)
        concrete_baseline_bits = occ * _CONCRETE_BITS
        storage_cost = skeleton_bits + occ * fill_b
        net_gain = concrete_baseline_bits - storage_cost

        # MDL acceptance rule (must also satisfy occ >= MDL_MIN_OCC)
        if not is_compose:
            decision = 'LEAF_REJECTED_NOT_COMPOSE'
            reason = 'Leaf skeleton: concrete representation is already minimal; no structure to share.'
        elif occ < MDL_MIN_OCC:
            decision = 'COMPOSE_SKIPPED_OCC_LOW'
            reason = f'occ={occ} < MDL_MIN_OCC={MDL_MIN_OCC}; cannot form macro.'
        elif net_gain <= 0:
            decision = 'COMPOSE_REJECTED_TAUTOLOGY'
            reason = (f'net_gain={net_gain:.2f} <= 0: fill cost ({occ}*{fill_b:.1f}={occ*fill_b:.1f}b) '
                      f'+ skeleton ({skeleton_bits:.1f}b) >= concrete baseline ({concrete_baseline_bits:.1f}b). '
                      f'Macro would not compress; skeleton repetition is TAUTOLOGICAL.')
        else:
            decision = 'COMPOSE_ACCEPTED'
            reason = (f'net_gain={net_gain:.2f}b > 0: compresses {occ} tasks. '
                      f'skeleton={skeleton_bits:.1f}b + fills={occ*fill_b:.1f}b < concrete={concrete_baseline_bits:.1f}b.')
            accepted += 1
            # Store accepted macro in library with provenance (source_task_ids, source_depths)
            src_recs = skel_recs[sk]
            src_task_ids = sorted(set(r['task_id'] for r in src_recs))
            src_depths   = sorted(set(r['depth'] for r in src_recs))
            macro_key = f'skel_macro_{sk}'
            new_lib[macro_key] = {
                'skel_type': sk, 'occurrences': occ,
                'source_task_ids': src_task_ids, 'source_depths': src_depths,
                'net_gain': round(net_gain, 2),
                'skeleton_bits': round(skeleton_bits, 2),
                'fill_bits_per_task': round(fill_b, 2),
                'total_fill_bits': round(occ * fill_b, 2),
                'concrete_baseline_bits': round(concrete_baseline_bits, 2),
            }

        funnel_candidates.append({
            'skel_type': sk,
            'unique_task_count': occ,
            'is_compose': is_compose,
            'skeleton_bits': round(skeleton_bits, 2),
            'fill_bits_per_task': round(fill_b, 2),
            'total_fill_bits': round(occ * fill_b, 2),
            'concrete_baseline_bits': round(concrete_baseline_bits, 2),
            'net_gain': round(net_gain, 2),
            'decision': decision,
            'reason': reason,
            'task_ids_sample': sorted(task_ids)[:5],
        })

    all_depth_vals = [rec['depth'] for rec in all_solved_records]
    all_depth0     = all(d == 0 for d in all_depth_vals)

    note_parts = []
    if all_depth0:
        note_parts.append(
            'All solved programs at depth-0. No compose skeletons in funnel: '
            'no depth-1 solves means no compose-skeleton candidates. '
            'Skeleton anti-unification finds nothing to anti-unify at compose level.'
        )
    if accepted == 0 and not all_depth0:
        note_parts.append(
            'Compose solutions found but non-tautology MDL rejects all skeleton macros '
            '(net_gain<=0: fill cost dominates skeleton reuse benefit).'
        )
    if accepted > 0:
        note_parts.append(
            f'{accepted} skeleton macro(s) accepted: net_gain>0 after fill-cost charging.'
        )
    concrete_occ_ge2 = {ps: cnt for ps, cnt in concrete_counts.items() if cnt >= MDL_MIN_OCC}
    note_parts.append(
        f'Concrete-level: {len(concrete_counts)} unique programs, '
        f'{len(concrete_occ_ge2)} with occ>={MDL_MIN_OCC} (concrete grouping, for comparison).'
    )

    return new_lib, {
        'total_unique_skeletons': len(skel_task_set),
        'concrete_unique_programs': len(concrete_counts),
        'concrete_occ_ge2_count': len(concrete_occ_ge2),
        'per_candidate': funnel_candidates,
        'accepted_skeleton_macros': accepted,
        'note': ' | '.join(note_parts) if note_parts else '',
    }

# ── Dream sleep and transfer test (identical to Stage 1c) ─────────────────────
def dream_sleep(library, task_sample_io, n_dreams=N_DREAMS):
    dl_weights = {name: 0 for name in library}
    for task_io in task_sample_io[:n_dreams]:
        for name, macro in library.items():
            prog = ('macro', name)
            for pair in task_io[:2]:
                try:
                    inp = np.array(pair['input'], dtype=np.int64)
                    out = eval_program(prog, inp, library)
                    if out is not None and not np.array_equal(inp, out):
                        dl_weights[name] += 1
                        break
                except Exception:
                    pass
    return dl_weights

def transfer_test(library, held_tasks, dl_weights=None, budget=3000):
    empty = {}
    bl_costs, bl_solved = [], 0
    lib_costs, lib_solved, lib_new = [], 0, 0
    macro_usage_map     = {}   # task_id -> [macro_names used in cheaper lib solution]
    selected_macro_count = 0   # tasks where lib solution used any macro
    selected_holed_count = 0   # tasks where lib solution used a HOLED (non-PRIM) macro

    # Determine which library keys are holed skeletons (vs concrete PRIM-only)
    holed_macro_keys = {k for k in library if k.startswith('skel_macro_')
                        and not library[k].get('skel_type', '').startswith('PRIM_')}

    for task in held_tasks:
        prog_b, cb, _, _ = search_task_holed(task['io'], empty, budget)
        bl_costs.append(cb)
        if prog_b: bl_solved += 1

        prog_l, cl, _, _ = search_task_holed(task['io'], library, budget, dl_weights)
        lib_costs.append(cl)
        is_cheaper = cl < cb  # library solution cost strictly cheaper than baseline
        new_solve  = prog_l is not None and prog_b is None
        if prog_l:
            lib_solved += 1
            if new_solve: lib_new += 1
        macros_used = prog_macro_names(prog_l) if prog_l else []
        holed_macros = [m for m in macros_used if m in holed_macro_keys]
        # Attribution gate: only record usage when library solution is actually CHEAPER
        # "used somewhere" ≠ "used in a cheaper solution" — path-(ii) requires is_cheaper
        macro_usage_map[task['id']] = {
            'baseline_cost': cb, 'library_cost': cl,
            'is_cheaper': is_cheaper, 'new_solve': new_solve,
            'macros_used': macros_used, 'holed_macros_used': holed_macros,
        }
        if macros_used and is_cheaper:
            selected_macro_count += 1
            if holed_macros:
                selected_holed_count += 1

    bl_mean  = float(np.mean(bl_costs))
    lib_mean = float(np.mean(lib_costs))
    delta    = (lib_mean - bl_mean) / bl_mean * 100 if bl_mean else 0

    if not library:
        verdict = f"VACUOUS: library empty (delta={delta:+.1f}% is a no-op comparison)."
    elif delta < -TRANSFER_MARGIN:
        verdict = f"TRANSFER_GENUINE: delta={delta:+.1f}% < -{TRANSFER_MARGIN:.0f}%."
    else:
        verdict = f"HOLLOW: delta={delta:+.1f}% >= -{TRANSFER_MARGIN:.0f}%."

    return {
        'baseline': {'mean_cost': bl_mean, 'solved': bl_solved},
        'with_library': {'mean_cost': lib_mean, 'solved': lib_solved, 'new_solves': lib_new},
        'delta_pct': delta, 'verdict': verdict,
        'selected_macro_counts': selected_macro_count,
        'selected_holed_counts': selected_holed_count,
        'macro_usage_map': macro_usage_map,
    }

# ── MBPP (PRESERVED from Stage 1c) ───────────────────────────────────────────
def _mbpp_search(test_cases, budget=3000):
    import itertools
    MBPP_PRIMS = {
        'sort_asc':  lambda l: sorted(l),
        'sort_desc': lambda l: sorted(l, reverse=True),
        'reverse':   lambda l: list(reversed(l)),
        'unique':    lambda l: list(dict.fromkeys(l)),
        'cumsum':    lambda l: [sum(l[:i+1]) for i in range(len(l))],
        'sum_list':  lambda l: sum(l),
        'len_list':  lambda l: len(l),
        'max_list':  lambda l: max(l) if l else 0,
        'min_list':  lambda l: min(l) if l else 0,
        'id':        lambda l: list(l),
    }
    param_ops = (
        [(lambda n: lambda l: [x for x in l if x > n])(n) for n in range(0, 6)] +
        [(lambda n: lambda l: [x for x in l if x < n])(n) for n in range(1, 7)] +
        [(lambda n: lambda l: [x + n for x in l])(n) for n in range(1, 6)] +
        [(lambda n: lambda l: [x * n for x in l])(n) for n in range(2, 5)] +
        [(lambda n: lambda l: [x % n for x in l] if n else l)(n) for n in range(2, 5)] +
        [(lambda n: lambda l: [x ** n for x in l])(n) for n in [2, 3]] +
        [(lambda n: lambda l: [x for x in l if x != n])(n) for n in range(0, 5)] +
        [(lambda n: lambda l: [x for x in l if x == n])(n) for n in range(0, 5)]
    )
    all_fns = list(MBPP_PRIMS.values()) + param_ops
    nodes = 0
    for fn in all_fns:
        nodes += 1
        if nodes > budget: return False
        try:
            if all(fn(list(inp)) == exp for inp, exp in test_cases): return True
        except Exception: pass
    for f1 in all_fns:
        for f2 in all_fns:
            nodes += 1
            if nodes > budget: return False
            try:
                if all(f1(f2(list(inp))) == exp for inp, exp in test_cases): return True
            except Exception: pass
    return False

def eval_mbpp(mbpp_items):
    import re
    solved = 0
    for item in mbpp_items:
        cases = []
        for s in item.get('test_list', [])[:3]:
            m = re.match(r'assert\s+\w+\((\[.+?\])\s*\)\s*==\s*(.+)', s.strip())
            if not m: continue
            try:
                inp = eval(m.group(1))
                exp = eval(m.group(2))
                if isinstance(inp, list) and all(isinstance(x, int) for x in inp):
                    cases.append((inp, exp))
            except Exception: pass
        if cases and _mbpp_search(cases, BUDGET):
            solved += 1
    return solved

# ── Data loading ──────────────────────────────────────────────────────────────
def load_arc():
    tasks = []
    for sub in ['training', 'evaluation']:
        d = os.path.join(DATA_PATH, sub)
        if not os.path.isdir(d): continue
        for fname in sorted(os.listdir(d)):
            if not fname.endswith('.json'): continue
            with open(os.path.join(d, fname)) as f:
                data = json.load(f)
            pairs = [p for p in data.get('train', []) if p.get('input') and p.get('output')]
            if pairs:
                tasks.append({'id': fname[:-5], 'io': pairs, 'split_dir': sub})
    return tasks

def load_mbpp():
    try:
        from datasets import load_dataset
        return list(load_dataset('google-research-datasets/mbpp', 'sanitized', split='test'))[:N_MBPP]
    except Exception as e:
        print(f"  MBPP load skipped: {e}")
        return []

# ── Stage 1d helpers: subprogram containment (Leo #11818) ────────────────────
def concrete_programs_for_skeleton(sk):
    """All concrete programs with the given skeleton type (for near-miss exhaustive check)."""
    leaves = all_leaves_holed()
    if not sk.startswith('COMPOSE('):
        return [p for p in leaves if prog_skeleton_type(p) == sk]
    inner = sk[8:-1]  # strip COMPOSE( and )
    depth = 0
    split_idx = None
    for i, c in enumerate(inner):
        if c == '(': depth += 1
        elif c == ')': depth -= 1
        elif c == ',' and depth == 0:
            split_idx = i; break
    if split_idx is None:
        return []
    sk_left  = inner[:split_idx].strip()
    sk_right = inner[split_idx + 1:].strip()
    left_progs  = [p for p in leaves if prog_skeleton_type(p) == sk_left]
    right_progs = [p for p in leaves if prog_skeleton_type(p) == sk_right]
    return [('compose', lp, rp) for lp in left_progs for rp in right_progs]

def task_io_match(prog, task_io):
    """Check if program correctly solves all training I/O pairs."""
    for pair in task_io:
        try:
            inp = np.array(pair['input'], dtype=np.int64)
            out = np.array(pair['output'], dtype=np.int64)
            result = eval_program(prog, inp, {})
            if result is None or not np.array_equal(result, out):
                return False
        except Exception:
            return False
    return True

def check_exact_subprogram_hits(prog, s1c_library):
    """Return list of s1c macro names whose skel_type matches the program's skeleton."""
    if prog is None:
        return []
    sk = prog_skeleton_type(prog)
    return [name for name, info in s1c_library.items()
            if info.get('skel_type') == sk]

def check_near_miss_hits_detailed(task_io, task_id, s1c_library, skel_concrete_cache,
                                   bfs_depth, bfs_cost, bfs_skeleton):
    """Minimality-aware near-miss with full dominance evidence (Leo #11832, Kai run card).

    Returns (raw_hits, dominated_hits, non_dominated_hits, dominance_entries).
    Dominance: depth-first only unless a globally comparable near-miss cost is
    available. The concrete-program list index is preserved for audit, but it is
    not comparable to BFS nodes searched. Raw hits are preserved for audit.
    """
    raw_hits, dominated_hits, non_dominated_hits, dominance_entries = [], [], [], []

    for macro_name, macro_info in s1c_library.items():
        sk = macro_info.get('skel_type')
        if not sk:
            continue
        macro_depth = 1 if sk.startswith('COMPOSE(') else 0

        near_miss_candidate_index = None
        for j, prog in enumerate(skel_concrete_cache.get(sk, [])):
            if task_io_match(prog, task_io):
                near_miss_candidate_index = j
                break
        if near_miss_candidate_index is None:
            continue

        raw_hits.append(macro_name)

        if bfs_depth is not None and macro_depth > bfs_depth:
            verdict = 'DOMINATED'
            reason = (f"macro_depth ({macro_depth}) > bfs_depth ({bfs_depth}); "
                      f"over-expressive alternative to simpler BFS solution ({bfs_skeleton})")
            dominated_hits.append(macro_name)
        else:
            verdict = 'NON_DOMINATED'
            reason = (f"macro_depth ({macro_depth}) <= bfs_depth ({bfs_depth}); "
                      "near-miss concrete candidate index is not comparable to bfs_cost")
            non_dominated_hits.append(macro_name)

        dominance_entries.append({
            'task_id': task_id,
            'macro_name': macro_name,
            'near_miss_skeleton': sk,
            'near_miss_depth': macro_depth,
            'near_miss_candidate_index': near_miss_candidate_index,
            'near_miss_cost_unknown': True,
            'best_solution_skeleton': bfs_skeleton,
            'best_solution_depth': bfs_depth,
            'best_solution_cost': bfs_cost,
            'dominance_verdict': verdict,
            'dominance_reason': reason,
        })

    return raw_hits, dominated_hits, non_dominated_hits, dominance_entries

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import subprocess
    t0 = time.time()
    print("STAGE 1d B30000+MINIMALITY: SUBPROGRAM CONTAINMENT FOLLOW-UP (Leo #11832, Kai #11845)")
    print("PREMISE: Minimality-aware near-miss. Does B30000 confirm budget saturation?")
    print("FIXED Stage 1c library — NOT re-learning. Budget curve: 3000/6000/10000/30000.")
    print("=" * 70)
    sys.stdout.flush()

    base_commit = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        text=True).strip()

    # Load Stage 1c fixed library
    with open(S1C_RESULT_PATH) as f:
        s1c_result = json.load(f)
    s1c_library = s1c_result.get('library_final', {})
    print(f"\nStage 1c library: {len(s1c_library)} macro(s)")
    for name, info in s1c_library.items():
        print(f"  {name}: skel_type={info.get('skel_type')}, "
              f"occ={info.get('occurrences')}, net_gain={info.get('net_gain')}")
    sys.stdout.flush()

    # Pre-compute concrete programs for near-miss exhaustive check
    skel_concrete_cache = {}
    for name, info in s1c_library.items():
        sk = info.get('skel_type')
        if sk:
            progs = concrete_programs_for_skeleton(sk)
            skel_concrete_cache[sk] = progs
            print(f"  Near-miss candidates for {sk}: {len(progs)} programs")
    sys.stdout.flush()

    # Load held tasks — same split as Stage 1c (SPLIT_SEED=42)
    all_arc = load_arc()
    training = [t for t in all_arc if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(training))
    held_tasks = [training[i] for i in idx[:N_HELD]]
    held_ids   = [t['id'] for t in held_tasks]
    print(f"\nHeld tasks: {len(held_tasks)}")
    s1c_src = [info.get('source_task_ids', []) for info in s1c_library.values()]
    print(f"Stage 1c source tasks: {s1c_src}")

    dominance_table = []
    # Coverage curve: enumerate HELD tasks WITHOUT library at each budget
    budget_results = []
    for budget in BUDGET_POINTS:
        print(f"\n{'='*50}")
        n_leaves = len(all_leaves_holed())
        n_d1 = n_leaves * n_leaves
        d1_cov = max(0, min(budget - n_leaves, n_d1))
        print(f"BUDGET = {budget}  ({d1_cov/n_d1*100:.1f}% depth-1)")
        t_b = time.time()
        sys.stdout.flush()

        per_task = {}
        d0_count = 0
        d1_count = 0
        exact_hits_total = 0
        raw_nm_total = 0
        dom_nm_total = 0
        nondom_nm_total = 0

        for i, task in enumerate(held_tasks):
            prog, cost, depth, sk = search_task_holed(task['io'], {}, budget)
            exact_hits = check_exact_subprogram_hits(prog, s1c_library)
            # near_miss_cost (index in skel list) != bfs nodes — pass bfs_cost=None, depth-only dominance
            raw_nm, dom_nm, nondom_nm, dom_entries = (
                check_near_miss_hits_detailed(task['io'], task['id'], s1c_library,
                                              skel_concrete_cache,
                                              bfs_depth=depth, bfs_cost=None,
                                              bfs_skeleton=sk)
                if prog is not None and not exact_hits else ([], [], [], []))
            dominance_table.extend(dom_entries)
            per_task[task['id']] = {
                'solved': prog is not None,
                'depth': depth if prog is not None else -1,
                'skeleton_type': sk,
                'exact_subprogram_hits': exact_hits,
                'raw_near_miss_hits': raw_nm,
                'dominated_near_miss_hits': dom_nm,
                'non_dominated_near_miss_hits': nondom_nm,
            }
            if prog is not None:
                if depth == 0: d0_count += 1
                elif depth == 1: d1_count += 1
                if exact_hits: exact_hits_total += 1
                if raw_nm: raw_nm_total += 1
                if dom_nm: dom_nm_total += 1
                if nondom_nm: nondom_nm_total += 1
            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{len(held_tasks)} tasks, elapsed {time.time()-t_b:.0f}s")
                sys.stdout.flush()

        n_solved = d0_count + d1_count
        coverage_pct = n_solved / len(held_tasks) * 100
        elapsed_b = time.time() - t_b
        print(f"  Held solved: {n_solved}/{len(held_tasks)} ({coverage_pct:.1f}%) "
              f"— depth-0={d0_count}, depth-1={d1_count}")
        print(f"  Exact subprogram hits: {exact_hits_total} tasks "
              f"({exact_hits_total/max(n_solved,1)*100:.0f}% of solved)")
        print(f"  Near-miss (raw/dom/nondom): {raw_nm_total}/{dom_nm_total}/{nondom_nm_total}")
        print(f"  Elapsed: {elapsed_b:.0f}s")
        sys.stdout.flush()

        budget_results.append({
            'budget': budget,
            'held_solved': n_solved,
            'depth_0': d0_count,
            'depth_1': d1_count,
            'coverage_pct': coverage_pct,
            'exact_subprogram_hits_total': exact_hits_total,
            'raw_near_miss_hits_total': raw_nm_total,
            'dominated_near_miss_hits_total': dom_nm_total,
            'non_dominated_near_miss_hits_total': nondom_nm_total,
            'elapsed_sec': elapsed_b,
            'per_task': per_task,
        })

    # ── PREMISE_* classification (Leo #11832, Kai #11845) ─────────────────────
    top = budget_results[-1]
    n_solved_top    = top['held_solved']
    exact_top       = top['exact_subprogram_hits_total']
    nondom_nm_top   = top['non_dominated_near_miss_hits_total']
    raw_nm_top_val  = top['raw_near_miss_hits_total']
    dom_nm_top_val  = top['dominated_near_miss_hits_total']

    # Budget-saturation: curve flat means representational ceiling, not search-depth ceiling
    curve_solved = [b['held_solved'] for b in budget_results]
    curve_is_flat = len(curve_solved) >= 2 and (max(curve_solved) == min(curve_solved))
    # Coverage increase at B30000 vs earlier budgets
    prev_solved = curve_solved[:-1]
    coverage_increased_at_top = (n_solved_top > max(prev_solved)) if prev_solved else False
    coverage_saturation = curve_is_flat

    if n_solved_top < COVERAGE_THRESHOLD:
        premise_class = "PREMISE_30000_INSTRUMENTATION_INCOMPLETE"
        premise_reason = (
            f"At max budget={BUDGET_POINTS[-1]}, only {n_solved_top}/{N_HELD} held tasks solved "
            f"({n_solved_top/N_HELD*100:.1f}%): fewer than threshold={COVERAGE_THRESHOLD}. "
            f"Insufficient held coverage to make subprogram test conclusive."
        )
    elif coverage_increased_at_top:
        premise_class = "PREMISE_30000_COVERAGE_INCREASED"
        premise_reason = (
            f"Coverage increased at B{BUDGET_POINTS[-1]}: {n_solved_top}/{N_HELD} "
            f"({n_solved_top/N_HELD*100:.1f}%) vs prior max {max(prev_solved)}/{N_HELD}. "
            f"Gate: exact_hits={exact_top}, non_dominated_near_miss={nondom_nm_top}. "
            f"Budget not yet saturated; inspect exact and non-dominated near-miss before branching."
        )
    elif exact_top > 0:
        premise_class = "PREMISE_SELECTION_BOTTLENECK"
        premise_reason = (
            f"Stage 1c macros appear as exact subprograms in {exact_top} held solutions "
            f"at budget={BUDGET_POINTS[-1]}. Transfer at budget=3000 was HOLLOW "
            f"because the proposer couldn't SELECT these macros within transfer budget. "
            f"Stage 2 proposer warranted."
        )
    elif nondom_nm_top > 0:
        premise_class = "PREMISE_NONDOMINATED_NEAR_MISS"
        premise_reason = (
            f"No exact subprogram hits at max budget (adequate coverage: "
            f"{n_solved_top}/{N_HELD} = {n_solved_top/N_HELD*100:.1f}%). "
            f"Non-dominated semantic near-misses: {nondom_nm_top} held tasks "
            f"(raw={raw_nm_top_val}, dominated={dom_nm_top_val}). "
            f"Macro depth <= BFS solution depth; concrete program with Stage 1c macro "
            f"skeleton solves the task. See dominance_table for per-candidate evidence."
        )
    elif curve_is_flat:
        premise_class = "PREMISE_BUDGET_SATURATED_FIXED_GRAMMAR"
        premise_reason = (
            f"Coverage flat at {n_solved_top}/{N_HELD} = {n_solved_top/N_HELD*100:.1f}% "
            f"across all {len(BUDGET_POINTS)} budget points "
            f"({BUDGET_POINTS[0]}→{BUDGET_POINTS[-1]}). Grammar is budget-saturated: "
            f"additional budget ({BUDGET_POINTS[-1]//BUDGET_POINTS[0]}×) finds no new held tasks. "
            f"Exact hits=0. Non-dominated near-miss=0 (raw={raw_nm_top_val}, "
            f"all dominated by simpler BFS solutions). "
            f"Fixed holed grammar representationally bound. More budget is the wrong lever. "
            f"Move to net-free matched-eval gate (Kai #11845)."
        )
    else:
        premise_class = "PREMISE_BUDGET_SATURATED_FIXED_GRAMMAR"
        premise_reason = (
            f"Adequate held coverage ({n_solved_top}/{N_HELD} = "
            f"{n_solved_top/N_HELD*100:.1f}% at max budget={BUDGET_POINTS[-1]}). "
            f"Stage 1c macros: exact hits=0, non-dominated near-miss=0. "
            f"Fixed holed grammar premise FALSE with mechanism evidence."
        )

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"PREMISE CLASSIFICATION: {premise_class}")
    print(f"REASON: {premise_reason}")
    print(f"Elapsed: {total:.1f}s")
    sys.stdout.flush()

    result = {
        'experiment': 'stage1d-premise-truth-b30000-minimality-follow-up',
        'note': ('Leo #11832, Kai #11845: minimality-aware near-miss follow-up. '
                 'B30000 confirm point. Original artifact preserved as base_artifact. '
                 'PREMISE_* labels per run card.'),
        'base_artifact': BASE_ARTIFACT,
        'base_commit': base_commit,
        'config': {
            'budget_points': BUDGET_POINTS,
            'n_held': N_HELD,
            'split_seed': SPLIT_SEED,
            'coverage_threshold': COVERAGE_THRESHOLD,
        },
        's1c_library': s1c_library,
        's1c_macro_skel_types': {
            name: info.get('skel_type') for name, info in s1c_library.items()
        },
        's1c_macro_source_tasks': {
            name: info.get('source_task_ids', []) for name, info in s1c_library.items()
        },
        'held_task_ids': held_ids,
        'near_miss_candidate_counts': {
            sk: len(progs) for sk, progs in skel_concrete_cache.items()
        },
        'budget_curve': budget_results,
        'slope_analysis': {
            'budgets': BUDGET_POINTS,
            'coverage_pcts': [b['coverage_pct'] for b in budget_results],
            'exact_hits_per_budget': [b['exact_subprogram_hits_total'] for b in budget_results],
            'raw_near_miss_per_budget': [b['raw_near_miss_hits_total'] for b in budget_results],
            'dominated_near_miss_per_budget': [b['dominated_near_miss_hits_total'] for b in budget_results],
            'non_dominated_near_miss_per_budget': [b['non_dominated_near_miss_hits_total'] for b in budget_results],
        },
        'coverage_saturation': coverage_saturation,
        'dominance_table': dominance_table,
        'premise_classification_corrected': premise_class,
        'premise_reason': premise_reason,
        'total_elapsed_sec': total,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Written: {RESULT_PATH}")

if __name__ == '__main__':
    main()
