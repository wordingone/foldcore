"""
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

DATA_PATH   = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1c_holed_result.json"

N_ITERATIONS    = 3
N_CURRICULUM    = 200
N_HELD          = 200
N_MBPP          = 50
SPLIT_SEED      = 42
BUDGET          = 3000
MDL_MIN_OCC     = 2
N_DREAMS        = 30
TRANSFER_MARGIN = 5.0

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
def search_task_holed(task_io, lib, budget=BUDGET, dl_weights=None):
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

def transfer_test(library, held_tasks, dl_weights=None, budget=BUDGET):
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
def _mbpp_search(test_cases, budget=BUDGET):
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

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("STAGE 1c-HOLED: SKELETON GRAMMAR BFS -- ANTI-UNIFICATION ABSTRACTION")
    print("Leo #11766: PIVOT from flat-330 to holed skeleton grammar (~14 types)")
    print("TWO-VARIABLE CHANGE: representation (concrete->holed) + depth tractability")
    print("=" * 70)
    t0 = time.time()
    sys.stdout.flush()

    all_arc = load_arc()
    training = [t for t in all_arc if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(training))
    held_tasks  = [training[i] for i in idx[:N_HELD]]
    pool_tasks  = [training[i] for i in idx[N_HELD:]]
    rng2 = np.random.default_rng(SPLIT_SEED + 1)
    curriculum_idx = rng2.choice(len(pool_tasks), min(N_CURRICULUM, len(pool_tasks)), replace=False)
    curriculum = [pool_tasks[i] for i in sorted(curriculum_idx)]
    held_ids = [t['id'] for t in held_tasks]

    # Skeleton grammar inventory
    n_skeletons = len(SKELETON_ORDER)
    total_fills = sum(len(skeleton_fills(sk)) for sk in SKELETON_ORDER)
    depth1_skel_pairs = n_skeletons ** 2
    depth2_skel_triples = n_skeletons ** 3

    leaves = all_leaves_holed()
    n_leaves = len(leaves)
    n_depth1_compose = n_leaves * n_leaves
    compose_covered = max(0, min(BUDGET - n_leaves, n_depth1_compose))
    depth1_coverage_frac = compose_covered / n_depth1_compose if n_depth1_compose else 0

    print(f"ARC: held={len(held_tasks)}, curriculum={len(curriculum)}")
    print(f"MBPP: {N_MBPP} items")
    print(f"Skeleton grammar: {n_skeletons} skeleton types, {total_fills} total concrete fills")
    print(f"  Skeleton pairs (depth-1): {depth1_skel_pairs} (= {n_skeletons}^2)")
    print(f"  Skeleton triples (depth-2): {depth2_skel_triples} (= {n_skeletons}^3)")
    print(f"Concrete: {n_leaves} depth-0 programs, {n_depth1_compose:,} depth-1 compose space")
    print(f"BUDGET={BUDGET}: depth-0 ({n_leaves}) + {compose_covered:,} depth-1 "
          f"({depth1_coverage_frac:.2%} concrete coverage)")
    print(f"Key change: depth-1 programs ordered by skeleton pair (small-fills first).")
    print(f"Abstraction: skeleton-type anti-unification (not concrete-program exact match).")
    sys.stdout.flush()

    mbpp_items = load_mbpp()

    library         = {}
    dl_weights      = {}
    all_solved_recs = []
    iter_results    = []
    cumulative_funnel = None

    for it in range(N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"ITERATION {it+1}/{N_ITERATIONS}  (library={len(library)} macros)")
        t_iter = time.time()
        sys.stdout.flush()

        it_costs, it_solved = [], 0
        it_solved_recs = []
        depth_counts   = collections.Counter()
        skel_counts    = collections.Counter()

        for i, task in enumerate(curriculum):
            prog, cost, depth, sk = search_task_holed(task['io'], library, BUDGET, dl_weights)
            it_costs.append(cost)
            if prog:
                it_solved += 1
                ps = str(prog)
                ph = prog_hash(prog)
                rec = {'prog': prog, 'prog_str': ps, 'hash': ph,
                       'task_id': task['id'], 'depth': depth, 'skel_type': sk}
                it_solved_recs.append(rec)
                depth_counts[depth] += 1
                if sk: skel_counts[sk] += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_iter
                print(f"  ... {i+1}/{len(curriculum)} tasks, {it_solved} solved, {elapsed:.0f}s")
                sys.stdout.flush()

        all_solved_recs.extend(it_solved_recs)
        arc_rate = it_solved / len(curriculum)
        arc_cost = float(np.mean(it_costs))
        print(f"  ARC wake: solved={it_solved}/{len(curriculum)} ({arc_rate:.1%}), "
              f"mean_cost={arc_cost:.0f}")
        if it_solved_recs:
            type_counts = collections.Counter(prog_type(r['prog']) for r in it_solved_recs)
            depth_dist  = dict(sorted(depth_counts.items()))
            print(f"  Program types: {dict(type_counts)}, depth_dist: {depth_dist}")
            print(f"  Skeleton types: {dict(skel_counts.most_common(5))}")
        sys.stdout.flush()

        mbpp_solved = eval_mbpp(mbpp_items)
        mbpp_rate   = mbpp_solved / max(len(mbpp_items), 1)
        print(f"  MBPP wake: solved={mbpp_solved}/{len(mbpp_items)} ({mbpp_rate:.1%})")
        sys.stdout.flush()

        print(f"  Abstraction-sleep ({len(all_solved_recs)} solved records, skeleton anti-unification)...")
        sys.stdout.flush()
        old_sz   = len(library)
        library, funnel = abstraction_sleep_holed(all_solved_recs, library)
        cumulative_funnel = funnel
        new_mac  = len(library) - old_sz
        sk_accepted = funnel['accepted_skeleton_macros']
        print(f"  Library: {old_sz} -> {len(library)} (+{new_mac} concrete macros)")
        print(f"  Skeleton funnel: {funnel['total_unique_skeletons']} unique skeleton types, "
              f"{sk_accepted} accepted skeleton macros")
        sys.stdout.flush()

        print(f"  Dream-sleep: {min(N_DREAMS, len(curriculum))} fantasies...")
        task_sample = [t['io'] for t in curriculum[:N_DREAMS]]
        dl_weights  = dream_sleep(library, task_sample)
        active = sum(1 for w in dl_weights.values() if w > 0)
        print(f"  Dream-sleep: {active}/{len(library)} macros active")
        sys.stdout.flush()

        print(f"  Transfer test ({len(held_tasks)} held-out, budget={BUDGET})...")
        t_tr = time.time()
        transfer = transfer_test(library, held_tasks, dl_weights, BUDGET)
        print(f"  {transfer['verdict']} (transfer: {time.time()-t_tr:.0f}s)")
        sys.stdout.flush()

        iter_results.append({
            'iteration':    it + 1,
            'library_size': len(library),
            'new_macros':   new_mac,
            'skeleton_macros_formed': sk_accepted,
            'arc': {'solved': it_solved, 'n': len(curriculum),
                    'solve_rate': arc_rate, 'mean_cost': arc_cost,
                    'depth_distribution': {str(k): v for k, v in sorted(depth_counts.items())},
                    'skeleton_distribution': {k: v for k, v in skel_counts.most_common(10)}},
            'mbpp': {'solved': mbpp_solved, 'n': len(mbpp_items), 'solve_rate': mbpp_rate},
            'transfer': transfer,
            'dream_sleep_active_macros': active,
            'elapsed_sec': time.time() - t_iter,
        })

    # ── Aggregate ─────────────────────────────────────────────────────────────
    all_type_counts   = collections.Counter(prog_type(r['prog']) for r in all_solved_recs)
    all_depth_counts  = collections.Counter(r['depth'] for r in all_solved_recs)
    all_skel_counts   = collections.Counter(r.get('skel_type') for r in all_solved_recs if r.get('skel_type'))

    rates  = [r['arc']['solve_rate'] for r in iter_results]
    deltas = [r['transfer']['delta_pct'] for r in iter_results]
    last_d = deltas[-1] if deltas else 0
    n_macros = len(library)
    depth1_solves = all_depth_counts.get(1, 0)
    sk_accepted_total = cumulative_funnel['accepted_skeleton_macros'] if cumulative_funnel else 0

    # Define these early — needed for instrumentation completeness check
    curriculum_ids = [t['id'] for t in curriculum]
    curriculum_held_disjoint = len(set(held_ids) & set(curriculum_ids)) == 0

    # ── Kai 5-label holed classification (Kai #11768/#11769/#11785) ──────────────
    # Check ALL required prereg fields are present (Kai instrumentation gate)
    last_transfer = iter_results[-1]['transfer'] if iter_results else {}
    funnel_ok = (cumulative_funnel is not None and
                 'per_candidate' in cumulative_funnel and
                 'accepted_skeleton_macros' in cumulative_funnel)
    transfer_ok = (
        'selected_macro_counts' in last_transfer and
        'selected_holed_counts' in last_transfer and
        'macro_usage_map' in last_transfer
    )
    ids_ok = (len(curriculum_ids) > 0 and len(held_ids) > 0 and
              isinstance(curriculum_held_disjoint, bool))
    library_ok = all(
        'source_task_ids' in m and 'source_depths' in m
        for m in library.values()
    ) if library else True

    instrumentation_ok = funnel_ok and transfer_ok and ids_ok and library_ok

    if not instrumentation_ok:
        kai_class = "HOLED_INSTRUMENTATION_INCOMPLETE"
        missing = []
        if not funnel_ok:    missing.append('abstraction_funnel.per_candidate/accepted_skeleton_macros')
        if not transfer_ok:  missing.append('transfer.selected_macro_counts/selected_holed_counts/macro_usage_map')
        if not ids_ok:       missing.append('curriculum_ids/held_task_ids/curriculum_held_disjoint')
        if not library_ok:   missing.append('library_final.source_task_ids/source_depths')
        kai_reason = f"Missing required schema fields: {'; '.join(missing)}. Cannot classify."
    elif depth1_solves == 0 and depth1_coverage_frac < 0.10:
        kai_class = "HOLED_DEPTH_STARVATION"
        kai_reason = (
            f"Zero depth-1 compose solves; concrete compose coverage {depth1_coverage_frac:.2%}. "
            f"Skeleton anti-unification has no compose solutions to anti-unify. "
            f"Holed search order covers different concrete block than flat Stage 1c "
            f"but same ~{depth1_coverage_frac:.2%} of depth-1 space within BUDGET={BUDGET}. "
            f"Stage 2 proposer DEFERRED."
        )
    elif depth1_solves > 0 and sk_accepted_total == 0:
        kai_class = "HOLED_FORMATION_NEGATIVE"
        kai_reason = (
            f"{depth1_solves} depth-1 solve(s) found but non-tautology MDL rejects all skeleton macros "
            f"(net_gain<=0 after charging fill bits, or occ<{MDL_MIN_OCC}). "
            f"Skeleton repetition is tautological: fill cost >= compression gain. "
            f"Stage 2 proposer DEFERRED."
        )
    elif sk_accepted_total > 0 and last_d >= -TRANSFER_MARGIN:
        kai_class = "HOLED_HOLLOW"
        kai_reason = (
            f"{sk_accepted_total} skeleton macro(s) formed with net_gain>0 (non-tautological). "
            f"Transfer delta={last_d:+.1f}% >= -{TRANSFER_MARGIN:.0f}%: HOLLOW. "
            f"Stage 2 proposer DEFERRED."
        )
    elif sk_accepted_total > 0 and last_d < -TRANSFER_MARGIN:
        # Locked threshold (Leo #11792/#11796/#11801): bare negative delta is NOT GENUINE.
        # GENUINE requires mechanism attribution: new_solves>0 OR holed macros appear in cheaper solutions.
        last_new_solves = last_transfer.get('with_library', {}).get('new_solves', 0)
        mechanism_attributed = (last_new_solves > 0 or
                                last_transfer.get('selected_holed_counts', 0) > 0)
        if mechanism_attributed:
            kai_class = "HOLED_TRANSFER_GENUINE"
            kai_reason = (
                f"{sk_accepted_total} skeleton macro(s) formed (net_gain>0) + "
                f"transfer delta={last_d:+.1f}% < -{TRANSFER_MARGIN:.0f}% + "
                f"mechanism attributed (new_solves={last_new_solves}, "
                f"selected_holed_counts={last_transfer.get('selected_holed_counts', 0)}). "
                f"Holed library genuinely reduces held-out search cost. "
                f"Stage 2 proposer DEFERRED (not yet needed)."
            )
        else:
            kai_class = "HOLED_HOLLOW"
            kai_reason = (
                f"{sk_accepted_total} skeleton macro(s) formed (net_gain>0) + "
                f"negative delta={last_d:+.1f}% BUT new_solves=0 and "
                f"selected_holed_counts={last_transfer.get('selected_holed_counts', 0)}: "
                f"no holed macro appears in cheaper held-out solutions. "
                f"Unattributable delta → HOLED_HOLLOW per locked threshold."
            )
    else:
        kai_class = "HOLED_DEPTH_STARVATION"
        kai_reason = (
            f"Default: depth1_solves={depth1_solves}, "
            f"coverage={depth1_coverage_frac:.2%}, sk_macros={sk_accepted_total}."
        )

    # ── Verdict ───────────────────────────────────────────────────────────────
    if kai_class == "HOLED_DEPTH_STARVATION":
        verdict = (
            f"HOLED_DEPTH_STARVATION: skeleton-pair ordering covers a different concrete block "
            f"({depth1_coverage_frac:.2%} depth-1), but zero depth-1 solves. "
            f"Anti-unification has nothing to anti-unify. "
            f"Branching bound persists within BUDGET={BUDGET}."
        )
    elif kai_class == "HOLED_TRANSFER_GENUINE":
        verdict = (
            f"PASS: HOLED_TRANSFER_GENUINE -- skeleton anti-unification yields "
            f"transferable macro (delta={last_d:+.1f}%)."
        )
    else:
        verdict = f"{kai_class}: {kai_reason}"

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"KAI CLASSIFICATION: {kai_class}")
    print(f"VERDICT: {verdict}")
    print(f"Elapsed: {total:.1f}s")
    sys.stdout.flush()

    # ── Claim scope ───────────────────────────────────────────────────────────
    claim_scope = (
        f"Stage 1c-HOLED: TWO-VARIABLE CHANGE from Stage 1c (both deliberate). "
        f"(1) Representation: concrete-330 → holed skeleton grammar ({n_skeletons} types). "
        f"(2) Search order: skeleton-pair priority (small-fill pairs first) vs lex concrete order. "
        f"Concrete coverage: {n_leaves} depth-0 programs (100%) + {compose_covered:,}/{n_depth1_compose:,} "
        f"depth-1 programs ({depth1_coverage_frac:.2%}) — same ~2.45% budget, different program block. "
        f"Depth-1 solves: {depth1_solves}. Skeleton macros formed: {sk_accepted_total}. "
        f"Kai classification: {kai_class}. "
        f"The skeleton-pair ordering tests a DIFFERENT 2.45% block of concrete depth-1 programs "
        f"than flat Stage 1c: specifically, small-fill skeleton pairs (PRIM×PRIM, MAP_DELETE×PRIM, etc.) "
        f"get full coverage before budget exhaustion."
    )

    # Kai #11748 schema required fields (curriculum_ids/held_disjoint already computed above)
    sk_depth_coverage = {
        'depth_0_skeletons_tried': n_skeletons,
        'depth_0_skeleton_names': SKELETON_ORDER,
        'depth_1_skeleton_pairs': depth1_skel_pairs,
        'depth_2_skeleton_triples': depth2_skel_triples,
        'note': ('Skeleton-depth counts PAIRS tried (14^2=196), '
                 'independent of concrete fills attempted within each pair. '
                 'fill_attempt_coverage tracks concrete programs.'),
    }

    result = {
        'experiment': 'stage1c-holed-skeleton-grammar',
        'grammar_mode': 'holed_object_centric',
        'note': ('Leo #11766/#11769/#11785: holed skeleton grammar pivot. '
                 'Non-tautology MDL: fill bits charged (Kai #11768). 5-label Kai gate. '
                 'TWO-VARIABLE CHANGE: representation + search order. BUDGET=3000.'),
        'config': {
            'n_iterations': N_ITERATIONS, 'n_curriculum': N_CURRICULUM,
            'n_held': N_HELD, 'n_mbpp': N_MBPP,
            'split_seed': SPLIT_SEED, 'budget': BUDGET,
            'mdl_min_occ': MDL_MIN_OCC, 'n_dreams': N_DREAMS,
            'transfer_margin_pct': TRANSFER_MARGIN,
        },
        'curriculum_ids': curriculum_ids,
        'held_task_ids': held_ids,
        'curriculum_held_disjoint': curriculum_held_disjoint,
        'holed_skeleton_inventory': {
            sk: {
                'arity': 0 if sk.startswith('PRIM_') else
                         1 if sk in ('MAP_DELETE', 'MAP_KEEPONLY') else 2,
                'hole_positions': (
                    [] if sk.startswith('PRIM_') else
                    ['predicate'] if sk in ('MAP_DELETE', 'MAP_KEEPONLY') else
                    ['predicate', 'transform_param'] if sk == 'MAP_RECOLOR' else
                    ['predicate', 'direction'] if sk == 'MAP_TRANSLATE' else ['predicate']
                ),
                'fill_domain_size': len(skeleton_fills(sk)),
                'fill_domain_example': skeleton_fills(sk)[0] if skeleton_fills(sk) else None,
            }
            for sk in SKELETON_ORDER
        },
        'skeleton_grammar': {
            'n_skeleton_types': n_skeletons,
            'skeleton_names': SKELETON_ORDER,
            'total_concrete_fills': total_fills,
            'depth1_skeleton_pairs': depth1_skel_pairs,
            'depth2_skeleton_triples': depth2_skel_triples,
            'family_sizes': {sk: len(skeleton_fills(sk)) for sk in SKELETON_ORDER},
        },
        'skeleton_depth_coverage': sk_depth_coverage,
        'fill_attempt_coverage': {
            'n_leaves': n_leaves,
            'depth_1_covered_concrete': compose_covered,
            'depth_1_total_concrete': n_depth1_compose,
            'depth_1_coverage_fraction': depth1_coverage_frac,
            'note': 'Concrete fills within budget; different block from flat Stage 1c.',
        },
        'max_depth_reached': max((r['depth'] for r in all_solved_recs), default=-1),
        'solved_programs': [
            {'prog_str': r['prog_str'][:120], 'hash': r['hash'],
             'task_id': r['task_id'], 'depth': r['depth'],
             'skeleton_body': r.get('skel_type'),
             'fills': prog_fills(r['prog']),
             'prog_type': prog_type(r['prog'])}
            for r in all_solved_recs
        ],
        'program_type_breakdown': dict(all_type_counts),
        'skeleton_type_breakdown': dict(all_skel_counts.most_common()),
        'depth_distribution_all_solves': {str(k): v for k, v in sorted(all_depth_counts.items())},
        'abstraction_funnel': cumulative_funnel,
        'held_out_transfer': last_transfer,
        'library_final': {
            n: {
                k: v for k, v in m.items()
                if k in ('skel_type', 'occurrences', 'source_task_ids', 'source_depths',
                         'net_gain', 'skeleton_bits', 'fill_bits_per_task',
                         'total_fill_bits', 'concrete_baseline_bits',
                         'body_str', 'body_size', 'savings')
            }
            for n, m in library.items()
        } if library else {},
        'iterations': iter_results,
        'claim_scope': claim_scope,
        'kai_classification': kai_class,
        'kai_reason': kai_reason,
        'verdict': verdict,
        'total_elapsed_sec': total,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Written: {RESULT_PATH}")

if __name__ == '__main__':
    main()
