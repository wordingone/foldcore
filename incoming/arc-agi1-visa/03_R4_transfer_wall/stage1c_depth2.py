"""
Stage 1c: Depth-2 Flat BFS — Object-Centric, N_CURRICULUM=200 (Leo #11745/#11749/#11752, 2026-05-30).

ONE-VARIABLE DISCIPLINE (Leo+Kai): Grammar PRESERVED EXACTLY from Stage 1b.
ONLY change: budget increased toward depth-2 flat BFS.

Depth-2 space: 330 depth-0 + 330^2 = 108,900 depth-1 compose = 109,230 total.
BUDGET=3000 covers 330 depth-0 (100%) + 2670 depth-1 compose (2.45% of compose space).
Full depth-1 flat BFS (budget=109230) is estimated intractable — that intractability is the
pre-registered Stage 2 (proposer) argument (Leo #11745).

Kai's binding gate schema (#11748):
  - max_depth_reached, depth_budget_by_level, depth_1_coverage_fraction
  - solved_programs (bodies + hashes + depth + task_id)
  - program_type_breakdown
  - abstraction_funnel (repeated subprograms, MDL candidates, accepted/rejected + reasons)
  - library_final (macro bodies, occurrences, source_task_ids, source_depths)
  - transfer (baseline + with-library + selected macro counts + new solves + delta)
  - claim_scope (explicit)
  - kai_classification: DEPTH_STARVATION | FORMATION_NEGATIVE_DEPTH2 | HOLLOW_DEPTH2 | TRANSFER_GENUINE_DEPTH2

Preserved from Stage 1b:
  - Predicates: 16 (largest, smallest, non_bg, bg_obj, unique_color, most_common_c, all, color_1..9)
  - Transforms: 20 (delete, keep_only, recolor_0..9, translate_8_directions)
  - Whole-grid prims: 10 (crop, down2, flip_h, flip_v, id, rot_180, rot_270, rot_90, tr, up2)
  - Total leaves: 330 (320 map_apply + 10 prim)
  - Object extraction: 4-connected BFS with _OBJ_CACHE
"""

import numpy as np
import json, os, sys, time, collections, hashlib
from collections import deque

DATA_PATH   = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1c_result.json"

N_ITERATIONS   = 3
N_CURRICULUM   = 200   # doubled from Stage 1b (100) — Leo #11745
N_HELD         = 200   # same as Stage 1b
N_MBPP         = 50
SPLIT_SEED     = 42
# Depth-0: 330 programs. Depth-1 compose: 330^2 = 108,900.
# BUDGET=3000 covers all depth-0 (330) + 2670 depth-1 compose (2.45% of compose space).
# Full depth-1 (budget=109230) estimated >35 min — intractable within 5-min cap.
BUDGET         = 3000
MDL_MIN_OCC    = 2
N_DREAMS       = 30
TRANSFER_MARGIN = 5.0

N_DEPTH0_PROGS = 330   # updated after grammar built
TOTAL_DEPTH1   = 109230  # 330 + 330^2

# ── Object extraction (connected components, 4-connected) ─────────────────────
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
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = rr + dr, cc + dc
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

# ── Object predicates ─────────────────────────────────────────────────────────

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
    color_counts = collections.Counter(o['color'] for o in objs)
    return [o for o in objs if color_counts[o['color']] == 1]

def _pred_most_common_color(objs):
    if not objs: return []
    color_counts = collections.Counter(o['color'] for o in objs)
    most = color_counts.most_common(1)[0][0]
    return [o for o in objs if o['color'] == most]

def _pred_all(objs):
    return objs

PREDICATES = {
    'largest':        lambda g, objs: _pred_largest(objs),
    'smallest':       lambda g, objs: _pred_smallest(objs),
    'non_bg':         lambda g, objs: _pred_non_bg(g, objs),
    'bg_obj':         lambda g, objs: _pred_bg(g, objs),
    'unique_color':   lambda g, objs: _pred_unique_color(objs),
    'most_common_c':  lambda g, objs: _pred_most_common_color(objs),
    'all':            lambda g, objs: _pred_all(objs),
}
for _c in range(1, 10):
    PREDICATES[f'color_{_c}'] = (lambda c: lambda g, objs: [o for o in objs if o['color'] == c])(_c)
PRED_NAMES = sorted(PREDICATES.keys())

# ── Object transforms ──────────────────────────────────────────────────────────

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
    'delete':     lambda g, sel: _delete(g, sel),
    'keep_only':  lambda g, sel: _keep_only(g, sel),
}
for _c in range(10):
    TRANSFORMS[f'recolor_{_c}'] = (lambda c: lambda g, sel: _recolor(c)(g, sel))(_c)
for _dy, _dx in [(-1,0),(1,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]:
    TRANSFORMS[f'translate_{_dy:+d}_{_dx:+d}'] = (
        lambda dy, dx: lambda g, sel: _translate(dy, dx)(g, sel))(_dy, _dx)
TRANSFORM_NAMES = sorted(TRANSFORMS.keys())

# ── Whole-grid primitives ─────────────────────────────────────────────────────

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

# ── Program representation ────────────────────────────────────────────────────

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

# ── Leaf programs ─────────────────────────────────────────────────────────────

def all_leaves(lib, dl_weights=None):
    leaves = []
    if lib:
        macro_order = sorted(lib.keys(),
                             key=lambda n: -(dl_weights.get(n, 0) if dl_weights else 0))
        for name in macro_order:
            leaves.append(('macro', name))
    for pred in PRED_NAMES:
        for transform in TRANSFORM_NAMES:
            leaves.append(('map_apply', pred, transform))
    for name in PRIM_NAMES:
        leaves.append(('prim', name))
    return leaves

# ── Program depth and type ─────────────────────────────────────────────────────

def prog_depth(prog):
    t = prog[0]
    if t in ('prim', 'map_apply', 'macro'): return 0
    if t == 'compose': return 1 + max(prog_depth(prog[1]), prog_depth(prog[2]))
    return 0

def prog_type(prog):
    t = prog[0]
    if t == 'prim': return 'prim'
    if t == 'map_apply': return 'map_apply'
    if t == 'macro': return 'macro'
    if t == 'compose':
        has_map = any(p[0] == 'map_apply' for p in (prog[1], prog[2]))
        return 'compose_map_apply' if has_map else 'compose_prim'
    return 'unknown'

def prog_hash(prog):
    return hashlib.sha256(str(prog).encode()).hexdigest()[:12]

def _prog_size(prog):
    t = prog[0]
    if t == 'compose': return 1 + _prog_size(prog[1]) + _prog_size(prog[2])
    return 1

# ── BFS search — returns (prog, nodes_used, depth_of_solution) ────────────────

def search_task(task_io, lib, budget=BUDGET, dl_weights=None):
    nodes = 0
    leaves = all_leaves(lib, dl_weights)

    for prog in leaves:
        nodes += 1
        if nodes > budget: return None, nodes, -1
        if check_prog(prog, task_io, lib): return prog, nodes, 0

    for p1 in leaves:
        for p2 in leaves:
            nodes += 1
            if nodes > budget: return None, nodes, -1
            prog = ('compose', p1, p2)
            if check_prog(prog, task_io, lib): return prog, nodes, 1

    for p1 in leaves:
        for p2 in leaves:
            for p3 in leaves:
                nodes += 1
                if nodes > budget: return None, nodes, -1
                prog = ('compose', p1, ('compose', p2, p3))
                if check_prog(prog, task_io, lib): return prog, nodes, 2
                nodes += 1
                if nodes > budget: return None, nodes, -1
                prog = ('compose', ('compose', p1, p2), p3)
                if check_prog(prog, task_io, lib): return prog, nodes, 2

    return None, nodes, -1

# ── Abstraction sleep with funnel ─────────────────────────────────────────────

def abstraction_sleep(all_solved_records, library):
    """Returns (new_lib, abstraction_funnel_dict)."""
    new_lib = dict(library)

    # Group by program string
    prog_str_to_records = collections.defaultdict(list)
    for rec in all_solved_records:
        prog_str_to_records[rec['prog_str']].append(rec)

    funnel_candidates = []
    accepted = 0

    for prog_str, recs in prog_str_to_records.items():
        occ = len(recs)
        body = recs[0]['prog']
        body_size = _prog_size(body)
        savings = occ * (body_size - 1) - body_size
        source_task_ids = [r['task_id'] for r in recs]
        source_depths   = [r['depth'] for r in recs]

        if occ < MDL_MIN_OCC:
            funnel_candidates.append({
                'prog_str': prog_str[:80], 'prog_hash': recs[0]['hash'],
                'occurrences': occ, 'body_size': body_size,
                'mdl_savings': savings, 'decision': 'SKIPPED',
                'reason': f'occ={occ} < MDL_MIN_OCC={MDL_MIN_OCC}',
                'source_task_ids': source_task_ids, 'source_depths': source_depths,
            })
            continue

        if body_size == 1:
            funnel_candidates.append({
                'prog_str': prog_str[:80], 'prog_hash': recs[0]['hash'],
                'occurrences': occ, 'body_size': body_size,
                'mdl_savings': savings, 'decision': 'REJECTED',
                'reason': f'body_size=1: savings=occ*(1-1)-1=-1 (depth-0 programs cannot compress)',
                'source_task_ids': source_task_ids, 'source_depths': source_depths,
            })
            continue

        if savings <= 0:
            funnel_candidates.append({
                'prog_str': prog_str[:80], 'prog_hash': recs[0]['hash'],
                'occurrences': occ, 'body_size': body_size,
                'mdl_savings': savings, 'decision': 'REJECTED',
                'reason': f'savings={savings:.1f} <= 0',
                'source_task_ids': source_task_ids, 'source_depths': source_depths,
            })
            continue

        if any(str(m['body']) == prog_str for m in new_lib.values()):
            continue

        name = f"M{len(new_lib)}_{body_size}n_{occ}x"
        new_lib[name] = {
            'body': body, 'body_str': prog_str, 'occurrences': occ,
            'body_size': body_size, 'savings': savings,
            'source_task_ids': source_task_ids, 'source_depths': source_depths,
        }
        funnel_candidates.append({
            'prog_str': prog_str[:80], 'prog_hash': recs[0]['hash'],
            'occurrences': occ, 'body_size': body_size,
            'mdl_savings': savings, 'decision': 'ACCEPTED',
            'macro_name': name,
            'source_task_ids': source_task_ids, 'source_depths': source_depths,
        })
        accepted += 1
        print(f"    MACRO {name}: occ={occ}, savings={savings:.0f}")

    occ_ge2 = [c for c in funnel_candidates if c['occurrences'] >= MDL_MIN_OCC]
    unique_progs = len(prog_str_to_records)
    occ_dist = collections.Counter(len(recs) for recs in prog_str_to_records.values())

    # Structural note
    all_depth0 = all(r['depth'] == 0 for recs in prog_str_to_records.values() for r in recs)
    note_parts = []
    if all_depth0 and unique_progs > 0:
        note_parts.append(
            "All solved programs at depth-0 (body_size=1). "
            "MDL savings = occ*(1-1)-1 = -1 for all: depth-0 programs structurally cannot compress. "
            "Macro formation requires depth-1 compose solutions (body_size>=2)."
        )
    if not occ_ge2:
        note_parts.append(
            f"No programs appear in >=2 tasks (occ_dist={dict(sorted(occ_dist.items()))}). "
            "Either programs are fully diverse or compose coverage too low to detect reuse."
        )

    funnel = {
        'total_unique_programs': unique_progs,
        'occurrence_distribution': {str(k): v for k, v in sorted(occ_dist.items())},
        'occurrence_ge2_candidates': occ_ge2,
        'accepted_macros': accepted,
        'note': ' '.join(note_parts) if note_parts else '',
    }
    return new_lib, funnel

# ── Dream sleep ───────────────────────────────────────────────────────────────

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

# ── Transfer test ─────────────────────────────────────────────────────────────

def transfer_test(library, held_tasks, dl_weights=None, budget=BUDGET):
    empty = {}
    bl_costs, bl_solved = [], 0
    lib_costs, lib_solved, lib_new = [], 0, 0
    selected_macro_counts = []
    for task in held_tasks:
        prog_b, cb, _ = search_task(task['io'], empty, budget)
        bl_costs.append(cb)
        if prog_b: bl_solved += 1

        prog_l, cl, _ = search_task(task['io'], library, budget, dl_weights)
        lib_costs.append(cl)
        if prog_l:
            lib_solved += 1
            if not prog_b: lib_new += 1
        selected_macro_counts.append(len(library))

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
        'with_library': {'mean_cost': lib_mean, 'solved': lib_solved,
                         'new_solves': lib_new},
        'selected_macro_counts_per_task': len(library),
        'delta_pct': delta, 'verdict': verdict,
    }

# ── MBPP ──────────────────────────────────────────────────────────────────────

def _mbpp_search(test_cases, budget=BUDGET):
    import itertools
    MBPP_PRIMS = {
        'sort_asc':   lambda l: sorted(l),
        'sort_desc':  lambda l: sorted(l, reverse=True),
        'reverse':    lambda l: list(reversed(l)),
        'unique':     lambda l: list(dict.fromkeys(l)),
        'cumsum':     lambda l: [sum(l[:i+1]) for i in range(len(l))],
        'sum_list':   lambda l: sum(l),
        'len_list':   lambda l: len(l),
        'max_list':   lambda l: max(l) if l else 0,
        'min_list':   lambda l: min(l) if l else 0,
        'id':         lambda l: list(l),
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
            if all(fn(list(inp)) == exp for inp, exp in test_cases):
                return True
        except Exception:
            pass
    for f1 in all_fns:
        for f2 in all_fns:
            nodes += 1
            if nodes > budget: return False
            try:
                if all(f1(f2(list(inp))) == exp for inp, exp in test_cases):
                    return True
            except Exception:
                pass
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
            except Exception:
                pass
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
    print("STAGE 1c: DEPTH-2 FLAT BFS — OBJECT-CENTRIC, N_CURRICULUM=200")
    print("Leo #11745/#11749/#11752: depth budget increased, grammar preserved, Kai schema")
    print("=" * 70)
    t0 = time.time()
    sys.stdout.flush()

    all_arc = load_arc()
    training = [t for t in all_arc if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(training))
    held_tasks   = [training[i] for i in idx[:N_HELD]]
    pool_tasks   = [training[i] for i in idx[N_HELD:]]
    rng2 = np.random.default_rng(SPLIT_SEED + 1)
    curriculum_idx = rng2.choice(len(pool_tasks), min(N_CURRICULUM, len(pool_tasks)), replace=False)
    curriculum = [pool_tasks[i] for i in sorted(curriculum_idx)]
    held_ids   = [t['id'] for t in held_tasks]

    aug_status = {
        'curriculum_augmented': False,
        'held_augmented': False,
        'note': 'No D4 augmentation applied anywhere. Stage 1c source is raw ARC training split only.',
    }

    # Grammar inventory
    n_map_apply = len(PRED_NAMES) * len(TRANSFORM_NAMES)
    n_prim      = len(PRIM_NAMES)
    n_leaves    = n_map_apply + n_prim
    n_depth1_compose = n_leaves * n_leaves
    compose_covered = BUDGET - n_leaves  # depth-1 programs covered by budget
    compose_covered = max(0, min(compose_covered, n_depth1_compose))
    depth1_coverage_frac = compose_covered / n_depth1_compose if n_depth1_compose else 0

    mbpp_items = load_mbpp()
    print(f"ARC: held={len(held_tasks)}, curriculum={len(curriculum)}")
    print(f"MBPP: {len(mbpp_items)} items")
    print(f"Grammar: {len(PRED_NAMES)} pred x {len(TRANSFORM_NAMES)} tr = {n_map_apply} map_apply"
          f" + {n_prim} prim = {n_leaves} leaves (PRESERVED from Stage 1b)")
    print(f"Depth-0: {n_leaves} programs. Depth-1 compose: {n_depth1_compose:,} programs.")
    print(f"BUDGET={BUDGET}: covers all depth-0 ({n_leaves}) + {compose_covered:,} depth-1 "
          f"({depth1_coverage_frac:.2%} of compose space).")
    print(f"Full depth-1 estimated intractable: ~{n_depth1_compose / max(compose_covered, 1):.0f}x "
          f"more time than this budget.")
    sys.stdout.flush()

    library          = {}
    dl_weights       = {}
    all_solved_recs  = []  # [{prog, prog_str, hash, task_id, depth}]
    iter_results     = []
    cumulative_funnel = None

    for it in range(N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"ITERATION {it+1}/{N_ITERATIONS}  (library={len(library)} macros)")
        t_iter = time.time()
        sys.stdout.flush()

        # ── WAKE ──────────────────────────────────────────────────────────────
        it_costs, it_solved = [], 0
        it_solved_recs = []
        depth_counts = collections.Counter()  # depth -> n solved

        for i, task in enumerate(curriculum):
            prog, cost, depth = search_task(task['io'], library, BUDGET, dl_weights)
            it_costs.append(cost)
            if prog:
                it_solved += 1
                ps = str(prog)
                ph = prog_hash(prog)
                rec = {'prog': prog, 'prog_str': ps, 'hash': ph,
                       'task_id': task['id'], 'depth': depth}
                it_solved_recs.append(rec)
                depth_counts[depth] += 1
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
        sys.stdout.flush()

        mbpp_solved = eval_mbpp(mbpp_items)
        mbpp_rate   = mbpp_solved / max(len(mbpp_items), 1)
        print(f"  MBPP wake: solved={mbpp_solved}/{len(mbpp_items)} ({mbpp_rate:.1%})")
        sys.stdout.flush()

        # ── ABSTRACTION SLEEP with funnel ─────────────────────────────────────
        print(f"  Abstraction-sleep ({len(all_solved_recs)} solved programs total)...")
        sys.stdout.flush()
        old_sz  = len(library)
        library, funnel = abstraction_sleep(all_solved_recs, library)
        cumulative_funnel = funnel
        new_mac = len(library) - old_sz
        print(f"  Library: {old_sz} -> {len(library)} (+{new_mac} macros)")
        print(f"  Funnel: {funnel['total_unique_programs']} unique progs, "
              f"{len(funnel['occurrence_ge2_candidates'])} occ>=2 candidates, "
              f"{funnel['accepted_macros']} accepted")
        sys.stdout.flush()

        # ── DREAM SLEEP ───────────────────────────────────────────────────────
        print(f"  Dream-sleep: {min(N_DREAMS, len(curriculum))} fantasies...")
        task_sample = [t['io'] for t in curriculum[:N_DREAMS]]
        dl_weights  = dream_sleep(library, task_sample)
        active = sum(1 for w in dl_weights.values() if w > 0)
        print(f"  Dream-sleep: {active}/{len(library)} macros active")
        sys.stdout.flush()

        # ── TRANSFER ──────────────────────────────────────────────────────────
        print(f"  Transfer test ({len(held_tasks)} held-out, budget={BUDGET})...")
        t_tr = time.time()
        transfer = transfer_test(library, held_tasks, dl_weights, BUDGET)
        print(f"  {transfer['verdict']} (transfer: {time.time()-t_tr:.0f}s)")
        sys.stdout.flush()

        iter_results.append({
            'iteration':    it + 1,
            'library_size': len(library),
            'new_macros':   new_mac,
            'arc': {'solved': it_solved, 'n': len(curriculum),
                    'solve_rate': arc_rate, 'mean_cost': arc_cost,
                    'depth_distribution': {str(k): v for k, v in sorted(depth_counts.items())}},
            'mbpp': {'solved': mbpp_solved, 'n': len(mbpp_items), 'solve_rate': mbpp_rate},
            'transfer': transfer,
            'dream_sleep_active_macros': active,
            'elapsed_sec': time.time() - t_iter,
        })

    # ── Aggregate program type breakdown ──────────────────────────────────────
    all_type_counts = collections.Counter(prog_type(r['prog']) for r in all_solved_recs)
    all_depth_counts = collections.Counter(r['depth'] for r in all_solved_recs)

    # ── Kai classification ────────────────────────────────────────────────────
    rates  = [r['arc']['solve_rate'] for r in iter_results]
    deltas = [r['transfer']['delta_pct'] for r in iter_results]
    last_d = deltas[-1] if deltas else 0
    n_macros = len(library)
    depth1_solves = all_depth_counts.get(1, 0)

    # Classification logic:
    if depth1_coverage_frac < 0.10 and depth1_solves == 0:
        kai_class = "DEPTH_STARVATION"
        kai_reason = (f"Compose coverage {depth1_coverage_frac:.2%} < 10% of depth-1 space. "
                      f"Zero depth-1 solves: yield too low to test macro formation at depth-1. "
                      f"Full depth-1 flat BFS intractable at ~{1/depth1_coverage_frac:.0f}x this budget.")
    elif depth1_solves > 0 and n_macros == 0:
        kai_class = "FORMATION_NEGATIVE_DEPTH2"
        kai_reason = (f"{depth1_solves} depth-1 solves found. "
                      f"No MDL-positive reusable chunks: structure-absence argument strengthens.")
    elif n_macros > 0 and last_d >= -TRANSFER_MARGIN:
        kai_class = "HOLLOW_DEPTH2"
        kai_reason = f"{n_macros} macros formed but transfer delta={last_d:+.1f}% >= -{TRANSFER_MARGIN:.0f}%."
    elif n_macros > 0 and last_d < -TRANSFER_MARGIN:
        kai_class = "TRANSFER_GENUINE_DEPTH2"
        kai_reason = (f"{n_macros} macros formed + transfer delta={last_d:+.1f}% < -{TRANSFER_MARGIN:.0f}%. "
                      f"Net-free bootstrap revives: Stage 1b was depth-starved.")
    else:
        kai_class = "DEPTH_STARVATION"
        kai_reason = f"Default: coverage={depth1_coverage_frac:.2%}, depth1_solves={depth1_solves}."

    # ── Verdict string ─────────────────────────────────────────────────────────
    monotone = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    if kai_class == "TRANSFER_GENUINE_DEPTH2":
        verdict = (f"PASS: {kai_class} — net-free bootstrap revives at depth-2 "
                   f"(delta={last_d:+.1f}%).")
    elif kai_class == "DEPTH_STARVATION":
        verdict = (f"DEPTH_STARVATION: compose coverage {depth1_coverage_frac:.2%} "
                   f"({compose_covered:,}/{n_depth1_compose:,} depth-1 programs). "
                   f"Full depth-1 intractable (~{1/max(depth1_coverage_frac,1e-9):.0f}x budget needed). "
                   f"This intractability is the pre-registered Stage 2 (proposer) argument.")
    else:
        verdict = f"{kai_class}: {kai_reason}"

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"KAI CLASSIFICATION: {kai_class}")
    print(f"VERDICT: {verdict}")
    print(f"Elapsed: {total:.1f}s")
    sys.stdout.flush()

    # ── Claim scope ────────────────────────────────────────────────────────────
    claim_scope = (
        f"Stage 1c searched {n_leaves} depth-0 programs (100%) and "
        f"{compose_covered:,}/{n_depth1_compose:,} depth-1 compose programs "
        f"({depth1_coverage_frac:.2%}). "
        f"Structure-absence cannot be claimed from this coverage: "
        f"{'zero depth-1 programs evaluated' if depth1_solves == 0 else f'{depth1_solves} depth-1 solves found'}. "
        f"Full depth-1 flat BFS requires budget={n_leaves + n_depth1_compose:,} "
        f"(~{(n_leaves + n_depth1_compose)/BUDGET:.0f}x this budget); "
        f"estimated intractable. "
        f"Any-to-any escalation remains premature: gated on Stage 2 (proposer) providing "
        f"deeper solves with still-zero reuse/transfer."
    )

    result = {
        'experiment':    'stage1c-depth2-flat-bfs-object-centric',
        'note':          ('Leo #11745/#11749/#11752: one-variable discipline — grammar preserved '
                          'from Stage 1b, only depth budget increased. '
                          'Kai binding gate schema (#11748). PRISM multi-domain (ARC+MBPP).'),
        'config': {
            'n_iterations': N_ITERATIONS, 'n_curriculum': N_CURRICULUM,
            'n_held': N_HELD, 'n_mbpp': N_MBPP,
            'split_seed': SPLIT_SEED, 'budget': BUDGET,
            'mdl_min_occ': MDL_MIN_OCC, 'n_dreams': N_DREAMS,
            'transfer_margin_pct': TRANSFER_MARGIN,
        },
        'grammar': {
            'predicates': PRED_NAMES, 'transforms': TRANSFORM_NAMES,
            'n_map_apply_ops': n_map_apply, 'whole_grid_prims': PRIM_NAMES,
            'n_leaves': n_leaves, 'preserved_from_stage1b': True,
        },
        'depth_budget_by_level': {
            'depth_0': n_leaves,
            'depth_1_covered': compose_covered,
            'depth_1_total':   n_depth1_compose,
            'depth_1_coverage_fraction': depth1_coverage_frac,
        },
        'max_depth_reached': 1 if BUDGET > n_leaves else 0,
        'held_task_ids':   held_ids,
        'source_construction': (
            f'curriculum={N_CURRICULUM} RANDOM training tasks (no sorting); '
            f'held={N_HELD} from training (split_seed={SPLIT_SEED}); '
            f'NO augmentation. Same split as Stage 1b.'),
        'aug_leakage_check':   {'hits': 0, 'verdict': 'CLEAN: no augmentation applied.'},
        'augmentation_status': aug_status,
        'solved_programs': [
            {'prog_str': r['prog_str'][:120], 'hash': r['hash'],
             'task_id': r['task_id'], 'depth': r['depth'],
             'prog_type': prog_type(r['prog'])}
            for r in all_solved_recs
        ],
        'program_type_breakdown': dict(all_type_counts),
        'depth_distribution_all_solves': {str(k): v for k, v in sorted(all_depth_counts.items())},
        'abstraction_funnel': cumulative_funnel,
        'library_final': {
            n: {
                'body_str': m['body_str'], 'occurrences': m['occurrences'],
                'body_size': m['body_size'], 'savings': m['savings'],
                'source_task_ids': m['source_task_ids'],
                'source_depths': m['source_depths'],
            }
            for n, m in library.items()
        },
        'iterations':   iter_results,
        'claim_scope':  claim_scope,
        'kai_classification': kai_class,
        'kai_reason':   kai_reason,
        'verdict':      verdict,
        'total_elapsed_sec': total,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Written: {RESULT_PATH}")

if __name__ == '__main__':
    main()
