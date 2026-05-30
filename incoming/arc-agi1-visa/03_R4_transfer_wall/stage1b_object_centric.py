"""
Stage 1b: Object-centric wake-sleep loop (Leo #11740, 2026-05-30).

Corrects Stage 1 grammar: whole-grid combinators CANNOT surface object-level shared
structure because two tasks that both "recolor the largest object red" have DIFFERENT
whole-grid programs (object at different positions/shapes) -> look unique -> no macro.
The object-centric grammar expresses them as the SAME program -> macros form.

Grammar:
  - object-extract(predicate) — segment grid into connected components, select by pred
  - map-apply(pred, transform) — apply transform to grid for selected objects
  - compose(f, g) — whole-program composition
  - Whole-grid prims retained for tasks that genuinely are whole-grid (geometric parity)

Fixes from Stage 1 review (Leo #11740):
  1. Object-centric combinators (the missing piece).
  2. N_CURRICULUM=100 random from pool (not 30 easiest — prevents easy-set bias).
  3. Augmentation status: explicitly confirmed CLEAN (no D4 augmentation anywhere).
  4. Transfer bug fixed: baseline searched once per task (not twice).
  5. Dream-sleep runs and DL prior actually updated (library feedback to next iter).
  6. MBPP: improved combinator coverage.

Kill criteria (unchanged from Leo #11734):
  PASS: solve-rate lifts + transfer delta < -5%.
  FAIL (informative): flat across >=3 iters.
  HOLLOW REDUX on THIS grammar: structure-absence established -> any-to-any escalation.

Gated result JSON: held_task_ids, source_construction, aug_leakage_check (CLEAN),
  augmentation_status (explicit), iterations with library per iter.
"""

import numpy as np
import json, os, sys, time, collections
from collections import deque

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1b_result.json"

N_ITERATIONS   = 3
N_CURRICULUM   = 100   # random from pool (not sorted-by-complexity)
N_HELD         = 200   # original held-out for transfer
N_MBPP         = 50
SPLIT_SEED     = 42
BUDGET         = 2000
MDL_MIN_OCC    = 2     # min task-count for macro to gate MDL
N_DREAMS       = 30    # fantasies per dream-sleep
TRANSFER_MARGIN = 5.0  # % threshold for TRANSFER_GENUINE

# ── Object extraction (connected components, 4-connected) ─────────────────────

def get_bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])

def extract_objects(grid):
    """Extract connected components (4-connected, same color). Returns list of obj dicts."""
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
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                objects.append({'color': color, 'cells': cells, 'area': len(cells)})
    return objects

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

def _make_color_pred(c):
    return lambda objs: [o for o in objs if o['color'] == c]

# Predicate registry: pred_name -> fn(grid, objects) -> selected_objects
# Note: preds that need grid get it; simple ones don't
PREDICATES = {
    'largest':        lambda g, objs: _pred_largest(objs),
    'smallest':       lambda g, objs: _pred_smallest(objs),
    'non_bg':         lambda g, objs: _pred_non_bg(g, objs),
    'bg_obj':         lambda g, objs: _pred_bg(g, objs),
    'unique_color':   lambda g, objs: _pred_unique_color(objs),
    'most_common_c':  lambda g, objs: _pred_most_common_color(objs),
    'all':            lambda g, objs: _pred_all(objs),
}
# Color-specific predicates
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
        # Erase old positions
        for obj in selected:
            for r, c in obj['cells']:
                g[r, c] = bg
        # Draw at new positions (clamp to grid bounds)
        for obj in selected:
            for r, c in obj['cells']:
                nr, nc = r + dy, c + dx
                if 0 <= nr < h and 0 <= nc < w:
                    g[nr, nc] = obj['color']
        return g
    return fn

# Transform registry: transform_name -> fn(grid, selected_objects) -> grid
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

# ── Object-centric program evaluation ─────────────────────────────────────────

def eval_map_apply(pred_name, transform_name, grid):
    """Apply transform to objects selected by predicate."""
    try:
        objs = extract_objects(grid)
        selected = PREDICATES[pred_name](grid, objs)
        if not selected:
            return grid.copy()
        return TRANSFORMS[transform_name](grid, selected)
    except Exception:
        return None

# ── Whole-grid primitives (retained for geometric-parity tasks) ───────────────

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
# ('prim', name)                 — whole-grid primitive
# ('map_apply', pred, transform) — object-centric op
# ('compose', p1, p2)            — compose p2 then p1
# ('macro', name)                — library macro (body in library dict)

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
        return eval_map_apply(prog[1], prog[2], g)
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
    """All depth-0 programs. Library macros come first (lower DL cost)."""
    leaves = []
    # Library macros first (treated as depth-0 primitives, lower DL cost)
    if lib:
        macro_order = sorted(lib.keys(),
                             key=lambda n: -(dl_weights.get(n, 0) if dl_weights else 0))
        for name in macro_order:
            leaves.append(('macro', name))
    # Object-centric: map_apply(pred, transform) for all pred × transform
    for pred in PRED_NAMES:
        for transform in TRANSFORM_NAMES:
            leaves.append(('map_apply', pred, transform))
    # Whole-grid prims
    for name in PRIM_NAMES:
        leaves.append(('prim', name))
    return leaves

# ── BFS search ────────────────────────────────────────────────────────────────

def search_task(task_io, lib, budget=BUDGET, dl_weights=None):
    nodes = 0
    leaves = all_leaves(lib, dl_weights)

    for prog in leaves:
        nodes += 1
        if nodes > budget: return None, nodes
        if check_prog(prog, task_io, lib): return prog, nodes

    for p1 in leaves:
        for p2 in leaves:
            nodes += 1
            if nodes > budget: return None, nodes
            prog = ('compose', p1, p2)
            if check_prog(prog, task_io, lib): return prog, nodes

    for p1 in leaves:
        for p2 in leaves:
            for p3 in leaves:
                nodes += 1
                if nodes > budget: return None, nodes
                prog = ('compose', p1, ('compose', p2, p3))
                if check_prog(prog, task_io, lib): return prog, nodes
                nodes += 1
                if nodes > budget: return None, nodes
                prog = ('compose', ('compose', p1, p2), p3)
                if check_prog(prog, task_io, lib): return prog, nodes

    return None, nodes

# ── Abstraction sleep ─────────────────────────────────────────────────────────

def _prog_shape(prog):
    """Replace param values with '?' to get skeleton — for now exact match only."""
    return prog  # no holes in first iteration (macros are exact programs)

def _prog_size(prog):
    t = prog[0]
    if t == 'compose': return 1 + _prog_size(prog[1]) + _prog_size(prog[2])
    return 1

def abstraction_sleep(all_solved, library):
    new_lib = dict(library)
    prog_tasks = collections.defaultdict(list)
    for prog in all_solved:
        prog_tasks[str(prog)].append(prog)

    for prog_str, progs in prog_tasks.items():
        occ = len(progs)
        if occ < MDL_MIN_OCC: continue
        body = progs[0]
        body_size = _prog_size(body)
        # Compression: each use costs 1 (macro call) instead of body_size
        savings = occ * (body_size - 1) - body_size
        if savings <= 0: continue
        if any(str(m['body']) == str(body) for m in new_lib.values()):
            continue
        name = f"M{len(new_lib)}_{body_size}n_{occ}x"
        new_lib[name] = {'body': body, 'occurrences': occ, 'body_size': body_size,
                          'savings': savings}
        print(f"    MACRO {name}: body={prog_str[:60]}, occ={occ}, savings={savings:.0f}")
    return new_lib

# ── Dream sleep ───────────────────────────────────────────────────────────────

def dream_sleep(library, task_sample_io, n_dreams=N_DREAMS):
    """Apply macros to sample task grids. Count non-trivial transformations -> DL weights."""
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

# ── Transfer test (E3 instrument — ONE baseline search per task) ──────────────

def transfer_test(library, held_tasks, dl_weights=None, budget=BUDGET):
    empty = {}
    bl_costs, bl_solved = [], 0
    lib_costs, lib_solved, lib_new = [], 0, 0
    for task in held_tasks:
        prog_b, cb = search_task(task['io'], empty, budget)
        bl_costs.append(cb)
        if prog_b: bl_solved += 1
        prog_l, cl = search_task(task['io'], library, budget, dl_weights)
        lib_costs.append(cl)
        if prog_l:
            lib_solved += 1
            if not prog_b: lib_new += 1
    bl_mean  = float(np.mean(bl_costs))
    lib_mean = float(np.mean(lib_costs))
    delta    = (lib_mean - bl_mean) / bl_mean * 100 if bl_mean else 0
    if delta < -TRANSFER_MARGIN:
        verdict = f"TRANSFER_GENUINE: delta={delta:+.1f}% < -{TRANSFER_MARGIN:.0f}%."
    else:
        verdict = f"HOLLOW: delta={delta:+.1f}% >= -{TRANSFER_MARGIN:.0f}%."
    return {'baseline': {'mean_cost': bl_mean, 'solved': bl_solved},
            'with_library': {'mean_cost': lib_mean, 'solved': lib_solved, 'new_solves': lib_new},
            'delta_pct': delta, 'verdict': verdict}

# ── MBPP (improved: list→list/int and int→int coverage) ──────────────────────

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
    # Try single ops
    for fn in all_fns:
        nodes += 1
        if nodes > budget: return False
        try:
            if all(fn(list(inp)) == exp for inp, exp in test_cases):
                return True
        except Exception:
            pass
    # Try compose of two ops
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
    print("STAGE 1b: OBJECT-CENTRIC WAKE-SLEEP LOOP")
    print("Leo #11740: object-extract + map-apply combinators for object-level shared structure")
    print("=" * 70)
    t0 = time.time()
    sys.stdout.flush()

    all_arc = load_arc()
    training = [t for t in all_arc if t['split_dir'] == 'training']
    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(training))
    held_tasks   = [training[i] for i in idx[:N_HELD]]
    pool_tasks   = [training[i] for i in idx[N_HELD:]]
    # Curriculum: RANDOM 100 from pool (not easiest-30 — avoids easy-set bias)
    rng2 = np.random.default_rng(SPLIT_SEED + 1)
    curriculum_idx = rng2.choice(len(pool_tasks), min(N_CURRICULUM, len(pool_tasks)), replace=False)
    curriculum = [pool_tasks[i] for i in sorted(curriculum_idx)]
    held_ids   = [t['id'] for t in held_tasks]

    # Augmentation status: explicitly CLEAN
    aug_status = {
        'curriculum_augmented': False,
        'held_augmented': False,
        'note': 'No D4 augmentation applied anywhere. Stage 1b source is raw ARC training split only.',
    }

    mbpp_items = load_mbpp()
    print(f"ARC: held={len(held_tasks)}, curriculum={len(curriculum)} (random from pool, not sorted)")
    print(f"MBPP: {len(mbpp_items)} items")
    print(f"Grammar: {len(PRED_NAMES)} predicates x {len(TRANSFORM_NAMES)} transforms = "
          f"{len(PRED_NAMES)*len(TRANSFORM_NAMES)} map_apply ops + {len(PRIM_NAMES)} whole-grid prims")
    print(f"Augmentation: CLEAN ({aug_status['note']})")
    sys.stdout.flush()

    library     = {}
    dl_weights  = {}
    all_solved  = []
    iter_results = []

    for it in range(N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"ITERATION {it+1}/{N_ITERATIONS}  (library={len(library)} macros)")
        t_iter = time.time()
        sys.stdout.flush()

        # ── WAKE ──────────────────────────────────────────────────────────────
        it_costs, it_solved, it_progs = [], 0, []
        for i, task in enumerate(curriculum):
            prog, cost = search_task(task['io'], library, BUDGET, dl_weights)
            it_costs.append(cost)
            if prog:
                it_solved += 1
                it_progs.append(prog)
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t_iter
                print(f"  ... {i+1}/{len(curriculum)} tasks, {it_solved} solved, {elapsed:.0f}s")
                sys.stdout.flush()
        all_solved.extend(it_progs)
        arc_rate = it_solved / len(curriculum)
        arc_cost = float(np.mean(it_costs))
        print(f"  ARC wake: solved={it_solved}/{len(curriculum)} ({arc_rate:.1%}), mean_cost={arc_cost:.0f}")
        sys.stdout.flush()

        # MBPP wake
        mbpp_solved = eval_mbpp(mbpp_items)
        mbpp_rate   = mbpp_solved / max(len(mbpp_items), 1)
        print(f"  MBPP wake: solved={mbpp_solved}/{len(mbpp_items)} ({mbpp_rate:.1%})")
        sys.stdout.flush()

        # ── ABSTRACTION SLEEP ─────────────────────────────────────────────────
        print(f"  Abstraction-sleep ({len(all_solved)} solved programs total)...")
        sys.stdout.flush()
        old_sz  = len(library)
        library = abstraction_sleep(all_solved, library)
        new_mac = len(library) - old_sz
        print(f"  Library: {old_sz} -> {len(library)} (+{new_mac} macros)")
        sys.stdout.flush()

        # ── DREAM SLEEP ───────────────────────────────────────────────────────
        print(f"  Dream-sleep: sampling {min(N_DREAMS, len(curriculum))} fantasies...")
        task_sample = [t['io'] for t in curriculum[:N_DREAMS]]
        dl_weights  = dream_sleep(library, task_sample)
        active = sum(1 for w in dl_weights.values() if w > 0)
        print(f"  Dream-sleep: {active}/{len(library)} macros active, "
              f"weights={sorted(dl_weights.values(), reverse=True)[:5]}")
        sys.stdout.flush()

        # ── TRANSFER (built-in from iteration 1, E3 instrument, ONE baseline call) ──
        print(f"  Transfer test ({len(held_tasks)} held-out tasks, budget={BUDGET})...")
        t_transfer = time.time()
        sys.stdout.flush()
        transfer = transfer_test(library, held_tasks, dl_weights, BUDGET)
        print(f"  {transfer['verdict']} (transfer elapsed: {time.time()-t_transfer:.0f}s)")
        sys.stdout.flush()

        # Solve type breakdown for first iteration
        if it == 0 and it_progs:
            obj_type = sum(1 for p in it_progs if p[0] == 'map_apply' or
                          (p[0] == 'compose' and (p[1][0] == 'map_apply' or p[2][0] == 'map_apply')))
            whole_type = len(it_progs) - obj_type
            print(f"  Program types: {obj_type}/{len(it_progs)} use map_apply (object-level), "
                  f"{whole_type}/{len(it_progs)} whole-grid only")
            sys.stdout.flush()

        iter_results.append({
            'iteration':    it + 1,
            'library_size': len(library),
            'new_macros':   new_mac,
            'arc': {'solved': it_solved, 'n': len(curriculum),
                    'solve_rate': arc_rate, 'mean_cost': arc_cost},
            'mbpp': {'solved': mbpp_solved, 'n': len(mbpp_items), 'solve_rate': mbpp_rate},
            'transfer': transfer,
            'dream_sleep_active_macros': active,
            'elapsed_sec': time.time() - t_iter,
        })

    # ── Verdict ───────────────────────────────────────────────────────────────
    rates    = [r['arc']['solve_rate'] for r in iter_results]
    deltas   = [r['transfer']['delta_pct'] for r in iter_results]
    monotone = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    last_d   = deltas[-1] if deltas else 0

    if monotone and last_d < -TRANSFER_MARGIN:
        verdict = f"PASS: solve-rate monotone + transfer genuine (delta={last_d:+.1f}%). Library generalizes."
    elif max(rates) - min(rates) < 0.02 and last_d >= -TRANSFER_MARGIN:
        verdict = (f"FAIL (informative): solve-rate flat ({min(rates):.1%}-{max(rates):.1%}), "
                   f"transfer HOLLOW redux (delta={last_d:+.1f}%). "
                   f"Structure-absence ESTABLISHED on object-centric grammar. "
                   f"Pre-registered escalation: any-to-any representation question.")
    else:
        verdict = (f"PARTIAL: rates={[f'{r:.1%}' for r in rates]}, transfer={last_d:+.1f}%.")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"Total elapsed: {total:.1f}s")
    sys.stdout.flush()

    result = {
        'experiment': 'stage1b-object-centric-wake-sleep',
        'note': ('Leo #11740: object-centric combinator grammar corrects Stage 1 whole-grid gap. '
                 'map_apply(predicate, transform) expresses object-level shared structure. '
                 'PRISM multi-domain (ARC+MBPP). Transfer from iteration 1. '
                 'Transfer bug fixed (one baseline search per task). '
                 'Kai-gated per E3 discipline.'),
        'config': {
            'n_iterations': N_ITERATIONS, 'n_curriculum': N_CURRICULUM,
            'n_held': N_HELD, 'n_mbpp': N_MBPP,
            'split_seed': SPLIT_SEED, 'budget': BUDGET,
            'mdl_min_occ': MDL_MIN_OCC, 'n_dreams': N_DREAMS,
            'transfer_margin_pct': TRANSFER_MARGIN,
        },
        'grammar': {
            'predicates': PRED_NAMES,
            'transforms': TRANSFORM_NAMES,
            'n_map_apply_ops': len(PRED_NAMES) * len(TRANSFORM_NAMES),
            'whole_grid_prims': PRIM_NAMES,
        },
        'held_task_ids':       held_ids,
        'source_construction': (f'curriculum={N_CURRICULUM} RANDOM training tasks (no sorting); '
                                f'held={N_HELD} from training (split_seed={SPLIT_SEED}); '
                                f'NO augmentation.'),
        'aug_leakage_check':   {'hits': 0, 'verdict': 'CLEAN: no augmentation applied.'},
        'augmentation_status': aug_status,
        'iterations':    iter_results,
        'library_final': {n: {'body_str': str(m['body']), 'occurrences': m['occurrences'],
                               'savings': m['savings']}
                          for n, m in library.items()},
        'verdict':       verdict,
        'total_elapsed_sec': total,
    }

    with open(RESULT_PATH, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Written: {RESULT_PATH}")

if __name__ == '__main__':
    main()
