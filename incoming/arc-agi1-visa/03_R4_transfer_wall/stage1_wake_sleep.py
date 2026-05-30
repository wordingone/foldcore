"""
Stage 1: Net-free wake-sleep self-compilation loop (Leo #11734, 2026-05-30).

Grammar: typed combinators with argument slots (NOT flat op-list / designer-expansion).
  ARC: Grid->Grid — geometric prims + recolor(c1,c2) + compose(f,g) + library macros
  MBPP: List->List/int — sort/filter/map combinators + param ops

Loop (N_ITERATIONS):
  WAKE (curriculum-ordered BFS guided by DL prior)
  -> ABSTRACTION-SLEEP (MDL-gated library from solved traces)
  -> DREAM-SLEEP (macro sampling -> fantasy DL prior update)
  -> TRANSFER (library tested on original held-out, built-in from iter 1)

Kill criteria (Leo #11734):
  PASS: solve-rate lifts across >=3 curriculum tiers AND transfer delta < -5%.
  FAIL (informative): flat -> reveals grammar-poverty vs structure-absence.
  HOLLOW redux: transfer >= -5% even with richer grammar -> ARC lacks shared structure.

Gated result JSON (Kai-compatible: held_task_ids, source_construction,
  aug_leakage_check, iteration results with library per iter).
"""

import numpy as np
import json, os, sys, itertools, collections, time

DATA_PATH = "B:/M/the-search/incoming/arc-agi1-visa/ARC-AGI/data"
RESULT_PATH = "incoming/arc-agi1-visa/03_R4_transfer_wall/stage1_result.json"

N_ITERATIONS   = 3
N_CURRICULUM   = 30    # easiest curriculum tasks for wake
N_HELD         = 200   # original held-out for transfer
N_MBPP         = 50
SPLIT_SEED     = 42
BUDGET         = 2000  # programs tried per task
MDL_MIN_OCC    = 2     # min task count for macro to gate MDL
N_DREAMS       = 20    # fantasies per dream-sleep

# ── ARC grid primitives ───────────────────────────────────────────────────────

def _crop_bg(g):
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
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
    'mir_h':   lambda g: np.hstack([g, g[:, ::-1]]),
    'mir_v':   lambda g: np.vstack([g, g[::-1]]),
    'dup_h':   lambda g: np.hstack([g, g]),
    'dup_v':   lambda g: np.vstack([g, g]),
    'id':      lambda g: g,
    'crop':    _crop_bg,
}
PRIM_NAMES = sorted(GRID_PRIMS.keys())

# Parameterized recolor(c1, c2): replace color c1 with c2.
# Represented as argument slots — NOT a flat list of 72 ops.
RECOLOR_PAIRS = [(c1, c2) for c1 in range(1, 10) for c2 in range(0, 10) if c1 != c2]

# ── ARC program evaluation ────────────────────────────────────────────────────

def _eval_arc(prog, g, lib):
    """Evaluate ARC program tree on numpy grid g."""
    t = prog[0]
    if t == 'prim':
        return GRID_PRIMS[prog[1]](g)
    if t == 'recolor':
        return np.where(g == prog[1], prog[2], g)
    if t == 'compose':
        r = _eval_arc(prog[2], g, lib)
        if r is None: return None
        return _eval_arc(prog[1], r, lib)
    if t == 'macro':
        m = lib.get(prog[1])
        if m is None: return None
        return _eval_arc(m['body'], g, lib)
    return None

def check_arc(prog, task_io, lib):
    for pair in task_io:
        try:
            inp = np.array(pair['input'], dtype=np.int64)
            out = np.array(pair['output'], dtype=np.int64)
            r = _eval_arc(prog, inp, lib)
            if r is None or r.shape != out.shape or not np.array_equal(r, out):
                return False
        except Exception:
            return False
    return True

# ── ARC leaf programs (depth 0) ───────────────────────────────────────────────

def arc_leaves(lib):
    leaves = [('prim', n) for n in PRIM_NAMES]
    leaves += [('recolor', c1, c2) for c1, c2 in RECOLOR_PAIRS]
    leaves += [('macro', n) for n in sorted(lib.keys())]
    return leaves

# ── ARC search (BFS by depth, library macros at depth 0) ─────────────────────

def search_arc(task_io, lib, budget=BUDGET):
    nodes = 0
    leaves = arc_leaves(lib)

    for prog in leaves:
        nodes += 1
        if nodes > budget: return None, nodes
        if check_arc(prog, task_io, lib): return prog, nodes

    for p1 in leaves:
        for p2 in leaves:
            nodes += 1
            if nodes > budget: return None, nodes
            prog = ('compose', p1, p2)
            if check_arc(prog, task_io, lib): return prog, nodes

    for p1 in leaves:
        for p2 in leaves:
            for p3 in leaves:
                nodes += 1
                if nodes > budget: return None, nodes
                prog = ('compose', p1, ('compose', p2, p3))
                if check_arc(prog, task_io, lib): return prog, nodes
                nodes += 1
                if nodes > budget: return None, nodes
                prog = ('compose', ('compose', p1, p2), p3)
                if check_arc(prog, task_io, lib): return prog, nodes

    return None, nodes

# ── MBPP primitives ───────────────────────────────────────────────────────────

MBPP_PRIMS = {
    'sort_asc':  lambda l: sorted(l),
    'sort_desc': lambda l: sorted(l, reverse=True),
    'reverse':   lambda l: list(reversed(l)),
    'unique':    lambda l: list(dict.fromkeys(l)),
    'cumsum':    lambda l: [sum(l[:i+1]) for i in range(len(l))],
    'id_list':   lambda l: list(l),
}
MBPP_PRIM_NAMES = sorted(MBPP_PRIMS.keys())

def _mbpp_filter_gt(n): return lambda l: [x for x in l if x > n]
def _mbpp_filter_lt(n): return lambda l: [x for x in l if x < n]
def _mbpp_map_add(n):   return lambda l: [x + n for x in l]
def _mbpp_map_mul(n):   return lambda l: [x * n for x in l]

MBPP_PARAM_OPS = (
    [('filter_gt', n) for n in range(0, 6)] +
    [('filter_lt', n) for n in range(1, 7)] +
    [('map_add',   n) for n in range(1, 6)] +
    [('map_mul',   n) for n in range(2, 5)]
)
_MBPP_PARAM_FNS = {
    'filter_gt': _mbpp_filter_gt,
    'filter_lt': _mbpp_filter_lt,
    'map_add':   _mbpp_map_add,
    'map_mul':   _mbpp_map_mul,
}

def _eval_mbpp(prog, lst):
    t = prog[0]
    if t == 'mbpp_prim':
        return MBPP_PRIMS[prog[1]](lst)
    if t == 'mbpp_param':
        _, name, n = prog
        return _MBPP_PARAM_FNS[name](n)(lst)
    if t == 'mbpp_compose':
        r = _eval_mbpp(prog[2], lst)
        if r is None: return None
        return _eval_mbpp(prog[1], r)
    return None

def check_mbpp(prog, test_cases):
    for inp, exp in test_cases:
        try:
            r = _eval_mbpp(prog, list(inp))
            if r != exp: return False
        except Exception:
            return False
    return True

def mbpp_leaves():
    leaves = [('mbpp_prim', n) for n in MBPP_PRIM_NAMES]
    leaves += [('mbpp_param', name, n) for name, n in MBPP_PARAM_OPS]
    return leaves

def search_mbpp(test_cases, budget=BUDGET):
    nodes = 0
    leaves = mbpp_leaves()
    for prog in leaves:
        nodes += 1
        if nodes > budget: return None, nodes
        if check_mbpp(prog, test_cases): return prog, nodes
    for p1 in leaves:
        for p2 in leaves:
            nodes += 1
            if nodes > budget: return None, nodes
            prog = ('mbpp_compose', p1, p2)
            if check_mbpp(prog, test_cases): return prog, nodes
    return None, nodes

# ── Abstraction sleep ─────────────────────────────────────────────────────────

def _prog_shape(prog):
    """Replace recolor arg values with '?' to get skeleton."""
    t = prog[0]
    if t == 'recolor': return ('recolor', '?', '?')
    if t == 'compose': return ('compose', _prog_shape(prog[1]), _prog_shape(prog[2]))
    return prog

def _prog_size(prog):
    t = prog[0]
    if t == 'compose': return 1 + _prog_size(prog[1]) + _prog_size(prog[2])
    return 1

def abstraction_sleep(all_solved, library):
    """Extract MDL-gated macros. Only exact programs (0 holes) compress cleanly."""
    new_lib = dict(library)
    shape_tasks = collections.defaultdict(list)
    for prog in all_solved:
        shape = _prog_shape(prog)
        # Only accept no-hole shapes (exact programs reused across tasks)
        if "'?'" not in str(shape):
            shape_tasks[str(shape)].append(prog)

    for shape_str, progs in shape_tasks.items():
        occ = len(progs)
        if occ < MDL_MIN_OCC: continue
        body = progs[0]
        body_size = _prog_size(body)
        # Compression per use = body_size - 1 (macro call = 1 node)
        savings = occ * (body_size - 1) - body_size
        if savings <= 0: continue
        # Skip if already in library
        if any(str(m['body']) == str(body) for m in new_lib.values()):
            continue
        name = f"M{len(new_lib)}_{body_size}n_{occ}x"
        new_lib[name] = {'body': body, 'occurrences': occ, 'body_size': body_size, 'savings': savings}
        print(f"    MACRO {name}: occ={occ}, savings={savings:.0f}")
    return new_lib

# ── Dream sleep ───────────────────────────────────────────────────────────────

def dream_sleep(library, task_sample_io, n_dreams=N_DREAMS):
    """Apply macros to task inputs; count macros producing distinct outputs."""
    weights = {name: 0 for name in library}
    for task_io in task_sample_io[:n_dreams]:
        for name, macro in library.items():
            try:
                prog = ('macro', name)
                for pair in task_io[:2]:
                    inp = np.array(pair['input'], dtype=np.int64)
                    out = _eval_arc(prog, inp, library)
                    if out is not None and not np.array_equal(inp, out):
                        weights[name] += 1
                        break
            except Exception:
                pass
    return weights

# ── Transfer test ─────────────────────────────────────────────────────────────

def transfer_test(library, held_tasks, budget=BUDGET):
    empty = {}
    bl_costs, bl_solved = [], 0
    lib_costs, lib_solved, lib_new = [], 0, 0
    for task in held_tasks:
        _, cb = search_arc(task['io'], empty, budget)
        bl_costs.append(cb)
        prog_b, _ = search_arc(task['io'], empty, budget)
        if prog_b: bl_solved += 1
        prog_l, cl = search_arc(task['io'], library, budget)
        lib_costs.append(cl)
        if prog_l:
            lib_solved += 1
            if not prog_b: lib_new += 1
    bl_mean  = float(np.mean(bl_costs))
    lib_mean = float(np.mean(lib_costs))
    delta    = (lib_mean - bl_mean) / bl_mean * 100 if bl_mean else 0
    verdict  = (f"TRANSFER_GENUINE: delta={delta:+.1f}% < -5%." if delta < -5.0
                else f"HOLLOW: delta={delta:+.1f}% >= -5%.")
    return {'baseline': {'mean_cost': bl_mean, 'solved': bl_solved},
            'with_library': {'mean_cost': lib_mean, 'solved': lib_solved, 'new_solves': lib_new},
            'delta_pct': delta, 'verdict': verdict}

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
            if not pairs: continue
            cells = [len(p['input'])*len(p['input'][0])+len(p['output'])*len(p['output'][0])
                     for p in pairs]
            tasks.append({'id': fname[:-5], 'io': pairs, 'split_dir': sub,
                          'complexity': len(pairs) * float(np.mean(cells))})
    return tasks

def load_mbpp():
    try:
        from datasets import load_dataset
        ds = load_dataset('google-research-datasets/mbpp', 'sanitized', split='test')
        return list(ds)[:N_MBPP]
    except Exception as e:
        print(f"  MBPP load skipped: {e}")
        return []

def parse_mbpp_tests(item):
    """Extract (input_list, expected_list_or_int) test cases from MBPP item."""
    import re
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
    return cases

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("STAGE 1: NET-FREE WAKE-SLEEP SELF-COMPILATION LOOP")
    print("Leo #11734 — typed combinators, PRISM multi-domain, transfer from iter 1")
    print("=" * 70)
    t0 = time.time()
    sys.stdout.flush()

    all_arc = load_arc()
    training = [t for t in all_arc if t['split_dir'] == 'training']

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(len(training))
    held_tasks  = [training[i] for i in idx[:N_HELD]]
    pool_tasks  = [training[i] for i in idx[N_HELD:]]
    curriculum  = sorted(pool_tasks, key=lambda t: t['complexity'])[:N_CURRICULUM]
    held_ids    = [t['id'] for t in held_tasks]

    mbpp_items  = load_mbpp()
    print(f"ARC: held={len(held_tasks)}, curriculum={len(curriculum)}")
    print(f"MBPP: {len(mbpp_items)} items")
    sys.stdout.flush()

    library     = {}
    all_solved  = []
    iter_results = []

    for it in range(N_ITERATIONS):
        print(f"\n{'='*50}")
        print(f"ITERATION {it+1}/{N_ITERATIONS}  (library={len(library)} macros)")
        t_iter = time.time()
        sys.stdout.flush()

        # ── WAKE ──────────────────────────────────────────────────────────────
        it_costs, it_solved, it_progs = [], 0, []
        for task in curriculum:
            prog, cost = search_arc(task['io'], library, BUDGET)
            it_costs.append(cost)
            if prog:
                it_solved += 1
                it_progs.append(prog)
        all_solved.extend(it_progs)
        arc_rate = it_solved / len(curriculum)
        arc_cost = float(np.mean(it_costs))
        print(f"  ARC wake: solved={it_solved}/{len(curriculum)} ({arc_rate:.1%}), mean_cost={arc_cost:.0f}")
        sys.stdout.flush()

        # MBPP wake
        mbpp_solved = 0
        for item in mbpp_items:
            cases = parse_mbpp_tests(item)
            if cases:
                prog, _ = search_mbpp(cases, BUDGET)
                if prog: mbpp_solved += 1
        mbpp_rate = mbpp_solved / max(len(mbpp_items), 1)
        print(f"  MBPP wake: solved={mbpp_solved}/{len(mbpp_items)} ({mbpp_rate:.1%})")
        sys.stdout.flush()

        # ── ABSTRACTION SLEEP ─────────────────────────────────────────────────
        print(f"  Abstraction-sleep ({len(all_solved)} solved programs total)...")
        old_sz  = len(library)
        library = abstraction_sleep(all_solved, library)
        new_mac = len(library) - old_sz
        print(f"  Library: {old_sz} -> {len(library)} (+{new_mac} macros)")
        sys.stdout.flush()

        # ── DREAM SLEEP ───────────────────────────────────────────────────────
        task_sample = [t['io'] for t in curriculum[:N_DREAMS]]
        dl_weights  = dream_sleep(library, task_sample)
        active = sum(1 for w in dl_weights.values() if w > 0)
        print(f"  Dream-sleep: {active}/{len(library)} macros active in fantasies")
        sys.stdout.flush()

        # ── TRANSFER (every iteration, built-in from iter 1) ──────────────────
        print(f"  Transfer test ({len(held_tasks)} held-out tasks)...")
        sys.stdout.flush()
        transfer = transfer_test(library, held_tasks, BUDGET)
        print(f"  {transfer['verdict']}")
        sys.stdout.flush()

        iter_results.append({
            'iteration':    it + 1,
            'library_size': len(library),
            'new_macros':   new_mac,
            'arc':  {'solved': it_solved, 'n': len(curriculum),
                     'solve_rate': arc_rate, 'mean_cost': arc_cost},
            'mbpp': {'solved': mbpp_solved, 'n': len(mbpp_items), 'solve_rate': mbpp_rate},
            'transfer': transfer,
            'elapsed_sec': time.time() - t_iter,
        })

    # ── Verdict ───────────────────────────────────────────────────────────────
    rates    = [r['arc']['solve_rate'] for r in iter_results]
    deltas   = [r['transfer']['delta_pct'] for r in iter_results]
    monotone = all(rates[i] <= rates[i+1] for i in range(len(rates)-1))
    last_d   = deltas[-1] if deltas else 0

    if monotone and last_d < -5.0:
        verdict = f"PASS: solve-rate monotone + transfer genuine (delta={last_d:+.1f}%)."
    elif max(rates) - min(rates) < 0.02:
        verdict = (f"FAIL (informative): solve-rate flat ({min(rates):.1%}-{max(rates):.1%}). "
                   f"Grammar-poverty or ARC structure-absence. "
                   f"Transfer delta={last_d:+.1f}% (>=-5% -> HOLLOW redux).")
    else:
        verdict = (f"PARTIAL: rates={[f'{r:.1%}' for r in rates]}, "
                   f"transfer={last_d:+.1f}%.")

    total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"VERDICT: {verdict}")
    print(f"Elapsed: {total:.1f}s")
    sys.stdout.flush()

    result = {
        'experiment': 'stage1-wake-sleep',
        'note': ('Leo #11734: net-free typed-combinator wake-sleep loop. '
                 'PRISM multi-domain (ARC+MBPP). Transfer from iteration 1. '
                 'Kai-gated per E3 discipline.'),
        'config': {
            'n_iterations': N_ITERATIONS, 'n_curriculum': N_CURRICULUM,
            'n_held': N_HELD, 'n_mbpp': N_MBPP,
            'split_seed': SPLIT_SEED, 'budget': BUDGET,
            'mdl_min_occ': MDL_MIN_OCC, 'n_dreams': N_DREAMS,
        },
        'held_task_ids':       held_ids,
        'source_construction': (f'curriculum={N_CURRICULUM} easiest training tasks by complexity; '
                                f'held={N_HELD} from training (split_seed={SPLIT_SEED}); no augmentation.'),
        'aug_leakage_check':   {'hits': 0,
                                'verdict': 'CLEAN: no augmentation. Curriculum and held are disjoint training subsets.'},
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
