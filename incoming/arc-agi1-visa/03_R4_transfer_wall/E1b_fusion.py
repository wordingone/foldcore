# E1b_fusion.py -- Fair test of intentional reprioritization.
#
# E1 REVISE: memoization (PASSIVE) works; reprioritization was UNTESTABLE because
# the h=0 compound at depth-1 dominated regardless of component ordering.
#
# E1b fixes both issues:
#   1. Anti-SYNTH: colors vary per pair so cellrule/colormap cannot generalize.
#   2. Multiple h=0 candidates at depth-2: {fh, rot} both preserve (k,k) shape
#      on square objects -> both give h=0 after crop. h-heuristic under-determines.
#   3. Biased family [rot x8, fh x2]: after 8 rot tasks, rot.success >> fh.success.
#      INTENTIONAL v2 (success-weighted heap) correctly prioritizes rot -> speedup
#      on rot tasks, slight slowdown on fh tasks. Net: PASS if 7 wins > 2 losses.
#
# Two solver implementations:
#   v1: original SUBSTRATE heap (push order does NOT affect heap ordering;
#       alphabetical seq tiebreak swamps component-priority)
#   v2: success-weighted heap (success count as explicit priority tiebreaker;
#       the "correct" ANIMA implementation)
#
# Kill criterion (pre-registered):
#   PASS             -- v1-INTENTIONAL beats v1-PASSIVE (original heap sufficient)
#   PASS-v2          -- v2-INTENTIONAL beats v2-PASSIVE but v1-INTENTIONAL = v1-PASSIVE
#                       -> intentional reprioritization is load-bearing IF heap priority
#                       incorporates success counts; GO E2 with v2 heap
#   STRONGER-REVISE  -- INTENTIONAL = PASSIVE in BOTH v1 and v2 on this family
#                       -> shape-distance heuristic swamps component-priority even
#                       with proper weighting; ESCALATE before E2
#
# Run from 03_R4_transfer_wall/ where SUBSTRATE.py and ARC-AGI/ exist.
import json, numpy as np, time, heapq
from scipy import ndimage
from collections import Counter

# -- Load SUBSTRATE -----------------------------------------------------------
substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

# -- Family design: {fh, rot}, fh < rot alphabetically -----------------------
# 8 rot tasks then 2 fh tasks. After task 1+, rot.success > fh.success.
# v2-INTENTIONAL prioritizes rot -> win on rot tasks; lose on fh tasks.
# Net: 7 wins (tasks 2-8) vs 2 losses (tasks 9-10). Expected PASS-v2.
SECOND_OPS = ['rot'] * 8 + ['fh'] * 2

def make_pair(rng, second_op):
    """output = second_op(crop(input)) on a (k,k) square object.
    FRESH colors per call: prevents cellrule/colormap SYNTHS from generalizing.
    fh and rot both preserve (k,k) -> both h=0 at depth-2 from crop state.
    """
    c1 = int(rng.randint(1, 9))
    c2 = int(rng.randint(1, 9))
    while c2 == c1:
        c2 = int(rng.randint(1, 9))
    k = 3  # fixed square size 3x3 to simplify
    for _ in range(60):
        H = int(rng.randint(9, 14))
        W = int(rng.randint(9, 14))
        r0 = int(rng.randint(2, H - k - 2))
        c0 = int(rng.randint(2, W - k - 2))
        grid = np.zeros((H, W), dtype=int)
        obj = np.full((k, k), c1, dtype=int)
        # Asymmetric accent pattern (non-palindrome first row):
        #   row0: [c2, c1, c1]   row1: [c1, c1, c2]   row2: [c2, c2, c1]
        # Ensures fh/fv/rot/tr each produce distinct first-row tobytes[:24]
        # so the SUBSTRATE seen-set fingerprint doesn't falsely skip rot or fh.
        obj[0, 0] = c2   # top-left
        obj[1, 2] = c2   # middle-right
        obj[2, 0] = c2   # bottom-left
        obj[2, 1] = c2   # bottom-middle
        grid[r0:r0+k, c0:c0+k] = obj
        try:
            cropped = BASIS['crop'](grid)
            if cropped.shape != (k, k):
                continue
            out = BASIS[second_op](cropped)
            if out is None or out.size == 0:
                continue
            return grid.tolist(), out.tolist()
        except:
            continue
    return None


def make_family(n=10):
    tasks = []
    seed = 0
    while len(tasks) < n:
        rng = np.random.RandomState(seed)
        seed += 1
        second_op = SECOND_OPS[len(tasks)]
        pairs = []
        # Each pair gets its own fresh rng state -> different colors -> anti-SYNTH
        for _ in range(60):
            p = make_pair(rng, second_op)
            if p is not None:
                pairs.append(p)
            if len(pairs) >= 3:
                break
        if len(pairs) < 3:
            continue
        test_task = {
            'train': [{'input': i, 'output': o} for i, o in pairs[:2]],
            'test':  [{'input': pairs[2][0], 'output': pairs[2][1]}],
        }
        # Pre-verify: solvable by BASIS heap search (no SYNTH shortcut)
        S_test = Substrate()
        solved, seq, exp = _solve_raw(S_test, test_task, budget=400, max_depth=2)
        if not solved:
            continue
        # Reject if SYNTHS solved it (seq contains non-BASIS name)
        if not all(nm in BASIS for nm in seq):
            continue
        tasks.append({'task': test_task, 'second_op': second_op, 'verify_exp': exp})
    return tasks


# -- v1 solver: original SUBSTRATE heap (push order irrelevant to heap order) --
def _solve_raw(s, d, budget=1000, max_depth=3):
    """Passive insertion-order heap search, NO SYNTHS. Used for pre-verification."""
    tr = [(G(p['input']), G(p['output'])) for p in d['train']]
    ti = G(d['test'][0]['input']); to = G(d['test'][0]['output'])
    tgt = tr[0][1].shape
    order = list(s.prims)
    frontier = [(0, (), tr[0][0])]; exp = 0; seen = set()
    while frontier and exp < budget:
        _, seq, cur = heapq.heappop(frontier); exp += 1
        try:
            tp = [(s.ap(seq, i), o) for i, o in tr]
            if (cur.shape == tgt and
                    all(np.array_equal(r, o) for r, o in tp) and
                    np.array_equal(s.ap(seq, ti), to)):
                s._absorb(seq)
                for n in seq:
                    s.success[n] += 1
                return True, seq, exp
        except: pass
        if len(seq) >= max_depth: continue
        for nm in order:
            try:
                nxt = s.prims[nm](cur)
                if nxt.size == 0 or nxt.size > 1600: continue
                k = (nxt.shape, nxt.tobytes()[:24])
                if k in seen: continue
                seen.add(k)
                h = abs(nxt.shape[0]-tgt[0]) + abs(nxt.shape[1]-tgt[1])
                heapq.heappush(frontier, (h*2+len(seq), seq+(nm,), nxt))
            except: pass
    return False, (), exp


def solve_v1(s, d, budget=1000, max_depth=3, priority_mode='passive'):
    """v1 solver: includes SYNTHS; priority_mode changes push ORDER only.
    For h-equal nodes heap uses alphabetical tiebreak on seq -- push order irrelevant.
    """
    tr = [(G(p['input']), G(p['output'])) for p in d['train']]
    ti = G(d['test'][0]['input']); to = G(d['test'][0]['output'])
    tgt = tr[0][1].shape

    for syn in SYNTHS:
        try:
            got = syn(tr)
            if got:
                nm, fn = got
                if np.array_equal(fn(ti), to):
                    s.success[nm] += 1
                    return True, (nm,), 0
        except: pass

    if priority_mode == 'intentional':
        order = sorted(s.prims, key=lambda n: -s.success.get(n, 0))
    else:
        order = list(s.prims)

    frontier = [(0, (), tr[0][0])]; exp = 0; seen = set()
    while frontier and exp < budget:
        _, seq, cur = heapq.heappop(frontier); exp += 1
        try:
            tp = [(s.ap(seq, i), o) for i, o in tr]
            if (cur.shape == tgt and
                    all(np.array_equal(r, o) for r, o in tp) and
                    np.array_equal(s.ap(seq, ti), to)):
                s._absorb(seq)
                for n in seq:
                    s.success[n] += 1
                return True, seq, exp
            for syn in SYNTHS[:3]:
                g = syn(tp)
                if g and np.array_equal(g[1](s.ap(seq, ti)), to):
                    s._absorb(seq)
                    return True, seq + (g[0],), exp
        except: pass
        if len(seq) >= max_depth: continue
        for nm in order:
            try:
                nxt = s.prims[nm](cur)
                if nxt.size == 0 or nxt.size > 1600: continue
                k = (nxt.shape, nxt.tobytes()[:24])
                if k in seen: continue
                seen.add(k)
                h = abs(nxt.shape[0]-tgt[0]) + abs(nxt.shape[1]-tgt[1])
                heapq.heappush(frontier, (h*2+len(seq), seq+(nm,), nxt))
            except: pass

    if all(np.array_equal(o, tr[0][1]) for i, o in tr) and np.array_equal(to, tr[0][1]):
        return True, ('constant',), exp
    return False, (), exp


def solve_v2(s, d, budget=1000, max_depth=3, priority_mode='passive'):
    """v2 solver: success count as EXPLICIT heap priority tiebreaker.
    priority = (h*2+depth, -success_of_last_prim, push_counter, seq)
    High-success prims pop FIRST when h is tied -> true INTENTIONAL reprioritization.
    """
    tr = [(G(p['input']), G(p['output'])) for p in d['train']]
    ti = G(d['test'][0]['input']); to = G(d['test'][0]['output'])
    tgt = tr[0][1].shape

    for syn in SYNTHS:
        try:
            got = syn(tr)
            if got:
                nm, fn = got
                if np.array_equal(fn(ti), to):
                    s.success[nm] += 1
                    return True, (nm,), 0
        except: pass

    def ap(seq, x):
        for nm in seq:
            x = s.prims[nm](x)
        return x

    ctr = [0]
    frontier = [(0, 0, ctr[0], (), tr[0][0])]; ctr[0] += 1
    exp = 0; seen = set()
    while frontier and exp < budget:
        _, _, _, seq, cur = heapq.heappop(frontier); exp += 1
        try:
            tp = [(ap(seq, i), o) for i, o in tr]
            if (cur.shape == tgt and
                    all(np.array_equal(r, o) for r, o in tp) and
                    np.array_equal(ap(seq, ti), to)):
                s._absorb(seq)
                for n in seq:
                    s.success[n] += 1
                return True, seq, exp
            for syn in SYNTHS[:3]:
                g = syn(tp)
                if g and np.array_equal(g[1](ap(seq, ti)), to):
                    s._absorb(seq)
                    return True, seq + (g[0],), exp
        except: pass
        if len(seq) >= max_depth: continue
        for nm in s.prims:
            try:
                nxt = s.prims[nm](cur)
                if nxt.size == 0 or nxt.size > 1600: continue
                k = (nxt.shape, nxt.tobytes()[:24])
                if k in seen: continue
                seen.add(k)
                h = abs(nxt.shape[0]-tgt[0]) + abs(nxt.shape[1]-tgt[1])
                # INTENTIONAL: success as tiebreaker; PASSIVE: 0 (falls back to counter)
                sb = -s.success.get(nm, 0) if priority_mode == 'intentional' else 0
                heapq.heappush(frontier, (h*2+len(seq), sb, ctr[0], seq+(nm,), nxt))
                ctr[0] += 1
            except: pass

    if all(np.array_equal(o, tr[0][1]) for i, o in tr) and np.array_equal(to, tr[0][1]):
        return True, ('constant',), exp
    return False, (), exp


# -- Build family --------------------------------------------------------------
print("Building E1b family (N=10, rot x8 + fh x2, varying colors)...")
FAMILY = make_family(10)
print(f"  {len(FAMILY)} tasks constructed.")
for k, t in enumerate(FAMILY):
    inp = np.array(t['task']['train'][0]['input'])
    cropped = BASIS['crop'](inp)
    print(f"  Task {k+1} [{t['second_op']}]: input={inp.shape} "
          f"crop={cropped.shape} verify_exp={t['verify_exp']}")

# -- Run 4 arms x 2 solvers ---------------------------------------------------
BUDGET = 500

ARMS = [
    # (label, persistence, mode, solver_fn)
    ('v1-OFF',          False, 'intentional', solve_v1),
    ('v1-PASSIVE',      True,  'passive',      solve_v1),
    ('v1-INTENTIONAL',  True,  'intentional',  solve_v1),
    ('v2-OFF',          False, 'intentional', solve_v2),
    ('v2-PASSIVE',      True,  'passive',      solve_v2),
    ('v2-INTENTIONAL',  True,  'intentional',  solve_v2),
]

results = {}
print()
print("=" * 72)
print(f"E1b: {len(ARMS)} arms x {len(FAMILY)} tasks  budget={BUDGET}")
print("=" * 72)

for label, persistent, mode, solver_fn in ARMS:
    print(f"\nArm: {label}")
    S = Substrate()
    arm = []
    for k, item in enumerate(FAMILY):
        if not persistent:
            S = Substrate()
        t0 = time.time()
        solved, seq, exp = solver_fn(S, item['task'], budget=BUDGET,
                                     max_depth=3, priority_mode=mode)
        elapsed = time.time() - t0
        grown = len(S.prims) - len(BASIS)
        abs_used = any('+' in nm for nm in seq)
        arm.append({'task': k+1, 'op': item['second_op'], 'solved': solved,
                    'seq': seq, 'exp': exp, 'time': elapsed,
                    'grown': grown, 'abs': abs_used,
                    'success_snap': dict(S.success)})
        flag = ' [ABS]' if abs_used else ''
        print(f"  Task {k+1}[{item['second_op']}]: {'SOLVED' if solved else 'MISS'}{flag} "
              f"exp={exp} seq={seq} grown={grown}")
    results[label] = arm

# -- Speedup table -------------------------------------------------------------
print()
print("=" * 72)
print("Speedup table (exp_count per task per arm)")
print("=" * 72)
header = (f"{'T':>2}  {'op':>4}  "
          f"{'v1OFF':>6} {'v1PAS':>6} {'v1INT':>6}  "
          f"{'v2OFF':>6} {'v2PAS':>6} {'v2INT':>6}")
print(header)
print("-" * len(header))
for k in range(len(FAMILY)):
    op = FAMILY[k]['second_op']
    exps = [results[a][k]['exp'] for a, *_ in ARMS]
    print(f"{k+1:>2}  {op:>4}  " + "  ".join(f"{e:>6}" for e in exps[:3])
          + "   " + "  ".join(f"{e:>6}" for e in exps[3:]))

# -- Task-by-task comparison: does INTENTIONAL beat PASSIVE? -------------------
print()
tasks_range = range(1, len(FAMILY))  # tasks 2..N (task 1 has no prior info)

def win_count(a_pass, a_int):
    wins   = sum(1 for k in tasks_range if results[a_int][k]['exp'] < results[a_pass][k]['exp'])
    equal  = sum(1 for k in tasks_range if results[a_int][k]['exp'] == results[a_pass][k]['exp'])
    losses = sum(1 for k in tasks_range if results[a_int][k]['exp'] > results[a_pass][k]['exp'])
    pass_med = float(np.median([results[a_pass][k]['exp'] for k in tasks_range]))
    int_med  = float(np.median([results[a_int][k]['exp'] for k in tasks_range]))
    return wins, equal, losses, pass_med, int_med

v1_w, v1_e, v1_l, v1_pm, v1_im = win_count('v1-PASSIVE', 'v1-INTENTIONAL')
v2_w, v2_e, v2_l, v2_pm, v2_im = win_count('v2-PASSIVE', 'v2-INTENTIONAL')

print(f"v1 (original heap):         INTL wins={v1_w}/9  equal={v1_e}/9  loses={v1_l}/9  "
      f"PAS_med={v1_pm:.0f}  INT_med={v1_im:.0f}")
print(f"v2 (success-weighted heap): INTL wins={v2_w}/9  equal={v2_e}/9  loses={v2_l}/9  "
      f"PAS_med={v2_pm:.0f}  INT_med={v2_im:.0f}")

# -- Stronger deletion test (v2-INTENTIONAL) -----------------------------------
print()
print("=" * 72)
print("Stronger deletion test: delete grown prims + block re-absorption")
print("=" * 72)

S_full = Substrate()
for item in FAMILY:
    solve_v2(S_full, item['task'], budget=BUDGET, max_depth=3, priority_mode='intentional')
grown_names = [nm for nm in S_full.prims if nm not in BASIS]
print(f"Grown prims after all 10 tasks: {grown_names}")

if not grown_names:
    print("  No grown prims. Deletion N/A.")
else:
    S_del = Substrate()
    solve_v2(S_del, FAMILY[0]['task'], budget=BUDGET, max_depth=3, priority_mode='intentional')
    deleted = [nm for nm in list(S_del.prims) if nm not in BASIS]
    for nm in deleted:
        del S_del.prims[nm]
    S_del._absorb = lambda seq: None   # block re-grow (stronger deletion test)
    print(f"  Deleted: {deleted}  Re-absorption blocked.")
    print(f"  Regressing tasks 2..{len(FAMILY)}:")
    for k in range(1, len(FAMILY)):
        orig = results['v2-INTENTIONAL'][k]['exp']
        _, _, exp_del = solve_v2(S_del, FAMILY[k]['task'], budget=BUDGET,
                                  max_depth=3, priority_mode='intentional')
        ratio = exp_del / max(orig, 1)
        tag = 'REGRESSED' if exp_del > orig * 1.5 else 'same'
        print(f"    Task {k+1}[{FAMILY[k]['second_op']}]: exp={exp_del} "
              f"(orig={orig}, ratio={ratio:.2f}) {tag}")

# -- Kill criterion verdict ----------------------------------------------------
print()
print("=" * 72)
print("KILL CRITERION VERDICT")
print("=" * 72)

SPEEDUP_THRESHOLD = 0.8    # INT_med < 80% of PAS_med
WIN_THRESHOLD     = 5      # >=5/9 task wins

v1_pass_crit = (v1_im < v1_pm * SPEEDUP_THRESHOLD) and (v1_w >= WIN_THRESHOLD)
v2_pass_crit = (v2_im < v2_pm * SPEEDUP_THRESHOLD) and (v2_w >= WIN_THRESHOLD)

if v1_pass_crit:
    verdict = "PASS"
    detail  = ("INTENTIONAL beats PASSIVE on original heap. "
               "Intentional reprioritization load-bearing without heap modification. GO E2.")
elif v2_pass_crit and not v1_pass_crit:
    verdict = "PASS-v2"
    detail  = ("INTENTIONAL beats PASSIVE ONLY with success-weighted heap (v2). "
               "v1 alphabetical tiebreak swamps component-priority. "
               "Intentional reprioritization is load-bearing IF heap explicitly "
               "incorporates success counts. GO E2 with v2 heap.")
else:
    verdict = "STRONGER-REVISE"
    detail  = ("INTENTIONAL = PASSIVE in BOTH v1 and v2. "
               "Shape-distance heuristic fully determines exploration order; "
               "accumulated state contribution is nil even with proper priority weighting. "
               "Intentional premise does not hold for symbolic heap search on this basis. "
               "ESCALATE to Leo before E2.")

print(f"  VERDICT: {verdict}")
print(f"  {detail}")
print()

unsolved = [(a, k+1) for a in results
            for k, r in enumerate(results[a]) if not r['solved']]
if unsolved:
    print(f"  WARNING: unsolved tasks: {unsolved}")
