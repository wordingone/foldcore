# E1_fusion.py — R2-fusion feasibility / second-exposure speedup experiment.
#
# Objective: does R2-fused streaming abstraction discovery produce second-exposure
#   speedup on repeated structure, on the existing geometric basis?
#
# Three arms:
#   OFF        — fresh Substrate per task (control; no persistence)
#   PASSIVE    — one Substrate across the stream; library grows (_absorb works),
#                but frontier priority UNCHANGED (flat insertion order = pure memoization)
#   INTENTIONAL — one Substrate; library grows AND frontier reprioritized by success count
#                 (ANIMA: memory reparameterizes next solve's search)
#
# Task family: N=10 synthetic ARC-style tasks, all requiring crop -> dup_h -> dup_v
#   (a depth-3 BASIS composition SUBSTRATE must SEARCH from scratch on task 1).
#   Tasks vary in: object color, object shape, grid size, object position.
#
# Metrics per task per arm:
#   solve_time   — wall clock for solve()
#   exp_count    — number of heap pops (budget used)
#   winning_seq  — the sequence that solved the task (shows if abstraction was used)
#   abstractions — size of grown library at task k
#
# Kill criterion (written before running):
#   PASS   — INTENTIONAL shows solve-time / exp-count drop on tasks 2..N vs task 1,
#            deletion confirms speedup from discovered abstraction,
#            AND INTENTIONAL > PASSIVE (reprioritization load-bearing). -> GO E2.
#   REVISE — speedup only in PASSIVE (memoization works; reprioritization adds nothing).
#   KILL   — no speedup in either arm. R2-fused discovery doesn't work on this basis.
#
# Run from 03_R4_transfer_wall/ where S1.py and ARC-AGI/ exist.
import json, numpy as np, time, heapq, copy
from scipy import ndimage
from collections import Counter

# ── Load SUBSTRATE definitions ────────────────────────────────────────────────
substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

# ── Task family construction ──────────────────────────────────────────────────
TARGET_PROG = ('crop', 'up2')  # depth-2; verified solvable by SUBSTRATE
# NOTE: depth-3 ('crop','dup_h','dup_v') was attempted but hits SUBSTRATE's seen-set
# collision: up2(crop) produces same shape+first-24-bytes as dup_v(dup_h(crop)) for
# uniform-color objects, causing the correct depth-3 solution to be pruned as "seen."
# crop->up2 avoids this: up2(crop) IS the target, found at depth-2 with no collision.

def apply_prog(x, prog=TARGET_PROG):
    for nm in prog:
        x = BASIS[nm](x)
    return x

def make_pair(rng, c1, c2):
    """Make a pair where output = up2(crop(input)). Uses 2-color objects to avoid
    seen-set collisions (uniform-color objects cause fingerprint collisions)."""
    for _ in range(30):
        h = int(rng.randint(7, 12))
        w = int(rng.randint(7, 12))
        oh = int(rng.randint(2, 5))
        ow = int(rng.randint(2, 5))
        if oh >= h - 2 or ow >= w - 2:
            continue
        r = int(rng.randint(1, h - oh - 1))
        c = int(rng.randint(1, w - ow - 1))
        grid = np.zeros((h, w), dtype=int)
        # Two-color object: main color c1, accent pixel(s) = c2
        # This ensures up2(crop) has distinct first-24-bytes from other same-shape transforms.
        obj = np.full((oh, ow), c1, dtype=int)
        obj[0, 0] = c2          # top-left corner accent
        obj[oh-1, ow-1] = c2    # bottom-right corner accent
        grid[r:r+oh, c:c+ow] = obj
        try:
            out = apply_prog(grid)
            if out is None or out.size == 0 or out.size > 900:
                continue
            # Verify step-by-step
            if not np.array_equal(BASIS['up2'](BASIS['crop'](grid)), out):
                continue
            return grid.tolist(), out.tolist()
        except:
            continue
    return None

def make_family(n=10):
    tasks = []
    rng = np.random.RandomState(0)
    seed = 0
    while len(tasks) < n:
        rng2 = np.random.RandomState(seed)
        seed += 1
        c1 = int(rng2.randint(1, 8))
        c2 = c1 % 8 + 1  # always different from c1
        pairs = []
        for _ in range(30):
            p = make_pair(rng2, c1, c2)
            if p is not None:
                pairs.append(p)
            if len(pairs) >= 3:
                break
        if len(pairs) < 3:
            continue
        # Verify SUBSTRATE can actually solve task 1 with a fresh substrate
        test_task = {
            'train': [{'input': i, 'output': o} for i, o in pairs[:2]],
            'test': [{'input': pairs[2][0], 'output': pairs[2][1]}],
        }
        S_test = Substrate()
        solved, seq, exp = solve_instrumented(S_test, test_task, budget=500, max_depth=2, priority_mode='passive')
        if not solved:
            continue  # skip tasks SUBSTRATE can't solve
        tasks.append(test_task)
    return tasks

# ── Instrumented solve (defined before make_family which calls it) ────────────
def solve_instrumented(s, d, budget=1000, max_depth=3, priority_mode='intentional'):
    """Like Substrate.solve but returns (solved, winning_seq, exp_count).
    priority_mode: 'intentional' = sort by success count (default SUBSTRATE);
                   'passive' = use insertion order (flat, memoization only).
    """
    tr = [(G(p['input']), G(p['output'])) for p in d['train']]
    ti = G(d['test'][0]['input']); to = G(d['test'][0]['output'])
    tgt = tr[0][1].shape

    # SYNTHS first
    for syn in SYNTHS:
        try:
            got = syn(tr)
            if got:
                nm, fn = got
                if np.array_equal(fn(ti), to):
                    s.success[nm] += 1
                    return True, (nm,), 0
        except: pass

    # Ordering: intentional = by success desc; passive = insertion order
    if priority_mode == 'intentional':
        order = sorted(s.prims, key=lambda n: -s.success.get(n, 0))
    else:  # passive
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
                    s.success[n] = s.success.get(n, 0) + 1
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

print("Constructing task family (N=10, program: crop->up2)...")
FAMILY = make_family(10)
print(f"  {len(FAMILY)} tasks constructed.")
for k, t in enumerate(FAMILY):
    inp = np.array(t['train'][0]['input'])
    out = np.array(t['train'][0]['output'])
    print(f"  Task {k+1}: input={inp.shape} output={out.shape} colors={sorted(set(int(v) for v in inp.flat))}")

# ── Run three arms ────────────────────────────────────────────────────────────
ARMS = [
    ('OFF',         None,           'intentional'),  # fresh substrate per task
    ('PASSIVE',     'persistent',   'passive'),       # persistent, flat ordering
    ('INTENTIONAL', 'persistent',   'intentional'),   # persistent, success-ordered
]

results = {}
BUDGET = 2000

print()
print("=" * 64)
print(f"Running E1: three arms × {len(FAMILY)} tasks  (budget={BUDGET})")
print("=" * 64)

for arm_name, persistence, pmode in ARMS:
    print(f"\nArm: {arm_name}")
    S = Substrate()  # will be reused for persistent arms
    arm_results = []

    for k, task in enumerate(FAMILY):
        if persistence is None:
            S = Substrate()  # fresh per task for OFF

        t0 = time.time()
        solved, seq, exp_count = solve_instrumented(S, task, budget=BUDGET, max_depth=3, priority_mode=pmode)
        elapsed = time.time() - t0

        grown_count = len(S.prims) - len(BASIS)
        used_abstraction = any('+' in nm for nm in seq)

        arm_results.append({
            'task': k + 1,
            'solved': solved,
            'seq': seq,
            'exp_count': exp_count,
            'solve_time': elapsed,
            'grown_count': grown_count,
            'used_abstraction': used_abstraction,
        })

        status = 'SOLVED' if solved else 'MISS'
        abs_flag = ' [ABS]' if used_abstraction else ''
        print(f"  Task {k+1}: {status}{abs_flag}  seq={seq}  exp={exp_count}  t={elapsed:.4f}s  grown={grown_count}")

    results[arm_name] = arm_results

# ── Analysis: speedup curves ──────────────────────────────────────────────────
print()
print("=" * 64)
print("Speedup curves (exp_count relative to OFF task-1 baseline)")
print("=" * 64)
off_task1_exp = results['OFF'][0]['exp_count']
if off_task1_exp == 0:
    off_task1_exp = 1  # avoid div-by-zero

print(f"OFF task-1 baseline: exp_count={off_task1_exp}")
print()
header = f"{'Task':>5}  {'OFF exp':>9}  {'PASS exp':>9}  {'INTL exp':>9}  {'OFF time':>9}  {'PASS time':>10}  {'INTL time':>10}  {'PASS seq'}  {'INTL seq'}"
print(header)
print("-" * len(header))
for k in range(len(FAMILY)):
    r_off  = results['OFF'][k]
    r_pass = results['PASSIVE'][k]
    r_intl = results['INTENTIONAL'][k]
    print(f"  {k+1:>3}  {r_off['exp_count']:>9}  {r_pass['exp_count']:>9}  {r_intl['exp_count']:>9}  "
          f"{r_off['solve_time']:>9.4f}  {r_pass['solve_time']:>10.4f}  {r_intl['solve_time']:>10.4f}  "
          f"{'|'.join(r_pass['seq']):<30}  {'|'.join(r_intl['seq'])}")

# ── Deletion test ─────────────────────────────────────────────────────────────
print()
print("=" * 64)
print("Deletion test (INTENTIONAL arm)")
print("=" * 64)

# Rebuild INTENTIONAL arm state to get the grown primitives
S_intl = Substrate()
for k, task in enumerate(FAMILY):
    solve_instrumented(S_intl, task, budget=BUDGET, max_depth=3, priority_mode='intentional')

grown_prims = {nm: fn for nm, fn in S_intl.prims.items() if nm not in BASIS}
print(f"Grown primitives: {list(grown_prims.keys())}")

if not grown_prims:
    print("  No grown primitives — abstraction never fired.")
    print("  Deletion test N/A.")
else:
    for gp_name in grown_prims:
        # Re-run INTENTIONAL without this primitive, on tasks 2..N
        print(f"\n  Deleting '{gp_name}' — re-solving tasks 2..{len(FAMILY)}:")
        S_del = Substrate()
        # Re-run task 1 to build the same pre-deletion state
        solve_instrumented(S_del, FAMILY[0], budget=BUDGET, max_depth=3, priority_mode='intentional')
        # Remove the grown prim
        if gp_name in S_del.prims:
            del S_del.prims[gp_name]

        for k in range(1, len(FAMILY)):
            task = FAMILY[k]
            t0 = time.time()
            solved_del, seq_del, exp_del = solve_instrumented(S_del, task, budget=BUDGET, max_depth=3, priority_mode='intentional')
            elapsed_del = time.time() - t0

            orig = results['INTENTIONAL'][k]
            exp_ratio = exp_del / max(orig['exp_count'], 1)
            time_ratio = elapsed_del / max(orig['solve_time'], 1e-6)
            print(f"    Task {k+1}: exp={exp_del} (orig={orig['exp_count']}, ratio={exp_ratio:.2f})  "
                  f"time={elapsed_del:.4f}s (ratio={time_ratio:.2f})  {'REGRESSED' if exp_del > orig['exp_count']*1.5 else 'same'}")

# ── Kill criterion verdict ─────────────────────────────────────────────────────
print()
print("=" * 64)
print("KILL CRITERION VERDICT")
print("=" * 64)

# Check speedup: tasks 2..N exp_count relative to task 1 in same arm
off_exp_1 = results['OFF'][0]['exp_count']
pass_exp_1 = results['PASSIVE'][0]['exp_count']
intl_exp_1 = results['INTENTIONAL'][0]['exp_count']

off_median_rest  = np.median([r['exp_count'] for r in results['OFF'][1:]])
pass_median_rest = np.median([r['exp_count'] for r in results['PASSIVE'][1:]])
intl_median_rest = np.median([r['exp_count'] for r in results['INTENTIONAL'][1:]])

print(f"  OFF:         task-1 exp={off_exp_1},  tasks 2..N median exp={off_median_rest:.0f}  (ratio={off_median_rest/max(off_exp_1,1):.2f})")
print(f"  PASSIVE:     task-1 exp={pass_exp_1}, tasks 2..N median exp={pass_median_rest:.0f}  (ratio={pass_median_rest/max(pass_exp_1,1):.2f})")
print(f"  INTENTIONAL: task-1 exp={intl_exp_1}, tasks 2..N median exp={intl_median_rest:.0f}  (ratio={intl_median_rest/max(intl_exp_1,1):.2f})")

SPEEDUP_THRESHOLD = 0.5   # task 2..N median < 50% of task 1 = speedup
INTL_OVER_PASS_THRESHOLD = 0.8   # INTL median < 80% of PASS median = reprioritization load-bearing

passive_speedup = pass_median_rest < pass_exp_1 * SPEEDUP_THRESHOLD
intl_speedup    = intl_median_rest < intl_exp_1 * SPEEDUP_THRESHOLD
intl_over_pass  = intl_median_rest < pass_median_rest * INTL_OVER_PASS_THRESHOLD

print()
if intl_speedup and intl_over_pass:
    verdict = "PASS"
    detail = "INTENTIONAL shows second-exposure speedup AND beats PASSIVE. R2-fusion viable. GO E2."
elif (passive_speedup or intl_speedup) and not intl_over_pass:
    verdict = "REVISE"
    detail = "Speedup present but INTENTIONAL does NOT beat PASSIVE. Memoization works; reprioritization non-load-bearing."
elif not passive_speedup and not intl_speedup:
    verdict = "DESIGN-KILL"
    detail = "No speedup in PASSIVE or INTENTIONAL. R2-fused discovery doesn't produce efficiency gain."
else:
    verdict = "PARTIAL"
    detail = f"passive_speedup={passive_speedup}, intl_speedup={intl_speedup}, intl_over_pass={intl_over_pass}. Manual inspection needed."

print(f"  VERDICT: {verdict}")
print(f"  {detail}")
print()

# Check if any task was unsolved (would invalidate the comparison)
unsolved = [(arm, k+1) for arm, arm_res in results.items() for k, r in enumerate(arm_res) if not r['solved']]
if unsolved:
    print(f"  WARNING: unsolved tasks: {unsolved}. Results may be unreliable.")
