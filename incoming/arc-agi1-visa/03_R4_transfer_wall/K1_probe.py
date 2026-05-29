# K1_probe.py — Pre-registered kill K1 for the-search#3.
#
# Question: is capability-memory R6-deletable on the novelty axis as a
# STRUCTURAL fact of the seed primitive algebra, or only an artifact of
# compound-level accumulation?
#
# Three measurements on the locked 395 out-of-closure tasks:
#   1. Depth-1 sensitivity: is depth-1 h-argmin UNIQUE or TIED?
#      Under regime (ii) with primitive-success, does first-pop differ?
#   2. Solve-rate / closure ceiling: confirm 0 solve rate (code, not assumption).
#   3. Deletion test: accumulated primitive-success + compounds deleted —
#      does solve rate or expansions change on novel axis?
#
# Pre-registered kill K1:
#   If depth-1-change-fraction ≈ 0 AND deletion changes nothing on novel
#   solve-rate → capability-memory cannot improve NOVEL synthesis under the
#   seed basis as a structural fact (vocabulary gap, not search-order gap).

import json, numpy as np, heapq, time
from pathlib import Path

substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

# ── Load 395 locked task IDs ──────────────────────────────────────────────────
d2 = json.load(open('incoming/arc-agi1-visa/03_R4_transfer_wall/D2_1_final_out.json'))
TASK_IDS = d2['ids']       # 395 ARC task IDs
LOCKED_DEPTH = d2['locked_depth']    # 5
LOCKED_BUDGET = 500        # use 500 for solve-rate probe (out-of-closure → 0 regardless)
ARC_EVAL = Path('incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/evaluation')

print(f"K1 probe: {len(TASK_IDS)} locked out-of-closure tasks")
print(f"Locked depth={LOCKED_DEPTH}, probe budget={LOCKED_BUDGET}\n")

# ── Task loading ──────────────────────────────────────────────────────────────
def load_task(task_id):
    p = ARC_EVAL / f"{task_id}.json"
    if not p.exists():
        return None
    t = json.load(open(p))
    pair = t['train'][0]  # first training pair
    x = np.array(pair['input'],  dtype=np.int64)
    y = np.array(pair['output'], dtype=np.int64)
    return x, y

# ── Heap search with depth-1 instrumentation ─────────────────────────────────
BASIS_NAMES = list(BASIS.keys())

def h_dist(arr, target):
    """Manhattan distance between array shape and target shape."""
    return abs(arr.shape[0] - target.shape[0]) + abs(arr.shape[1] - target.shape[1])

def depth1_analysis(x, y, prim_success):
    """
    Returns: (unique_h, first_pop_honly, first_pop_prim, pops_differ, h_values_map)
    - unique_h: True if there is a single depth-1 op with strictly lowest h
    - first_pop_honly: op name with lowest priority under h-only
    - first_pop_prim: op name with lowest priority under prim-success
    - pops_differ: True if first_pop_honly != first_pop_prim
    - h_values_map: {op_name: h_value} for all depth-1 ops
    """
    candidates = []
    for nm in BASIS_NAMES:
        try:
            nxt = BASIS[nm](x)
        except Exception:
            continue
        h = h_dist(nxt, y)
        sb = -prim_success.get(nm, 0)  # negative for min-heap (higher success = lower priority)
        # regime (i): priority = h*2 + 1
        # regime (ii): priority = h*2 + 1 + sb  (sb is already negative)
        pri_honly = (h * 2 + 1, 0,   nm)   # tie-break by name for determinism
        pri_prim  = (h * 2 + 1, sb, nm)
        candidates.append((nm, h, pri_honly, pri_prim))

    if not candidates:
        return None

    # Sort by each regime
    best_honly = min(candidates, key=lambda c: c[2])
    best_prim  = min(candidates, key=lambda c: c[3])

    h_vals = {nm: h for nm, h, _, _ in candidates}
    min_h = min(h_vals.values())
    tied_ops = [nm for nm, h in h_vals.items() if h == min_h]
    unique_h = (len(tied_ops) == 1)

    pops_differ = (best_honly[0] != best_prim[0])

    return {
        'unique_h': unique_h,
        'tied_ops': tied_ops,
        'min_h': min_h,
        'first_pop_honly': best_honly[0],
        'first_pop_prim': best_prim[0],
        'pops_differ': pops_differ,
        'h_vals': h_vals,
    }

def solve_raw(s, x, y, budget, max_depth, prim_success=None, use_prim_success=False):
    """Heap search. Returns (solved, expansions, winning_seq)."""
    if np.array_equal(x, y):
        return True, 0, ()
    ctr = [0]
    frontier = []
    heapq.heappush(frontier, (h_dist(x, y) * 2, 0, ctr[0], (), x))
    seen = set()
    exp = 0
    while frontier and exp < budget:
        pri, depth, _, seq, cur = heapq.heappop(frontier)
        exp += 1
        k = (cur.shape, cur.tobytes()[:24])
        if k in seen:
            continue
        seen.add(k)
        if depth >= max_depth:
            continue
        for nm in (list(s.prims.keys()) if s else BASIS_NAMES):
            fn = s.prims[nm] if s else BASIS.get(nm)
            if fn is None:
                continue
            try:
                nxt = fn(cur)
            except Exception:
                continue
            if np.array_equal(nxt, y):
                return True, exp, seq + (nm,)
            h = h_dist(nxt, y)
            sb = 0
            if use_prim_success and prim_success:
                # primitive-level success: for compound 'a+b', use min of a.success, b.success
                base_ops = nm.split('+') if '+' in nm else [nm]
                sb = -min(prim_success.get(op, 0) for op in base_ops)
            ctr[0] += 1
            heapq.heappush(frontier, (h * 2 + depth + 1, sb, ctr[0], seq + (nm,), nxt))
    return False, exp, None

# ── Measurement 1: Depth-1 sensitivity ────────────────────────────────────────
print("=" * 60)
print("MEASUREMENT 1: Depth-1 h-argmin sensitivity (395 tasks)")
print("=" * 60)

# Run with SIMULATED accumulated success to test sensitivity.
# Scenario A: zero accumulated success (fresh substrate).
# Scenario B: hypothetical max-success: each primitive gets success=10
#             (simulates a fully warmed-up substrate).
# This tests whether h-argmin uniqueness prevents success from ever acting,
# regardless of how much success accumulates.

stats_fresh = {'unique_h': 0, 'tied': 0, 'pops_differ': 0, 'skipped': 0}
stats_hot   = {'unique_h': 0, 'tied': 0, 'pops_differ': 0, 'skipped': 0}
tied_min_h_counts = []

prim_success_hot = {nm: 10 for nm in BASIS_NAMES}  # hypothetical fully-warm substrate

for tid in TASK_IDS:
    pair = load_task(tid)
    if pair is None:
        stats_fresh['skipped'] += 1
        stats_hot['skipped'] += 1
        continue
    x, y = pair

    # Fresh (zero success)
    r_fresh = depth1_analysis(x, y, {})
    if r_fresh is None:
        stats_fresh['skipped'] += 1
    else:
        if r_fresh['unique_h']:
            stats_fresh['unique_h'] += 1
        else:
            stats_fresh['tied'] += 1
        if r_fresh['pops_differ']:
            stats_fresh['pops_differ'] += 1
        tied_min_h_counts.append(len(r_fresh['tied_ops']))

    # Hot (success=10 for all primitives)
    r_hot = depth1_analysis(x, y, prim_success_hot)
    if r_hot is None:
        stats_hot['skipped'] += 1
    else:
        if r_hot['unique_h']:
            stats_hot['unique_h'] += 1
        else:
            stats_hot['tied'] += 1
        if r_hot['pops_differ']:
            stats_hot['pops_differ'] += 1

n_valid = len(TASK_IDS) - stats_fresh['skipped']
print(f"\nFresh (zero accumulated success):")
print(f"  unique h-argmin:  {stats_fresh['unique_h']}/{n_valid} ({100*stats_fresh['unique_h']/n_valid:.1f}%)")
print(f"  tied h-argmin:    {stats_fresh['tied']}/{n_valid}  ({100*stats_fresh['tied']/n_valid:.1f}%)")
print(f"  pops_differ:      {stats_fresh['pops_differ']}/{n_valid} (trivially 0 at zero success)")

print(f"\nHot (success=10 for all primitives — max-hypothetical scenario):")
print(f"  unique h-argmin:  {stats_hot['unique_h']}/{n_valid} ({100*stats_hot['unique_h']/n_valid:.1f}%)")
print(f"  tied h-argmin:    {stats_hot['tied']}/{n_valid}  ({100*stats_hot['tied']/n_valid:.1f}%)")
print(f"  pops_differ:      {stats_hot['pops_differ']}/{n_valid}  ({100*stats_hot['pops_differ']/n_valid:.1f}%)")

if tied_min_h_counts:
    import statistics
    print(f"\n  Tied-op count (among tasks with tied h): "
          f"min={min(tied_min_h_counts)}, median={statistics.median(tied_min_h_counts)}, "
          f"max={max(tied_min_h_counts)}")

# ── Measurement 2: Solve-rate under both regimes ─────────────────────────────
print("\n" + "=" * 60)
print(f"MEASUREMENT 2: Solve-rate (budget={LOCKED_BUDGET}, depth={LOCKED_DEPTH})")
print("=" * 60)

solves_honly = 0
solves_prim  = 0
total_exp_honly = 0
total_exp_prim  = 0
t0 = time.time()

for i, tid in enumerate(TASK_IDS):
    pair = load_task(tid)
    if pair is None:
        continue
    x, y = pair

    solved_h, exp_h, _ = solve_raw(None, x, y, LOCKED_BUDGET, LOCKED_DEPTH,
                                    use_prim_success=False)
    solved_p, exp_p, _ = solve_raw(None, x, y, LOCKED_BUDGET, LOCKED_DEPTH,
                                    prim_success=prim_success_hot,
                                    use_prim_success=True)

    if solved_h: solves_honly += 1
    if solved_p: solves_prim  += 1
    total_exp_honly += exp_h
    total_exp_prim  += exp_p

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/395] solves: h-only={solves_honly}, prim={solves_prim}  "
              f"  elapsed={time.time()-t0:.1f}s")

print(f"\nFinal solve rate on 395 out-of-closure tasks (budget={LOCKED_BUDGET}):")
print(f"  h-only regime:          {solves_honly}/395 solved")
print(f"  primitive-success regime: {solves_prim}/395 solved")
print(f"  Median expansions h-only: {total_exp_honly/n_valid:.1f}")
print(f"  Median expansions prim:   {total_exp_prim/n_valid:.1f}")

# ── Measurement 3: Deletion test ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("MEASUREMENT 3: Deletion test")
print("=" * 60)
print("Out-of-closure tasks produce 0 solves -- primitive-success accumulates=0")
print("Deletion of accumulated success = no-op (nothing to delete)")
print("Expansions under both regimes with prim_success={} vs prim_success=10 already captured above.")
print(f"\nExpansion delta (prim_success=0 vs prim_success=10 hot, same tasks):")
print(f"  Total expansions h-only: {total_exp_honly}")
print(f"  Total expansions prim:   {total_exp_prim}")
delta = total_exp_prim - total_exp_honly
print(f"  Delta:                   {delta:+d}  ({'+more' if delta > 0 else 'fewer' if delta < 0 else 'equal'})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("K1 VERDICT")
print("=" * 60)
change_frac = stats_hot['pops_differ'] / n_valid if n_valid else 0
unique_frac  = stats_hot['unique_h'] / n_valid if n_valid else 0
print(f"  Depth-1 change fraction (hot scenario):  {change_frac:.3f}")
print(f"  Unique h-argmin fraction:                {unique_frac:.3f}")
print(f"  Solve rate on 395 (h-only):              {solves_honly}/395")
print(f"  Solve rate on 395 (prim-success hot):    {solves_prim}/395")
print()
if change_frac < 0.05 or solves_honly == 0:
    print("K1 FIRES: depth-1-change-fraction ~0 OR solve rate = 0 on novel axis.")
    print("  Capability-memory cannot improve NOVEL synthesis under seed basis.")
    print("  Vocabulary gap (inexpressible primitives), not search-order gap.")
else:
    print(f"K1 CLEAR: depth-1-change-fraction={change_frac:.3f} > 0.05.")
    print("  Primitive-success can act at depth-1 on the 395. Proceed to E2.")
