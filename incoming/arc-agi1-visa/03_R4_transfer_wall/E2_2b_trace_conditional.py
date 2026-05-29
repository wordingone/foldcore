# E2_2b_trace_conditional.py — Terminal selection-axis test (the-search#3).
#
# Purpose: close the one unmeasured cell in signal-cleanliness:
#   E2.1 biased core + dirty signal:   meta +0.00 (redundant — core pre-correct)
#   E2.1 uniform core + dirty signal:  meta -0.13 (harmful — noise contamination)
#   E2.2(b) UNIFORM core + CLEAN signal: meta is the SOLE bias source; unambiguous.
#
# Trace-conditional accumulation: only accumulate from sequences that MATCH the
# GROUND-TRUTH generating program. Alternative solutions are discarded.
# This eliminates the noise from underdetermined task families (E2.1 root cause).
#
# Pre-registered kill (K3): deletion-delta <= 0 with clean signal → selection axis
#   DEFINITIVELY dead → hard pivot to E2.2' (coverage-expansion core, C6).
# Pre-registered positive: deletion-delta > 0 → clean-signal meta helps; carry into E2.2'.
#
# Leo directive: #11525. One variable: accumulation cleanliness. Uniform prior = sole bias.
# Refs the-search#3

import json, numpy as np, heapq, time
from collections import defaultdict

substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

BASIS_NAMES = list(BASIS.keys())

# ── Task generation (same as E2.1) ────────────────────────────────────────────
rng_gen = np.random.default_rng(7)

def make_random_grid(rng, min_h=3, max_h=7, min_w=3, max_w=7):
    h = rng.integers(min_h, max_h + 1)
    w = rng.integers(min_w, max_w + 1)
    colors = rng.integers(1, 9, size=3)
    grid = rng.choice(np.append([0], colors), size=(h, w))
    return grid.astype(np.int64)

def apply_program(prog, grid):
    x = grid.copy()
    for op in prog:
        x = BASIS[op](x)
    return x

def generate_task(rng, prog, n_tries=30):
    for _ in range(n_tries):
        try:
            x = make_random_grid(rng)
            y = apply_program(prog, x)
            if (not np.array_equal(x, y)
                    and y.size > 0
                    and y.shape[0] >= 2
                    and y.shape[1] >= 2):
                return x, y
        except Exception:
            continue
    return None

ACCUM_FIRST  = ['crop', 'fh', 'tr']
ACCUM_SECOND = ['rot', 'up2', 'mir_h']
HELD_FIRST   = ['fv', 'dup_h']

ACCUM_PROGRAMS = [(f, s) for f in ACCUM_FIRST  for s in ACCUM_SECOND]
HELD_PROGRAMS  = [(f, s) for f in HELD_FIRST   for s in ACCUM_SECOND]

N_ACCUM_PER = 10
N_HELD_PER  = 5

accum_tasks = []
for prog in ACCUM_PROGRAMS:
    for _ in range(N_ACCUM_PER):
        t = generate_task(rng_gen, prog)
        if t:
            accum_tasks.append((prog, t[0], t[1]))

held_tasks = []
for prog in HELD_PROGRAMS:
    for _ in range(N_HELD_PER):
        t = generate_task(rng_gen, prog)
        if t:
            held_tasks.append((prog, t[0], t[1]))

print(f"Tasks: {len(accum_tasks)} accum, {len(held_tasks)} held-out")

# ── UNIFORM prior (meta is the sole bias source) ──────────────────────────────
UNIFORM_D0 = {nm: 1.0 / len(BASIS_NAMES) for nm in BASIS_NAMES}
UNIFORM_D1 = {nm: 1.0 / len(BASIS_NAMES) for nm in BASIS_NAMES}

# ── Trace-conditional meta-layer ──────────────────────────────────────────────
class TraceConditionalMeta:
    """Only accumulate from sequences that MATCH the ground-truth program."""

    def __init__(self):
        self.depth2_success = defaultdict(int)
        self.accumulated = 0
        self.rejected = 0

    def update(self, solved_seq, ground_truth_prog):
        """
        Accumulate only if solved_seq matches ground_truth_prog (d0, d1).
        Rejects: alternative solutions that solve the task but aren't ground truth.
        """
        gt = tuple(ground_truth_prog)
        if len(solved_seq) >= 2 and solved_seq[:2] == gt:
            self.depth2_success[solved_seq[1]] += 1
            self.accumulated += 1
        else:
            self.rejected += 1

    def get_boost(self, nm):
        return 1.0 + self.depth2_success.get(nm, 0)

    def get_weighted_d1(self, core_d1):
        raw = {nm: core_d1[nm] * self.get_boost(nm) for nm in BASIS_NAMES}
        total = sum(raw.values())
        return {nm: raw[nm] / total for nm in BASIS_NAMES}

def h_dist(a, b):
    return abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])

def solve(x, y, budget, max_depth, core_d0, core_d1, meta=None):
    if np.array_equal(x, y):
        return True, 0, ()
    eff_d1 = meta.get_weighted_d1(core_d1) if meta else core_d1
    ctr = [0]
    frontier = []
    heapq.heappush(frontier, (h_dist(x, y) * 2, 0.0, ctr[0], (), x))
    seen = set()
    exp = 0
    while frontier and exp < budget:
        pri, neg_s, _, seq, cur = heapq.heappop(frontier)
        exp += 1
        k = (cur.shape, cur.tobytes()[:24])
        if k in seen:
            continue
        seen.add(k)
        d = len(seq)
        if d >= max_depth:
            continue
        for nm in BASIS_NAMES:
            try:
                nxt = BASIS[nm](cur)
            except Exception:
                continue
            if np.array_equal(nxt, y):
                return True, exp, seq + (nm,)
            h = h_dist(nxt, y)
            prob = core_d0[nm] if d == 0 else eff_d1[nm]
            ctr[0] += 1
            heapq.heappush(frontier, (h * 2 + d + 1, -prob, ctr[0], seq + (nm,), nxt))
    return False, exp, None

# ── Accumulation stream ───────────────────────────────────────────────────────
BUDGET    = 500
MAX_DEPTH = 3

print("\n" + "=" * 60)
print("ACCUMULATION STREAM (trace-conditional, uniform core)")
print("=" * 60)

meta = TraceConditionalMeta()
accum_exp_core, accum_exp_meta = [], []

for prog, x, y in accum_tasks:
    _, exp_c, _      = solve(x, y, BUDGET, MAX_DEPTH, UNIFORM_D0, UNIFORM_D1, meta=None)
    _, exp_m, seq_m  = solve(x, y, BUDGET, MAX_DEPTH, UNIFORM_D0, UNIFORM_D1, meta=meta)
    if seq_m:
        meta.update(seq_m, prog)  # trace-conditional: only accumulate if matches ground truth
    accum_exp_core.append(exp_c)
    accum_exp_meta.append(exp_m)

print(f"CORE_ONLY  mean exp = {np.mean(accum_exp_core):.2f}")
print(f"CORE_META  mean exp = {np.mean(accum_exp_meta):.2f}")
print(f"Accumulated traces: {meta.accumulated} (match ground truth)")
print(f"Rejected traces:    {meta.rejected} (alternative solutions)")
print(f"Meta depth2_success: {dict(meta.depth2_success)}")

# ── Held-out evaluation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT EVALUATION (meta fixed, uniform core)")
print("=" * 60)

held_exp_c, held_exp_m = [], []
held_solve_c = held_solve_m = 0
for prog, x, y in held_tasks:
    sc, ec, _ = solve(x, y, BUDGET, MAX_DEPTH, UNIFORM_D0, UNIFORM_D1, meta=None)
    sm, em, _ = solve(x, y, BUDGET, MAX_DEPTH, UNIFORM_D0, UNIFORM_D1, meta=meta)
    held_exp_c.append(ec); held_exp_m.append(em)
    if sc: held_solve_c += 1
    if sm: held_solve_m += 1

N_held = len(held_tasks)
print(f"CORE_ONLY: {held_solve_c}/{N_held} solved, mean_exp={np.mean(held_exp_c):.2f}")
print(f"CORE_META: {held_solve_m}/{N_held} solved, mean_exp={np.mean(held_exp_m):.2f}")

by_prog = defaultdict(lambda: {'ec': [], 'em': [], 'sc': 0, 'sm': 0, 'n': 0})
for i, (prog, x, y) in enumerate(held_tasks):
    p = by_prog[prog]
    p['ec'].append(held_exp_c[i]); p['em'].append(held_exp_m[i])
    p['sc'] += (1 if held_exp_c[i] < BUDGET else 0)
    p['sm'] += (1 if held_exp_m[i] < BUDGET else 0)
    p['n'] += 1

print("\nPer-program:")
for prog, p in sorted(by_prog.items()):
    print(f"  {prog}: core={np.mean(p['ec']):.1f} meta={np.mean(p['em']):.1f}")

# ── R6 deletion test ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("R6 DELETION TEST")
print("=" * 60)

meta_del = TraceConditionalMeta()
held_exp_del = []
for prog, x, y in held_tasks:
    _, ed, _ = solve(x, y, BUDGET, MAX_DEPTH, UNIFORM_D0, UNIFORM_D1, meta=meta_del)
    held_exp_del.append(ed)

del_mean  = float(np.mean(held_exp_del))
meta_mean = float(np.mean(held_exp_m))
delta     = del_mean - meta_mean

print(f"Post-deletion mean_exp = {del_mean:.2f}")
print(f"Pre-deletion  mean_exp = {meta_mean:.2f}")
print(f"Deletion delta         = {delta:+.3f}")

# ── Pre-registered verdict ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("E2.2(b) VERDICT — Pre-registered kill/positive")
print("=" * 60)

if delta <= 0:
    verdict = "K3_FIRES"
    print("K3 FIRES (3/3 mechanism kills): trace-conditional meta non-load-bearing.")
    print(f"  deletion-delta={delta:+.3f} <= 0. Signal cleanliness does not rescue selection.")
    print("  HARD PIVOT: selection axis DEFINITIVELY dead.")
    print("  Next: E2.2' (coverage-expansion core = C6 operationalized).")
else:
    verdict = "POSITIVE"
    print(f"POSITIVE: deletion-delta={delta:+.3f} > 0. Clean signal makes meta load-bearing.")
    print("  Carry trace-conditional meta-layer into E2.2' coverage-expansion design.")

result = {
    "experiment": "E2.2b",
    "issue_ref": "the-search#3",
    "date": "2026-05-29",
    "design": {
        "core": "UNIFORM (1/12 for all ops)",
        "accumulation": "trace-conditional (match ground-truth only)",
        "accum_programs": ACCUM_PROGRAMS,
        "held_programs": HELD_PROGRAMS,
        "n_accum_tasks": len(accum_tasks),
        "n_held_tasks": N_held,
        "budget": BUDGET,
        "max_depth": MAX_DEPTH
    },
    "accumulation": {
        "core_mean_exp": float(np.mean(accum_exp_core)),
        "meta_mean_exp": float(np.mean(accum_exp_meta)),
        "accumulated_traces": meta.accumulated,
        "rejected_traces": meta.rejected,
        "meta_depth2_success": dict(meta.depth2_success)
    },
    "held_out": {
        "CORE_ONLY": {"solved": int(held_solve_c), "total": N_held, "mean_exp": float(np.mean(held_exp_c))},
        "CORE_META": {"solved": int(held_solve_m), "total": N_held, "mean_exp": float(np.mean(held_exp_m))}
    },
    "r6_deletion": {
        "post_deletion_mean_exp": del_mean,
        "pre_deletion_mean_exp":  meta_mean,
        "delta": float(delta),
        "k3_fires": bool(delta <= 0)
    },
    "verdict": verdict,
    "preregistered_pivot": "K3_FIRES -> E2.2' (coverage-expansion core, C6 operationalized)"
}

with open('incoming/arc-agi1-visa/03_R4_transfer_wall/E2_2b_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nResult written to E2_2b_result.json")
