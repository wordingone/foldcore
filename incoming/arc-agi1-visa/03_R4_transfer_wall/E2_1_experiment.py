# E2_1_experiment.py — E2.1 combinatorial meta-layer gate (the-search#3).
#
# Architecture:
#   Core (frozen): logistic prior over shape-features → BASIS-op distribution.
#                  Trained once on synthetic forward-run triples, then frozen.
#   Meta-layer (studied RSI process): LGG-accumulated depth-2 success counts,
#                  R2-fused at solve-time, reweights core's depth-2 prior.
#
# Experimental design:
#   Accumulation programs (9): ACCUM_FIRST × ACCUM_SECOND (crop/fh/tr × rot/up2/mir_h).
#     System streams through these tasks, accumulating LGG signal.
#   Held-out programs (6): NEW_FIRST × ACCUM_SECOND (fv/dup_h × rot/up2/mir_h).
#     Novel first-op combinations. Second ops ARE in the accumulated LGG set.
#
# Arms:
#   CORE_ONLY: frozen prior, no meta-layer.
#   CORE_META: frozen prior + accumulated LGG boost at depth-2.
#
# R6-load-bearing kill: delete meta-layer -> if held-out efficiency unchanged
#   or improves -> meta-layer decorative -> 2nd mechanism kill of the-search#3.
#
# MEM-HEAVY: N/A (core is ~1MB, zero VRAM).

import json, numpy as np, heapq, time
from collections import defaultdict

substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

BASIS_NAMES = list(BASIS.keys())
N_OPS = len(BASIS_NAMES)

# ── Task generation ───────────────────────────────────────────────────────────
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

# ── Program split ─────────────────────────────────────────────────────────────
ACCUM_FIRST  = ['crop', 'fh', 'tr']
ACCUM_SECOND = ['rot', 'up2', 'mir_h']
HELD_FIRST   = ['fv', 'dup_h']

ACCUM_PROGRAMS = [(f, s) for f in ACCUM_FIRST  for s in ACCUM_SECOND]  # 9
HELD_PROGRAMS  = [(f, s) for f in HELD_FIRST   for s in ACCUM_SECOND]  # 6

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

print(f"Tasks generated: {len(accum_tasks)} accum, {len(held_tasks)} held-out")

# ── Core: frozen logistic prior ───────────────────────────────────────────────
# Train on 600 synthetic programs biased toward ACCUM_FIRST at d=0, ACCUM_SECOND at d=1.
# Smoothing ensures all ops have non-zero probability.
rng_train = np.random.default_rng(13)
N_TRAIN = 600
d0_counts = defaultdict(lambda: 1.0)  # Laplace smoothing
d1_counts = defaultdict(lambda: 1.0)

OTHER_FIRST  = [nm for nm in BASIS_NAMES if nm not in ACCUM_FIRST]
OTHER_SECOND = [nm for nm in BASIS_NAMES if nm not in ACCUM_SECOND]

for i in range(N_TRAIN):
    # Bias: 70% toward ACCUM ops, 30% others
    d0 = (rng_train.choice(ACCUM_FIRST)   if rng_train.random() < 0.7
          else rng_train.choice(OTHER_FIRST))
    d1 = (rng_train.choice(ACCUM_SECOND)  if rng_train.random() < 0.7
          else rng_train.choice(OTHER_SECOND))
    d0_counts[d0] += 1
    d1_counts[d1] += 1

total_d0 = sum(d0_counts[nm] for nm in BASIS_NAMES)
total_d1 = sum(d1_counts[nm] for nm in BASIS_NAMES)
CORE_D0 = {nm: d0_counts[nm] / total_d0 for nm in BASIS_NAMES}
CORE_D1 = {nm: d1_counts[nm] / total_d1 for nm in BASIS_NAMES}

top3_d0 = sorted(CORE_D0.items(), key=lambda x: -x[1])[:3]
top3_d1 = sorted(CORE_D1.items(), key=lambda x: -x[1])[:3]
print(f"Core D0 top-3: {top3_d0}")
print(f"Core D1 top-3: {top3_d1}")

# ── LGG meta-layer ────────────────────────────────────────────────────────────
class MetaLayer:
    """R2-fused: updates at solve-time. Reweights core's D1 prior by depth-2 success counts."""

    def __init__(self):
        self.depth2_success = defaultdict(int)

    def update(self, seq):
        if len(seq) >= 2:
            self.depth2_success[seq[1]] += 1

    def get_boost(self, nm):
        return 1.0 + self.depth2_success.get(nm, 0)

    def get_weighted_d1(self, core_d1):
        """Returns core_d1 × meta_boost, normalized."""
        raw = {nm: core_d1[nm] * self.get_boost(nm) for nm in BASIS_NAMES}
        total = sum(raw.values())
        return {nm: raw[nm] / total for nm in BASIS_NAMES}

# ── Heap search ───────────────────────────────────────────────────────────────
def h_dist(a, b):
    return abs(a.shape[0] - b.shape[0]) + abs(a.shape[1] - b.shape[1])

def solve(x, y, budget, max_depth, core_d0, core_d1, meta=None):
    """
    Priority key: (h*2 + depth, -prob_score, counter, seq, state).
    At d=0: prob_score = core_d0[nm].
    At d=1: prob_score = (core_d1 × meta_boost normalized)[nm] if meta else core_d1[nm].
    h-heuristic primary; prob_score breaks ties within same h-bucket.
    """
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
            heapq.heappush(frontier,
                           (h * 2 + d + 1, -prob, ctr[0], seq + (nm,), nxt))
    return False, exp, None

# ── Accumulation stream ───────────────────────────────────────────────────────
BUDGET    = 500
MAX_DEPTH = 3

print("\n" + "=" * 60)
print("ACCUMULATION STREAM")
print("=" * 60)

meta = MetaLayer()
accum_exp_core, accum_exp_meta = [], []

for prog, x, y in accum_tasks:
    _, exp_c, _    = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=None)
    _, exp_m, seq_m = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=meta)
    if seq_m:
        meta.update(seq_m)   # R2-fused update
    accum_exp_core.append(exp_c)
    accum_exp_meta.append(exp_m)

print(f"CORE_ONLY  mean exp = {np.mean(accum_exp_core):.2f}")
print(f"CORE_META  mean exp = {np.mean(accum_exp_meta):.2f}")
print(f"Meta depth2_success: {dict(meta.depth2_success)}")

# ── Held-out evaluation ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HELD-OUT EVALUATION (meta fixed — no further updates)")
print("=" * 60)

held_rows = []
for prog, x, y in held_tasks:
    sc, exp_c, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=None)
    sm, exp_m, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=meta)
    held_rows.append({'prog': prog,
                      'solved_c': sc, 'exp_c': exp_c,
                      'solved_m': sm, 'exp_m': exp_m})

held_exp_c   = [r['exp_c'] for r in held_rows]
held_exp_m   = [r['exp_m'] for r in held_rows]
held_solve_c = sum(r['solved_c'] for r in held_rows)
held_solve_m = sum(r['solved_m'] for r in held_rows)
N_held = len(held_rows)

print(f"CORE_ONLY  solved={held_solve_c}/{N_held}  mean_exp={np.mean(held_exp_c):.2f}")
print(f"CORE_META  solved={held_solve_m}/{N_held}  mean_exp={np.mean(held_exp_m):.2f}")

by_prog = defaultdict(lambda: {'exp_c': [], 'exp_m': [], 'sc': 0, 'sm': 0, 'n': 0})
for r in held_rows:
    p = by_prog[r['prog']]
    p['exp_c'].append(r['exp_c']); p['exp_m'].append(r['exp_m'])
    p['sc'] += r['solved_c']; p['sm'] += r['solved_m']; p['n'] += 1

print("\nPer-program:")
for prog, p in sorted(by_prog.items()):
    print(f"  {prog}: "
          f"core={np.mean(p['exp_c']):.1f}({p['sc']}/{p['n']}) "
          f"meta={np.mean(p['exp_m']):.1f}({p['sm']}/{p['n']})")

# ── R6 deletion test ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("R6 DELETION TEST")
print("=" * 60)

meta_del = MetaLayer()  # fresh empty meta (deletion)
held_exp_del = []
for prog, x, y in held_tasks:
    _, exp_d, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=meta_del)
    held_exp_del.append(exp_d)

del_mean = np.mean(held_exp_del)
meta_mean = np.mean(held_exp_m)
delta = del_mean - meta_mean

print(f"Post-deletion mean_exp = {del_mean:.2f}")
print(f"Pre-deletion  mean_exp = {meta_mean:.2f}")
print(f"Deletion delta         = {delta:+.2f} expansions "
      f"({'DEGRADED -- meta was load-bearing' if delta > 0 else 'no change or improved -- meta decorative'})")

# ── Verdict ───────────────────────────────────────────────────────────────────
meta_beats_core = (np.mean(held_exp_m) < np.mean(held_exp_c)
                   or held_solve_m > held_solve_c)
r6_fires = delta <= 0

print("\n" + "=" * 60)
print("E2.1 VERDICT")
print("=" * 60)

if r6_fires:
    verdict = "R6_FIRES"
    print("R6 FIRES (2nd mechanism kill): meta-layer non-load-bearing.")
    print("Deletion does not degrade held-out efficiency. Meta is decorative.")
elif meta_beats_core:
    verdict = "PASS"
    print("PASS: CORE_META > CORE_ONLY on held-out novel compositions.")
    print(f"R6 confirmed load-bearing: deletion costs +{delta:.2f} expansions.")
    print("Proceed to E2.2 (structural novelty, richer core).")
else:
    verdict = "REVISE"
    print("REVISE: Meta-layer R6-load-bearing (deletion costs expansions)")
    print("but CORE_META does not beat CORE_ONLY on held-out.")
    print("Core prior already captures enough structure; meta is redundant signal.")
    print("Implication: E2.2 needs a richer core (Option 1) where core is not already")
    print("pre-biased toward the correct second ops.")

result = {
    "experiment": "E2.1",
    "issue_ref": "the-search#3",
    "date": "2026-05-29",
    "design": {
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
        "meta_depth2_success": dict(meta.depth2_success)
    },
    "held_out": {
        "CORE_ONLY": {"solved": int(held_solve_c), "total": N_held, "mean_exp": float(np.mean(held_exp_c))},
        "CORE_META": {"solved": int(held_solve_m), "total": N_held, "mean_exp": float(np.mean(held_exp_m))}
    },
    "r6_deletion": {
        "post_deletion_mean_exp": float(del_mean),
        "pre_deletion_mean_exp":  float(meta_mean),
        "delta": float(delta),
        "r6_fires": bool(r6_fires)
    },
    "verdict": verdict
}

with open('incoming/arc-agi1-visa/03_R4_transfer_wall/E2_1_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nResult written to E2_1_result.json")
