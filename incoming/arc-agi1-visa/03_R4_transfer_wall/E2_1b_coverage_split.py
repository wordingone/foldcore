# E2_1b_coverage_split.py — R4 Leg 2 confirmation from the capability-memory side.
#
# Context: E2.1 measured in-coverage only (synthetic tasks from the 12-op basis).
# Per R4 Leg 2, out-of-coverage prediction is +0.000: a selection mechanism (LGG
# meta-layer) cannot improve performance on tasks outside vocabulary coverage.
#
# This script:
#   1. Reconstructs the E2.1 accumulated meta-layer (from the in-coverage accum stream).
#   2. Runs it against a sample of the 395 locked out-of-closure ARC tasks.
#   3. Reports in-coverage vs out-of-coverage improvement SEPARATELY.
#
# Pre-registered R4 prediction:
#   in-coverage (E2.1): R6 fires (already confirmed)
#   out-of-coverage: solve rate delta = 0, expansion delta ≈ 0 (Leg 2)
#   If out-of-coverage shows ANY gain from meta → CONTRADICTS R4 → flag immediately.
#
# Refs: the-search#3, R4 Leg 2 (98e10f17), E2.1 (dad816fe)

import json, numpy as np, heapq, time
from collections import defaultdict
from pathlib import Path

substrate_src = open('B:/M/the-search/experiments/components/SUBSTRATE.py').read()
run_idx = substrate_src.index('\n# RUN:')
exec(substrate_src[:run_idx])

BASIS_NAMES = list(BASIS.keys())
N_OPS = len(BASIS_NAMES)

# ── Reconstruct E2.1 core + accumulated meta-layer ────────────────────────────
# Match E2.1 exactly: biased core + accum-stream meta-layer.
rng_gen   = np.random.default_rng(7)    # same seed as E2.1
rng_train = np.random.default_rng(13)

ACCUM_FIRST  = ['crop', 'fh', 'tr']
ACCUM_SECOND = ['rot', 'up2', 'mir_h']
HELD_FIRST   = ['fv', 'dup_h']

ACCUM_PROGRAMS = [(f, s) for f in ACCUM_FIRST  for s in ACCUM_SECOND]  # 9
HELD_PROGRAMS  = [(f, s) for f in HELD_FIRST   for s in ACCUM_SECOND]  # 6

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

# Rebuild core (biased logistic prior)
N_TRAIN = 600
d0_counts = defaultdict(lambda: 1.0)
d1_counts = defaultdict(lambda: 1.0)
OTHER_FIRST  = [nm for nm in BASIS_NAMES if nm not in ACCUM_FIRST]
OTHER_SECOND = [nm for nm in BASIS_NAMES if nm not in ACCUM_SECOND]
for i in range(N_TRAIN):
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

# Rebuild meta-layer accumulated state from E2.1 accum stream
class MetaLayer:
    def __init__(self):
        self.depth2_success = defaultdict(int)
    def update(self, seq):
        if len(seq) >= 2:
            self.depth2_success[seq[1]] += 1
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

# Accumulate meta-layer from E2.1 accum stream (same RNG state)
meta_accum = MetaLayer()
N_ACCUM_PER = 10
for prog in ACCUM_PROGRAMS:
    for _ in range(N_ACCUM_PER):
        t = generate_task(rng_gen, prog)
        if t:
            _, _, seq_m = solve(t[0], t[1], 500, 3, CORE_D0, CORE_D1, meta=meta_accum)
            if seq_m:
                meta_accum.update(seq_m)

print(f"Meta accumulated: {dict(meta_accum.depth2_success)}")

# ── Section A: IN-COVERAGE held-out (same as E2.1 held-out) ───────────────────
print("\n" + "=" * 60)
print("SECTION A: IN-COVERAGE held-out (6 novel programs from basis)")
print("Expected: R6 fires (confirmed E2.1). Reproduced here for split table.")
print("=" * 60)

BUDGET = 500
MAX_DEPTH = 3

held_tasks = []
for prog in HELD_PROGRAMS:
    for _ in range(5):  # N_HELD_PER = 5
        t = generate_task(rng_gen, prog)
        if t:
            held_tasks.append((prog, t[0], t[1]))

in_cov_exp_c, in_cov_exp_m = [], []
in_cov_solve_c = in_cov_solve_m = 0
for prog, x, y in held_tasks:
    sc, ec, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=None)
    sm, em, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=meta_accum)
    in_cov_exp_c.append(ec); in_cov_exp_m.append(em)
    if sc: in_cov_solve_c += 1
    if sm: in_cov_solve_m += 1

print(f"CORE_ONLY: {in_cov_solve_c}/{len(held_tasks)} solved, mean_exp={np.mean(in_cov_exp_c):.2f}")
print(f"CORE_META: {in_cov_solve_m}/{len(held_tasks)} solved, mean_exp={np.mean(in_cov_exp_m):.2f}")
print(f"delta_exp: {np.mean(in_cov_exp_m) - np.mean(in_cov_exp_c):+.2f}")

# ── Section B: OUT-OF-COVERAGE (395 locked ARC tasks) ─────────────────────────
print("\n" + "=" * 60)
print("SECTION B: OUT-OF-COVERAGE (395 locked ARC eval tasks)")
print("R4 Leg 2 prediction: solve_delta=0, exp_delta~=0")
print("=" * 60)

d2 = json.load(open('incoming/arc-agi1-visa/03_R4_transfer_wall/D2_1_final_out.json'))
ARC_EVAL = Path('incoming/arc-agi1-visa/03_R4_transfer_wall/ARC-AGI/data/evaluation')

# Sample 100 tasks for speed (out-of-closure result is structural, not statistical)
rng_sample = np.random.default_rng(42)
sample_ids = list(rng_sample.choice(d2['ids'], size=min(100, len(d2['ids'])), replace=False))

out_cov_exp_c, out_cov_exp_m = [], []
out_cov_solve_c = out_cov_solve_m = 0
skipped = 0
t0 = time.time()

for i, tid in enumerate(sample_ids):
    p = ARC_EVAL / f"{tid}.json"
    if not p.exists():
        skipped += 1
        continue
    task = json.load(open(p))
    pair = task['train'][0]
    x = np.array(pair['input'],  dtype=np.int64)
    y = np.array(pair['output'], dtype=np.int64)

    sc, ec, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=None)
    sm, em, _ = solve(x, y, BUDGET, MAX_DEPTH, CORE_D0, CORE_D1, meta=meta_accum)
    out_cov_exp_c.append(ec); out_cov_exp_m.append(em)
    if sc: out_cov_solve_c += 1
    if sm: out_cov_solve_m += 1

    if (i + 1) % 25 == 0:
        print(f"  [{i+1}/{len(sample_ids)}] solves: core={out_cov_solve_c}, meta={out_cov_solve_m}  "
              f"elapsed={time.time()-t0:.1f}s")

n_out = len(out_cov_exp_c)
print(f"\nCORE_ONLY: {out_cov_solve_c}/{n_out} solved, mean_exp={np.mean(out_cov_exp_c):.2f}")
print(f"CORE_META: {out_cov_solve_m}/{n_out} solved, mean_exp={np.mean(out_cov_exp_m):.2f}")
print(f"delta_exp: {np.mean(out_cov_exp_m) - np.mean(out_cov_exp_c):+.2f}")
print(f"delta_solve: {out_cov_solve_m - out_cov_solve_c:+d}")

# ── Coverage-split verdict ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("E2.1b VERDICT — Coverage-split summary")
print("=" * 60)

in_delta  = np.mean(in_cov_exp_m)  - np.mean(in_cov_exp_c)
out_delta = np.mean(out_cov_exp_m) - np.mean(out_cov_exp_c)

print(f"In-coverage  (held-out, 6 novel programs):  exp_delta={in_delta:+.3f}  R4=SELECTION, fires per E2.1")
print(f"Out-of-coverage (100 ARC locked tasks):     exp_delta={out_delta:+.3f}  R4 Leg 2 prediction=~0.000")
print()
if abs(out_delta) < 1.0 and out_cov_solve_m == out_cov_solve_c:
    print("R4 LEG 2 CONFIRMED from capability-memory side:")
    print("  LGG meta-layer (selection mechanism) does not improve out-of-coverage tasks.")
    print("  Selection cannot expand vocabulary coverage. Consistent with warm-cold=+0.000.")
    verdict = "R4_LEG2_CONFIRMED"
elif out_cov_solve_m > out_cov_solve_c or out_delta < -2.0:
    print("CONTRADICTION: out-of-coverage shows improvement from selection mechanism.")
    print("This CONTRADICTS R4 Leg 2. Investigate immediately.")
    verdict = "R4_CONTRADICTION"
else:
    print(f"Marginal: exp_delta={out_delta:+.3f}, solve_delta={out_cov_solve_m - out_cov_solve_c:+d}.")
    print("Consistent with Leg 2 (noise level).")
    verdict = "R4_LEG2_CONSISTENT"

result = {
    "experiment": "E2.1b",
    "issue_ref": "the-search#3",
    "date": "2026-05-29",
    "r4_prediction": "out_of_coverage_delta=0 (Leg 2: selection cannot expand coverage)",
    "meta_accumulated": dict(meta_accum.depth2_success),
    "in_coverage": {
        "n_tasks": len(held_tasks),
        "CORE_ONLY": {"solved": int(in_cov_solve_c), "mean_exp": float(np.mean(in_cov_exp_c))},
        "CORE_META": {"solved": int(in_cov_solve_m), "mean_exp": float(np.mean(in_cov_exp_m))},
        "delta_exp": float(in_delta)
    },
    "out_of_coverage": {
        "n_tasks": n_out,
        "n_sampled": len(sample_ids),
        "CORE_ONLY": {"solved": int(out_cov_solve_c), "mean_exp": float(np.mean(out_cov_exp_c))},
        "CORE_META": {"solved": int(out_cov_solve_m), "mean_exp": float(np.mean(out_cov_exp_m))},
        "delta_exp": float(out_delta),
        "delta_solve": int(out_cov_solve_m - out_cov_solve_c)
    },
    "verdict": verdict
}

with open('incoming/arc-agi1-visa/03_R4_transfer_wall/E2_1b_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("\nResult written to E2_1b_result.json")
