#!/usr/bin/env python3
"""
Step 292 -- IO-guided composition search for a%b.

Spec. Anti-inflation check: can IO-consistency VERIFY that the correct
3-step decomposition of a%b is the highest-scoring depth-3 composition?

This is NOT the substrate discovering composition autonomously.
It tests whether the IO-consistency landscape discriminates correctly:
  success = correct composition (sub(a, mul(b, fdiv(a,b)))) is top-scoring at depth 3.

Primitive library: add, sub, mul, fdiv (floor division)
Composition: linear chain — each step adds one value, result is the last value.
  values = [a, b]
  step k: values.append(op(values[i], values[j]))  for i,j in 0..n_avail-1
  output = values[-1]

IO pairs: (a,b) -> a%b for a,b in 1..20 (400 pairs).
Score: fraction of IO pairs where |output - target| < 0.5.

Success criterion (Avir): correct 3-step composition scores 100% AND is top-scoring.
Kill criterion: correct composition is NOT top-scoring at depth-3.
"""

import math
import time

# ─── Primitives ───────────────────────────────────────────────────────────────

def _add(x, y):  return x + y
def _sub(x, y):  return x - y
def _mul(x, y):  return x * y
def _fdiv(x, y): return math.floor(x / y) if y != 0 else float('inf')

OPS = [
    ('add',  _add),
    ('sub',  _sub),
    ('mul',  _mul),
    ('fdiv', _fdiv),
]

# ─── IO pairs ─────────────────────────────────────────────────────────────────

TRAIN_MAX = 20
IO_PAIRS = [(a, b, a % b) for a in range(1, TRAIN_MAX + 1)
                           for b in range(1, TRAIN_MAX + 1)]  # 400 pairs

# ─── Program representation ───────────────────────────────────────────────────
# A program is a list of steps: (op_name, op_fn, i, j)
#   i, j = indices into `values` at time of execution
# values = [a, b, r0, r1, ...]
# output = values[-1]

def eval_program(steps, a, b):
    values = [float(a), float(b)]
    for (_, op_fn, i, j) in steps:
        try:
            r = op_fn(values[i], values[j])
            if not math.isfinite(r) or abs(r) > 1e9:
                return None
        except Exception:
            return None
        values.append(r)
    return values[-1]


def score(steps):
    correct = 0
    for a, b, target in IO_PAIRS:
        r = eval_program(steps, a, b)
        if r is not None and abs(r - target) < 0.5:
            correct += 1
    return correct / len(IO_PAIRS)


def prog_str(steps):
    """Human-readable symbolic expression."""
    names = ['a', 'b']
    parts = []
    for (op_name, _, i, j) in steps:
        expr = f"{op_name}({names[i]}, {names[j]})"
        names.append(f"r{len(parts)}")
        parts.append(f"r{len(parts)-1}={expr}")
    return ' | '.join(parts)


# ─── Enumerate ────────────────────────────────────────────────────────────────

def enumerate_programs(depth):
    """Enumerate all linear-chain programs of given depth."""
    results = []

    def build(steps, n_avail):
        if len(steps) == depth:
            results.append(steps[:])
            return
        for op_name, op_fn in OPS:
            for i in range(n_avail):
                for j in range(n_avail):
                    steps.append((op_name, op_fn, i, j))
                    build(steps, n_avail + 1)
                    steps.pop()

    build([], 2)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_depth(depth, label):
    t0 = time.time()
    programs = enumerate_programs(depth)
    n = len(programs)
    print(f"\n--- Depth {depth}: {n} programs ---", flush=True)

    best_score = -1.0
    best_prog  = None
    n_perfect  = 0
    score_dist = {}  # score bucket -> count

    for prog in programs:
        s = score(prog)
        bucket = round(s, 2)
        score_dist[bucket] = score_dist.get(bucket, 0) + 1
        if s == 1.0:
            n_perfect += 1
        if s > best_score:
            best_score = s
            best_prog  = prog

    elapsed = time.time() - t0
    print(f"  Best score: {best_score*100:.1f}%  ({n_perfect} perfect programs)",
          flush=True)
    print(f"  Best program: {prog_str(best_prog)}", flush=True)
    print(f"  Time: {elapsed:.2f}s", flush=True)

    # Score distribution (top buckets)
    top_buckets = sorted(score_dist.items(), key=lambda x: -x[0])[:5]
    print(f"  Score distribution (top 5): "
          + ", ".join(f"{int(s*100)}%:{c}" for s, c in top_buckets), flush=True)

    return best_score, best_prog, n_perfect


def main():
    t0 = time.time()
    print("Step 292 -- IO-guided Composition Search for a%b", flush=True)
    print(f"IO pairs: {len(IO_PAIRS)} (a,b in 1..{TRAIN_MAX})", flush=True)
    print(f"Primitives: {[n for n,_ in OPS]}", flush=True)
    print(f"Target: a%b = sub(a, mul(b, fdiv(a,b)))", flush=True)

    # The correct 3-step program:
    # r0 = fdiv(a, b)     values[2]
    # r1 = mul(b, r0)     values[3]  -- b=values[1], r0=values[2]
    # r2 = sub(a, r1)     output     -- a=values[0], r1=values[3]
    correct_steps = [
        ('fdiv', _fdiv, 0, 1),   # fdiv(a, b)
        ('mul',  _mul,  1, 2),   # mul(b, r0)
        ('sub',  _sub,  0, 3),   # sub(a, r1)
    ]
    correct_score = score(correct_steps)
    print(f"\nCorrect decomposition score: {correct_score*100:.1f}%", flush=True)
    print(f"  {prog_str(correct_steps)}", flush=True)

    if correct_score < 1.0:
        print("  WARNING: correct decomposition doesn't score 100% — check bugs.",
              flush=True)

    results = {}
    for depth in [1, 2, 3]:
        best_score, best_prog, n_perfect = run_depth(depth, f"depth-{depth}")
        results[depth] = (best_score, best_prog, n_perfect)

    elapsed = time.time() - t0

    # ─── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("STEP 292 SUMMARY", flush=True)
    print("=" * 70, flush=True)

    best_score_d3, best_prog_d3, n_perfect_d3 = results[3]
    correct_is_top = (correct_score == 1.0 and best_score_d3 == 1.0)

    # Check if best_prog_d3 IS the correct composition (by score and structure)
    print(f"\nDepth-3 best: {prog_str(best_prog_d3)}", flush=True)
    print(f"Depth-3 best score: {best_score_d3*100:.1f}% | "
          f"Perfect programs: {n_perfect_d3}", flush=True)
    print(f"Correct decomposition score: {correct_score*100:.1f}%", flush=True)
    print(flush=True)

    print("SUCCESS CRITERION (Spec):", flush=True)
    if correct_score == 1.0 and best_score_d3 == 1.0:
        print(f"  PASSES — correct composition scores 100% and is among top-scoring.",
              flush=True)
        if n_perfect_d3 == 1:
            print(f"  STRONG PASS — correct composition is the UNIQUE perfect program.",
                  flush=True)
        else:
            print(f"  WEAK PASS — {n_perfect_d3} programs score 100% (IO not fully "
                  f"discriminating for depth-3).", flush=True)
    else:
        print(f"  KILLED — correct composition does NOT emerge as top-scoring.",
              flush=True)
        print(f"  IO-consistency landscape does not discriminate correctly.",
              flush=True)

    print(f"\nElapsed: {elapsed:.2f}s", flush=True)


if __name__ == '__main__':
    main()
