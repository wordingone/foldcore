"""
step0809_action0_always.py -- Diagnostic: always action 0 on LS20.

Tests the action-0 dominance hypothesis from steps 779/803 results.
If action 0 always >> random (36.4/seed), LS20 is action-0-dominated.
This changes what "beating random" means as a benchmark.

NOT numbered as official step — diagnostic to calibrate baseline.
25K steps, 10 seeds. Compare to step807 (36.4/seed random).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np

N_SEEDS = 10
N_STEPS = 25_000
N_ACTIONS = 4


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


print("=" * 70)
print("DIAGNOSTIC: ALWAYS ACTION 0 ON LS20")
print("=" * 70)
print(f"25K steps x {N_SEEDS} seeds")
print()

results = []
for seed in range(N_SEEDS):
    t0 = time.time()
    env = _make_ls20()
    obs = env.reset(seed=seed * 1000)
    l1_count = 0
    current_level = 0
    step = 0

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000)
            current_level = 0
            continue

        obs, reward, done, info = env.step(0)  # always action 0
        step += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            l1_count += (cl - current_level)
            current_level = cl

        if done:
            obs = env.reset(seed=seed * 1000)
            current_level = 0

    elapsed = time.time() - t0
    results.append(l1_count)
    print(f"  seed={seed:02d}  L1={l1_count}  ({elapsed:.1f}s)")

print()
print("=" * 70)
print("RESULTS — ALWAYS ACTION 0")
print("=" * 70)
total = sum(results)
mean = np.mean(results)
print(f"  Total L1: {total}  ({N_SEEDS} seeds × {N_STEPS} steps)")
print(f"  Mean L1/seed: {mean:.1f}")
print(f"  Random baseline (step807): 36.4/seed")
print(f"  Ratio vs random: {mean/36.4:.2f}x")
print()
if mean > 100:
    print("FINDING: Action 0 dominates LS20. Level completion ~requires sustained action 0.")
    print(f"New action-0 ceiling: {mean:.1f}/seed. Random baseline (36.4) was confounded by random seed-0 sequences.")
elif mean > 50:
    print(f"FINDING: Action 0 is above random ({mean:.1f} vs 36.4/seed) but not overwhelmingly dominant.")
else:
    print(f"FINDING: Action 0 is NOT dominant ({mean:.1f}/seed). Random (36.4/seed) may be comparable.")
print()
print("DIAGNOSTIC DONE")
