"""
step0807_ft09_baseline.py -- FT09 random baseline (68 actions).

FT09 post-ban floor: random action with 674 encoding.
25K steps, 10 seeds. Compare to LS20 baseline (36.4/seed).
"""
import sys, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import _enc_frame as _enc_674

print("=" * 70)
print("STEP 807 FT09 — RANDOM BASELINE (68 ACTIONS)")
print("=" * 70)
print("674 encoding + NO graph + RANDOM actions on FT09")
print("25K steps x 10 seeds")
print()

N_SEEDS = 10
N_STEPS = 25_000
N_ACTIONS = 68  # FT09: 4 dirs + 64 clicks


def _make_ft09():
    try:
        import arcagi3
        return arcagi3.make("FT09")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("FT09")


results = []
for seed in range(N_SEEDS):
    t0 = time.time()
    rng = np.random.RandomState(seed)
    env = _make_ft09()
    obs = env.reset(seed=seed * 1000)
    l1_count = 0
    current_level = 0
    step = 0
    running_mean = np.zeros(256, np.float32)
    n_obs = 0

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000)
            current_level = 0
            continue

        x = _enc_674(np.asarray(obs, dtype=np.float32))
        n_obs += 1
        alpha = 1.0 / n_obs
        running_mean = (1 - alpha) * running_mean + alpha * x

        action = rng.randint(0, N_ACTIONS)
        obs, reward, done, info = env.step(action)
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
print("FT09 RANDOM BASELINE RESULTS")
print("=" * 70)
total = sum(results)
mean = np.mean(results)
print(f"  Total L1: {total}  ({N_SEEDS} seeds x {N_STEPS} steps)")
print(f"  Mean L1/seed: {mean:.1f}")
print(f"  LS20 baseline: 36.4/seed (for comparison)")
print()
print("STEP 807_FT09 DONE")
