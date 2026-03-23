"""
step0807_random_baseline.py — Post-ban random baseline calibration.

Substrate: 674 encoding (avgpool16, centered_enc, LSH k=12, running-mean).
NO graph. NO forward model. RANDOM actions.

Establishes the post-ban floor: what does random exploration achieve with
only encoding (no navigation mechanism)?

Metrics per seed:
  - L1 count (level completions)
  - unique_cells (distinct LSH cells visited)
  - episodes (done=True events = level timeouts)
  - steps_per_episode (mean episode length)

Budget: 25K steps, 10 seeds. ~5 min.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import _enc_frame as _enc_674

print("=" * 70)
print("STEP 807 — RANDOM BASELINE (POST-BAN CALIBRATION)")
print("=" * 70)
print("674 encoding + NO graph + RANDOM actions")
print("25K steps × 10 seeds")
print()

# ---- Parameters ----
N_SEEDS = 10
N_STEPS = 25_000
N_ACTIONS = 4

# ---- 674 encoding (no graph) ----
K_NAV = 12
DIM = 256


def _enc_frame(obs):
    """Use 674's proven encoding."""
    return _enc_674(np.asarray(obs, dtype=np.float32))


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_seed(seed: int):
    """Run one seed. Returns metrics dict."""
    rng = np.random.RandomState(seed)
    H_nav = rng.randn(K_NAV, DIM).astype(np.float32)
    running_mean = np.zeros(DIM, np.float32)
    n_obs = 0

    env = _make_ls20()
    obs = env.reset(seed=seed * 1000)

    unique_cells = set()
    l1_count = 0
    current_level = 0
    episode_lengths = []
    ep_len = 0
    step = 0
    t0 = time.time()

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000)
            current_level = 0
            continue

        # Encode
        x = np.asarray(obs, dtype=np.float32)
        x = _enc_frame(x)
        n_obs += 1
        alpha = 1.0 / n_obs
        running_mean = (1 - alpha) * running_mean + alpha * x
        x_c = x - running_mean

        # Cell hash (nav only, no graph)
        bits = (H_nav @ x_c > 0).astype(np.uint8)
        cell = int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)
        unique_cells.add(cell)

        # Random action
        action = rng.randint(0, N_ACTIONS)
        obs, reward, done, info = env.step(action)
        step += 1
        ep_len += 1

        # Level completion check
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            l1_count += (cl - current_level)
            current_level = cl

        if done:
            episode_lengths.append(ep_len)
            ep_len = 0
            obs = env.reset(seed=seed * 1000)
            current_level = 0

    elapsed = time.time() - t0
    mean_ep_len = np.mean(episode_lengths) if episode_lengths else N_STEPS

    print(f"  seed={seed:02d}  L1={l1_count}  unique_cells={len(unique_cells)}  "
          f"episodes={len(episode_lengths)}  ep_len={mean_ep_len:.0f}  ({elapsed:.1f}s)")
    return {
        "seed": seed,
        "l1": l1_count,
        "unique_cells": len(unique_cells),
        "episodes": len(episode_lengths),
        "mean_ep_len": mean_ep_len,
        "elapsed": elapsed,
    }


results = []
for seed in range(N_SEEDS):
    r = run_seed(seed)
    results.append(r)

# ---- Summary ----
print()
print("=" * 70)
print("STEP 807 RESULTS — POST-BAN RANDOM BASELINE")
print("=" * 70)
print()
total_l1 = sum(r["l1"] for r in results)
mean_cells = np.mean([r["unique_cells"] for r in results])
mean_ep = np.mean([r["mean_ep_len"] for r in results])
mean_eps = np.mean([r["episodes"] for r in results])
print(f"  Total L1 completions: {total_l1} / {N_SEEDS} seeds × {N_STEPS} steps")
print(f"  Mean unique cells:    {mean_cells:.1f}")
print(f"  Mean episodes:        {mean_eps:.1f} (timeouts)")
print(f"  Mean episode length:  {mean_ep:.1f} steps")
print()
if total_l1 == 0:
    print("FINDING: Random + 674 encoding achieves L1=0. No navigation without action mechanism.")
    print("Post-ban floor: unique cell coverage only. 674 encoding does not help navigation without graph.")
else:
    print(f"FINDING: Random + 674 encoding achieves {total_l1} L1 completions. Some navigation possible.")
print()
print("STEP 807 DONE")
