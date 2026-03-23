"""
step0817_null_hypothesis.py — Is 674 encoding the only contribution?

Compare:
  (A) 674 encoding (avgpool16, centered_enc, LSH k=12, running_mean) + random actions
  (B) Random projection to 256D (no real encoding) + random actions

If A >> B: encoding is the contribution (even without action mechanism).
If A ≈ B: encoding doesn't matter without action mechanism.

Metrics: unique cells visited, L1 count, episode length.
Budget: 25K steps, 10 seeds per variant. ~5 min.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')

import numpy as np
from substrates.step0674 import _enc_frame as _enc_674

print("=" * 70)
print("STEP 817 — NULL HYPOTHESIS: IS 674 ENCODING THE CONTRIBUTION?")
print("=" * 70)
print()

N_SEEDS = 10
N_STEPS = 25_000
N_ACTIONS = 4
K_NAV = 12
DIM = 256


def _enc_frame_674(obs_arr: np.ndarray) -> np.ndarray:
    """674 encoding: avgpool16 → 256-dim → mean-center."""
    return _enc_674(np.asarray(obs_arr, dtype=np.float32))


def _enc_random(obs_arr: np.ndarray, W_rand: np.ndarray) -> np.ndarray:
    """Random projection encoding: flatten → random 256D projection."""
    flat = obs_arr.flatten().astype(np.float32)[:4096]  # take first 4096 pixels
    if len(flat) < 4096:
        flat = np.pad(flat, (0, 4096 - len(flat)))
    x = W_rand @ flat  # (256, 4096) @ (4096,) → 256
    return x - x.mean()


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


def run_variant(variant: str, seed: int, H_nav: np.ndarray, W_rand=None) -> dict:
    running_mean = np.zeros(DIM, np.float32)
    n_obs = 0

    rng = np.random.RandomState(seed + 100)
    env = _make_ls20()
    obs = env.reset(seed=seed * 1000)

    unique_cells = set()
    l1_count = 0
    current_level = 0
    episodes = 0
    ep_lengths = []
    ep_len = 0
    step = 0

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000)
            current_level = 0
            continue

        x = np.asarray(obs, dtype=np.float32)

        if variant == "674_encoding":
            x_enc = _enc_frame_674(x)
        else:  # random_encoding
            x_enc = _enc_random(x, W_rand)

        n_obs += 1
        alpha = 1.0 / n_obs
        running_mean = (1 - alpha) * running_mean + alpha * x_enc
        x_c = x_enc - running_mean

        bits = (H_nav @ x_c > 0).astype(np.uint8)
        cell = int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)
        unique_cells.add(cell)

        action = rng.randint(0, N_ACTIONS)
        obs, reward, done, info = env.step(action)
        step += 1
        ep_len += 1

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            l1_count += (cl - current_level)
            current_level = cl

        if done:
            ep_lengths.append(ep_len)
            ep_len = 0
            episodes += 1
            obs = env.reset(seed=seed * 1000)
            current_level = 0

    return {
        "l1": l1_count,
        "unique_cells": len(unique_cells),
        "episodes": episodes,
        "mean_ep_len": np.mean(ep_lengths) if ep_lengths else N_STEPS,
    }


# Initialize shared nav planes and random projection matrix
rng0 = np.random.RandomState(0)
H_nav = rng0.randn(K_NAV, DIM).astype(np.float32)
W_rand = rng0.randn(DIM, 4096).astype(np.float32) * 0.01  # random projection

print("Running both variants (10 seeds × 25K steps each)...")
print()

results_674 = []
results_rand = []

for seed in range(N_SEEDS):
    t0 = time.time()
    r_674 = run_variant("674_encoding", seed, H_nav)
    r_rand = run_variant("random_encoding", seed, H_nav, W_rand)
    elapsed = time.time() - t0
    results_674.append(r_674)
    results_rand.append(r_rand)
    print(f"  seed={seed:02d}  674: L1={r_674['l1']} cells={r_674['unique_cells']}  |  "
          f"rand: L1={r_rand['l1']} cells={r_rand['unique_cells']}  ({elapsed:.1f}s)")

# Summary
print()
print("=" * 70)
print("STEP 817 RESULTS — NULL HYPOTHESIS")
print("=" * 70)
print()
for name, results in [("674 encoding", results_674), ("Random encoding", results_rand)]:
    total_l1 = sum(r["l1"] for r in results)
    mean_cells = np.mean([r["unique_cells"] for r in results])
    print(f"  {name:20s}: L1={total_l1}  mean_cells={mean_cells:.1f}")

print()
cells_674 = np.mean([r["unique_cells"] for r in results_674])
cells_rand = np.mean([r["unique_cells"] for r in results_rand])
ratio = cells_674 / max(cells_rand, 1)
print(f"  674 encoding / Random = {ratio:.2f}x (unique cells)")
print()
if ratio > 1.2:
    print("FINDING: 674 encoding provides MORE cell diversity than random projection.")
    print("The encoding IS the contribution to exploration (even without action mechanism).")
elif ratio < 0.8:
    print("FINDING: Random encoding explores MORE cells than 674 encoding.")
    print("674 encoding is too coarse — random projection covers space better.")
else:
    print("FINDING: 674 encoding and random projection explore similar cell counts.")
    print("The encoding structure has minimal effect on exploration without action mechanism.")
print()
print("STEP 817 DONE")
