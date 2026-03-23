"""
step0855_compression_progress.py — R3_cf + L1 test: Schmidhuber compression progress.

R3 hypothesis: going where prediction accuracy is IMPROVING avoids noisy TV and
enables purposeful exploration. Anti-noisy-TV mechanism: stochastic transitions
have constant error (no progress) → not selected.

Metrics: L1 count (does it navigate?) + prediction accuracy R3_cf.
Budget: 25K steps, 10 test seeds. ~5 min.
"""
import sys, os, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import numpy as np

print("=" * 70)
print("STEP 855 — COMPRESSION PROGRESS (SCHMIDHUBER ANTI-NOISY-TV)")
print("=" * 70)

from substrates.step0855 import CompressionProgress855

N_ACTIONS = 4
N_SEEDS = 10
N_STEPS = 25_000


def _make_ls20():
    try:
        import arcagi3
        return arcagi3.make("LS20")
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make("LS20")


from substrates.step0674 import _enc_frame

results = []
for seed in range(N_SEEDS):
    t0 = time.time()
    sub = CompressionProgress855(n_actions=N_ACTIONS, seed=seed)
    sub.reset(seed)

    env = _make_ls20()
    obs = env.reset(seed=seed * 1000)
    l1_count = 0
    current_level = 0
    unique_cells_set = set()
    pred_errors = []
    step = 0

    while step < N_STEPS:
        if obs is None:
            obs = env.reset(seed=seed * 1000)
            sub.on_level_transition()
            continue

        obs_arr = np.asarray(obs, dtype=np.float32)
        prev_enc = sub._last_enc.copy() if sub._last_enc is not None else None
        prev_action = sub._prev_action

        action = sub.process(obs_arr)
        obs_next, reward, done, info = env.step(action % N_ACTIONS)
        step += 1

        # Prediction accuracy measurement
        if prev_enc is not None and prev_action is not None and obs_next is not None:
            next_arr = np.asarray(obs_next, dtype=np.float32)
            next_enc = sub._encode_for_pred(next_arr)
            pred = sub.predict_next(prev_enc, prev_action)
            err = float(np.sum((pred - next_enc) ** 2))
            norm = float(np.sum(next_enc ** 2)) + 1e-8
            pred_errors.append((err, norm))

        # Track unique cells
        if sub._last_enc is not None:
            h = hash(sub._last_enc.tobytes()) % (2**32)
            unique_cells_set.add(h)

        obs = obs_next
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > current_level:
            l1_count += (cl - current_level)
            current_level = cl
            sub.on_level_transition()

        if done:
            obs = env.reset(seed=seed * 1000)
            current_level = 0
            sub.on_level_transition()

    elapsed = time.time() - t0
    # Prediction accuracy
    if pred_errors:
        total_err = sum(e for e, n in pred_errors)
        total_norm = sum(n for e, n in pred_errors)
        pred_acc = max(0.0, 1.0 - total_err / total_norm) * 100.0
    else:
        pred_acc = None

    results.append({
        "seed": seed, "l1": l1_count,
        "unique_cells": len(unique_cells_set),
        "pred_acc": pred_acc, "elapsed": elapsed
    })
    pa_str = f"  pred_acc={pred_acc:.1f}%" if pred_acc is not None else ""
    print(f"  seed={seed:02d}  L1={l1_count}  unique_cells={len(unique_cells_set)}{pa_str}  ({elapsed:.1f}s)")

print()
print("=" * 70)
print("STEP 855 RESULTS")
print("=" * 70)
total_l1 = sum(r["l1"] for r in results)
mean_cells = np.mean([r["unique_cells"] for r in results])
mean_acc = np.mean([r["pred_acc"] for r in results if r["pred_acc"] is not None]) if any(r["pred_acc"] for r in results) else None
print(f"  Total L1: {total_l1}/{N_SEEDS} seeds")
print(f"  Mean unique cells: {mean_cells:.1f}")
if mean_acc is not None:
    print(f"  Mean prediction accuracy: {mean_acc:.2f}%")
print()
if total_l1 > 0:
    print("FINDING: Compression progress NAVIGATES LS20. L1 > 0. Anti-noisy-TV mechanism works.")
    print("Run R3_cf protocol to test transfer.")
else:
    print("FINDING: Compression progress achieves L1=0. Does not navigate LS20.")
    print("But prediction accuracy above may show D(s) transfer via secondary metric.")
print()
print("STEP 855 DONE")
