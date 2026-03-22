#!/usr/bin/env python3
"""
Step 350 -- ARC-AGI-3: 16x16 resolution diagnostic on LS20.

1. avgpool kernel=4 (64->16x16 = 256 dims)
2. Print initial frame as 16x16 text grid
3. Take ACTION1 10 times, print frame + diff
4. Test ACTION3/4 from various positions
5. Cosine similarity at 16x16 (raw and centered)

Script: scripts/run_step350_resolution.py
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    """64x64 -> 16x16 (kernel=4). Returns (16,16) float32 in [0,1]."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3))  # (16,16)


def avgpool8(frame):
    """64x64 -> 8x8 (kernel=8). Returns (8,8) float32 in [0,1]."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3))


def print_grid(grid, label=""):
    if label:
        print(f"  {label}:")
    for row in grid:
        print("    " + " ".join(f"{v:.2f}" for v in row))


def cos_sim(a, b):
    """Cosine similarity between two flat arrays."""
    ta = torch.from_numpy(a.flatten().astype(np.float32))
    tb = torch.from_numpy(b.flatten().astype(np.float32))
    return float(F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)).item())


def main():
    import arc_agi
    from arcengine import GameState

    print("Step 350 -- 16x16 resolution diagnostic on LS20", flush=True)
    print("=" * 60, flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    action_names = ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4']

    # -------------------------------------------------------------------------
    # 1. Initial frame at 16x16
    # -------------------------------------------------------------------------
    print("\n[1] Initial frame at 16x16 (avgpool kernel=4):", flush=True)
    env = arc.make(ls20.game_id)
    obs = env.reset()
    f0_16 = avgpool16(obs.frame)
    f0_8  = avgpool8(obs.frame)
    print_grid(f0_16, "16x16 frame")

    # -------------------------------------------------------------------------
    # 2. Take ACTION1 x10, track frames
    # -------------------------------------------------------------------------
    print("\n[2] 10x ACTION1 (left): frame diffs at 16x16", flush=True)
    env2 = arc.make(ls20.game_id)
    obs2 = env2.reset()
    prev16 = avgpool16(obs2.frame)
    for i in range(10):
        obs2 = env2.step(env2.action_space[0])  # ACTION1
        if obs2 is None or obs2.state in [GameState.GAME_OVER, GameState.WIN]:
            print(f"  step {i+1}: game ended")
            break
        curr16 = avgpool16(obs2.frame)
        diff = np.abs(curr16 - prev16)
        n_changed = (diff > 0.01).sum()
        max_diff = diff.max()
        print(f"  step {i+1}: max_diff={max_diff:.4f}  cells_changed(>0.01)={n_changed}", flush=True)
        prev16 = curr16

    print("\n  Frame after 10x ACTION1:", flush=True)
    print_grid(curr16, "16x16 frame")
    diff_from_start = np.abs(curr16 - f0_16)
    print_grid(diff_from_start, "diff from initial")

    # -------------------------------------------------------------------------
    # 3. ACTION3/4 at multiple positions (after 0, 5, 10 ACTION1 presses)
    # -------------------------------------------------------------------------
    print("\n[3] ACTION3/4 effect at various positions (16x16):", flush=True)
    for warmup_a1 in [0, 5, 10, 20]:
        print(f"\n  After {warmup_a1}x ACTION1:", flush=True)
        for act_idx in range(4):
            env3 = arc.make(ls20.game_id)
            obs3 = env3.reset()
            for _ in range(warmup_a1):
                obs3 = env3.step(env3.action_space[0])
                if obs3 is None or obs3.state in [GameState.GAME_OVER, GameState.WIN]:
                    break
            if obs3 is None:
                continue
            base16 = avgpool16(obs3.frame)
            obs3 = env3.step(env3.action_space[act_idx])
            if obs3 is None:
                continue
            after16 = avgpool16(obs3.frame)
            diff = np.abs(after16 - base16)
            n = (diff > 0.01).sum()
            mx = diff.max()
            cs = cos_sim(base16, after16)
            print(f"    {action_names[act_idx]}: max_diff={mx:.4f}  cells={n:3d}  cos_sim={cs:.6f}", flush=True)

    # -------------------------------------------------------------------------
    # 4. Cosine similarity at 16x16: consecutive frames (raw and centered)
    # -------------------------------------------------------------------------
    print("\n[4] Cosine similarity: consecutive frames at 16x16", flush=True)
    env4 = arc.make(ls20.game_id)
    obs4 = env4.reset()
    frames16 = [avgpool16(obs4.frame)]
    for act_idx in [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]:
        obs4 = env4.step(env4.action_space[act_idx])
        if obs4 is None or obs4.state in [GameState.GAME_OVER, GameState.WIN]:
            break
        frames16.append(avgpool16(obs4.frame))

    print("  Raw cosine sims (consecutive frames):", flush=True)
    for i in range(len(frames16) - 1):
        cs = cos_sim(frames16[i], frames16[i+1])
        print(f"    frame {i}->{i+1}: {cs:.6f}", flush=True)

    # Center by mean
    all_f = np.stack(frames16)  # (N, 16, 16)
    mean_f = all_f.mean(axis=0)
    centered = all_f - mean_f[np.newaxis]
    print("\n  Centered cosine sims (subtract mean frame):", flush=True)
    for i in range(len(frames16) - 1):
        fa = centered[i].flatten().astype(np.float32)
        fb = centered[i+1].flatten().astype(np.float32)
        ta = torch.from_numpy(fa)
        tb = torch.from_numpy(fb)
        cs = float(F.cosine_similarity(ta.unsqueeze(0), tb.unsqueeze(0)).item())
        print(f"    frame {i}->{i+1}: {cs:.6f}", flush=True)

    # -------------------------------------------------------------------------
    # 5. Look for distinct goal region
    # -------------------------------------------------------------------------
    print("\n[5] Goal region search: variance across 20 frames (high variance = dynamic)", flush=True)
    env5 = arc.make(ls20.game_id)
    obs5 = env5.reset()
    import random
    frames_var = []
    step_v = 0
    while step_v < 20 and obs5 is not None:
        frames_var.append(avgpool16(obs5.frame))
        obs5 = env5.step(random.choice(env5.action_space[:2]))  # ACTION1/2
        step_v += 1
        if obs5 is None or obs5.state in [GameState.GAME_OVER, GameState.WIN]:
            break

    var_grid = np.stack(frames_var).var(axis=0)
    print("  Variance grid (16x16) — high values = cells that change with movement:", flush=True)
    print_grid(var_grid, "variance")

    # Find top-5 most variable cells
    flat_var = var_grid.flatten()
    top5 = np.argsort(flat_var)[-5:][::-1]
    print("\n  Top 5 most variable cells (row, col, variance):", flush=True)
    for idx in top5:
        r, c = divmod(idx, 16)
        print(f"    ({r:2d},{c:2d}): var={flat_var[idx]:.6f}", flush=True)

    # Static cells (zero or near-zero variance) = background/walls
    n_static = (flat_var < 1e-6).sum()
    n_dynamic = (flat_var > 0.0001).sum()
    print(f"\n  Static cells (var<1e-6): {n_static}/256", flush=True)
    print(f"  Dynamic cells (var>0.001): {n_dynamic}/256", flush=True)


if __name__ == '__main__':
    main()
