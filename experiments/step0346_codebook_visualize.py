#!/usr/bin/env python3
"""
Step 346 -- Visualize ARC-AGI-3 frames.

For each game:
  - Frame at step 0
  - Frame after each of 10 random actions
  - Frame at GAME_OVER (if triggered)
  - Frame after reset

For first 5 frames: print 8x8 avgpool as text grid (values 0-15).
For LS20: diff between ACTION1 vs ACTION4 from same start state.

Script: scripts/run_step346_visualize.py
"""

import logging
import random
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def frame_to_grid(frame, label=""):
    """Print 64x64 frame[0] as 8x8 avgpool text grid (values 0-15, 2-char wide)."""
    arr = np.array(frame[0], dtype=np.float32)
    pooled = arr.reshape(8, 8, 8, 8).mean(axis=(2, 3))  # (8, 8)
    lines = []
    if label:
        lines.append(f"  {label}")
    lines.append("  " + " ".join(f"{int(v*15):2d}" for v in pooled[0]))
    for row in pooled:
        lines.append("  " + " ".join(f"{int(v*15):2d}" for row_v in [row] for v in row_v))
    return "\n".join(lines[1:])  # skip duplicate first line


def avgpool8(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(8, 8, 8, 8).mean(axis=(2, 3)).flatten()


def print_frame_grid(frame, label=""):
    arr = np.array(frame[0], dtype=np.float32)
    pooled = arr.reshape(8, 8, 8, 8).mean(axis=(2, 3))  # (8, 8)
    if label:
        print(f"  [{label}]")
    for row in pooled:
        print("  " + " ".join(f"{int(v*15):2d}" for v in row))


def print_diff_grid(diff_arr, label=""):
    """diff_arr: (8,8) float. Show non-zero cells."""
    if label:
        print(f"  [{label}]")
    nz = []
    for r in range(8):
        for c in range(8):
            v = diff_arr[r, c]
            if abs(v) > 0.001:
                nz.append(f"    ({r},{c}): {v:+.3f}")
    if nz:
        for line in nz:
            print(line)
    else:
        print("    (no change — zero diff)")
    # Also print as grid
    for row in diff_arr:
        print("  " + " ".join(f"{v:+5.2f}" for v in row))


def explore_game(arc, game_id, title):
    from arcengine import GameState

    print(f"\n{'='*60}")
    print(f"GAME: {title} ({game_id})")
    print('='*60)

    env  = arc.make(game_id)
    obs  = env.reset()
    n_acts = len(env.action_space)
    print(f"  action_space ({n_acts}): {[a.name for a in env.action_space]}")
    print(f"  state: {obs.state}  levels_completed: {obs.levels_completed}  win_levels: {obs.win_levels}")
    print()

    frames = []   # list of (label, frame, pooled)
    game_over_frame = None
    post_reset_frame = None

    # Step 0
    p0 = avgpool8(obs.frame)
    frames.append(("step_0_initial", obs.frame, p0))
    print(f"  step 0 (initial):")
    print_frame_grid(obs.frame, "8x8 avgpool (0-15)")
    print()

    # 10 random actions
    prev_obs = obs
    for i in range(10):
        action = random.choice(env.action_space)
        data = {}
        if action.is_complex():
            arr = np.array(prev_obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}

        obs = env.step(action, data=data)
        if obs is None:
            print(f"  step {i+1}: obs=None after {action.name}")
            break

        p = avgpool8(obs.frame)
        label = f"step_{i+1}_{action.name}_state={obs.state.name if hasattr(obs.state,'name') else obs.state}"
        frames.append((label, obs.frame, p))

        # Diff from previous
        prev_p = frames[-2][2]
        diff = p - prev_p
        nz_count = np.sum(np.abs(diff) > 0.001)

        print(f"  step {i+1}: action={action.name}  state={obs.state}  "
              f"levels={obs.levels_completed}  nonzero_diff={nz_count}/64")

        if obs.state == GameState.GAME_OVER:
            game_over_frame = obs.frame
            print(f"    -> GAME_OVER at step {i+1}")
            break
        if obs.state == GameState.WIN:
            print(f"    -> WIN at step {i+1}")
            break

        prev_obs = obs

    # Show first 5 frames as text grids
    print()
    print("  First 5 frames (8x8 avgpool):")
    for label, frame, p in frames[:5]:
        print(f"  [{label}]")
        arr = np.array(frame[0], dtype=np.float32)
        pooled = arr.reshape(8, 8, 8, 8).mean(axis=(2, 3))
        for row in pooled:
            print("  " + " ".join(f"{int(v*15):2d}" for v in row))
        print()

    # GAME_OVER frame
    if game_over_frame is not None:
        print("  GAME_OVER frame (8x8 avgpool):")
        print_frame_grid(game_over_frame, "game_over")
        print()

    # Reset
    obs_r = env.reset()
    if obs_r is not None:
        post_reset_frame = obs_r.frame
        print("  Post-reset frame (8x8 avgpool):")
        print_frame_grid(post_reset_frame, "post_reset")
        print()
        # Compare initial vs post-reset
        p_init  = frames[0][2]
        p_reset = avgpool8(post_reset_frame)
        diff_reset = p_reset - p_init
        nz = np.sum(np.abs(diff_reset) > 0.001)
        print(f"  initial vs post-reset: nonzero diff cells = {nz}/64")
        if nz > 0:
            pooled_diff = diff_reset.reshape(8, 8)
            print("  diff grid (post_reset - initial):")
            for row in pooled_diff:
                print("  " + " ".join(f"{v:+5.2f}" for v in row))
        print()

    return frames


def compare_actions_ls20(arc, game_id):
    """Compare ACTION1 vs ACTION4 from same start state."""
    from arcengine import GameState

    print(f"\n{'='*60}")
    print(f"LS20: ACTION comparison from same start state")
    print('='*60)

    for trial in range(3):
        print(f"\n  Trial {trial+1}:")

        # ACTION1
        env1 = arc.make(game_id)
        obs1 = env1.reset()
        p_base = avgpool8(obs1.frame)
        act1 = env1.action_space[0]  # ACTION1
        obs_a1 = env1.step(act1)
        if obs_a1 is None:
            print(f"    ACTION1 -> obs=None")
            continue
        p_a1   = avgpool8(obs_a1.frame)
        diff_a1 = p_a1 - p_base
        nz_a1   = np.sum(np.abs(diff_a1) > 0.001)

        # ACTION4 (same initial state via fresh env)
        env4 = arc.make(game_id)
        obs4 = env4.reset()
        act4 = env4.action_space[3]  # ACTION4
        obs_a4 = env4.step(act4)
        if obs_a4 is None:
            print(f"    ACTION4 -> obs=None")
            continue
        p_a4   = avgpool8(obs_a4.frame)
        diff_a4 = p_a4 - p_base

        # Verify base frames are same
        base_match = np.allclose(p_base, avgpool8(obs4.frame), atol=0.001)
        print(f"    base frames identical: {base_match}")
        print(f"    {act1.name}: nonzero_diff={np.sum(np.abs(diff_a1)>0.001)}/64  "
              f"state={obs_a1.state}  levels={obs_a1.levels_completed}")
        print(f"    {act4.name}: nonzero_diff={np.sum(np.abs(diff_a4)>0.001)}/64  "
              f"state={obs_a4.state}  levels={obs_a4.levels_completed}")

        print(f"\n    {act1.name} diff (non-zero cells, 8x8 avgpool):")
        d1 = diff_a1.reshape(8, 8)
        for row in d1:
            cells = " ".join(f"{v:+5.2f}" for v in row)
            print(f"      {cells}")

        print(f"\n    {act4.name} diff (non-zero cells, 8x8 avgpool):")
        d4 = diff_a4.reshape(8, 8)
        for row in d4:
            cells = " ".join(f"{v:+5.2f}" for v in row)
            print(f"      {cells}")

        # Where do they differ from each other?
        diff_between = d4 - d1
        nz_between = np.sum(np.abs(diff_between) > 0.001)
        print(f"\n    diff(ACTION4 - ACTION1): nonzero={nz_between}/64")
        if nz_between > 0:
            for r in range(8):
                for c in range(8):
                    v = diff_between[r, c]
                    if abs(v) > 0.001:
                        print(f"      ({r},{c}): {v:+.4f}")


def main():
    import arc_agi
    random.seed(42)

    print("Step 346 -- ARC-AGI-3 frame visualization", flush=True)
    print()

    arc   = arc_agi.Arcade()
    games = arc.get_environments()
    print(f"Games: {[g.game_id for g in games]}")

    # Explore all three games
    for g in games:
        explore_game(arc, g.game_id, g.title)

    # LS20 action comparison
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if ls20:
        compare_actions_ls20(arc, ls20.game_id)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
