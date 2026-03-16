#!/usr/bin/env python3
"""
Step 374 -- VC33 deep visual diagnostic. Pure observation.
Script: scripts/run_step374_vc33_diagnostic.py
"""

import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def main():
    import arc_agi
    from arcengine import GameState

    print("Step 374 -- VC33 deep diagnostic", flush=True)
    print("=" * 60, flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    # 1. Game metadata
    print(f"\n[1] Game metadata:", flush=True)
    print(f"  title: {vc33.title}", flush=True)
    print(f"  game_id: {vc33.game_id}", flush=True)
    # Check for win_levels attribute
    for attr in dir(vc33):
        if not attr.startswith('_'):
            try:
                val = getattr(vc33, attr)
                if not callable(val):
                    print(f"  {attr}: {val}", flush=True)
            except:
                pass

    env = arc.make(vc33.game_id)
    obs = env.reset()
    action_space = env.action_space

    print(f"\n  action_space: {len(action_space)} actions", flush=True)
    for i, a in enumerate(action_space):
        print(f"    [{i}] {a.name} is_complex={a.is_complex()}", flush=True)

    # Check for game attributes
    print(f"\n  obs attributes:", flush=True)
    for attr in dir(obs):
        if not attr.startswith('_'):
            try:
                val = getattr(obs, attr)
                if not callable(val):
                    if attr != 'frame':
                        print(f"    {attr}: {val}", flush=True)
            except:
                pass

    # 2. Full life observation (50 steps)
    print(f"\n[2] Full life observation (50 steps):", flush=True)
    env2 = arc.make(vc33.game_id)
    obs2 = env2.reset()

    snapshot_steps = [0, 1, 10, 25, 40, 49]
    frames = {}
    prev_frame = None

    for step in range(52):
        frame = np.array(obs2.frame[0])

        if step in snapshot_steps:
            frames[step] = frame.copy()
            # Print non-background cells
            print(f"\n  Step {step}: state={obs2.state} levels={obs2.levels_completed}", flush=True)
            # Find distinctive values (not background)
            unique_vals = np.unique(frame)
            print(f"    unique pixel values: {sorted(unique_vals)}", flush=True)
            # Show non-zero, non-background cells
            bg_val = np.median(frame)  # assume most common = background
            non_bg = np.argwhere(frame != bg_val)
            if len(non_bg) > 0 and len(non_bg) < 100:
                print(f"    non-background cells ({len(non_bg)} total, bg={bg_val}):", flush=True)
                for r, c in non_bg[:30]:
                    print(f"      ({r:2d},{c:2d}) = {frame[r,c]}", flush=True)
                if len(non_bg) > 30:
                    print(f"      ... ({len(non_bg) - 30} more)", flush=True)
            elif len(non_bg) >= 100:
                print(f"    {len(non_bg)} non-background cells (too many to list, bg={bg_val})", flush=True)
                # Show value distribution
                vals, counts = np.unique(frame, return_counts=True)
                for v, c in zip(vals, counts):
                    print(f"      value {v:2d}: {c:4d} cells ({c/4096*100:.1f}%)", flush=True)

        if prev_frame is not None:
            diff = (frame != prev_frame)
            n_changed = diff.sum()
            if n_changed > 0 and step <= 50:
                changed_pos = np.argwhere(diff)[:5]
                print(f"  Step {step}: {n_changed} cells changed"
                      f" (e.g. {[(r,c,int(prev_frame[r,c]),int(frame[r,c])) for r,c in changed_pos]})", flush=True)
        prev_frame = frame.copy()

        if obs2.state == GameState.GAME_OVER:
            print(f"  Step {step}: GAME_OVER", flush=True)
            break
        if obs2.state == GameState.WIN:
            print(f"  Step {step}: WIN levels={obs2.levels_completed}", flush=True)
            break

        obs2 = env2.step(action_space[0], data={"x": 32, "y": 32})
        if obs2 is None:
            print(f"  Step {step+1}: obs=None", flush=True)
            break

    # 3. Click on distinctive elements
    print(f"\n[3] Click on distinctive elements:", flush=True)

    # Find distinctive cells from frame at step 10
    if 10 in frames:
        f10 = frames[10]
        bg = int(np.median(f10))
        non_bg_pos = np.argwhere(f10 != bg)
        if len(non_bg_pos) > 0:
            # Try clicking on first few distinctive cells
            test_clicks = [(int(r), int(c)) for r, c in non_bg_pos[:5]]
        else:
            test_clicks = [(0, 0), (32, 32), (63, 63)]
    else:
        test_clicks = [(0, 0), (32, 32), (63, 63)]

    # Also add generic positions
    test_clicks.extend([(0, 0), (32, 32), (63, 63), (10, 10)])

    for y, x in test_clicks[:8]:
        env3 = arc.make(vc33.game_id)
        obs3 = env3.reset()
        # Advance to step 10
        for _ in range(10):
            obs3 = env3.step(action_space[0], data={"x": 32, "y": 32})
            if obs3 is None or obs3.state in [GameState.GAME_OVER, GameState.WIN]:
                break
        if obs3 is None or obs3.state != GameState.NOT_FINISHED:
            continue
        base = np.array(obs3.frame[0])
        obs3 = env3.step(action_space[0], data={"x": x, "y": y})
        if obs3 is None:
            print(f"  click ({x:2d},{y:2d}): obs=None", flush=True)
            continue
        after = np.array(obs3.frame[0])
        diff = (after != base).sum()
        state = obs3.state
        lvls = obs3.levels_completed
        print(f"  click ({x:2d},{y:2d}): state={state} levels={lvls}"
              f" cells_changed={diff}/4096", flush=True)

    # 4. Does clicking EVER produce a different state?
    print(f"\n[4] Exhaustive click test from step 10:", flush=True)
    states_seen = set()
    for x in range(0, 64, 4):
        for y in range(0, 64, 4):
            env4 = arc.make(vc33.game_id)
            obs4 = env4.reset()
            for _ in range(10):
                obs4 = env4.step(action_space[0], data={"x": 32, "y": 32})
                if obs4 is None or obs4.state != GameState.NOT_FINISHED: break
            if obs4 is None or obs4.state != GameState.NOT_FINISHED: continue
            obs4 = env4.step(action_space[0], data={"x": x, "y": y})
            if obs4 is None: continue
            frame_hash = hash(np.array(obs4.frame[0]).tobytes())
            states_seen.add(frame_hash)

    print(f"  Unique frames after clicking at 256 positions: {len(states_seen)}", flush=True)
    if len(states_seen) == 1:
        print(f"  CONFIRMED: click position has ZERO effect on frame.", flush=True)
    elif len(states_seen) > 1:
        print(f"  FOUND: click position produces {len(states_seen)} different frames!", flush=True)


if __name__ == '__main__':
    main()
