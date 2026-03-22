#!/usr/bin/env python3
"""
Step 360 -- FT09 resolution sweep + VC33 action space inspection.

FT09: try 32x32 and 64x64. Take 10 random actions at each resolution.
VC33: inspect action_space, is_complex(), data dict.
Script: scripts/run_step360_ft09_resolution.py
"""

import logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool(frame, kernel):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    h, w = arr.shape
    bh, bw = h // kernel, w // kernel
    return arr.reshape(bh, kernel, bw, kernel).mean(axis=(1, 3))


def main():
    import arc_agi
    from arcengine import GameState

    print("Step 360 -- FT09 resolution sweep + VC33 action inspection", flush=True)
    print("=" * 60, flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()

    # ---- FT09 resolution sweep ----
    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())
    print(f"\n[FT09] {ft09.title} ({ft09.game_id})", flush=True)

    for kernel, label in [(4, "16x16"), (2, "32x32"), (1, "64x64")]:
        res = 64 // kernel
        print(f"\n  Resolution: {label} ({res}x{res} = {res*res} dims)", flush=True)

        env = arc.make(ft09.game_id)
        obs = env.reset()
        action_space = env.action_space

        prev = avgpool(obs.frame, kernel).flatten()
        n_changed_list = []

        for i in range(10):
            act = action_space[i % len(action_space)]
            data = {}
            if act.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs = env.step(act, data=data)
            if obs is None or obs.state in [GameState.GAME_OVER, GameState.WIN]:
                print(f"    step {i+1}: game ended ({obs.state if obs else 'None'})", flush=True)
                break
            curr = avgpool(obs.frame, kernel).flatten()
            diff = np.abs(curr - prev)
            n_changed = (diff > 0.01).sum()
            max_diff = diff.max()
            n_changed_list.append(n_changed)
            print(f"    step {i+1} ({act.name}): max_diff={max_diff:.4f}"
                  f"  cells_changed(>0.01)={n_changed}/{res*res}", flush=True)
            prev = curr

        if n_changed_list:
            print(f"    avg cells changed: {np.mean(n_changed_list):.1f}/{res*res}", flush=True)

    # ---- FT09 action space details ----
    print(f"\n  FT09 action_space ({len(action_space)} actions):", flush=True)
    for i, a in enumerate(action_space):
        print(f"    [{i}] {a.name}  is_complex={a.is_complex()}", flush=True)

    # ---- VC33 action space inspection ----
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())
    print(f"\n[VC33] {vc33.title} ({vc33.game_id})", flush=True)

    env2 = arc.make(vc33.game_id)
    obs2 = env2.reset()
    action_space2 = env2.action_space

    print(f"  action_space ({len(action_space2)} actions):", flush=True)
    for i, a in enumerate(action_space2):
        print(f"    [{i}] {a.name}  is_complex={a.is_complex()}", flush=True)

    # Try VC33 with different data dicts
    print(f"\n  Testing VC33 ACTION6 with various data:", flush=True)
    test_data = [
        {},
        {"x": 0, "y": 0},
        {"x": 32, "y": 32},
        {"x": 10, "y": 50},
    ]
    for d in test_data:
        env3 = arc.make(vc33.game_id)
        obs3 = env3.reset()
        base = avgpool(obs3.frame, 4).flatten()
        obs3 = env3.step(action_space2[0], data=d)
        if obs3 is None:
            print(f"    data={d}: obs=None", flush=True)
            continue
        after = avgpool(obs3.frame, 4).flatten()
        diff = np.abs(after - base)
        n = (diff > 0.01).sum()
        print(f"    data={d}: max_diff={diff.max():.4f}  cells_changed={n}/256"
              f"  state={obs3.state}", flush=True)

    # VC33 full resolution test
    print(f"\n  VC33 at full 64x64:", flush=True)
    for d in [{"x": 0, "y": 0}, {"x": 32, "y": 32}]:
        env4 = arc.make(vc33.game_id)
        obs4 = env4.reset()
        base64 = avgpool(obs4.frame, 1).flatten()
        obs4 = env4.step(action_space2[0], data=d)
        if obs4 is None:
            print(f"    data={d}: obs=None", flush=True)
            continue
        after64 = avgpool(obs4.frame, 1).flatten()
        diff64 = np.abs(after64 - base64)
        n64 = (diff64 > 0.01).sum()
        print(f"    data={d}: max_diff={diff64.max():.4f}  cells_changed={n64}/4096", flush=True)


if __name__ == '__main__':
    main()
