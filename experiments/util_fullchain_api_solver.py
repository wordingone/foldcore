"""
Fullchain API-based solver for m0r0, ka59, wa30.
Uses the arc_agi API directly for BFS, with frame hashing.
Replays from scratch for each state but limits search depth per level.

Usage: PYTHONUTF8=1 python experiments/util_fullchain_api_solver.py [game_id]
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
from collections import deque

import arc_agi
from arcengine import GameAction, GameState


def frame_hash(obs):
    """Hash frame from observation for state dedup."""
    if obs is None or obs.frame is None:
        return None
    arr = np.asarray(obs.frame, dtype=np.int8)
    if arr.ndim == 3:
        arr = arr[-1]
    return hashlib.md5(arr.tobytes()).hexdigest()


def make_env(arcade, game_id):
    """Create a fresh game env."""
    games = arcade.get_environments()
    info = next(g for g in games if game_id in g.game_id.lower())
    return arcade.make(info.game_id)


def replay_actions(arcade, game_id, actions):
    """Replay an action sequence, return (obs, env)."""
    env = make_env(arcade, game_id)
    obs = env.reset()
    for a in actions:
        if a >= 7:
            ci = a - 7
            obs = env.step(GameAction.ACTION6, data={"x": ci % 64, "y": ci // 64})
        else:
            ga = list(GameAction)[a + 1]
            if ga == GameAction.ACTION6 and any(
                ga2 == GameAction.ACTION6 for ga2 in env.action_space
            ):
                obs = env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
            else:
                obs = env.step(ga)
        if obs is None:
            break
    return obs, env


def get_kb_actions(game_id):
    """Get keyboard action IDs for a game."""
    if game_id == "wa30":
        return [0, 1, 2, 3, 4]  # UP DOWN LEFT RIGHT GRAB
    elif game_id == "ka59":
        return [0, 1, 2, 3]  # UP DOWN LEFT RIGHT (click handled separately)
    elif game_id == "m0r0":
        return [0, 1, 2, 3, 4]  # UP DOWN LEFT RIGHT ACTION5
    return [0, 1, 2, 3]


def get_click_targets(obs, game_id, env):
    """Extract click targets from frame/game state."""
    if game_id == "ka59":
        # Click on player sprites to switch active player
        game = env._game
        players = game.current_level.get_sprites_by_tag("xlfuqjygey")
        active = game.ascpmvdpwj
        grid_w, grid_h = game.current_level.grid_size or (63, 63)
        scale = min(64 // grid_w, 64 // grid_h)
        x_off = (64 - grid_w * scale) // 2
        y_off = (64 - grid_h * scale) // 2

        clicks = []
        for p in players:
            if p is not active:
                # Center of player sprite
                px = int((p.x + p.width / 2) * scale + x_off)
                py = int((p.y + p.height / 2) * scale + y_off)
                px = max(0, min(63, px))
                py = max(0, min(63, py))
                clicks.append(7 + py * 64 + px)
        return clicks

    elif game_id == "m0r0":
        # Click on cvcer toggle blocks
        game = env._game
        toggles = game.current_level.get_sprites_by_name("cvcer")
        if not toggles:
            return []
        grid_w, grid_h = game.current_level.grid_size or (13, 13)
        scale = min(64 // grid_w, 64 // grid_h)
        x_off = (64 - grid_w * scale) // 2
        y_off = (64 - grid_h * scale) // 2

        clicks = []
        for t in toggles:
            px = int((t.x + 0.5) * scale + x_off)
            py = int((t.y + 0.5) * scale + y_off)
            px = max(0, min(63, px))
            py = max(0, min(63, py))
            clicks.append(7 + py * 64 + px)
        return clicks

    return []


def bfs_level(arcade, game_id, prior_actions, max_depth=80, max_states=100000):
    """
    BFS a single level using API replay.

    Returns: list of actions for this level, or None.
    """
    kb_actions = get_kb_actions(game_id)

    # Get initial state for this level
    obs, env = replay_actions(arcade, game_id, prior_actions)
    if obs is None:
        print("  ERROR: can't reach level")
        return None

    start_level = obs.levels_completed
    h0 = frame_hash(obs)

    # Get click targets
    click_actions = get_click_targets(obs, game_id, env)

    all_actions = kb_actions + click_actions
    print(f"  BFS: {len(all_actions)} actions ({len(kb_actions)} kb + {len(click_actions)} click)")
    print(f"  Start level: {start_level}")

    visited = {h0}
    queue = deque()
    queue.append([])

    states_explored = 0
    max_d = 0
    t0 = time.time()

    while queue and states_explored < max_states:
        seq = queue.popleft()
        depth = len(seq)

        if depth > max_depth:
            break

        if depth > max_d:
            max_d = depth
            elapsed = time.time() - t0
            print(f"    d={depth}, q={len(queue)}, v={len(visited)}, "
                  f"explored={states_explored}, {elapsed:.1f}s")

        for a in all_actions:
            states_explored += 1
            new_seq = seq + [a]

            obs2, env2 = replay_actions(arcade, game_id, prior_actions + new_seq)
            if obs2 is None:
                continue
            if obs2.state == GameState.GAME_OVER:
                continue
            if obs2.levels_completed > start_level:
                elapsed = time.time() - t0
                print(f"  SOLVED! depth={len(new_seq)}, states={states_explored}, {elapsed:.1f}s")
                return new_seq

            h = frame_hash(obs2)
            if h and h not in visited:
                visited.add(h)

                # Update click targets (player positions may have changed)
                new_clicks = get_click_targets(obs2, game_id, env2)
                for c in new_clicks:
                    if c not in all_actions:
                        all_actions.append(c)

                queue.append(new_seq)

    elapsed = time.time() - t0
    print(f"  BFS exhausted: {states_explored} states, depth={max_d}, {elapsed:.1f}s")
    return None


def solve_game(game_id, n_levels, l1_actions=None, max_depth=80, max_states=100000):
    """Solve a game through all levels."""
    print(f"\n{'='*60}")
    print(f"SOLVING {game_id} ({n_levels} levels)")
    print(f"{'='*60}")

    arcade = arc_agi.Arcade()

    # Use existing L1 solution
    if l1_actions:
        # Verify L1
        obs, env = replay_actions(arcade, game_id, l1_actions)
        if obs and obs.levels_completed >= 1:
            # Trim to exact solution
            for trim_len in range(len(l1_actions), 0, -1):
                obs2, _ = replay_actions(arcade, game_id, l1_actions[:trim_len])
                if obs2 and obs2.levels_completed >= 1:
                    l1_actions = l1_actions[:trim_len]
                else:
                    l1_actions = l1_actions[:trim_len + 1]
                    break
            print(f"L1 verified: {len(l1_actions)} actions")
        else:
            print(f"L1 solution FAILED verification!")
            return None
    else:
        # BFS L1 from scratch
        l1_actions_result = bfs_level(arcade, game_id, [], max_depth=max_depth, max_states=max_states)
        if l1_actions_result is None:
            print("FAILED to solve L1")
            return None
        l1_actions = l1_actions_result

    all_actions = list(l1_actions)
    per_level = {"L1": {"actions": l1_actions, "count": len(l1_actions)}}

    for level_num in range(2, n_levels + 1):
        print(f"\n{'='*60}")
        print(f"{game_id} Level {level_num}/{n_levels}")
        print(f"{'='*60}")

        t0 = time.time()
        solution = bfs_level(arcade, game_id, all_actions,
                             max_depth=max_depth, max_states=max_states)
        elapsed = time.time() - t0

        if solution is None:
            print(f"FAILED L{level_num} ({elapsed:.1f}s)")
            break

        per_level[f"L{level_num}"] = {
            "actions": solution,
            "count": len(solution),
            "time": round(elapsed, 2)
        }
        all_actions.extend(solution)
        print(f"Solved L{level_num}: {len(solution)} actions in {elapsed:.1f}s")

    # Verify
    obs, env = replay_actions(arcade, game_id, all_actions)
    max_level = obs.levels_completed if obs else 0
    won = obs.state == GameState.WIN if obs else False
    print(f"\nVerification: max_level={max_level}, won={won}")

    result = {
        "game": game_id,
        "source": "api_bfs_solver",
        "type": "analytical",
        "total_actions": len(all_actions),
        "max_level": max_level,
        "won": won,
        "per_level": per_level,
        "all_actions": all_actions,
    }

    out_path = f"B:/M/the-search/experiments/results/prescriptions/{game_id}_fullchain.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")
    return result


if __name__ == "__main__":
    os.chdir("B:/M/the-search")
    game = sys.argv[1] if len(sys.argv) > 1 else "all"

    if game in ["wa30", "all"]:
        with open("experiments/results/prescriptions/wa30_analytical.json") as f:
            wa30_l1 = json.load(f)["all_actions"]
        solve_game("wa30", 9, l1_actions=wa30_l1, max_depth=60, max_states=50000)

    if game in ["ka59", "all"]:
        with open("experiments/results/prescriptions/ka59_analytical.json") as f:
            ka59_l1 = json.load(f)["all_actions"]
        solve_game("ka59", 7, l1_actions=ka59_l1, max_depth=60, max_states=50000)

    if game in ["m0r0", "all"]:
        with open("experiments/results/prescriptions/m0r0_full_seq.json") as f:
            m0r0_l1 = json.load(f)["full_sequence"]
        solve_game("m0r0", 6, l1_actions=m0r0_l1, max_depth=60, max_states=50000)
