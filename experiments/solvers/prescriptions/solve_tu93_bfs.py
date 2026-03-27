"""
TU93 solver: BFS using game engine as simulator.
For each level, try all possible move sequences and find the shortest one that wins.
Uses the actual game engine to handle enemy movement correctly.
"""

import sys
import os
import json
import logging
import numpy as np
from collections import deque
from copy import deepcopy

from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo


def create_game():
    info = EnvironmentInfo(
        game_id='tu93',
        local_dir='environment_files/tu93/2b534c15',
        class_name='Tu93'
    )
    logger = logging.getLogger('tu93')
    logger.setLevel(logging.WARNING)
    return LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)


def get_frame_hash(game):
    """Get a hashable representation of the current game state."""
    # Access the internal game to get sprite positions
    g = game._game
    # Get player position
    players = g.current_level.get_sprites_by_tag("albwnmiahg")
    if not players:
        return None
    player = players[0]

    # Get all enemy positions and types
    enemies_vll = g.current_level.get_sprites_by_tag("vllvfeggte")
    enemies_zzu = g.current_level.get_sprites_by_tag("zzuxulcort")
    enemies_nat = g.current_level.get_sprites_by_tag("natiyqayts")

    state_parts = [player.x, player.y]
    for e in enemies_vll:
        state_parts.extend([e.x, e.y, e.rotation])
    for e in enemies_zzu:
        state_parts.extend([e.x, e.y, e.rotation])
    for e in enemies_nat:
        state_parts.extend([e.x, e.y, e.rotation])

    return tuple(state_parts)


def solve_level_brute(prefix_actions, level_idx, max_depth=50):
    """
    Solve a single level by brute-force BFS over the game simulator.
    prefix_actions: actions to play to reach this level (all previous level solutions)
    """
    print(f"\n  BFS for level {level_idx + 1}...")

    # We need to replay from scratch for each candidate path
    # To make this tractable, we use iterative deepening

    actions = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT

    # Try DFS with iterative deepening
    for depth in range(1, max_depth + 1):
        game = create_game()
        game.reset()
        # Play prefix
        for a in prefix_actions:
            game.step(a)

        # Now do DFS at this depth
        result = dfs_solve(game, prefix_actions, level_idx, depth)
        if result is not None:
            print(f"  SOLVED level {level_idx + 1} in {len(result)} moves!")
            return result

        if depth <= 3:
            print(f"    depth {depth}: no solution")

    print(f"  FAILED to solve level {level_idx + 1}")
    return None


def dfs_solve(game, prefix_actions, level_idx, max_depth):
    """DFS with the game engine. Need to replay from start for each path."""
    # BFS is more appropriate but requires state save/restore which we can't do
    # Instead, use BFS with full replay

    target_lc = level_idx + 1  # levels_completed we want to reach

    # BFS over move sequences
    queue = deque([[]])
    best = None

    while queue:
        path = queue.popleft()

        if len(path) >= max_depth:
            continue

        # Replay game to test this path
        game = create_game()
        game.reset()
        for a in prefix_actions:
            game.step(a)

        # Play the path
        alive = True
        for a in path:
            result = game.step(a)
            if result.state == 'GAME_OVER':
                alive = False
                break
            if result.levels_completed >= target_lc:
                return path  # Solved!

        if not alive:
            continue

        # Expand
        for action in [1, 2, 3, 4]:
            queue.append(path + [action])

    return None


def solve_level_replay_bfs(prefix_actions, level_idx, step_limit=50):
    """
    BFS using full game replay for each candidate.
    Optimization: use state hashing to prune visited states.
    """
    target_lc = level_idx + 1

    # Phase 1: Just try the simple BFS path from maze graph
    # Phase 2: If that fails (enemies), use iterative DFS

    queue = deque([[]])
    visited_states = set()

    iterations = 0
    max_iterations = 500000  # Safety limit

    while queue and iterations < max_iterations:
        path = queue.popleft()
        iterations += 1

        if iterations % 10000 == 0:
            print(f"    Iteration {iterations}, queue size {len(queue)}, path length {len(path)}")

        if len(path) > step_limit:
            continue

        # Replay
        game = create_game()
        game.reset()
        for a in prefix_actions:
            game.step(a)

        alive = True
        for a in path:
            result = game.step(a)
            if result.state == 'GAME_OVER':
                alive = False
                break
            if result.levels_completed >= target_lc:
                return path

        if not alive:
            continue

        # Get state hash
        state = get_frame_hash(game)
        if state is None or state in visited_states:
            continue
        visited_states.add(state)

        # Expand
        for action in [1, 2, 3, 4]:
            if len(path) + 1 <= step_limit:
                queue.append(path + [action])

    print(f"    BFS exhausted after {iterations} iterations")
    return None


def main():
    # Known solutions for L1 and L2
    l1_actions_0idx = [3, 1, 1, 3, 0, 3, 1, 1, 2, 2, 1, 3, 3, 1, 3, 0, 3, 1]
    l2_actions_0idx = [0, 3, 3, 1, 3, 3, 0, 3, 3, 0]

    # Convert to 1-indexed for game engine
    l1 = [a + 1 for a in l1_actions_0idx]
    l2 = [a + 1 for a in l2_actions_0idx]

    # Verify L1 + L2
    game = create_game()
    game.reset()
    for a in l1:
        result = game.step(a)
    print(f"After L1: lc={result.levels_completed}, state={result.state}")

    for a in l2:
        result = game.step(a)
    print(f"After L2: lc={result.levels_completed}, state={result.state}")

    # Build prefix for each level
    all_level_actions = {0: l1, 1: l2}

    for level_idx in range(2, 9):
        prefix = []
        for i in range(level_idx):
            prefix.extend(all_level_actions[i])

        print(f"\n{'='*50}")
        print(f"Solving Level {level_idx + 1}")
        print(f"{'='*50}")

        solution = solve_level_replay_bfs(prefix, level_idx, step_limit=35)

        if solution is not None:
            all_level_actions[level_idx] = solution
            # Convert to 0-indexed for output
            actions_0idx = [a - 1 for a in solution]
            move_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
            path_str = ' '.join([move_names[a] for a in actions_0idx])
            print(f"  Solution: {path_str}")
            print(f"  Actions (0-indexed): {actions_0idx}")
        else:
            print(f"  FAILED!")
            break

    # Verify full chain
    print(f"\n{'='*50}")
    print("VERIFICATION")
    print(f"{'='*50}")

    game = create_game()
    game.reset()
    total = 0
    for level_idx in range(9):
        if level_idx not in all_level_actions:
            print(f"Level {level_idx + 1}: NOT SOLVED")
            break
        actions = all_level_actions[level_idx]
        for a in actions:
            result = game.step(a)
            if result.state == 'GAME_OVER':
                print(f"Level {level_idx + 1}: DIED after {total} total actions")
                break
        else:
            total += len(actions)
            print(f"Level {level_idx + 1}: OK ({len(actions)} moves, lc={result.levels_completed})")
            continue
        break

    # Save results
    all_0idx = []
    level_0idx = {}
    for i in sorted(all_level_actions.keys()):
        acts = [a - 1 for a in all_level_actions[i]]
        level_0idx[i] = acts
        all_0idx.extend(acts)

    output = {
        "game": "tu93",
        "source": "bfs_maze_solver_v2",
        "type": "analytical",
        "total_actions": len(all_0idx),
        "max_level": max(all_level_actions.keys()) + 1,
        "all_actions": all_0idx,
    }
    for i in sorted(level_0idx.keys()):
        output[f"l{i+1}_actions"] = level_0idx[i]

    outpath = "experiments/results/prescriptions/tu93_fullchain.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Total actions: {len(all_0idx)}")


if __name__ == "__main__":
    main()
