"""
SU15 solver using game engine as black-box simulator.
BFS with state deduplication to find click sequences for each level.

Encoding: action_id = y * 64 + x + 7
Game API: step(GameAction.ACTION6, data={"x": x, "y": y})
"""

import sys
import os
import json
import logging
import numpy as np
from collections import deque

from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo
from arcengine import GameAction


def create_game():
    info = EnvironmentInfo(
        game_id='su15',
        local_dir='environment_files/su15/4c352900',
        class_name='Su15'
    )
    logger = logging.getLogger('su15')
    logger.setLevel(logging.WARNING)
    return LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)


def encode_action(x, y):
    """Encode x,y to action_id for JSON output."""
    return y * 64 + x + 7


def get_state_hash(game):
    """Get hashable state representation."""
    g = game._game
    parts = []
    for sprite in sorted(g.hmeulfxgy, key=lambda s: (s.x, s.y)):
        size = g.amnmgwpkeb.get(sprite, 0)
        parts.append((size, sprite.x, sprite.y))
    for sprite in sorted(g.peiiyyzum, key=lambda s: (s.x, s.y)):
        parts.append((-1, sprite.x, sprite.y))
    return tuple(parts)


def get_candidate_clicks(level_idx):
    """Get candidate (x,y) positions to try for each level."""

    # Level definitions from source
    levels = [
        # Level 1 (solved)
        {'fruits': [(3,58,2)], 'goals': [(44,11)], 'steps': 32},
        # Level 2 (solved)
        {'fruits': [(41,37,0),(18,37,0),(37,40,0),(16,41,0),(14,55,0),(16,57,0),(49,54,0),(47,56,0)],
         'goals': [(29,23)], 'steps': 32},
        # Level 3
        {'fruits': [(55,23,0),(61,23,0),(31,22,0),(31,15,0),(12,23,0),(8,28,0),
                    (46,22,1),(30,32,1),(18,16,1)],
         'goals': [(5,46),(19,46)], 'steps': 48},
        # Level 4
        {'fruits': [(5,26,0),(11,26,0),(31,27,0),(36,29,0),(33,47,0),(30,51,0),(12,47,0),(8,41,0)],
         'goals': [(1,53)], 'enemies': [(52,19)], 'steps': 48},
        # Level 5
        {'fruits': [(58,59,0),(44,53,0),(3,60,0),(14,54,0),
                    (14,28,1),(53,26,1),(6,25,1),(42,26,1)],
         'goals': [(28,11)], 'enemies': [(4,37),(46,37)], 'steps': 32},
        # Level 6
        {'fruits': [(33,32,5)],
         'goals': [(2,12),(52,53)], 'enemies': [(16,34)], 'steps': 32},
        # Level 7
        {'fruits': [(9,25,1),(20,35,1),(6,35,1),(30,37,1),(51,46,5)],
         'goals': [(19,13),(40,18)], 'enemies': [(12,51),(52,56)], 'steps': 32},
        # Level 8
        {'fruits': [(13,42,3),(3,40,3),(20,24,5)],
         'goals': [(52,15),(3,15),(52,51),(3,51)], 'enemies': [(43,31),(29,53),(47,48)], 'steps': 48},
        # Level 9
        {'fruits': [(18,46,1),(23,52,1),(35,48,5)],
         'goals': [(7,37),(49,51),(7,51)], 'enemies': [(51,13),(14,12),(15,22),(54,33)], 'steps': 48},
    ]

    data = levels[level_idx]
    fruits = data['fruits']
    goals = data['goals']
    steps = data['steps']

    # Generate candidate click positions
    # Key insight: clicks need to be near fruits (to pull them) or between fruits (to merge)
    # or near goals (to position fruits)
    positions = set()

    # Every 2-pixel grid point near fruits
    for fx, fy, fs in fruits:
        for dx in range(-10, 11, 2):
            for dy in range(-10, 11, 2):
                x, y = fx + dx, fy + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    positions.add((x, y))

    # Near goals
    for gx, gy in goals:
        for dx in range(-12, 13, 2):
            for dy in range(-12, 13, 2):
                x, y = gx + dx, gy + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    positions.add((x, y))

    # Midpoints between same-size fruit pairs
    for i in range(len(fruits)):
        for j in range(i+1, len(fruits)):
            if fruits[i][2] == fruits[j][2]:
                mx = (fruits[i][0] + fruits[j][0]) // 2
                my = (fruits[i][1] + fruits[j][1]) // 2
                for dx in range(-4, 5, 2):
                    for dy in range(-4, 5, 2):
                        x, y = mx + dx, my + dy
                        if 0 <= x <= 63 and 10 <= y <= 63:
                            positions.add((x, y))

    # Path from fruit clusters to goals
    for gx, gy in goals:
        for fx, fy, fs in fruits:
            # Add points along the line from fruit to goal
            dist = max(abs(gx-fx), abs(gy-fy))
            if dist > 0:
                for t in range(0, dist+1, max(1, dist//10)):
                    x = int(fx + (gx - fx) * t / dist)
                    y = int(fy + (gy - fy) * t / dist)
                    if 0 <= x <= 63 and 10 <= y <= 63:
                        positions.add((x, y))

    return sorted(positions), steps


def solve_level(prefix_clicks, level_idx, max_search_steps=None):
    """
    BFS to find a click sequence that solves the level.
    prefix_clicks: list of (x,y) tuples to play before this level
    """
    candidates, step_limit = get_candidate_clicks(level_idx)

    # For levels with many candidates, we need to be smarter
    # First try a coarser grid, then refine
    if len(candidates) > 500:
        # Start with coarser grid
        coarse = [(x, y) for x, y in candidates if x % 4 == 0 and y % 4 == 2]
        if len(coarse) > 200:
            coarse = [(x, y) for x, y in candidates if x % 8 == 0 and y % 8 == 2]
        candidates = coarse

    print(f"  Level {level_idx+1}: {len(candidates)} candidate positions, step_limit={step_limit}")

    target_lc = level_idx + 1
    if max_search_steps is None:
        max_search_steps = step_limit

    queue = deque([[]])
    visited = set()
    iterations = 0
    max_iter = 500000

    while queue and iterations < max_iter:
        path = queue.popleft()
        iterations += 1

        if iterations % 10000 == 0:
            print(f"    iter={iterations}, queue={len(queue)}, depth={len(path)}, visited={len(visited)}")

        if len(path) >= max_search_steps:
            continue

        # Replay
        game = create_game()
        game.reset()
        for x, y in prefix_clicks:
            game.step(GameAction.ACTION6, data={'x': x, 'y': y})

        alive = True
        for x, y in path:
            result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            if result.state == 'GAME_OVER':
                alive = False
                break
            if result.levels_completed >= target_lc:
                return path

        if not alive:
            continue

        state = get_state_hash(game)
        if state in visited:
            continue
        visited.add(state)

        # Check remaining steps
        g = game._game
        remaining = g.step_counter_ui.current_steps
        if remaining <= 0:
            continue

        for cx, cy in candidates:
            if len(path) + 1 <= max_search_steps:
                queue.append(path + [(cx, cy)])

    print(f"  BFS exhausted: {iterations} iter, {len(visited)} states")
    return None


def main():
    # Known L1 and L2 solutions
    l1_clicks = [(8,54),(13,49),(19,43),(25,37),(30,32),(36,26),(42,20),(47,15)]
    l2_clicks = [(39,38),(17,39),(15,56),(48,55),(32,38),(25,38),(23,38),(41,55),(35,55),(21,55),(27,55),(27,48),(25,43),(24,40),(28,35),(32,32),(33,28)]

    # Verify L1+L2
    game = create_game()
    game.reset()
    for x, y in l1_clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    print(f"After L1: lc={result.levels_completed}")
    for x, y in l2_clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    print(f"After L2: lc={result.levels_completed}")

    all_level_clicks = {0: l1_clicks, 1: l2_clicks}

    for level_idx in range(2, 9):
        prefix = []
        for i in range(level_idx):
            prefix.extend(all_level_clicks[i])

        print(f"\n{'='*50}")
        print(f"Solving Level {level_idx + 1}")
        print(f"{'='*50}")

        solution = solve_level(prefix, level_idx)

        if solution is not None:
            all_level_clicks[level_idx] = solution
            print(f"  Solution ({len(solution)} clicks):")
            for x, y in solution:
                print(f"    Click ({x},{y})")
        else:
            print(f"  FAILED to solve level {level_idx + 1}")
            break

    # Save results
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_level_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_level_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    output = {
        "game": "su15",
        "source": "analytical_bfs_v2",
        "type": "analytical",
        "total_actions": len(all_action_ids),
        "max_level": max(all_level_clicks.keys()) + 1 if all_level_clicks else 0,
        "all_actions": all_action_ids,
    }
    for i in sorted(level_action_ids.keys()):
        output[f"l{i+1}_actions"] = level_action_ids[i]

    outpath = "experiments/results/prescriptions/su15_fullchain.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
