"""
SU15 solver: BFS using game engine as simulator.
For each level, try sequences of click actions and find working solutions.
Uses the actual game engine to handle vacuum/merge/enemy mechanics correctly.
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
        game_id='su15',
        local_dir='environment_files/su15/4c352900',
        class_name='Su15'
    )
    logger = logging.getLogger('su15')
    logger.setLevel(logging.WARNING)
    return LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)


def get_su15_state(game):
    """Extract relevant state from SU15 game for hashing."""
    g = game._game
    state_parts = []

    # Fruit positions and sizes
    for sprite in g.hmeulfxgy:
        size = g.amnmgwpkeb.get(sprite, 0)
        state_parts.append(('f', size, sprite.x, sprite.y))

    # Enemy positions
    for sprite in g.peiiyyzum:
        etype = g.hirdajbmj.get(sprite, '')
        state_parts.append(('e', etype, sprite.x, sprite.y))

    # Sort for canonical form
    state_parts.sort()
    return tuple(state_parts)


def check_level_complete(game, target_level):
    """Check if we've advanced past target level."""
    g = game._game
    return g.level_index > target_level


def solve_level_bfs(prefix_actions, level_idx, step_limit=None, candidate_actions=None):
    """
    BFS over click action sequences using game replay.
    """
    target_lc = level_idx + 1

    if step_limit is None:
        # Get step limit from level data
        step_limits = [32, 32, 48, 48, 32, 32, 32, 48, 48]
        step_limit = step_limits[level_idx]

    # Determine candidate actions - filter to only those near fruits/goals
    if candidate_actions is None:
        candidate_actions = get_candidate_actions(level_idx)

    print(f"  Candidate actions: {len(candidate_actions)} positions")
    print(f"  Step limit: {step_limit}")

    queue = deque([[]])
    visited_states = set()
    iterations = 0
    max_iterations = 200000

    while queue and iterations < max_iterations:
        path = queue.popleft()
        iterations += 1

        if iterations % 5000 == 0:
            print(f"    iter={iterations}, queue={len(queue)}, depth={len(path)}, visited={len(visited_states)}")

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

        # State hash
        state = get_su15_state(game)
        if state in visited_states:
            continue
        visited_states.add(state)

        # Check if too many steps used (game-internal step counter)
        g = game._game
        if g.step_counter_ui.current_steps <= 0:
            continue

        # Expand with candidate actions only
        remaining = g.step_counter_ui.current_steps
        for action in candidate_actions:
            if len(path) + 1 <= step_limit and remaining > 0:
                queue.append(path + [action])

    print(f"  BFS exhausted after {iterations} iterations, {len(visited_states)} states")
    return None


def get_candidate_actions(level_idx):
    """
    Get candidate click positions for each level based on fruit/goal positions.
    Action index = i * 14 + j where x = i*4, y = 10 + j*4
    """
    # For SU15, ACTION6 takes x,y data. The action_space has 224 actions (16x14 grid)

    # Level-specific fruit and goal positions
    level_data = [
        # Level 1
        {'fruits': [(3, 58, 2)], 'goals': [(44, 11)]},
        # Level 2
        {'fruits': [(41,37,0),(18,37,0),(37,40,0),(16,41,0),(14,55,0),(16,57,0),(49,54,0),(47,56,0)],
         'goals': [(29, 23)]},
        # Level 3
        {'fruits': [(55,23,0),(61,23,0),(31,22,0),(31,15,0),(12,23,0),(8,28,0),
                    (46,22,1),(30,32,1),(18,16,1)],
         'goals': [(5, 46), (19, 46)]},
        # Level 4
        {'fruits': [(5,26,0),(11,26,0),(31,27,0),(36,29,0),(33,47,0),(30,51,0),(12,47,0),(8,41,0)],
         'goals': [(1, 53)],
         'enemies': [(52, 19)]},
        # Level 5
        {'fruits': [(58,59,0),(44,53,0),(3,60,0),(14,54,0),
                    (14,28,1),(53,26,1),(6,25,1),(42,26,1)],
         'goals': [(28, 11)],
         'enemies': [(4, 37), (46, 37)]},
        # Level 6
        {'fruits': [(33, 32, 5)],
         'goals': [(2, 12), (52, 53)],
         'enemies': [(16, 34)]},
        # Level 7
        {'fruits': [(9,25,1),(20,35,1),(6,35,1),(30,37,1),(51,46,5)],
         'goals': [(19, 13), (40, 18)],
         'enemies': [(12, 51), (52, 56)]},
        # Level 8
        {'fruits': [(13,42,3),(3,40,3),(20,24,5)],
         'goals': [(52,15),(3,15),(52,51),(3,51)],
         'enemies': [(43,31),(29,53),(47,48)]},
        # Level 9
        {'fruits': [(18,46,1),(23,52,1),(35,48,5)],
         'goals': [(7,37),(49,51),(7,51)],
         'enemies': [(51,13),(14,12),(15,22),(54,33)]},
    ]

    data = level_data[level_idx]
    fruits = data['fruits']
    goals = data['goals']

    # Collect all interesting x,y positions: near fruits, between fruit pairs, near goals
    positions = set()

    # Near each fruit
    for fx, fy, fs in fruits:
        for dx in range(-8, 9, 4):
            for dy in range(-8, 9, 4):
                x, y = fx + dx, fy + dy
                if 0 <= x <= 60 and 10 <= y <= 62:
                    positions.add((x, y))

    # Near each goal (for final placement)
    for gx, gy in goals:
        for dx in range(-12, 13, 4):
            for dy in range(-12, 13, 4):
                x, y = gx + dx, gy + dy
                if 0 <= x <= 60 and 10 <= y <= 62:
                    positions.add((x, y))

    # Between fruit pairs (for merging)
    for i in range(len(fruits)):
        for j in range(i+1, len(fruits)):
            if fruits[i][2] == fruits[j][2]:  # Same size
                mx = (fruits[i][0] + fruits[j][0]) // 2
                my = (fruits[i][1] + fruits[j][1]) // 2
                for dx in range(-4, 5, 4):
                    for dy in range(-4, 5, 4):
                        x, y = mx + dx, my + dy
                        if 0 <= x <= 60 and 10 <= y <= 62:
                            positions.add((x, y))

    # Convert to action indices
    actions = set()
    for x, y in positions:
        i = round(x / 4)
        j = round((y - 10) / 4)
        i = max(0, min(15, i))
        j = max(0, min(13, j))
        actions.add(i * 14 + j)

    return sorted(actions)


def main():
    # Known L1 and L2 solutions (action indices into the 224-element action space)
    l1_actions = [3471, 3156, 2778, 2400, 2085, 1707, 1329, 1014]
    l2_actions = [2478, 2520, 3606, 3575, 2471, 2464, 2462, 3568, 3562, 3548, 3554, 3106, 2784, 2591, 2275, 2087, 1832]

    # Wait - these action indices are much larger than 224. The SU15 action space must be different.
    # Let me check what action space the game actually provides.

    game = create_game()
    obs = game.reset()
    print(f"Action space: {game.action_space}")
    print(f"Action space length: {len(game.action_space)}")

    # Check what the action IDs actually are
    # From the source: actions are ActionInput(id=GameAction.ACTION6.value, data={"x": i*4, "y": y})
    # The game.step() takes an action id - but what format?
    # Let me check the available_actions
    print(f"Available actions from obs: {obs.available_actions}")

    # Check the game's internal action list
    g = game._game
    print(f"Internal actions count: {len(g.actions)}")
    if g.actions:
        print(f"First action: id={g.actions[0].id}, data={g.actions[0].data}")
        print(f"Last action: id={g.actions[-1].id}, data={g.actions[-1].data}")

    # The actions in the existing prescription have IDs > 1000
    # These might be the internal action IDs mapping to ACTION6 with specific x,y data
    # Let me check how step() works

if __name__ == "__main__":
    main()
