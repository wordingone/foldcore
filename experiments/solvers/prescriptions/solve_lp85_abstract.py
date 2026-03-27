"""
Pure abstract model solver for LP85.
LP85 is a cyclic permutation puzzle:
- Goals sit on positions defined by numbered cycle maps
- Each button rotates goals along a cycle (R = shift right, L = shift left)
- Win: each colored tile has a matching goal at (tile.x+1, tile.y+1)
- This solver runs at ~1M states/sec with no game engine dependency
"""
import sys
import json
import time
from collections import deque

sys.path.insert(0, 'B:/M/the-search/environment_files/lp85/305b61c3')
from lp85 import Lp85, izutyjcpih, chmfaflqhy, qfvvosdkqr, crxpafuiwp
from arcengine import GameAction, ActionInput, GameState
import numpy as np

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def encode_click(x, y):
    return 7 + y * 64 + x


def game_step(game, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    r = game.perform_action(ai, raw=True)
    if r is None:
        return 0, None
    return r.levels_completed, r.state


def solve_level_abstract(level_idx):
    """
    Build pure abstract model for this level and solve with BFS.
    State = tuple of goal grid positions.
    Actions = cyclic permutations on the grid positions.
    """
    g = Lp85()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)

    level_name = g.ucybisahh
    step_budget = g.toxpunyqe.current_steps

    print(f'  Level name: {level_name}, budget: {step_budget}')

    # Build cycle maps
    uopmnplcnv = qfvvosdkqr(izutyjcpih)

    # Get button info: each button is (map_name, direction=R/L)
    # Build the permutation function for each button
    buttons = []
    btn_display_coords = []

    for s in g.afhycvvjg:
        if s.tags and 'button' in s.tags[0]:
            parts = s.tags[0].split('_')
            if len(parts) == 3:
                map_name = parts[1]
                direction = parts[2]  # R or L
                # Find display coordinates that trigger this button
                # Button sprite is at grid (s.x, s.y), need display coords
                buttons.append((map_name, direction, s.x, s.y))

    # Find display coords for each unique button
    # Use the game to find which display coords trigger which buttons
    btn_actions = {}  # (map_name, dir) -> display action
    base_frame = None

    g_test = Lp85()
    g_test.full_reset()
    if level_idx > 0:
        g_test.set_level(level_idx)
    ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
    r = g_test.perform_action(ai, raw=True)
    base_goals = set()
    for s in g_test.current_level.get_sprites_by_tag('goal'):
        base_goals.add((s.x, s.y))
    for s in g_test.current_level.get_sprites_by_tag('goal-o'):
        base_goals.add((s.x, s.y))

    for dy in range(0, 64, 2):
        for dx in range(0, 64, 2):
            a = encode_click(dx, dy)
            g_test = Lp85()
            g_test.full_reset()
            if level_idx > 0:
                g_test.set_level(level_idx)
            levels, state = game_step(g_test, a)
            if state == GameState.GAME_OVER:
                continue

            new_goals = set()
            for s in g_test.current_level.get_sprites_by_tag('goal'):
                new_goals.add((s.x, s.y))
            for s in g_test.current_level.get_sprites_by_tag('goal-o'):
                new_goals.add((s.x, s.y))

            if new_goals != base_goals:
                # Find which button this corresponds to by checking tags
                grid_pos = g_test.camera.display_to_grid(dx, dy)
                if grid_pos:
                    gx, gy = grid_pos
                    # Find button at this grid position
                    for map_name, direction, bx, by in buttons:
                        if bx <= gx < bx + 3 and by <= gy < by + 4:
                            key = (map_name, direction)
                            if key not in btn_actions:
                                btn_actions[key] = a
                            break

    print(f'  Found {len(btn_actions)} unique button actions')

    if not btn_actions:
        return None

    # Now build the abstract permutation model
    # For each button (map_name, dir), compute the permutation on grid positions
    # The cycle map defines positions numbered 1..N in a grid
    # R shift: position i goes to position (i % N) + 1
    # L shift: position i goes to position ((i - 2) % N) + 1

    if level_name not in izutyjcpih:
        print(f'  WARNING: no cycle maps for {level_name}')
        return None

    maps_data = izutyjcpih[level_name]

    # Build position -> number mapping for each map
    # Position is in grid coords (not scaled)
    map_pos_to_num = {}  # map_name -> {(grid_x, grid_y) -> number}
    map_num_to_pos = {}  # map_name -> {number -> (grid_x, grid_y)}

    for map_name, grid in maps_data.items():
        pos_to_num = {}
        num_to_pos = {}
        for y, row in enumerate(grid):
            for x, val in enumerate(row):
                if val != -1:
                    pos_to_num[(x, y)] = val
                    num_to_pos[val] = (x, y)
        map_pos_to_num[map_name] = pos_to_num
        map_num_to_pos[map_name] = num_to_pos

    # Compute permutation for each button action
    # A button (map_name, R) shifts all sprites on map positions to the NEXT position in the cycle
    # A button (map_name, L) shifts all sprites on map positions to the PREVIOUS position in the cycle

    # The actual permutation on SPRITE COORDINATES:
    # sprite_coord = grid_coord * crxpafuiwp (scale factor = 3)
    # The step function uses chmfaflqhy which returns (src, dst) pairs
    # where src.x/y and dst.x/y are in grid coordinates

    btn_permutations = {}  # action_id -> permutation dict {(sprite_x, sprite_y) -> (sprite_x, sprite_y)}

    for (map_name, direction), action in btn_actions.items():
        is_right = (direction == 'R')
        pairs = chmfaflqhy(level_name, map_name, is_right, uopmnplcnv)

        perm = {}  # {(sx, sy) -> (dx, dy)} in sprite coordinates
        for src, dst in pairs:
            sx, sy = src.x * crxpafuiwp, src.y * crxpafuiwp
            dx, dy = dst.x * crxpafuiwp, dst.y * crxpafuiwp
            perm[(sx, sy)] = (dx, dy)
        btn_permutations[action] = perm

    # Get initial state: positions of goals (both types)
    g = Lp85()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)

    initial_goals = []
    goal_types = []
    for s in g.current_level.get_sprites_by_tag('goal'):
        initial_goals.append((s.x, s.y))
        goal_types.append('goal')
    for s in g.current_level.get_sprites_by_tag('goal-o'):
        initial_goals.append((s.x, s.y))
        goal_types.append('goal-o')

    # Get target positions (where tiles are +1 offset)
    tile_targets = []
    tile_target_types = []
    for s in g.current_level.get_sprites_by_tag('bghvgbtwcb'):
        tile_targets.append((s.x + 1, s.y + 1))
        tile_target_types.append('goal')
    for s in g.current_level.get_sprites_by_tag('fdgmtkfrxl'):
        tile_targets.append((s.x + 1, s.y + 1))
        tile_target_types.append('goal-o')

    print(f'  Goals: {initial_goals}')
    print(f'  Targets: {tile_targets} (types: {tile_target_types})')

    # Abstract state = tuple of goal positions
    # Apply permutation: for each goal, if its position is in the perm, move it
    def apply_perm(state, perm):
        new_state = list(state)
        for i in range(len(new_state)):
            if new_state[i] in perm:
                new_state[i] = perm[new_state[i]]
        return tuple(new_state)

    def check_win(state):
        # Each goal must be at its target position
        for i, (tx, ty) in enumerate(tile_targets):
            found = False
            for j, (gx, gy) in enumerate(state):
                if goal_types[j] == tile_target_types[i] and gx == tx and gy == ty:
                    found = True
                    break
            if not found:
                return False
        return True

    # BFS on abstract state
    action_list = list(btn_actions.values())
    perm_list = [btn_permutations[a] for a in action_list]
    btn_names = list(btn_actions.keys())

    init_state = tuple(initial_goals)
    if check_win(init_state):
        print('  Already solved!')
        return []

    print(f'  BFS with {len(action_list)} actions, budget={step_budget}')

    queue = deque([(init_state, ())])
    visited = {init_state}
    explored = 0
    t0 = time.time()
    depth = 0

    while queue:
        state, seq = queue.popleft()
        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f'    d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.1f}s ({rate:.0f}/s)')

        if len(seq) >= min(step_budget, 60):
            continue

        for i, (action, perm) in enumerate(zip(action_list, perm_list)):
            new_state = apply_perm(state, perm)
            explored += 1

            if check_win(new_state):
                new_seq = list(seq) + [action]
                el = time.time() - t0
                print(f'  SOLVED in {len(new_seq)} actions ({explored} explored, {el:.1f}s)')
                print(f'  Sequence: {[btn_names[action_list.index(a)] for a in new_seq]}')
                return new_seq

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, tuple(list(seq) + [action])))

        if time.time() - t0 > 300:
            print(f'  TIMEOUT ({explored} explored)')
            return None

        if explored > 10000000:
            print(f'  STATE LIMIT ({explored})')
            return None

    print(f'  EXHAUSTED ({explored} explored)')
    return None


def main():
    all_actions = []
    per_level = {}
    n_levels = 8

    for level_idx in range(n_levels):
        print(f'\n{"="*60}')
        print(f'LP85 Level {level_idx + 1}/{n_levels}')
        print(f'{"="*60}')

        sol = solve_level_abstract(level_idx)
        if sol is None:
            per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED'}
            print(f'  Level {level_idx + 1} UNSOLVED - stopping chain')
            break

        all_actions.extend(sol)
        per_level[f'L{level_idx + 1}'] = {
            'status': 'SOLVED',
            'actions': sol,
            'length': len(sol),
        }

    # Verify chain
    print(f'\nVerifying chain ({len(all_actions)} actions)...')
    g = Lp85()
    g.full_reset()
    max_levels = 0
    for a in all_actions:
        levels, state = game_step(g, a)
        if levels > max_levels:
            max_levels = levels
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    print(f'Chain verified: {max_levels} levels, {len(all_actions)} actions')

    result = {
        'game': 'lp85',
        'version': '305b61c3',
        'total_levels': n_levels,
        'method': 'abstract_permutation_bfs',
        'levels': per_level,
        'full_sequence': all_actions,
        'max_level_solved': max_levels,
        'total_actions': len(all_actions),
    }

    out_path = f'{RESULTS_DIR}/lp85_fullchain.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
