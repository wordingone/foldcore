"""
Fullchain solver for m0r0, ka59, wa30.
Solves ALL levels of each game by:
1. Extracting level data from game source files
2. Running BFS on abstract game state
3. Verifying solutions via arc_agi API
4. Saving results to prescriptions/{game}_fullchain.json

Usage: PYTHONUTF8=1 python experiments/util_fullchain_solver.py [game_id]
"""
import os
import sys
import json
import time
import numpy as np
from collections import deque

import arc_agi
from arcengine import GameAction, GameState

STEP_WA30 = 4  # wa30 movement step size
STEP_KA59 = 3  # ka59 movement step size


# ============================================================================
# wa30 solver - Grab-carry Sokoban with facing direction
# ============================================================================

def extract_wa30_level(level):
    """Extract abstract game state from a wa30 level."""
    sps = level.get_sprites()
    walls = set()
    blocks = []
    targets = set()
    targets2 = set()
    forbidden = set()
    player = None
    ai_movers = []
    destroyable = []

    S = STEP_WA30
    for sp in sps:
        tags = list(sp.tags) if hasattr(sp, 'tags') else []
        pos = (sp.x, sp.y)

        if 'wbmdvjhthc' in tags:
            player = pos
        elif 'geezpjgiyd' in tags:
            blocks.append(pos)
        elif 'fsjjayjoeg' in tags:
            for dy in range(sp.height):
                for dx in range(sp.width):
                    targets.add((sp.x + dx, sp.y + dy))
        elif 'kdweefinfi' in tags:
            ai_movers.append(pos)
        elif 'ysysltqlke' in tags:
            destroyable.append(pos)
        elif 'bnzklblgdk' in tags:
            forbidden.add(pos)
        elif 'zqxwgacnue' in tags:
            for dy in range(sp.height):
                for dx in range(sp.width):
                    targets2.add((sp.x + dx, sp.y + dy))
        elif sp.is_collidable:
            for dy in range(0, sp.height, S):
                for dx in range(0, sp.width, S):
                    walls.add((sp.x + dx, sp.y + dy))

    # Boundary walls
    for i in range(0, 64, S):
        walls.add((-S, i))
        walls.add((64, i))
        walls.add((i, -S))
        walls.add((i, 64))

    budget = level.get_data('StepCounter')
    return {
        'player': player,
        'blocks': tuple(sorted(blocks)),
        'targets': targets,
        'targets2': targets2,
        'walls': walls,
        'forbidden': forbidden,
        'ai_movers': ai_movers,
        'destroyable': destroyable,
        'budget': budget,
    }


def wa30_bfs(level_data, max_states=3000000):
    """
    BFS solver for wa30 with proper facing-direction grab.

    State: (player_pos, facing, blocks_tuple, carrying_block_idx_or_-1)
    - facing: 0=up, 1=down, 2=left, 3=right (set by last move direction)
    - carrying: index into sorted blocks tuple, or -1

    Actions: UP(0), DOWN(1), LEFT(2), RIGHT(3), GRAB(4)
    """
    S = STEP_WA30
    walls = level_data['walls']
    targets = level_data['targets']
    forbidden = level_data['forbidden']
    initial_blocks = level_data['blocks']
    player_start = level_data['player']
    budget = level_data['budget']

    # Direction -> (dx, dy, facing)
    DIRS = [
        (0, -S, 0),   # UP -> facing 0
        (0, S, 1),    # DOWN -> facing 1 (rotation 180)
        (-S, 0, 2),   # LEFT -> facing 2 (rotation 270)
        (S, 0, 3),    # RIGHT -> facing 3 (rotation 90)
    ]

    # Facing -> grab offset (where grabbed block must be relative to player)
    GRAB_OFFSETS = {
        0: (0, -S),   # facing up: block above
        1: (0, S),    # facing down: block below
        2: (-S, 0),   # facing left: block to left
        3: (S, 0),    # facing right: block to right
    }

    def is_free(pos, blocks_set):
        """Position is walkable (no wall, no forbidden, no block, in bounds)."""
        return (pos not in walls and pos not in forbidden and
                pos not in blocks_set and
                0 <= pos[0] < 64 and 0 <= pos[1] < 64)

    def check_win(blocks_tuple):
        """All blocks on target positions and not carried."""
        return all((bx, by) in targets for bx, by in blocks_tuple)

    # State: (player, facing, blocks, carrying)
    # Start facing down (default rotation=0 means facing up actually)
    # The player sprite starts with default rotation, let's assume rotation 0 (facing up)
    # Actually rotation is set by pjedoipwee on move, initial is whatever. Let's use 0.
    initial_state = (player_start, 0, initial_blocks, -1)

    visited = set()
    visited.add(initial_state)
    queue = deque()
    queue.append((initial_state, []))

    states_explored = 0
    max_depth = 0

    while queue:
        state, actions = queue.popleft()
        player, facing, blocks, carrying = state
        cur_depth = len(actions)

        if cur_depth >= budget:
            continue

        if cur_depth > max_depth:
            max_depth = cur_depth
            if max_depth % 5 == 0:
                print(f"    depth={max_depth}, queue={len(queue)}, visited={len(visited)}")

        states_explored += 1
        if states_explored > max_states:
            print(f"    BFS exhausted at {states_explored} states, depth={max_depth}")
            return None

        blocks_set = set(blocks)

        for action_id in range(5):
            if action_id < 4:
                # Movement
                dx, dy, new_facing = DIRS[action_id]
                new_player = (player[0] + dx, player[1] + dy)

                if carrying >= 0:
                    # Moving with carried block
                    block_pos = blocks[carrying]
                    offset = (block_pos[0] - player[0], block_pos[1] - player[1])
                    new_block = (new_player[0] + offset[0], new_player[1] + offset[1])

                    other_blocks = blocks_set - {block_pos}

                    # Both player and block positions must be valid
                    player_ok = (new_player not in walls and new_player not in forbidden and
                                 new_player not in other_blocks and
                                 0 <= new_player[0] < 64 and 0 <= new_player[1] < 64 and
                                 new_player != new_block)
                    block_ok = (new_block not in walls and new_block not in forbidden and
                                new_block not in other_blocks and
                                0 <= new_block[0] < 64 and 0 <= new_block[1] < 64)

                    if player_ok and block_ok:
                        new_blocks = list(blocks)
                        new_blocks[carrying] = new_block
                        new_blocks_sorted = tuple(sorted(new_blocks))

                        # Find new carrying index
                        new_carry = -1
                        for ci, bp in enumerate(new_blocks_sorted):
                            if bp == new_block:
                                new_carry = ci
                                break

                        # Note: when carrying, don't change facing (rotation not set)
                        # Actually looking at source, qnmfimgpwc is called which does set rotation
                        # when NOT carrying. When carrying, wqwsvmhhzj is called directly.
                        # Let me check: the step function calls yygfcvqoyx which calls
                        # qnmfimgpwc(player, dx, dy). qnmfimgpwc sets rotation if not carrying.
                        # When carrying, it still calls wqwsvmhhzj but doesn't set rotation.
                        # Actually: qnmfimgpwc does: "if player not in nsevyuople: set rotation"
                        # So when carrying, rotation is NOT updated.
                        new_state = (new_player, facing, new_blocks_sorted, new_carry)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((new_state, actions + [action_id]))
                    # If invalid, we could still update facing but player/block don't move
                    # In the game, a blocked move still consumes a step and updates facing
                    # Actually qnmfimgpwc: "if player not in nsevyuople: set rotation"
                    # followed by wqwsvmhhzj which checks validity. If invalid, nothing happens.
                    # But rotation IS set before the move check (when not carrying).
                    # When carrying, rotation is NOT changed even on invalid moves.

                else:
                    # Not carrying - just move
                    if is_free(new_player, blocks_set):
                        new_state = (new_player, new_facing, blocks, -1)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((new_state, actions + [action_id]))
                    else:
                        # Move blocked but facing changes
                        new_state = (player, new_facing, blocks, -1)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((new_state, actions + [action_id]))

            else:
                # GRAB/RELEASE (action 4)
                if carrying >= 0:
                    # Release
                    # Check win after release
                    new_state = (player, facing, blocks, -1)
                    if new_state not in visited:
                        visited.add(new_state)
                        new_actions = actions + [4]
                        if check_win(blocks):
                            print(f"    SOLVED! depth={len(new_actions)}, states={states_explored}")
                            return new_actions
                        queue.append((new_state, new_actions))
                else:
                    # Grab - only the block in front (based on facing)
                    gx, gy = GRAB_OFFSETS[facing]
                    target_pos = (player[0] + gx, player[1] + gy)

                    for bi, bp in enumerate(blocks):
                        if bp == target_pos:
                            new_state = (player, facing, blocks, bi)
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append((new_state, actions + [4]))
                            break

    print(f"    BFS complete: {states_explored} states, depth={max_depth}, no solution found")
    return None


def solve_wa30_all():
    """Solve all 9 levels of wa30."""
    sys.path.insert(0, 'environment_files/wa30/ee6fef47')
    from wa30 import levels as WA30_LEVELS

    all_actions = []
    per_level = {}
    total_t0 = time.time()

    for level_idx in range(9):
        print(f"\n{'='*60}")
        print(f"wa30 Level {level_idx+1}/9")
        print(f"{'='*60}")

        level_data = extract_wa30_level(WA30_LEVELS[level_idx])
        print(f"  Player: {level_data['player']}")
        print(f"  Blocks ({len(level_data['blocks'])}): {level_data['blocks']}")
        print(f"  Targets: {len(level_data['targets'])} cells")
        print(f"  AI movers: {level_data['ai_movers']}")
        print(f"  Destroyable: {level_data['destroyable']}")
        print(f"  Budget: {level_data['budget']}")

        if level_data['ai_movers'] or level_data['destroyable']:
            print(f"  NOTE: Level has AI movers/destroyable - solving abstractly first")

        t0 = time.time()
        solution = wa30_bfs(level_data)
        elapsed = time.time() - t0

        if solution is None:
            print(f"  FAILED ({elapsed:.1f}s)")
            break

        per_level[f"L{level_idx+1}"] = {
            "actions": solution,
            "count": len(solution),
            "time": round(elapsed, 2)
        }
        all_actions.extend(solution)
        print(f"  Solved: {len(solution)} actions in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    max_level = verify_solution("wa30", all_actions)

    return {
        "game": "wa30",
        "source": "analytical_bfs_solver",
        "type": "analytical",
        "total_actions": len(all_actions),
        "max_level": max_level,
        "per_level": per_level,
        "all_actions": all_actions,
        "total_time": round(total_elapsed, 2),
        "action_map": {"0": "UP", "1": "DOWN", "2": "LEFT", "3": "RIGHT", "4": "INTERACT"}
    }


# ============================================================================
# ka59 solver - Sokoban push puzzle
# ============================================================================

def extract_ka59_level(level):
    """Extract abstract game state from a ka59 level."""
    sps = level.get_sprites()
    S = STEP_KA59
    walls = set()
    blocks = []
    goals = set()
    block_goals = set()
    players = []
    explodables = []

    for sp in sps:
        tags = list(sp.tags) if hasattr(sp, 'tags') else []
        pos = (sp.x, sp.y)
        size = (sp.width, sp.height)

        if 'xlfuqjygey' in tags:
            players.append({'pos': pos, 'size': size})
        elif 'nnckfubbhi' in tags:
            blocks.append({'pos': pos, 'size': size})
        elif 'gobzaprasa' in tags:
            explodables.append({'pos': pos, 'size': size})
        elif 'rktpmjcpkt' in tags:
            # Goal for players - player fits inside goal (goal size > player size)
            # Goal is 5x5, player is 3x3, offset +1,+1
            goals.add((sp.x + 1, sp.y + 1))
        elif 'ucjzrlvfkb' in tags:
            block_goals.add((sp.x + 1, sp.y + 1))
        elif 'vwjqkxkyxm' in tags or 'divgcilurm' in tags:
            # Wall sprites - pixel -1 = transparent, others = opaque (wall)
            for dy in range(sp.height):
                for dx in range(sp.width):
                    pix_val = sp.pixels[dy, dx] if hasattr(sp.pixels, '__getitem__') else 0
                    if pix_val != -1:
                        walls.add((sp.x + dx, sp.y + dy))

    budget = level.get_data('StepCounter')
    return {
        'players': [p['pos'] for p in players],
        'blocks': tuple(sorted([b['pos'] for b in blocks])),
        'goals': goals,
        'block_goals': block_goals,
        'walls': walls,
        'explodables': [e['pos'] for e in explodables],
        'budget': budget,
        'player_sizes': [p['size'] for p in players],
        'block_sizes': [b['size'] for b in blocks],
    }


def ka59_bfs(level_data, grid_size, max_states=3000000):
    """
    BFS solver for ka59 (push Sokoban).
    Player pushes blocks by moving into them.
    Win: all goals have a player, all block_goals have a block.
    Movement step = 3.
    Can switch active player by clicking on another player sprite.
    """
    S = STEP_KA59
    walls = level_data['walls']
    goals = level_data['goals']
    block_goals = level_data['block_goals']
    initial_players = tuple(sorted(level_data['players']))
    initial_blocks = level_data['blocks']
    budget = level_data['budget']
    gs = grid_size

    DIRS = [(0, -S), (0, S), (-S, 0), (S, 0)]  # UP, DOWN, LEFT, RIGHT

    player_w, player_h = 3, 3

    def collides_wall(pos, w=3, h=3):
        for dy in range(h):
            for dx in range(w):
                if (pos[0] + dx, pos[1] + dy) in walls:
                    return True
        return False

    def sprites_overlap(p1, s1, p2, s2):
        return (p1[0] < p2[0] + s2[0] and p1[0] + s1[0] > p2[0] and
                p1[1] < p2[1] + s2[1] and p1[1] + s1[1] > p2[1])

    def check_win(players_t, blocks_t):
        for g in goals:
            if g not in players_t:
                return False
        for bg in block_goals:
            if bg not in blocks_t:
                return False
        return True

    def game_to_pixel(gx, gy, player_w=3, player_h=3):
        """Convert game coords to pixel coords for clicking center of player."""
        scale = min(64 // gs[0], 64 // gs[1])
        x_off = (64 - gs[0] * scale) // 2
        y_off = (64 - gs[1] * scale) // 2
        # Center of player sprite in pixel coords
        px = int((gx + player_w / 2) * scale + x_off)
        py = int((gy + player_h / 2) * scale + y_off)
        return px, py

    n_players = len(initial_players)

    # State: (active_player_idx, players, blocks)
    initial_state = (0, initial_players, initial_blocks)
    visited = {initial_state}
    queue = deque([(initial_state, [])])
    states_explored = 0
    max_depth = 0

    while queue:
        state, actions = queue.popleft()
        active_idx, players, blocks = state
        cur_depth = len(actions)

        if cur_depth >= budget:
            continue
        if cur_depth > max_depth:
            max_depth = cur_depth
            if max_depth % 5 == 0:
                print(f"    depth={max_depth}, queue={len(queue)}, visited={len(visited)}")

        states_explored += 1
        if states_explored > max_states:
            print(f"    BFS exhausted at {states_explored} states")
            return None

        active_player = players[active_idx]

        # Action: switch active player (click on another player)
        for pi in range(n_players):
            if pi != active_idx:
                new_state = (pi, players, blocks)
                if new_state not in visited:
                    visited.add(new_state)
                    # Click action: encode as click at player center
                    px, py = game_to_pixel(players[pi][0], players[pi][1])
                    click_action = 7 + py * 64 + px
                    queue.append((new_state, actions + [click_action]))

        # Directional actions
        for action_id in range(4):
            dx, dy = DIRS[action_id]
            new_pos = (active_player[0] + dx, active_player[1] + dy)

            if collides_wall(new_pos, player_w, player_h):
                continue

            hit_player = False
            for pi, pp in enumerate(players):
                if pi != active_idx:
                    if sprites_overlap(new_pos, (player_w, player_h), pp, (player_w, player_h)):
                        hit_player = True
                        break
            if hit_player:
                continue

            pushed_idx = -1
            for bi, bp in enumerate(blocks):
                if sprites_overlap(new_pos, (player_w, player_h), bp, (3, 3)):
                    pushed_idx = bi
                    break

            if pushed_idx >= 0:
                bp = blocks[pushed_idx]
                new_bp = (bp[0] + dx, bp[1] + dy)

                if collides_wall(new_bp, 3, 3):
                    continue

                block_hit = False
                for bi2, bp2 in enumerate(blocks):
                    if bi2 != pushed_idx:
                        if sprites_overlap(new_bp, (3, 3), bp2, (3, 3)):
                            block_hit = True
                            break
                if block_hit:
                    continue

                for pi, pp in enumerate(players):
                    if pi != active_idx:
                        if sprites_overlap(new_bp, (3, 3), pp, (player_w, player_h)):
                            block_hit = True
                            break
                if block_hit:
                    continue

                new_blocks = list(blocks)
                new_blocks[pushed_idx] = new_bp
                new_blocks_t = tuple(sorted(new_blocks))

                new_players = list(players)
                new_players[active_idx] = new_pos
                new_players_t = tuple(sorted(new_players))

                new_active = new_players_t.index(new_pos)
                new_state = (new_active, new_players_t, new_blocks_t)

                if new_state not in visited:
                    visited.add(new_state)
                    new_actions = actions + [action_id]
                    if check_win(new_players_t, new_blocks_t):
                        print(f"    SOLVED! depth={len(new_actions)}, states={states_explored}")
                        return new_actions
                    queue.append((new_state, new_actions))
            else:
                new_players = list(players)
                new_players[active_idx] = new_pos
                new_players_t = tuple(sorted(new_players))

                new_active = new_players_t.index(new_pos)
                new_state = (new_active, new_players_t, blocks)

                if new_state not in visited:
                    visited.add(new_state)
                    new_actions = actions + [action_id]
                    if check_win(new_players_t, blocks):
                        print(f"    SOLVED! depth={len(new_actions)}, states={states_explored}")
                        return new_actions
                    queue.append((new_state, new_actions))

    print(f"    BFS complete: {states_explored} states, depth={max_depth}")
    return None


def solve_ka59_all():
    """Solve all 7 levels of ka59."""
    sys.path.insert(0, 'environment_files/ka59/9f096b4a')
    from ka59 import levels as KA59_LEVELS

    all_actions = []
    per_level = {}
    total_t0 = time.time()

    for level_idx in range(7):
        print(f"\n{'='*60}")
        print(f"ka59 Level {level_idx+1}/7")
        print(f"{'='*60}")

        level_data = extract_ka59_level(KA59_LEVELS[level_idx])
        print(f"  Players: {level_data['players']}")
        print(f"  Blocks: {level_data['blocks']}")
        print(f"  Goals: {sorted(level_data['goals'])}")
        print(f"  Block goals: {sorted(level_data['block_goals'])}")
        print(f"  Budget: {level_data['budget']}")

        t0 = time.time()
        grid_size = KA59_LEVELS[level_idx].grid_size or (63, 63)
        solution = ka59_bfs(level_data, grid_size)
        elapsed = time.time() - t0

        if solution is None:
            print(f"  FAILED ({elapsed:.1f}s)")
            break

        per_level[f"L{level_idx+1}"] = {
            "actions": solution,
            "count": len(solution),
            "time": round(elapsed, 2)
        }
        all_actions.extend(solution)
        print(f"  Solved: {len(solution)} actions in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    max_level = verify_solution("ka59", all_actions)

    return {
        "game": "ka59",
        "source": "analytical_bfs_solver",
        "type": "analytical",
        "total_actions": len(all_actions),
        "max_level": max_level,
        "per_level": per_level,
        "all_actions": all_actions,
        "total_time": round(total_elapsed, 2),
        "action_map": {"0": "UP", "1": "DOWN", "2": "LEFT", "3": "RIGHT"}
    }


# ============================================================================
# m0r0 solver - Mirror twin blocks
# ============================================================================

def extract_m0r0_level(level):
    """Extract abstract state for m0r0 level."""
    sps = level.get_sprites()
    grid_w, grid_h = level.grid_size or (64, 64)

    blocks = []
    toggles = []
    walls = set()
    jggua_walls = set()
    colored_walls = {}
    colored_triggers = {}
    npwxa = level.get_data('npwxa')

    for sp in sps:
        name = sp.name
        tags = list(sp.tags) if hasattr(sp, 'tags') else []

        if name.startswith('qzfkx-ubwff') or name.startswith('qzfkx-kncqr'):
            mirror_x = 'crkfz' in name
            mirror_y = 'kncqr' in name
            blocks.append({
                'name': name, 'pos': (sp.x, sp.y),
                'mirror_x': mirror_x, 'mirror_y': mirror_y
            })
        elif name == 'cvcer':
            toggles.append((sp.x, sp.y))
        elif name == 'wyiex':
            walls.add((sp.x, sp.y))
        elif 'jggua' in tags:
            # In arcengine, pixel value -1 = transparent (passable), 0+ = opaque (wall)
            for dy in range(sp.height):
                for dx in range(sp.width):
                    if sp.pixels[dy, dx] != -1:  # Non-transparent = wall
                        jggua_walls.add((sp.x + dx, sp.y + dy))
        elif name.startswith('dfnuk-'):
            color_type = name.split('-')[1]
            if color_type not in colored_walls:
                colored_walls[color_type] = []
            # dfnuk sprites - pixel -1 = transparent, others = solid
            for dy in range(sp.height):
                for dx in range(sp.width):
                    if sp.pixels[dy, dx] != -1:
                        colored_walls[color_type].append((sp.x + dx, sp.y + dy))
        elif name.startswith('hnutp-'):
            color_type = name.split('-')[1]
            if color_type not in colored_triggers:
                colored_triggers[color_type] = []
            colored_triggers[color_type].append((sp.x, sp.y))

    return {
        'grid_size': (grid_w, grid_h),
        'blocks': blocks,
        'toggles': toggles,
        'walls': walls,
        'jggua_walls': jggua_walls,
        'colored_walls': colored_walls,
        'colored_triggers': colored_triggers,
        'npwxa': npwxa,
    }


def m0r0_bfs(level_data, max_states=3000000):
    """
    BFS solver for m0r0.

    Blocks move simultaneously in mirrored directions.
    ubwff-idtiq: (dx, dy)
    ubwff-crkfz: (-dx, dy)
    kncqr-idtiq: (dx, -dy)
    kncqr-crkfz: (-dx, -dy)

    Block touching wyiex = reset (flash animation then all blocks reset to start).
    Blocks overlapping = both removed.
    All blocks removed = win.

    Colored walls: toggled by triggers (hnutp) when a block is on trigger position.
    cvcer toggles: clicked to open/close paths (not direction-based, uses ACTION6 click).
    """
    grid_w, grid_h = level_data['grid_size']
    jggua_walls = level_data['jggua_walls']
    wyiex_walls = level_data['walls']
    toggles = level_data['toggles']
    colored_walls = level_data['colored_walls']
    colored_triggers = level_data['colored_triggers']
    block_defs = level_data['blocks']

    n_blocks = len(block_defs)
    if n_blocks == 0:
        return []

    color_types = sorted(colored_walls.keys())

    def is_wall(pos, color_state):
        if pos in jggua_walls:
            return True
        if pos[0] < 0 or pos[0] >= grid_w or pos[1] < 0 or pos[1] >= grid_h:
            return True
        # Check colored walls - active when trigger NOT activated
        for ci, ct in enumerate(color_types):
            if not color_state[ci]:
                for wp in colored_walls.get(ct, []):
                    if pos == wp:
                        return True
        return False

    def get_trigger_state(positions, active):
        """Check which color triggers are activated."""
        state = []
        for ct in color_types:
            triggered = False
            for tpos in colored_triggers.get(ct, []):
                for bi in range(n_blocks):
                    if active[bi] and positions[bi] == tpos:
                        triggered = True
                        break
                if triggered:
                    break
            state.append(triggered)
        return tuple(state)

    initial_positions = tuple(b['pos'] for b in block_defs)
    initial_active = tuple([True] * n_blocks)
    initial_color_state = get_trigger_state(initial_positions, initial_active)

    initial_state = (initial_positions, initial_active, initial_color_state)
    visited = {initial_state}
    queue = deque([(initial_state, [])])
    states_explored = 0
    max_depth = 0

    DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    while queue:
        state, actions = queue.popleft()
        positions, active, color_state = state
        cur_depth = len(actions)

        if cur_depth > 150:
            continue
        if cur_depth > max_depth:
            max_depth = cur_depth
            if max_depth % 10 == 0:
                print(f"    depth={max_depth}, queue={len(queue)}, visited={len(visited)}")

        states_explored += 1
        if states_explored > max_states:
            print(f"    BFS exhausted at {states_explored} states")
            return None

        for action_id in range(4):
            dx, dy = DIRS[action_id]
            new_positions = list(positions)

            for bi in range(n_blocks):
                if not active[bi]:
                    continue
                info = block_defs[bi]
                bdx = -dx if info['mirror_x'] else dx
                bdy = -dy if info['mirror_y'] else dy
                new_pos = (positions[bi][0] + bdx, positions[bi][1] + bdy)

                # Check wall collision
                if is_wall(new_pos, color_state):
                    new_pos = positions[bi]  # Stay

                # Also check collision with nhiae (toggle) sprites
                if new_pos in toggles:
                    new_pos = positions[bi]  # Toggles are collidable

                new_positions[bi] = new_pos

            new_positions_t = tuple(new_positions)

            # Check wyiex collision (blocks landing on wall markers)
            any_on_wall = False
            for bi in range(n_blocks):
                if active[bi] and new_positions_t[bi] in wyiex_walls:
                    any_on_wall = True
                    break

            if any_on_wall:
                # All blocks reset to original positions
                reset_pos = list(new_positions_t)
                for bi in range(n_blocks):
                    if active[bi]:
                        reset_pos[bi] = block_defs[bi]['pos']
                new_positions_t = tuple(reset_pos)

            # Check overlaps
            new_active = list(active)
            pos_groups = {}
            for bi in range(n_blocks):
                if new_active[bi]:
                    p = new_positions_t[bi]
                    if p not in pos_groups:
                        pos_groups[p] = []
                    pos_groups[p].append(bi)

            for p, bis in pos_groups.items():
                if len(bis) >= 2:
                    for bi in bis[:2]:
                        new_active[bi] = False

            new_active_t = tuple(new_active)
            new_color = get_trigger_state(new_positions_t, new_active_t)

            # Win check
            if not any(new_active_t):
                print(f"    SOLVED! depth={cur_depth+1}, states={states_explored}")
                return actions + [action_id]

            new_state = (new_positions_t, new_active_t, new_color)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, actions + [action_id]))

    print(f"    BFS complete: {states_explored} states, depth={max_depth}")
    return None


def solve_m0r0_all():
    """Solve all 6 levels of m0r0."""
    sys.path.insert(0, 'environment_files/m0r0/dadda488')
    from m0r0 import levels as M0R0_LEVELS

    # Load existing L1 solution
    presc_path = "B:/M/the-search/experiments/results/prescriptions/m0r0_full_seq.json"
    with open(presc_path) as f:
        l1_data = json.load(f)
    l1_raw = l1_data["full_sequence"]

    all_actions = list(l1_raw)
    per_level = {"L1": {"count": len(l1_raw), "actions": l1_raw, "source": "random_search"}}
    total_t0 = time.time()

    for level_idx in range(1, 6):
        print(f"\n{'='*60}")
        print(f"m0r0 Level {level_idx+1}/6")
        print(f"{'='*60}")

        level_data = extract_m0r0_level(M0R0_LEVELS[level_idx])
        print(f"  Grid: {level_data['grid_size']}")
        print(f"  Blocks: {len(level_data['blocks'])}")
        for b in level_data['blocks']:
            print(f"    {b['name']} at {b['pos']} mirror_x={b['mirror_x']} mirror_y={b['mirror_y']}")
        print(f"  Toggles: {level_data['toggles']}")
        print(f"  Walls: {len(level_data['walls'])} wyiex, {len(level_data['jggua_walls'])} maze")
        print(f"  ColoredWalls: {list(level_data['colored_walls'].keys())}")

        t0 = time.time()
        solution = m0r0_bfs(level_data)
        elapsed = time.time() - t0

        if solution is None:
            print(f"  FAILED ({elapsed:.1f}s)")
            break

        per_level[f"L{level_idx+1}"] = {
            "actions": solution,
            "count": len(solution),
            "time": round(elapsed, 2)
        }
        all_actions.extend(solution)
        print(f"  Solved: {len(solution)} actions in {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    max_level = verify_solution("m0r0", all_actions)

    return {
        "game": "m0r0",
        "source": "analytical_bfs_solver",
        "type": "analytical",
        "total_actions": len(all_actions),
        "max_level": max_level,
        "per_level": per_level,
        "all_actions": all_actions,
        "total_time": round(total_elapsed, 2),
    }


# ============================================================================
# Verification and main
# ============================================================================

def verify_solution(game_id, actions):
    """Verify a solution sequence via the arc_agi API.

    For m0r0, uses util_arcagi3 encoding: 0-6=keyboard, 7+=click at pixel.
    For ka59/wa30, actions 0-4 map to ACTION1-ACTION5 (pure keyboard).
    """
    print(f"\nVerifying {game_id} ({len(actions)} actions)...")

    arcade = arc_agi.Arcade()
    games = arcade.get_environments()
    info = next(g for g in games if game_id in g.game_id.lower())
    env = arcade.make(info.game_id)
    obs = env.reset()

    # Detect click support
    supports_click = any(ga == GameAction.ACTION6 for ga in env.action_space)

    max_level = 0
    for i, a in enumerate(actions):
        if a >= 7:
            # Click action
            click_idx = a - 7
            px = click_idx % 64
            py = click_idx // 64
            obs = env.step(GameAction.ACTION6, data={"x": px, "y": py})
        elif a == 5 and supports_click:
            # ACTION6 as keyboard in click game - needs data
            obs = env.step(GameAction.ACTION6, data={"x": 0, "y": 0})
        else:
            ga = list(GameAction)[a + 1]  # 0->ACTION1, 1->ACTION2, etc.
            obs = env.step(ga)
        if obs is None:
            continue
        if obs.levels_completed > max_level:
            max_level = obs.levels_completed
            print(f"  Level {max_level} completed at action {i}")
        if obs.state == GameState.WIN:
            print(f"  WIN! levels={obs.levels_completed}")
            break
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at action {i}, levels={obs.levels_completed}")
            break

    print(f"  Result: max_level={max_level}")
    return max_level


def save_result(result, game_id):
    out_path = f"B:/M/the-search/experiments/results/prescriptions/{game_id}_fullchain.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    os.chdir("B:/M/the-search")
    game = sys.argv[1] if len(sys.argv) > 1 else "wa30"

    if game in ["wa30", "all"]:
        result = solve_wa30_all()
        if result:
            save_result(result, "wa30")

    if game in ["ka59", "all"]:
        result = solve_ka59_all()
        if result:
            save_result(result, "ka59")

    if game in ["m0r0", "all"]:
        result = solve_m0r0_all()
        if result:
            save_result(result, "m0r0")
