"""
M0R0 Full Chain Solver — all 6 levels.

Mechanics (from source code analysis):
- 2 or 4 blocks (qzfkx sprites) move simultaneously
  - ubwff-idtiq: normal (dx, dy)
  - ubwff-crkfz: X-mirrored (-dx, dy)
  - kncqr-idtiq: Y-mirrored (dx, -dy)
  - kncqr-crkfz: XY-mirrored (-dx, -dy)
- Blocks collide with: jggua walls (0-valued pixels), cvcer toggles (nhiae tag),
  dfnuk doors (when closed)
- wyiex: checkered kill zones — if ANY block lands on one, ALL blocks revert to
  pre-move positions (wasted action)
- cvcer: toggle blocks — click to select (ACTION6), move with arrows, click elsewhere to deselect
- dfnuk-{color}: door walls (3px, horizontal when rot=90). Opened when ANY active block
  sits on a matching hnutp-{color} pressure plate
- Win: all active blocks pair up at same positions → removed. When 0 remain → next level.

Block pairing rules:
- After movement, if two blocks were previously horizontally adjacent (abs(prev_x)==1, same y)
  and have now swapped (each at other's prev pos), they merge to average position
- Then any 2+ blocks at same position are removed (paired off, 2 at a time)

Actions: ACTION1=UP(dy=-1), ACTION2=DOWN(dy+1), ACTION3=LEFT(dx=-1), ACTION4=RIGHT(dx+1)
         ACTION5=noop, ACTION6=click(x,y)
"""

import json
import sys
import os
from collections import deque

# ============================================================
# MAZE DATA (from m0r0.py sprite definitions)
# ============================================================

MAZES = {
    "Level1": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0],
        [0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level2": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level6": [
        [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level9": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level11": [
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
    ],
}


def rotate_180(pixels):
    """Rotate pixel array 180 degrees."""
    rows = len(pixels)
    cols = len(pixels[0]) if rows > 0 else 0
    return [[pixels[rows - 1 - r][cols - 1 - c] for c in range(cols)] for r in range(rows)]


# ============================================================
# LEVEL DEFINITIONS
# ============================================================

LEVELS = [
    # Level 1: Level6 maze rotated 180, 2 ubwff blocks
    {
        "maze": "Level6", "maze_pos": (0, 0), "maze_rot": 180,
        "blocks": [
            ("ubwff-idtiq", 3, 9),
            ("ubwff-crkfz", 7, 9),
        ],
        "toggles": [],
        "wyiex": [],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (11, 11),
    },
    # Level 2: Level11 maze at (2,0), 2 ubwff blocks + many wyiex
    {
        "maze": "Level11", "maze_pos": (2, 0),
        "blocks": [
            ("ubwff-idtiq", 4, 1),
            ("ubwff-crkfz", 8, 1),
        ],
        "toggles": [],
        "wyiex": [
            (5, 5), (5, 6), (12, 8), (11, 8), (10, 8), (9, 8), (8, 8),
            (4, 8), (2, 8), (1, 8), (0, 8), (5, 8), (5, 7),
            (4, 12), (3, 12), (2, 12), (1, 12), (0, 12),
            (9, 12), (8, 12), (7, 12), (6, 12), (5, 12),
            (12, 12), (11, 12), (10, 12),
        ],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (13, 13),
    },
    # Level 3: Level1 maze, 3 cvcer toggles, 2 ubwff blocks
    {
        "maze": "Level1", "maze_pos": (0, 0),
        "blocks": [
            ("ubwff-idtiq", 4, 10),
            ("ubwff-crkfz", 8, 10),
        ],
        "toggles": [(1, 3), (6, 2), (8, 6)],
        "wyiex": [],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (13, 13),
    },
    # Level 4: Level9 maze, 1 cvcer toggle, lots of wyiex
    {
        "maze": "Level9", "maze_pos": (0, 0),
        "blocks": [
            ("ubwff-idtiq", 2, 6),
            ("ubwff-crkfz", 8, 4),
        ],
        "toggles": [(5, 5)],
        "wyiex": [
            (1, 1), (2, 1), (3, 1), (9, 1), (8, 1), (7, 1),
            (7, 9), (9, 9), (8, 9), (3, 9), (2, 9), (1, 9),
            (4, 6), (5, 6), (6, 6), (4, 4), (5, 4), (6, 4),
        ],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (11, 11),
    },
    # Level 5: Level2 maze, doors + pressure plates, 2 ubwff blocks
    {
        "maze": "Level2", "maze_pos": (0, 0),
        "blocks": [
            ("ubwff-idtiq", 13, 12),
            ("ubwff-crkfz", 1, 12),
        ],
        "toggles": [],
        "wyiex": [],
        "dfnuk": {
            "qeazm": [(10, 9, 90), (10, 5, 90)],
            "raixb": [(3, 5, 90)],
            "ujcze": [(2, 9, 90)],
        },
        "hnutp": {
            "qeazm": [(3, 12), (3, 1)],
            "raixb": [(8, 6)],
            "ujcze": [(14, 6)],
        },
        "grid_size": (15, 15),
    },
    # Level 6: No jggua maze, walls made of wyiex. 1 cvcer toggle, doors + plates
    {
        "maze": None,
        "blocks": [
            ("ubwff-idtiq", 3, 4),
            ("ubwff-crkfz", 9, 4),
        ],
        "toggles": [(6, 9)],
        "wyiex": [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),
            (0, 11), (0, 9), (0, 10), (0, 5), (0, 6), (0, 7), (0, 8), (0, 1), (0, 2), (0, 3), (0, 4),
            (12, 11), (12, 9), (12, 10), (12, 5), (12, 6), (12, 7), (12, 8), (12, 1), (12, 2), (12, 3), (12, 4),
            (6, 11), (6, 10), (6, 5), (6, 6), (6, 7), (6, 8), (6, 1), (6, 2), (6, 3), (6, 4),
            (5, 6), (1, 6), (7, 6), (11, 6),
        ],
        "dfnuk": {
            "raixb": [(2, 6, 90)],
            "ujcze": [(8, 6, 90)],
        },
        "hnutp": {
            "raixb": [(9, 2), (3, 9)],
            "ujcze": [(3, 2)],
        },
        "grid_size": (13, 13),
    },
]


def build_maze_walls(level):
    """Build static wall set from jggua maze."""
    walls = set()
    maze_name = level.get("maze")
    if maze_name is None:
        return walls

    maze_pixels = MAZES[maze_name]
    mx, my = level.get("maze_pos", (0, 0))
    maze_rot = level.get("maze_rot", 0)

    if maze_rot == 180:
        maze_pixels = rotate_180(maze_pixels)

    for r, row in enumerate(maze_pixels):
        for c, val in enumerate(row):
            if val == 0:
                walls.add((mx + c, my + r))

    return walls


def build_door_cells(dfnuk_dict):
    """Build door cell positions per color.
    dfnuk sprites are 3x1 (vertical). With rotation 90, they become 1x3 (horizontal).
    Actually — the sprite is [[color],[color],[color]] which is 1 wide, 3 tall.
    With rot=90, it becomes 3 wide, 1 tall.
    Position (x,y) is top-left. So rot=90: cells at (x,y), (x+1,y), (x+2,y)."""
    doors = {}
    for color, door_list in dfnuk_dict.items():
        cells = []
        for item in door_list:
            x, y, rot = item
            if rot == 90:
                cells.extend([(x, y), (x + 1, y), (x + 2, y)])
            else:
                cells.extend([(x, y), (x, y + 1), (x, y + 2)])
        doors[color] = cells
    return doors


def get_block_movement(block_type, dx, dy):
    """Get actual movement for a block type given input direction (dx, dy)."""
    if "ubwff-idtiq" in block_type:
        return (dx, dy)
    elif "ubwff-crkfz" in block_type:
        return (-dx, dy)
    elif "kncqr-idtiq" in block_type:
        return (dx, -dy)
    elif "kncqr-crkfz" in block_type:
        return (-dx, -dy)
    return (dx, dy)


# ============================================================
# SOLVER
# ============================================================

def solve_level(level_idx, max_states=5000000):
    """Solve a single M0R0 level using BFS.

    State = (block_positions_tuple, toggle_positions_tuple, doors_open_tuple)

    For levels without toggles: simple BFS over block positions + door states.
    For levels with toggles: BFS with toggle select/move/deselect.
    """
    level = LEVELS[level_idx]
    gw, gh = level["grid_size"]

    # Build static walls (maze)
    maze_walls = build_maze_walls(level)

    # Build wyiex set
    wyiex_set = set(level["wyiex"])

    # Door info
    door_cells = build_door_cells(level.get("dfnuk", {}))
    plate_positions = level.get("hnutp", {})
    has_doors = len(door_cells) > 0

    # Block info
    blocks = level["blocks"]  # list of (type, x, y)
    n_blocks = len(blocks)
    block_types = [b[0] for b in blocks]
    initial_block_pos = tuple((b[1], b[2]) for b in blocks)

    # Toggle info
    toggles = level.get("toggles", [])
    has_toggles = len(toggles) > 0
    initial_toggle_pos = tuple(toggles)

    # Door colors
    door_colors = sorted(door_cells.keys())

    # Action deltas: 0=UP(dy=-1), 1=DOWN(dy=1), 2=LEFT(dx=-1), 3=RIGHT(dx=1)
    action_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def compute_doors_open(block_positions, toggle_positions):
        """Check which doors are open based on block/toggle positions on plates."""
        if not has_doors:
            return ()

        # Positions that can press plates: active blocks only
        occupied = set(block_positions)

        result = []
        for color in door_colors:
            plates = plate_positions.get(color, [])
            pressed = any((px, py) in occupied for px, py in plates)
            result.append(pressed)
        return tuple(result)

    def get_door_walls(doors_open_tuple):
        """Get set of door wall positions for closed doors."""
        if not has_doors:
            return set()
        walls = set()
        for i, color in enumerate(door_colors):
            if not doors_open_tuple[i]:
                for cell in door_cells[color]:
                    walls.add(cell)
        return walls

    def try_move_block(bx, by, dx, dy, all_walls):
        """Try to move a block. Returns new position (may be same if blocked)."""
        nx, ny = bx + dx, by + dy
        if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
            return (bx, by)
        if (nx, ny) in all_walls:
            return (bx, by)
        return (nx, ny)

    if not has_toggles:
        # ---- SIMPLE BFS (no toggles) ----
        initial_doors = compute_doors_open(initial_block_pos, initial_toggle_pos)

        # State: (block_positions, doors_open)
        # For toggle-less levels, toggles are always static (treated as static walls in collision)
        # Actually toggles in these levels are empty, so no toggle walls

        initial_state = (initial_block_pos, initial_doors)
        queue = deque([(initial_state, [])])
        visited = {initial_state}
        states_explored = 0

        while queue and states_explored < max_states:
            state, path = queue.popleft()
            states_explored += 1

            if states_explored % 500000 == 0:
                print(f"  L{level_idx+1}: {states_explored} states explored, queue={len(queue)}")

            block_pos, doors_open = state

            # Win check: all blocks paired (at same position)
            pos_set = {}
            for bp in block_pos:
                pos_set[bp] = pos_set.get(bp, 0) + 1
            all_paired = all(c >= 2 for c in pos_set.values()) and len(block_pos) > 0
            if all_paired:
                print(f"  L{level_idx+1}: SOLVED in {len(path)} moves ({states_explored} states)")
                return path

            # Build wall set for this state
            all_walls = set(maze_walls)
            # Add toggle positions as walls (static)
            for tx, ty in initial_toggle_pos:
                all_walls.add((tx, ty))
            # Add closed door walls
            all_walls |= get_door_walls(doors_open)

            for ai, (adx, ady) in enumerate(action_deltas):
                new_positions = []
                for bi in range(n_blocks):
                    bx, by = block_pos[bi]
                    bdx, bdy = get_block_movement(block_types[bi], adx, ady)
                    new_positions.append(try_move_block(bx, by, bdx, bdy, all_walls))

                new_pos_tuple = tuple(new_positions)

                # Check wyiex: if any block lands on wyiex, revert all
                hit_wyiex = any((nx, ny) in wyiex_set for nx, ny in new_pos_tuple)
                if hit_wyiex:
                    continue  # Revert = no state change, waste of action

                # Check block overlap merge
                # First: handle the "swap" merge (horizontally adjacent blocks that swap)
                # For simplicity in BFS, we just check if any two blocks are at same position
                # The swap-merge is a refinement that moves them to midpoint, but if they're
                # at same position already, they'd be removed anyway

                new_doors = compute_doors_open(new_pos_tuple, initial_toggle_pos)
                new_state = (new_pos_tuple, new_doors)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [ai]))

        print(f"  L{level_idx+1}: NOT SOLVED ({states_explored} states)")
        return None

    else:
        # ---- BFS WITH TOGGLES ----
        # State: (block_positions, toggle_positions, doors_open, selected_toggle_idx)
        # selected_toggle_idx = -1 means block movement mode

        initial_doors = compute_doors_open(initial_block_pos, initial_toggle_pos)
        initial_state = (initial_block_pos, initial_toggle_pos, initial_doors, -1)

        queue = deque([(initial_state, [])])
        visited = {initial_state}
        states_explored = 0

        while queue and states_explored < max_states:
            state, path = queue.popleft()
            states_explored += 1

            if states_explored % 500000 == 0:
                print(f"  L{level_idx+1}: {states_explored} states explored, queue={len(queue)}")

            block_pos, toggle_pos, doors_open, sel_idx = state

            # Win check (only when not in toggle mode)
            if sel_idx == -1:
                pos_set = {}
                for bp in block_pos:
                    pos_set[bp] = pos_set.get(bp, 0) + 1
                all_paired = all(c >= 2 for c in pos_set.values())
                if all_paired:
                    print(f"  L{level_idx+1}: SOLVED in {len(path)} actions ({states_explored} states)")
                    return path

            # Build wall set
            all_walls = set(maze_walls)
            for ti, (tx, ty) in enumerate(toggle_pos):
                if ti != sel_idx:  # Selected toggle doesn't block
                    all_walls.add((tx, ty))
            all_walls |= get_door_walls(doors_open)

            if sel_idx == -1:
                # BLOCK MOVEMENT MODE
                for ai, (adx, ady) in enumerate(action_deltas):
                    new_positions = []
                    for bi in range(n_blocks):
                        bx, by = block_pos[bi]
                        bdx, bdy = get_block_movement(block_types[bi], adx, ady)
                        new_positions.append(try_move_block(bx, by, bdx, bdy, all_walls))

                    new_pos_tuple = tuple(new_positions)

                    # Check wyiex
                    hit_wyiex = any((nx, ny) in wyiex_set for nx, ny in new_pos_tuple)
                    if hit_wyiex:
                        continue

                    new_doors = compute_doors_open(new_pos_tuple, toggle_pos)
                    new_state = (new_pos_tuple, toggle_pos, new_doors, -1)

                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [ai]))

                # TRY CLICKING EACH TOGGLE (select it)
                for ti in range(len(toggle_pos)):
                    new_state = (block_pos, toggle_pos, doors_open, ti)
                    action_code = 100 + ti
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append((new_state, path + [action_code]))

            else:
                # TOGGLE MOVEMENT MODE
                tx, ty = toggle_pos[sel_idx]

                for ai, (adx, ady) in enumerate(action_deltas):
                    ntx, nty = tx + adx, ty + ady
                    if 0 <= ntx < gw and 0 <= nty < gh and (ntx, nty) not in all_walls:
                        # Also can't move toggle onto a block
                        if not any((ntx, nty) == bp for bp in block_pos):
                            new_toggles = list(toggle_pos)
                            new_toggles[sel_idx] = (ntx, nty)
                            new_toggle_tuple = tuple(new_toggles)
                            new_doors = compute_doors_open(block_pos, new_toggle_tuple)
                            new_state = (block_pos, new_toggle_tuple, new_doors, sel_idx)
                            if new_state not in visited:
                                visited.add(new_state)
                                queue.append((new_state, path + [ai]))

                # DESELECT (click empty space)
                new_state = (block_pos, toggle_pos, doors_open, -1)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [200]))

        print(f"  L{level_idx+1}: NOT SOLVED ({states_explored} states)")
        return None


def abstract_to_prism_actions(actions, level_idx):
    """Convert abstract BFS actions to PRISM action IDs.

    Movement: 0=UP→ACTION1(id=1), 1=DOWN→ACTION2(id=2), 2=LEFT→ACTION3(id=3), 3=RIGHT→ACTION4(id=4)
    Click toggle: 100+ti → ACTION6(id=6) with click coordinates at toggle grid position
    Deselect: 200 → ACTION6(id=6) clicking on a wall/empty position

    For ACTION6, the game uses display_to_grid via Camera. The click coords are pixel coords
    in the 64x64 frame. grid→pixel: px = grid_x * scale + x_offset + scale//2
    """
    level = LEVELS[level_idx]
    gw, gh = level["grid_size"]
    scale = min(64 // gw, 64 // gh)
    scaled_w = gw * scale
    scaled_h = gh * scale
    x_off = (64 - scaled_w) // 2
    y_off = (64 - scaled_h) // 2

    toggles = list(level.get("toggles", []))
    current_toggles = list(toggles)
    sel_idx = -1

    prism = []
    for a in actions:
        if a < 4:
            # Movement: abstract 0=UP→id 1, 1=DOWN→id 2, 2=LEFT→id 3, 3=RIGHT→id 4
            prism.append(a + 1)
            # If in toggle mode, track toggle movement
            if sel_idx >= 0:
                dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a]
                tx, ty = current_toggles[sel_idx]
                current_toggles[sel_idx] = (tx + dx, ty + dy)
        elif 100 <= a < 200:
            # Select toggle
            ti = a - 100
            sel_idx = ti
            tx, ty = current_toggles[ti]
            px = tx * scale + x_off + scale // 2
            py = ty * scale + y_off + scale // 2
            px = min(px, 63)
            py = min(py, 63)
            prism.append(("click", px, py))
        elif a == 200:
            # Deselect: click on a wall position
            # Find a wall position in the maze that's safe to click
            # Use (0,0) grid position which is almost always a wall
            px = 0 * scale + x_off + scale // 2
            py = 0 * scale + y_off + scale // 2
            # But we need to click somewhere that is NOT a cvcer toggle
            # Actually, clicking anywhere that is NOT a cvcer should deselect
            # Let's click at pixel (0, 0) — outside the grid = no sprite there
            # Actually the game uses camera.display_to_grid which needs valid coords
            # Let's click on a known wall position
            # The corner (0,0) in grid should be wall for all mazes
            px = min(px, 63)
            py = min(py, 63)
            prism.append(("click", px, py))
            sel_idx = -1

    return prism


def prism_to_action_ids(prism_actions):
    """Convert PRISM actions to flat action IDs for the API.

    For arc_agi API:
    - Simple actions (1-5): just the action ID
    - ACTION6 (click): action_id=6, x=px, y=py

    For the chain file, we encode clicks as: 7 + y * 64 + x
    """
    action_ids = []
    for a in prism_actions:
        if isinstance(a, tuple):
            # Click action
            _, px, py = a
            action_ids.append(7 + py * 64 + px)
        else:
            action_ids.append(a)
    return action_ids


def verify_solution_api(level_solutions):
    """Verify solutions against the real game API."""
    try:
        from arc_agi import arc_env
    except ImportError:
        print("arc_agi not available, skipping API verification")
        return False

    env = arc_env("m0r0")
    obs = env.reset()

    total_actions = 0
    for li, sol in enumerate(level_solutions):
        if sol is None:
            print(f"  L{li+1}: No solution, cannot verify")
            return False

        prism = abstract_to_prism_actions(sol, li)
        flat = prism_to_action_ids(prism)

        print(f"  L{li+1}: Executing {len(flat)} actions...")
        for ai, action in enumerate(flat):
            if action >= 7:
                # Click action
                py = (action - 7) // 64
                px = (action - 7) % 64
                obs, reward, done, info = env.step(6, x=px, y=py)
            else:
                obs, reward, done, info = env.step(action)

            if done:
                break

        levels_completed = info.get("levels_completed", 0)
        print(f"  After L{li+1}: levels_completed={levels_completed}, state={info.get('state', '?')}")

        if levels_completed < li + 1:
            print(f"  FAILED at level {li+1}")
            return False

        total_actions += len(flat)

    final_state = info.get("state", "")
    print(f"\n  Final state: {final_state}")
    print(f"  Total actions: {total_actions}")
    return final_state == "WIN"


def verify_with_mcp(level_solutions):
    """Verify solutions using the arc-game MCP (interactive step-by-step)."""
    print("\n--- MCP Verification ---")
    print("Run the actions through arc-game MCP manually or use this output")

    all_actions = []
    for li, sol in enumerate(level_solutions):
        if sol is None:
            print(f"L{li+1}: UNSOLVED")
            continue
        prism = abstract_to_prism_actions(sol, li)
        flat = prism_to_action_ids(prism)
        all_actions.extend(flat)
        print(f"L{li+1}: {len(flat)} actions: {flat[:20]}{'...' if len(flat) > 20 else ''}")

    return all_actions


def main():
    print("=" * 60)
    print("M0R0 Full Chain Solver")
    print("=" * 60)

    solutions = []
    all_flat_actions = []
    per_level = {}

    for li in range(6):
        print(f"\n--- Level {li+1} ---")
        level = LEVELS[li]
        print(f"  Grid: {level['grid_size']}, Blocks: {len(level['blocks'])}")
        print(f"  Toggles: {len(level.get('toggles', []))}")
        print(f"  Doors: {list(level.get('dfnuk', {}).keys())}")
        print(f"  Wyiex: {len(level.get('wyiex', []))}")

        sol = solve_level(li)
        solutions.append(sol)

        if sol is not None:
            prism = abstract_to_prism_actions(sol, li)
            flat = prism_to_action_ids(prism)
            per_level[f"L{li+1}"] = {
                "count": len(flat),
                "actions": flat,
                "abstract_actions": sol,
            }
            all_flat_actions.extend(flat)
            print(f"  -> {len(flat)} actions (abstract: {len(sol)})")
        else:
            per_level[f"L{li+1}"] = {"count": 0, "actions": [], "status": "UNSOLVED"}

    # Save results
    result = {
        "game": "m0r0",
        "total_actions": len(all_flat_actions),
        "per_level": per_level,
        "all_actions": all_flat_actions,
    }

    output_file = "m0r0_fullchain.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_file}")
    print(f"Total actions: {len(all_flat_actions)}")

    solved = sum(1 for s in solutions if s is not None)
    print(f"Solved: {solved}/6 levels")

    if solved == 6:
        print("\nAll levels solved! Verifying against API...")
        success = verify_solution_api(solutions)
        if success:
            print("VERIFICATION PASSED!")
        else:
            print("VERIFICATION FAILED — check action encoding")

    return solutions, per_level


if __name__ == "__main__":
    main()
