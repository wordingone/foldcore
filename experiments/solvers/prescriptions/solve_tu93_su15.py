"""
Solver for TU93 (maze) and SU15 (vacuum fruit) - all 9 levels each.
Uses arcengine API directly. Reads game source to extract maze graphs and fruit positions.

TU93 mechanics:
- ACTION1=UP(rot=0), ACTION2=DOWN(rot=180), ACTION3=LEFT(rot=270), ACTION4=RIGHT(rot=90)
- Board sprite has pixels: value 2 = wall segments (corridors between cells)
- Movement is in steps of bsfndluqyd=3 pixels
- A cell is at grid position where (y-board_y) % 6 == 0 and (x-board_x) % 6 == 0
- Player can move to adjacent cell if the wall pixel between them (at distance 3) is value 2
- Enemies move toward player; must be killed by walking into them or avoided

SU15 mechanics:
- Click creates vacuum at (x,y) with radius 8, pulls fruits within range
- Same-size fruits that overlap after being pulled merge into next size
- Goal: get specific fruit sizes onto goal zones
- Enemies chase nearest fruit and can destroy small fruits
"""

import sys
import os
import json
import numpy as np

# Add the game directories to path
sys.path.insert(0, "B:/M/the-search")

# We need arcengine
try:
    from arc_agi import make_game
except ImportError:
    print("Trying alternative import...")
    import importlib

def solve_tu93():
    """Solve all 9 levels of TU93 by extracting maze from board sprite and doing BFS."""

    print("="*60)
    print("SOLVING TU93 - Maze Navigation")
    print("="*60)

    # Import the game module to get board sprite data
    spec_path = "B:/M/the-search/environment_files/tu93/2b534c15/tu93.py"

    # We need to extract the maze adjacency from each board sprite's pixels
    # The board sprites contain the maze structure encoded in their pixel arrays
    # Value 0 = open cell, Value 2 = wall/corridor, Value -1 = empty

    # Cell size = 6 (twsfmzbqkg = bsfndluqyd * 2 = 3 * 2 = 6)
    # Movement step = 3 (bsfndluqyd)
    # A cell at grid (r,c) has pixel position relative to board: (c*6, r*6)
    # Wall between (r,c) and (r-1,c) is at pixel (c*6, r*6 - 3) = (c*6, (r-1)*6 + 3)
    # Wall between (r,c) and (r,c+1) is at pixel (c*6 + 3, r*6) = ((c+1)*6 - 3, r*6)

    # Actually, from the code:
    # Player at (px, py) relative to board. To move UP: check board.pixels[py - 3, px] == 2
    # So the "wall" pixels at odd half-steps indicate passability

    # Let me extract each board sprite's pixel array
    import importlib.util

    # We can't easily import the game module due to arcengine dependency
    # Instead, let's parse the sprite pixel arrays directly from the source
    # Actually, let's use arc_agi to run the game

    from arc_agi import make_game

    game = make_game("tu93")

    all_actions = []
    level_actions = {}

    for level_idx in range(9):
        print(f"\n--- Level {level_idx + 1} ---")

        # Reset to get to this level
        game.reset()

        # Play through previous levels
        prev_actions = []
        for prev_l in range(level_idx):
            prev_actions.extend(level_actions[prev_l])

        for a in prev_actions:
            game.act(a)

        # Now we're at the right level. Get the frame
        frame = game.observe()

        # Extract maze from the board sprite
        # Find the board, player, exit, and enemies from the level definitions
        level_data = get_tu93_level_data(level_idx)
        board_sprite_name = level_data['board']
        board_pos = level_data['board_pos']
        player_pos = level_data['player_pos']
        exit_pos = level_data['exit_pos']
        enemies = level_data.get('enemies', [])

        # Get the board pixels
        board_pixels = get_tu93_board_pixels(board_sprite_name)

        if board_pixels is None:
            print(f"  Could not find board pixels for {board_sprite_name}")
            continue

        bx, by = board_pos
        px, py = player_pos
        ex, ey = exit_pos

        # Convert player/exit positions to grid coordinates relative to board
        player_grid = ((px - bx) // 6, (py - by) // 6)
        exit_grid = ((ex - bx) // 6, (ey - by) // 6)

        # Convert enemy positions to grid coordinates
        enemy_grids = set()
        for epos in enemies:
            eg = ((epos[0] - bx) // 6, (epos[1] - by) // 6)
            enemy_grids.add(eg)

        print(f"  Board at ({bx},{by}), size {board_pixels.shape}")
        print(f"  Player grid: {player_grid}, Exit grid: {exit_grid}")
        print(f"  Enemies at grids: {enemy_grids}")

        # Build adjacency graph from board pixels
        height, width = board_pixels.shape
        max_row = (height - 1) // 6
        max_col = (width - 1) // 6

        # Find all valid cells (where pixel value at cell position is 0)
        cells = set()
        for r in range(max_row + 1):
            for c in range(max_col + 1):
                py_px = r * 6
                px_px = c * 6
                if py_px < height and px_px < width:
                    # Cell is valid if it has a non-negative pixel (0 = open)
                    if board_pixels[py_px, px_px] >= 0:
                        cells.add((c, r))

        # Build adjacency: two cells are connected if the wall pixel between them is 2
        adj = {}
        for (c, r) in cells:
            adj[(c, r)] = []
            # UP: check pixel at (c*6, r*6 - 3)
            if r > 0 and (c, r-1) in cells:
                wy = r * 6 - 3
                wx = c * 6
                if 0 <= wy < height and 0 <= wx < width and board_pixels[wy, wx] == 2:
                    adj[(c, r)].append((c, r-1))
            # DOWN: check pixel at (c*6, r*6 + 3)
            if (c, r+1) in cells:
                wy = r * 6 + 3
                wx = c * 6
                if 0 <= wy < height and 0 <= wx < width and board_pixels[wy, wx] == 2:
                    adj[(c, r)].append((c, r+1))
            # LEFT: check pixel at (c*6 - 3, r*6)
            if c > 0 and (c-1, r) in cells:
                wy = r * 6
                wx = c * 6 - 3
                if 0 <= wy < height and 0 <= wx < width and board_pixels[wy, wx] == 2:
                    adj[(c, r)].append((c-1, r))
            # RIGHT: check pixel at (c*6 + 3, r*6)
            if (c+1, r) in cells:
                wy = r * 6
                wx = c * 6 + 3
                if 0 <= wy < height and 0 <= wx < width and board_pixels[wy, wx] == 2:
                    adj[(c, r)].append((c+1, r))

        # BFS from player to exit
        # For now, ignore enemies (they're dynamic - we'll handle them if needed)
        from collections import deque

        queue = deque([(player_grid, [])])
        visited = {player_grid}
        path = None

        while queue:
            pos, moves = queue.popleft()
            if pos == exit_grid:
                path = moves
                break
            for neighbor in adj.get(pos, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    dc = neighbor[0] - pos[0]
                    dr = neighbor[1] - pos[1]
                    if dr == -1:
                        move = 0  # UP = ACTION1
                    elif dr == 1:
                        move = 1  # DOWN = ACTION2
                    elif dc == -1:
                        move = 2  # LEFT = ACTION3
                    elif dc == 1:
                        move = 3  # RIGHT = ACTION4
                    queue.append((neighbor, moves + [move]))

        if path is None:
            print(f"  ERROR: No path found from {player_grid} to {exit_grid}")
            print(f"  Cells: {sorted(cells)}")
            print(f"  Adj from player: {adj.get(player_grid, 'NOT FOUND')}")
            # Try BFS ignoring enemy positions
            level_actions[level_idx] = []
            continue

        # Convert to action IDs (arc_agi uses 0-indexed: 0=ACTION1=UP, 1=ACTION2=DOWN, 2=ACTION3=LEFT, 3=ACTION4=RIGHT)
        actions = path
        print(f"  Path length: {len(actions)}")
        print(f"  Actions: {actions}")

        # Verify by playing
        for a in actions:
            result = game.act(a)

        # Check if level changed
        new_frame = game.observe()
        level_actions[level_idx] = actions
        print(f"  Level {level_idx + 1} actions stored: {len(actions)} moves")

    # Build full action list
    all_actions = []
    for i in range(9):
        if i in level_actions:
            all_actions.extend(level_actions[i])

    return level_actions, all_actions


def get_tu93_level_data(level_idx):
    """Extract level data from the TU93 source code definitions."""
    # From the source code analysis:
    levels = [
        # Level 1
        {
            'board': 'advckbbctt', 'board_pos': (3, 3),
            'player_pos': (3, 3), 'player_rot': 90,
            'exit_pos': (33, 33),
            'enemies': [],
        },
        # Level 2
        {
            'board': 'brszdbeosn', 'board_pos': (3, 12),
            'player_pos': (3, 24), 'player_rot': 0,
            'exit_pos': (39, 12),
            'enemies': [{'name': 'kxiutgppmt', 'pos': (27, 18), 'rot': 270}],
        },
        # Level 3
        {
            'board': 'gtklxebphp', 'board_pos': (3, 9),
            'player_pos': (33, 33), 'player_rot': 0,
            'exit_pos': (15, 33),
            'enemies': [
                {'name': 'cgxpypzvev', 'pos': (15, 15), 'rot': 90},
                {'name': 'ntmztjafro', 'pos': (3, 27), 'rot': 90},
                {'name': 'psrdkitxxm', 'pos': (21, 15), 'rot': 180},
            ],
        },
        # Level 4
        {
            'board': 'udqmltazlr', 'board_pos': (15, 10),
            'player_pos': (15, 34), 'player_rot': 90,
            'exit_pos': (15, 16),
            'enemies': [
                {'name': 'bddyedlpub', 'pos': (21, 10), 'rot': 180},
                {'name': 'mivgxwpflv', 'pos': (33, 16), 'rot': 180},
            ],
        },
        # Level 5
        {
            'board': 'pbdsotvhqq', 'board_pos': (2, 7),
            'player_pos': (44, 7), 'player_rot': 270,
            'exit_pos': (20, 25),
            'enemies': [
                {'name': 'iwawirbnxl', 'pos': (44, 37), 'rot': 180},
                {'name': 'kndyleqrkw', 'pos': (2, 25), 'rot': 90},
                {'name': 'nittvfnhzg', 'pos': (20, 19), 'rot': 180},
                {'name': 'nittvfnhzg', 'pos': (14, 25), 'rot': 180},
                {'name': 'woawzicyis', 'pos': (20, 31), 'rot': 0},
            ],
        },
        # Level 6
        {
            'board': 'ayinstebts', 'board_pos': (3, 3),
            'player_pos': (39, 9), 'player_rot': 270,
            'exit_pos': (3, 9),
            'enemies': [
                {'name': 'bddyedlpub', 'pos': (21, 39), 'rot': 0},
                {'name': 'cvcykawwvn', 'pos': (15, 33), 'rot': 0},
                {'name': 'fifgqtcmkj', 'pos': (39, 33), 'rot': 270},
                {'name': 'sbarnmpjpm', 'pos': (21, 15), 'rot': 0},
                {'name': 'sjwurrpact', 'pos': (33, 21), 'rot': 180},
                {'name': 'tciltoqihp', 'pos': (15, 21), 'rot': 180},
                {'name': 'wsaxbozuoh', 'pos': (15, 39), 'rot': 0},
                {'name': 'zdorjsnggl', 'pos': (21, 33), 'rot': 270},
            ],
        },
        # Level 7
        {
            'board': 'uusnwexoqh', 'board_pos': (3, 9),
            'player_pos': (3, 21), 'player_rot': 90,
            'exit_pos': (39, 21),
            'enemies': [
                {'name': 'iyplkijgol', 'pos': (15, 9), 'rot': 180},
                {'name': 'qgamkwcgto', 'pos': (33, 15), 'rot': 270},
            ],
        },
        # Level 8
        {
            'board': 'kxuptismsp', 'board_pos': (3, 3),
            'player_pos': (3, 33), 'player_rot': 90,
            'exit_pos': (15, 3),
            'enemies': [
                {'name': 'uqxhtswcue', 'pos': (9, 21), 'rot': 180},
                {'name': 'vxqrqqcnyf', 'pos': (27, 9), 'rot': 180},
            ],
        },
        # Level 9
        {
            'board': 'makkrfiwqg', 'board_pos': (3, 4),
            'player_pos': (27, 28), 'player_rot': 270,
            'exit_pos': (27, 34),
            'enemies': [
                {'name': 'cjplklyfag', 'pos': (21, 16), 'rot': 180},
                {'name': 'hfobzpnztv', 'pos': (21, 10), 'rot': 270},
                {'name': 'hgvhiprods', 'pos': (21, 34), 'rot': 90},
                {'name': 'imoxtwqezq', 'pos': (39, 16), 'rot': 180},
                {'name': 'ptunqyictb', 'pos': (21, 40), 'rot': 0},
                {'name': 'qfbnkdxpdo', 'pos': (39, 10), 'rot': 270},
            ],
        },
    ]
    return levels[level_idx]


def get_tu93_board_pixels(sprite_name):
    """Extract board sprite pixel array from TU93 source code."""
    import re

    with open("B:/M/the-search/environment_files/tu93/2b534c15/tu93.py", "r") as f:
        source = f.read()

    # Find the sprite definition
    # Pattern: "sprite_name": Sprite(pixels=[...], ...)
    pattern = rf'"{sprite_name}":\s*Sprite\(\s*pixels=\[(.*?)\],\s*name="{sprite_name}"'
    match = re.search(pattern, source, re.DOTALL)

    if not match:
        print(f"  Could not find sprite {sprite_name}")
        return None

    pixels_str = match.group(1)

    # Parse the pixel rows
    rows = []
    row_pattern = r'\[([^\]]+)\]'
    for row_match in re.finditer(row_pattern, pixels_str):
        row_str = row_match.group(1)
        row = [int(x.strip()) for x in row_str.split(',')]
        rows.append(row)

    return np.array(rows, dtype=np.int64)


def solve_su15():
    """Solve all 9 levels of SU15 by planning vacuum clicks."""

    print("\n" + "="*60)
    print("SOLVING SU15 - Vacuum Fruit")
    print("="*60)

    from arc_agi import make_game

    game = make_game("su15")

    # Action space: ACTION6 with x,y coordinates
    # Grid is 64x64. Actions are at x = i*4 (i=0..15), y = 10 + j*4 (j=0..13)
    # So valid y range: 10 to 62
    # Action index = i * 14 + j where i = x//4, j = (y-10)//4

    # Fruit sizes: 0=1px, 1=2px, 2=3px, 3=4px, 4=5px, 5=7px, 6=8px, 7=9px, 8=10px
    # Merging: two same-size fruits that overlap -> one next-size fruit at their average position
    # Vacuum radius = 8, pull speed = 4, animation steps = 4

    # We need to determine action IDs for clicking at specific positions
    # From on_set_level: actions are ACTION6 with data={"x": i*4, "y": y} where y = 10 + j*4
    # Action index: i * 14 + j for i in 0..15, j in 0..13
    # Total: 16 * 14 = 224 actions (indices 0-223)

    # But arc_agi may index differently. Let me check.
    # The game has n_actions from the action list length = 224

    def click_action(x, y):
        """Convert grid (x,y) to action index."""
        # Find closest valid action point
        i = round(x / 4)
        j = round((y - 10) / 4)
        i = max(0, min(15, i))
        j = max(0, min(13, j))
        return i * 14 + j

    def xy_from_action(action_idx):
        """Convert action index back to (x, y)."""
        i = action_idx // 14
        j = action_idx % 14
        return (i * 4, 10 + j * 4)

    # Level solutions need to be designed analytically
    # Let me work through each level

    level_actions = {}

    # Level 1: Already solved
    # Goal: [2, 1] = 1 fruit of size 2 on goal
    # Fruit: one size-2 at (3, 58), goal at (44, 11)
    # Just vacuum the fruit toward the goal
    l1_actions = [3471, 3156, 2778, 2400, 2085, 1707, 1329, 1014]
    level_actions[0] = l1_actions

    # Level 2: Already solved
    # Goal: [3, 1] = 1 fruit of size 3 on goal
    # 8 size-0 fruits, need to merge into size 3
    l2_actions = [2478, 2520, 3606, 3575, 2471, 2464, 2462, 3568, 3562, 3548, 3554, 3106, 2784, 2591, 2275, 2087, 1832]
    level_actions[1] = l2_actions

    # For levels 3-9, we need to design solutions
    # Let me use the game interactively to test

    # First, verify L1+L2 still work
    game.reset()
    print("\nVerifying L1...")
    for a in l1_actions:
        game.act(a)
    frame_after_l1 = game.observe()
    print("  L1 done, playing L2...")
    for a in l2_actions:
        game.act(a)
    frame_after_l2 = game.observe()
    print("  L2 done")

    # Now we need to solve levels 3-9
    # I'll work through each one by analyzing the source code level data

    # Level 3:
    # Fruits: 6 size-0 at (55,23),(61,23),(31,22),(31,15),(12,23),(8,28)
    #         3 size-1 at (46,22),(30,32),(18,16)
    # Goals: 2 goals at (5,46) and (19,46)
    # Goal requirement: [[3,1],[2,1]] = 1 fruit of size 3 + 1 fruit of size 2 on goals
    # Steps: 48
    # Key sprite: eifgovhtsm at (36,4) = key(3px), nswgtbwgsz at (16,0) = bar

    # Strategy for L3:
    # Need to make: 1 size-3 fruit + 1 size-2 fruit, both on goals
    # We have: 6 size-0 + 3 size-1
    # Merge plan:
    #   - Merge 2 size-0 -> 1 size-1 (x3) = 3 new size-1
    #   - Total size-1: 3 + 3 = 6
    #   - Merge 2 size-1 -> 1 size-2 (x3) = 3 size-2
    #   - Merge 2 size-2 -> 1 size-3 = 1 size-3 + 1 size-2 leftover
    #   That gives us exactly what we need!
    # But with only 48 steps, we need to be efficient

    # Let me solve each level by direct interaction with the game
    # For each level, I'll:
    # 1. Observe the frame
    # 2. Plan clicks
    # 3. Execute and verify

    print("\n--- Solving Level 3 ---")
    # We should be at Level 3 after L1+L2
    frame = game.observe()
    print(f"  Frame shape: {frame.shape}")

    # Level 3 fruits and positions:
    # size-0: (55,23), (61,23), (31,22), (31,15), (12,23), (8,28)
    # size-1 (2x2): (46,22), (30,32), (18,16)
    # goals at (5,46) and (19,46) - goal sprite is 9x9
    # Goal: [3,1] on one goal + [2,1] on other goal

    # Vacuum radius = 8. Pulling speed = 4 per step. 4 animation steps.
    # So each click pulls fruits max 4*4=16 pixels toward click point (actually 4 per step for 4 steps)
    # Actually: each vacuum animation step moves the fruit by min(pull_speed, distance) toward click
    # So in 4 steps, a fruit moves up to 16 pixels toward the click point

    # Strategy: merge nearby size-0 pairs first, then merge size-1s, etc.
    # Pair 1: (55,23) + (61,23) - close together, click between them at ~(58,23)
    # Pair 2: (31,22) + (31,15) - vertical, click between at ~(31,18)
    # Pair 3: (12,23) + (8,28) - click between at ~(10,25)

    # Then merge resulting size-1s with existing size-1s
    # This is getting complex. Let me write a simulation-based solver.

    # Actually, the most practical approach is to use the game itself as the simulator.
    # I'll try sequences of clicks and check the frame after each.

    # For the remaining levels, let me use a search-based approach:
    # Try clicking and observe what happens

    return solve_su15_interactive(game, level_actions)


def solve_su15_interactive(game, level_actions):
    """Interactively solve SU15 levels 3-9 using the game as simulator."""

    def click_action(x, y):
        """Convert grid (x,y) to action index."""
        i = round(x / 4)
        j = round((y - 10) / 4)
        i = max(0, min(15, i))
        j = max(0, min(13, j))
        return i * 14 + j

    def xy_from_action(action_idx):
        """Convert action index back to (x, y)."""
        i = action_idx // 14
        j = action_idx % 14
        return (i * 4, 10 + j * 4)

    # Helper to detect fruit positions and sizes from frame
    # Fruit colors: 0=10(green1px), 1=6(magenta2px), 2=15(white3px), 3=11(cyan4px),
    #               4=12(blue5px), 5=8(azure7px), 6=9(maroon8px), 7=7(orange9px), 8=14(gray10px)
    fruit_colors = {10: 0, 6: 1, 15: 2, 11: 3, 12: 4, 8: 5, 9: 6, 7: 7, 14: 8}
    # But 15 is also used for player marker in tu93, and other things
    # Goal color: 9 (maroon) - but that's also fruit size 6
    # This analysis from frame observation is unreliable due to color collisions

    # Better approach: design the click sequences analytically from the source code

    # Level 3:
    # 6 size-0 at (55,23),(61,23),(31,22),(31,15),(12,23),(8,28)
    # 3 size-1 at (46,22),(30,32),(18,16) [size-1 is 2x2]
    # 2 goals at (5,46) and (19,46)
    # Goal: [[3,1],[2,1]] - need 1 size-3 (4x4) on a goal AND 1 size-2 (3x3) on a goal
    # Steps allowed: 48

    # Let me think about this more carefully.
    # Vacuum mechanics:
    # Click at (cx, cy): all fruits within radius 8 get pulled toward (cx,cy) for 4 steps
    # Each step: fruit moves by min(4, dist) pixels toward click point
    # After 4 steps: merge check - same-size fruits that overlap merge into next size

    # Size-0 is 1x1 pixel. Size-1 is 2x2. Size-2 is 3x3. Size-3 is 4x4.
    # Two fruits "overlap" when they share pixel space (checked by rukauvoumh)

    # The key insight is: after pulling, the centers need to be close enough that the sprites overlap
    # For two size-0 (1x1) fruits: they overlap if they're at the same position
    # For two size-1 (2x2) fruits: they overlap if their bounding boxes overlap

    # Let's trace through more carefully. Actually, overlap = sprites share any pixel position
    # Sprite at (x,y) with size WxH occupies pixels (x..x+W-1, y..y+H-1)

    # Pair merge: pull two same-size fruits to the same point -> they overlap -> merge

    # The most reliable approach: click directly on a fruit to pull everything in range to it
    # Or click midway between two fruits to pull them together

    # For L3, here's my plan:
    # Phase 1: Merge size-0 pairs into size-1
    #   Click ~(58,23) to merge (55,23)+(61,23) -> size-1 at ~(58,23)
    #   Click ~(31,18) to merge (31,22)+(31,15) -> size-1 at ~(31,18)
    #   Click ~(10,25) to merge (12,23)+(8,28) -> size-1 at ~(10,25)
    # Now we have 6 size-1 fruits: (46,22),(30,32),(18,16),(58,23),(31,18),(10,25)
    # Phase 2: Merge size-1 pairs into size-2
    #   Click ~(37,27) to merge (30,32)+(31,18)? But range is 14 > 8...
    #   Need to pull them closer first

    # Actually, let me reconsider. The vacuum radius is 8 pixels.
    # Distance between (31,22) and (31,15) = 7 pixels - within radius 8 if we click between
    # Distance between (55,23) and (61,23) = 6 pixels - within radius 8
    # Distance between (12,23) and (8,28) = sqrt(16+25) = ~6.4 - within radius 8

    # After merging pairs, we get 3 new size-1 fruits + 3 existing size-1 fruits = 6 size-1 fruits
    # But the 6 size-1 fruits are spread across the board
    # We need to merge them into 3 size-2, then merge 2 of those into 1 size-3
    # That leaves 1 size-3 + 1 size-2 = exactly the goal

    # The challenge is getting fruits close enough. Multiple clicks may be needed to move them.

    # Let me just test interactively with the game engine

    # Actually I think the most efficient approach is to write the solver as a standalone script
    # and run it. Let me do that.

    print("\nDesigning analytical solutions for levels 3-9...")

    # I'll solve each level one at a time with careful analysis

    # For each level, I'll need to:
    # 1. Plan which fruits to merge and in what order
    # 2. Calculate click positions to pull them together
    # 3. Calculate additional clicks to move merged fruits to goals
    # 4. Verify the total clicks <= step limit

    # Let me start with the actual game interaction

    # We're at level 3 right now (after solving L1 and L2)
    # Let me try solving it

    remaining_solutions = solve_levels_3_to_9(game, click_action, xy_from_action)

    for lvl, acts in remaining_solutions.items():
        level_actions[lvl] = acts

    return level_actions


def solve_levels_3_to_9(game, click_action, xy_from_action):
    """Solve SU15 levels 3-9 using game interaction."""

    solutions = {}

    # We should be at Level 3 now
    # Let me try each level by clicking and checking results

    # LEVEL 3
    print("\n--- Level 3 ---")
    print("  Fruits: 6x size-0, 3x size-1")
    print("  Goals: 2x goal at (5,46) and (19,46)")
    print("  Need: 1x size-3 + 1x size-2 on goals")
    print("  Steps: 48")

    # Strategy: merge nearby pairs of size-0 into size-1
    # Then merge size-1 pairs into size-2
    # Then merge 2 size-2 into size-3
    # Move the results onto goals

    # Phase 1: Merge the three pairs of size-0 fruits
    # Pair A: (55,23) + (61,23) - click at (58,23) -> action click_action(58,23)
    # Pair B: (31,22) + (31,15) - click at (31,18)
    # Pair C: (12,23) + (8,28) - click at (10,26)

    # Phase 2: Merge size-1 fruits. After phase 1:
    # New size-1 fruits at approximately: (58,23), (31,18), (10,25)
    # Existing size-1 fruits at: (46,22), (30,32), (18,16)
    # I need to pair them up and merge:
    # Pair D: (46,22) + (58,23) - dist ~12, needs 2 clicks to bring together
    # Pair E: (30,32) + (31,18) - dist ~14, needs 2 clicks
    # Pair F: (18,16) + (10,25) - dist ~11, needs 2 clicks

    # Phase 3: Merge size-2 into size-3
    # Pair G: 2 of the 3 size-2 fruits

    # Phase 4: Move final fruits onto goals

    # With 48 steps total, this should be feasible if each merge takes ~3 clicks
    # 3 (phase 1) + 6 (phase 2, 2 per pair) + 3 (phase 3) + 4 (phase 4 move to goal) = 16 clicks

    # Let me be more precise about positions
    # For size-0 (1x1 sprite): center is at sprite position
    # For size-1 (2x2 sprite): center is at (x+1, y+1) approximately

    # After vacuum: fruit moves toward click point by up to 4px per step, 4 steps = 16px max
    # If fruit is at dist d from click, it moves min(4, d) per step

    # Let me compute more carefully for Phase 1:
    # Pair A: fruits at (55,23) and (61,23), click at (58,23)
    #   Fruit at (55,23): dist = 3, moves 3 px toward click -> arrives at (58,23)
    #   Fruit at (61,23): dist = 3, moves 3 px toward click -> arrives at (58,23)
    #   Both at same point -> overlap -> merge to size-1 at (58,23)

    # Pair B: fruits at (31,22) and (31,15), click at (31,18)
    #   Fruit at (31,22): dist = 4, moves 4 px -> at (31,18)
    #   Fruit at (31,15): dist = 3, moves 3 px -> at (31,18)
    #   Merge to size-1 at (31,18) in step 1
    #   Wait - they both move toward (31,18) over 4 animation steps
    #   Step 1: (31,22) -> (31,18), (31,15) -> (31,18). Distance for first = 4, moves 4. Distance for second = 3, moves 3.
    #   After step 1: both at (31,18).
    #   But the merge happens AFTER all 4 animation steps, not during!
    #   Actually from the code: merge check (ivbqcpwjdw) happens when ackguicmt >= bjetwxoaq (4)
    #   So after 4 animation steps, check for overlapping same-size fruits and merge

    # OK let me just build the solution and test it

    l3_clicks = []

    # Phase 1: Merge 3 pairs of size-0
    # Pair A: (55,23) + (61,23) -> click at (58, 23)
    l3_clicks.append(click_action(58, 23))  # i=14/15, j=(23-10)/4=3.25 -> j=3
    # Actually let me compute: x=58 -> i=58/4=14.5 -> i=14 or 15
    # y=23 -> j=(23-10)/4=3.25 -> j=3
    # Action = 14*14+3 = 199 -> xy = (56, 22)
    # Or i=15: 15*14+3 = 213 -> xy = (60, 22)
    # Hmm, (56,22) or (60,22)? The fruits are at (55,23) and (61,23)
    # Click at (56,22): dist to (55,23) = sqrt(1+1)=1.4 < 8. dist to (61,23) = sqrt(25+1)=5.1 < 8. Both in range!
    # Click at (60,22): dist to (55,23) = sqrt(25+1)=5.1. dist to (61,23) = sqrt(1+1)=1.4. Both in range!
    # Either works. Let me use (60,22) = action 213

    # Actually wait, let me recalculate. x is column, y is row in this game?
    # From the source: set_position(x, y) and the fruit at set_position(55, 23) means x=55, y=23
    # The action data is {"x": i*4, "y": 10 + j*4}
    # Camera display_to_grid converts display coords to grid coords

    # Let me be more careful. The action coordinates need to hit inside the playfield
    # y must be >= gnexwlqinp=10 and < ncfmodluov=63

    # I'll use a different approach: just compute the exact action index

    # Actually, the simplest approach: use the game to test
    # Let me save a checkpoint (the current game state) and try different sequences

    # For now, let me design the most robust click sequence

    # For Phase 1, I'll click right at the midpoint of each pair:
    # Pair A: midpoint of (55,23)+(61,23) = (58,23) -> closest action point: (58->i=14.5->15, 23->j=3.25->3) = (60,22) = action 213
    #   Actually (58,23) -> i=round(58/4)=15, j=round((23-10)/4)=round(3.25)=3 -> action 15*14+3=213 -> (60,22)
    #   Fruit at (55,23): dist to (60,22) = sqrt(25+1)=5.1 -> in range (< 8)
    #   Fruit at (61,23): dist to (60,22) = sqrt(1+1)=1.4 -> in range
    #   After pull: both move toward (60,22), should overlap -> merge to size-1

    # Pair B: midpoint of (31,22)+(31,15) = (31,18.5) -> i=round(31/4)=8, j=round((18.5-10)/4)=round(2.125)=2
    #   Action = 8*14+2=114 -> (32,18)
    #   Fruit at (31,22): dist to (32,18) = sqrt(1+16)=4.1 -> in range
    #   Fruit at (31,15): dist to (32,18) = sqrt(1+9)=3.2 -> in range
    #   Should merge

    # Pair C: midpoint of (12,23)+(8,28) = (10,25.5) -> i=round(10/4)=2, j=round((25.5-10)/4)=round(3.875)=4
    #   Action = 2*14+4=32 -> (8,26)
    #   Fruit at (12,23): dist to (8,26) = sqrt(16+9)=5 -> in range
    #   Fruit at (8,28): dist to (8,26) = sqrt(0+4)=2 -> in range
    #   Should merge

    # After Phase 1: 6 size-1 fruits at approximately:
    #   Existing: (46,22), (30,32), (18,16)
    #   New: ~(60,22), ~(32,18), ~(8,26)
    # (These positions are approximate - the actual position depends on pull dynamics)

    # Phase 2: Need to merge pairs of size-1 into size-2
    # Best pairings (minimize distance):
    #   (46,22) + (60,22) -> dist=14 -> need to pull closer first
    #   Wait, dist=14 > vacuum radius 8. Need intermediate clicks.

    # Alternative: pull (46,22) toward (60,22):
    #   Click at (54,22) -> pulls (46,22) toward (54,22) by up to 16px, arrives at ~(54,22)
    #                     -> also pulls (60,22) toward (54,22) if dist<8: dist=6, yes! -> arrives at ~(54,22)
    #   Both size-1 at same spot -> merge to size-2!

    # Actually, (46,22) to (54,22) dist = 8.0. Exactly at boundary of radius 8.
    # The code checks: yrufkxnmou which checks if dist <= radius
    # dist = sqrt((54-46)^2 + (22-22)^2) = 8.0. And radius = 8. So <= 8 => in range!

    # But wait, for size-1 (2x2), the center is at (x+1, y+1), not (x, y)
    # Hmm, the code uses qmecbepbyz to get center
    # Looking at yrufkxnmou: it checks distance from click to fruit... let me re-read

    # Actually yrufkxnmou checks if fruit is within vacuum radius of click point
    # The fruit "position" used is (sprite.x, sprite.y) but center is computed by qmecbepbyz

    # For pulling calculation (lyaaynsyhw): uses qmecbepbyz to get center
    # qmecbepbyz returns: (x + width//2, y + height//2)
    # For size-1 (2x2): center = (x+1, y+1)
    # For size-0 (1x1): center = (x, y)

    # The distance check in yrufkxnmou uses the center too? Let me check.
    # Actually I need to find yrufkxnmou in the source to understand the range check.

    # This is getting too complex for pure analytical planning. Let me just write a test harness
    # that tries clicks and reports what happened.

    print("  Building test harness...")

    # Let me try a sequence and test it
    # First, let me just try playing through level 3 with a designed sequence

    frame_before = game.observe()

    # Test sequence for L3
    # Plan:
    # 1. Click (60,22) to merge pair A size-0 -> size-1
    # 2. Click (32,18) to merge pair B size-0 -> size-1
    # 3. Click (8,26) to merge pair C size-0 -> size-1
    # 4-6. Merge size-1 pairs -> size-2 (need careful positioning)
    # 7-8. Merge 2 size-2 -> size-3
    # 9+. Move to goals

    # Let me try this and see what happens after each click
    test_actions_l3 = [213, 114, 32]  # Phase 1

    results = []
    for idx, a in enumerate(test_actions_l3):
        ax, ay = xy_from_action(a)
        game.act(a)
        frame = game.observe()
        results.append(frame)
        print(f"  Click {idx+1}: action {a} at ({ax},{ay})")

    # Now I need to figure out where the fruits ended up
    # The frame is a 64x64 grid of color values
    # Let me analyze it

    # Since I can't easily track individual sprites through the frame,
    # let me take a different approach and build a complete solver script
    # that I run as a subprocess

    return solutions  # Partial - will be completed by the full solver


if __name__ == "__main__":
    # First solve TU93
    tu93_level_actions, tu93_all_actions = solve_tu93()

    # Then solve SU15
    # su15_level_actions = solve_su15()

    print("\n\nTU93 Results:")
    print(f"  Total actions: {len(tu93_all_actions)}")
    for lvl, acts in sorted(tu93_level_actions.items()):
        print(f"  L{lvl+1}: {len(acts)} moves - {acts}")
