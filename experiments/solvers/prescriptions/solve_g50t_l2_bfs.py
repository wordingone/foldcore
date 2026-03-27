import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput
from collections import deque

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()
game.set_level(1)
gs = game.vgwycxsxjz
room = gs.afbbgvkpip
STEP = 6

def can_move_to(x, y, doors_positions=None):
    """Check if player can occupy position (x,y)"""
    cx, cy = x+3, y+3
    # Floor check
    if room.esidlbhbhw(cx, cy) == -1:
        return False
    # Door collision check: player 7x7 at (x,y) vs door 7x7 at (dx,dy)
    if doors_positions is None:
        doors_positions = [(d.x, d.y) for d in gs.uwxkstolmf]
    for dx, dy in doors_positions:
        if abs(x - dx) < 7 and abs(y - dy) < 7:
            return False
    return True

def bfs_reachable(start_x, start_y, doors_positions=None):
    """BFS from start position, return all reachable positions"""
    visited = set()
    queue = deque([(start_x, start_y)])
    visited.add((start_x, start_y))
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,-STEP),(0,STEP),(-STEP,0),(STEP,0)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and can_move_to(nx, ny, doors_positions):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited

# Doors initially at (13,19) and (37,49)
initial_doors = [(13,19), (37,49)]
reachable_closed = bfs_reachable(49, 25, initial_doors)

print("L2 Reachable with both doors closed:")
for y in range(1, 56, 6):
    line = f"  y={y:2d}: "
    for x in range(1, 56, 6):
        if (x,y) in reachable_closed:
            if (x,y) == (49,25): line += " P "
            elif (x,y) == (25,19): line += " E "
            else: line += " . "
        elif can_move_to(x, y, []):  # floor exists but blocked by door
            line += " D "
        elif room.esidlbhbhw(x+3, y+3) != -1:
            line += " X "
        else:
            line += " # "
    print(line)

# With D2 open (button B2 at (37,25) pressed -> D2 moves from (37,49) to (37,43))
print("\nL2 Reachable with D2 open (D2 at 37,43):")
d2_open = [(13,19), (37,43)]
reachable_d2 = bfs_reachable(49, 25, d2_open)
for y in range(1, 56, 6):
    line = f"  y={y:2d}: "
    for x in range(1, 56, 6):
        if (x,y) in reachable_d2:
            if (x,y) == (49,25): line += " P "
            elif (x,y) == (25,19): line += " E "
            elif (x,y) == (37,25): line += " B2"
            elif (x,y) == (13,37): line += " B1"
            else: line += " . "
        elif can_move_to(x, y, []):
            line += " D "
        elif room.esidlbhbhw(x+3, y+3) != -1:
            line += " X "
        else:
            line += " # "
    print(line)

# With both doors open (D1 at 13,25, D2 at 37,43)
print("\nL2 Reachable with both doors open:")
both_open = [(13,25), (37,43)]
reachable_both = bfs_reachable(49, 25, both_open)
for y in range(1, 56, 6):
    line = f"  y={y:2d}: "
    for x in range(1, 56, 6):
        if (x,y) in reachable_both:
            if (x,y) == (49,25): line += " P "
            elif (x,y) == (25,19): line += " E "
            elif (x,y) == (37,25): line += " B2"
            elif (x,y) == (13,37): line += " B1"
            else: line += " . "
        elif can_move_to(x, y, []):
            line += " D "
        elif room.esidlbhbhw(x+3, y+3) != -1:
            line += " X "
        else:
            line += " # "
    print(line)

# Check: is exit reachable with just D2 open?
print(f"\nExit (25,19) reachable with D2 open: {(25,19) in reachable_d2}")
print(f"Exit (25,19) reachable with both open: {(25,19) in reachable_both}")
print(f"B1 (13,37) reachable with D2 open: {(13,37) in reachable_d2}")
print(f"B2 (37,25) reachable with closed: {(37,25) in reachable_closed}")

