import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput
from collections import deque

game = G50t()
game.set_level(1)
gs = game.vgwycxsxjz
room = gs.afbbgvkpip
STEP = 6

def can_move_to(x, y, doors_positions):
    cx, cy = x+3, y+3
    if room.esidlbhbhw(cx, cy) == -1:
        return False
    for dx, dy in doors_positions:
        if abs(x - dx) < 7 and abs(y - dy) < 7:
            return False
    return True

def bfs_reachable(start_x, start_y, doors_positions):
    visited = set()
    queue = deque([(start_x, start_y)])
    if can_move_to(start_x, start_y, doors_positions):
        visited.add((start_x, start_y))
    while queue:
        x, y = queue.popleft()
        for dx, dy in [(0,-STEP),(0,STEP),(-STEP,0),(STEP,0)]:
            nx, ny = x+dx, y+dy
            if (nx, ny) not in visited and can_move_to(nx, ny, doors_positions):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return visited

# D1 at (13,19) rot=0 -> hluvhlvimq: rot=0 -> (0,1) -> moves (0, 6) = to (13,25)
# D2 at (37,49) rot=180 -> hluvhlvimq: rot=180 -> (0,-1) -> moves (0,-6) = to (37,43)
# 
# When OPENED: D1 at (13,25), D2 at (37,43)
# When CLOSED: D1 at (13,19), D2 at (37,49)

# Let me try ALL combinations:
configs = [
    ("Both closed", [(13,19), (37,49)]),
    ("D1 open, D2 closed", [(13,25), (37,49)]),
    ("D1 closed, D2 open", [(13,19), (37,43)]),
    ("Both open", [(13,25), (37,43)]),
]

for name, doors in configs:
    reachable = bfs_reachable(49, 25, doors)
    has_exit = (25,19) in reachable
    has_b1 = (13,37) in reachable
    has_b2 = (37,25) in reachable
    print(f"{name}: exit={has_exit}, B1={has_b1}, B2={has_b2}, cells={len(reachable)}")
    
    # Show map
    for y in range(7, 56, 6):
        line = f"  y={y:2d}: "
        for x in range(7, 56, 6):
            if (x,y) in reachable:
                if (x,y) == (49,25): line += " P "
                elif (x,y) == (25,19): line += " E "
                elif (x,y) == (37,25): line += "B2 "
                elif (x,y) == (13,37): line += "B1 "
                else: line += " . "
            else:
                line += " # "
        print(line)
    print()

# Hmm, the exit check might be wrong. Win condition is safkknjslo:
# exit.x + 1 == player.x AND exit.y + 1 == player.y
# exit at (24,18), so player needs (25,19)
# But wait, maybe the exit position in the grid isn't exactly aligned?
# Exit is 9x9 at (24,18). Center at (28,22). 
# The player needs to overlap with the exit.
# safkknjslo: self.whftgckbcu.x + 1 == self.dzxunlkwxt.x and self.whftgckbcu.y + 1 == self.dzxunlkwxt.y
# exit.x=24, exit.y=18. So player needs x=25, y=19. That IS (25,19).

# Why can't we reach (25,19)?
# From (7,7): RIGHT to (13,7) ok, (19,7) ok, (25,7) ok
# From (25,7): DOWN to (25,13) - check room
cx, cy = 25+3, 13+3
print(f"Room at (25,13): {room.esidlbhbhw(cx, cy)}")
cx, cy = 25+3, 19+3  
print(f"Room at (25,19): {room.esidlbhbhw(cx, cy)}")
# Even (19,13):
cx, cy = 19+3, 13+3
print(f"Room at (19,13): {room.esidlbhbhw(cx, cy)}")

# (25,13) is blocked by room pixels. The corridor from top to exit is not direct.
# The exit area seems only accessible from left side at y=19 level
# (7,19) -> RIGHT -> (13,19) -> (19,19) -> (25,19)
# But (13,19) is where door D1 is, and (7,19) is blocked by D1 even when open

# Wait let me check more carefully. Maybe player x=19, y=19 is accessible from (19,7)?
# (19,7) -> DOWN -> (19,13): room pixel?
cx, cy = 19+3, 13+3
print(f"Room at (19,13): {room.esidlbhbhw(cx, cy)}")
# It's -1 (blocked). So we can't go down from row 7 to row 19 at x=19.

# What about x=7? (7,7) -> (7,13) -> (7,19) -> (7,25)?
# (7,13): room open, but D1 at (13,19) blocks (7,13)?
# D1 door at (13,19): |7-13|=6 < 7, |13-19|=6 < 7 -> YES BLOCKS
# So (7,13) is blocked by D1 when D1 is at (13,19).
# When D1 opens to (13,25): |7-13|=6 < 7, |13-25|=12 >= 7 -> NOT blocked
# So when D1 is open: (7,13) is accessible!
# But D1 at (13,25) blocks (7,19)? |7-13|=6 < 7, |19-25|=6 < 7 -> YES blocks!
# So (7,19) is ALWAYS blocked by D1 regardless of state!

# Hmm but (7,19) has room floor. What if we approach from (13,19)?
# When D1 opens: D1 at (13,25). (13,19): |13-13|=0 < 7, |19-25|=6 < 7 -> BLOCKED
# Still blocked. (19,19): |19-13|=6 < 7, |19-25|=6 < 7 -> BLOCKED
# So (19,19) is also blocked by D1!

# This means: when D1 is at (13,19), blocks: (7..19, 13..25)
# When D1 is at (13,25), blocks: (7..19, 19..31)
# D1 blocks an entire corridor in both states!

# But wait - maybe D1 needs to move MORE than 6 pixels?
# hluvhlvimq returns (0,1) for rot=0. Then dx*jarvstobjt = 0*6=0, dy*1*6=6.
# So it moves 6 pixels down. That's correct.

# Is there a way to get D1 to be in a state where the exit path is clear?
# Maybe the yellow toggle changes behavior? D1 is not yellow (rot=0, not xiibindvfw).

# Let me look at this differently. Maybe the exit is reached from a different direction.
# From row 37: (13,37) is B1. (19,37) open. (25,37) open.
# From (25,37): UP to (25,31) - check room
cx, cy = 25+3, 31+3
print(f"Room at (25,31): {room.esidlbhbhw(cx, cy)}")
# (25,31): blocked
cx, cy = 25+3, 25+3
print(f"Room at (25,25): {room.esidlbhbhw(cx, cy)}")
# Also blocked

# How about from (25,43)?
cx, cy = 25+3, 43+3
print(f"Room at (25,43): {room.esidlbhbhw(cx, cy)}")
# Let me check again with the high-res room map
print("\nDetailed room around exit area:")
rs = game.current_level.get_sprites_by_tag("rsrdfsruqh")[0]
import numpy as np
pixels = np.array(rs.pixels)
for y in range(10, 20):
    line = f"  y={y+7:2d}: "
    for x in range(10, 30):
        val = pixels[y, x]
        if val == -1:
            line += "#"
        else:
            line += "."
    print(line)

