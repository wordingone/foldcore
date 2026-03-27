import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()
game.set_level(1)
gs = game.vgwycxsxjz
room = gs.afbbgvkpip

# Full room map with 3-pixel resolution
print("L2 Full room map (3px resolution, room at (7,7) size 49x49):")
print("     ", end="")
for x in range(7, 56, 3):
    print(f"{x:3d}", end="")
print()
for y in range(7, 56, 3):
    print(f"  {y:2d}: ", end="")
    for x in range(7, 56, 3):
        cx, cy = x+3, y+3
        val = room.esidlbhbhw(cx, cy)
        if val == -1:
            print("  #", end="")
        else:
            print("  .", end="")
    print()

# Check 6-pixel grid positions that overlap with buttons/doors
print("\n\nNavigable 6x6 grid positions (step=6):")
for y in range(7, 56, 6):
    line = f"  y={y:2d}: "
    for x in range(7, 56, 6):
        cx, cy = x+3, y+3
        floor = room.esidlbhbhw(cx, cy) != -1
        is_door = any(abs(d.x-x) < 7 and abs(d.y-y) < 7 for d in gs.uwxkstolmf)
        is_button = any(abs(h.x-x) < 7 and abs(h.y-y) < 7 for h in gs.hamayflsib)
        if not floor:
            line += " # "
        elif is_door:
            line += " D "
        elif is_button:
            line += " B "
        else:
            line += " . "
    print(line)

# The key insight: paths with 6-pixel steps
# Grid positions (x,y) where x and y are on the 6-pixel grid starting from player
# Player starts at (49,25). Each move is 6 pixels.
# Valid x positions from player: ..., 25, 31, 37, 43, 49, ...
# Valid y positions from player: ..., 7, 13, 19, 25, 31, 37, 43, 49, ...

# Wait but L1 player starts at (13,7) and grid is 13+6k, 7+6k
# L2 player starts at (49,25). Grid is 49-6k for x, 25-6k for y
# x: 1, 7, 13, 19, 25, 31, 37, 43, 49
# y: 1, 7, 13, 19, 25, 31, 37, 43, 49
# These all line up! Good.

# So the navigable grid IS the one I printed.
# Now trace paths to B1 at (13,37):
# From bottom: (49,49) -> LEFT through bottom row
# Bottom row y=49: all open, but D2 at (37,49) blocks
# From left: (7,7) -> DOWN -> (7,13) -> (7,19) -> (7,25) blocked
# From (7,37): accessible from (7,49) via UP*2 = (7,43 blocked), hmm

# Check (7,43):
cx, cy = 7+3, 43+3
val = room.esidlbhbhw(cx, cy)
print(f"\n(7,43): {'OPEN' if val != -1 else 'BLOCKED'}")

# Check bottom-left path
for y in [37, 43, 49]:
    for x in [7, 13, 19, 25]:
        cx, cy = x+3, y+3
        val = room.esidlbhbhw(cx, cy)
        print(f"  ({x},{y}): {'O' if val != -1 else '#'}", end="")
    print()

