"""Map G50T L2 room layout precisely with all objects."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
import numpy as np

game = G50t()
game.set_level(1)

rs = game.current_level.get_sprites_by_tag("rsrdfsruqh")[0]
pixels = np.array(rs.pixels)

# Create a map where each cell is a 6x6 block
# Room starts at (7,7), step=6, so grid positions are:
# (7,7), (13,7), (19,7), (25,7), (31,7), (37,7), (43,7), (49,7)
# Grid row = (y-7)/6, grid col = (x-7)/6

print("L2 Grid Map (6x6 blocks, room starts at 7,7):")
print("  Cols:  7  13  19  25  31  37  43  49")
print("        (0) (1) (2) (3) (4) (5) (6) (7)")

for row in range(8):
    y = 7 + row * 6
    line = f"  y={y:2d} ({row}): "
    for col in range(8):
        x = 7 + col * 6
        # Check center of 7x7 player at this position
        cx, cy = x + 3, y + 3
        rx, ry = cx - rs.x, cy - rs.y
        if 0 <= rx < rs.width and 0 <= ry < rs.height:
            val = pixels[ry, rx]
            if val == -1:
                line += "  X "
            else:
                line += "  . "
        else:
            line += "  O "
    print(line)

# Now overlay objects
print("\nObjects:")
print("  P = Player start (49,25) -> grid (7,3)")
print("  E = Exit (24,18) -> grid (~2.8, 1.8)")
print("  T1 = Trigger (13,19) -> grid (1,2)")
print("  T2 = Trigger (37,49) -> grid (5,7)")
print("  G1 = Gate (13,37) -> grid (1,5)")
print("  G2 = Gate (37,25) -> grid (5,3)")

# Which cells can the player actually reach?
# Let me do a BFS
room_obj = game.vgwycxsxjz.afbbgvkpip

def can_reach(x, y):
    """Check if player center at (x+3, y+3) is on valid floor"""
    cx, cy = x + 3, y + 3
    return room_obj.esidlbhbhw(cx, cy) != -1

# Check all grid positions
print("\nReachable grid positions:")
for row in range(8):
    y = 7 + row * 6
    line = f"  y={y:2d}: "
    for col in range(8):
        x = 7 + col * 6
        if can_reach(x, y):
            # Label special positions
            if (x,y) == (49,25): line += " P "
            elif (x,y) == (13,19): line += " T1"
            elif (x,y) == (37,49): line += " T2"
            elif (x,y) == (13,37): line += " G1"
            elif (x,y) == (37,25): line += " G2"
            elif (x,y) == (25,19): line += " E "
            else: line += " . "
        else:
            line += " # "
    print(line)

