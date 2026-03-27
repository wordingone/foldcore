"""Debug G50T L2 - render the room sprite to see walkable area."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
import numpy as np

game = G50t()
game.set_level(1)

# The room sprite is rsrdfsruqh  
level = game.current_level
room_sprites = level.get_sprites_by_tag("rsrdfsruqh")
print(f"Room sprites: {len(room_sprites)}")
for rs in room_sprites:
    print(f"  Room at ({rs.x},{rs.y}) size={rs.width}x{rs.height}")
    pixels = np.array(rs.pixels)
    print(f"  Pixel shape: {pixels.shape}")
    print(f"  Unique values: {np.unique(pixels)}")
    
    # Print the room as a grid showing which cells are walkable
    # Scale down by showing every 6th pixel (step size)
    print("  Room layout (each char = roughly 6x6 block):")
    for y in range(0, pixels.shape[0], 3):
        row = ""
        for x in range(0, pixels.shape[1], 3):
            val = pixels[y,x]
            if val == -1:
                row += " "  # transparent/no floor
            elif val == 0:
                row += "#"  # black/wall
            else:
                row += "."  # floor
        print(f"    {row}")

# Also check the movement validation more carefully
# esidlbhbhw checks if a point has a non-(-1) pixel
# Let me check what pixels are at specific grid positions
gs = game.vgwycxsxjz
room = gs.afbbgvkpip  # the room wrapper

# Check specific positions the player would move through
# Player center = player.x + 3, player.y + 3 (since player is 7x7)
test_positions = [
    (49+3,25+3, "Start"),
    (43+3,25+3, "1L from start"),
    (37+3,25+3, "2L from start (gate)"),
    (49+3,19+3, "1U from start"),
    (49+3,13+3, "2U from start"),
    (43+3,13+3, "2U1L"),
    (43+3,7+3, "3U1L"),
    (37+3,7+3, "3U2L"),
    (31+3,7+3, "3U3L"),
    (25+3,7+3, "3U4L"),
    (19+3,7+3, "3U5L"),
    (13+3,7+3, "3U6L"),
    (7+3,7+3, "3U7L"),
    (7+3,13+3, "corner test"),
    (13+3,19+3, "trigger 1"),
    (37+3,49+3, "trigger 2"),
    (25+3,19+3, "exit area"),
]

print("\nPosition check (center pixel in room):")
for x, y, label in test_positions:
    rx, ry = x - room.x, y - room.y
    if 0 <= rx < room.width and 0 <= ry < room.height:
        val = room.esidlbhbhw(x, y)
        print(f"  ({x},{y}) [{label}]: pixel={val}")
    else:
        print(f"  ({x},{y}) [{label}]: OUT OF BOUNDS")

# Let's also check what the room pixel map looks like at key positions
print("\nRoom pixels at y=16 (2U row), x scan:")
rs = room_sprites[0]
pixels = np.array(rs.pixels)
y_offset = 16 - rs.y  # = 16 - 7 = 9
if 0 <= y_offset < pixels.shape[0]:
    for x in range(0, pixels.shape[1], 3):
        val = pixels[y_offset, x]
        print(f"  x={x+rs.x}: {val}", end="")
    print()

