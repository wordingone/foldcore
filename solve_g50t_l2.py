import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()

# L2 Strategy:
# Record path: UP*3, LEFT*2, DOWN*3 = 8 moves to button B2 at (37,25)
# Clone replays: UP*3, LEFT*2, DOWN*3
# Clone at step 8 reaches (37,25) = button B2
# Button B2 controls door D2 at (37,49) -> door moves UP to (37,43)
# 
# BUT: we need 8 successful player moves after RECORD for clone to finish its path.
# Player starts at (49,25) after RECORD.
#
# Clone path simultaneous with player:
# Player move 1: clone does UP (49,25)->(49,19)
# Player move 2: clone does UP (49,19)->(49,13)
# Player move 3: clone does UP (49,13)->(49,7)
# Player move 4: clone does LEFT (49,7)->(43,7)
# Player move 5: clone does LEFT (43,7)->(37,7)
# Player move 6: clone does DOWN (37,7)->(37,13)
# Player move 7: clone does DOWN (37,13)->(37,19)
# Player move 8: clone does DOWN (37,19)->(37,25) -> BUTTON ACTIVATED
#
# So after 8 player moves, door D2 opens.
# Player needs 8 moves that are valid AND lead toward exit.
# Player at (49,25), exit needs player at (25,19).
# 
# But the door at (13,19) rot=0 might be in the way...
# Door D1 at (13,19) blocks the left area.
# Button B1 at (13,37) controls D1.
#
# Do we need B1? The exit is at (24,18), player needs (25,19).
# (25,19) is in column x=25, which is accessible from the top corridor.
# Door D1 at (13,19) is at x=13, which is left of x=25.
# So we don't need B1 for the exit path!
#
# Player path after RECORD: UP*3, LEFT*4, DOWN*2
# That's 3+4+2 = 9 moves. But clone only has 8 moves.
# After clone finishes its 8 moves, door D2 opens.
# But wait - does the door need to open for the player's path?
#
# Player path: (49,25) UP -> (49,19) UP -> (49,13) UP -> (49,7)
#              LEFT -> (43,7) LEFT -> (37,7) LEFT -> (31,7) LEFT -> (25,7)
#              DOWN -> (25,13) should be blocked (pixel check)
#
# From the grid map, (25,13) is '#' (blocked). Let me check...
# At y=13, x=25: blocked in room pixels.
# The room has a corridor at y=7 (top) and y=19 on the left side.
# (25,13) is in the blocked area.
#
# So player can go from (25,7) DOWN only if there's floor at (25,13).
# But there's no floor there!
#
# Hmm. The exit at (24,18) is 9x9. Does the exit sprite provide floor?
# Actually, the movement check is rhvduhvfwn which checks xvkyljflji:
#   other.esidlbhbhw(player.center_x, player.center_y) != -1
# This checks the ROOM sprite (afbbgvkpip/rsrdfsruqh).
# The room sprite has blocked areas. The exit is separate.
#
# But wait - in L1, the player goes DOWN from (13,7) through (13,49).
# Let me re-examine the L1 room layout...

# Actually, let me just explore the room layout at key positions
game.set_level(1)
gs = game.vgwycxsxjz
room = gs.afbbgvkpip

print("L2 - Check all grid positions for reachability:")
for y in range(7, 56, 6):
    line = f"  y={y:2d}: "
    for x in range(7, 56, 6):
        cx, cy = x+3, y+3
        val = room.esidlbhbhw(cx, cy)
        if val != -1:
            line += f" . "
        else:
            line += f" # "
    print(line)

# Hmm I need a different approach for L2.
# The top corridor at y=7 goes across.
# At y=13, most is blocked except x=7 and x=37,49
# At y=19, x=7,13,19,25 are open, then blocked at 31, then x=37,49
# 
# So from (25,7) going DOWN: (25,13) is blocked.
# From (19,7) going DOWN: let me check (19,13)
print("\nSpecific position checks:")
for x, y in [(19,13), (25,13), (31,13), (7,13), (13,13)]:
    cx, cy = x+3, y+3
    val = room.esidlbhbhw(cx, cy)
    print(f"  ({x},{y}): {'OPEN' if val != -1 else 'BLOCKED'}")

# So the path to exit at (25,19) must go through the left side
# Left corridor: (7,7) -> (7,13) -> (7,19) -> then RIGHT to (13,19) etc.
# But door D1 at (13,19) blocks!
#
# Ugh. So we need BOTH buttons activated.
# B1 at (13,37) controls D1 at (13,19) -> opens by moving DOWN 6
# B2 at (37,25) controls D2 at (37,49) -> opens by moving UP 6
#
# We need two clones: one on B1, one on B2.
# With 3 record slots, we can make 2 recordings.
#
# Strategy:
# Recording 1: go to B1 at (13,37)
#   Player path: from (49,25): LEFT*? can't because of gate... 
#   From top: UP*3 to (49,7), LEFT*7 to (7,7), DOWN*5 to (7,37)
#   Then RIGHT to B1 at (13,37)
#   Total: 3+7+5+1 = 16 moves
#
# Recording 2: go to B2 at (37,25)
#   Player returns to start after recording 1
#   UP*3, LEFT*2, DOWN*3 = 8 moves
#
# After recording 2, clone 1 needs 16 moves, clone 2 needs 8 moves
# Player needs max(16,8)=16 successful moves after second RECORD
# But clone 2 only has 8 moves. Clones stop replaying when their path is exhausted.
# Clone 2 reaches B2 after 8 player moves.
# Clone 1 reaches B1 after 16 player moves.
# Both need to be on their buttons at the same time for both doors to be open.
# Wait - does a clone STAY on the button after reaching it?
# When the path is exhausted, the clone just stays put.
# So clone 2 reaches B2 at move 8 and stays there -> D2 stays open.
# Clone 1 reaches B1 at move 16 and stays there -> D1 opens at move 16.
# After move 16, BOTH doors are open!
# Player needs to be near exit after 16 moves.

# Actually wait. The clones move in PARALLEL with the player.
# Clone 1 path = recording 1 path. Clone 2 path = recording 2 path.
# Clone 1 at step i does path1[i], clone 2 at step i does path2[i].
# So clone 2 (8 moves) reaches B2 at step 8.
# Clone 1 (16 moves) reaches B1 at step 16.
# Player needs to make 16 successful moves.

# But do we know the player can make 16 valid moves after RECORD?
# The player starts at (49,25). The exit is at (25,19).
# Route: UP*3 (49,7), LEFT*7 (7,7), DOWN*2 (7,19), RIGHT*3 (25,19) = 15 moves
# But D1 at (13,19) blocks RIGHT from (7,19) to (13,19) until move 16!
# That's too late.

# Alternative: longer recording 1 path OR different order
# What if we record B2 first (shorter), then B1 (longer)?
# Recording 1: to B2 = UP*3, LEFT*2, DOWN*3 = 8 moves
# Recording 2: to B1 = UP*3, LEFT*7, DOWN*5, RIGHT*1 = 16 moves
# After RECORD 2:
# Clone 1 (B2 path, 8 moves): reaches B2 at step 8
# Clone 2 (B1 path, 16 moves): reaches B1 at step 16
# Player needs 16 successful moves, and after step 16, both doors are open.
# But D1 opens at step 16 when clone 2 reaches B1.
# Player needs to be positioned to reach exit after D1 opens.
# Player route: make 16 moves that end near exit.
# E.g.: UP*3 (49,7), LEFT*7 (7,7), DOWN*2 (7,19), 
# now at step 12, need 4 more. D1 is still closed, can't go RIGHT.
# RIGHT: blocked by D1. Try DOWN: (7,25) is '#'
# Actually (7,25): let me check
for x, y in [(7,25), (7,31), (7,37)]:
    cx, cy = x+3, y+3
    val = room.esidlbhbhw(cx, cy)
    print(f"  ({x},{y}): {'OPEN' if val != -1 else 'BLOCKED'}")

