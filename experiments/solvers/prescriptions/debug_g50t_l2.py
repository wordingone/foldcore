"""Debug G50T L2 movement constraints."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction, jarvstobjt
from arcengine import ActionInput

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
RECORD = GameAction.ACTION5

action_names = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', RECORD: 'RECORD'}

def debug_play(game, lvl_idx, actions_list):
    game.set_level(lvl_idx)
    gs = game.vgwycxsxjz
    player = gs.dzxunlkwxt
    print(f"  Start: player=({player.x},{player.y})")
    
    total = 0
    for action, count in actions_list:
        for _ in range(count):
            old_lvl = game.level_index
            game.perform_action(ActionInput(id=action))
            total += 1
            gs = game.vgwycxsxjz
            player = gs.dzxunlkwxt
            aname = action_names.get(action, '?')
            print(f"  [{total}] {aname}: player=({player.x},{player.y}) lvl={game.level_index}")
            if game.level_index != old_lvl:
                print(f"  LEVEL COMPLETE!")
                return total
    return total

game = G50t()

# L2: Let me check what's at various positions
# Room is (7,7) to (56,56)
# Player starts at (49,25)
# Let me try moving in different directions
print("=== L2 - exploration ===")
# Try going LEFT from start
print("-- Move LEFT from (49,25)")
debug_play(game, 1, [(LEFT,7)])

print("\n-- Move DOWN then LEFT")
debug_play(game, 1, [(DOWN,5),(LEFT,7)])

print("\n-- Move DOWN*2, LEFT from (49,37)")
debug_play(game, 1, [(DOWN,2),(LEFT,7)])

print("\n-- Move UP from start")
debug_play(game, 1, [(UP,4)])

# Check the gate positions more carefully
# Gate at (37,25) rot=0 - this moves in direction based on rotation
# rot=0 -> (0,1) -> down, then *6 -> moves 6 down
# Gate at (13,37) rot=0 -> same, moves 6 down when triggered

# Triggers:
# (13,19) rot=0 -> opens based on connections through paths
# (37,49) rot=180 -> opens based on connections
# 
# Trigger mechanism: when player/clone stands on trigger location,
# it signals the connected gate to open.
# Connection is via alyzsfkumg (path markers) -> they connect triggers and gates

# The path at (16,22) 1x19 is a vertical line from y=22, height=19, so y=22 to y=40
# The trigger at (13,19) center is at (13+3, 19+3) = (16,22) - that matches the path start!
# The gate at (13,37) center is at (13+3, 37+3) = (16,40) - near end of path (16,22 to 16,40)!
# So trigger (13,19) -> path (16,22) -> gate (13,37)

# The path at (40,28) 1x25 is a vertical line from y=28, height=25, so y=28 to y=52  
# The trigger at (37,49) center is at (37+3, 49+3) = (40,52) - near end of this path!
# The gate at (37,25) center is at (37+3, 25+3) = (40,28) - matches path start!
# So trigger (37,49) -> path (40,28) -> gate (37,25)

# So: trigger (37,49) controls gate (37,25)
# And: trigger (13,19) controls gate (13,37)

# For L2: player needs to reach exit at (24,18)+1 = (25,19)
# Gate (37,25) blocks the path LEFT from player
# Trigger (37,49) opens gate (37,25)
# We need a clone on trigger (37,49)

# But the LEFT from (43,49) to (37,49) failed!
# Let me check - maybe the trigger sprite itself blocks movement
# Triggers are 7x7, at (37,49). The "room" is (7,7) 49x49 = extends to (56,56)
# Can the player stand ON the trigger? The trigger check is in game logic...
# Actually: triggers are 7x7 at position (37,49). They overlap with the trigger sensor area.
# The player (7x7) at (37,49) would be ON the trigger.
# But maybe the player can't move there because something blocks it?

# Let me check - what's the rinmohgkoo sprite?
# It's at (15,26) 3x11 in L2 - that's an obstacle!

print("\n-- Check L2 obstacle positions")
game.set_level(1)
level = game.current_level
for s in level.get_sprites():
    tags = list(s.tags)
    if not tags or tags == ['uxqxubarib'] or tags == ['ppfvilwwnk'] or tags == ['gpkhwmwioo'] or tags == ['ovhuyqtghw']:
        continue
    print(f"  ({s.x},{s.y}) {s.width}x{s.height} tags={tags}")

# Check untagged sprites (they might be obstacles)
print("\n-- Untagged sprites in L2:")
for s in level.get_sprites():
    tags = list(s.tags)
    if not tags:
        print(f"  ({s.x},{s.y}) {s.width}x{s.height}")

