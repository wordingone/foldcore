"""Debug G50T L2 - understand movement constraints with obstacles."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction, jarvstobjt
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5
action_names = {UP:'UP', DOWN:'DOWN', LEFT:'LEFT', RIGHT:'RIGHT', RECORD:'RECORD'}

def dp(game, lvl, actions):
    game.set_level(lvl)
    gs = game.vgwycxsxjz
    p = gs.dzxunlkwxt
    print(f"  Start: ({p.x},{p.y})")
    for i,(a,c) in enumerate(actions):
        for j in range(c):
            old_lvl = game.level_index
            game.perform_action(ActionInput(id=a))
            p = game.vgwycxsxjz.dzxunlkwxt
            print(f"  {action_names[a]}: ({p.x},{p.y})")
            if game.level_index != old_lvl:
                print(f"  COMPLETE!")
                return
    return

game = G50t()

# L2 obstacles:
# Wall at (39,32) 3x17 = x:39-41, y:32-48
# Wall at (15,26) 3x11 = x:15-17, y:26-36
# Gate (37,25) 7x7 = x:37-43, y:25-31 (blocks LEFT movement)
# Gate (13,37) 7x7 = x:13-19, y:37-43 
# Trigger (13,19) 7x7 = x:13-19, y:19-25
# Trigger (37,49) 7x7 = x:37-43, y:49-55
# Exit (24,18) 9x9 = x:24-32, y:18-26

# The player (7x7) at position P means pixels at P to P+6
# Movement check: rhvduhvfwn checks if target pos is inside the room area (rsrdfsruqh)
# AND not blocked by uwxkstolmf (triggers when visible)
# vjpujwqrto checks collision with triggers

# Wait, triggers BLOCK movement? Let me re-read vjpujwqrto...
# vjpujwqrto returns True if player overlaps with any VISIBLE trigger
# rdgrjozeoh: if rhvduhvfwn returns False (target NOT in area OR blocked by trigger), move fails
# So triggers block player movement!

# From (43,49): player (43,49)-(49,55). Moving LEFT to (37,49): 
# Check trigger at (37,49)-(43,55). Player at (37,49) would be ON the trigger.
# vjpujwqrto would return True since player overlaps trigger.
# So the move is blocked!

# This means the clone mechanism has a DIFFERENT rule for ending up on triggers.
# Actually wait - let me re-read. The clones use the same move function rdgrjozeoh.
# But the clone IS the player until it becomes a clone after RECORD.
# Actually - the enemies/roamers (kgvnkyaimw) have different collision rules.

# Let me check: can the player move ONTO a trigger from a different approach?
# From above: player at (37,43) -> DOWN to (37,49)?

print("L2 - Try approach trigger from above")
dp(game, 1, [(LEFT,2),(DOWN,3)])
# (49,25) -> LEFT -> (43,25) -> LEFT -> blocked by gate at (37,25)
# Player stuck at (37,25)

# Hmm. Let me try a completely different approach.
# Gate (37,25) - when triggered it moves DOWN 6 pixels to (37,31)
# If the trigger at (37,49) controls this gate...
# But player can't reach trigger (37,49) directly.

# What if the player goes UP first, then records a path DOWN?
print("\nL2 - Go UP then explore")
dp(game, 1, [(UP,3),(LEFT,7)])

# Player at (49,7). Going LEFT...
# Exit at (24,18). 
# The room area center-ish. 

# L2 is 49x49 room from (7,7). 
# Trigger at (13,19) needs someone on it to open gate at (13,37)
# Trigger at (37,49) needs someone on it to open gate at (37,25)

# But can the player step on a trigger?
# Let me try reaching trigger (13,19) from different angles
print("\nL2 - Try reaching trigger (13,19)")
dp(game, 1, [(UP,2),(LEFT,6)])
# From (49,25) up 2 = (49,13), left 6 should go to (13,13)... if room allows
# Then DOWN should reach (13,19)

print("\nL2 - UP*2, LEFT*6, DOWN*1")
dp(game, 1, [(UP,2),(LEFT,6),(DOWN,1)])

