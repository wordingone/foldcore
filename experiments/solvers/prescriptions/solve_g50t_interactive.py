"""Interactive G50T solver - test action sequences level by level."""
import sys, os, json
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction, jarvstobjt
from arcengine import ActionInput

UP = GameAction.ACTION1     # dy=-1 -> y-6
DOWN = GameAction.ACTION2   # dy=+1 -> y+6
LEFT = GameAction.ACTION3   # dx=-1 -> x-6
RIGHT = GameAction.ACTION4  # dx=+1 -> x+6
RECORD = GameAction.ACTION5

STEP = jarvstobjt  # 6

action_names = {UP: 'UP', DOWN: 'DOWN', LEFT: 'LEFT', RIGHT: 'RIGHT', RECORD: 'RECORD'}

def try_solve(game, lvl_idx, actions_list, label=""):
    """Try to solve a level with given actions. Returns action count or -1."""
    game.set_level(lvl_idx)
    gs = game.vgwycxsxjz
    player = gs.dzxunlkwxt
    
    total = 0
    for action, count in actions_list:
        for _ in range(count):
            old_lvl = game.level_index
            game.perform_action(ActionInput(id=action))
            total += 1
            if game.level_index != old_lvl:
                print(f"  {label}L{lvl_idx+1}: SOLVED in {total} actions")
                return total
    
    # Pump animations
    for _ in range(300):
        old_lvl = game.level_index
        game.perform_action(ActionInput(id=UP))
        if game.level_index != old_lvl:
            print(f"  {label}L{lvl_idx+1}: SOLVED in {total} actions (+ animation)")
            return total
    
    print(f"  {label}L{lvl_idx+1}: FAILED after {total} actions")
    # Print player position
    gs = game.vgwycxsxjz
    player = gs.dzxunlkwxt
    print(f"    Player at ({player.x}, {player.y}), exit at ({gs.whftgckbcu.x}, {gs.whftgckbcu.y})")
    return -1

def debug_play(game, lvl_idx, actions_list):
    """Play with debug output showing player position after each move."""
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
            print(f"  [{total}] {aname}: player=({player.x},{player.y}) level={game.level_index}")
            if game.level_index != old_lvl:
                print(f"  LEVEL COMPLETE!")
                return total
    return total

game = G50t()

# L1: Known solution - RIGHT*4, RECORD, DOWN*7, RIGHT*5 = 17 actions
try_solve(game, 0, [(RIGHT,4),(RECORD,1),(DOWN,7),(RIGHT,5)], "")

# L2: Player at (49,25), Exit at (24,18)
# Triggers: (13,19) rot=0, (37,49) rot=180  
# Gates: (13,37) rot=0, (37,25) rot=0
# Gate rot=0 means it moves DOWN when triggered (dy=+1 -> y+6)
# 
# Gate at (37,25) is in the path from player to exit
# Gate at (13,37) is below
#
# Trigger at (37,49) rot=180 -> trigger opens gate... need to check which gate connects
# Trigger at (13,19) rot=0 -> trigger opens gate...
#
# The connection is through path markers. Let me check the path marker overlay.
# Path at (16,22) 1x19 - vertical column from y=22 to y=40
# Path at (40,28) 1x25 - vertical column from y=28 to y=52
#
# From L1 logic: player goes RIGHT, records, clone replays to hold trigger
# L2: 2 triggers, need 2 clones (but only 3 record slots total and we need one slot free)
#
# Actually, record slots: L2 has 3 slots. Slot 0 is active at start.
# First RECORD: creates clone #1 from slot 0, activates slot 1
# Second RECORD: creates clone #2 from slot 1, activates slot 2  
# Then player moves freely (slot 2 is just the "play" slot)
#
# Strategy for L2:
# Player at (49,25). Exit at (24,18). 
# Gate at (37,25) blocks LEFT path. Need trigger at... 
# Let me trace: trigger->path->gate connections
# 
# Move player LEFT to (37,25) - blocked by gate
# Actually, gate at (37,25) rot=0 means when opened it moves DOWN by 6 -> (37,31)
# When triggered, gate moves; when untriggered it moves back
#
# Let me think step-by-step:
# Player at (49,25). Move LEFT -> (43,25), (37,25) blocked by gate
# Need to get a clone on trigger at (37,49) or (13,19)
#
# First recording: go DOWN from (49,25) to reach trigger at (37,49)?
# (49,25) -> DOWN -> (49,31) -> DOWN -> (49,37) -> DOWN -> (49,43) -> DOWN -> (49,49)
# Then LEFT to (43,49) -> (37,49) -- that's the trigger!
# So: DOWN*4, LEFT*2, RECORD
# Clone replays: DOWN*4, LEFT*2, sits on trigger (37,49)
# This should open gate at (37,25) (moving it DOWN to 31)
#
# After recording, player rewinds to start (49,25) then moves:
# LEFT to get past gate (37,25) which is now open -> (43,25), (37,25)?
# Wait, gate moved from (37,25) to (37,31), so (37,25) is clear
# LEFT -> (43,25) -> (37,25) -> (31,25) -> (25,25) -> (19,25) -> (13,25)
# Then UP -> (13,19) -- that's the other trigger!
# But we need to get to exit at (24,18)
#
# Hmm wait. Let me reconsider. Exit is at (24,18). Player needs to reach (25,19) 
# since win condition is player at exit+1,exit+1 (exit width 9)
# Actually safkknjslo: exit.x+1 == player.x and exit.y+1 == player.y
# So player needs to be at (25, 19)
#
# From (49,25):
# Recording 1: move DOWN*4, LEFT*2 to trigger (37,49)
# After clone created, player back at (49,25)
# Now gate (37,25) is open (clone on trigger)
# Move LEFT: (43,25) -> (37,25) should be passable -> (31,25) -> (25,25) -> (19,25)
# Move UP: (19,19)
# But exit is at (24,18), player needs (25,19)
# So from start: after recording, move LEFT*4 to (25,25), UP*1 to (25,19)
# That's (25,19) = exit (24,18) + (1,1)! 
#
# Wait: but which gate does trigger at (37,49) control?
# Let me verify by checking path connections

print("\n=== L2 Debug ===")
# Test: DOWN*4, LEFT*2, RECORD, LEFT*4, UP*1
debug_play(game, 1, [(DOWN,4),(LEFT,2),(RECORD,1)])

