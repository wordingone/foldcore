import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5
an = {UP:'U', DOWN:'D', LEFT:'L', RIGHT:'R', RECORD:'REC'}

game = G50t()

def dp(game, lvl, actions):
    game.set_level(lvl)
    gs = game.vgwycxsxjz
    p = gs.dzxunlkwxt
    total = 0
    print(f"  Start: ({p.x},{p.y})")
    for a,c in actions:
        for _ in range(c):
            old = game.level_index
            game.perform_action(ActionInput(id=a))
            total += 1
            p = gs.dzxunlkwxt
            clones = [(c.x,c.y) for c in gs.rloltuowth]
            doors = [(d.x,d.y) for d in gs.uwxkstolmf]
            print(f"  [{total:2d}] {an[a]:>3s}: player=({p.x},{p.y}) clones={clones} doors={doors}")
            if game.level_index != old:
                print(f"  COMPLETE in {total} actions!")
                return total
    return total

# L2 strategy:
# Recording 1: path to B2 at (37,25)
#   UP*3, LEFT*2, DOWN*3 = 8 moves
# Recording 2: path to B1 at (13,37) 
#   UP*3, LEFT*7, DOWN*5, RIGHT*1 = 16 moves
#
# After both recordings:
# Clone1 (B2 path, 8 moves): reaches B2 at player step 8
# Clone2 (B1 path, 16 moves): reaches B1 at player step 16
#
# Player needs to make 16 successful moves.
# After step 8: D2 opens. After step 16: D1 opens.
# Player then needs to reach (25,19).
#
# Player route (16 successful moves):
# From (49,25): UP*3 to (49,7), LEFT*7 to (7,7), DOWN*1 to (7,13), 
# then need 5 more moves... 
# DOWN to (7,19), then at step 13 door D1 is still closed.
# Can go DOWN more? (7,25) is blocked. So bounce: UP*3 to (7,7)
# Wait, that only gets us to step 12+3 = 15
# Then need 1 more... DOWN to (7,13) = step 16
# Now D1 opens! Door D1 was at (13,19) moves DOWN to (13,25)
# Then RIGHT from (7,13) to (13,13) - but (13,13) is BLOCKED in room!
#
# Hmm. The door is at (13,19) and blocks movement. When it opens,
# it moves to (13,25). So (13,19) is now clear.
# But player at (7,13) going RIGHT to (13,13): (13,13) is blocked room pixel!
#
# Different approach: player should go to (7,19), then after D1 opens,
# go RIGHT from (7,19) to (13,19) which is now clear.
# But need 16 moves before door opens at step 16.
#
# Route: UP*3(49,7), LEFT*7(7,7), DOWN*2(7,19) = 12 moves
# Need 4 more: RIGHT*(7,19): can go to (13,19)? D1 blocks! Not yet.
# Bounce: UP*2(7,7), DOWN*2(7,19) = 4 more moves = step 16
# Now D1 opens! Player at (7,19)
# Then RIGHT*3 to (25,19) = exit!

# Let me test:
print("L2 approach: REC1(B2), REC2(B1), then navigate")
# Recording 1: to B2
game.set_level(1)
gs = game.vgwycxsxjz
for a in [UP]*3 + [LEFT]*2 + [DOWN]*3:
    game.perform_action(ActionInput(id=a))
p = gs.dzxunlkwxt
print(f"  After path to B2: ({p.x},{p.y})")
game.perform_action(ActionInput(id=RECORD))
p = gs.dzxunlkwxt
print(f"  After REC1: ({p.x},{p.y}), clones={len(gs.rloltuowth)}")

# Recording 2: to B1
for a in [UP]*3 + [LEFT]*7 + [DOWN]*5 + [RIGHT]*1:
    game.perform_action(InputAction := ActionInput(id=a))
p = gs.dzxunlkwxt
print(f"  After path to B1: ({p.x},{p.y})")

# Check: can the player reach (13,37) via this path?
# (49,25) UP*3 -> (49,7), LEFT*7 -> (7,7), DOWN*5 -> (7,37), RIGHT -> (13,37)
# (7,25) is BLOCKED, (7,31) is BLOCKED. Can't go DOWN*5 from (7,7)!
# (7,13) is open, (7,19) is open, (7,25) blocked.
# So path from (7,7) going DOWN: (7,7)->(7,13)->(7,19) then stuck.

# Need different path to B1!
# From (7,19): can go RIGHT to (13,19)? That's the door D1.
# Door D1 is in uwxkstolmf (blockers). So NO.
# From (7,19): DOWN to (7,25) is blocked.
# From (7,37): how to get there? Need to go from (7,7) to (7,37) without going through blocked areas.
# 
# Alternative: go via bottom. From (49,25): DOWN*4 to (49,49), LEFT*7 to (7,49),
# UP*2 to (7,37), RIGHT to (13,37).
# Path: DOWN*4, LEFT*7, UP*2, RIGHT*1 = 14 moves

# Let me restart and try
print("\n\nRetry with different B1 path:")
game.set_level(1)
gs = game.vgwycxsxjz

# Recording 1 (shorter): to B2 = UP*3, LEFT*2, DOWN*3 = 8 moves
for a in [UP]*3 + [LEFT]*2 + [DOWN]*3:
    game.perform_action(ActionInput(id=a))
print(f"  At B2: ({gs.dzxunlkwxt.x},{gs.dzxunlkwxt.y})")
game.perform_action(ActionInput(id=RECORD))
print(f"  After REC1: ({gs.dzxunlkwxt.x},{gs.dzxunlkwxt.y})")

# Recording 2 (longer): to B1 via bottom = DOWN*4, LEFT*7, UP*2, RIGHT*1 = 14 moves
# But wait: from (49,25) DOWN*4 -> check reachability
# (49,31): open. (49,37): open. (49,43): open. (49,49): open. Good.
# LEFT*7 from (49,49): (43,49)->(37,49): blocked by door D2!
# D2 at (37,49) is a door/blocker. Can't pass.
# 
# Hmm. But clone1 is replaying toward B2. At what step does D2 open?
# Clone1 path is 8 moves. Clone1 replays alongside player's recording2 moves.
# At step 8 of recording2, clone1 reaches B2, D2 opens.
# But we need to go through D2 position before step 8 finishes recording.
# That won't work.

# Different approach entirely:
# Maybe record B1 first (go to bottom), THEN record B2 (go to right side)
# Recording 1: path to B1
#   DOWN*4 to (49,49), LEFT*7 to (7,49), UP*2 to (7,37), RIGHT to (13,37) = 14 moves
#   But LEFT*7 from (49,49): (43,49) ok, (37,49) has door D2 blocking...

# Actually let me check: is D2 initially at (37,49)?
# And the room pixel at (37,49) - is it walkable if the door weren't there?
game.set_level(1)
gs = game.vgwycxsxjz
room = gs.afbbgvkpip
cx, cy = 37+3, 49+3
val = room.esidlbhbhw(cx, cy)
print(f"\n  Room pixel at (37,49): {'OPEN' if val != -1 else 'BLOCKED'}")

# And at bottom row y=49:
for x in range(7, 56, 6):
    cx, cy = x+3, 49+3
    val = room.esidlbhbhw(cx, cy)
    is_door = any(d.x == x and d.y == 49 for d in gs.uwxkstolmf)
    blocked_by = "DOOR" if is_door else ""
    status = 'OPEN' if val != -1 else 'BLOCKED'
    print(f"  ({x},49): {status} {blocked_by}")

# So the door at (37,49) occupies a cell that the room shows as open.
# The door BLOCKS because vjpujwqrto checks uwxkstolmf for collision.
# If the door moves away, the room cell is passable.

# So to reach B1 via bottom, we need D2 to be open first.
# D2 opens when B2 is pressed. 
# If we record path to B2 first, clone1 replays during recording2.
# During recording2 moves, clone1 advances alongside.
# At recording2 move 8, clone1 finishes B2 path -> D2 opens.
# So if recording2 reaches (37,49) at move 9+, D2 is already open!

# Recording 1: B2 path = UP*3, LEFT*2, DOWN*3 = 8 moves
# Recording 2: B1 path needs to pass through (37,49) after 8 moves
#   Move 1-8: go to some position near (37,49)
#   At step 8: clone1 is on B2, D2 opens
#   Step 9+: pass through (37,49)
#
# Recording 2 path from (49,25):
#   DOWN*4 to (49,49) = 4 moves (steps 1-4)
#   LEFT to (43,49) = step 5
#   Need to wait for step 8... 
#   But we need to make SUCCESSFUL moves!
#   Go: UP to (43,43)? Check if passable.

game.set_level(1)
gs = game.vgwycxsxjz
# Check (43,43): 
cx, cy = 43+3, 43+3
val = room.esidlbhbhw(cx, cy)
print(f"\n  Room pixel at (43,43): {'OPEN' if val != -1 else 'BLOCKED'}")
cx, cy = 43+3, 37+3
val = room.esidlbhbhw(cx, cy)
print(f"  Room pixel at (43,37): {'OPEN' if val != -1 else 'BLOCKED'}")

