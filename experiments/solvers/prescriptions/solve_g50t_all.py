import sys, os, json
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction, jarvstobjt
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5
an = {UP:'U', DOWN:'D', LEFT:'L', RIGHT:'R', RECORD:'REC'}

def play(game, actions, verbose=False):
    p = game.vgwycxsxjz.dzxunlkwxt
    if verbose: print(f"  ({p.x},{p.y})", end="")
    total = 0
    for a,c in actions:
        for _ in range(c):
            old = game.level_index
            game.perform_action(ActionInput(id=a))
            total += 1
            p = game.vgwycxsxjz.dzxunlkwxt
            if verbose: print(f" {an[a]}>({p.x},{p.y})", end="")
            if game.level_index != old:
                if verbose: print(" WIN!")
                return total, True
    if verbose: print()
    return total, False

def pump_win(game, old_level, max_steps=300):
    for _ in range(max_steps):
        game.perform_action(ActionInput(id=UP))
        if game.level_index != old_level:
            return True
    return False

def solve_level(game, actions, label, verbose=True):
    old_level = game.level_index
    n, won = play(game, actions, verbose)
    if not won:
        won = pump_win(game, old_level)
    if won:
        print(f"  {label}: SOLVED in {n} actions")
    else:
        print(f"  {label}: FAILED")
        p = game.vgwycxsxjz.dzxunlkwxt
        ex = game.vgwycxsxjz.whftgckbcu
        print(f"    Player at ({p.x},{p.y}), exit at ({ex.x},{ex.y})")
    return n if won else -1

def check_doors(game):
    gs = game.vgwycxsxjz
    for t in gs.uwxkstolmf:
        print(f"    Door at ({t.x},{t.y})")

game = G50t()

# ===================== L1 =====================
# Known solution: RIGHT*4, RECORD, DOWN*7, RIGHT*5 = 17
print("=== L1 ===")
n1 = solve_level(game, [(RIGHT,4),(RECORD,1),(DOWN,7),(RIGHT,5)], "L1")

# ===================== L2 =====================
# Grid map (step 6, origin 7,7):
#   y\x  7  13  19  25  31  37  43  49
#    7:  .   .   .   .   .   .   .   .   <- top corridor
#   13:  .   #   #   #   #   .   #   .
#   19:  .  D1   .   E   #   .   #   .   <- D1=door at (13,19)
#   25:  #   #   #   #   #  B2   .   P   <- B2=button at (37,25), P=player
#   31:  #   #   #   #   #   #   #   .
#   37:  .  B1   .   .   #   #   #   .   <- B1=button at (13,37)
#   43:  #   #   #   .   #   #   #   .
#   49:  .   .   .   .   .  D2   .   .   <- D2=door at (37,49)
#
# Button B2 at (37,25) controls Door D2 at (37,49) rot=180 -> UP by 6 to (37,43)
# Button B1 at (13,37) controls Door D1 at (13,19) rot=0 -> DOWN by 6 to (13,25)
#
# Player at (49,25). Exit at (24,18) -> player needs to be at (25,19).
#
# Strategy:
# 1. Move UP to top corridor: (49,25) -> UP*3 -> (49,7)
# 2. Move LEFT along top: (49,7) -> LEFT*2 -> (37,7)
# 3. Move DOWN to button B2: (37,7) -> DOWN*3 -> (37,25)
# 4. This activates button B2! Door D2 at (37,49) moves UP to (37,43)
# 5. RECORD to save path and create clone on button B2
# 6. Player returns to start (49,25)? No - player returns to (49,7)? 
#    Actually RECORD rewinds to the START of recording. But the recording starts
#    from when the player entered the level. The areahjypvy tracks ALL moves.
#    So RECORD rewinds ALL the way back to the original start position.
#    Then clone replays ALL moves.
#
# Let me test this theory:
print("\n=== L2 ===")
game.set_level(1)
print("  Test: UP*3, LEFT*2, DOWN*3 (reach button B2)")
play(game, [(UP,3),(LEFT,2),(DOWN,3)], verbose=True)
p = game.vgwycxsxjz.dzxunlkwxt
print(f"  Player at ({p.x},{p.y})")
check_doors(game)

# Now RECORD
print("  RECORD:")
play(game, [(RECORD,1)], verbose=True)
p = game.vgwycxsxjz.dzxunlkwxt
print(f"  Player at ({p.x},{p.y})")
check_doors(game)

# After record, clone should replay: UP*3, LEFT*2, DOWN*3
# Clone ends up at button B2 (37,25), holding door D2 open
# Player is back at start (49,25)
# Now player needs to reach exit at (25,19)
# Path: UP*3 to (49,7), LEFT*4 to (25,7), DOWN*2 to (25,19) = exit!
print("  Continue: UP*3, LEFT*4, DOWN*2")
n2, won = play(game, [(UP,3),(LEFT,4),(DOWN,2)], verbose=True)
if not won:
    won = pump_win(game, 1)
if won:
    print(f"  L2 total path: UP*3, LEFT*2, DOWN*3, RECORD, UP*3, LEFT*4, DOWN*2")
    print(f"  = 3+2+3+1+3+4+2 = 18 actions")

