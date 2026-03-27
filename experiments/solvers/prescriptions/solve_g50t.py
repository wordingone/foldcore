"""Solve G50T analytically - all 7 levels."""
import sys, os, json
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2  
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
RECORD = GameAction.ACTION5

def act(game, action_id):
    game.perform_action(ActionInput(id=action_id))

def act_seq(game, actions):
    """Execute a sequence of (action, count) tuples"""
    total = 0
    for action, count in actions:
        for _ in range(count):
            act(game, action)
            total += 1
    return total

def wait_animations(game, max_steps=200):
    """Keep stepping until animations resolve"""
    old_level = game.level_index
    for _ in range(max_steps):
        # Check if level changed or game won
        if game.level_index != old_level:
            return True
        # Try a no-op by doing reset action to pump animations
        try:
            act(game, GameAction.RESET)
        except:
            break
    return game.level_index != old_level

def solve_level(game, actions_seq, level_name):
    """Solve a level with given action sequence, return action count"""
    old_level = game.level_index
    total_actions = 0
    
    for action, count in actions_seq:
        for _ in range(count):
            act(game, action)
            total_actions += 1
            # After each action, let animations play out
            while game.vgwycxsxjz.jqpwhiraaj:
                act(game, UP)  # Dummy action while animation plays
            if game.level_index != old_level:
                print(f"  {level_name}: SOLVED in {total_actions} actions")
                return total_actions
    
    # Pump remaining animations
    for _ in range(300):
        if game.level_index != old_level:
            print(f"  {level_name}: SOLVED in {total_actions} actions")
            return total_actions
        try:
            act(game, UP)
        except:
            break
    
    if game.level_index != old_level:
        print(f"  {level_name}: SOLVED in {total_actions} actions")
        return total_actions
    else:
        print(f"  {level_name}: FAILED after {total_actions} actions (still at level {game.level_index})")
        return -1

# G50T game mechanics:
# Player starts at qftsebtxuc position
# Each move = 6 pixels in that direction
# RECORD: saves path, rewinds, creates clone
# Triggers (kjrcloicja): need something standing on them to open gates
# Gates (medyellngi): open/close based on trigger state
# Exit (gilbljmfbc): reach to complete level
# Purple teleporters (mpreboxmgc): connected pairs, teleport player
# jarvstobjt = 6 (step size)

game = G50t()
all_actions = []

# === LEVEL 1 ===
# Player at (13,7), Exit at (42,48)
# Trigger at (13,37) rot=270, Gate at (37,7)
# One record slot available
# Move RIGHT to trigger area, RECORD, then move to exit
# L1 solved: RIGHT*4, RECORD, DOWN*7, RIGHT*5
print("=== G50T Level 1 ===")
l1_actions = [(RIGHT, 4), (RECORD, 1), (DOWN, 7), (RIGHT, 5)]
n = solve_level(game, l1_actions, "L1")
all_actions.append(("L1", n, l1_actions))

# === LEVEL 2 ===  
# Player at (49,25), Exit at (24,18)
# 2 triggers: (13,19) rot=0, (37,49) rot=180
# 2 gates: (13,37), (37,25)
# Clone paths: hxztohfdlx at (16,22) 1x19 and (40,28) 1x25
# 3 record slots (need 2 recordings)
# Strategy: 
#   Player needs to go from (49,25) to exit at (24,18)
#   Gate at (37,25) blocks path to left
#   Gate at (13,37) blocks path down
#   Trigger at (13,19) controls gate - need clone to stand on it
#   Trigger at (37,49) controls another gate
#
# Let me think about grid positions in steps of 6:
# Player at (49,25). In grid-steps from player: 
#   LEFT goes x-6, RIGHT goes x+6, UP goes y-6, DOWN goes y+6
print("\n=== G50T Level 2 ===")
# Let me explore L2 interactively
game2 = G50t()
game2.set_level(1)
# Check initial positions
player = game2.vgwycxsxjz.dzxunlkwxt
print(f"  Player: ({player.x}, {player.y})")
print(f"  Exit: ({game2.vgwycxsxjz.whftgckbcu.x}, {game2.vgwycxsxjz.whftgckbcu.y})")
# Triggers and gates  
for item in game2.vgwycxsxjz.uwxkstolmf:
    print(f"  Trigger: ({item.x}, {item.y}) rot={item.rotation}")
for item in game2.vgwycxsxjz.hamayflsib:
    print(f"  Gate/door: ({item.x}, {item.y}) rot={getattr(item, 'rotation', '?')}")

