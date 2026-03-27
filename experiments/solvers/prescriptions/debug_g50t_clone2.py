import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()
game.set_level(1)
gs = game.vgwycxsxjz

# Record path to button: UP*3, LEFT*2, DOWN*3
for a in [UP, UP, UP, LEFT, LEFT, DOWN, DOWN, DOWN]:
    game.perform_action(ActionInput(id=a))

print(f"At button: ({gs.dzxunlkwxt.x},{gs.dzxunlkwxt.y})")

# RECORD
game.perform_action(ActionInput(id=RECORD))
print(f"After RECORD: ({gs.dzxunlkwxt.x},{gs.dzxunlkwxt.y})")
print(f"  Clones: {len(gs.rloltuowth)}")
for c, path in gs.rloltuowth.items():
    print(f"  Clone at ({c.x},{c.y}), path length={len(path)}")

# Now try UP moves - player should be able to go UP from (49,25)
for i in range(12):
    game.perform_action(ActionInput(id=UP))
    p = gs.dzxunlkwxt
    clone_pos = [(c.x,c.y) for c in gs.rloltuowth]
    doors = [(d.x,d.y) for d in gs.uwxkstolmf]
    print(f"  UP #{i+1}: player=({p.x},{p.y}) clone={clone_pos} doors={doors}")
    if game.level_index != 1:
        print("  LEVEL!")
        break

