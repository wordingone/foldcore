import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()

# L2: Let me watch clone behavior step by step
game.set_level(1)
gs = game.vgwycxsxjz

# Record a short path first
for a in [UP, UP, UP, LEFT, LEFT, DOWN, DOWN, DOWN]:  # 8 moves to button B2
    game.perform_action(ActionInput(id=a))

p = gs.dzxunlkwxt
print(f"Before RECORD: player at ({p.x},{p.y})")
print(f"  areahjypvy (recorded moves): {gs.areahjypvy}")

# RECORD
game.perform_action(ActionInput(id=RECORD))
p = gs.dzxunlkwxt
print(f"After RECORD: player at ({p.x},{p.y})")
print(f"  Clones (rloltuowth): {len(gs.rloltuowth)}")
print(f"  uocsatwnyt (saved path): {gs.uocsatwnyt}")
print(f"  record slot: {gs.rlazdofsxb}")
print(f"  dofntsemri (rewinding): {gs.dofntsemri}")
print(f"  pohkooyzds: {gs.pohkooyzds}")
print(f"  areahjypvy: {gs.areahjypvy}")

# Now step through and watch clone appear
for i in range(20):
    game.perform_action(ActionInput(id=RIGHT))
    p = gs.dzxunlkwxt
    clones = gs.rloltuowth
    clone_info = []
    for c, path in clones.items():
        clone_info.append(f"clone at ({c.x},{c.y})")
    doors_info = [f"({d.x},{d.y})" for d in gs.uwxkstolmf]
    print(f"  Step {i+1}: player=({p.x},{p.y}) clones={clone_info} doors={doors_info} anim={gs.jqpwhiraaj}")
    if game.level_index != 1:
        print("  LEVEL CHANGED!")
        break

