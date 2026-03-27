import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput

UP, DOWN, LEFT, RIGHT, RECORD = GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5

game = G50t()

# L1: check what medyellngi and kjrcloicja really do
game.set_level(0)
gs = game.vgwycxsxjz

print("L1 initial state:")
print(f"  hamayflsib (buttons/gates): {len(gs.hamayflsib)}")
for h in gs.hamayflsib:
    print(f"    ({h.x},{h.y}) type={type(h).__name__}")
print(f"  uwxkstolmf (doors/blockers): {len(gs.uwxkstolmf)}")
for t in gs.uwxkstolmf:
    print(f"    ({t.x},{t.y}) rot={t.rotation} type={type(t).__name__}")

# Move RIGHT*4 
for _ in range(4):
    game.perform_action(ActionInput(id=RIGHT))

p = gs.dzxunlkwxt
print(f"\nAfter RIGHT*4: player at ({p.x},{p.y})")
print(f"  uwxkstolmf after:")
for t in gs.uwxkstolmf:
    print(f"    door at ({t.x},{t.y})")

# Now for L2
print("\n\nL2 analysis:")
game.set_level(1)
gs = game.vgwycxsxjz
print(f"  hamayflsib: {len(gs.hamayflsib)}")
for h in gs.hamayflsib:
    print(f"    ({h.x},{h.y}) type={type(h).__name__}")
print(f"  uwxkstolmf: {len(gs.uwxkstolmf)}")
for t in gs.uwxkstolmf:
    print(f"    ({t.x},{t.y}) rot={t.rotation} type={type(t).__name__}")
