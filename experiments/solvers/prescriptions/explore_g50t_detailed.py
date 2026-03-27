"""Detailed G50T level exploration."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction, jarvstobjt, xiibindvfw
from arcengine import ActionInput
import numpy as np

UP = GameAction.ACTION1
DOWN = GameAction.ACTION2
LEFT = GameAction.ACTION3
RIGHT = GameAction.ACTION4
RECORD = GameAction.ACTION5

STEP = jarvstobjt  # = 6

def explore_level(game, lvl_idx):
    game.set_level(lvl_idx)
    gs = game.vgwycxsxjz  # game state
    
    player = gs.dzxunlkwxt
    exit_sprite = gs.whftgckbcu
    
    print(f"\n{'='*60}")
    print(f"Level {lvl_idx+1}")
    print(f"{'='*60}")
    print(f"  Player start: ({player.x}, {player.y})")
    print(f"  Exit: ({exit_sprite.x}, {exit_sprite.y})")
    print(f"  Step size: {STEP}")
    print(f"  Record slots: {len(gs.drofvwhbxb)}")
    
    # Triggers
    print(f"  Triggers ({len(gs.uwxkstolmf)}):")
    for t in gs.uwxkstolmf:
        # lzwacefckd tells us the color = whether it's yellow (11) toggle
        color = getattr(t, 'lzwacefckd', '?')
        is_yellow = (color == xiibindvfw)  # xiibindvfw = 11 = yellow
        print(f"    Trigger at ({t.x},{t.y}) rot={t.rotation} yellow_toggle={is_yellow}")
    
    # Gates/doors
    print(f"  Gates/doors ({len(gs.hamayflsib)}):")
    for g in gs.hamayflsib:
        color = '?'
        if hasattr(g, 'lzwacefckd'):
            color = g.lzwacefckd
        print(f"    Gate at ({g.x},{g.y}) rot={getattr(g, 'rotation', '?')}")
    
    # Enemies/roamers
    print(f"  Enemies/roamers ({len(gs.kgvnkyaimw)}):")
    for e in gs.kgvnkyaimw:
        print(f"    Enemy at ({e.x},{e.y})")
    
    # Teleporters  
    teleporters = game.current_level.get_sprites_by_tag("mpreboxmgc")
    if teleporters:
        print(f"  Teleporters ({len(teleporters)}):")
        for tp in teleporters:
            print(f"    Teleporter at ({tp.x},{tp.y})")
    
    # hgglgttaui sprites (teleporter mechanism)
    hg = game.current_level.get_sprites_by_tag("hgglgttaui")
    if hg:
        print(f"  Teleporter mechanisms ({len(hg)}):")
        for h in hg:
            print(f"    At ({h.x},{h.y})")
    
    # vtwcsmdoqp (special sprites on L6/L7)
    vt = game.current_level.get_sprites_by_tag("vtwcsmdoqp")
    if vt:
        print(f"  Special (vtwcsmdoqp) ({len(vt)}):")
        for v in vt:
            print(f"    At ({v.x},{v.y})")
    
    # irlvvbptzu sprites (yellow obstacles L3)
    ir = game.current_level.get_sprites_by_tag("irlvvbptzu")
    if ir:
        print(f"  Yellow obstacles ({len(ir)}):")
        for i in ir:
            print(f"    At ({i.x},{i.y}) size={i.width}x{i.height}")
    
    # Paths (clone markers)
    paths = game.current_level.get_sprites_by_tag("hxztohfdlx")
    if paths:
        print(f"  Path markers ({len(paths)}):")
        for p in paths:
            print(f"    Path at ({p.x},{p.y}) size={p.width}x{p.height} rot={p.rotation}")
    
    # Room/area
    area = game.current_level.get_sprites_by_tag("rsrdfsruqh")
    if area:
        for a in area:
            print(f"  Room: ({a.x},{a.y}) size={a.width}x{a.height}")
    
    # Border
    border = game.current_level.get_sprites_by_tag("uxqxubarib")
    if border:
        for b in border:
            print(f"  Border: ({b.x},{b.y}) size={b.width}x{b.height}")

game = G50t()
for i in range(7):
    explore_level(game, i)

