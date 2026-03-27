"""Explore all three games - dump sprite positions for each level."""
import sys, os
os.environ['PYTHONUTF8'] = '1'

# === G50T ===
sys.path.insert(0, 'environment_files/g50t/5849a774')
from g50t import G50t, GameAction
from arcengine import ActionInput
import numpy as np

def act(game, action_id, count=1):
    for _ in range(count):
        game.perform_action(ActionInput(id=action_id))

def pump(game, n=50):
    """Pump actions to let animations finish"""
    for _ in range(n):
        try:
            game.perform_action(ActionInput(id=GameAction.RESET))
        except:
            pass
        if hasattr(game, '_current_level_index'):
            pass

def describe_sprites(game, label):
    level = game.current_level
    sprites = level.get_sprites()
    print(f"\n{label} (Level {game.level_index + 1})")
    tag_groups = {}
    for s in sprites:
        tags = tuple(sorted(s.tags)) if hasattr(s, 'tags') else ()
        key = tags[0] if tags else 'untagged'
        if key not in tag_groups:
            tag_groups[key] = []
        tag_groups[key].append(s)
    
    for tag, slist in sorted(tag_groups.items()):
        for s in slist:
            rot = s.rotation if hasattr(s, 'rotation') else 0
            print(f"  [{tag}] pos=({s.x},{s.y}) size={s.width}x{s.height} rot={rot}")

# G50T - map through levels
print("=" * 60)
print("G50T - Ghost Replay Game")
print("=" * 60)
print("Actions: 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=RECORD")

game = G50t()

# Level by level sprite dump
for lvl_idx in range(7):
    game.set_level(lvl_idx)
    describe_sprites(game, f"G50T")

print("\n" + "=" * 60)

# Tag meanings from enum:
# qftsebtxuc = azxhtyyauk = player start
# gpkhwmwioo = ekfhaifjds = record slots 
# kjrcloicja = mcqullpcwz = trigger/claw
# medyellngi = mxmeqbrmab = gate/door  
# gilbljmfbc = lrtamslcit = exit
# ovhuyqtghw = inaylmmhhy = UI
# ppfvilwwnk = ofihnvwckg = timer
# hxztohfdlx = dryrnuvljg = clone path marker
# rsrdfsruqh = bognpjtvzt = area/room
# vtwcsmdoqp = qeixtoeawu = ?
# mpreboxmgc = ugfrhsffov = purple teleporters
# Ghost = gnezmcccob

