"""Explore G50T levels by playing through them using arcengine directly."""
import sys, os
sys.path.insert(0, 'environment_files/g50t/5849a774')
os.environ['PYTHONUTF8'] = '1'

from g50t import G50t, GameAction
from arcengine import ActionInput
import numpy as np

def act(game, action_id, count=1):
    results = []
    for _ in range(count):
        game.perform_action(ActionInput(id=action_id))
        results.append({
            'level': game.level_index,
        })
    return results

def describe_level(game):
    level = game.current_level
    sprites = level.get_sprites()
    print(f"\n=== Level {game.level_index + 1} ===")
    print(f"  Number of sprites: {len(sprites)}")
    for s in sprites:
        tags = list(s.tags) if hasattr(s, 'tags') else []
        print(f"  Sprite: pos=({s.x},{s.y}) size=({s.width}x{s.height}) tags={tags}")

game = G50t()
describe_level(game)

# Solve L1: RIGHT*4, RECORD, DOWN*7, RIGHT*5
print("=== Solving L1 ===")
act(game, GameAction.ACTION4, 4)  # RIGHT*4
act(game, GameAction.ACTION5, 1)  # RECORD
act(game, GameAction.ACTION2, 7)  # DOWN*7
r = act(game, GameAction.ACTION4, 5)  # RIGHT*5
print(f"After L1 actions: level={r[-1]['level']}")

# Check if we need more steps
for i in range(20):
    try:
        act(game, GameAction.ACTION4, 1)
    except:
        break
    if game.level_index >= 1:
        print(f"Advanced to level {game.level_index + 1}")
        break

print(f"\nNow at level index: {game.level_index}")
describe_level(game)
