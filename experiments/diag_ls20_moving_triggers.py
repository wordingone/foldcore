"""
Check which LS20 levels have moving triggers (xfmluydglp/dboxixicic).
These break the static BFS model — triggers change position each step.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util

spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018b_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS

import arcagi3
env = arcagi3.make('LS20')

# Play through levels and inspect each one
obs = env.reset(seed=0)
game = env._env._game

for lvl_idx in range(7):
    if lvl_idx > 0:
        # Advance to next level (play the BFS solution or fake-advance)
        sol = SOLUTIONS[lvl_idx - 1]
        if sol:
            for a in sol:
                obs, _, done, info = env.step(a)
                if info.get('level', 0) >= lvl_idx:
                    break
        else:
            print(f"  L{lvl_idx+1}: can't advance (no BFS solution)")
            break

    # Read game state at this level
    game = env._env._game

    # Moving triggers
    n_moving = len(game.wsoslqeku)

    # Static triggers
    rot_t = game.current_level.get_sprites_by_tag('rhsxkxzdjz')
    col_t = game.current_level.get_sprites_by_tag('soyhouuebz')
    shp_t = game.current_level.get_sprites_by_tag('ttfwljgohq')

    # Movers (xfmluydglp)
    movers = game.current_level.get_sprites_by_tag('xfmluydglp')

    print(f"L{lvl_idx+1} (level_index={game.level_index}):")
    print(f"  Moving trigger objects (dboxixicic): {n_moving}")
    print(f"  Mover sprites (xfmluydglp): {len(movers)} at {[(m.x, m.y) for m in movers]}")
    print(f"  Rot triggers: {[(t.x, t.y) for t in rot_t]}")
    print(f"  Color triggers: {[(t.x, t.y) for t in col_t]}")
    print(f"  Shape triggers: {[(t.x, t.y) for t in shp_t]}")
    print(f"  Player start: ({game.gudziatsk.x}, {game.gudziatsk.y}) | ltwrkifkx={game.ltwrkifkx}, zyoimjaei={game.zyoimjaei}")

    # If moving triggers exist, show their state after 5 steps
    if n_moving > 0:
        print(f"  Moving trigger positions over 5 steps:")
        for step_i in range(5):
            for aril in game.wsoslqeku:
                aril.step()
            positions = [(aril._sprite.x, aril._sprite.y) for aril in game.wsoslqeku]
            print(f"    After step {step_i+1}: {positions}")
        # Reset mover positions
        for aril in game.wsoslqeku:
            aril.bkuguqrpvq()
        print(f"    After reset: {[(aril._sprite.x, aril._sprite.y) for aril in game.wsoslqeku]}")
    print()
