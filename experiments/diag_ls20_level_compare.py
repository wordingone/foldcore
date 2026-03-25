"""
Compare mod.levels[i] (mock arcengine) vs game._levels[i] (real arcengine).
Find why L5 has different trigger data in each.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import arcagi3
real_env = arcagi3.make('LS20')
real_env.reset(seed=0)
real_game = real_env._env._game

# Print real game level summary
print("=== REAL GAME _levels ===")
for i, lv in enumerate(real_game._levels):
    movers = lv.get_sprites_by_tag('xfmluydglp')
    rot_t = lv.get_sprites_by_tag('rhsxkxzdjz')
    col_t = lv.get_sprites_by_tag('soyhouuebz')
    shp_t = lv.get_sprites_by_tag('ttfwljgohq')
    px = lv.get_sprites_by_tag('sfqyzhzkij')
    print(f"  L{i+1}: player={[(s.x,s.y) for s in px]}, movers={[(m.x,m.y,m.width,m.height) for m in movers]}")
    print(f"        rot_t={[(s.x,s.y) for s in rot_t]}, col_t={[(s.x,s.y) for s in col_t]}, shp_t={[(s.x,s.y) for s in shp_t]}")

# Now load via mock arcengine (same as BFS does)
import importlib.util
spec = importlib.util.spec_from_file_location('solver1018e', 'B:/M/the-search/experiments/step1018e_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
# Only exec the mock arcengine + level loading without computing solutions
# Actually we have to load the whole thing - but it already computed solutions
# So let's just use the already-loaded mod
spec.loader.exec_module(mod)
