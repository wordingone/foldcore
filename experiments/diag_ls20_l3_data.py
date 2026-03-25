"""
Dump L3 actual game state at each step of the BFS path.
Shows exact position and step counter to find where model diverges.
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
ACTION_NAMES = mod.ACTION_NAMES
SOL_L3 = SOLUTIONS[2]

import arcagi3
env = arcagi3.make('LS20')

# Play L1 and L2 first
obs = env.reset(seed=0)
for a in SOLUTIONS[0]: env.step(a)
for a in SOLUTIONS[1]: env.step(a)

# Access game internals
game = env._env._game

def game_state(game):
    px = game.gudziatsk.x
    py = game.gudziatsk.y
    sh = game.fwckfzsyc
    co = game.hiaauhahz
    ro = game.cklxociuu
    lvl = game.level_index
    lives = game.aqygnziho
    akoad = game.akoadfsur
    # Step counter
    counter = game._step_counter_ui.osgviligwp
    # Collectibles remaining
    n_coll = len(game.ofoahudlo)  # collected so far
    return px, py, sh, co, ro, lvl, lives, akoad, counter, n_coll

print("=== L3 game state trace ===")
print(f"{'Step':>4} {'Act':5} {'px':>4} {'py':>4} {'sh':>3} {'co':>3} {'ro':>3} {'lives':>5} {'akoad':>5} {'steps':>5} {'ncoll':>5} {'cl':>3} {'done':>5}")

# Print initial state
px,py,sh,co,ro,lvl,lives,akoad,counter,ncoll = game_state(game)
print(f"  0  (init) {px:4} {py:4} {sh:3} {co:3} {ro:3} {lives:5} {akoad:5} {counter:5} {ncoll:5}")

for i, action in enumerate(SOL_L3):
    obs, reward, done, info = env.step(action)
    cl = info.get('level', 0) if isinstance(info, dict) else 0

    px,py,sh,co,ro,lvl,lives,akoad,counter,ncoll = game_state(game)
    print(f"{i+1:4}  {ACTION_NAMES[action][0]:5} {px:4} {py:4} {sh:3} {co:3} {ro:3} {lives:5} {akoad:5} {counter:5} {ncoll:5}  {cl:3} {str(done):>5}")

    if cl > 2:
        print(f"*** L3 COMPLETE at step {i+1} ***")
        break
    if lives == 0 or (done and cl <= 2):
        print(f"*** GAME OVER at step {i+1} ***")
        break

# Also dump collectible positions for L3 (from BFS model)
print("\n=== BFS model - L3 collectibles ===")
ld3 = mod._extract_level(game._levels[2], 2)
print(f"  collectibles: {ld3['collectibles']}")
print(f"  moves_per_life: {ld3['moves_per_life']}")
print(f"  goals: {ld3['goals']}")
print(f"  player start: ({ld3['px0']}, {ld3['py0']})")
print(f"  start_shape={ld3['start_shape']} start_color={ld3['start_color']} start_rot={ld3['start_rot']}")

# Try to get actual collectibles from the real game at L3 start
print("\n=== Real game - L3 collectibles (from fresh reset to L3) ===")
env2 = arcagi3.make('LS20')
obs2 = env2.reset(seed=0)
game2 = env2._env._game
for a in SOLUTIONS[0]: env2.step(a)
for a in SOLUTIONS[1]: env2.step(a)
# Now at L3
game2 = env2._env._game
colls = game2.current_level.get_sprites_by_tag('npxgalaybz')
print(f"  collectibles in real game: {[(c.x, c.y, c.width, c.height) for c in colls]}")
goals_real = game2.current_level.get_sprites_by_tag('rjlbuycveu')
print(f"  goals in real game: {[(g.x, g.y, g.width, g.height) for g in goals_real]}")
players_real = game2.current_level.get_sprites_by_tag('sfqyzhzkij')
print(f"  player in real game: {[(p.x, p.y, p.width, p.height) for p in players_real]}")
print(f"  game start_shape={game2.fwckfzsyc} start_color={game2.hiaauhahz} start_rot={game2.cklxociuu}")
print(f"  game level_index={game2.level_index}, lives={game2.aqygnziho}")
