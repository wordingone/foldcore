"""
Trace L5 BFS solution step by step.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util

spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018d_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
ACTION_NAMES = mod.ACTION_NAMES
SOL_L5 = SOLUTIONS[4]  # 0-indexed: L5 = index 4

print(f"L5 BFS solution ({len(SOL_L5)} steps): {''.join(ACTION_NAMES[a][0] for a in SOL_L5)}")

import arcagi3
env = arcagi3.make('LS20')
obs = env.reset(seed=0)
for a in SOLUTIONS[0]: env.step(a)  # L1
for a in SOLUTIONS[1]: env.step(a)  # L2
for a in SOLUTIONS[2]: env.step(a)  # L3
for a in SOLUTIONS[3]: env.step(a)  # L4
print("Now at L5")

game = env._env._game

def gs():
    return (game.gudziatsk.x, game.gudziatsk.y,
            game.fwckfzsyc, game.hiaauhahz, game.cklxociuu,
            game.aqygnziho, game.akoadfsur, game.level_index)

print(f"{'Step':>4} {'Act':5} {'px':>4} {'py':>4} {'sh':>3} {'co':>3} {'ro':>3} {'lives':>5} {'akoad':>5} {'cl':>3} {'done':>5}")
px0,py0,sh0,co0,ro0,lv0,ak0,li0 = gs()
print(f"   0  (init) {px0:4} {py0:4} {sh0:3} {co0:3} {ro0:3} {lv0:5} {ak0:5}")

for i, action in enumerate(SOL_L5):
    obs, reward, done, info = env.step(action)
    cl = info.get('level', 0) if isinstance(info, dict) else 0
    px,py,sh,co,ro,lives,akoad,li = gs()
    print(f"  {i+1:3}  {ACTION_NAMES[action][0]:5} {px:4} {py:4} {sh:3} {co:3} {ro:3} {lives:5} {akoad:5}  {cl:3} {str(done):>5}")
    if cl > 4:
        print(f"  *** L5 COMPLETE at step {i+1} ***")
        break
    if done and cl <= 4:
        print(f"  *** GAME OVER at step {i+1} ***")
        break

# Print goal info for L5
goals = game.current_level.get_sprites_by_tag('rjlbuycveu')
print(f"\nGoals in real game: {[(g.x, g.y, g.width, g.height) for g in goals]}")
print(f"Required state: sh={game.ldxlnycps}, co_idx={game.yjdexjsoa}, ro_idx={game.ehwheiwsk}")
print(f"Current state: sh={game.fwckfzsyc}, co={game.hiaauhahz}, ro={game.cklxociuu}")
print(f"Level done flags: {game.lvrnuajbl}")
