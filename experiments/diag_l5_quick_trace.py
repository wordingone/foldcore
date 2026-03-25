"""Quick L5 trace without running BFS. Hardcode the known 44-step solution."""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
os.environ['PYTHONUTF8'] = '1'
import arcagi3

env = arcagi3.make('LS20')
obs = env.reset(seed=0)
game = env._env._game

# Play L1-L4 solutions (hardcoded from previous known-good run)
L1 = [2,2,2,0,0,0,3,3,3,0,0,0]  # LLLUUUURRRUUU → action indices: U=0,D=1,L=2,R=3
L2_str = "URUUUUURRDRDDDDDDDLLRURUDUUUUUUULLLLLLDLDDDDD"
L3_str = "UUUUUUUULDDDDDDDDUUULLURRRRRRRUUULUDURD"
L4_str = "LLLDDDLDDLLUDUDUDUULLUDLLUUUDDRUUUURURUULLL"

ACTION_MAP = {'U':0,'D':1,'L':2,'R':3}
def s2a(s): return [ACTION_MAP[c] for c in s]

for a in s2a("LLLUUUURRRUUU"): env.step(a)  # L1
for a in s2a(L2_str): env.step(a)  # L2
for a in s2a(L3_str): env.step(a)  # L3
for a in s2a(L4_str): env.step(a)  # L4
print(f"After L4: level={game.level_index}")

# L5 solution
L5_str = "UUULUULLLRLRLRUULLLLULLLRRDDDDDRRDDDRRRRRRRU"
L5_sol = s2a(L5_str)

print(f"\nL5 44-step trace:")
print(f"Initial: px={game.gudziatsk.x} py={game.gudziatsk.y} sh={game.fwckfzsyc} co={game.hiaauhahz} ro={game.cklxociuu}")

# Get moving triggers
for i, d in enumerate(game.wsoslqeku):
    print(f"Mover {i}: sprite at ({d._sprite.x},{d._sprite.y}), mover ({d.bfdcztirdu.x},{d.bfdcztirdu.y} w={d.bfdcztirdu.width})")

goals_real = game.current_level.get_sprites_by_tag('rjlbuycveu')
print(f"Goal: {[(g.x,g.y) for g in goals_real]}")
print(f"Required: sh={game.ldxlnycps}, co={game.yjdexjsoa}, ro={game.ehwheiwsk}")

print(f"\n{'St':>3} {'Act':4} {'px':>4} {'py':>4} {'sh':>3} {'co':>3} {'ro':>3} {'trig':>12} {'cl':>3} {'done':>5}")

for i, action in enumerate(L5_sol):
    act_name = 'UDLR'[action]
    trig_before = [(d._sprite.x, d._sprite.y) for d in game.wsoslqeku]
    obs, reward, done, info = env.step(action)
    cl = info.get('level', 0) if isinstance(info, dict) else 0
    px, py = game.gudziatsk.x, game.gudziatsk.y
    sh, co, ro = game.fwckfzsyc, game.hiaauhahz, game.cklxociuu
    trig_after = [(d._sprite.x, d._sprite.y) for d in game.wsoslqeku]
    print(f"{i+1:3} {act_name:4} {px:4} {py:4} {sh:3} {co:3} {ro:3} {str(trig_after[0] if trig_after else '?'):>12}  {cl:3} {str(done):>5}")
    if cl > 4:
        print(f"  *** L5 COMPLETE ***")
        break
    if done and cl <= 4:
        print(f"  *** DIED ***")
        break

print(f"\nFinal: px={game.gudziatsk.x} py={game.gudziatsk.y} sh={game.fwckfzsyc} co={game.hiaauhahz} ro={game.cklxociuu}")
print(f"Goal: sh={game.ldxlnycps}, co={game.yjdexjsoa}, ro={game.ehwheiwsk}")
