"""
Trace L5 BFS solution (from step1018e) step by step, comparing BFS state vs real game.
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'

import importlib.util
spec = importlib.util.spec_from_file_location('sub', 'B:/M/the-search/experiments/step1018e_ls20_solver.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SOLUTIONS = mod._SOLUTIONS
ACTION_NAMES = mod.ACTION_NAMES
SOL_L5 = SOLUTIONS[4]

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

def real_state():
    return (game.gudziatsk.x, game.gudziatsk.y,
            game.fwckfzsyc, game.hiaauhahz, game.cklxociuu,
            game.level_index)

def get_trigger_positions():
    """Get current positions of all moving triggers (wsoslkeku sprites)."""
    return [(d._sprite.x, d._sprite.y) for d in game.wsoslqeku]

# Extract BFS-level model for L5
import arcagi3 as _ag
levels_raw = _ag.make('LS20')._env._game._levels
l5 = levels_raw[4]

# Get BFS info
print("\nBFS level data for L5:")
print(f"  Moving trigger cycles:")
for idx, d in enumerate(game.wsoslqeku):
    print(f"    trigger {idx}: sprite at ({d._sprite.x},{d._sprite.y}), mover ({d.bfdcztirdu.x},{d.bfdcztirdu.y} w={d.bfdcztirdu.width} h={d.bfdcztirdu.height})")

print(f"\nInitial real state: {real_state()}")
print(f"Trigger positions: {get_trigger_positions()}")

# Track BFS state
# BFS cycles from mod for L5
# Find the level model data from mod
# We'll simulate BFS state manually

# Get cycles from the BFS model
# L5 is level index 4
# Re-extract using mod's internal function
l5_model = mod._extract_level(l5, 4)
print(f"\nBFS model for L5:")
print(f"  start=(sh={l5_model['start_shape']}, co={l5_model['start_color']}, ro={l5_model['start_rot']})")
print(f"  cycle_lcm={l5_model['cycle_lcm']}")
print(f"  moving rot cycles: {l5_model['moving_rot_cycles']}")
print(f"  moving col cycles: {l5_model['moving_col_cycles']}")
print(f"  moving shp cycles: {l5_model['moving_shp_cycles']}")
print(f"  goals: {l5_model['goals']}")
print(f"  ramps: {[(r[0],r[1],r[4],r[5]) for r in l5_model['ramps']]}")

DIRS_BFS = [(0, -5), (0, 5), (-5, 0), (5, 0)]
PLAYER_SIZE = 5

def overlaps(ax, ay, bx, by, size=5):
    return ax == bx and ay == by

def bfs_apply_triggers(npx, npy, sh, co, ro, n_moves_new, model):
    nsh, nco, nro = sh, co, ro
    for tx, ty in model['rot_triggers']:
        if overlaps(tx, ty, npx, npy): nro = (nro + 1) % 4
    for cycle in model['moving_rot_cycles']:
        mtx, mty = cycle[n_moves_new % len(cycle)]
        if overlaps(mtx, mty, npx, npy): nro = (nro + 1) % 4
    for tx, ty in model['color_triggers']:
        if overlaps(tx, ty, npx, npy): nco = (nco + 1) % 4
    for cycle in model['moving_col_cycles']:
        mtx, mty = cycle[n_moves_new % len(cycle)]
        if overlaps(mtx, mty, npx, npy): nco = (nco + 1) % 4
    for tx, ty in model['shape_triggers']:
        if overlaps(tx, ty, npx, npy): nsh = (nsh + 1) % 6
    for cycle in model['moving_shp_cycles']:
        mtx, mty = cycle[n_moves_new % len(cycle)]
        if overlaps(mtx, mty, npx, npy): nsh = (nsh + 1) % 6
    return nsh, nco, nro

walls_set = set(model['walls'] if 'walls' in l5_model else [])
walls_set = set(l5_model['walls'])

def bfs_is_blocked(npx, npy, sh, co, ro, gdone, model):
    if npx < 0 or npy < 0 or npx + PLAYER_SIZE > 64 or npy + PLAYER_SIZE > 64:
        return True
    for wx, wy in model['walls']:
        if overlaps(wx, wy, npx, npy): return True
    for gi, (gx, gy, rs, rc, rr) in enumerate(model['goals']):
        if gi in gdone: continue
        if overlaps(gx, gy, npx, npy):
            if not (sh == rs and co == rc and ro == rr): return True
    return False

def bfs_check_ramp(npx, npy, sh, co, ro, n_moves_new, model):
    for rx, ry, rw, rh, dest_dx, dest_dy in model['ramps']:
        if overlaps(rx, ry, npx, npy):
            fx, fy = npx + dest_dx, npy + dest_dy
            nsh, nco, nro = bfs_apply_triggers(fx, fy, sh, co, ro, n_moves_new, model)
            return (fx, fy, nsh, nco, nro)
    return None

# Simulate BFS step by step
px, py = l5_model['px0'], l5_model['py0']
sh, co, ro = l5_model['start_shape'], l5_model['start_color'], l5_model['start_rot']
n_moves_mod = 0
gdone = frozenset()
model = l5_model

print(f"\n{'Step':>4} {'Act':5} {'Real px':>7} {'Real py':>7} {'BFS px':>6} {'BFS py':>6} {'Rsh':>4} {'Bsh':>4} {'Rco':>4} {'Bco':>4} {'Rro':>4} {'Bro':>4} {'Rnm':>4} {'Bnm':>4} {'Trig_real':>20} {'Match':>6} {'cl':>3}")

def get_nm_real():
    """Estimate real trigger phase by checking wsoslkeku sprite position."""
    if not game.wsoslqeku:
        return -1
    # First trigger sprite position
    sx, sy = game.wsoslqeku[0]._sprite.x, game.wsoslqeku[0]._sprite.y
    # Find index in cycle
    if model['moving_rot_cycles']:
        cyc = model['moving_rot_cycles'][0]
        for i, (cx, cy) in enumerate(cyc):
            if cx == sx and cy == sy:
                return i
    return -1

px0r, py0r, sh0r, co0r, ro0r, lvl0 = real_state()
print(f"   0  (init) {px0r:7} {py0r:7} {px:6} {py:6} {sh0r:4} {sh:4} {co0r:4} {co:4} {ro0r:4} {ro:4} {'?':>4} {'0':>4} {str(get_trigger_positions()):>20}  {'?':>6}   0")

for i, action in enumerate(SOL_L5):
    obs, reward, done, info = env.step(action)
    cl = info.get('level', 0) if isinstance(info, dict) else 0
    rpx, rpy, rsh, rco, rro, rlvl = real_state()
    trig_real = get_trigger_positions()

    # BFS simulate
    dx, dy = DIRS_BFS[action]
    npx, npy = px + dx, py + dy
    blocked = bfs_is_blocked(npx, npy, sh, co, ro, gdone, model)

    if blocked:
        # n_moves_mod unchanged
        bfs_sh, bfs_co, bfs_ro = sh, co, ro
        bfs_px, bfs_py = px, py
        bfs_nm = n_moves_mod
    else:
        n_moves_new = (n_moves_mod + 1) % model['cycle_lcm']
        nsh, nco, nro = bfs_apply_triggers(npx, npy, sh, co, ro, n_moves_new, model)
        ramp = bfs_check_ramp(npx, npy, nsh, nco, nro, n_moves_new, model)
        if ramp is not None:
            bfs_px, bfs_py, bfs_sh, bfs_co, bfs_ro = ramp
        else:
            bfs_px, bfs_py, bfs_sh, bfs_co, bfs_ro = npx, npy, nsh, nco, nro
        n_moves_mod = n_moves_new
        bfs_nm = n_moves_mod
        px, py, sh, co, ro = bfs_px, bfs_py, bfs_sh, bfs_co, bfs_ro

    if blocked:
        px_upd, py_upd = px, py
    else:
        px_upd, py_upd = bfs_px, bfs_py
        px, py = px_upd, py_upd
        sh, co, ro = bfs_sh, bfs_co, bfs_ro

    match = (rpx == bfs_px and rpy == bfs_py and rsh == bfs_sh and rco == bfs_co and rro == bfs_ro)
    nm_real = get_nm_real()

    print(f"  {i+1:3}  {ACTION_NAMES[action][0]:5} {rpx:7} {rpy:7} {bfs_px:6} {bfs_py:6} {rsh:4} {bfs_sh:4} {rco:4} {bfs_co:4} {rro:4} {bfs_ro:4} {nm_real:4} {bfs_nm:4} {str(trig_real):>20} {str(match):>6} {cl:3}")

    if cl > 4:
        print(f"  *** L5 COMPLETE at step {i+1} ***")
        break
    if done and cl <= 4:
        print(f"  *** GAME OVER at step {i+1} ***")
        break

print(f"\nFinal real state: {real_state()}")
print(f"BFS expected final: px={bfs_px}, py={bfs_py}, sh={bfs_sh}, co={bfs_co}, ro={bfs_ro}, n_moves={bfs_nm}")
goals_real = game.current_level.get_sprites_by_tag('rjlbuycveu')
print(f"Goals: {[(g.x,g.y) for g in goals_real]}")
print(f"Required: sh={game.ldxlnycps}, co_idx={game.yjdexjsoa}, ro_idx={game.ehwheiwsk}")
print(f"Current: sh={game.fwckfzsyc}, co={game.hiaauhahz}, ro={game.cklxociuu}")
