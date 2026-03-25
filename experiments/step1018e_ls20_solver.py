"""
step1018e_ls20_solver.py — LS20/9607627b Multi-Level BFS Solver v5.

BFS FIXES FROM 1018d:
  1. Goals included in ramp wall set (fjzuynaokm bug):
     Real game includes rjlbuycveu (goal) sprites in ramp destination wall search.
     BFS was missing this — ramp (54,51) in L5 went past goal (54,5), player at y=5
     instead of correct y=10. Now included: ramp_walls = ihdgageizm | rjlbuycveu.

  2. Moving triggers modeled (dboxixicic oscillation):
     L5 rotation trigger (14,35) is a MOVING trigger (wsoslqeku count=1).
     Mover sprite ajdspzphhd at (14,35,w=11,h=1) carries the rotation trigger
     through cycle: (14,35)→(19,35)→(24,35)→(19,35)→(14,35)→... period 4.
     wsoslqeku.step() called once per NON-BLOCKED action (before txnfzvzetn).
     fwtnsrvkrz() reverts on blocked actions (net-zero phase advance).
     BFS state now includes n_moves_mod (= non-blocked moves % LCM of cycle lengths).
     Trigger fires when player's new position overlaps trigger's CURRENT phase position.

DIRECTION 2: Per-game prescription allowed (Jun directive 2026-03-24).
BUDGET: 10K steps, 10 seeds.
"""
import sys
import os
import types
import math
from collections import deque
import numpy as np

# ─── Constants ───
TNKEKOEUK = [12, 9, 14, 8]
DHKSVILBB = [0, 90, 180, 270]
N_SHAPES = 6
N_COLORS = 4
N_ROTS = 4
PLAYER_SIZE = 5

DIRS = [(0, -PLAYER_SIZE), (0, PLAYER_SIZE), (-PLAYER_SIZE, 0), (PLAYER_SIZE, 0)]
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

CONFIG = {
    "solver": "bfs_ls20_ramp_aware_moving_triggers",
    "levels": 7,
    "direction": 2,
}


# ─── Mock arcengine ───

class _Sprite:
    def __init__(self, pixels=None, name='', visible=True, collidable=True, tags=None, layer=0):
        self.pixels = pixels
        self.name = name
        self.tags = list(tags) if tags else []
        self.x = 0
        self.y = 0
        self.width = len(pixels[0]) if pixels else 0
        self.height = len(pixels) if pixels else 0

    def clone(self):
        s = _Sprite(self.pixels, self.name, tags=self.tags)
        s.width = self.width
        s.height = self.height
        s.x = self.x
        s.y = self.y
        return s

    def set_position(self, x, y):
        self.x = x
        self.y = y
        return self

    def set_rotation(self, r):
        return self

    def set_scale(self, s):
        return self

    def color_remap(self, a, b):
        return self


class _Level:
    def __init__(self, sprites=None, grid_size=None, data=None):
        self._sprites = sprites or []
        self._data = data or {}

    def get_data(self, key):
        return self._data.get(key)

    def get_sprites_by_tag(self, tag):
        return [s for s in self._sprites if tag in s.tags]


class _Camera:
    def __init__(self, **kw):
        pass


class _ARCBaseGame:
    pass


class _GameAction:
    pass


class _BlockingMode:
    pass


class _RenderableUserDisplay:
    def __init__(self, *a, **kw):
        pass


def _install_mock_arcengine():
    prev = sys.modules.get('arcengine')
    m = types.ModuleType('arcengine')
    m.ARCBaseGame = _ARCBaseGame
    m.BlockingMode = _BlockingMode
    m.Camera = _Camera
    m.GameAction = _GameAction
    m.Level = _Level
    m.RenderableUserDisplay = _RenderableUserDisplay
    m.Sprite = _Sprite
    sys.modules['arcengine'] = m
    return prev


def _restore_arcengine(prev):
    if prev is None:
        del sys.modules['arcengine']
    else:
        sys.modules['arcengine'] = prev


# ─── Math helpers ───

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a * b // _gcd(a, b)


# ─── Moving trigger simulation (dboxixicic) ───

# dboxixicic directions: 0=(0,1)=down, 1=(1,0)=right, 2=(0,-1)=up, 3=(-1,0)=left
_MOVER_DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def _compute_mover_cycle(mover, trigger_x, trigger_y, cell=5):
    """
    Simulate dboxixicic movement to get trigger position at each step count.

    Returns list of (x, y) positions indexed by [step_count % len(cycle)].
    The cycle begins at step 0 = trigger's initial position (trigger_x, trigger_y).

    Uses position-only cycle detection: simulate from start, stop when trigger
    returns to (trigger_x, trigger_y). This gives correct cycle starting at index 0.
    """
    mx, my, mw, mh = mover.x, mover.y, mover.width, mover.height
    pixels = mover.pixels

    def valid(x, y):
        ax = x - mx
        ay = y - my
        if ax < 0 or ax >= mw or ay < 0 or ay >= mh:
            return False
        row = pixels[ay]
        return int(row[ax]) >= 0

    tx, ty = trigger_x, trigger_y
    direction = 0
    positions = [(tx, ty)]  # index 0 = initial position

    for _ in range(200):
        dirs_to_try = [direction, (direction - 1) % 4, (direction + 1) % 4, (direction + 2) % 4]
        moved = False
        for d in dirs_to_try:
            dx, dy = _MOVER_DIRS[d]
            nx, ny = tx + dx * cell, ty + dy * cell
            if valid(nx, ny):
                direction = d
                tx, ty = nx, ny
                moved = True
                break
        if not moved:
            break
        if tx == trigger_x and ty == trigger_y:
            break  # returned to start: cycle complete
        positions.append((tx, ty))

    return positions if positions else [(trigger_x, trigger_y)]


# ─── Level data extraction ───

def _color_idx(val):
    return TNKEKOEUK.index(val)


def _rot_idx(val):
    return DHKSVILBB.index(val)


def _overlaps(wx, wy, px, py):
    """Sprite top-left corner (wx,wy) size PLAYER_SIZE overlaps player bounding box."""
    return wx >= px and wx < px + PLAYER_SIZE and wy >= py and wy < py + PLAYER_SIZE


def _box_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    """Full bounding-box overlap check between two rectangles."""
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _compute_ramp_dest(rx, ry, dx, dy, ramp_w, ramp_h, ramp_walls):
    """
    Compute player destination delta when entering ramp at (rx, ry) direction (dx, dy).
    ramp_walls: frozenset of (x, y) positions that stop the ramp slide.
    Range matches real game: range(1, 12) = k from 1 to 11.
    """
    wall_cx = rx + dx
    wall_cy = ry + dy
    cells = 0
    for k in range(1, 12):
        nskortmtde = wall_cx + dx * ramp_w * k
        wsotwbexvh = wall_cy + dy * ramp_h * k
        if (nskortmtde, wsotwbexvh) in ramp_walls:
            cells = max(0, k - 1)
            break
    if cells <= 0:
        return None
    return (dx * ramp_w * cells, dy * ramp_h * cells)


def _extract_level(level, level_idx):
    data = level._data
    step_counter = data.get('StepCounter', 42)
    steps_dec = data.get('StepsDecrement', 2)
    moves_per_life = step_counter // steps_dec

    start_shape = data.get('StartShape', 0)
    start_color = _color_idx(data.get('StartColor', 9))
    start_rot = _rot_idx(data.get('StartRotation', 0))

    kvynsvxbpi = data.get('kvynsvxbpi', 0)
    goal_colors_raw = data.get('GoalColor', 9)
    goal_rots_raw = data.get('GoalRotation', 0)
    if isinstance(kvynsvxbpi, int):
        kvynsvxbpi = [kvynsvxbpi]
    if isinstance(goal_colors_raw, int):
        goal_colors_raw = [goal_colors_raw]
    if isinstance(goal_rots_raw, int):
        goal_rots_raw = [goal_rots_raw]

    # Walls (ihdgageizm)
    walls = set()
    for s in level.get_sprites_by_tag('ihdgageizm'):
        walls.add((s.x, s.y))

    # Player start
    ps = level.get_sprites_by_tag('sfqyzhzkij')
    px0, py0 = ps[0].x, ps[0].y

    # Goals (rjlbuycveu)
    goal_sprites = level.get_sprites_by_tag('rjlbuycveu')
    goals = []
    for i, gs in enumerate(goal_sprites):
        req_shape = kvynsvxbpi[i]
        req_color = _color_idx(goal_colors_raw[i])
        req_rot = _rot_idx(goal_rots_raw[i])
        goals.append((gs.x, gs.y, req_shape, req_color, req_rot))

    # Triggers (static)
    rot_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('rhsxkxzdjz')]
    color_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('soyhouuebz')]
    shape_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('ttfwljgohq')]

    # Collectibles
    collectibles = [(s.x, s.y) for s in level.get_sprites_by_tag('npxgalaybz')]

    # Ramp wall set: ihdgageizm + rjlbuycveu (real game's fjzuynaokm includes goals)
    goals_set = frozenset((s.x, s.y) for s in goal_sprites)
    ramp_walls = frozenset(walls) | goals_set

    # Ramps (gbvqrjtaqo)
    ramps = []
    for s in level.get_sprites_by_tag('gbvqrjtaqo'):
        dx, dy = 0, 0
        if s.name.endswith('r'):
            dx = 1
        elif s.name.endswith('l'):
            dx = -1
        elif s.name.endswith('t'):
            dy = -1
        elif s.name.endswith('b'):
            dy = 1
        dest = _compute_ramp_dest(s.x, s.y, dx, dy, s.width, s.height, ramp_walls)
        if dest is not None:
            ramps.append((s.x, s.y, s.width, s.height, dest[0], dest[1]))

    # Moving triggers (xfmluydglp movers carrying rhsxkxzdjz/soyhouuebz/ttfwljgohq)
    # Real game: dboxixicic links mover to trigger if collides_with(ignoreMode=True).
    # We use _box_overlap as approximation.
    moving_rot_cycles = []   # list of [(x,y), ...] per moving rot trigger
    moving_col_cycles = []
    moving_shp_cycles = []

    movers = level.get_sprites_by_tag('xfmluydglp')
    for mover in movers:
        for tag, cycle_list in [('rhsxkxzdjz', moving_rot_cycles),
                                 ('soyhouuebz', moving_col_cycles),
                                 ('ttfwljgohq', moving_shp_cycles)]:
            for trigger_s in level.get_sprites_by_tag(tag):
                if _box_overlap(mover.x, mover.y, mover.width, mover.height,
                                trigger_s.x, trigger_s.y, trigger_s.width, trigger_s.height):
                    cycle = _compute_mover_cycle(mover, trigger_s.x, trigger_s.y)
                    cycle_list.append(cycle)
                    # Remove this trigger from static list (it's moving, not static)
                    key = (trigger_s.x, trigger_s.y)
                    if tag == 'rhsxkxzdjz' and key in rot_triggers:
                        rot_triggers.remove(key)
                    elif tag == 'soyhouuebz' and key in color_triggers:
                        color_triggers.remove(key)
                    elif tag == 'ttfwljgohq' and key in shape_triggers:
                        shape_triggers.remove(key)

    # Compute LCM of all cycle lengths (for trigger_phase modulo)
    all_cycle_lens = [len(c) for c in moving_rot_cycles + moving_col_cycles + moving_shp_cycles]
    if all_cycle_lens:
        cycle_lcm = all_cycle_lens[0]
        for cl in all_cycle_lens[1:]:
            cycle_lcm = _lcm(cycle_lcm, cl)
    else:
        cycle_lcm = 1  # No moving triggers: phase is always 0

    if moving_rot_cycles or moving_col_cycles or moving_shp_cycles:
        print(f"    moving triggers: rot={len(moving_rot_cycles)} col={len(moving_col_cycles)} shp={len(moving_shp_cycles)} cycle_lcm={cycle_lcm}")
        for c in moving_rot_cycles:
            print(f"      rot cycle: {c}")

    return {
        'level_idx': level_idx,
        'px0': px0, 'py0': py0,
        'start_shape': start_shape,
        'start_color': start_color,
        'start_rot': start_rot,
        'goals': goals,
        'walls': frozenset(walls),
        'rot_triggers': rot_triggers,
        'color_triggers': color_triggers,
        'shape_triggers': shape_triggers,
        'moving_rot_cycles': moving_rot_cycles,
        'moving_col_cycles': moving_col_cycles,
        'moving_shp_cycles': moving_shp_cycles,
        'cycle_lcm': cycle_lcm,
        'collectibles': collectibles,
        'ramps': ramps,
        'moves_per_life': moves_per_life,
        'step_counter': step_counter,
        'n_goals': len(goals),
    }


# ─── BFS solver ───

def _bfs_solve(ld, max_path_len=250):
    """
    BFS with ramp-aware and moving-trigger-aware mechanics.

    State: (px, py, shape, color, rot, goals_done_frozenset, collected_frozenset,
            steps_left, n_moves_mod, deaths_remaining)

    n_moves_mod = total non-blocked moves % cycle_lcm.
    Trigger positions depend on (n_moves_mod + 1) % cycle_len at move time.
    Blocked moves: n_moves_mod unchanged (wsoslqeku.step() + fwtnsrvkrz() = net 0).
    Death (sleft -> 0): respawn at initial state with deaths_remaining - 1.
    max_lives = 3 (aqygnziho = 3 in on_set_level).
    """
    px0, py0 = ld['px0'], ld['py0']
    s0, c0, r0 = ld['start_shape'], ld['start_color'], ld['start_rot']
    goals = ld['goals']
    walls = ld['walls']
    rot_t = ld['rot_triggers']
    col_t = ld['color_triggers']
    shp_t = ld['shape_triggers']
    mov_rot = ld['moving_rot_cycles']
    mov_col = ld['moving_col_cycles']
    mov_shp = ld['moving_shp_cycles']
    cycle_lcm = ld['cycle_lcm']
    colls = ld['collectibles']
    ramps = ld['ramps']
    mpl = ld['moves_per_life']
    n_goals = ld['n_goals']

    all_done = frozenset(range(n_goals))
    n_colls = len(colls)
    all_colls = frozenset(range(n_colls))

    max_lives = 3  # aqygnziho = 3 in on_set_level
    init_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl, 0, max_lives - 1)

    queue = deque([(init_state, [])])
    visited = {init_state}

    def _apply_triggers(npx, npy, sh, co, ro, n_moves_new):
        """Apply trigger effects at position (npx, npy) with n_moves_new phase."""
        nsh, nco, nro = sh, co, ro
        # Static rot triggers
        for tx, ty in rot_t:
            if _overlaps(tx, ty, npx, npy):
                nro = (nro + 1) % N_ROTS
        # Moving rot triggers
        for cycle in mov_rot:
            mtx, mty = cycle[n_moves_new % len(cycle)]
            if _overlaps(mtx, mty, npx, npy):
                nro = (nro + 1) % N_ROTS
        # Static col triggers
        for tx, ty in col_t:
            if _overlaps(tx, ty, npx, npy):
                nco = (nco + 1) % N_COLORS
        # Moving col triggers
        for cycle in mov_col:
            mtx, mty = cycle[n_moves_new % len(cycle)]
            if _overlaps(mtx, mty, npx, npy):
                nco = (nco + 1) % N_COLORS
        # Static shp triggers
        for tx, ty in shp_t:
            if _overlaps(tx, ty, npx, npy):
                nsh = (nsh + 1) % N_SHAPES
        # Moving shp triggers
        for cycle in mov_shp:
            mtx, mty = cycle[n_moves_new % len(cycle)]
            if _overlaps(mtx, mty, npx, npy):
                nsh = (nsh + 1) % N_SHAPES
        return nsh, nco, nro

    def _apply_collectibles(npx, npy, coll, sleft):
        new_coll = set(coll)
        new_sleft = sleft
        for ci in list(coll):
            cx, cy = colls[ci]
            if _overlaps(cx, cy, npx, npy):
                new_coll.discard(ci)
                new_sleft = mpl
        return frozenset(new_coll), new_sleft

    def _check_goals_at(px, py, sh, co, ro, gdone):
        new_gdone = set(gdone)
        for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
            if gi in gdone:
                continue
            if px == gx and py == gy and sh == rs and co == rc and ro == rr:
                new_gdone.add(gi)
        return frozenset(new_gdone)

    def _check_ramp(npx, npy, sh, co, ro, coll, sleft, gdone, n_moves_new):
        """Check ramp at (npx, npy). Returns result or None."""
        for rx, ry, rw, rh, dest_dx, dest_dy in ramps:
            if _box_overlap(rx, ry, rw, rh, npx, npy, PLAYER_SIZE, PLAYER_SIZE):
                final_x = npx + dest_dx
                final_y = npy + dest_dy
                # Apply triggers at destination (same n_moves_new — ramp frames don't advance)
                nsh, nco, nro = _apply_triggers(final_x, final_y, sh, co, ro, n_moves_new)
                new_coll, new_sleft = _apply_collectibles(final_x, final_y, coll, sleft)
                if new_sleft == sleft:
                    new_sleft = sleft
                return (final_x, final_y, nsh, nco, nro, new_coll, new_sleft)
        return None

    while queue:
        state, path = queue.popleft()

        if len(path) >= max_path_len:
            continue

        px, py, sh, co, ro, gdone, coll, sleft, n_moves_mod, deaths_remaining = state

        if gdone == all_done:
            return path

        for action, (dx, dy) in enumerate(DIRS):
            npx, npy = px + dx, py + dy

            # Determine if blocked
            is_blocked = False

            if npx < 0 or npy < 0 or npx + PLAYER_SIZE > 64 or npy + PLAYER_SIZE > 64:
                is_blocked = True
            else:
                for wx, wy in walls:
                    if _overlaps(wx, wy, npx, npy):
                        is_blocked = True
                        break

                if not is_blocked:
                    for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                        if gi in gdone:
                            continue
                        if _overlaps(gx, gy, npx, npy):
                            if not (sh == rs and co == rc and ro == rr):
                                is_blocked = True
                                break

            if is_blocked:
                # Blocked: n_moves_mod UNCHANGED (fwtnsrvkrz undo), sleft-1
                new_sleft = sleft - 1
                if new_sleft <= 0:
                    if deaths_remaining > 0:
                        death_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl, 0, deaths_remaining - 1)
                        if death_state not in visited:
                            visited.add(death_state)
                            queue.append((death_state, path + [action]))
                    continue
                new_gdone = _check_goals_at(px, py, sh, co, ro, gdone)
                new_state = (px, py, sh, co, ro, new_gdone, coll, new_sleft, n_moves_mod, deaths_remaining)
                if new_state not in visited:
                    visited.add(new_state)
                    if new_gdone == all_done:
                        return path + [action]
                    queue.append((new_state, path + [action]))
                continue

            # Non-blocked move: advance trigger phase FIRST
            n_moves_new = (n_moves_mod + 1) % cycle_lcm

            # Apply triggers at (npx, npy) with new phase
            nsh, nco, nro = _apply_triggers(npx, npy, sh, co, ro, n_moves_new)

            # Check ramp at (npx, npy) — ramp entry costs 1 step (mfyzdfvxsm called before ramp check)
            ramp_result = _check_ramp(npx, npy, nsh, nco, nro, coll, sleft - 1, gdone, n_moves_new)
            if ramp_result is not None:
                final_x, final_y, rsh, rco, rro, rcoll, rsleft = ramp_result
                if rsleft <= 0:
                    if deaths_remaining > 0:
                        death_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl, 0, deaths_remaining - 1)
                        if death_state not in visited:
                            visited.add(death_state)
                            queue.append((death_state, path + [action]))
                    continue
                new_state = (final_x, final_y, rsh, rco, rro, gdone, rcoll, rsleft, n_moves_new, deaths_remaining)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action]))
                continue

            # Normal move
            new_coll, new_sleft = _apply_collectibles(npx, npy, coll, sleft - 1)

            new_gdone = set(gdone)
            for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                if gi in gdone:
                    continue
                if _overlaps(gx, gy, npx, npy):
                    if nsh == rs and nco == rc and nro == rr:
                        new_gdone.add(gi)

            if frozenset(new_gdone) == all_done:
                return path + [action]

            if new_sleft <= 0:
                if deaths_remaining > 0:
                    death_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl, 0, deaths_remaining - 1)
                    if death_state not in visited:
                        visited.add(death_state)
                        queue.append((death_state, path + [action]))
                continue

            new_state = (npx, npy, nsh, nco, nro,
                         frozenset(new_gdone), new_coll, new_sleft, n_moves_new, deaths_remaining)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [action]))

    return None


# ─── Compute solutions at module load ───

def _compute_solutions():
    # Use real arcagi3 env to get correct level data (mock arcengine gives wrong sprite dims)
    try:
        import arcagi3
        real_env = arcagi3.make('LS20')
        real_env.reset(seed=0)
        ls20_levels = real_env._env._game._clean_levels
        print(f"  Loaded {len(ls20_levels)} LS20 levels")
    except Exception as e:
        print(f"WARNING: failed to load LS20 via arcagi3: {e} — using fallback")
        return None

    solutions = []
    for i, level in enumerate(ls20_levels):
        ld = _extract_level(level, i)
        n_ramps = len(ld['ramps'])
        has_moving = bool(ld['moving_rot_cycles'] or ld['moving_col_cycles'] or ld['moving_shp_cycles'])
        print(f"  Level {i+1}: player=({ld['px0']},{ld['py0']}), "
              f"goals={len(ld['goals'])}, walls={len(ld['walls'])}, "
              f"moves_per_life={ld['moves_per_life']}, ramps={n_ramps}"
              f"{', moving_triggers' if has_moving else ''}")
        if n_ramps > 0:
            for rx, ry, rw, rh, ddx, ddy in ld['ramps']:
                print(f"    ramp at ({rx},{ry}) -> delta ({ddx},{ddy})")

        sol = None
        for max_len in [100, 200, 300, 400]:
            sol = _bfs_solve(ld, max_path_len=max_len)
            if sol is not None:
                break

        if sol is not None:
            action_str = ''.join(ACTION_NAMES[a][0] for a in sol)
            print(f"  Level {i+1}: SOLVED in {len(sol)} steps: {action_str}")
        else:
            print(f"  Level {i+1}: NO SOLUTION FOUND — using random fallback")

        solutions.append(sol)

    return solutions


print("Step 1018e: Computing BFS solutions for LS20/9607627b levels (ramp+moving-trigger-aware)...")
_SOLUTIONS = _compute_solutions()
if _SOLUTIONS is None:
    _SOLUTIONS = [None] * 7

print(f"  Solutions: {['OK' if s else 'FAIL' for s in _SOLUTIONS]}")
print(f"  Lengths: {[len(s) if s else 0 for s in _SOLUTIONS]}")


# ─── Substrate class ───

class Ls20SolverSubstrate:
    """LS20 multi-level BFS solver substrate (Step 1018e)."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.RandomState(seed)
        self._n_actions = 4
        self._is_ls20 = False
        self._level_idx = 0
        self._step_in_seq = 0

    def set_game(self, n_actions: int):
        self._n_actions = n_actions
        self._is_ls20 = (n_actions == 4)
        self._level_idx = 0
        self._step_in_seq = 0

    def process(self, obs: np.ndarray) -> int:
        if not self._is_ls20:
            return int(self._rng.randint(0, self._n_actions))

        if self._level_idx >= len(_SOLUTIONS):
            return int(self._rng.randint(0, self._n_actions))

        sol = _SOLUTIONS[self._level_idx]
        if sol is None or len(sol) == 0:
            return int(self._rng.randint(0, self._n_actions))

        action = sol[self._step_in_seq % len(sol)]
        self._step_in_seq += 1
        return int(action)

    def on_level_transition(self):
        if not self._is_ls20:
            return
        sol = _SOLUTIONS[self._level_idx] if self._level_idx < len(_SOLUTIONS) else None
        if sol is not None and self._step_in_seq >= len(sol):
            self._level_idx += 1
        self._step_in_seq = 0


SUBSTRATE_CLASS = Ls20SolverSubstrate
