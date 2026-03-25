"""
step1018d_ls20_solver.py — LS20/9607627b Multi-Level BFS Solver v4 (Ramp-Aware).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1018d --substrate experiments/step1018d_ls20_solver.py --steps 10000 --seeds 10

FIX FROM 1018b: BFS didn't model ramp/conveyor mechanic (gbvqrjtaqo/yjgargdic_* sprites).
  Root cause of L3 failure: ramp at (8, 5) with direction=right. Player entering (9, 5)
  gets slid RIGHT by ramp.width * cells = 5*5 = 25px → ends at (34, 5).
  BFS modeled player navigating from (9, 5) → complete path divergence from real game.
  Ramps consume 14 game steps per activation (8 push + 6 return) but do NOT decrement
  the step counter (euemavvxz active → mfyzdfvxsm() not called).
  Levels with ramps: L3 (1 ramp), L4 (2 ramps), L5 (2 ramps). L1, L2, L6, L7: no ramps.

FIX FROM 1018c: arc_agi GAME_OVER reset bug — see util_arcagi3.py.

MECHANICS MODELED:
  1. Player movement with wall/goal blocking.
  2. Triggers (shape/color/rotation) → update state.
  3. Collectibles → reset step counter.
  4. Goal completion: player at goal position with matching state.
  5. Multi-life: sleft tracks step budget per life (moves_per_life).
  6. Ramps (gbvqrjtaqo): proper bounding-box detection, slide to wall-bounded destination,
     step counter unchanged, triggers/collectibles at destination, pbznecvnfr NOT called
     this step (need blocked-move next step if goal at destination).
  7. Blocked moves: player stays at current position, sleft decrements, goal check at
     current position (handles ramp-delivered-to-goal and edge cases).

DIRECTION 2: Per-game prescription allowed (Jun directive 2026-03-24).
KILL: N/A — diagnostic only.
BUDGET: 10K steps, 10 seeds.
"""
import sys
import os
import types
from collections import deque
import numpy as np

# ─── Constants from ls20.py ───
TNKEKOEUK = [12, 9, 14, 8]
DHKSVILBB = [0, 90, 180, 270]
N_SHAPES = 6
N_COLORS = 4
N_ROTS = 4
PLAYER_SIZE = 5

# Action mapping (0-indexed):
# 0 = ACTION1 = y -= 5 = UP
# 1 = ACTION2 = y += 5 = DOWN
# 2 = ACTION3 = x -= 5 = LEFT
# 3 = ACTION4 = x += 5 = RIGHT
DIRS = [(0, -PLAYER_SIZE), (0, PLAYER_SIZE), (-PLAYER_SIZE, 0), (PLAYER_SIZE, 0)]
ACTION_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT']

CONFIG = {
    "solver": "bfs_ls20_ramp_aware",
    "levels": 7,
    "direction": 2,
}


# ─── Mock arcengine for parsing ls20.py ───

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


# ─── Level data extraction ───

def _color_idx(val):
    return TNKEKOEUK.index(val)


def _rot_idx(val):
    return DHKSVILBB.index(val)


def _overlaps(wx, wy, px, py):
    """Sprite at (wx,wy) size PLAYER_SIZE overlaps player bounding box at (px,py)."""
    return wx >= px and wx < px + PLAYER_SIZE and wy >= py and wy < py + PLAYER_SIZE


def _box_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    """Full bounding-box overlap check between two rectangles."""
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _compute_ramp_dest(rx, ry, dx, dy, ramp_w, ramp_h, walls):
    """
    Compute player destination delta when entering ramp at (rx, ry) direction (dx, dy).
    Uses same logic as twkzhcfelv.ullzqnksoj().

    Returns (player_dx, player_dy) if ramp fires, or None if no valid slide.
    """
    wall_cx = rx + dx
    wall_cy = ry + dy
    cells = 0
    for k in range(1, 13):
        nskortmtde = wall_cx + dx * ramp_w * k
        wsotwbexvh = wall_cy + dy * ramp_h * k
        if (nskortmtde, wsotwbexvh) in walls:
            cells = max(0, k - 1)
            break
    if cells <= 0:
        return None
    # Ramp moves by (dx * ramp_w * cells, dy * ramp_h * cells)
    # Player moves by the same delta
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

    # Triggers
    rot_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('rhsxkxzdjz')]
    color_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('soyhouuebz')]
    shape_triggers = [(s.x, s.y) for s in level.get_sprites_by_tag('ttfwljgohq')]

    # Collectibles
    collectibles = [(s.x, s.y) for s in level.get_sprites_by_tag('npxgalaybz')]

    # Ramps (gbvqrjtaqo / yjgargdic_*)
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
        dest = _compute_ramp_dest(s.x, s.y, dx, dy, s.width, s.height, frozenset(walls))
        if dest is not None:
            ramps.append((s.x, s.y, s.width, s.height, dest[0], dest[1]))
            # dest = (player_dx, player_dy) when ramp fires

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
        'collectibles': collectibles,
        'ramps': ramps,  # [(rx, ry, rw, rh, dest_dx, dest_dy), ...]
        'moves_per_life': moves_per_life,
        'step_counter': step_counter,
        'n_goals': len(goals),
    }


# ─── BFS solver ───

def _bfs_solve(ld, max_path_len=250):
    """
    BFS with ramp-aware mechanics.

    State: (px, py, shape, color, rot, goals_done_frozenset, collected_frozenset, steps_left)

    Transitions:
      1. Normal move (not blocked): player at new pos, apply triggers/collectibles, sleft-1.
         Goal check at new pos.
      2. Blocked move: player stays at current pos, sleft-1.
         Goal check at current pos (pbznecvnfr fires at current pos when blocked).
      3. Ramp: player at new pos, ramp fires → slide to destination.
         Apply triggers/collectibles at destination. sleft UNCHANGED.
         Goal check at destination via next step's blocked move (NOT immediate).
         Mark as post-ramp: add destination as new state without goal check.
    """
    px0, py0 = ld['px0'], ld['py0']
    s0, c0, r0 = ld['start_shape'], ld['start_color'], ld['start_rot']
    goals = ld['goals']
    walls = ld['walls']
    rot_t = ld['rot_triggers']
    col_t = ld['color_triggers']
    shp_t = ld['shape_triggers']
    colls = ld['collectibles']
    ramps = ld['ramps']
    mpl = ld['moves_per_life']
    n_goals = ld['n_goals']

    all_done = frozenset(range(n_goals))
    n_colls = len(colls)
    all_colls = frozenset(range(n_colls))

    # Goal positions for quick lookup
    goal_positions = {}  # (gx, gy) -> gi (only unmatched-state goals are walls)
    for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
        goal_positions[(gx, gy)] = gi

    # Precompute ramp overlap: (rx, ry, rw, rh, dest_dx, dest_dy)
    # Ramp fires when player's bounding box overlaps ramp's bounding box

    # State: (px, py, sh, co, ro, gdone_frozenset, coll_frozenset, steps_left)
    init_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl)

    queue = deque([(init_state, [])])
    visited = {init_state}

    def _apply_triggers(npx, npy, sh, co, ro):
        """Apply trigger effects at position (npx, npy)."""
        nsh, nco, nro = sh, co, ro
        for tx, ty in rot_t:
            if _overlaps(tx, ty, npx, npy):
                nro = (nro + 1) % N_ROTS
        for tx, ty in col_t:
            if _overlaps(tx, ty, npx, npy):
                nco = (nco + 1) % N_COLORS
        for tx, ty in shp_t:
            if _overlaps(tx, ty, npx, npy):
                nsh = (nsh + 1) % N_SHAPES
        return nsh, nco, nro

    def _apply_collectibles(npx, npy, coll, sleft):
        """Apply collectible pickup at position. Returns (new_coll, new_sleft)."""
        new_coll = set(coll)
        new_sleft = sleft
        for ci in list(coll):
            cx, cy = colls[ci]
            if _overlaps(cx, cy, npx, npy):
                new_coll.discard(ci)
                new_sleft = mpl  # reset
        return frozenset(new_coll), new_sleft

    def _check_goals_at(px, py, sh, co, ro, gdone):
        """Check pbznecvnfr at (px, py) — goal completion when player is here."""
        new_gdone = set(gdone)
        for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
            if gi in gdone:
                continue
            if px == gx and py == gy and sh == rs and co == rc and ro == rr:
                new_gdone.add(gi)
        return frozenset(new_gdone)

    def _check_ramp(npx, npy, sh, co, ro, coll, sleft, gdone):
        """
        Check if player at (npx, npy) triggers a ramp.
        Returns (final_x, final_y, nsh, nco, nro, new_coll, new_sleft, True)
        or None if no ramp.
        Sleft is UNCHANGED for ramp (step counter not decremented).
        """
        for rx, ry, rw, rh, dest_dx, dest_dy in ramps:
            if _box_overlap(rx, ry, rw, rh, npx, npy, PLAYER_SIZE, PLAYER_SIZE):
                # Ramp fires: slide player
                final_x = npx + dest_dx
                final_y = npy + dest_dy
                # Apply triggers at destination (txnfzvzetn after ramp)
                nsh, nco, nro = _apply_triggers(final_x, final_y, sh, co, ro)
                # Apply collectibles at destination
                new_coll, new_sleft = _apply_collectibles(final_x, final_y, coll, sleft)
                # sleft is UNCHANGED if no collectible reset (ramp doesn't decrement)
                if new_sleft == sleft:
                    new_sleft = sleft  # explicitly unchanged
                # pbznecvnfr NOT called after ramp — return destination as new state
                return (final_x, final_y, nsh, nco, nro, new_coll, new_sleft)
        return None

    while queue:
        state, path = queue.popleft()

        if len(path) >= max_path_len:
            continue

        px, py, sh, co, ro, gdone, coll, sleft = state

        if gdone == all_done:
            return path

        if sleft <= 0:
            continue

        for action, (dx, dy) in enumerate(DIRS):
            npx, npy = px + dx, py + dy

            # Determine if blocked
            is_blocked = False

            # Bounds check
            if npx < 0 or npy < 0 or npx + PLAYER_SIZE > 64 or npy + PLAYER_SIZE > 64:
                is_blocked = True
            else:
                # Wall check
                for wx, wy in walls:
                    if _overlaps(wx, wy, npx, npy):
                        is_blocked = True
                        break

                if not is_blocked:
                    # Goal check: unmatched goals act as walls
                    for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                        if gi in gdone:
                            continue
                        if _overlaps(gx, gy, npx, npy):
                            if not (sh == rs and co == rc and ro == rr):
                                is_blocked = True
                                break

            if is_blocked:
                # Player stays at (px, py). sleft-1. pbznecvnfr at (px, py).
                new_sleft = sleft - 1
                if new_sleft <= 0:
                    continue
                new_gdone = _check_goals_at(px, py, sh, co, ro, gdone)
                new_state = (px, py, sh, co, ro, new_gdone, coll, new_sleft)
                if new_state not in visited:
                    visited.add(new_state)
                    if new_gdone == all_done:
                        return path + [action]
                    queue.append((new_state, path + [action]))
                continue

            # Not blocked: player moves to (npx, npy)
            # Apply triggers at (npx, npy)
            nsh, nco, nro = _apply_triggers(npx, npy, sh, co, ro)

            # Check ramp at (npx, npy)
            ramp_result = _check_ramp(npx, npy, nsh, nco, nro, coll, sleft, gdone)
            if ramp_result is not None:
                final_x, final_y, rsh, rco, rro, rcoll, rsleft = ramp_result
                # After ramp: player at (final_x, final_y), sleft UNCHANGED
                # pbznecvnfr NOT called — add as new state, goal check happens next step
                new_state = (final_x, final_y, rsh, rco, rro, gdone, rcoll, rsleft)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action]))
                continue

            # Normal move: collectibles at (npx, npy), then goal check
            new_coll, new_sleft = _apply_collectibles(npx, npy, coll, sleft - 1)

            # Goal check at (npx, npy)
            new_gdone = set(gdone)
            for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                if gi in gdone:
                    continue
                if _overlaps(gx, gy, npx, npy):
                    if nsh == rs and nco == rc and nro == rr:
                        new_gdone.add(gi)

            new_state = (npx, npy, nsh, nco, nro,
                         frozenset(new_gdone), new_coll, new_sleft)

            if new_state not in visited:
                visited.add(new_state)
                if frozenset(new_gdone) == all_done:
                    return path + [action]
                if new_sleft > 0:
                    queue.append((new_state, path + [action]))

    return None


# ─── Compute solutions at module load ───

def _compute_solutions():
    """Parse ls20.py and run BFS for all 7 levels. Returns list of action sequences."""
    prev_arcengine = _install_mock_arcengine()

    ls20_path = None
    search_roots = [
        'B:/M/the-search',
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    for root in search_roots:
        candidate = os.path.join(root, 'environment_files', 'ls20', '9607627b', 'ls20.py')
        if os.path.exists(candidate):
            ls20_path = candidate
            break

    if ls20_path is None:
        _restore_arcengine(prev_arcengine)
        print("WARNING: ls20.py not found — using fallback random actions")
        return None

    import importlib.util
    spec = importlib.util.spec_from_file_location('_ls20_env', ls20_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        _restore_arcengine(prev_arcengine)
        print(f"WARNING: failed to load ls20.py: {e} — using fallback")
        return None

    ls20_levels = mod.levels
    print(f"  Loaded {len(ls20_levels)} LS20 levels")

    _restore_arcengine(prev_arcengine)

    solutions = []
    for i, level in enumerate(ls20_levels):
        ld = _extract_level(level, i)
        n_ramps = len(ld['ramps'])
        print(f"  Level {i+1}: player=({ld['px0']},{ld['py0']}), "
              f"goals={len(ld['goals'])}, walls={len(ld['walls'])}, "
              f"moves_per_life={ld['moves_per_life']}, ramps={n_ramps}")
        if n_ramps > 0:
            for rx, ry, rw, rh, ddx, ddy in ld['ramps']:
                print(f"    ramp at ({rx},{ry}) → delta ({ddx},{ddy})")

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


# Compute solutions at module import time
print("Step 1018d: Computing BFS solutions for LS20/9607627b levels (ramp-aware)...")
_SOLUTIONS = _compute_solutions()
if _SOLUTIONS is None:
    _SOLUTIONS = [None] * 7

print(f"  Solutions: {['OK' if s else 'FAIL' for s in _SOLUTIONS]}")
print(f"  Lengths: {[len(s) if s else 0 for s in _SOLUTIONS]}")


# ─── Substrate class (unchanged from 1018b) ───

class Ls20SolverSubstrate:
    """
    LS20 multi-level BFS solver substrate (Step 1018d).

    For LS20 (n_actions=4): replays BFS solutions level by level.
    For other games (FT09/VC33/CIFAR): random actions.
    """

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

        # Replay action (cycling: handles death/restart)
        action = sol[self._step_in_seq % len(sol)]
        self._step_in_seq += 1
        return int(action)

    def on_level_transition(self):
        """Called by ArcGameWrapper on cl>level OR done=True (death/end).

        Only advance _level_idx if the BFS solution was completed (not a death).
        """
        if not self._is_ls20:
            return
        sol = _SOLUTIONS[self._level_idx] if self._level_idx < len(_SOLUTIONS) else None
        if sol is not None and self._step_in_seq >= len(sol):
            self._level_idx += 1
        self._step_in_seq = 0


SUBSTRATE_CLASS = Ls20SolverSubstrate
