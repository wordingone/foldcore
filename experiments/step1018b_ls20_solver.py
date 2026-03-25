"""
step1018b_ls20_solver.py — LS20/9607627b Multi-Level BFS Solver v2 (Direction 2, Sub-mode A).

SUBSTRATE-ONLY FILE. No main(), no harness, no run loop.
Run via: python run_experiment.py --step 1018b --substrate experiments/step1018b_ls20_solver.py --steps 10000 --seeds 10

FIX FROM 1018: on_level_transition() was advancing _level_idx on BOTH level-complete AND death.
  ArcGameWrapper calls on_level_transition() for: (1) cl>level, (2) done=True (death/end).
  Both can fire in same step on level completion → double advance (_level_idx jumps by 2).
  On death alone → premature advance → substrate plays wrong level's BFS.

  Fix: only advance _level_idx if BFS solution was completed (_step_in_seq >= len(sol)).
  Death: _step_in_seq < len(sol) → stay at current level, restart from step 0.
  Level complete: _step_in_seq >= len(sol) → advance to next level.
  Double-fire: first call advances + resets step to 0; second call sees 0 < len → stays.

DIRECTION 2: Per-game prescription allowed (Jun directive 2026-03-24).
  - Reads LS20 source directly for level geometry
  - BFS to find optimal path through each level (shape/color/rotation matching + walls)
  - Solutions replayed per-level; restart from step 0 on death.

PURPOSE: LS20 multi-level ceiling measurement after execution fix.
  Expected: LS20 L1=100% (all seeds complete Level 1 with correct tracking).
  Higher levels depend on BFS model accuracy (Level 3+ still under investigation).

KILL: N/A — diagnostic only.
BUDGET: 10K steps, 10 seeds.
"""
import sys
import os
import types
from collections import deque
import numpy as np

# ─── Constants from ls20.py ───
# tnkekoeuk = [epqvqkpffo=12, jninpsotet=9, bejggpjowv=14, tqogkgimes=8]
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
    "solver": "bfs_ls20_direction2",
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
    """Install mock arcengine. Returns previous value (None if not installed)."""
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
    """Restore arcengine to previous value after mock use."""
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
    """Does sprite at (wx,wy) overlap with player's bounding box at (px,py)?"""
    return wx >= px and wx < px + PLAYER_SIZE and wy >= py and wy < py + PLAYER_SIZE


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

    # Walls (ihdgageizm tagged sprites)
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
        'moves_per_life': moves_per_life,
        'step_counter': step_counter,
        'n_goals': len(goals),
    }


# ─── BFS solver ───

def _bfs_solve(ld, max_path_len=200):
    """
    BFS to find shortest action sequence that achieves all goals in one life attempt.

    State: (px, py, shape, color, rot, goals_done_frozenset, collected_frozenset)
    Budget: moves_per_life per "life segment" between collectibles.
    We track budget with steps_remaining included in state.
    """
    px0, py0 = ld['px0'], ld['py0']
    s0, c0, r0 = ld['start_shape'], ld['start_color'], ld['start_rot']
    goals = ld['goals']
    walls = ld['walls']
    rot_t = ld['rot_triggers']
    col_t = ld['color_triggers']
    shp_t = ld['shape_triggers']
    colls = ld['collectibles']
    mpl = ld['moves_per_life']
    n_goals = ld['n_goals']

    all_done = frozenset(range(n_goals))
    n_colls = len(colls)
    all_colls = frozenset(range(n_colls))

    # Pre-compute goal wall positions (goals are walls until matched)
    goal_positions = {(gx, gy): gi for gi, (gx, gy, *_) in enumerate(goals)}

    # State: (px, py, sh, co, ro, gdone_frozenset, coll_frozenset, steps_left)
    init_state = (px0, py0, s0, c0, r0, frozenset(), all_colls, mpl)

    queue = deque([(init_state, [])])
    visited = {init_state}

    while queue:
        state, path = queue.popleft()

        if len(path) >= max_path_len:
            continue

        px, py, sh, co, ro, gdone, coll, sleft = state

        if gdone == all_done:
            return path

        if sleft <= 0:
            continue  # ran out of budget in this approach

        for action, (dx, dy) in enumerate(DIRS):
            npx, npy = px + dx, py + dy

            # Bounds check
            if npx < 0 or npy < 0 or npx + PLAYER_SIZE > 64 or npy + PLAYER_SIZE > 64:
                continue

            # Wall check (ihdgageizm)
            blocked = False
            for wx, wy in walls:
                if _overlaps(wx, wy, npx, npy):
                    blocked = True
                    break
            if blocked:
                continue

            # Goal check: unmatched goals act as walls
            goal_entered = None
            for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                if gi in gdone:
                    continue  # goal already achieved, not blocking
                if _overlaps(gx, gy, npx, npy):
                    # Check state match BEFORE triggers (conservative — correct for most cases)
                    if sh == rs and co == rc and ro == rr:
                        goal_entered = gi
                    else:
                        blocked = True
                        break
            if blocked:
                continue

            # Apply triggers at new position
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

            # Update goals with post-trigger state
            new_gdone = set(gdone)
            if goal_entered is not None:
                new_gdone.add(goal_entered)
            # Also check goals at this position with updated state
            for gi, (gx, gy, rs, rc, rr) in enumerate(goals):
                if gi in new_gdone:
                    continue
                if _overlaps(gx, gy, npx, npy):
                    if nsh == rs and nco == rc and nro == rr:
                        new_gdone.add(gi)

            # Collectible check (resets step counter)
            new_coll = set(coll)
            new_sleft = sleft - 1
            for ci in list(coll):
                cx, cy = colls[ci]
                if _overlaps(cx, cy, npx, npy):
                    new_coll.discard(ci)
                    new_sleft = mpl  # reset step counter

            new_state = (npx, npy, nsh, nco, nro,
                         frozenset(new_gdone), frozenset(new_coll), new_sleft)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [action]))

    return None  # no solution found within budget


# ─── Compute solutions at module load ───

def _compute_solutions():
    """Parse ls20.py and run BFS for all 7 levels. Returns list of action sequences."""
    prev_arcengine = _install_mock_arcengine()

    # Find ls20.py
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

    # Import ls20 module with mock arcengine active
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

    # Restore real arcengine BEFORE running BFS (BFS doesn't need arcengine)
    _restore_arcengine(prev_arcengine)

    solutions = []
    for i, level in enumerate(ls20_levels):
        ld = _extract_level(level, i)
        print(f"  Level {i+1}: player=({ld['px0']},{ld['py0']}), "
              f"goals={len(ld['goals'])}, walls={len(ld['walls'])}, "
              f"moves_per_life={ld['moves_per_life']}")

        # Try BFS with increasing max_path_len
        sol = None
        for max_len in [50, 100, 150, 200]:
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
print("Step 1018b: Computing BFS solutions for LS20/9607627b levels...")
_SOLUTIONS = _compute_solutions()
if _SOLUTIONS is None:
    _SOLUTIONS = [None] * 7

print(f"  Solutions: {['OK' if s else 'FAIL' for s in _SOLUTIONS]}")
print(f"  Lengths: {[len(s) if s else 0 for s in _SOLUTIONS]}")


# ─── Substrate class ───

class Ls20SolverSubstrate:
    """
    LS20 multi-level BFS solver substrate (Step 1018).

    For LS20 (n_actions=4): replays BFS solutions level by level.
    For other games (FT09/VC33/CIFAR): random actions.

    The substrate detects level transitions via on_level_transition().
    Within each level, the solution sequence is repeated (handles death/restart).
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

        # Get current level's solution
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
        This prevents: (1) death corrupting level tracking, (2) double-fire
        on level completion (cl>level + done both firing same step).
        """
        if not self._is_ls20:
            return  # non-LS20 game — ignore, set_game() will reset on next LS20 run
        sol = _SOLUTIONS[self._level_idx] if self._level_idx < len(_SOLUTIONS) else None
        if sol is not None and self._step_in_seq >= len(sol):
            # Solution completed → advance to next level
            self._level_idx += 1
        # else: death or double-fire → stay at current level
        self._step_in_seq = 0  # always restart from beginning of current level's BFS


SUBSTRATE_CLASS = Ls20SolverSubstrate
