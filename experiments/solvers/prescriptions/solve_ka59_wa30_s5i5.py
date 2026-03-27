"""
Full-chain solver for KA59, WA30, S5I5.
Uses direct game module import with deepcopy-based BFS.
Each game gets custom action enumeration for efficiency.

Usage: PYTHONUTF8=1 python solve_ka59_wa30_s5i5.py [game_id]
"""
import sys, json, time, os, copy, copyreg, hashlib, importlib
import numpy as np
from collections import deque
import logging
logging.disable(logging.INFO)

from arcengine import GameAction, GameState, ActionInput

# Fix deepcopy for GameAction enum
def pickle_gameaction(ga):
    return (GameAction.__getitem__, (ga.name,))
copyreg.pickle(GameAction, pickle_gameaction)

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

GAME_PATHS = {
    'ka59': ('B:/M/the-search/environment_files/ka59/9f096b4a', 'ka59', 'Ka59'),
    'wa30': ('B:/M/the-search/environment_files/wa30/ee6fef47', 'wa30', 'Wa30'),
    's5i5': ('B:/M/the-search/environment_files/s5i5/a48e4b1d', 's5i5', 'S5i5'),
}

GAME_LEVELS = {'ka59': 7, 'wa30': 9, 's5i5': 8}

ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}


def encode_click(x, y):
    return 7 + y * 64 + x

def decode_action(a):
    if a < 7: return f"KB{a}"
    ci = a - 7
    return f"CL({ci%64},{ci//64})"


def load_game_class(game_id):
    path, mod_name, cls_name = GAME_PATHS[game_id]
    if path not in sys.path:
        sys.path.insert(0, path)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def game_step(game, action):
    """Execute action. Return (frame_2d, levels_completed, done)."""
    if action < 7:
        ga = ARCAGI3_TO_GA[action]
        if ga == GameAction.ACTION6:
            ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
        else:
            ai = ActionInput(id=ga, data={})
    else:
        ci = action - 7
        ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    try:
        r = game.perform_action(ai, raw=True)
    except Exception as e:
        return None, 0, False
    if r is None:
        return None, 0, False
    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    done = r.state in (GameState.GAME_OVER, GameState.WIN)
    return f, r.levels_completed, done


def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def make_at_level(game_cls, level_idx):
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


# ============================================================
# KA59 ABSTRACT MODEL
# ============================================================
def ka59_extract_state(game):
    """Extract abstract state from KA59 game: (active_idx, block_positions_tuple)."""
    players = game.current_level.get_sprites_by_tag("xlfuqjygey")
    active = game.ascpmvdpwj
    # Bombs/explosives
    bombs = game.current_level.get_sprites_by_tag("nnckfubbhi")
    # Explosive strips
    strips = game.current_level.get_sprites_by_tag("gobzaprasa")

    active_idx = -1
    player_pos = []
    for i, p in enumerate(players):
        player_pos.append((p.x, p.y))
        if p is active:
            active_idx = i

    bomb_pos = tuple((b.x, b.y) for b in bombs)
    strip_states = tuple((s.x, s.y, int(s.pixels[0, 0])) for s in strips)

    return (active_idx, tuple(player_pos), bomb_pos, strip_states)


def ka59_get_actions(game):
    """Get valid actions for KA59 at current state."""
    actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
    # Click on non-active players
    players = game.current_level.get_sprites_by_tag("xlfuqjygey")
    active = game.ascpmvdpwj
    for p in players:
        if p is not active:
            cx = p.x + p.width // 2
            cy = p.y + p.height // 2
            actions.append(encode_click(cx, cy))
    return actions


# ============================================================
# WA30 ABSTRACT MODEL
# ============================================================
def wa30_extract_state(game):
    """Extract abstract state from WA30 game."""
    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    blocks = game.current_level.get_sprites_by_tag("geezpjgiyd")
    # AI movers
    ai_k = game.current_level.get_sprites_by_tag("kdweefinfi")
    ai_y = game.current_level.get_sprites_by_tag("ysysltqlke")

    block_pos = tuple(sorted((b.x, b.y) for b in blocks))
    ai_k_pos = tuple(sorted((a.x, a.y) for a in ai_k))
    ai_y_pos = tuple(sorted((a.x, a.y) for a in ai_y))

    # Carrying state
    carrying = {}
    for a in ai_k + ai_y:
        if a in game.nsevyuople:
            carried = game.nsevyuople[a]
            carrying[id(a)] = (carried.x, carried.y)

    player_carry = None
    if player in game.nsevyuople:
        carried = game.nsevyuople[player]
        player_carry = (carried.x, carried.y)

    return (player.x, player.y, player.rotation, player_carry,
            block_pos, ai_k_pos, ai_y_pos)


# ============================================================
# S5I5 ABSTRACT MODEL
# ============================================================
def s5i5_extract_state(game):
    """Extract abstract state from S5I5 game."""
    pieces = game.current_level.get_sprites_by_tag("zylvdxoiuq")
    arms = game.current_level.get_sprites_by_tag("agujdcrunq")

    piece_pos = tuple(sorted((p.x, p.y) for p in pieces))
    arm_states = tuple(sorted((a.x, a.y, a.width, a.height) for a in arms))

    return (piece_pos, arm_states)


def s5i5_get_click_actions(game):
    """Get click actions for S5I5: panels and rotation buttons."""
    actions = []

    # Panels (gdgcpukdrl) - click left or right half
    panels = game.current_level.get_sprites_by_tag("gdgcpukdrl")
    for p in panels:
        is_horiz = p.width > p.height
        if is_horiz:
            # Left half
            lx = p.x + p.width // 4
            ly = p.y + p.height // 2
            actions.append(encode_click(lx, ly))
            # Right half
            rx = p.x + 3 * p.width // 4
            ry = p.y + p.height // 2
            actions.append(encode_click(rx, ry))
        else:
            # Top half
            tx = p.x + p.width // 2
            ty = p.y + p.height // 4
            actions.append(encode_click(tx, ty))
            # Bottom half
            bx = p.x + p.width // 2
            by = p.y + 3 * p.height // 4
            actions.append(encode_click(bx, by))

    # Rotation buttons (myzmclysbl)
    rot_buttons = game.current_level.get_sprites_by_tag("myzmclysbl")
    for r in rot_buttons:
        cx = r.x + r.width // 2
        cy = r.y + r.height // 2
        actions.append(encode_click(cx, cy))

    return actions


# ============================================================
# DEEPCOPY BFS SOLVER
# ============================================================
def deepcopy_bfs(game_cls, level_idx, game_id,
                 max_depth=80, max_states=2000000, time_limit=300):
    """BFS with deepcopy for state branching."""
    t0 = time.time()

    game = make_at_level(game_cls, level_idx)

    # Get initial frame for hash
    init_game = copy.deepcopy(game)
    init_frame, _, _ = game_step(init_game, encode_click(0, 0))

    # Determine actions based on game
    if game_id == 'ka59':
        get_actions = lambda g: ka59_get_actions(g)
        get_state = lambda g: ka59_extract_state(g)
    elif game_id == 'wa30':
        get_actions = lambda g: [0, 1, 2, 3, 4]  # Fixed KB actions
        get_state = lambda g: wa30_extract_state(g)
    elif game_id == 's5i5':
        get_actions = lambda g: s5i5_get_click_actions(g)
        get_state = lambda g: s5i5_extract_state(g)
    else:
        return None

    # Get initial state hash
    init_state = get_state(game)

    # BFS: (game_copy, action_sequence)
    queue = deque([(game, [])])
    visited = {init_state}
    explored = 0
    max_d = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, explored={explored}, depth={max_d}, visited={len(visited)})")
            return None

        cur_game, actions = queue.popleft()
        depth = len(actions)

        if depth > max_d:
            max_d = depth
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={max_d} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)")

        if depth >= max_depth:
            continue

        cur_actions = get_actions(cur_game)

        for action in cur_actions:
            g2 = copy.deepcopy(cur_game)
            frame, levels, done = game_step(g2, action)
            explored += 1

            if levels > 0:
                sol = actions + [action]
                print(f"    SOLVED! {len(sol)} actions, {explored} explored, {time.time()-t0:.1f}s")
                return sol

            if done or frame is None:
                continue

            state = get_state(g2)
            if state not in visited:
                visited.add(state)
                queue.append((g2, actions + [action]))

            if len(visited) >= max_states:
                print(f"    STATE LIMIT ({max_states}, d={max_d})")
                return None

    print(f"    EXHAUSTED (d={max_d}, explored={explored})")
    return None


# ============================================================
# REPLAY-BASED BFS (fallback for games where deepcopy is slow)
# ============================================================
def replay_bfs(game_cls, level_idx, game_id,
               max_depth=50, max_states=500000, time_limit=300):
    """BFS using replay from scratch (slower but safer)."""
    t0 = time.time()

    # Get initial frame
    g0 = make_at_level(game_cls, level_idx)
    init_frame, _, _ = game_step(g0, encode_click(0, 0))
    init_hash = frame_hash(init_frame)

    # Scan unique actions
    print(f"    Scanning actions...")
    if game_id == 'ka59':
        # Get click targets from initial state
        g0 = make_at_level(game_cls, level_idx)
        base_actions = [0, 1, 2, 3]
        players = g0.current_level.get_sprites_by_tag("xlfuqjygey")
        active = g0.ascpmvdpwj
        for p in players:
            if p is not active:
                cx = p.x + p.width // 2
                cy = p.y + p.height // 2
                base_actions.append(encode_click(cx, cy))
    elif game_id == 'wa30':
        base_actions = [0, 1, 2, 3, 4]
    elif game_id == 's5i5':
        g0 = make_at_level(game_cls, level_idx)
        base_actions = s5i5_get_click_actions(g0)
    else:
        return None

    # Filter to unique-effect actions
    groups = {}
    for a in base_actions:
        g = make_at_level(game_cls, level_idx)
        frame, levels, _ = game_step(g, a)
        if frame is None: continue
        if levels > 0:
            return [a]
        h = frame_hash(frame)
        if h != init_hash and h not in groups:
            groups[h] = a

    unique = list(groups.values())
    print(f"    {len(unique)} unique actions: {[decode_action(a) for a in unique[:20]]}")
    if not unique:
        return None

    # BFS
    queue = deque([()])
    visited = {init_hash}
    explored = 0
    max_d = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, e={explored}, d={max_d})")
            return None

        seq = queue.popleft()
        depth = len(seq)
        if depth > max_d:
            max_d = depth
            el = time.time() - t0
            print(f"      d={max_d} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s")

        if depth >= max_depth:
            continue

        for action in unique:
            actions = list(seq) + [action]
            g = make_at_level(game_cls, level_idx)
            for a in actions:
                frame, levels, done = game_step(g, a)
                if done or frame is None:
                    break
            explored += 1

            if levels > 0:
                print(f"    SOLVED! {len(actions)} actions, {explored} explored, {time.time()-t0:.1f}s")
                return actions

            if done or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={max_d})")
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


# ============================================================
# KA59 ABSTRACT SOLVER
# ============================================================
def ka59_abstract_solve(game_cls, level_idx, time_limit=300):
    """
    Abstract model solver for KA59.
    State = (active_idx, player_positions, bomb_positions)
    Movements use step_size=3.
    """
    t0 = time.time()
    print(f"    KA59 abstract solver level {level_idx+1}")

    game = make_at_level(game_cls, level_idx)
    step = 3  # jesnwclftg

    # Extract entities
    players = game.current_level.get_sprites_by_tag("xlfuqjygey")
    walls = game.current_level.get_sprites_by_tag("divgcilurm")
    gulches = game.current_level.get_sprites_by_tag("vwjqkxkyxm")
    goals_r = game.current_level.get_sprites_by_tag("rktpmjcpkt")
    goals_u = game.current_level.get_sprites_by_tag("ucjzrlvfkb")
    bombs = game.current_level.get_sprites_by_tag("nnckfubbhi")
    strips = game.current_level.get_sprites_by_tag("gobzaprasa")
    enemies = game.current_level.get_sprites_by_tag("Enemy")

    grid_w, grid_h = game.current_level.grid_size

    print(f"      Grid: {grid_w}x{grid_h}, {len(players)} players, {len(goals_r)} goals_r, {len(goals_u)} goals_u")
    print(f"      {len(bombs)} bombs, {len(strips)} strips, {len(enemies)} enemies")

    # Build wall set (solid obstacles at pixel level)
    wall_rects = []
    for w in walls:
        wall_rects.append((w.x, w.y, w.width, w.height))

    # Build gulch set
    gulch_rects = []
    for g_spr in gulches:
        gulch_rects.append((g_spr.x, g_spr.y, g_spr.width, g_spr.height))

    print(f"      Walls: {wall_rects}")
    print(f"      Gulches: {gulch_rects}")
    print(f"      Players: {[(p.x, p.y, p.width, p.height) for p in players]}")
    print(f"      Goals_r: {[(g.x, g.y, g.width, g.height) for g in goals_r]}")
    print(f"      Goals_u: {[(g.x, g.y, g.width, g.height) for g in goals_u]}")
    print(f"      Bombs: {[(b.x, b.y) for b in bombs]}")

    # This game is too complex for a pure abstract model due to:
    # - Collision detection using pixel overlap
    # - Explosive mechanics (strips that push when detonated)
    # - Multi-step animations (push chains take multiple steps)
    # - Enemy movement
    # Fall through to deepcopy BFS
    return None


# ============================================================
# WA30 ABSTRACT SOLVER
# ============================================================
def wa30_abstract_solve(game_cls, level_idx, time_limit=300):
    """
    Abstract model solver for WA30.
    State = (player_pos, player_rot, carrying?, block_positions, ai_positions)
    """
    t0 = time.time()
    print(f"    WA30 abstract solver level {level_idx+1}")

    game = make_at_level(game_cls, level_idx)
    step = 4  # celomdfhbh

    player = game.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
    blocks = game.current_level.get_sprites_by_tag("geezpjgiyd")
    targets1 = game.current_level.get_sprites_by_tag("fsjjayjoeg")
    targets2 = game.current_level.get_sprites_by_tag("zqxwgacnue")
    ai_k = game.current_level.get_sprites_by_tag("kdweefinfi")
    ai_y = game.current_level.get_sprites_by_tag("ysysltqlke")
    hazards = game.current_level.get_sprites_by_tag("bnzklblgdk")

    print(f"      Player: ({player.x},{player.y}), {len(blocks)} blocks")
    print(f"      Targets1: {len(targets1)}, Targets2: {len(targets2)}")
    print(f"      AI_k: {len(ai_k)}, AI_y: {len(ai_y)}")
    print(f"      Hazards: {len(hazards)}")

    # Fall through to deepcopy BFS (AI movers make abstract model complex)
    return None


# ============================================================
# MAIN SOLVER
# ============================================================
def solve_game(game_id, start_level=0, known_solutions=None):
    """Solve all levels of a game."""
    print(f"\n{'='*60}")
    print(f"SOLVING {game_id.upper()} ({GAME_LEVELS[game_id]} levels)")
    print(f"{'='*60}")

    game_cls = load_game_class(game_id)
    total = GAME_LEVELS[game_id]

    results = {
        'game': game_id, 'total_levels': total,
        'method': 'deepcopy_bfs',
        'levels': {}, 'full_sequence': [],
    }

    # Load known solutions if provided
    if known_solutions is None:
        known_solutions = {}

    current = 0

    for lnum in range(1, total + 1):
        level_idx = lnum - 1
        lkey = f'L{lnum}'

        if lnum <= start_level and lkey in known_solutions:
            sol = known_solutions[lkey]
            print(f"\n  {lkey}: Using known solution ({len(sol)} actions)")
            results['levels'][lkey] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'method': 'known'
            }
            results['full_sequence'].extend(sol)
            current = lnum
            continue

        if current < lnum - 1:
            print(f"\n  {lkey}: Skipping (previous level unsolved)")
            break

        print(f"\n  {lkey} (level_idx={level_idx}):")
        t0 = time.time()

        # Try deepcopy BFS first
        sol = deepcopy_bfs(game_cls, level_idx, game_id,
                          max_depth=80, max_states=2000000, time_limit=300)

        elapsed = time.time() - t0

        if sol is not None:
            results['levels'][lkey] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'time': round(elapsed, 2),
                'method': 'deepcopy_bfs'
            }
            results['full_sequence'].extend(sol)
            current = lnum

            # Verify
            g = make_at_level(game_cls, level_idx)
            levels = 0
            for a in sol:
                _, levels, done = game_step(g, a)
                if done: break
            print(f"    Verify: levels_completed={levels}")
        else:
            results['levels'][lkey] = {
                'status': 'UNSOLVED', 'time': round(elapsed, 2)
            }
            print(f"    UNSOLVED ({elapsed:.1f}s)")
            # Don't break - try next level anyway with set_level
            # break

    results['max_level_solved'] = current
    results['total_actions'] = len(results['full_sequence'])

    # Save
    outpath = os.path.join(RESULTS_DIR, f'{game_id}_fullchain.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  SAVED: {outpath}")
    print(f"  {current}/{total} levels, {results['total_actions']} actions")

    return results


if __name__ == '__main__':
    games_arg = sys.argv[1:] if len(sys.argv) > 1 else ['wa30', 'ka59', 's5i5']

    # Known solutions from previous runs
    WA30_KNOWN = {
        'L1': [0, 0, 2, 0, 0, 0, 2, 2, 4, 3, 3, 3, 4, 0, 3, 3, 4, 1, 2, 2, 4, 1, 4, 0, 0, 4],
    }

    S5I5_KNOWN = {
        'L1': [2847, 2847, 2847, 2847, 2847, 2847, 1396, 1396, 1396, 1396, 1396, 1396, 1396],
        'L2': [3473, 3473, 3473, 3473, 3473, 3473, 3473, 3473, 3489, 3489, 3489, 3489, 3489, 3489, 3489, 3489, 3473, 3503, 3503, 3503, 3519, 3519, 3519, 3519, 3519, 3519],
    }

    for gid in games_arg:
        known = {}
        if gid == 'wa30':
            known = WA30_KNOWN
            start = 1
        elif gid == 's5i5':
            known = S5I5_KNOWN
            start = 2
        elif gid == 'ka59':
            known = {}
            start = 0
        else:
            start = 0

        solve_game(gid, start_level=start, known_solutions=known)
