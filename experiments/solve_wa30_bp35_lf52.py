"""
Abstract model solvers for WA30, BP35, and LF52.
Uses arc_agi API as ground-truth simulator for verification.
BFS/DFS on abstract state models extracted from game source.

Usage: PYTHONUTF8=1 python experiments/solve_wa30_bp35_lf52.py [game_id]
"""
import os
import sys
import json
import time
import copy
import hashlib
from collections import deque

os.chdir("B:/M/the-search")

import numpy as np
import arc_agi
from arcengine import GameAction, GameState


# =============================================================================
# COMMON UTILITIES
# =============================================================================

def frame_hash(frame_list):
    """Hash a list of frames for state deduplication."""
    if isinstance(frame_list, list) and len(frame_list) > 0:
        arr = np.array(frame_list[-1]) if isinstance(frame_list[-1], list) else frame_list[-1]
        return hashlib.md5(arr.tobytes()).hexdigest()
    return ""


def api_bfs(game_id, available_actions, max_depth=200, max_states=2000000):
    """
    Generic BFS using the actual game API as simulator.
    Uses frame hashing for state deduplication.

    available_actions: list of (action_id, data_or_None) tuples
                       For keyboard: (GameAction.ACTION3, None)
                       For click: (GameAction.ACTION6, {"x": px, "y": py})
    """
    arcade = arc_agi.Arcade()
    games = arcade.get_environments()
    info = next(g for g in games if game_id in g.game_id.lower())

    # We can't easily BFS with the API because we can't save/restore state.
    # Instead, we need to use the abstract model approach.
    # The API is only used for final verification.
    pass


def verify_solution(game_id, actions):
    """Verify solution via arc_agi API."""
    if not actions:
        print(f"  No actions to verify for {game_id}")
        return 0

    print(f"\nVerifying {game_id} ({len(actions)} actions)...")
    arcade = arc_agi.Arcade()
    games = arcade.get_environments()
    info = next(g for g in games if game_id in g.game_id.lower())
    env = arcade.make(info.game_id)
    obs = env.reset()

    max_level = 0
    for i, a in enumerate(actions):
        try:
            if a >= 7:
                click_idx = a - 7
                px = click_idx % 64
                py = click_idx // 64
                obs = env.step(GameAction.ACTION6, data={"x": px, "y": py})
            else:
                ga = list(GameAction)[a + 1]
                obs = env.step(ga)
        except Exception as e:
            print(f"  Error at action {i} (a={a}): {e}")
            break
        if obs is None:
            continue
        if obs.levels_completed > max_level:
            max_level = obs.levels_completed
            print(f"  Level {max_level} completed at action {i}")
        if obs.state == GameState.WIN:
            print(f"  WIN! levels={obs.levels_completed}")
            break
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at action {i}, levels={obs.levels_completed}")
            break

    print(f"  Verified: max_level={max_level}")
    return max_level


def save_result(result, game_id):
    out_path = f"B:/M/the-search/experiments/results/prescriptions/{game_id}_fullchain.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


# =============================================================================
# WA30 SOLVER
# =============================================================================

STEP = 4

def extract_wa30_level(level):
    """Extract abstract state from wa30 level."""
    sps = level.get_sprites()
    S = STEP
    walls = set()
    blocks = []
    target_cells = set()
    target2_cells = set()
    forbidden = set()
    player = None
    ai_type1 = []
    ai_type2 = []

    for sp in sps:
        tags = list(sp.tags) if hasattr(sp, 'tags') else []
        pos = (sp.x, sp.y)
        if 'wbmdvjhthc' in tags:
            player = pos
        elif 'geezpjgiyd' in tags:
            blocks.append(pos)
        elif 'fsjjayjoeg' in tags:
            for dy in range(sp.height):
                for dx in range(sp.width):
                    target_cells.add((sp.x + dx, sp.y + dy))
        elif 'zqxwgacnue' in tags:
            for dy in range(sp.height):
                for dx in range(sp.width):
                    target2_cells.add((sp.x + dx, sp.y + dy))
        elif 'kdweefinfi' in tags:
            ai_type1.append(pos)
        elif 'ysysltqlke' in tags:
            ai_type2.append(pos)
        elif 'bnzklblgdk' in tags:
            forbidden.add(pos)
        elif sp.is_collidable:
            for dy in range(0, sp.height, S):
                for dx in range(0, sp.width, S):
                    walls.add((sp.x + dx, sp.y + dy))

    for i in range(0, 64, S):
        walls.add((-S, i)); walls.add((64, i))
        walls.add((i, -S)); walls.add((i, 64))

    budget = level.get_data('StepCounter')
    return {
        'player': player, 'blocks': tuple(sorted(blocks)),
        'target_cells': target_cells, 'target2_cells': target2_cells,
        'walls': walls, 'forbidden': forbidden,
        'ai_type1': tuple(sorted(ai_type1)),
        'ai_type2': tuple(sorted(ai_type2)),
        'budget': budget,
    }


def wa30_bfs(ld, max_states=5000000):
    """BFS for wa30. Handles AI movers by skipping them for simple levels."""
    S = STEP
    walls = ld['walls']
    targets = ld['target_cells']
    forbidden = ld['forbidden']
    blocks_init = ld['blocks']
    player_start = ld['player']
    budget = ld['budget']
    has_ai = bool(ld['ai_type1'] or ld['ai_type2'])

    DIRS = [(0,-S,0),(0,S,1),(-S,0,2),(S,0,3)]
    GRAB = {0:(0,-S), 1:(0,S), 2:(-S,0), 3:(S,0)}

    def is_free(p, bs):
        return p not in walls and p not in forbidden and p not in bs and 0<=p[0]<64 and 0<=p[1]<64

    def check_win(bt):
        return all(b in targets for b in bt)

    if has_ai:
        print(f"    WARNING: Level has AI movers - solving without AI simulation")
        print(f"    AI may move blocks; solution might not verify")

    init = (player_start, 0, blocks_init, -1)
    visited = {init}
    q = deque([(init, [])])
    explored = 0; md = 0

    while q:
        (player, facing, blocks, carry), acts = q.popleft()
        d = len(acts)
        if d >= budget: continue
        if d > md:
            md = d
            if md % 10 == 0: print(f"    d={md}, q={len(q)}, v={len(visited)}")
        explored += 1
        if explored > max_states:
            print(f"    Exhausted at {explored} states, d={md}")
            return None

        bs = set(blocks)
        for aid in range(5):
            if aid < 4:
                dx,dy,nf = DIRS[aid]
                np_ = (player[0]+dx, player[1]+dy)
                if carry >= 0:
                    bp = blocks[carry]
                    off = (bp[0]-player[0], bp[1]-player[1])
                    nb = (np_[0]+off[0], np_[1]+off[1])
                    obs = bs - {bp}
                    if (np_ not in walls and np_ not in forbidden and np_ not in obs and
                        0<=np_[0]<64 and 0<=np_[1]<64 and np_!=nb and
                        nb not in walls and nb not in forbidden and nb not in obs and
                        0<=nb[0]<64 and 0<=nb[1]<64):
                        nl = list(blocks); nl[carry] = nb
                        nbt = tuple(sorted(nl))
                        nc = next(i for i,b in enumerate(nbt) if b==nb)
                        ns = (np_, facing, nbt, nc)
                        if ns not in visited:
                            visited.add(ns)
                            q.append((ns, acts+[aid]))
                else:
                    if is_free(np_, bs):
                        ns = (np_, nf, blocks, -1)
                        if ns not in visited:
                            visited.add(ns)
                            q.append((ns, acts+[aid]))
                    else:
                        ns = (player, nf, blocks, -1)
                        if ns not in visited:
                            visited.add(ns)
                            q.append((ns, acts+[aid]))
            else:
                if carry >= 0:
                    ns = (player, facing, blocks, -1)
                    if ns not in visited:
                        visited.add(ns)
                        na = acts + [4]
                        if check_win(blocks):
                            print(f"    SOLVED! d={len(na)}, explored={explored}")
                            return na
                        q.append((ns, na))
                else:
                    gx,gy = GRAB[facing]
                    tp = (player[0]+gx, player[1]+gy)
                    for bi,bp in enumerate(blocks):
                        if bp == tp:
                            ns = (player, facing, blocks, bi)
                            if ns not in visited:
                                visited.add(ns)
                                q.append((ns, acts+[4]))
                            break

    print(f"    BFS done: {explored} states, d={md}, no solution")
    return None


def solve_wa30():
    sys.path.insert(0, 'environment_files/wa30/ee6fef47')
    from wa30 import levels as WA30_LEVELS

    all_acts = []; per_level = {}
    for li in range(9):
        print(f"\n{'='*60}\nWA30 Level {li+1}/9\n{'='*60}")
        ld = extract_wa30_level(WA30_LEVELS[li])
        print(f"  Player: {ld['player']}, Blocks: {len(ld['blocks'])}, Budget: {ld['budget']}")
        print(f"  AI-1: {len(ld['ai_type1'])}, AI-2: {len(ld['ai_type2'])}")

        t0 = time.time()
        sol = wa30_bfs(ld)
        el = time.time() - t0

        if sol is None:
            per_level[f"L{li+1}"] = {"status":"UNSOLVED","time":round(el,2)}
            print(f"  FAILED ({el:.1f}s)")
            continue  # Try next level anyway
        per_level[f"L{li+1}"] = {"status":"SOLVED","actions":sol,"count":len(sol),"time":round(el,2)}
        all_acts.extend(sol)
        print(f"  Solved: {len(sol)} actions in {el:.1f}s")

    max_lv = verify_solution("wa30", all_acts)
    return {"game":"wa30","version":"ee6fef47","method":"abstract_bfs",
            "max_level":max_lv,"total_actions":len(all_acts),
            "levels":per_level,"all_actions":all_acts,
            "action_map":{"0":"UP","1":"DOWN","2":"LEFT","3":"RIGHT","4":"INTERACT"}}


# =============================================================================
# BP35 SOLVER - Using game API for BFS via saved states
# =============================================================================

def bp35_api_solve_level(env, level_num, max_actions=80):
    """
    Solve a single BP35 level using BFS on game frames.
    Since we can't save/restore state, we use UNDO (ACTION7) to backtrack.

    Strategy: DFS with undo. Try actions, if dead end, undo.
    Available: ACTION3(LEFT), ACTION4(RIGHT), ACTION6(CLICK), ACTION7(UNDO)
    """
    # Actually, bp35 has UNDO (ACTION7)! We can DFS with undo.
    # But this is very slow through the API.

    # Better approach: extract the grid data and solve abstractly.
    # Let me do this properly.
    pass


def parse_bp35_grid(grid_lines, legend):
    """Parse BP35 grid into abstract tiles. Grid is in top-to-bottom order (already reversed from source)."""
    tiles = {}
    player = None
    gem = None
    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch == ' ': continue
            tnames = legend.get(ch, [])
            if not tnames: continue
            name = tnames[0]
            tiles[(x,y)] = name
            if name == "player_right": player = (x,y)
            elif name == "fjlzdjxhant": gem = (x,y)
    return tiles, player, gem


def bp35_bfs_level(level_num, grid_lines, legend, max_states=3000000):
    """
    BFS for BP35 level with constrained click set.

    Tile types:
    - xcjjwqfzjfe: wall (o)
    - qclfkhjnaac: breakable (x)
    - fjlzdjxhant: gem (+) = win
    - ubhhgljbnpu: spike-down (v)
    - hzusueifitk: spike-up (u)
    - aknlbboysnc/jcyhkseuorf: ceiling (m/w)
    - lrpkmzabbfa: gravity-flip (g) - click to flip
    - yuuqpmlxorv: toggle-on (1) - click toggles between solid/not
    - oonshderxef: toggle-off (2) - click toggles
    - etlsaqqtjvn: expand (y) - click creates blocks in 4 adjacent empty cells

    State: (player_pos, gravity_up, broken:frozenset, toggled:frozenset, expanded:frozenset)
    """
    # Grid is stored bottom=index 0 (source grids use [::-1])
    # Keep as-is: index 0 = bottom of level, higher index = higher on screen
    tiles = {}; player = None; gem = None
    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch == ' ': continue
            tnames = legend.get(ch, [])
            if not tnames: continue
            name = tnames[0]
            tiles[(x,y)] = name
            if name == "player_right": player = (x,y)
            elif name == "fjlzdjxhant": gem = (x,y)

    if not player or not gem:
        print(f"  No player ({player}) or gem ({gem})")
        return None

    print(f"  Player: {player}, Gem: {gem}, Tiles: {len(tiles)}")

    # Classify tiles
    walls = set()
    breakable = set()
    spikes_d = set()
    spikes_u = set()
    gflips = set()
    tog_on = set()
    tog_off = set()
    expand = set()
    ceilings = set()

    for pos, name in tiles.items():
        if name == "xcjjwqfzjfe": walls.add(pos)
        elif name == "qclfkhjnaac": breakable.add(pos)
        elif name == "ubhhgljbnpu": spikes_d.add(pos)
        elif name == "hzusueifitk": spikes_u.add(pos)
        elif name == "lrpkmzabbfa": gflips.add(pos)
        elif name == "yuuqpmlxorv": tog_on.add(pos)
        elif name == "oonshderxef": tog_off.add(pos)
        elif name == "etlsaqqtjvn": expand.add(pos)
        elif name in ("aknlbboysnc","jcyhkseuorf"): ceilings.add(pos)

    print(f"  Walls:{len(walls)} Break:{len(breakable)} Spikes:{len(spikes_d)+len(spikes_u)} Gflip:{len(gflips)} Toggle:{len(tog_on)+len(tog_off)} Expand:{len(expand)}")

    def is_solid(p, broken, toggled, expanded):
        if p in walls: return True
        if p in breakable and p not in broken: return True
        if p in tog_on and p not in toggled: return True
        if p in tog_off and p in toggled: return True
        if p in expanded: return True
        return False

    def gravity_fall(pos, grav_up, broken, toggled, expanded):
        """Returns (final_pos, hit_gem, hit_spike).
        grav_up=True: dy=-1 (fall toward index 0 = bottom of level)
        grav_up=False: dy=+1 (fall toward higher indices = top of level)
        Spikes: ubhhgljbnpu (v) = spike pointing down, kills when falling down onto it
                hzusueifitk (u) = spike pointing up, kills when falling up onto it
        """
        dy = -1 if grav_up else 1
        x, y = pos
        while True:
            ny = y + dy
            np_ = (x, ny)
            if np_ == gem: return (np_, True, False)
            # Spike check: v-spikes kill when gravity is down (grav_up=True, falling toward bottom)
            # u-spikes kill when gravity is up (grav_up=False, falling toward top)
            if np_ in spikes_d and grav_up:
                return ((x,y), False, True)
            if np_ in spikes_u and not grav_up:
                return ((x,y), False, True)
            if is_solid(np_, broken, toggled, expanded):
                return ((x,y), False, False)
            if ny < -5 or ny > 60:
                return ((x,y), False, True)
            y = ny

    # Initial state: apply gravity to player
    p0, g0, s0 = gravity_fall(player, True, frozenset(), frozenset(), frozenset())
    if g0:
        return []  # instant win
    if s0:
        print(f"  Player dies on spawn")
        return None

    # State: (pos, grav_up, broken, toggled, expanded)
    init = (p0, True, frozenset(), frozenset(), frozenset())
    visited = {init}
    q = deque([(init, [])])
    explored = 0; md = 0

    while q:
        (pos, gup, broken, toggled, expanded), acts = q.popleft()
        d = len(acts)
        if d >= 80: continue
        if d > md:
            md = d
            if md % 5 == 0: print(f"    d={md}, q={len(q)}, v={len(visited)}")
        explored += 1
        if explored > max_states:
            print(f"    Exhausted at {explored} states, d={md}")
            return None

        # Move LEFT/RIGHT
        for dx in [-1, 1]:
            np_ = (pos[0]+dx, pos[1])
            if np_ == gem:
                na = acts + [2 if dx<0 else 3]
                print(f"    SOLVED (move)! d={len(na)}, explored={explored}")
                return na
            if is_solid(np_, broken, toggled, expanded): continue

            fp, fg, fs = gravity_fall(np_, gup, broken, toggled, expanded)
            if fg:
                na = acts + [2 if dx<0 else 3]
                print(f"    SOLVED (fall to gem)! d={len(na)}, explored={explored}")
                return na
            if fs: continue

            ns = (fp, gup, broken, toggled, expanded)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, acts + [2 if dx<0 else 3]))

        # Helper to encode click action with camera-relative pixel coords
        def click_action(grid_pos, player_grid_y):
            cam_y = player_grid_y * 6 - 36
            px = grid_pos[0] * 6 + 3
            py = grid_pos[1] * 6 - cam_y + 3
            px = max(0, min(63, px))
            py = max(0, min(63, py))
            return 7 + py * 64 + px

        # Smart click pruning: only consider blocks in the fall column or reachable columns
        # A block is "relevant" if it's between the player and the gem in gravity direction,
        # or could affect the path after gravity flip.
        # For efficiency: only try blocks that the player could land on after breaking them
        # (i.e., blocks in x range [pos[0]-5, pos[0]+5] and in the gravity direction)
        def get_relevant_breaks(ppos, gup, broken):
            dy_g = -1 if gup else 1
            relevant = set()
            for bp in breakable:
                if bp in broken: continue
                # Only consider blocks within reasonable x range
                if abs(bp[0] - ppos[0]) > 6: continue
                # Only consider blocks in the gravity direction from player
                if gup and bp[1] >= ppos[1]: continue  # block above player when falling down
                if not gup and bp[1] <= ppos[1]: continue  # block below when falling up
                relevant.add(bp)
            return relevant

        # Click: breakable blocks (pruned)
        for bp in get_relevant_breaks(pos, gup, broken):
            nb = broken | {bp}
            fp, fg, fs = gravity_fall(pos, gup, nb, toggled, expanded)
            if fg:
                na = acts + [click_action(bp, pos[1])]
                print(f"    SOLVED (break to gem)! d={len(na)}, explored={explored}")
                return na
            if fs: continue
            ns = (fp, gup, nb, toggled, expanded)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, acts + [click_action(bp, pos[1])]))

        # Click: gravity flips (try all - usually few)
        for gp in gflips:
            ng = not gup
            fp, fg, fs = gravity_fall(pos, ng, broken, toggled, expanded)
            if fg:
                na = acts + [click_action(gp, pos[1])]
                print(f"    SOLVED (gflip to gem)! d={len(na)}, explored={explored}")
                return na
            if fs: continue
            ns = (fp, ng, broken, toggled, expanded)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, acts + [click_action(gp, pos[1])]))

        # Click: toggle blocks
        for tp in (tog_on | tog_off):
            nt = toggled - {tp} if tp in toggled else toggled | {tp}
            fp, fg, fs = gravity_fall(pos, gup, broken, nt, expanded)
            if fg:
                na = acts + [click_action(tp, pos[1])]
                print(f"    SOLVED (toggle to gem)! d={len(na)}, explored={explored}")
                return na
            if fs: continue
            ns = (fp, gup, broken, nt, expanded)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, acts + [click_action(tp, pos[1])]))

        # Click: expand blocks
        for ep in expand:
            ne = set(expanded)
            for edx, edy in [(-1,0),(1,0),(0,-1),(0,1)]:
                epos = (ep[0]+edx, ep[1]+edy)
                if not is_solid(epos, broken, toggled, expanded):
                    ne.add(epos)
            nef = frozenset(ne)
            if nef == expanded: continue
            fp, fg, fs = gravity_fall(pos, gup, broken, toggled, nef)
            if fg:
                na = acts + [click_action(ep, pos[1])]
                print(f"    SOLVED (expand to gem)! d={len(na)}, explored={explored}")
                return na
            if fs: continue
            ns = (fp, gup, broken, toggled, nef)
            if ns not in visited:
                visited.add(ns)
                q.append((ns, acts + [click_action(ep, pos[1])]))

    print(f"    BFS done: {explored} states, d={md}, no solution")
    return None


def bp35_api_dfs(game_id="bp35", max_depth=50, max_nodes=200000):
    """
    Solve BP35 using the actual game API with UNDO for backtracking.
    This is the most reliable approach since the game is the ground truth.

    BP35 available_actions: [ACTION3(LEFT), ACTION4(RIGHT), ACTION6(CLICK), ACTION7(UNDO)]
    Each action takes 2 steps (render + complete).

    Strategy: DFS with undo. At each state:
    1. Try LEFT, RIGHT
    2. Try clicking on visible blocks (breakable, gravity flip, toggle, expand)
    3. Use UNDO to backtrack
    """
    arcade = arc_agi.Arcade()
    games = arcade.get_environments()
    info = next(g for g in games if game_id in g.game_id.lower())
    env = arcade.make(info.game_id)
    obs = env.reset()

    # We need to call step twice per action (BP35 uses two-phase step)
    def do_action(ga, data=None):
        if data:
            o1 = env.step(ga, data=data)
        else:
            o1 = env.step(ga)
        o2 = env.step(ga)  # Second step to complete
        return o2

    # Get initial frame to identify clickable positions
    def get_frame():
        return np.array(obs.frame[-1]) if obs.frame else None

    max_level = 0
    all_actions = []

    print(f"Starting API-based DFS for {game_id}...")

    # For each level, do DFS with undo
    level = 0
    while level < 9:
        print(f"\n  Level {level+1}/9")

        # Get current frame and try to find a path
        # Simple DFS: try actions in order, undo on dead end
        path = []
        best_path = [None]
        visited_frames = set()

        frame = get_frame()
        if frame is not None:
            fh = hashlib.md5(frame.tobytes()).hexdigest()
            visited_frames.add(fh)

        def dfs_api(depth, current_obs):
            if best_path[0]: return
            if depth >= max_depth: return
            if len(visited_frames) > max_nodes: return

            if current_obs.state == GameState.GAME_OVER:
                return
            if current_obs.levels_completed > level:
                best_path[0] = list(path)
                return

            # Try LEFT (ACTION3 = action index 2)
            for action_data in [(GameAction.ACTION3, None, 2),
                                (GameAction.ACTION4, None, 3)]:
                ga, data, aidx = action_data
                try:
                    obs_new = env.step(ga)
                    obs_new = env.step(ga)  # Complete
                except:
                    continue

                if obs_new is None: continue
                if obs_new.levels_completed > level:
                    path.append(aidx)
                    best_path[0] = list(path)
                    return
                if obs_new.state == GameState.GAME_OVER:
                    # Undo
                    env.step(GameAction.ACTION7)
                    env.step(GameAction.ACTION7)
                    continue

                frame_new = np.array(obs_new.frame[-1]) if obs_new.frame else None
                if frame_new is not None:
                    fh = hashlib.md5(frame_new.tobytes()).hexdigest()
                    if fh not in visited_frames:
                        visited_frames.add(fh)
                        path.append(aidx)
                        dfs_api(depth + 1, obs_new)
                        if best_path[0]: return
                        path.pop()

                # Undo
                env.step(GameAction.ACTION7)
                env.step(GameAction.ACTION7)

            # Try clicks: scan the frame for clickable positions
            # In BP35, clickable tiles have specific colors
            # For now, just try a grid of click positions
            frame = get_frame()
            if frame is not None:
                # Try clicking at various positions
                for click_y in range(5, 60, 6):
                    for click_x in range(5, 60, 6):
                        try:
                            obs_new = env.step(GameAction.ACTION6, data={"x": click_x, "y": click_y})
                            obs_new = env.step(GameAction.ACTION6, data={"x": click_x, "y": click_y})
                        except:
                            continue

                        if obs_new is None: continue
                        if obs_new.levels_completed > level:
                            path.append(7 + click_y * 64 + click_x)
                            best_path[0] = list(path)
                            return
                        if obs_new.state == GameState.GAME_OVER:
                            env.step(GameAction.ACTION7)
                            env.step(GameAction.ACTION7)
                            continue

                        frame_new = np.array(obs_new.frame[-1]) if obs_new.frame else None
                        if frame_new is not None:
                            fh = hashlib.md5(frame_new.tobytes()).hexdigest()
                            if fh not in visited_frames:
                                visited_frames.add(fh)
                                path.append(7 + click_y * 64 + click_x)
                                dfs_api(depth + 1, obs_new)
                                if best_path[0]: return
                                path.pop()

                        env.step(GameAction.ACTION7)
                        env.step(GameAction.ACTION7)

        # This API-based DFS is too slow for click-heavy levels.
        # For now, store what we have and move on.
        print(f"  API DFS not practical for BP35 - using abstract model instead")
        break

    return None


def solve_bp35():
    # Grid data from bp35.py source (stored reversed with [::-1] in source)
    grids = {
        1: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "oo +     oo","oo       oo","oooooxxxooo","oo       oo","oo       oo","oo  xxx  oo","oo       oo",
             "oo       oo","ooxxxoooooo","ooxxx    oo","ooxxx    oo","ooooo    oo","oooooxxxxoo","oo       oo",
             "oo       oo","ooooooo ooo","oo n     oo","oo       oo","oo       oo","oo       oo","oo       oo",
             "oo       oo","oo       oo","mmmmmmmmmmm","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww",
             "wwwwwwwwwww"],
            {"x":["qclfkhjnaac"],"o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"m":["aknlbboysnc"],"w":["jcyhkseuorf"],"+":["fjlzdjxhant"]}),
        2: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "oo   +vvvoo","oo       oo","ooxxxxxxxoo","oo     xxoo","ooooo    oo","oovvv    oo","oo       oo",
             "oov vvvvvoo","oo       oo","ooxxxoooooo","oo      voo","oo       oo","oooooxooooo","oooooxooooo",
             "oo    vvvoo","oo       oo","oo       oo","ooxoooxxxoo","ooxxxx   oo","ooxxxx   oo","oooooo   oo",
             "oovvvo   oo","oo   o   oo","oo   o   oo","ooxxxxxxxoo","ooxxxxxxxoo","oo n     oo","oo       oo",
             "oo       oo","oo       oo","oo       oo","oo       oo","oo       oo","mmmmmmmmmmm","wwwwwwwwwww",
             "wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo"],
            {"x":["qclfkhjnaac"],"o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"m":["aknlbboysnc"],"w":["jcyhkseuorf"],"+":["fjlzdjxhant"]}),
        3: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "oovvvvvvvoo","oo       oo","oo1111111oo","oo   1 + oo","oo   1   oo","oo  ooooooo","oo  vvvoooo",
             "oo     oooo","oo11222oooo","oo       oo","oo       oo","ooooooo  oo","ooooovv  oo","ooooo22  oo",
             "ooo  11  oo","oo       oo","oo oooo  oo","oo vvvv  oo","oo       oo","oo 222ooooo","oo       oo",
             "oo       oo","oo       oo","ooooooxxxoo","oo n 1   oo","oo   1   oo","oo       oo","oo       oo",
             "oo       oo","oo       oo","oo       oo","mmmmmmmmmmm","wwwwwwwwwww","wwwwwwwwwww","wwwwwwwwwww",
             "wwwwwwwwwww","wwwwwwwwwww"],
            {"x":["qclfkhjnaac"],"1":["yuuqpmlxorv"],"2":["oonshderxef"],"o":["xcjjwqfzjfe"],"n":["player_right"],
             "v":["ubhhgljbnpu"],"m":["aknlbboysnc"],"w":["jcyhkseuorf"],"+":["fjlzdjxhant"]}),
        4: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo","ooooooooooo",
             "oo       oo","oo       oo","oooooooo oo","ooooooo  oo","oooooo   oo","oovvn    oo","oo       oo",
             "oo       oo","ooxxooooooo","oo       oo","oo   xxxxoo","oo       oo","oo    o  oo","ooooooo  oo",
             "ooogogoxxoo","oooooooxxoo","oovv vv  oo","oo       oo","oo  +    oo","oo       oo","oo       oo",
             "ooooooooooo","oooogoooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"g":["lrpkmzabbfa"],"o":["xcjjwqfzjfe"],"n":["player_right"],
             "v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
        5: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo","ooooooooooo",
             "oooo + vvvo","oooo   xxxo","oooo   xxxo","ooooooooo o","ooooooooo o","oo n    g o","oo      ooo",
             "oouuuu  ooo","oooooo  ooo","ooooooxxooo","oo      ooo","oo ooo  vvo","oo ooo    o","oo vvoxx  o",
             "oo 22oooxxo","oo        o","oo        o","ooo     ooo","ooo  oooooo","oooxxoooooo","ooo  oooooo",
             "ooouuoooooo","oooooooogoo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"g":["lrpkmzabbfa"],"o":["xcjjwqfzjfe"],"n":["player_right"],
             "v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
        6: (["ooooooooooo","oooooooogoo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooo  oo","oooo     oo","oooo oooooo","oooo vvvvoo","oo       oo","oo       oo","oo2222122oo",
             "oo       oo","oouuuu   oo","oooooo oooo","oooooo oooo","oovvv    oo","oo     o oo","oo222ooo oo",
             "oo       oo","oooooogoooo","oo n     oo","oo       oo","oo    22 oo","oouuuuuu oo","oooooooo oo",
             "oooooooo oo","oooooooo oo","o   o    oo","o + g    oo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"g":["lrpkmzabbfa"],"1":["yuuqpmlxorv"],"2":["oonshderxef"],
             "o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
        7: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooooo2ovo","goooooo2o o","goooooo2  o","goooooo2  o","go 2v v2o o","go 22 12o o","go 2 1 2o o",
             "go 2 u uo o","go oooooo o","go o222 o o","go o222 o o","go  222   o","go  222   o","go  222   o",
             "go u222u  o","gooooooo  o","go n  2o  o","go        o","go    2o  o","gooooooo  o","go        o",
             "go      u o","go + oo o o","go   oo   o","go   oooooo","ooooooooooo","ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"g":["lrpkmzabbfa"],"1":["yuuqpmlxorv"],"2":["oonshderxef"],
             "o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
        8: (["ooooooooooo","ooooooooooo","ooooooooooo",
             "ooooogooooo","ooooooooooo","ooooooooooo",
             "oooovvvoooo","oooo   oooo","ooov   vooo","oov     voo","ov       vo","o         o","o         o",
             "o         o","o         o","o         o","o         o","o         o","o  y   o1oo","o      o +o",
             "o         o","o      oooo","o111111111o","o         o","o      1  o","oooooooo  o","ovvvvvvv  o",
             "o         o","o y       o","o         o","ooooo   ooo","o  n      o","o         o","o         o",
             "ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"y":["etlsaqqtjvn"],"g":["lrpkmzabbfa"],"1":["yuuqpmlxorv"],"2":["oonshderxef"],
             "o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
        9: (["ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo",
             "ogggggggggo","ooooooooooo","oovvvvvvvvo"," o        o"," x     y  o"," o        o"," o        o",
             " o        o"," o        o"," o  x     o"," o        o","go11111111o"," o        o"," ouuuuuuu o",
             " oooooooo o","govvvvvvo o"," o      o o"," o      o o","go      o o"," o   o  o o"," o   o    o",
             " o   o    o","go   ouuuuo"," oxxxoooooo"," o   vvvvvo"," o        o"," o        o"," o        o",
             " o    y   o"," o        o","go        o"," oooo   ooo","go n      o"," o        o"," o        o",
             " oooooooooo","    ooooooo","  + ooooooo","ooooooooooo","ooooooooooo","ooooooooooo","ooooooooooo"],
            {"x":["qclfkhjnaac"],"y":["etlsaqqtjvn"],"g":["lrpkmzabbfa"],"1":["yuuqpmlxorv"],"2":["oonshderxef"],
             "o":["xcjjwqfzjfe"],"n":["player_right"],"v":["ubhhgljbnpu"],"u":["hzusueifitk"],"+":["fjlzdjxhant"]}),
    }

    per_level = {}; all_acts = []
    for li in range(1, 10):
        print(f"\n{'='*60}\nBP35 Level {li}/9\n{'='*60}")
        grid_lines, legend = grids[li]
        t0 = time.time()
        sol = bp35_bfs_level(li, grid_lines, legend)
        el = time.time() - t0
        if sol is None:
            per_level[f"L{li}"] = {"status":"UNSOLVED","time":round(el,2)}
            print(f"  FAILED ({el:.1f}s)")
            continue
        per_level[f"L{li}"] = {"status":"SOLVED","actions":sol,"length":len(sol),"time":round(el,2)}
        all_acts.extend(sol)
        print(f"  Solved: {len(sol)} actions in {el:.1f}s")

    max_lv = verify_solution("bp35", all_acts) if all_acts else 0
    return {"game":"bp35","version":"0a0ad940","method":"abstract_bfs_gravity",
            "max_level":max_lv,"total_actions":len(all_acts),
            "levels":per_level,"all_actions":all_acts}


# =============================================================================
# LF52 SOLVER - Peg solitaire with DFS
# =============================================================================

def lf52_solve_level(level_num, grid_lines, legend, max_depth=100):
    """DFS backtracking for LF52 peg solitaire."""
    board = set(); pieces = set(); red_pieces = set(); blue_pieces = set()
    obstacles = set(); moveable = set()

    for y, row in enumerate(grid_lines):
        for x, ch in enumerate(row):
            if ch == ' ' or ch == '': continue
            tt = legend.get(ch, [])
            if not tt: continue
            if "hupkpseyuim" in tt: board.add((x,y))
            if "hupkpseyuim2" in tt: board.add((x,y)); moveable.add((x,y))
            if "fozwvlovdui" in tt: pieces.add((x,y))
            if "fozwvlovdui_red" in tt: red_pieces.add((x,y))
            if "fozwvlovdui_blue" in tt: blue_pieces.add((x,y))
            if "dgxfozncuiz" in tt: obstacles.add((x,y))

    all_p = pieces | red_pieces | blue_pieces
    n = len(all_p)
    print(f"  Board:{len(board)} Pieces:{len(pieces)} Red:{len(red_pieces)} Blue:{len(blue_pieces)} Obs:{len(obstacles)} Move:{len(moveable)}")

    if n <= 1: return []
    if moveable:
        print(f"  WARNING: Level has moveable tiles - DFS only handles jumps")

    DIRS = [(0,-1),(1,0),(0,1),(-1,0)]

    # DFS
    best = [None]
    def dfs(ps, moves, d):
        if best[0]: return
        if len(ps) == 1:
            best[0] = list(moves)
            return
        if d >= max_depth: return
        for p in list(ps):
            for dx,dy in DIRS:
                mid = (p[0]+dx, p[1]+dy)
                land = (p[0]+2*dx, p[1]+2*dy)
                if mid in ps and land in board and land not in ps and land not in obstacles:
                    np_ = ps - {p, mid} | {land}
                    moves.append((p, mid, land))
                    dfs(np_, moves, d+1)
                    if best[0]: return
                    moves.pop()

    t0 = time.time()
    dfs(frozenset(all_p), [], 0)
    el = time.time() - t0

    if not best[0]:
        print(f"  No solution ({el:.1f}s)")
        return None

    print(f"  Found: {len(best[0])} jumps in {el:.1f}s")

    # Convert to click actions
    off = {1:(10,5), 2:(6,8), 3:(5,5), 4:(5,5), 5:(5,5),
           6:(5,5), 7:(5,5), 8:(5,5), 9:(5,5), 10:(5,3)}
    ox, oy = off.get(level_num, (5,5))

    acts = []
    for piece, mid, land in best[0]:
        # Click piece
        px = piece[0]*6 + ox + 3; py = piece[1]*6 + oy + 3
        acts.append(7 + py*64 + px)
        # Click direction arrow (appears at 2x distance from piece)
        dx = land[0]-piece[0]; dy = land[1]-piece[1]
        adx = (dx//abs(dx)) if dx else 0
        ady = (dy//abs(dy)) if dy else 0
        ax = piece[0]*6 + ox + 3 + adx*12
        ay = piece[1]*6 + oy + 3 + ady*12
        acts.append(7 + ay*64 + ax)

    return acts


def solve_lf52():
    grids = {
        1: (["",".......",".xx.x..",".....x.","    ...","    .x.","    ...","    ..."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"]}),
        2: (["....... ",".xx.x.x->","....... |","        |"," <--,---3"," |      "," |    .."," L----x."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        3: ([".. ..     ...",".x .x-,--x...",".x.x      .xx.","..x.      ..","          .xx.",
             "      <-> ..","      | | x..",".x.x-,3 L-.x","...       .."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        4: (["","",".......",".xp.p.x-,---x...x.",".......     .p..p.","            ......",
             "            .p..p.","        <---,..p..","        |","    .x. | ","   .p p.p..",
             "   .----,-.",   "   .p p....","    .x."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "p":["dgxfozncuiz","hupkpseyuim"],"-":["kraubslpehi"],"|":["kraubslpehi-up"],
             "L":["kraubslpehi-L"],"3":["kraubslpehi-3"],"<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        5: ([" -,-T-P-      ..p..x","    |        <---T-P>"," ...|  <--> .|...|. |",
             " xp.|.x|  Lx.|.p.|..|"," ...|  |    .|..p|x.|","    |  |     L---t--3",
             "  <-t--3       .","  |            p","  |           ..x"],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "p":["dgxfozncuiz","hupkpseyuim"],"P":["dgxfozncuiz","hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"],"T":["kraubslpehi-T"],"t":["kraubslpehi-t"]}),
        6: ([""," ....         ....   "," .r..         .x.........>. ",
             " .x..         p..p     p |x"," ....         |  |    .?.|.",
             " ......       |  |    x| |"," ......,,-----t--3  ...L-3",
             " x.....             x","   x                ."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "p":["dgxfozncuiz","hupkpseyuim"],"r":["fozwvlovdui_red","hupkpseyuim"],
             "?":["hupkpseyuim2","kraubslpehi-up"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"],"T":["kraubslpehi-T"],"t":["kraubslpehi-t"]}),
        7: (["                 <--->","x     r    <-->  |   |","p     p    |  ; <tTPTt>",
             "L--T--3  <-t-p. | D D |","   |     |    .   | | ;",
             " <-t-,> <t->  .p.p. | ."," |    | |  |  .p.p.   x",
             " p    p L->p           "," .p.p.....3."],
            {"x":["fozwvlovdui","hupkpseyuim"],".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             ";":["hupkpseyuim2","kraubslpehi-up"],"D":["dgxfozncuiz","hupkpseyuim2","kraubslpehi-up"],
             "p":["dgxfozncuiz","hupkpseyuim"],"r":["fozwvlovdui_red","hupkpseyuim"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"],"T":["kraubslpehi-T"],"t":["kraubslpehi-t"]}),
        8: (["       ","",
             " ........"," xp...p.."," ......p.","<-p...p..","|...b....","|...b...x",
             "|       |","L-,>   <3","   |   ;"," ......bb"," L--P,P-3"],
            {"x":["fozwvlovdui","hupkpseyuim"],"b":["fozwvlovdui_blue","hupkpseyuim"],
             ".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             ";":["hupkpseyuim2","kraubslpehi-up"],
             "p":["dgxfozncuiz","hupkpseyuim"],"P":["dgxfozncuiz","hupkpseyuim2","kraubslpehi"],
             "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
             "<":["kraubslpehi-<"],">":["kraubslpehi->"]}),
        9: (["       ","           x..p.p......","  ..b..    .........bb.",
             "  ...b.    .p.....p.p..","  .....              ..","  ....,--------------..",
             "  xb..   ","  .b..x  "],
            {"x":["fozwvlovdui","hupkpseyuim"],"b":["fozwvlovdui_blue","hupkpseyuim"],
             ".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
             "p":["dgxfozncuiz","hupkpseyuim"],"-":["kraubslpehi"]}),
        10: (["   .x. ","<-T-T-T->","| ; ; ; |","| L-t-3 |","|       |",
              "| ...bb |","L--b... |","  ..b.. |","  b.... |","  ....x 7",
              "        7","        7","        7","        7"],
             {"x":["fozwvlovdui","hupkpseyuim"],"b":["fozwvlovdui_blue","hupkpseyuim"],
              ".":["hupkpseyuim"],",":["hupkpseyuim2","kraubslpehi"],
              ";":["hupkpseyuim2","kraubslpehi-up"],
              "7":["fozwvlovdui_blue","hupkpseyuim2","kraubslpehi-up"],
              "p":["dgxfozncuiz","hupkpseyuim"],
              "-":["kraubslpehi"],"|":["kraubslpehi-up"],"L":["kraubslpehi-L"],"3":["kraubslpehi-3"],
              "<":["kraubslpehi-<"],">":["kraubslpehi->"],"T":["kraubslpehi-T"],"t":["kraubslpehi-t"]}),
    }

    per_level = {}; all_acts = []
    for li in range(1, 11):
        print(f"\n{'='*60}\nLF52 Level {li}/10\n{'='*60}")
        grid_lines, legend = grids[li]
        t0 = time.time()
        sol = lf52_solve_level(li, grid_lines, legend)
        el = time.time() - t0
        if sol is None:
            per_level[f"L{li}"] = {"status":"UNSOLVED","time":round(el,2)}
            print(f"  FAILED ({el:.1f}s)")
            continue
        per_level[f"L{li}"] = {"status":"SOLVED","actions":sol,"length":len(sol),"time":round(el,2)}
        all_acts.extend(sol)
        print(f"  Solved: {len(sol)} actions in {el:.1f}s")

    max_lv = verify_solution("lf52", all_acts) if all_acts else 0
    return {"game":"lf52","version":"271a04aa","method":"abstract_dfs_peg_solitaire",
            "max_level":max_lv,"total_actions":len(all_acts),
            "levels":per_level,"all_actions":all_acts}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    game = sys.argv[1] if len(sys.argv) > 1 else "all"

    if game in ["lf52", "all"]:
        result = solve_lf52()
        if result: save_result(result, "lf52")

    if game in ["wa30", "all"]:
        result = solve_wa30()
        if result: save_result(result, "wa30")

    if game in ["bp35", "all"]:
        result = solve_bp35()
        if result: save_result(result, "bp35")
