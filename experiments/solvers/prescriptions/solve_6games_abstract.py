"""
Abstract model solvers for 6 ARC-AGI-3 games.
Uses game engine for state extraction + abstract BFS for speed.
Games: SK48, AR25, TN36, LP85, R11L, S5I5
"""
import sys
import os
import json
import time
import hashlib
import copy
import numpy as np
from collections import deque
from itertools import product

os.chdir('B:/M/the-search')

# Add game paths
GAME_PATHS = {
    'sk48': 'B:/M/the-search/environment_files/sk48/41055498',
    'ar25': 'B:/M/the-search/environment_files/ar25/e3c63847',
    'tn36': 'B:/M/the-search/environment_files/tn36/ab4f63cc',
    'lp85': 'B:/M/the-search/environment_files/lp85/305b61c3',
    'r11l': 'B:/M/the-search/environment_files/r11l/aa269680',
    's5i5': 'B:/M/the-search/environment_files/s5i5/a48e4b1d',
}

for p in GAME_PATHS.values():
    if p not in sys.path:
        sys.path.insert(0, p)

import logging
logging.disable(logging.INFO)

from arcengine import GameAction, ActionInput, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def encode_click(x, y):
    return 7 + y * 64 + x


def decode_action(a):
    if a < 7:
        return f"KB{a}"
    ci = a - 7
    return f"CL({ci % 64},{ci // 64})"


def game_step(game, action):
    """Execute an action on the game. Returns (frame, levels_completed, state)."""
    if action < 7:
        GA_MAP = {
            0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
            3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
            6: GameAction.ACTION7,
        }
        ga = GA_MAP[action]
        if ga == GameAction.ACTION6:
            ai = ActionInput(id=ga, data={'x': 0, 'y': 0})
        else:
            ai = ActionInput(id=ga, data={})
    else:
        ci = action - 7
        ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    try:
        r = game.perform_action(ai, raw=True)
    except Exception:
        return None, 0, None
    if r is None:
        return None, 0, None
    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3:
        f = f[-1]
    return f, r.levels_completed, r.state


def frame_hash(arr):
    if arr is None:
        return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def verify_chain(game_cls, full_seq):
    """Verify full chain from L1."""
    g = game_cls()
    g.full_reset()
    max_levels = 0
    for a in full_seq:
        _, levels, state = game_step(g, a)
        if levels > max_levels:
            max_levels = levels
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    return max_levels


def save_result(game_key, result):
    out_path = os.path.join(RESULTS_DIR, f'{game_key}_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {out_path}')


# =============================================================
# LP85 — Cyclic permutation puzzle (abstract model)
# =============================================================
def solve_lp85():
    """
    LP85: A tile permutation puzzle.
    - Tiles sit on positions defined by cycle maps (izutyjcpih)
    - Each button rotates tiles along a numbered cycle
    - R = shift right (+1), L = shift left (-1)
    - Win: all colored tiles land on goal positions
    """
    from lp85 import Lp85
    print(f'\n{"="*70}')
    print(f'SOLVING LP85')
    print(f'{"="*70}')

    game_cls = Lp85

    # First: discover per-level buttons and build abstract model
    all_actions = []
    per_level = {}
    n_levels = 8

    for level_idx in range(n_levels):
        print(f'\n--- Level {level_idx + 1}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        # Get level state
        level_name = g.ucybisahh
        print(f'  Level name: {level_name}')

        # Find buttons
        buttons = []
        for s in g.afhycvvjg:
            if s.tags and 'button' in s.tags[0]:
                parts = s.tags[0].split('_')
                if len(parts) == 3:
                    map_name = parts[1]
                    direction = parts[2]
                    buttons.append({
                        'sprite': s,
                        'x': s.x, 'y': s.y,
                        'map': map_name,
                        'dir': direction,
                        'action': encode_click(s.x + 1, s.y + 1),
                    })
        print(f'  Found {len(buttons)} buttons')

        # Find colored tiles and goals
        bghvgbtwcb = g.current_level.get_sprites_by_tag("bghvgbtwcb")
        fdgmtkfrxl = g.current_level.get_sprites_by_tag("fdgmtkfrxl")
        goals = g.current_level.get_sprites_by_tag("goal")
        goals_o = g.current_level.get_sprites_by_tag("goal-o")

        print(f'  Tiles: {len(bghvgbtwcb)} bghvgbtwcb + {len(fdgmtkfrxl)} fdgmtkfrxl')
        print(f'  Goals: {len(goals)} goal + {len(goals_o)} goal-o')

        # Build abstract state: positions of all moveable tiles
        # State = tuple of (x, y) for each tile
        crxpafuiwp = 3  # scale factor

        def get_tile_positions(game):
            """Extract tile positions as abstract state."""
            positions = []
            for t in game.current_level.get_sprites_by_tag("bghvgbtwcb"):
                positions.append((t.x, t.y))
            for t in game.current_level.get_sprites_by_tag("fdgmtkfrxl"):
                positions.append((t.x, t.y))
            return tuple(sorted(positions))

        def check_win(game):
            return game.khartslnwa()

        # Try each button to see which ones change state
        unique_buttons = []
        base_state = get_tile_positions(g)

        for btn in buttons:
            g2 = game_cls()
            g2.full_reset()
            if level_idx > 0:
                g2.set_level(level_idx)

            _, levels, state = game_step(g2, btn['action'])
            if levels > 0:
                print(f'  INSTANT SOLVE with button {btn["map"]}_{btn["dir"]}!')
                level_actions = [btn['action']]
                all_actions.extend(level_actions)
                per_level[f'L{level_idx + 1}'] = {
                    'status': 'SOLVED',
                    'actions': level_actions,
                    'length': len(level_actions),
                }
                break
            new_state = get_tile_positions(g2)
            if new_state != base_state:
                unique_buttons.append(btn)
        else:
            print(f'  {len(unique_buttons)} effective buttons')

            if len(unique_buttons) == 0:
                per_level[f'L{level_idx + 1}'] = {'status': 'NO_ACTIONS'}
                print(f'  NO effective buttons found!')
                break

            # BFS on abstract state using game engine
            # Replay from set_level each time (state is positions)
            btn_actions = [b['action'] for b in unique_buttons]
            btn_names = [f"{b['map']}_{b['dir']}" for b in unique_buttons]

            sol = bfs_engine(game_cls, level_idx, btn_actions, max_depth=60,
                             max_states=5000000, time_limit=240,
                             state_fn=get_tile_positions, win_fn=check_win)

            elapsed = time.time() - t0
            if sol is not None:
                # Verify in chain
                test_seq = all_actions + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= level_idx + 1:
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED',
                        'actions': sol,
                        'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    all_actions.extend(sol)
                    print(f'  SOLVED L{level_idx + 1}: {len(sol)} actions')
                else:
                    per_level[f'L{level_idx + 1}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    print(f'  CHAIN FAIL (got {chain_max})')
                    break
            else:
                per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 'lp85',
        'version': '305b61c3',
        'total_levels': n_levels,
        'method': 'abstract_bfs',
        'levels': per_level,
        'full_sequence': all_actions,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(all_actions),
    }
    save_result('lp85', result)
    return result


def bfs_engine(game_cls, level_idx, actions, max_depth=50, max_states=2000000,
               time_limit=300, state_fn=None, win_fn=None):
    """
    BFS using game engine with custom state hash.
    Replays from set_level each time.
    """
    t0 = time.time()

    # Get initial state
    g0 = game_cls()
    g0.full_reset()
    if level_idx > 0:
        g0.set_level(level_idx)
    init_state = state_fn(g0) if state_fn else frame_hash(np.zeros(1))

    print(f'    BFS: {len(actions)} actions, max_depth={max_depth}, limit={max_states}')

    queue = deque([()])
    visited = {init_state}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f'    TIMEOUT ({time_limit}s, e={explored}, d={depth})')
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f'      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)')

        if len(seq) >= max_depth:
            continue

        for action in actions:
            new_seq = list(seq) + [action]
            explored += 1

            g = game_cls()
            g.full_reset()
            if level_idx > 0:
                g.set_level(level_idx)

            won = False
            game_over = False
            for a in new_seq:
                _, levels, state = game_step(g, a)
                if levels > 0:
                    won = True
                    break
                if state == GameState.GAME_OVER:
                    game_over = True
                    break

            if won:
                el = time.time() - t0
                print(f'    SOLVED! {len(new_seq)} steps, {explored} states, {el:.1f}s')
                return list(new_seq)

            if game_over:
                continue

            if win_fn and win_fn(g):
                el = time.time() - t0
                print(f'    SOLVED (win_fn)! {len(new_seq)} steps, {explored} states, {el:.1f}s')
                return list(new_seq)

            sh = state_fn(g) if state_fn else None
            if sh is not None and sh not in visited:
                visited.add(sh)
                queue.append(tuple(new_seq))

            if explored >= max_states:
                print(f'    LIMIT ({max_states}, d={depth})')
                return None

    print(f'    EXHAUSTED ({explored})')
    return None


# =============================================================
# SK48 — Rail puzzle (abstract model)
# =============================================================
def solve_sk48():
    """
    SK48: Rail-based puzzle.
    - Pushers extend/contract segments along rails
    - Colored items sit on segments
    - Pairs must match colors at matching positions
    - ACTION1-4 = move (up/down/left/right)
    - ACTION6 = click to select pusher pair
    - ACTION7 = undo
    """
    from sk48 import Sk48
    print(f'\n{"="*70}')
    print(f'SOLVING SK48')
    print(f'{"="*70}')

    game_cls = Sk48
    all_actions = []
    per_level = {}
    n_levels = 8

    for level_idx in range(n_levels):
        print(f'\n--- Level {level_idx + 1}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        # State function for SK48: positions of colored items + segment counts per pusher
        def sk48_state(game):
            parts = []
            for s in game.vbelzuaian:
                parts.append(('item', s.x, s.y, int(s.pixels[1, 1])))
            for pusher, segs in game.mwfajkguqx.items():
                parts.append(('seg', pusher.x, pusher.y, len(segs)))
            parts.append(('sel', game.vzvypfsnt.x, game.vzvypfsnt.y))
            return tuple(sorted(parts))

        # Find available actions
        # KB actions: 0=up, 1=down, 2=left, 3=right, 6=undo
        # Click actions: select different pushers
        kb_actions = [0, 1, 2, 3]  # movement actions

        # Find clickable pushers
        click_actions = []
        pushers = g.current_level.get_sprites_by_tag("epdquznwmq")
        for p in pushers:
            if "sys_click" in p.tags:
                cx = p.x + 3
                cy = p.y + 3
                click_actions.append(encode_click(cx, cy))

        all_level_actions = kb_actions + click_actions
        print(f'  {len(kb_actions)} KB + {len(click_actions)} click = {len(all_level_actions)} actions')

        # Discover unique actions
        base_state = sk48_state(g)
        unique_actions = []
        for a in all_level_actions:
            g2 = game_cls()
            g2.full_reset()
            if level_idx > 0:
                g2.set_level(level_idx)
            _, levels, state = game_step(g2, a)
            if levels > 0:
                print(f'  INSTANT SOLVE!')
                sol = [a]
                all_actions.extend(sol)
                per_level[f'L{level_idx + 1}'] = {'status': 'SOLVED', 'actions': sol, 'length': 1}
                break
            new_state = sk48_state(g2)
            if new_state != base_state and new_state not in [sk48_state(g2) for _ in []]:
                unique_actions.append(a)
        else:
            # Deduplicate by effect
            seen_states = {}
            deduped = []
            for a in unique_actions:
                g2 = game_cls()
                g2.full_reset()
                if level_idx > 0:
                    g2.set_level(level_idx)
                game_step(g2, a)
                st = sk48_state(g2)
                if st not in seen_states:
                    seen_states[st] = a
                    deduped.append(a)

            unique_actions = deduped
            print(f'  {len(unique_actions)} unique actions: {[decode_action(a) for a in unique_actions[:20]]}')

            sol = bfs_engine(game_cls, level_idx, unique_actions,
                             max_depth=40, max_states=3000000, time_limit=240,
                             state_fn=sk48_state)

            elapsed = time.time() - t0
            if sol is not None:
                test_seq = all_actions + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= level_idx + 1:
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    all_actions.extend(sol)
                    print(f'  SOLVED L{level_idx + 1}: {len(sol)} actions')
                else:
                    per_level[f'L{level_idx + 1}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    print(f'  CHAIN FAIL')
                    break
            else:
                per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 'sk48',
        'version': '41055498',
        'total_levels': n_levels,
        'method': 'abstract_bfs',
        'levels': per_level,
        'full_sequence': all_actions,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(all_actions),
    }
    save_result('sk48', result)
    return result


# =============================================================
# TN36 — Programming puzzle (engine BFS)
# =============================================================
def solve_tn36():
    """
    TN36: Programming puzzle with click-only interface.
    Complex internal state with programs, rotations, scales.
    Timer bar moving left = lose condition.
    """
    from tn36 import Tn36
    print(f'\n{"="*70}')
    print(f'SOLVING TN36')
    print(f'{"="*70}')

    game_cls = Tn36
    all_actions = []
    per_level = {}
    n_levels = 7

    for level_idx in range(n_levels):
        print(f'\n--- Level {level_idx + 1}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        # State: frame hash (complex internal state)
        def tn36_state(game):
            # Use sprite positions as state
            parts = []
            for s in game.current_level.get_sprites():
                if s.visible:
                    parts.append((s.name[:8], s.x, s.y))
            return hash(tuple(sorted(parts)))

        # Find click actions that change state
        base_state = tn36_state(g)
        unique_actions = []
        seen = set()

        for y in range(0, 64, 1):
            for x in range(0, 64, 1):
                a = encode_click(x, y)
                g2 = game_cls()
                g2.full_reset()
                if level_idx > 0:
                    g2.set_level(level_idx)

                # Need to wait for animations
                f, levels, state = game_step(g2, a)
                # Step through animations
                while g2.tsflfunycx.nwjrtjcxpo:
                    f, levels, state = game_step(g2, a)

                if levels > 0:
                    print(f'  INSTANT SOLVE at ({x},{y})!')
                    sol = [a]
                    all_actions.extend(sol)
                    per_level[f'L{level_idx + 1}'] = {'status': 'SOLVED', 'actions': sol, 'length': 1}
                    break

                st = tn36_state(g2)
                if st != base_state and st not in seen:
                    seen.add(st)
                    unique_actions.append(a)
            else:
                continue
            break
        else:
            print(f'  {len(unique_actions)} unique actions')

            if not unique_actions:
                per_level[f'L{level_idx + 1}'] = {'status': 'NO_ACTIONS'}
                print(f'  NO effective actions!')
                break

            # For TN36, BFS needs to handle animations (multi-step per action)
            sol = bfs_tn36(game_cls, level_idx, unique_actions,
                           max_depth=30, max_states=2000000, time_limit=240,
                           state_fn=tn36_state)

            elapsed = time.time() - t0
            if sol is not None:
                test_seq = all_actions + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= level_idx + 1:
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    all_actions.extend(sol)
                else:
                    per_level[f'L{level_idx + 1}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    break
            else:
                per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 'tn36',
        'version': 'ab4f63cc',
        'total_levels': n_levels,
        'method': 'engine_bfs',
        'levels': per_level,
        'full_sequence': all_actions,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(all_actions),
    }
    save_result('tn36', result)
    return result


def bfs_tn36(game_cls, level_idx, actions, max_depth=30, max_states=2000000,
             time_limit=300, state_fn=None):
    """BFS for TN36 with animation handling."""
    t0 = time.time()

    g0 = game_cls()
    g0.full_reset()
    if level_idx > 0:
        g0.set_level(level_idx)
    init_state = state_fn(g0)

    print(f'    BFS: {len(actions)} actions, max_depth={max_depth}')

    queue = deque([()])
    visited = {init_state}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f'    TIMEOUT ({time_limit}s, e={explored}, d={depth})')
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f'      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)')

        if len(seq) >= max_depth:
            continue

        for action in actions:
            new_seq = list(seq) + [action]
            explored += 1

            g = game_cls()
            g.full_reset()
            if level_idx > 0:
                g.set_level(level_idx)

            won = False
            game_over = False
            for a in new_seq:
                _, levels, state = game_step(g, a)
                if levels > 0:
                    won = True
                    break
                if state == GameState.GAME_OVER:
                    game_over = True
                    break
                # Process animations
                for _ in range(100):
                    if not g.tsflfunycx.nwjrtjcxpo:
                        break
                    _, levels, state = game_step(g, a)
                    if levels > 0:
                        won = True
                        break
                    if state == GameState.GAME_OVER:
                        game_over = True
                        break
                if won or game_over:
                    break

            if won:
                el = time.time() - t0
                print(f'    SOLVED! {len(new_seq)} steps, {explored} states, {el:.1f}s')
                return list(new_seq)

            if game_over:
                continue

            sh = state_fn(g)
            if sh not in visited:
                visited.add(sh)
                queue.append(tuple(new_seq))

            if explored >= max_states:
                print(f'    LIMIT ({max_states}, d={depth})')
                return None

    print(f'    EXHAUSTED ({explored})')
    return None


# =============================================================
# AR25 — Mirror reflection puzzle (engine BFS)
# =============================================================
def solve_ar25():
    """
    AR25: Mirror/reflection puzzle.
    - Move pieces with arrows (ACTION1-4)
    - Select pieces (ACTION5 cycles, ACTION6 clicks)
    - Mirrors reflect pieces
    - Win: all target squares covered
    """
    from ar25 import Ar25
    print(f'\n{"="*70}')
    print(f'SOLVING AR25')
    print(f'{"="*70}')

    game_cls = Ar25
    all_actions = []
    per_level = {}
    n_levels = 8

    for level_idx in range(n_levels):
        print(f'\n--- Level {level_idx + 1}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        def ar25_state(game):
            parts = []
            for s in game.migkdsjrwk:
                parts.append(('piece', s.x, s.y, s.pixels.shape))
            for s in game.khupblbrxc:
                tag = s.tags[0] if s.tags else ''
                parts.append(('axis', s.x, s.y, tag))
            if game.llludejph:
                parts.append(('sel', game.llludejph.x, game.llludejph.y))
            return hash(tuple(sorted(parts)))

        # Available actions: KB 0-4 (arrows + select), click on pieces
        kb_actions = [0, 1, 2, 3, 4]  # up, down, left, right, cycle-select

        # Find clickable positions
        click_actions = []
        for s in g.xefwpvwoh:
            cx = s.x + s.pixels.shape[1] // 2
            cy = s.y + s.pixels.shape[0] // 2
            click_actions.append(encode_click(cx, cy))

        all_level_actions = kb_actions + click_actions
        print(f'  {len(kb_actions)} KB + {len(click_actions)} click = {len(all_level_actions)} actions')

        # Deduplicate
        base_state = ar25_state(g)
        unique_actions = []
        seen_states = {}
        for a in all_level_actions:
            g2 = game_cls()
            g2.full_reset()
            if level_idx > 0:
                g2.set_level(level_idx)
            _, levels, state = game_step(g2, a)
            if levels > 0:
                print(f'  INSTANT SOLVE!')
                sol = [a]
                all_actions.extend(sol)
                per_level[f'L{level_idx + 1}'] = {'status': 'SOLVED', 'actions': sol, 'length': 1}
                break
            st = ar25_state(g2)
            if st != base_state and st not in seen_states:
                seen_states[st] = a
                unique_actions.append(a)
        else:
            print(f'  {len(unique_actions)} unique actions')

            sol = bfs_engine(game_cls, level_idx, unique_actions,
                             max_depth=40, max_states=3000000, time_limit=240,
                             state_fn=ar25_state)

            elapsed = time.time() - t0
            if sol is not None:
                test_seq = all_actions + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= level_idx + 1:
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    all_actions.extend(sol)
                else:
                    per_level[f'L{level_idx + 1}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    break
            else:
                per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 'ar25',
        'version': 'e3c63847',
        'total_levels': n_levels,
        'method': 'engine_bfs',
        'levels': per_level,
        'full_sequence': all_actions,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(all_actions),
    }
    save_result('ar25', result)
    return result


# =============================================================
# R11L — Spider body puzzle (engine BFS, extend existing L1-L2)
# =============================================================
def solve_r11l():
    """
    R11L: Spider body puzzle.
    - Bodies have legs
    - Click legs to select, click positions to move
    - Body = average of leg centers
    - Win: bodies on targets
    """
    from r11l import R11l
    print(f'\n{"="*70}')
    print(f'SOLVING R11L (extending from L2)')
    print(f'{"="*70}')

    game_cls = R11l

    # Load existing
    with open(os.path.join(RESULTS_DIR, 'r11l_fullchain.json')) as f:
        existing = json.load(f)

    full_seq = list(existing.get('full_sequence', []))
    max_lev = verify_chain(game_cls, full_seq)
    print(f'  Existing chain: {max_lev} levels, {len(full_seq)} actions')

    per_level = {}
    for k, v in existing.get('levels', {}).items():
        per_level[k] = v

    n_levels = 6

    for level_idx in range(max_lev, n_levels):
        lnum = level_idx + 1
        if lnum <= max_lev:
            continue
        print(f'\n--- Level {lnum}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        def r11l_state(game):
            parts = []
            for leg in game.ftmaz:
                parts.append(('leg', leg.x, leg.y))
            for name, data in game.brdck.items():
                body = data['kignw']
                if body:
                    parts.append(('body', name, body.x, body.y))
            if game.mjdkn:
                parts.append(('sel', game.mjdkn.x, game.mjdkn.y))
            return hash(tuple(sorted(parts)))

        # Discover unique click actions
        base_state = r11l_state(g)
        unique_actions = []
        seen = set()

        for y in range(0, 64, 2):
            for x in range(0, 64, 2):
                a = encode_click(x, y)
                g2 = game_cls()
                g2.full_reset()
                if level_idx > 0:
                    g2.set_level(level_idx)

                _, levels, state = game_step(g2, a)
                # Process animation
                while g2.bmtib:
                    _, levels, state = game_step(g2, encode_click(0, 0))
                    if levels > 0:
                        break

                if levels > 0:
                    print(f'  INSTANT SOLVE!')
                    sol = [a]
                    full_seq.extend(sol)
                    per_level[f'L{lnum}'] = {'status': 'SOLVED', 'actions': sol, 'length': 1}
                    break

                st = r11l_state(g2)
                if st != base_state and st not in seen:
                    seen.add(st)
                    unique_actions.append(a)
        else:
            print(f'  {len(unique_actions)} unique actions')

            if not unique_actions:
                per_level[f'L{lnum}'] = {'status': 'NO_ACTIONS'}
                break

            sol = bfs_engine(game_cls, level_idx, unique_actions,
                             max_depth=30, max_states=2000000, time_limit=240,
                             state_fn=r11l_state)

            elapsed = time.time() - t0
            if sol is not None:
                test_seq = full_seq + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= lnum:
                    per_level[f'L{lnum}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    full_seq.extend(sol)
                else:
                    per_level[f'L{lnum}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    break
            else:
                per_level[f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 'r11l',
        'version': 'aa269680',
        'total_levels': n_levels,
        'method': 'engine_bfs',
        'levels': per_level,
        'full_sequence': full_seq,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(full_seq),
    }
    save_result('r11l', result)
    return result


# =============================================================
# S5I5 — Extend existing L1-L2 with faster BFS
# =============================================================
def solve_s5i5():
    """
    S5I5: Rotating arm puzzle.
    - Click color buttons to rotate all arms of that color
    - Click rails to extend/shorten arms
    - Win: moveable markers on goal positions
    """
    from s5i5 import S5i5
    print(f'\n{"="*70}')
    print(f'SOLVING S5I5 (extending from L2)')
    print(f'{"="*70}')

    game_cls = S5i5

    # Load existing
    with open(os.path.join(RESULTS_DIR, 's5i5_fullchain.json')) as f:
        existing = json.load(f)

    full_seq = list(existing.get('full_sequence', []))
    max_lev = verify_chain(game_cls, full_seq)
    print(f'  Existing chain: {max_lev} levels, {len(full_seq)} actions')

    per_level = {}
    for k, v in existing.get('levels', {}).items():
        per_level[k] = v

    n_levels = 8

    for level_idx in range(max_lev, n_levels):
        lnum = level_idx + 1
        if lnum <= max_lev:
            continue
        print(f'\n--- Level {lnum}/{n_levels} ---')
        t0 = time.time()

        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        def s5i5_state(game):
            parts = []
            for s in game.current_level.get_sprites_by_tag('agujdcrunq'):
                parts.append(('arm', s.x, s.y, s.width, s.height))
            for s in game.current_level.get_sprites_by_tag('zylvdxoiuq'):
                parts.append(('mark', s.x, s.y))
            return hash(tuple(sorted(parts)))

        # Discover unique click actions
        base_state = s5i5_state(g)
        unique_actions = []
        seen = set()

        for y in range(0, 64, 2):
            for x in range(0, 64, 2):
                a = encode_click(x, y)
                g2 = game_cls()
                g2.full_reset()
                if level_idx > 0:
                    g2.set_level(level_idx)
                _, levels, state = game_step(g2, a)
                if levels > 0:
                    print(f'  INSTANT SOLVE at ({x},{y})!')
                    sol = [a]
                    full_seq.extend(sol)
                    per_level[f'L{lnum}'] = {'status': 'SOLVED', 'actions': sol, 'length': 1}
                    break
                if state == GameState.GAME_OVER:
                    continue
                st = s5i5_state(g2)
                if st != base_state and st not in seen:
                    seen.add(st)
                    unique_actions.append(a)
        else:
            print(f'  {len(unique_actions)} unique actions')

            if not unique_actions:
                per_level[f'L{lnum}'] = {'status': 'NO_ACTIONS'}
                break

            sol = bfs_engine(game_cls, level_idx, unique_actions,
                             max_depth=40, max_states=3000000, time_limit=300,
                             state_fn=s5i5_state)

            elapsed = time.time() - t0
            if sol is not None:
                test_seq = full_seq + sol
                chain_max = verify_chain(game_cls, test_seq)
                print(f'  Chain: {chain_max} levels')
                if chain_max >= lnum:
                    per_level[f'L{lnum}'] = {
                        'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                        'time': round(elapsed, 2),
                    }
                    full_seq.extend(sol)
                else:
                    per_level[f'L{lnum}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                    break
            else:
                per_level[f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                print(f'  UNSOLVED ({elapsed:.1f}s)')
                break
            continue

    result = {
        'game': 's5i5',
        'version': 'a48e4b1d',
        'total_levels': n_levels,
        'method': 'engine_bfs',
        'levels': per_level,
        'full_sequence': full_seq,
        'max_level_solved': max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0),
        'total_actions': len(full_seq),
    }
    save_result('s5i5', result)
    return result


# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    games = sys.argv[1:] if len(sys.argv) > 1 else ['lp85', 'sk48', 'ar25', 'tn36', 'r11l', 's5i5']

    results = {}
    for game in games:
        try:
            if game == 'lp85':
                results[game] = solve_lp85()
            elif game == 'sk48':
                results[game] = solve_sk48()
            elif game == 'ar25':
                results[game] = solve_ar25()
            elif game == 'tn36':
                results[game] = solve_tn36()
            elif game == 'r11l':
                results[game] = solve_r11l()
            elif game == 's5i5':
                results[game] = solve_s5i5()
        except Exception as e:
            print(f'\nERROR solving {game}: {e}')
            import traceback
            traceback.print_exc()
            results[game] = {'error': str(e)}

    print(f'\n\n{"="*70}')
    print(f'SUMMARY')
    print(f'{"="*70}')
    for game, r in results.items():
        if isinstance(r, dict) and 'error' not in r:
            ml = r.get('max_level_solved', 0)
            nl = r.get('total_levels', '?')
            ta = r.get('total_actions', 0)
            print(f'  {game.upper()}: {ml}/{nl} levels, {ta} actions')
        else:
            print(f'  {game.upper()}: ERROR')
