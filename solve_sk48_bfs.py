"""
BFS solver for SK48 using game engine with state hashing.
SK48: Rail puzzle with pushers and colored items.
Actions: KB 0-3 (up/down/left/right), KB 6 (undo), click to select pushers.
"""
import sys
import os
import json
import time
import hashlib
import numpy as np
from collections import deque

sys.path.insert(0, 'B:/M/the-search/environment_files/sk48/41055498')
os.chdir('B:/M/the-search')

from sk48 import Sk48
from arcengine import GameAction, ActionInput, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def encode_click(x, y):
    return 7 + y * 64 + x


def game_step(game, action):
    if action < 7:
        GA_MAP = {0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
                  3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
                  6: GameAction.ACTION7}
        ga = GA_MAP[action]
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


def sk48_state(game):
    """Extract abstract state from SK48 game."""
    parts = []
    # Colored items positions and colors
    for s in game.vbelzuaian:
        parts.append(('item', s.x, s.y, int(s.pixels[1, 1])))
    # Segment counts per pusher
    for pusher, segs in game.mwfajkguqx.items():
        parts.append(('seg', pusher.x, pusher.y, len(segs)))
    # Currently selected pusher
    parts.append(('sel', game.vzvypfsnt.x, game.vzvypfsnt.y))
    return tuple(sorted(parts))


def make_at_level(level_idx):
    g = Sk48()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


def find_actions(level_idx):
    """Find unique actions for a level."""
    g = make_at_level(level_idx)
    base = sk48_state(g)

    # KB actions: up=0, down=1, left=2, right=3
    kb_actions = [0, 1, 2, 3]

    # Click actions: find clickable pushers
    pushers = g.current_level.get_sprites_by_tag("epdquznwmq")
    click_actions = []
    for p in pushers:
        if "sys_click" in p.tags:
            cx = p.x + 3
            cy = p.y + 3
            click_actions.append(encode_click(cx, cy))

    all_actions = kb_actions + click_actions
    print(f'  {len(kb_actions)} KB + {len(click_actions)} click')

    # Deduplicate by effect
    unique = []
    seen = set()
    for a in all_actions:
        g2 = make_at_level(level_idx)
        f, levels, state = game_step(g2, a)
        if levels > 0:
            return [a], True
        st = sk48_state(g2)
        if st != base and st not in seen:
            seen.add(st)
            unique.append(a)

    return unique, False


def solve():
    all_actions = []
    per_level = {}
    n_levels = 8

    for level_idx in range(n_levels):
        print(f'\n=== SK48 Level {level_idx + 1}/{n_levels} ===')
        t0 = time.time()

        actions, instant = find_actions(level_idx)
        if instant:
            all_actions.extend(actions)
            per_level[f'L{level_idx + 1}'] = {'status': 'SOLVED', 'actions': actions, 'length': 1}
            print(f'  INSTANT SOLVE!')
            continue

        print(f'  {len(actions)} unique actions')

        if not actions:
            per_level[f'L{level_idx + 1}'] = {'status': 'NO_ACTIONS'}
            break

        # BFS
        g0 = make_at_level(level_idx)
        init_state = sk48_state(g0)
        queue = deque([()])
        visited = {init_state}
        explored = 0
        depth = 0
        solved = False

        while queue:
            if time.time() - t0 > 240:
                print(f'  TIMEOUT (e={explored}, d={depth})')
                break
            seq = queue.popleft()
            if len(seq) > depth:
                depth = len(seq)
                el = time.time() - t0
                rate = explored / max(el, 0.1)
                print(f'    d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)')
            if len(seq) >= 40:
                continue
            for action in actions:
                new_seq = list(seq) + [action]
                explored += 1
                g = make_at_level(level_idx)
                won = False
                dead = False
                for a in new_seq:
                    f, levels, state = game_step(g, a)
                    if levels > 0:
                        won = True
                        break
                    if state == GameState.GAME_OVER:
                        dead = True
                        break
                if won:
                    el = time.time() - t0
                    print(f'  SOLVED in {len(new_seq)} actions ({explored} explored, {el:.1f}s)')
                    all_actions.extend(new_seq)
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED', 'actions': list(new_seq),
                        'length': len(new_seq), 'time': round(el, 2),
                    }
                    solved = True
                    break
                if dead:
                    continue
                st = sk48_state(g)
                if st not in visited:
                    visited.add(st)
                    queue.append(tuple(new_seq))
            if solved:
                break
            if explored >= 3000000:
                print(f'  LIMIT (e={explored})')
                break

        if not solved:
            el = time.time() - t0
            per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(el, 2)}
            print(f'  UNSOLVED')
            break

    # Verify
    print(f'\nVerifying chain ({len(all_actions)} actions)...')
    g = Sk48()
    g.full_reset()
    max_levels = 0
    for a in all_actions:
        f, levels, state = game_step(g, a)
        if levels > max_levels:
            max_levels = levels
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    print(f'Chain: {max_levels} levels')

    result = {
        'game': 'sk48', 'version': '41055498', 'total_levels': n_levels,
        'method': 'engine_bfs',
        'levels': per_level, 'full_sequence': all_actions,
        'max_level_solved': max_levels, 'total_actions': len(all_actions),
    }
    with open(f'{RESULTS_DIR}/sk48_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved sk48_fullchain.json')


if __name__ == '__main__':
    solve()
