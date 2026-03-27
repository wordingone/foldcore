"""
Fast BFS solver using deepcopy for state checkpointing.
Instead of replaying from set_level() each time (~100/s),
deepcopy the game state and advance by one action (~1000+/s).
"""
import sys
import os
import json
import time
import copy
import copyreg
import hashlib
import numpy as np
from collections import deque

os.chdir('B:/M/the-search')

import logging
logging.disable(logging.INFO)

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

from arcengine import GameAction, ActionInput, GameState

# Fix deepcopy for GameAction
def pickle_gameaction(ga):
    return (GameAction.__getitem__, (ga.name,))
copyreg.pickle(GameAction, pickle_gameaction)

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'
GA_MAP = {0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
          3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
          6: GameAction.ACTION7}


def encode_click(x, y):
    return 7 + y * 64 + x


def decode_action(a):
    if a < 7:
        return f"KB{a}"
    ci = a - 7
    return f"CL({ci % 64},{ci // 64})"


def do_action(game, action):
    """Execute one action on game, return (frame_hash, levels_completed, state)."""
    if action < 7:
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
        return None, 0, GameState.GAME_OVER
    if r is None:
        return None, 0, GameState.GAME_OVER
    h = frame_hash_from_result(r)
    return h, r.levels_completed, r.state


def frame_hash_from_result(r):
    """Get frame hash from perform_action result."""
    if r is None:
        return None
    f = np.array(r.frame, dtype=np.uint8)
    return hashlib.md5(f.tobytes()).hexdigest()


def get_frame_hash(game, noop_action=None):
    """Get frame hash by performing a noop action."""
    if noop_action is None:
        noop_action = encode_click(63, 63)  # Click on edge - should be noop
    g = copy.deepcopy(game)
    ai = ActionInput(id=GameAction.ACTION6, data={'x': 63, 'y': 63})
    r = g.perform_action(ai, raw=True)
    if r is None:
        return None
    f = np.array(r.frame, dtype=np.uint8)
    return hashlib.md5(f.tobytes()).hexdigest()


def find_unique_actions(game_cls, level_idx, kb_actions=[], click_step=2):
    """Discover unique actions by frame hashing."""
    g0 = game_cls()
    g0.full_reset()
    if level_idx > 0:
        g0.set_level(level_idx)

    # Get base frame via noop
    g_noop = game_cls()
    g_noop.full_reset()
    if level_idx > 0:
        g_noop.set_level(level_idx)
    ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
    r = g_noop.perform_action(ai, raw=True)
    base_h = frame_hash_from_result(r) if r else None

    groups = {}
    for kb in kb_actions:
        g = game_cls()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)
        ga = GA_MAP[kb]
        ai = ActionInput(id=ga, data={})
        r = g.perform_action(ai, raw=True)
        if r and r.levels_completed > 0:
            return [kb], True
        h = frame_hash_from_result(r)
        if h and h != base_h and h not in groups:
            groups[h] = kb

    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            g = game_cls()
            g.full_reset()
            if level_idx > 0:
                g.set_level(level_idx)
            ci = a - 7
            ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
            r = g.perform_action(ai, raw=True)
            if r and r.levels_completed > 0:
                return [a], True
            h = frame_hash_from_result(r)
            if h and h != base_h and h not in groups:
                groups[h] = a

    return list(groups.values()), False


def fast_bfs(game_cls, level_idx, actions, max_depth=50, max_states=5000000, time_limit=300):
    """
    BFS using deepcopy for state checkpointing.
    Instead of replaying from scratch, we store (game_copy, action_sequence) in the queue.
    """
    t0 = time.time()

    g0 = game_cls()
    g0.full_reset()
    if level_idx > 0:
        g0.set_level(level_idx)

    # Get initial frame hash
    g_init = copy.deepcopy(g0)
    h_init, _, _ = do_action(g_init, encode_click(0, 0))
    if h_init is None:
        print('    Cannot hash initial state')
        return None

    # Queue stores (game_state, action_sequence)
    queue = deque([(g0, ())])
    visited = {h_init}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f'    TIMEOUT ({time_limit}s, e={explored}, d={depth})')
            return None

        g_state, seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f'      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)')

        if len(seq) >= max_depth:
            del g_state
            continue

        for action in actions:
            g = copy.deepcopy(g_state)
            h, lev, st = do_action(g, action)
            explored += 1

            if lev > 0:
                new_seq = list(seq) + [action]
                el = time.time() - t0
                print(f'    SOLVED! {len(new_seq)} steps, {explored} states, {el:.1f}s')
                return new_seq

            if st == GameState.GAME_OVER:
                continue

            if h and h not in visited:
                visited.add(h)
                queue.append((g, tuple(list(seq) + [action])))

            if explored >= max_states:
                print(f'    LIMIT ({max_states}, d={depth})')
                return None

        del g_state

    print(f'    EXHAUSTED ({explored})')
    return None


def verify_chain(game_cls, full_seq):
    g = game_cls()
    g.full_reset()
    max_levels = 0
    for a in full_seq:
        _, lev, st = do_action(g, a)
        if lev > max_levels:
            max_levels = lev
        if st in (GameState.GAME_OVER, GameState.WIN):
            break
    return max_levels


def solve_game(game_id, game_cls, n_levels, kb_actions=[], existing_seq=None,
               click_step=2, max_depth=50, time_limit=300, max_states=5000000):
    print(f'\n{"="*70}')
    print(f'SOLVING {game_id.upper()} ({n_levels} levels)')
    print(f'{"="*70}')

    full_seq = list(existing_seq) if existing_seq else []
    per_level = {}
    start_level = 0

    if full_seq:
        max_lev = verify_chain(game_cls, full_seq)
        print(f'  Existing chain: {max_lev} levels, {len(full_seq)} actions')
        start_level = max_lev
    else:
        start_level = 0

    for level_idx in range(start_level, n_levels):
        lnum = level_idx + 1
        print(f'\n--- Level {lnum}/{n_levels} ---')
        t0 = time.time()

        actions, instant = find_unique_actions(game_cls, level_idx, kb_actions, click_step)
        if instant:
            full_seq.extend(actions)
            per_level[f'L{lnum}'] = {'status': 'SOLVED', 'actions': actions, 'length': len(actions)}
            print(f'  INSTANT SOLVE!')
            continue

        print(f'  {len(actions)} unique actions: {[decode_action(a) for a in actions[:15]]}')

        if not actions:
            per_level[f'L{lnum}'] = {'status': 'NO_ACTIONS'}
            break

        sol = fast_bfs(game_cls, level_idx, actions, max_depth=max_depth,
                       max_states=max_states, time_limit=time_limit)
        elapsed = time.time() - t0

        if sol is not None:
            test_seq = full_seq + sol
            chain_max = verify_chain(game_cls, test_seq)
            print(f'  Chain: {chain_max} levels')
            if chain_max >= lnum:
                full_seq.extend(sol)
                per_level[f'L{lnum}'] = {
                    'status': 'SOLVED', 'actions': sol, 'length': len(sol),
                    'time': round(elapsed, 2),
                }
            else:
                per_level[f'L{lnum}'] = {'status': 'CHAIN_FAIL', 'time': round(elapsed, 2)}
                break
        else:
            per_level[f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            break

    max_solved = max((int(k[1:]) for k, v in per_level.items() if v.get('status') == 'SOLVED'), default=0)
    if full_seq and not max_solved:
        max_solved = verify_chain(game_cls, full_seq)

    return {
        'game': game_id, 'total_levels': n_levels,
        'method': 'fast_bfs_deepcopy',
        'levels': per_level, 'full_sequence': full_seq,
        'max_level_solved': max_solved, 'total_actions': len(full_seq),
    }


def main():
    games = sys.argv[1:] if len(sys.argv) > 1 else ['ar25', 'sk48', 'tn36', 's5i5', 'r11l']

    for gid in games:
        try:
            if gid == 'sk48':
                from sk48 import Sk48
                result = solve_game('sk48', Sk48, 8, kb_actions=[0,1,2,3,6],
                                    click_step=2, max_depth=40, time_limit=240)
            elif gid == 'ar25':
                from ar25 import Ar25
                with open(f'{RESULTS_DIR}/ar25_fullchain.json') as f:
                    existing = json.load(f)
                result = solve_game('ar25', Ar25, 8, kb_actions=[0,1,2,3,4],
                                    existing_seq=existing.get('full_sequence', []),
                                    click_step=4, max_depth=40, time_limit=240)
            elif gid == 'tn36':
                from tn36 import Tn36
                result = solve_game('tn36', Tn36, 7, click_step=1, max_depth=40, time_limit=240)
            elif gid == 'r11l':
                from r11l import R11l
                with open(f'{RESULTS_DIR}/r11l_fullchain.json') as f:
                    existing = json.load(f)
                result = solve_game('r11l', R11l, 6,
                                    existing_seq=existing.get('full_sequence', []),
                                    click_step=2, max_depth=40, time_limit=240)
            elif gid == 's5i5':
                from s5i5 import S5i5
                with open(f'{RESULTS_DIR}/s5i5_fullchain.json') as f:
                    existing = json.load(f)
                result = solve_game('s5i5', S5i5, 8,
                                    existing_seq=existing.get('full_sequence', []),
                                    click_step=2, max_depth=40, time_limit=240)
            else:
                print(f'Unknown game: {gid}')
                continue

            out_path = f'{RESULTS_DIR}/{gid}_fullchain.json'
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f'\nSaved: {out_path}')
            ml = result.get('max_level_solved', 0)
            nl = result.get('total_levels', 0)
            ta = result.get('total_actions', 0)
            print(f'{gid.upper()}: {ml}/{nl} levels, {ta} actions')

        except Exception as e:
            print(f'\nERROR solving {gid}: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
