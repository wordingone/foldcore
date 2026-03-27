"""
BFS solver for SP80 (L3-L6) and S5I5 (L3-L8).
Uses set_level() to jump directly to each level. Frame hashing for dedup.
"""
import sys
import os
import json
import time
import hashlib
import importlib
import numpy as np
from collections import deque
from arcengine import GameAction, ActionInput, GameState

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

def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def game_step(game, action):
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
    except:
        return None, 0, None
    if r is None:
        return None, 0, None
    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    return f, r.levels_completed, r.state


def make_at_level(game_cls, level_idx):
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


def replay_at_level(game_cls, level_idx, actions):
    g = make_at_level(game_cls, level_idx)
    frame = None
    levels = 0
    state = None
    for a in actions:
        frame, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN) or levels > 0:
            break
    return frame, levels, state


def find_unique_actions(game_cls, level_idx, kb_actions=[], click_step=2):
    base_f, _, _ = replay_at_level(game_cls, level_idx, [encode_click(0, 0)])
    base_h = frame_hash(base_f)
    groups = {}

    for kb in kb_actions:
        f, lev, st = replay_at_level(game_cls, level_idx, [kb])
        if f is None: continue
        if lev > 0: return [kb], True
        h = frame_hash(f)
        if h != base_h and h not in groups:
            groups[h] = kb

    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            f, lev, st = replay_at_level(game_cls, level_idx, [a])
            if f is None: continue
            if lev > 0: return [a], True
            h = frame_hash(f)
            if h != base_h and h not in groups:
                groups[h] = a

    return list(groups.values()), False


def bfs_solve(game_cls, level_idx, actions, max_depth=50, max_states=1000000, time_limit=300):
    """BFS with frame hashing."""
    t0 = time.time()

    base_f, _, _ = replay_at_level(game_cls, level_idx, [encode_click(0, 0)])
    init_h = frame_hash(base_f)

    queue = deque([()])
    visited = {init_h}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, e={explored}, d={depth})")
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.0f}s ({rate:.0f}/s)")

        if len(seq) >= max_depth:
            continue

        for action in actions:
            new_actions = list(seq) + [action]
            frame, levels, state = replay_at_level(game_cls, level_idx, new_actions)
            explored += 1

            if levels > 0:
                el = time.time() - t0
                print(f"    SOLVED! {len(new_actions)} steps, {explored} states, {el:.1f}s")
                return new_actions

            if state == GameState.GAME_OVER or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(new_actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


def replay_full_chain(game_cls, full_sequence):
    """Replay from scratch, return levels_completed."""
    g = game_cls()
    g.full_reset()
    levels = 0
    for a in full_sequence:
        _, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    return levels


def solve_game(game_id, game_cls, start_level, end_level, existing_seq, kb_actions=[]):
    results = {
        'game': game_id,
        'levels': {},
        'full_sequence': list(existing_seq),
    }

    # Verify existing
    g = game_cls()
    g.full_reset()
    levels = 0
    for a in existing_seq:
        _, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN): break
    print(f"  Existing chain: {levels} levels")
    current_max = levels

    for lnum in range(start_level, end_level + 1):
        level_idx = lnum - 1
        print(f"\n{'='*60}")
        print(f"{game_id.upper()} Level {lnum}")
        print(f"{'='*60}")

        t0 = time.time()

        unique, instant = find_unique_actions(game_cls, level_idx, kb_actions, click_step=2)

        if instant:
            sol = unique
            print(f"  INSTANT!")
        else:
            print(f"  {len(unique)} unique actions: {[decode_action(a) for a in unique]}")
            sol = bfs_solve(game_cls, level_idx, unique,
                          max_depth=50, max_states=1000000, time_limit=300)

        elapsed = time.time() - t0

        if sol is not None:
            # Verify in chain
            test_seq = results['full_sequence'] + sol
            chain_levels = replay_full_chain(game_cls, test_seq)
            print(f"  Chain: {chain_levels} levels")

            if chain_levels >= lnum:
                results['levels'][f'L{lnum}'] = {
                    'status': 'SOLVED',
                    'actions': sol,
                    'length': len(sol),
                    'time': round(elapsed, 2),
                    'decoded': [decode_action(a) for a in sol],
                }
                results['full_sequence'].extend(sol)
                current_max = lnum
                print(f"  SOLVED L{lnum}: {len(sol)} actions")
            else:
                results['levels'][f'L{lnum}'] = {'status': 'CHAIN_FAIL'}
                print(f"  CHAIN FAIL (got {chain_levels})")
                break
        else:
            results['levels'][f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            break

    results['max_level'] = current_max
    results['total_actions'] = len(results['full_sequence'])
    return results


if __name__ == '__main__':
    os.chdir('B:/M/the-search')
    games = sys.argv[1:] if len(sys.argv) > 1 else ['sp80', 's5i5']

    for gid in games:
        if gid == 'sp80':
            sp80_path = 'B:/M/the-search/environment_files/sp80/0ee2d095'
            if sp80_path not in sys.path:
                sys.path.insert(0, sp80_path)
            from sp80 import Sp80

            with open('B:/M/the-search/experiments/results/prescriptions/sp80_fullchain.json') as f:
                existing = json.load(f)

            result = solve_game('sp80', Sp80, 3, 6, existing['sequence'], kb_actions=[0,1,2,3,4])

            out = 'B:/M/the-search/experiments/results/prescriptions/sp80_fullchain.json'
            with open(out, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {out}")
            print(f"Result: {result['max_level']}/6, {result['total_actions']} actions")

        elif gid == 's5i5':
            s5i5_path = 'B:/M/the-search/environment_files/s5i5/a48e4b1d'
            if s5i5_path not in sys.path:
                sys.path.insert(0, s5i5_path)
            from s5i5 import S5i5

            with open('B:/M/the-search/experiments/results/prescriptions/s5i5_fullchain.json') as f:
                existing = json.load(f)

            result = solve_game('s5i5', S5i5, 3, 8, existing['full_sequence'])

            out = 'B:/M/the-search/experiments/results/prescriptions/s5i5_fullchain.json'
            with open(out, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved: {out}")
            print(f"Result: {result['max_level']}/8, {result['total_actions']} actions")
