"""
Fast BFS solver for S5I5 levels 3-8.
Uses state hashing from sprite positions instead of frame hashing.
Replays action sequences from set_level() for each state check.
Saves solutions to s5i5_fullchain.json.
"""
import sys
import os
import json
import time
import hashlib
import numpy as np
from collections import deque
from arcengine import GameAction, ActionInput, GameState

s5i5_path = 'B:/M/the-search/environment_files/s5i5/a48e4b1d'
if s5i5_path not in sys.path:
    sys.path.insert(0, s5i5_path)
from s5i5 import S5i5

def encode_click(x, y):
    return 7 + y * 64 + x

def decode_action(a):
    ci = a - 7
    return f"CL({ci%64},{ci//64})"

def make_at_level(level_idx):
    g = S5i5()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g

def do_action(g, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    return g.perform_action(ai, raw=True)

def state_hash(g):
    """Fast state hash from sprite positions."""
    parts = []
    for s in g.current_level.get_sprites_by_tag('agujdcrunq'):
        parts.append((s.name, s.x, s.y, s.width, s.height))
    for s in g.current_level.get_sprites_by_tag('zylvdxoiuq'):
        parts.append(('P', s.x, s.y))
    return hash(tuple(sorted(parts)))

def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()

def replay_and_check(level_idx, actions):
    """Replay actions at level, return (state_h, levels_completed, game_over)."""
    g = make_at_level(level_idx)
    for a in actions:
        r = do_action(g, a)
        if r is None:
            return None, 0, True
        if r.state == GameState.GAME_OVER:
            return None, 0, True
        if r.levels_completed > 0:
            return None, r.levels_completed, False
    sh = state_hash(g)
    return sh, 0, False

def find_unique_actions(level_idx, click_step=2):
    """Find distinct actions by state hash."""
    g0 = make_at_level(level_idx)
    # Get base state hash with noop
    do_action(g0, encode_click(0, 0))
    base_h = state_hash(g0)

    groups = {}
    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            g = make_at_level(level_idx)
            r = do_action(g, a)
            if r is None: continue
            if r.levels_completed > 0:
                return [a], True
            sh = state_hash(g)
            if sh != base_h and sh not in groups:
                groups[sh] = a
    return list(groups.values()), False

def bfs_solve(level_idx, actions, max_depth=50, max_states=2000000, time_limit=600):
    """BFS with state hashing. Returns action list or None."""
    t0 = time.time()

    g0 = make_at_level(level_idx)
    do_action(g0, encode_click(0, 0))
    init_h = state_hash(g0)

    print(f"    BFS: {len(actions)} actions, max_depth={max_depth}")

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
            new_seq = list(seq) + [action]
            explored += 1

            sh, levels, game_over = replay_and_check(level_idx, new_seq)

            if levels > 0:
                el = time.time() - t0
                print(f"    SOLVED! {len(new_seq)} steps, {explored} states, {el:.1f}s")
                return new_seq

            if game_over or sh is None:
                continue

            if sh not in visited:
                visited.add(sh)
                queue.append(tuple(new_seq))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


def game_step(game, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    r = game.perform_action(ai, raw=True)
    if r is None: return 0, None
    return r.levels_completed, r.state


def verify_chain(full_seq):
    """Verify full chain from L1."""
    g = S5i5()
    g.full_reset()
    levels = 0
    for a in full_seq:
        levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    return levels


if __name__ == '__main__':
    os.chdir('B:/M/the-search')

    # Load existing solution
    with open('experiments/results/prescriptions/s5i5_fullchain.json') as f:
        existing = json.load(f)

    full_seq = list(existing['full_sequence'])

    # Verify existing
    max_lev = verify_chain(full_seq)
    print(f"Existing chain: {max_lev} levels, {len(full_seq)} actions")

    results = {
        'game': 's5i5',
        'version': 'a48e4b1d',
        'total_levels': 8,
        'method': 'bfs_set_level + analytical',
        'levels': {
            'L1': existing['levels'].get('L1', {'status': 'SOLVED'}),
            'L2': existing['levels'].get('L2', {'status': 'SOLVED'}),
        },
        'full_sequence': list(full_seq),
    }

    current_max = max_lev

    for lnum in range(3, 9):
        if lnum <= current_max:
            continue

        level_idx = lnum - 1
        print(f"\n{'='*60}")
        print(f"S5I5 Level {lnum}")
        print(f"{'='*60}")

        t0 = time.time()

        # Find unique actions
        unique, instant = find_unique_actions(level_idx, click_step=2)

        if instant:
            sol = unique
            print(f"  INSTANT!")
        else:
            print(f"  {len(unique)} unique actions: {[decode_action(a) for a in unique]}")
            sol = bfs_solve(level_idx, unique, max_depth=40, max_states=2000000, time_limit=500)

        elapsed = time.time() - t0

        if sol is not None:
            # Verify in chain
            test_seq = results['full_sequence'] + sol
            chain_levels = verify_chain(test_seq)
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
                print(f"  CHAIN FAIL")
                break
        else:
            results['levels'][f'L{lnum}'] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            break

    results['max_level_solved'] = current_max
    results['max_level_solved_chain'] = current_max
    results['total_actions'] = len(results['full_sequence'])

    out_path = 'experiments/results/prescriptions/s5i5_fullchain.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Result: {current_max}/8 levels, {results['total_actions']} actions")
