"""
Solve remaining levels for SK48, AR25, TN36, LP85, R11L.

Uses set_level() to jump directly to each level, then:
1. Scans for unique effective actions (grid click scan + keyboard)
2. BFS with frame hashing
3. Falls back to random search for high-action-space levels

Usage: PYTHONUTF8=1 python experiments/solve_5games.py [game_id] [level]
"""
import json
import time
import sys
import os
import hashlib
import importlib
import copy
import random
import numpy as np
from collections import deque
from arcengine import GameAction, ActionInput, GameState

GAME_PATHS = {
    'sk48': ('B:/M/the-search/environment_files/sk48/41055498', 'sk48', 'Sk48'),
    'ar25': ('B:/M/the-search/environment_files/ar25/e3c63847', 'ar25', 'Ar25'),
    'tn36': ('B:/M/the-search/environment_files/tn36/ab4f63cc', 'tn36', 'Tn36'),
    'lp85': ('B:/M/the-search/environment_files/lp85/305b61c3', 'lp85', 'Lp85'),
    'r11l': ('B:/M/the-search/environment_files/r11l/aa269680', 'r11l', 'R11l'),
}

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

# Number of levels per game
GAME_LEVELS = {'sk48': 8, 'ar25': 8, 'tn36': 7, 'lp85': 8, 'r11l': 6}

# Action mapping
ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}

# Keyboard actions per game (derived from available_actions in game class)
KB_ACTIONS = {
    'sk48': [0, 1, 2, 3, 5, 6],  # ACTION1-4, ACTION6(click), ACTION7
    'ar25': [0, 1, 2, 3, 4, 5, 6],  # ACTION1-7
    'tn36': [],  # click only (ACTION6)
    'lp85': [],  # click only (ACTION6)
    'r11l': [],  # click only (ACTION6)
}

# Existing L1 solutions from full_seq files
L1_SOLUTIONS = {}


def encode_click(x, y):
    return 7 + y * 64 + x

def decode_action(a):
    if a < 7: return f"KB{a}"
    ci = a - 7
    return f"CL({ci%64},{ci//64})"

def frame_hash(arr):
    if arr is None: return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def load_game_class(game_id):
    path, mod_name, cls_name = GAME_PATHS[game_id]
    if path not in sys.path:
        sys.path.insert(0, path)
    # Force reload in case module was cached
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def game_step(game, action):
    """Execute arcagi3 action. Return (frame_2d, levels_completed, state)."""
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
        return None, 0, None

    if r is None:
        return None, 0, None

    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3: f = f[-1]
    return f, r.levels_completed, r.state


def game_step_multiframe(game, action, max_frames=50):
    """
    Execute action and consume multi-frame animations.
    Keep stepping until the game stops consuming frames (action completed).
    Returns (final_frame, levels_completed, state).
    """
    f, lc, st = game_step(game, action)
    if f is None:
        return None, 0, None

    # For multi-frame games, we need to keep stepping to resolve animations
    # The game's step() function handles this internally through perform_action
    return f, lc, st


def make_at_level(game_cls, level_idx):
    """Create game set to specific level (0-indexed)."""
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


def replay_at_level(game_cls, level_idx, actions):
    """Create game at level, replay actions. Return (frame, levels_completed, state)."""
    g = make_at_level(game_cls, level_idx)
    frame = None
    levels = 0
    state = None
    for a in actions:
        frame, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN) or levels > 0:
            break
    return frame, levels, state


def get_base_frame(game_cls, level_idx):
    """Get base frame at start of level via a no-op click."""
    g = make_at_level(game_cls, level_idx)
    frame, levels, state = game_step(g, encode_click(0, 0))
    return frame


def find_unique_actions(game_cls, level_idx, game_id, click_step=2, verbose=True):
    """Find unique effective actions at this level by scanning grid."""
    kb = KB_ACTIONS[game_id]

    # Get base state
    base_frame = get_base_frame(game_cls, level_idx)
    if base_frame is None:
        # Try different no-op
        g = make_at_level(game_cls, level_idx)
        frame, levels, state = game_step(g, 0)
        base_frame = frame

    base_hash = frame_hash(base_frame) if base_frame is not None else ''
    groups = {}
    instant_sol = None

    # Keyboard actions
    for kb_a in kb:
        if kb_a == 5:  # ACTION6 = click, handle via grid
            continue
        frame, levels, state = replay_at_level(game_cls, level_idx, [kb_a])
        if frame is None:
            continue
        if levels > 0:
            return [kb_a], True
        h = frame_hash(frame)
        if h != base_hash and h not in groups:
            groups[h] = kb_a

    # Click grid scan
    clicks_checked = 0
    for y in range(0, 64, click_step):
        for x in range(0, 64, click_step):
            a = encode_click(x, y)
            frame, levels, state = replay_at_level(game_cls, level_idx, [a])
            clicks_checked += 1
            if frame is None:
                continue
            if levels > 0:
                return [a], True
            h = frame_hash(frame)
            if h != base_hash and h not in groups:
                groups[h] = a

    if verbose:
        print(f"    Scanned {clicks_checked} clicks + {len(kb)} kb -> {len(groups)} unique actions")
    return list(groups.values()), False


def bfs_solve_level(game_cls, level_idx, game_id, max_depth=50, max_states=500000, time_limit=300):
    """BFS solve level using set_level + unique actions."""
    t0 = time.time()

    print(f"    Scanning actions...")
    unique, instant = find_unique_actions(game_cls, level_idx, game_id, click_step=2)

    if instant:
        print(f"    INSTANT SOLVE!")
        return unique

    # If very few unique actions, try finer grid
    if len(unique) < 3:
        print(f"    Only {len(unique)} actions, retrying step=1...")
        unique, instant = find_unique_actions(game_cls, level_idx, game_id, click_step=1)
        if instant:
            return unique

    print(f"    {len(unique)} unique actions: {[decode_action(a) for a in unique[:20]]}{'...' if len(unique) > 20 else ''}")
    if not unique:
        return None

    # If too many unique actions, BFS won't scale; try random search
    if len(unique) > 50:
        print(f"    Too many actions ({len(unique)}), using random search...")
        return random_solve_level(game_cls, level_idx, unique, time_limit=time_limit)

    # BFS
    base_frame = get_base_frame(game_cls, level_idx)
    init_hash = frame_hash(base_frame)

    queue = deque([()])
    visited = {init_hash}
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

        for action in unique:
            actions = list(seq) + [action]
            frame, levels, state = replay_at_level(game_cls, level_idx, actions)
            explored += 1

            if levels > 0:
                print(f"    SOLVED! {len(actions)} actions, {explored} states, {time.time()-t0:.1f}s")
                return actions

            if state in (GameState.GAME_OVER, GameState.WIN) or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                # Fall back to random search
                remaining_time = time_limit - (time.time() - t0)
                if remaining_time > 10:
                    print(f"    Falling back to random search ({remaining_time:.0f}s left)...")
                    return random_solve_level(game_cls, level_idx, unique, time_limit=remaining_time)
                return None

    print(f"    EXHAUSTED ({explored})")
    return None


def random_solve_level(game_cls, level_idx, actions, time_limit=300, max_depth=100):
    """Random walk search for a level. Tries many random sequences."""
    t0 = time.time()
    best_depth = 0
    attempts = 0

    while time.time() - t0 < time_limit:
        attempts += 1
        seq = []
        g = make_at_level(game_cls, level_idx)

        depth = random.randint(1, max_depth)
        for _ in range(depth):
            a = random.choice(actions)
            seq.append(a)
            frame, levels, state = game_step(g, a)

            if levels > 0:
                elapsed = time.time() - t0
                print(f"    RANDOM SOLVED! {len(seq)} actions, {attempts} attempts, {elapsed:.1f}s")
                return seq

            if state in (GameState.GAME_OVER, GameState.WIN) or frame is None:
                break

        if attempts % 10000 == 0:
            elapsed = time.time() - t0
            print(f"      {attempts} attempts, {elapsed:.0f}s, rate={attempts/elapsed:.0f}/s")

    elapsed = time.time() - t0
    print(f"    RANDOM FAILED: {attempts} attempts in {elapsed:.1f}s")
    return None


def replay_full_chain(game_cls, full_sequence):
    """Replay entire chain from L1 to verify. Return (levels_completed, state)."""
    g = game_cls()
    g.full_reset()
    levels = 0
    state = None
    for a in full_sequence:
        _, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    return levels, state


def load_l1_solution(game_id):
    """Load existing L1 solution from prescription files."""
    # Try full_seq first
    seq_path = os.path.join(RESULTS_DIR, f'{game_id}_full_seq.json')
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            data = json.load(f)
        return data.get('full_sequence', [])

    # Try prescription
    pres_path = os.path.join(RESULTS_DIR, f'{game_id}_prescription.json')
    if os.path.exists(pres_path):
        with open(pres_path) as f:
            data = json.load(f)
        if '1' in data.get('levels', {}):
            return data['levels']['1'].get('trimmed_sequence', [])

    return []


def solve_game(game_id, start_level=None, max_depth=50, max_states=500000, time_limit=300):
    """Solve all remaining levels of a game."""
    print(f"\n{'='*70}")
    print(f"SOLVING {game_id.upper()} ({GAME_LEVELS[game_id]} levels)")
    print(f"{'='*70}")

    game_cls = load_game_class(game_id)
    total = GAME_LEVELS[game_id]

    # Load existing results
    existing_path = os.path.join(RESULTS_DIR, f'{game_id}_fullchain.json')
    if os.path.exists(existing_path):
        with open(existing_path) as f:
            results = json.load(f)
    else:
        results = {
            'game': game_id,
            'total_levels': total,
            'levels': {},
            'full_sequence': [],
            'method': 'bfs_set_level + random_search',
            'version': GAME_PATHS[game_id][0].split('/')[-1],
        }

    # Get L1 solution
    l1_seq = load_l1_solution(game_id)
    if not l1_seq:
        print(f"  No L1 solution found! Trying to solve L1...")
        # Try BFS for L1
        sol = bfs_solve_level(game_cls, 0, game_id, max_depth=max_depth,
                              max_states=max_states, time_limit=time_limit)
        if sol is None:
            print(f"  FAILED to solve L1")
            return results
        l1_seq = sol

    # Verify L1
    print(f"\n  L1: Verifying ({len(l1_seq)} actions)...")
    g = game_cls()
    g.full_reset()
    levels = 0
    for a in l1_seq:
        _, levels, state = game_step(g, a)
        if state in (GameState.GAME_OVER, GameState.WIN):
            break

    if levels >= 1:
        print(f"  L1 OK (verified)")
        results['levels']['L1'] = {
            'status': 'SOLVED', 'actions': l1_seq,
            'length': len(l1_seq), 'method': 'existing'
        }
        results['full_sequence'] = list(l1_seq)
    else:
        print(f"  L1 FAIL (sequence doesn't solve L1)")
        results['max_level_solved'] = 0
        return results

    current_max = 1

    # Determine start level
    if start_level is None:
        # Check which levels are already solved
        for lnum in range(2, total + 1):
            if f'L{lnum}' in results['levels'] and results['levels'][f'L{lnum}'].get('status') == 'SOLVED':
                current_max = lnum
            else:
                break
        start_level = current_max + 1

    # Solve remaining levels
    for lnum in range(start_level, total + 1):
        print(f"\n{'='*70}")
        print(f"  {game_id.upper()} Level {lnum}/{total} (level_idx={lnum-1})")
        print(f"{'='*70}")

        t0 = time.time()
        sol = bfs_solve_level(game_cls, lnum - 1, game_id,
                               max_depth=max_depth, max_states=max_states,
                               time_limit=time_limit)
        elapsed = time.time() - t0

        if sol is not None:
            results['levels'][f'L{lnum}'] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'time': round(elapsed, 2),
                'method': 'bfs_set_level'
            }
            results['full_sequence'].extend(sol)
            current_max = lnum

            # Verify full chain
            vlev, vstate = replay_full_chain(game_cls, results['full_sequence'])
            print(f"  Chain verified: levels={vlev}, state={vstate}")
            acts = [decode_action(a) for a in sol[:20]]
            print(f"  Actions: {acts}{'...' if len(sol) > 20 else ''}")
        else:
            results['levels'][f'L{lnum}'] = {
                'status': 'UNSOLVED', 'time': round(elapsed, 2)
            }
            print(f"  UNSOLVED ({elapsed:.1f}s)")
            # Don't break - try next levels with set_level
            # Actually break since we need chain continuity
            break

    results['max_level_solved'] = current_max
    results['total_actions'] = len(results['full_sequence'])

    # Save
    outpath = os.path.join(RESULTS_DIR, f'{game_id}_fullchain.json')
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  SAVED: {outpath}")
    print(f"  {current_max}/{total} levels, {len(results['full_sequence'])} total actions")

    return results


if __name__ == '__main__':
    os.chdir("B:/M/the-search")

    game = sys.argv[1] if len(sys.argv) > 1 else None
    level = int(sys.argv[2]) if len(sys.argv) > 2 else None

    games_to_solve = [game] if game else ['r11l', 'tn36', 'lp85', 'sk48', 'ar25']

    all_results = {}
    for gid in games_to_solve:
        if gid not in GAME_PATHS:
            print(f"Unknown game: {gid}")
            continue

        t0 = time.time()
        res = solve_game(gid, start_level=level, max_depth=60, max_states=500000, time_limit=300)
        total_t = time.time() - t0

        print(f"\n  {gid}: {res.get('max_level_solved', 0)}/{GAME_LEVELS[gid]} levels, "
              f"{res.get('total_actions', 0)} actions, {total_t:.0f}s total")
        all_results[gid] = res

    print(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for gid, res in all_results.items():
        print(f"  {gid}: {res.get('max_level_solved', 0)}/{GAME_LEVELS[gid]} levels, "
              f"{res.get('total_actions', 0)} total actions")
