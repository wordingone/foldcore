"""
Full-chain solver for DC22 (L3-L6) and SC25 (L2-L6).
Uses set_level() for fast BFS with frame hashing.
"""
import sys, json, time, os, hashlib, importlib, copy
import numpy as np
from collections import deque
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')

from arcengine import GameAction, ActionInput, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

GAME_PATHS = {
    'dc22': ('B:/M/the-search/environment_files/dc22/4c9bff3e', 'dc22', 'Dc22'),
    'sc25': ('B:/M/the-search/environment_files/sc25/f9b21a2f', 'sc25', 'Sc25'),
}

# Known solutions
DC22_L1 = [2416, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1264, 0, 0, 0, 2416, 0, 0, 3, 3]
DC22_L2 = [2612, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1460, 1, 1, 1, 1, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2036, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
SC25_L1 = [2, 3230, 3545, 3555, 3870, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

ARCAGI3_TO_GA = {
    0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
    3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
    6: GameAction.ACTION7,
}


def encode_click(x, y):
    return 7 + y * 64 + x


def decode_action(a):
    if a < 7:
        names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'ACT5', 'CLICK(0,0)', 'ACT7']
        return names[a] if a < len(names) else f"KB{a}"
    ci = a - 7
    return f"CL({ci % 64},{ci // 64})"


def frame_hash(arr):
    if arr is None:
        return ''
    return hashlib.md5(arr.astype(np.uint8).tobytes()).hexdigest()


def load_game_class(game_id):
    path, mod_name, cls_name = GAME_PATHS[game_id]
    if path not in sys.path:
        sys.path.insert(0, path)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def game_step(game, action):
    """Execute action. Return (frame_2d, levels_completed, done, state)."""
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
        return None, 0, False, None
    if r is None:
        return None, 0, False, None

    f = np.array(r.frame, dtype=np.uint8)
    if f.ndim == 3:
        f = f[-1]
    done = r.state in (GameState.GAME_OVER, GameState.WIN)
    return f, r.levels_completed, done, r.state


def make_at_level(game_cls, level_idx):
    """Create game at specific level."""
    g = game_cls()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    return g


def replay_at_level(game_cls, level_idx, actions):
    """Create game at level, replay actions."""
    g = make_at_level(game_cls, level_idx)
    frame = None
    levels = 0
    done = False
    state = None
    for a in actions:
        frame, levels, done, state = game_step(g, a)
        if done or levels > 0:
            break
    return frame, levels, done, state, g


def find_unique_actions(game_cls, level_idx, kb_actions, click_step=2, max_y=64, max_x=64):
    """Find unique effective actions at this level."""
    g0 = make_at_level(game_cls, level_idx)
    # Get base frame with a no-op
    base_f, _, _, _ = game_step(g0, encode_click(0, 0))
    base_hash = frame_hash(base_f)
    groups = {}

    # Keyboard actions
    for kb in kb_actions:
        frame, levels, _, _, _ = replay_at_level(game_cls, level_idx, [kb])
        if frame is None:
            continue
        if levels > 0:
            return [kb], True
        h = frame_hash(frame)
        if h != base_hash and h not in groups:
            groups[h] = kb

    # Click grid
    for y in range(0, max_y, click_step):
        for x in range(0, max_x, click_step):
            a = encode_click(x, y)
            frame, levels, _, _, _ = replay_at_level(game_cls, level_idx, [a])
            if frame is None:
                continue
            if levels > 0:
                return [a], True
            h = frame_hash(frame)
            if h != base_hash and h not in groups:
                groups[h] = a

    return list(groups.values()), False


def bfs_solve_level(game_cls, level_idx, kb_actions, max_depth=80, max_states=600000,
                    time_limit=240, click_step=2):
    """BFS solve a level using set_level + unique actions + frame hashing."""
    t0 = time.time()

    # Get base frame
    g0 = make_at_level(game_cls, level_idx)
    base_f, _, _, _ = game_step(g0, encode_click(0, 0))
    init_hash = frame_hash(base_f)

    print(f"    Scanning actions (step={click_step})...")
    unique, instant = find_unique_actions(game_cls, level_idx, kb_actions,
                                          click_step=click_step)
    if instant:
        print(f"    INSTANT: {[decode_action(a) for a in unique]}")
        return unique

    # If very few, try finer scan
    if len(unique) < 3 and click_step > 1:
        print(f"    Only {len(unique)} actions, retrying step=1...")
        unique2, instant2 = find_unique_actions(game_cls, level_idx, kb_actions,
                                                 click_step=1)
        if instant2:
            print(f"    INSTANT: {[decode_action(a) for a in unique2]}")
            return unique2
        if len(unique2) > len(unique):
            unique = unique2

    print(f"    {len(unique)} unique actions: {[decode_action(a) for a in unique[:20]]}"
          f"{'...' if len(unique) > 20 else ''}")
    if not unique:
        print(f"    No effective actions found!")
        return None

    # BFS
    queue = deque([()])
    visited = {init_hash}
    explored = 0
    depth = 0

    while queue:
        if time.time() - t0 > time_limit:
            print(f"    TIMEOUT ({time_limit}s, explored={explored}, depth={depth}, "
                  f"visited={len(visited)})")
            return None

        seq = queue.popleft()

        if len(seq) > depth:
            depth = len(seq)
            el = time.time() - t0
            rate = explored / max(el, 0.1)
            print(f"      d={depth} v={len(visited)} q={len(queue)} "
                  f"e={explored} t={el:.0f}s ({rate:.0f}/s)")

        if len(seq) >= max_depth:
            continue

        for action in unique:
            actions = list(seq) + [action]
            frame, levels, done, state, g = replay_at_level(game_cls, level_idx, actions)
            explored += 1

            if levels > 0:
                print(f"    SOLVED! {len(actions)} actions, {explored} explored, "
                      f"{time.time() - t0:.1f}s")
                return actions

            if done or frame is None:
                continue

            h = frame_hash(frame)
            if h not in visited:
                visited.add(h)
                queue.append(tuple(actions))

            if explored >= max_states:
                print(f"    LIMIT ({max_states}, d={depth})")
                return None

    print(f"    EXHAUSTED ({explored} explored, {len(visited)} states)")
    return None


def replay_full_chain(game_cls, full_sequence):
    """Replay entire chain. Return (levels_completed, state)."""
    g = game_cls()
    g.full_reset()
    levels = 0
    state = None
    for a in full_sequence:
        _, levels, done, state = game_step(g, a)
        if done:
            break
    return levels, state


# ============================================================
# DC22 SOLVER
# ============================================================
def solve_dc22():
    print(f"\n{'=' * 70}")
    print(f"SOLVING DC22 (6 levels)")
    print(f"{'=' * 70}")

    game_cls = load_game_class('dc22')
    n_levels = 6

    # L1 + L2 known
    full_seq = list(DC22_L1) + list(DC22_L2)
    per_level = {'L1': len(DC22_L1), 'L2': len(DC22_L2)}

    # Verify L1+L2
    print(f"\nVerifying L1+L2 ({len(full_seq)} actions)...")
    levels, state = replay_full_chain(game_cls, full_seq)
    print(f"  After L1+L2: levels_completed={levels}")
    if levels < 2:
        print(f"  FAILED to verify L1+L2!")
        return None

    # DC22: actions are UP(0), DOWN(1), LEFT(2), RIGHT(3), clicks
    kb_actions = [0, 1, 2, 3]

    for lnum in range(3, n_levels + 1):
        print(f"\n--- Level {lnum}/{n_levels} ---")
        t0 = time.time()

        sol = bfs_solve_level(game_cls, lnum - 1, kb_actions,
                               max_depth=80, max_states=600000,
                               time_limit=240, click_step=2)
        elapsed = time.time() - t0

        if sol is not None:
            full_seq.extend(sol)
            per_level[f'L{lnum}'] = len(sol)
            print(f"  L{lnum}: {len(sol)} actions in {elapsed:.1f}s")
            print(f"  Actions: {[decode_action(a) for a in sol[:30]]}"
                  f"{'...' if len(sol) > 30 else ''}")

            # Verify chain so far
            vlev, vst = replay_full_chain(game_cls, full_seq)
            print(f"  Chain verify: levels={vlev}")
        else:
            print(f"  L{lnum}: UNSOLVED ({elapsed:.1f}s)")
            break

    # Save
    result = {
        'game': 'dc22',
        'source': 'bfs_set_level_solver',
        'type': 'fullchain',
        'total_actions': len(full_seq),
        'max_level': len(per_level),
        'n_levels': n_levels,
        'per_level': per_level,
        'all_actions': full_seq,
    }

    out_path = os.path.join(RESULTS_DIR, 'dc22_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Result: {len(per_level)}/{n_levels} levels, {len(full_seq)} actions")
    return result


# ============================================================
# SC25 SOLVER
# ============================================================
def solve_sc25():
    print(f"\n{'=' * 70}")
    print(f"SOLVING SC25 (6 levels)")
    print(f"{'=' * 70}")

    game_cls = load_game_class('sc25')
    n_levels = 6

    # L1 known
    full_seq = list(SC25_L1)
    per_level = {'L1': len(SC25_L1)}

    # Verify L1
    print(f"\nVerifying L1 ({len(SC25_L1)} actions)...")
    levels, state = replay_full_chain(game_cls, full_seq)
    print(f"  After L1: levels_completed={levels}")
    if levels < 1:
        print(f"  FAILED to verify L1!")
        return None

    # SC25: actions are UP(0), DOWN(1), LEFT(2), RIGHT(3), clicks
    # Spell slot clicks are the key mechanic
    kb_actions = [0, 1, 2, 3]

    for lnum in range(2, n_levels + 1):
        print(f"\n--- Level {lnum}/{n_levels} ---")
        t0 = time.time()

        sol = bfs_solve_level(game_cls, lnum - 1, kb_actions,
                               max_depth=50, max_states=600000,
                               time_limit=240, click_step=2)
        elapsed = time.time() - t0

        if sol is not None:
            full_seq.extend(sol)
            per_level[f'L{lnum}'] = len(sol)
            print(f"  L{lnum}: {len(sol)} actions in {elapsed:.1f}s")
            print(f"  Actions: {[decode_action(a) for a in sol[:30]]}"
                  f"{'...' if len(sol) > 30 else ''}")

            # Verify chain
            vlev, vst = replay_full_chain(game_cls, full_seq)
            print(f"  Chain verify: levels={vlev}")
        else:
            print(f"  L{lnum}: UNSOLVED ({elapsed:.1f}s)")
            break

    # Save
    result = {
        'game': 'sc25',
        'source': 'bfs_set_level_solver',
        'type': 'fullchain',
        'total_actions': len(full_seq),
        'max_level': len(per_level),
        'n_levels': n_levels,
        'per_level': per_level,
        'all_actions': full_seq,
    }

    out_path = os.path.join(RESULTS_DIR, 'sc25_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Result: {len(per_level)}/{n_levels} levels, {len(full_seq)} actions")
    return result


if __name__ == '__main__':
    games = sys.argv[1:] if len(sys.argv) > 1 else ['dc22', 'sc25']

    for gid in games:
        if gid == 'dc22':
            solve_dc22()
        elif gid == 'sc25':
            solve_sc25()
