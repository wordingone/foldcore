"""
IDDFS solver for ARC-AGI-3 games.
Uses single-env reset+replay approach.
For games with few effective actions, this is very fast.
"""
import sys, json, os, time, hashlib, random
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def get_click_positions(env):
    """Get actual clickable positions from sys_click sprites."""
    game = env._game
    try:
        click_sprites = game.current_level.get_sprites_by_tag('sys_click')
    except:
        click_sprites = []

    positions = set()
    for s in click_sprites:
        cx = s.x + s.width // 2
        cy = s.y + s.height // 2
        positions.add((cx, cy))

    if not positions:
        # Fallback: scan grid
        obs_before = env.reset()
        f_before = np.array(obs_before.frame, dtype=np.uint8)
        h_before = hashlib.md5(f_before.tobytes()).hexdigest()

        for y in range(0, 64, 4):
            for x in range(0, 64, 4):
                env.reset()
                obs = env.step(GameAction.ACTION6, data={'x': x, 'y': y})
                if obs:
                    f = np.array(obs.frame, dtype=np.uint8)
                    h = hashlib.md5(f.tobytes()).hexdigest()
                    if h != h_before:
                        positions.add((x, y))

    return sorted(positions)


def replay_seq(env, actions_list, action_set, prefix=[]):
    """Reset env, replay prefix + actions. Returns obs."""
    obs = env.reset()
    for act in prefix:
        if isinstance(act, tuple):
            obs = env.step(act[0], data=act[1])
        else:
            obs = env.step(act)
    for a_idx in actions_list:
        act = action_set[a_idx]
        if isinstance(act, tuple):
            obs = env.step(act[0], data=act[1])
        else:
            obs = env.step(act)
    return obs


def solve_level_bfs(env, action_set, prefix=[], max_depth=40, time_limit=300,
                    max_states=500000, verbose=True):
    """
    BFS with hash dedup for one level.
    Uses reset+replay but with optimized queue (stores action index sequences).
    """
    from collections import deque

    # Get initial hash
    obs = replay_seq(env, [], action_set, prefix)
    f = np.array(obs.frame, dtype=np.uint8)
    init_hash = hashlib.md5(f.tobytes()).hexdigest()

    queue = deque()
    queue.append([])
    visited = {init_hash}
    states = 0
    t0 = time.time()

    while queue:
        seq = queue.popleft()
        if len(seq) >= max_depth:
            continue

        for a_idx in range(len(action_set)):
            new_seq = seq + [a_idx]
            obs = replay_seq(env, new_seq, action_set, prefix)
            states += 1

            if states % 5000 == 0 and verbose:
                elapsed = time.time() - t0
                print(f'  BFS: {states} states, queue={len(queue)}, depth={len(new_seq)}, '
                      f'visited={len(visited)}, {elapsed:.1f}s')

            if obs and obs.state == GameState.WIN:
                if verbose:
                    elapsed = time.time() - t0
                    print(f'  SOLVED at depth {len(new_seq)}! States: {states}, {elapsed:.1f}s')
                return new_seq

            if obs is None or obs.state == GameState.GAME_OVER:
                continue

            f = np.array(obs.frame, dtype=np.uint8)
            h = hashlib.md5(f.tobytes()).hexdigest()

            if h not in visited:
                visited.add(h)
                queue.append(new_seq)

            if time.time() - t0 > time_limit:
                if verbose:
                    print(f'  Time limit ({time_limit}s). States: {states}, visited: {len(visited)}')
                return None

            if states >= max_states:
                if verbose:
                    print(f'  State limit ({max_states}). Visited: {len(visited)}')
                return None

    if verbose:
        print(f'  BFS exhausted. States: {states}, visited: {len(visited)}')
    return None


def solve_game_full(game_key, verbose=True):
    """Solve all levels of a game."""
    arc = arc_agi.Arcade()
    all_games = arc.get_environments()
    info = next(g for g in all_games if game_key in g.game_id.lower())

    env = arc.make(info.game_id)
    obs = env.reset()

    kb_actions = [a for a in env.action_space if a != GameAction.ACTION6]
    has_click = GameAction.ACTION6 in env.action_space
    is_click_only = info.tags == ['click']
    n_levels = len(info.baseline_actions)

    if verbose:
        print(f'\n{"="*70}')
        print(f'SOLVING {game_key.upper()} ({n_levels} levels)')
        print(f'  Tags: {info.tags}')
        print(f'  KB: {[a.name for a in kb_actions]}')
        print(f'  Baseline: {info.baseline_actions}')
        print(f'{"="*70}')

    # Build action set
    action_set = []
    if is_click_only:
        click_pos = get_click_positions(env)
        for x, y in click_pos:
            action_set.append((GameAction.ACTION6, {'x': x, 'y': y}))
        if verbose:
            print(f'  Click positions: {len(click_pos)}')
    else:
        action_set = list(kb_actions)
        if has_click:
            click_pos = get_click_positions(env)
            for x, y in click_pos:
                action_set.append((GameAction.ACTION6, {'x': x, 'y': y}))

    if verbose:
        print(f'  Action set size: {len(action_set)}')

    prefix = []
    per_level = {}
    all_raw_actions = []

    for level_idx in range(n_levels):
        if verbose:
            print(f'\n--- Level {level_idx + 1}/{n_levels} (baseline: {info.baseline_actions[level_idx]}) ---')

        baseline = info.baseline_actions[level_idx]
        max_d = min(50, max(5, int(baseline * 1.5)))

        # For click games, update click positions each level
        if is_click_only and level_idx > 0:
            obs = replay_seq(env, [], action_set, prefix)
            new_clicks = get_click_positions(env)
            if new_clicks:
                action_set = [(GameAction.ACTION6, {'x': x, 'y': y}) for x, y in new_clicks]
                if verbose:
                    print(f'  Updated click positions: {len(new_clicks)}')

        result = solve_level_bfs(env, action_set, prefix, max_depth=max_d,
                                time_limit=180, verbose=verbose)

        if result is None:
            if verbose:
                print(f'  FAILED level {level_idx + 1}')
            break

        # Convert to raw actions and extend prefix
        level_actions = []
        for a_idx in result:
            act = action_set[a_idx]
            if isinstance(act, tuple):
                level_actions.append(act)
            else:
                level_actions.append(act)

        prefix.extend(level_actions)
        all_raw_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(result)

        if verbose:
            print(f'  Solution: {len(result)} actions')

        # Verify
        obs = replay_seq(env, [], action_set, prefix)
        if obs and obs.state == GameState.WIN:
            if verbose:
                print(f'  GAME COMPLETE!')
            break

    # Serialize
    serialized = []
    for act in all_raw_actions:
        if isinstance(act, tuple):
            ga, data = act
            serialized.append(data['y'] * 64 + data['x'])
        else:
            serialized.append(act.value - 1)

    total = sum(per_level.values())
    result = {
        'game': game_key,
        'source': 'iddfs_solver',
        'type': 'analytical',
        'total_actions': total,
        'max_level': len(per_level),
        'n_levels': n_levels,
        'per_level': per_level,
        'baseline': info.baseline_actions,
        'all_actions': serialized,
    }

    return result


def main():
    games = sys.argv[1:] if len(sys.argv) > 1 else ['tn36', 'lp85']

    for gid in games:
        t0 = time.time()
        try:
            result = solve_game_full(gid, verbose=True)
            result['elapsed'] = round(time.time() - t0, 2)

            out_path = os.path.join(RESULTS_DIR, f'{gid}_fullchain.json')
            with open(out_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f'\nSaved: {out_path}')
            print(f'  {result["max_level"]}/{result["n_levels"]} levels, '
                  f'{result["total_actions"]} actions, {result["elapsed"]}s')

        except Exception as e:
            print(f'ERROR: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
