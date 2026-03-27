"""
BFS solver for DC22 — maze + crane puzzle.
L1-L2: existing solution.
L3+: BFS with replay-based state exploration.
"""
import sys, json, os, time
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState, InteractionMode
from collections import deque

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def make_env_at(info, raw_actions):
    """Create fresh env and replay actions to reach a specific state."""
    arc = arc_agi.Arcade()
    env = arc.make(info.game_id)
    obs = env.reset()
    for raw in raw_actions:
        if raw <= 3:
            obs = env.step([GameAction.ACTION1, GameAction.ACTION2,
                           GameAction.ACTION3, GameAction.ACTION4][raw])
        else:
            obs = env.step(GameAction.ACTION6,
                          data={'x': raw % 64, 'y': raw // 64})
    return env, obs


def get_state_key(game):
    """Compact hashable state. The moveable piece is fdvakicpimr, NOT bqxa."""
    mx, my = game.fdvakicpimr.x, game.fdvakicpimr.y
    # Track wall/sprite interaction states
    sprite_states = []
    for s in game.current_level.get_sprites():
        if 'wbze' in s.tags or 'itki' in s.tags:
            sprite_states.append((s.name, s.x, s.y, s.interaction.value))
    return (mx, my, tuple(sorted(sprite_states)))


def get_actions(game):
    """Get available actions: 4 directions + button clicks."""
    actions = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
    cam = game.camera
    seen = set()
    for s in game.current_level.get_sprites():
        if ('jpug' in s.tags or 'sys_click' in s.tags) and s.is_visible:
            sx = s.x - cam.x + s.width // 2
            sy = s.y - cam.y + s.height // 2
            raw = sy * 64 + sx
            if raw not in seen:
                seen.add(raw)
                actions.append(raw)
    return actions


def bfs_solve(info, prefix_actions, max_actions=300, time_limit=240):
    """BFS with replay-based state exploration."""
    start = time.time()

    env0, _ = make_env_at(info, prefix_actions)
    game0 = env0._game
    initial_level = game0.level_index
    initial_key = get_state_key(game0)
    available_actions = get_actions(game0)

    print(f'  Initial: moveable=({game0.fdvakicpimr.x},{game0.fdvakicpimr.y}), '
          f'target=({game0.bqxa.x},{game0.bqxa.y}), '
          f'actions={len(available_actions)}')

    queue = deque()
    queue.append([])
    visited = {initial_key}
    explored = 0

    while queue:
        if time.time() - start > time_limit:
            print(f'  Timeout after {explored} nodes, {len(visited)} states')
            return None

        actions_so_far = queue.popleft()

        if len(actions_so_far) >= max_actions:
            continue

        for action_raw in available_actions:
            new_actions = actions_so_far + [action_raw]

            # Replay
            env2, obs2 = make_env_at(info, prefix_actions + new_actions)
            game2 = env2._game
            explored += 1

            # Check level advanced
            if game2.level_index > initial_level:
                elapsed = time.time() - start
                print(f'  SOLVED in {len(new_actions)} actions, {explored} nodes, {elapsed:.1f}s')
                return new_actions

            # Skip game over
            if obs2 and obs2.state == GameState.GAME_OVER:
                continue

            # Check state
            key = get_state_key(game2)
            if key not in visited:
                visited.add(key)
                queue.append(new_actions)

            if explored % 500 == 0:
                elapsed = time.time() - start
                px, py = game2.bqxa.x, game2.bqxa.y
                print(f'  [{explored}] depth={len(new_actions)} states={len(visited)} '
                      f'queue={len(queue)} player=({px},{py}) {elapsed:.1f}s')

    print(f'  Exhausted: {explored} nodes, {len(visited)} states')
    return None


def main():
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    info = next(g for g in games if 'dc22' in g.game_id.lower())

    print(f'DC22: {len(info.baseline_actions)} levels, baseline={info.baseline_actions}')

    # Load existing L1+L2 from analytical solver
    l1_actions = [2416, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 1264, 0, 0, 0, 2416, 0, 0, 3, 3]
    l2_actions = [2612, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 1460, 1, 1, 1, 1, 1, 1,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2036, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    all_raw = l1_actions + l2_actions
    per_level = {'L1': len(l1_actions), 'L2': len(l2_actions)}

    # Verify
    env, obs = make_env_at(info, all_raw)
    print(f'After L1+L2: level={env._game.level_index}')

    for level_idx in range(2, len(info.baseline_actions)):
        print(f'\n--- Level {level_idx + 1}/{len(info.baseline_actions)} (baseline={info.baseline_actions[level_idx]}) ---')

        level_actions = bfs_solve(info, all_raw,
                                  max_actions=info.baseline_actions[level_idx] + 20,
                                  time_limit=240)

        if level_actions is None:
            print(f'  FAILED')
            break

        all_raw.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        env, obs = make_env_at(info, all_raw)
        print(f'  After: level={env._game.level_index}, state={obs.state.name}')

        if obs and obs.state == GameState.WIN:
            print('  GAME COMPLETE!')
            break

    # Save
    result = {
        'game': 'dc22',
        'source': 'bfs_solver',
        'type': 'fullchain',
        'total_actions': len(all_raw),
        'max_level': len(per_level),
        'n_levels': len(info.baseline_actions),
        'per_level': per_level,
        'baseline': info.baseline_actions,
        'all_actions': all_raw,
    }

    out_path = os.path.join(RESULTS_DIR, 'dc22_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
