"""
LP85 abstract solver - goals move along cycles, tiles are fixed.
State = goal positions. Target = goals at tile_pos + (1,1).
"""
import sys
import json
import time
from collections import deque

sys.path.insert(0, 'B:/M/the-search/environment_files/lp85/305b61c3')
from lp85 import Lp85, izutyjcpih, chmfaflqhy, qfvvosdkqr, crxpafuiwp
from arcengine import GameAction, ActionInput, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def encode_click(x, y):
    return 7 + y * 64 + x


def game_step(game, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    r = game.perform_action(ai, raw=True)
    if r is None:
        return 0, None
    return r.levels_completed, r.state


def solve():
    all_actions = []
    per_level = {}
    uopmnplcnv = qfvvosdkqr(izutyjcpih)

    for level_idx in range(8):
        print(f'\n=== Level {level_idx + 1} ===')
        t0 = time.time()

        g = Lp85()
        g.full_reset()
        if level_idx > 0:
            g.set_level(level_idx)

        level_name = g.ucybisahh
        step_budget = g.toxpunyqe.current_steps

        # Get target positions (fixed tiles)
        targets_b = []  # (x+1, y+1) positions where goals must be
        for s in g.current_level.get_sprites_by_tag('bghvgbtwcb'):
            targets_b.append((s.x + 1, s.y + 1))
        targets_f = []
        for s in g.current_level.get_sprites_by_tag('fdgmtkfrxl'):
            targets_f.append((s.x + 1, s.y + 1))

        # Get initial goal positions
        goals = []
        goal_types = []
        for s in g.current_level.get_sprites_by_tag('goal'):
            goals.append((s.x, s.y))
            goal_types.append('goal')
        for s in g.current_level.get_sprites_by_tag('goal-o'):
            goals.append((s.x, s.y))
            goal_types.append('goal-o')

        n_goals = len(goals)
        print(f'  {n_goals} goals, targets_b={targets_b}, targets_f={targets_f}, budget={step_budget}')

        # Find button display coords
        btn_info = []
        for s in g.afhycvvjg:
            if s.tags and 'button' in s.tags[0]:
                parts = s.tags[0].split('_')
                if len(parts) == 3:
                    btn_info.append((parts[1], parts[2], s.x, s.y, s.width, s.height))

        btn_display = {}
        for dy in range(0, 64):
            for dx in range(0, 64):
                grid = g.camera.display_to_grid(dx, dy)
                if grid:
                    gx, gy = grid
                    for map_name, direction, bx, by, bw, bh in btn_info:
                        key = (map_name, direction)
                        if key not in btn_display:
                            if bx <= gx < bx + bw and by <= gy < by + bh:
                                btn_display[key] = encode_click(dx, dy)

        print(f'  {len(btn_display)} buttons')

        # Build permutations: each button moves sprites at cycle positions
        # The permutation applies to ALL sprites, including goals
        # A sprite at position (sx, sy) moves to perm[(sx, sy)]
        btn_perms = {}
        for (map_name, direction), action in btn_display.items():
            is_right = (direction == 'R')
            pairs = chmfaflqhy(level_name, map_name, is_right, uopmnplcnv)
            perm = {}
            for src, dst in pairs:
                sx, sy = src.x * crxpafuiwp, src.y * crxpafuiwp
                dxx, dyy = dst.x * crxpafuiwp, dst.y * crxpafuiwp
                perm[(sx, sy)] = (dxx, dyy)
            btn_perms[action] = perm

        # State = tuple of goal positions
        init_state = tuple(goals)

        def check_win(state):
            # Each target_b must have a 'goal' at that position
            # Each target_f must have a 'goal-o' at that position
            goal_pos = set()
            goal_o_pos = set()
            for i, pos in enumerate(state):
                if goal_types[i] == 'goal':
                    goal_pos.add(pos)
                else:
                    goal_o_pos.add(pos)
            for t in targets_b:
                if t not in goal_pos:
                    return False
            for t in targets_f:
                if t not in goal_o_pos:
                    return False
            return True

        def apply_perm(state, perm):
            new_state = list(state)
            for i in range(len(new_state)):
                if new_state[i] in perm:
                    new_state[i] = perm[new_state[i]]
            return tuple(new_state)

        if check_win(init_state):
            print('  Already won!')
            per_level[f'L{level_idx + 1}'] = {'status': 'SOLVED', 'actions': [], 'length': 0}
            continue

        # BFS
        action_list = list(btn_perms.keys())
        perm_list = [btn_perms[a] for a in action_list]
        action_names = [k for k in btn_display.keys()]

        queue = deque([(init_state, ())])
        visited = {init_state}
        explored = 0
        depth = 0
        solved = False
        max_d = min(step_budget, 80)

        while queue:
            if time.time() - t0 > 300:
                print(f'  TIMEOUT (e={explored}, d={depth})')
                break
            state, seq = queue.popleft()
            if len(seq) > depth:
                depth = len(seq)
                el = time.time() - t0
                rate = explored / max(el, 0.1)
                print(f'    d={depth} v={len(visited)} q={len(queue)} e={explored} t={el:.1f}s ({rate:.0f}/s)')
            if len(seq) >= max_d:
                continue
            for i, (action, perm) in enumerate(zip(action_list, perm_list)):
                new_state = apply_perm(state, perm)
                explored += 1
                if check_win(new_state):
                    new_seq = list(seq) + [action]
                    el = time.time() - t0
                    rate = explored / max(el, 0.1)
                    print(f'  SOLVED in {len(new_seq)} actions ({explored} explored, {el:.1f}s, {rate:.0f}/s)')
                    all_actions.extend(new_seq)
                    per_level[f'L{level_idx + 1}'] = {
                        'status': 'SOLVED', 'actions': list(new_seq),
                        'length': len(new_seq), 'time': round(el, 2),
                    }
                    solved = True
                    break
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, tuple(list(seq) + [action])))
            if solved:
                break
            if explored > 50000000:
                print(f'  STATE LIMIT')
                break

        if not solved:
            el = time.time() - t0
            per_level[f'L{level_idx + 1}'] = {'status': 'UNSOLVED', 'time': round(el, 2)}
            print(f'  UNSOLVED')
            break

    # Verify chain
    print(f'\nVerifying chain ({len(all_actions)} actions)...')
    g = Lp85()
    g.full_reset()
    max_levels = 0
    for a in all_actions:
        levels, state = game_step(g, a)
        if levels > max_levels:
            max_levels = levels
        if state in (GameState.GAME_OVER, GameState.WIN):
            break
    print(f'Chain: {max_levels} levels, {len(all_actions)} actions')

    result = {
        'game': 'lp85', 'version': '305b61c3', 'total_levels': 8,
        'method': 'abstract_permutation_bfs',
        'levels': per_level, 'full_sequence': all_actions,
        'max_level_solved': max_levels, 'total_actions': len(all_actions),
    }
    out_path = f'{RESULTS_DIR}/lp85_fullchain.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    solve()
