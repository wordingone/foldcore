"""LP85 solver v6: fast abstract BFS with composed multi-button permutations."""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/lp85/305b61c3')
from lp85 import Lp85, izutyjcpih, qfvvosdkqr, chmfaflqhy, crxpafuiwp
from arcengine import ActionInput, GameAction, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def get_ops_for_level(g):
    """Compute (display_enc, composed_move_map) for each unique button position in level."""
    uopmnplcnv = qfvvosdkqr(izutyjcpih)
    level_name = g.ucybisahh

    # Group all button sprites by unique game position
    from collections import defaultdict
    btn_by_pos = defaultdict(list)
    for s in g.current_level.get_sprites():
        if s.tags and s.tags[0].startswith('button'):
            parts = s.tags[0].split('_')
            if len(parts) == 3:
                btn_name, direction = parts[1], parts[2]
                is_r = (direction == 'R')
                btn_by_pos[(s.x, s.y)].append((btn_name, is_r))

    # For each button position, compute composed permutation
    # Map: game_pos -> {old_game_xy: new_game_xy}
    composed_ops = {}  # game_pos -> move_map
    for (gx, gy), btns in btn_by_pos.items():
        # Compose all button perms at this position
        # Apply each one as delta: old_pos -> intermediate -> ... -> final_pos
        # Since each perm is a bijection, we compose them sequentially
        # Build combined move map for all cycle positions
        full_cycle = {}  # pos -> pos (all positions in any of the cycles)
        for btn_name, is_r in btns:
            perms = chmfaflqhy(level_name, btn_name, is_r, uopmnplcnv)
            for p, q in perms:
                old = (p.x * crxpafuiwp, p.y * crxpafuiwp)
                new = (q.x * crxpafuiwp, q.y * crxpafuiwp)
                full_cycle[old] = new
        if full_cycle:
            composed_ops[(gx, gy)] = full_cycle

    # For each button game pos, find a display encoding
    # pubeyzotzr(gx, gy) covers bounding box of sprites — scan display grid
    # We know button sprite size is typically 3-4 pixels
    # Use camera.display_to_grid to find display positions that map to each button pos
    ops = []
    found_game_pos = set()
    for dy in range(64):
        for dx in range(64):
            gpos = g.camera.display_to_grid(dx, dy)
            if gpos is None:
                continue
            # Check if this game pos is inside any button bounding box
            for (bx, by), move_map in composed_ops.items():
                if bx <= gpos[0] < bx + 4 and by <= gpos[1] < by + 4:  # approx bbox
                    if (bx, by) not in found_game_pos:
                        found_game_pos.add((bx, by))
                        enc = 7 + dy * 64 + dx
                        ops.append((enc, move_map))
            if len(found_game_pos) == len(composed_ops):
                break
        if len(found_game_pos) == len(composed_ops):
            break

    return ops


def get_state(g):
    goals = tuple(sorted((s.x, s.y) for s in g.current_level.get_sprites_by_tag('goal')))
    goals_o = tuple(sorted((s.x, s.y) for s in g.current_level.get_sprites_by_tag('goal-o')))
    return goals + goals_o


def game_step(game, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    r = game.perform_action(ai, raw=True)
    if r is None:
        return 0, None
    return r.levels_completed, r.state


def solve_level(level_idx):
    g_init = Lp85()
    g_init.full_reset()
    if level_idx > 0:
        g_init.set_level(level_idx)

    n_b = len(g_init.current_level.get_sprites_by_tag('goal'))
    targets_b = frozenset((s.x + 1, s.y + 1) for s in g_init.current_level.get_sprites_by_tag('bghvgbtwcb'))
    targets_f = frozenset((s.x + 1, s.y + 1) for s in g_init.current_level.get_sprites_by_tag('fdgmtkfrxl'))
    budget = min(g_init.toxpunyqe.current_steps, 200)
    init_state = get_state(g_init)
    print(f'  L{level_idx+1}: n_b={n_b}, targets_b={sorted(targets_b)}, targets_f={sorted(targets_f)}, budget={budget}')
    print(f'  Init state: {init_state}')

    def check_win(state):
        goals_b = frozenset(state[:n_b])
        goals_f = frozenset(state[n_b:])
        return targets_b <= goals_b and targets_f <= goals_f

    if check_win(init_state):
        return []

    t0 = time.time()

    # Get abstract operations
    ops = get_ops_for_level(g_init)
    print(f'  Ops: {len(ops)} button positions, t={time.time()-t0:.1f}s')

    if not ops:
        print('  No ops found!')
        return None

    def apply_op(state, move_map):
        """Apply composed permutation to goals in state."""
        goals_b = list(state[:n_b])
        goals_f = list(state[n_b:])
        changed = False
        for i, pos in enumerate(goals_b):
            if pos in move_map:
                goals_b[i] = move_map[pos]
                changed = True
        for i, pos in enumerate(goals_f):
            if pos in move_map:
                goals_f[i] = move_map[pos]
                changed = True
        if not changed:
            return state
        return tuple(sorted(goals_b)) + tuple(sorted(goals_f))

    # BFS
    visited = {init_state: []}
    queue = deque([(init_state, [])])
    found = None

    while queue:
        state, path = queue.popleft()
        if len(path) >= budget:
            continue

        for enc, move_map in ops:
            new_s = apply_op(state, move_map)
            if new_s == state:
                continue

            if check_win(new_s):
                found = path + [enc]
                print(f'  SOLVED! {len(found)} actions, t={time.time()-t0:.1f}s')
                break

            if new_s not in visited:
                visited[new_s] = path + [enc]
                queue.append((new_s, path + [enc]))

        if found:
            break
        if len(visited) > 500000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 120:
            print(f'  Timeout ({len(visited)} states), t={time.time()-t0:.1f}s')
            break

    if not found:
        print(f'  UNSOLVED. States: {len(visited)}, t={time.time()-t0:.1f}s')
    return found


def verify_solution(level_idx, actions):
    """Verify solution works with engine."""
    g = Lp85()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    for a in actions:
        lev, state = game_step(g, a)
        if lev > 0:
            return True
        if state == GameState.GAME_OVER:
            return False
    return False


def main():
    existing = json.load(open(f'{RESULTS_DIR}/lp85_fullchain.json'))
    per_level = {}
    all_actions = []
    for k, v in existing.get('levels', {}).items():
        if isinstance(v, dict) and v.get('actions'):
            per_level[k] = v
            all_actions.extend(v['actions'])
    print(f'Loaded {len(per_level)} solved levels, {len(all_actions)} actions')

    for level_idx in range(8):
        lname = f'L{level_idx+1}'
        if lname in per_level:
            print(f'{lname}: skip (solved)')
            continue
        print(f'\n=== {lname} ===')
        t0 = time.time()
        sol = solve_level(level_idx)
        elapsed = time.time() - t0
        if sol is not None:
            # Verify
            ok = verify_solution(level_idx, sol)
            print(f'  Engine verify: {ok}')
            if ok:
                all_actions.extend(sol)
                per_level[lname] = {'status': 'SOLVED', 'actions': sol, 'length': len(sol), 'time': round(elapsed, 2)}
                print(f'  => SOLVED in {len(sol)} actions ({elapsed:.1f}s)')
            else:
                print(f'  => VERIFY FAILED, skipping')
                per_level[lname] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
                break
        else:
            per_level[lname] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            print(f'  => UNSOLVED ({elapsed:.1f}s)')
            break

    # Chain verify
    print(f'\nChain verify ({len(all_actions)} total actions)...')
    g = Lp85()
    g.full_reset()
    max_lvl = 0
    for a in all_actions:
        lev, state = game_step(g, a)
        if lev > max_lvl:
            max_lvl = lev
        if state in (GameState.WIN, GameState.GAME_OVER):
            print(f'Game ended: {state}')
            break
    print(f'Chain result: {max_lvl} levels completed')

    result = {
        'game': 'lp85', 'total_levels': 8, 'method': 'abstract_bfs_v6_composed',
        'levels': per_level, 'full_sequence': all_actions,
        'max_level_solved': max_lvl, 'total_actions': len(all_actions),
    }
    with open(f'{RESULTS_DIR}/lp85_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {len(per_level)} levels, max_chain={max_lvl}')


if __name__ == '__main__':
    main()
