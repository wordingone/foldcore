"""DC22 fresh BFS solver - engine-based, no assumptions about mechanics."""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/dc22/4c9bff3e')
from dc22 import Dc22
from arcengine import ActionInput, GameAction, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'
KB = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT

def apply_action(g, a):
    if a < 7:
        return g.perform_action(ActionInput(id=a), raw=True)
    ci = a - 7
    return g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': ci%64, 'y': ci//64}), raw=True)

def get_state(g):
    """Get hashable game state."""
    # Player pos + all toggleable sprite states
    player_x = g.fdvakicpimr.x
    player_y = g.fdvakicpimr.y
    # Get visible/invisible sprites (buttons toggle walls)
    sprite_states = tuple(sorted(
        (s.x, s.y, s.visible) for s in g.current_level._sprites
        if hasattr(s, 'visible') and s.tags and any(t in ['wbze', 'bgeg', 'efzv', 'jbyz', 'jpug'] for t in s.tags)
    ))
    return (player_x, player_y, sprite_states)

def solve_level_bfs(level_idx, full_seq_prefix):
    """Solve level using BFS with engine simulation."""
    g = Dc22()
    g.full_reset()
    # Replay prefix to get to start of this level
    for a in full_seq_prefix:
        apply_action(g, a)

    n_levels_before = 0
    # Count completed levels from prefix
    g2 = Dc22()
    g2.full_reset()
    for a in full_seq_prefix:
        r = apply_action(g2, a)
        if r:
            n_levels_before = max(n_levels_before, r.levels_completed)

    init_state = get_state(g)
    budget = 150  # max actions per level

    print(f'  L{level_idx+1}: n_levels_before={n_levels_before}, budget={budget}')
    print(f'  Player start: ({g.fdvakicpimr.x},{g.fdvakicpimr.y})')

    # Find clickable positions (buttons)
    click_positions = []
    for s in g.current_level._sprites:
        if s.tags and any('button' in str(t) or t in ['jpug', 'bqxa'] for t in s.tags):
            # Click on this sprite's center
            # Convert game pos to display pos via camera
            try:
                dp = g.camera.grid_to_display(s.x, s.y)
                if dp and 0 <= dp[0] <= 63 and 0 <= dp[1] <= 63:
                    click_positions.append(7 + dp[1]*64 + dp[0])
            except:
                pass

    # Also try all display positions (dense sampling for finding buttons)
    found_clicks = set()
    for dy in range(0, 64, 2):
        for dx in range(0, 64, 2):
            g_test = Dc22()
            g_test.full_reset()
            for a in full_seq_prefix:
                apply_action(g_test, a)
            r = apply_action(g_test, 7 + dy*64 + dx)
            ns = get_state(g_test)
            if ns != init_state:
                enc = 7 + dy*64 + dx
                if enc not in found_clicks:
                    found_clicks.add(enc)

    print(f'  Found {len(found_clicks)} clickable positions')

    actions = KB + list(found_clicks)

    # BFS
    t0 = time.time()
    visited = {init_state: []}
    queue = deque([(init_state, [], g)])
    found = None

    while queue:
        state, path, _ = queue.popleft()
        if len(path) >= budget:
            continue

        for a in actions:
            # Reconstruct game state
            g_new = Dc22()
            g_new.full_reset()
            for pa in full_seq_prefix:
                apply_action(g_new, pa)
            for pa in path:
                apply_action(g_new, pa)

            r = apply_action(g_new, a)

            if r and r.levels_completed > n_levels_before:
                found = path + [a]
                print(f'  SOLVED! {len(found)} actions, lvl {n_levels_before}->{r.levels_completed}, t={time.time()-t0:.1f}s')
                break

            ns = get_state(g_new)
            if ns not in visited:
                visited[ns] = path + [a]
                queue.append((ns, path + [a], g_new))

        if found:
            break
        if len(visited) > 5000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 60:
            print(f'  Timeout, t={time.time()-t0:.1f}s')
            break

    if not found:
        print(f'  UNSOLVED. States: {len(visited)}, t={time.time()-t0:.1f}s')

    return found


def main():
    # Get current dc22 fullchain
    try:
        existing = json.load(open(f'{RESULTS_DIR}/dc22_fullchain.json'))
    except:
        existing = {}

    per_level = {}
    full_seq = []

    # Try solving level by level
    for level_idx in range(6):
        lname = f'L{level_idx+1}'
        print(f'\n=== {lname} ===')
        t0 = time.time()
        sol = solve_level_bfs(level_idx, full_seq)
        if sol:
            full_seq.extend(sol)
            per_level[lname] = {'status': 'SOLVED', 'actions': sol, 'length': len(sol), 'time': round(time.time()-t0, 2)}
            print(f'  => SOLVED {len(sol)} actions')
        else:
            per_level[lname] = {'status': 'UNSOLVED', 'time': round(time.time()-t0, 2)}
            print(f'  => UNSOLVED')
            break

    # Verify chain
    print(f'\nChain verify ({len(full_seq)} actions)...')
    g = Dc22()
    g.full_reset()
    max_lvl = 0
    for a in full_seq:
        r = apply_action(g, a)
        if r and r.levels_completed > max_lvl:
            max_lvl = r.levels_completed
            print(f'  Level {max_lvl} reached')
        if r and r.state == GameState.WIN:
            print('WIN!')
            break
    print(f'Chain: {max_lvl} levels')

    result = {
        'game': 'dc22', 'total_levels': 6, 'method': 'engine_bfs_v2',
        'n_levels': 6, 'max_level': max_lvl, 'total_actions': len(full_seq),
        'per_level': {k: v.get('length', 0) for k, v in per_level.items()},
        'all_actions': full_seq, 'levels': per_level,
    }
    with open(f'{RESULTS_DIR}/dc22_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved ({len(per_level)} levels, {max_lvl} chain-verified)')


if __name__ == '__main__':
    main()
