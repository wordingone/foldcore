"""DC22 BFS solver v2: proper engine-based BFS with efficient state representation."""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/dc22/4c9bff3e')
from dc22 import Dc22
from arcengine import ActionInput, GameAction, GameState, InteractionMode

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

KB = [1, 2, 3, 4]  # UP, DOWN, LEFT, RIGHT in new engine


def apply_action(g, a):
    if a < 7:
        return g.perform_action(ActionInput(id=a), raw=True)
    ci = a - 7
    return g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64}), raw=True)


def get_state(g):
    """Compact state: player pos + toggle interactions of wbze/jpug sprites."""
    px, py = g.fdvakicpimr.x, g.fdvakicpimr.y
    # Track interaction of toggling sprites
    toggles = tuple(
        s._interaction.value
        for s in g.current_level._sprites
        if any(t in s.tags for t in ['wbze', 'jpug']) and 'path' not in s.tags
    )
    return (px, py, toggles)


def find_unique_clicks(prefix_actions):
    """Find effective click positions by scanning and deduplicating by toggle effect."""
    g_base = Dc22()
    g_base.full_reset()
    for a in prefix_actions:
        apply_action(g_base, a)
    base_toggles = tuple(s._interaction.value for s in g_base.current_level._sprites
                         if any(t in s.tags for t in ['wbze', 'jpug']) and 'path' not in s.tags)

    seen_effects = {}  # effect_tuple -> representative enc
    for dy in range(64):
        for dx in range(64):
            enc = 7 + dy * 64 + dx
            g = Dc22()
            g.full_reset()
            for a in prefix_actions:
                apply_action(g, a)
            apply_action(g, enc)
            new_toggles = tuple(s._interaction.value for s in g.current_level._sprites
                                if any(t in s.tags for t in ['wbze', 'jpug']) and 'path' not in s.tags)
            if new_toggles != base_toggles:
                if new_toggles not in seen_effects:
                    seen_effects[new_toggles] = enc

    unique_clicks = list(seen_effects.values())
    print(f'  Unique clicks: {len(unique_clicks)} (scanned all 64x64)')
    return unique_clicks


def clone_game(prefix_actions, path):
    """Reconstruct game state by replaying prefix + path."""
    g = Dc22()
    g.full_reset()
    for a in prefix_actions:
        apply_action(g, a)
    for a in path:
        apply_action(g, a)
    return g


def solve_level_bfs(level_idx, prefix_actions, budget=150):
    """BFS from current state (after prefix) to complete next level."""
    t0 = time.time()

    # Find unique clicks for this level
    unique_clicks = find_unique_clicks(prefix_actions)
    all_actions = KB + unique_clicks
    print(f'  Actions: {len(KB)} moves + {len(unique_clicks)} unique clicks = {len(all_actions)} total')

    # Initial state
    g_init = clone_game(prefix_actions, [])
    init_state = get_state(g_init)
    init_lvl = sum(1 for _ in range(level_idx))  # We've completed level_idx levels
    goal = (g_init.bqxa.x, g_init.bqxa.y)
    print(f'  Player: ({g_init.fdvakicpimr.x},{g_init.fdvakicpimr.y}), Goal: {goal}, Budget: {budget}')

    # BFS with path stored per state
    visited = {init_state: []}
    queue = deque([(init_state, [])])
    found = None

    while queue:
        state, path = queue.popleft()
        if len(path) >= budget:
            continue

        for a in all_actions:
            # Reconstruct game and apply action
            g = clone_game(prefix_actions, path)
            r = apply_action(g, a)

            # Check win (level completed)
            if r and r.levels_completed > level_idx:
                found = path + [a]
                print(f'  SOLVED! {len(found)} actions, t={time.time()-t0:.1f}s')
                return found

            ns = get_state(g)
            if ns not in visited:
                visited[ns] = path + [a]
                queue.append((ns, path + [a]))

        if len(visited) % 500 == 0 and len(visited) > 0:
            pass  # progress tracking
        if len(visited) > 3000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 120:
            print(f'  Timeout ({len(visited)} states), t={time.time()-t0:.1f}s')
            break

    print(f'  UNSOLVED. States: {len(visited)}, t={time.time()-t0:.1f}s')
    return None


def main():
    try:
        existing = json.load(open(f'{RESULTS_DIR}/dc22_fullchain.json'))
    except Exception:
        existing = {}

    per_level = {}
    full_seq = []

    # Load already-solved levels
    if 'per_level' in existing:
        for k, v in existing.get('per_level', {}).items():
            pass  # we'll re-solve from scratch for correctness

    # Solve level by level
    for level_idx in range(6):
        lname = f'L{level_idx + 1}'
        print(f'\n=== {lname} ===')
        t0 = time.time()

        # Choose budget based on level
        budgets = [50, 80, 120, 150, 150, 150]
        budget = budgets[level_idx]

        sol = solve_level_bfs(level_idx, full_seq, budget=budget)
        elapsed = time.time() - t0

        if sol:
            full_seq.extend(sol)
            per_level[lname] = {
                'status': 'SOLVED', 'actions': sol,
                'length': len(sol), 'time': round(elapsed, 2)
            }
            print(f'  => SOLVED {len(sol)} actions ({elapsed:.1f}s)')
        else:
            per_level[lname] = {'status': 'UNSOLVED', 'time': round(elapsed, 2)}
            print(f'  => UNSOLVED ({elapsed:.1f}s)')
            break

    # Chain verify
    print(f'\nChain verify ({len(full_seq)} actions)...')
    g = Dc22()
    g.full_reset()
    max_lvl = 0
    for a in full_seq:
        r = apply_action(g, a)
        if r and r.levels_completed > max_lvl:
            max_lvl = r.levels_completed
            print(f'  Level {max_lvl} reached')
        if r and r.state in (GameState.WIN, GameState.GAME_OVER):
            print(f'Game ended: {r.state}')
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
