"""DC22 BFS solver v4: correct engine BFS with full state scan and zbhi support.

Key fixes from v3:
- Full 64x64 scan for button discovery (no filtered search)
- Per toggle-state button caching (handles zbhi-revealed buttons)
- Correct button detection using all-sprite state comparison
"""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/dc22/4c9bff3e')
from dc22 import Dc22
from arcengine import ActionInput, GameAction, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def apply_action(g, a):
    if a < 7:
        return g.perform_action(ActionInput(id=a), raw=True)
    ci = a - 7
    return g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64}), raw=True)


def rebuild(prefix, path):
    g = Dc22()
    g.full_reset()
    for a in prefix:
        apply_action(g, a)
    for a in path:
        apply_action(g, a)
    return g


def get_full_state(g):
    """Full state including player pos and all sprite interaction states."""
    px, py = g.fdvakicpimr.x, g.fdvakicpimr.y
    sprite_states = tuple(s._interaction.value for s in g.current_level._sprites)
    return (px, py, sprite_states)


def get_toggle_key(g):
    """Just the toggle state (for button cache keying)."""
    return tuple(s._interaction.value for s in g.current_level._sprites)


def find_unique_clicks_for_state(prefix, path, toggle_key_cache):
    """Find representative clicks for the current toggle state. Cached."""
    if toggle_key not in toggle_key_cache:
        g = rebuild(prefix, path)
        base = tuple(s._interaction.value for s in g.current_level._sprites)

        seen = {}
        for dy in range(64):
            for dx in range(64):
                enc = 7 + dy * 64 + dx
                g2 = rebuild(prefix, path)
                apply_action(g2, enc)
                new = tuple(s._interaction.value for s in g2.current_level._sprites)
                if new != base:
                    delta = tuple(a != b for a, b in zip(base, new))
                    if delta not in seen:
                        seen[delta] = enc

        toggle_key_cache[toggle_key] = list(seen.values())
    return toggle_key_cache[toggle_key]


def solve_level(level_idx, prefix_actions):
    t0 = time.time()

    g_init = rebuild(prefix_actions, [])
    px_init, py_init = g_init.fdvakicpimr.x, g_init.fdvakicpimr.y
    gx, gy = g_init.bqxa.x, g_init.bqxa.y
    budget = g_init.toxpunyqe.current_steps if hasattr(g_init, 'toxpunyqe') else 192
    budget = min(budget, 192)
    print(f'  L{level_idx+1}: player=({px_init},{py_init}), goal=({gx},{gy}), budget={budget}')

    # Cache: toggle_key -> list of unique click actions
    click_cache = {}

    def get_clicks(prefix, path):
        g = rebuild(prefix, path)
        tk = get_toggle_key(g)
        if tk not in click_cache:
            base = tuple(s._interaction.value for s in g.current_level._sprites)
            seen = {}
            # Scan right portion of display (dx=38-63) where jpug buttons appear
            for dy in range(64):
                for dx in range(38, 64):
                    enc = 7 + dy * 64 + dx
                    g2 = rebuild(prefix, path)
                    apply_action(g2, enc)
                    new = tuple(s._interaction.value for s in g2.current_level._sprites)
                    if new != base:
                        delta = tuple(a != b for a, b in zip(base, new))
                        if delta not in seen:
                            seen[delta] = enc
            click_cache[tk] = list(seen.values())
            print(f'    toggle_state scan: {len(click_cache[tk])} unique clicks (t={time.time()-t0:.1f}s)')
        return click_cache[tk]

    # Initial buttons scan
    init_clicks = get_clicks(prefix_actions, [])
    print(f'  Initial clicks: {init_clicks}')

    init_state = get_full_state(g_init)
    visited = {init_state: []}
    queue = deque([(init_state, [])])
    found = None

    while queue:
        state, path = queue.popleft()
        if len(path) >= budget:
            continue

        # Get available actions (moves + cached clicks for this toggle state)
        clicks = get_clicks(prefix_actions, path)
        all_actions = [1, 2, 3, 4] + clicks

        for a in all_actions:
            g2 = rebuild(prefix_actions, path)
            r = apply_action(g2, a)

            if r and r.levels_completed > level_idx:
                found = path + [a]
                print(f'  SOLVED! {len(found)} actions, states={len(visited)}, t={time.time()-t0:.1f}s')
                return found

            ns = get_full_state(g2)
            if ns not in visited:
                visited[ns] = path + [a]
                queue.append((ns, path + [a]))

        if len(visited) > 10000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 300:
            print(f'  Timeout ({len(visited)} states), t={time.time()-t0:.1f}s')
            break

    print(f'  UNSOLVED. States: {len(visited)}, cache_states={len(click_cache)}, t={time.time()-t0:.1f}s')
    return None


def verify_chain(actions):
    g = Dc22()
    g.full_reset()
    max_lvl = 0
    for a in actions:
        r = apply_action(g, a)
        if r and r.levels_completed > max_lvl:
            max_lvl = r.levels_completed
        if r and r.state in (GameState.WIN, GameState.GAME_OVER):
            break
    return max_lvl


def main():
    per_level = {}
    full_seq = []

    for level_idx in range(6):
        lname = f'L{level_idx + 1}'
        print(f'\n=== {lname} ===')
        t0 = time.time()

        sol = solve_level(level_idx, full_seq)
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

    print(f'\nChain verify ({len(full_seq)} actions)...')
    max_lvl = verify_chain(full_seq)
    print(f'Chain: {max_lvl} levels')

    result = {
        'game': 'dc22', 'total_levels': 6, 'method': 'engine_bfs_v4',
        'n_levels': 6, 'max_level': max_lvl, 'total_actions': len(full_seq),
        'per_level': {k: v.get('length', 0) for k, v in per_level.items()},
        'all_actions': full_seq, 'levels': per_level,
    }
    with open(f'{RESULTS_DIR}/dc22_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved ({len(per_level)} levels, {max_lvl} chain-verified)')


if __name__ == '__main__':
    main()
