"""DC22 BFS solver v3: engine BFS with compact state representation.

State = (player_x, player_y, toggle_key).
One reconstruction per (state, action) pair. Fast click discovery.
"""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/dc22/4c9bff3e')
from dc22 import Dc22
from arcengine import ActionInput, GameAction, GameState, InteractionMode

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def apply_action(g, a):
    if a < 7:
        return g.perform_action(ActionInput(id=a), raw=True)
    ci = a - 7
    return g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64}), raw=True)


def rebuild(prefix_actions, path):
    g = Dc22()
    g.full_reset()
    for a in prefix_actions:
        apply_action(g, a)
    for a in path:
        apply_action(g, a)
    return g


def get_state(g):
    px, py = g.fdvakicpimr.x, g.fdvakicpimr.y
    toggles = tuple(
        s._interaction.value
        for s in g.current_level._sprites
        if any(t in s.tags for t in ['wbze', 'jpug', 'ordebgeg', 'gigzqgcfncq'])
    )
    return (px, py, toggles)


def find_unique_clicks(prefix_actions):
    """Find representative clicks by first scanning near jpug sprites, then full scan if needed."""
    g0 = rebuild(prefix_actions, [])

    def get_toggles(g):
        return tuple(
            s._interaction.value for s in g.current_level._sprites
            if any(t in s.tags for t in ['wbze', 'jpug', 'ordebgeg', 'gigzqgcfncq'])
        )

    base_toggles = get_toggles(g0)

    # Collect candidate display positions: scan all display pos and filter near jpug/zbhi game pos
    # Build set of jpug/zbhi game positions
    button_game_pos = set()
    for s in g0.current_level._sprites:
        if 'jpug' in s.tags or 'zbhi' in s.tags:
            pixels = s.render()
            h = len(pixels)
            w = len(pixels[0]) if h > 0 else 0
            for dy in range(h):
                for dx in range(w):
                    if h > 0 and w > 0 and pixels[dy][dx] >= 0:
                        button_game_pos.add((s.x + dx, s.y + dy))

    candidates = set()
    for ddy in range(64):
        for ddx in range(64):
            gpos = g0.camera.display_to_grid(ddx, ddy)
            if gpos and gpos in button_game_pos:
                candidates.add(7 + ddy * 64 + ddx)

    # Also add a dense neighborhood around any found candidates
    if not candidates:
        # Fallback: scan all
        candidates = {7 + dy * 64 + dx for dy in range(64) for dx in range(64)}
    else:
        extra = set()
        for enc in list(candidates):
            ci = enc - 7
            dx, dy = ci % 64, ci // 64
            for ex in range(-3, 4):
                for ey in range(-3, 4):
                    if 0 <= dx + ex < 64 and 0 <= dy + ey < 64:
                        extra.add(7 + (dy + ey) * 64 + (dx + ex))
        candidates |= extra

    seen = {}  # delta_tuple -> enc
    for enc in sorted(candidates):
        g = rebuild(prefix_actions, [])
        apply_action(g, enc)
        new_toggles = get_toggles(g)
        if new_toggles != base_toggles:
            delta = tuple(a != b for a, b in zip(base_toggles, new_toggles))
            if delta not in seen:
                seen[delta] = enc

    print(f'  Clicks: tested {len(candidates)} candidates, found {len(seen)} unique effects')
    return list(seen.values())


def solve_level(level_idx, prefix_actions):
    t0 = time.time()

    g_init = rebuild(prefix_actions, [])
    px_init, py_init = g_init.fdvakicpimr.x, g_init.fdvakicpimr.y
    gx, gy = g_init.bqxa.x, g_init.bqxa.y
    budget = g_init.toxpunyqe.current_steps if hasattr(g_init, 'toxpunyqe') else 192
    budget = min(budget, 192)
    print(f'  player=({px_init},{py_init}), goal=({gx},{gy}), budget={budget}')

    unique_clicks = find_unique_clicks(prefix_actions)
    all_actions = [1, 2, 3, 4] + unique_clicks
    print(f'  {len(unique_clicks)} clicks, {len(all_actions)} total actions')

    init_state = get_state(g_init)
    visited = {init_state: []}
    queue = deque([(init_state, [])])
    found = None

    while queue:
        state, path = queue.popleft()
        if len(path) >= budget:
            continue

        for a in all_actions:
            g2 = rebuild(prefix_actions, path)
            r = apply_action(g2, a)

            if r and r.levels_completed > level_idx:
                found = path + [a]
                print(f'  SOLVED! {len(found)} actions, states={len(visited)}, t={time.time()-t0:.1f}s')
                return found

            ns = get_state(g2)
            if ns not in visited:
                visited[ns] = path + [a]
                queue.append((ns, path + [a]))

        if len(visited) > 10000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 120:
            print(f'  Timeout ({len(visited)} states), t={time.time()-t0:.1f}s')
            break

    print(f'  UNSOLVED. States: {len(visited)}, t={time.time()-t0:.1f}s')
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
        'game': 'dc22', 'total_levels': 6, 'method': 'engine_bfs_v3',
        'n_levels': 6, 'max_level': max_lvl, 'total_actions': len(full_seq),
        'per_level': {k: v.get('length', 0) for k, v in per_level.items()},
        'all_actions': full_seq, 'levels': per_level,
    }
    with open(f'{RESULTS_DIR}/dc22_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved ({len(per_level)} levels, {max_lvl} chain-verified)')


if __name__ == '__main__':
    main()
