"""LP85 solver v5: cycle-trace BFS with engine simulation."""
import sys, time, json
from collections import deque
sys.path.insert(0, 'B:/M/the-search/environment_files/lp85/305b61c3')
from lp85 import Lp85
from arcengine import ActionInput, GameAction, GameState

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'

def game_step(game, action):
    ci = action - 7
    ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})
    r = game.perform_action(ai, raw=True)
    if r is None:
        return 0, None
    return r.levels_completed, r.state

def encode_click(x, y):
    return 7 + y * 64 + x

def get_state(g):
    b = tuple(sorted((s.x, s.y) for s in g.current_level.get_sprites_by_tag('goal')))
    o = tuple(sorted((s.x, s.y) for s in g.current_level.get_sprites_by_tag('goal-o')))
    return b + o

def replay_to_state(level_idx, path):
    g = Lp85()
    g.full_reset()
    if level_idx > 0:
        g.set_level(level_idx)
    for a in path:
        ci = a - 7
        g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64}))
    return g

def map_transitions(level_idx, path, state):
    """Map all button effects from given state (via replay)."""
    trans = []
    for dy in range(64):
        for dx in range(64):
            g = replay_to_state(level_idx, path)
            g.perform_action(ActionInput(id=GameAction.ACTION6, data={'x': dx, 'y': dy}))
            new_s = get_state(g)
            if new_s != state:
                enc = encode_click(dx, dy)
                trans.append((enc, new_s))
    # Deduplicate by (enc, new_s) - keep first occurrence per unique new_s
    seen = set()
    deduped = []
    for enc, new_s in trans:
        if new_s not in seen:
            seen.add(new_s)
            deduped.append((enc, new_s))
    return deduped

def solve_level(level_idx):
    g_init = Lp85()
    g_init.full_reset()
    if level_idx > 0:
        g_init.set_level(level_idx)
    n_b = len(g_init.current_level.get_sprites_by_tag('goal'))
    targets_b = frozenset((s.x+1, s.y+1) for s in g_init.current_level.get_sprites_by_tag('bghvgbtwcb'))
    targets_f = frozenset((s.x+1, s.y+1) for s in g_init.current_level.get_sprites_by_tag('fdgmtkfrxl'))
    budget = min(g_init.toxpunyqe.current_steps, 80)
    init_state = get_state(g_init)
    print(f'  L{level_idx+1}: init={init_state}, targets_b={sorted(targets_b)}, budget={budget}')

    def check_win(state):
        return targets_b <= frozenset(state[:n_b]) and targets_f <= frozenset(state[n_b:])

    if check_win(init_state):
        return []

    t0 = time.time()

    # Map transitions from initial state
    init_trans = map_transitions(level_idx, [], init_state)
    print(f'  Init: {len(init_trans)} transitions, t={time.time()-t0:.1f}s')

    # Build perm cache: each button always applies same delta to same position
    known_perms = {}  # enc -> {pos -> new_pos}
    for enc, new_s in init_trans:
        perm = {}
        for i, (old_p, new_p) in enumerate(zip(init_state, new_s)):
            if old_p != new_p:
                perm[old_p] = new_p
        if perm:
            if enc not in known_perms:
                known_perms[enc] = {}
            known_perms[enc].update(perm)

    # BFS: track state -> shortest path
    visited = {init_state: []}
    bfs_q = deque([(init_state, [])])
    found = None

    def apply_known_perm(state, enc):
        perm = known_perms.get(enc, {})
        if not perm:
            return state
        new_s = list(state)
        changed = False
        for i in range(len(new_s)):
            if new_s[i] in perm:
                new_s[i] = perm[new_s[i]]
                changed = True
        if not changed:
            return state
        return tuple(sorted(new_s[:n_b])) + tuple(sorted(new_s[n_b:]))

    max_replay = 30
    replays_done = 0
    replay_paths = {}  # state -> path (for replaying)
    replay_paths[init_state] = []

    while bfs_q:
        state, path = bfs_q.popleft()
        if len(path) >= budget:
            continue

        # Get transitions for this state
        if state == init_state:
            state_trans = init_trans
        else:
            # Apply known perms
            state_trans = []
            for enc, perm in known_perms.items():
                new_s = apply_known_perm(state, enc)
                if new_s != state:
                    state_trans.append((enc, new_s))

            # If we have a path and haven't done too many replays, verify/expand
            if replays_done < max_replay and len(path) <= 6 and state in replay_paths:
                actual_trans = map_transitions(level_idx, replay_paths[state], state)
                replays_done += 1
                # Add newly discovered transitions and update perm cache
                known_enc = set(enc for enc, _ in state_trans)
                for enc, new_s in actual_trans:
                    if new_s != state:
                        perm_new = {state[i]: new_s[i] for i in range(len(state)) if state[i] != new_s[i]}
                        if enc not in known_perms:
                            known_perms[enc] = {}
                        known_perms[enc].update(perm_new)
                        if enc not in known_enc:
                            state_trans.append((enc, new_s))
                if replays_done % 5 == 0:
                    print(f'  Replay {replays_done}: depth={len(path)}, perms={len(known_perms)}, states={len(visited)}, t={time.time()-t0:.1f}s')

        for enc, new_s in state_trans:
            if check_win(new_s):
                found = path + [enc]
                print(f'  SOLVED! {len(found)} actions, t={time.time()-t0:.1f}s')
                break
            if new_s not in visited:
                visited[new_s] = path + [enc]
                bfs_q.append((new_s, path + [enc]))
                if new_s not in replay_paths:
                    replay_paths[new_s] = path + [enc]

        if found:
            break
        if len(visited) > 100000:
            print(f'  State limit ({len(visited)}), t={time.time()-t0:.1f}s')
            break
        if time.time() - t0 > 120:
            print(f'  Timeout, t={time.time()-t0:.1f}s')
            break

    if not found:
        print(f'  UNSOLVED. States: {len(visited)}, t={time.time()-t0:.1f}s')
    return found


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
            all_actions.extend(sol)
            per_level[lname] = {'status': 'SOLVED', 'actions': sol, 'length': len(sol), 'time': round(elapsed, 2)}
            print(f'  => SOLVED in {len(sol)} actions ({elapsed:.1f}s)')
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
        'game': 'lp85', 'total_levels': 8, 'method': 'cycle_trace_bfs_v5',
        'levels': per_level, 'full_sequence': all_actions,
        'max_level_solved': max_lvl, 'total_actions': len(all_actions),
    }
    with open(f'{RESULTS_DIR}/lp85_fullchain.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {len(per_level)} levels')


if __name__ == '__main__':
    main()
