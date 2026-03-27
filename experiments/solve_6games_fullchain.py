"""
Full-chain analytical solver for 6 games: re86, tr87, sk48, ar25, tn36, lp85.
Reads game source to understand mechanics, then computes optimal solutions.
"""
import sys, json, time, os, copy, copyreg, hashlib, random
import numpy as np
from collections import deque
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState, ActionInput

# Fix deepcopy for GameAction
def pickle_gameaction(ga):
    return (GameAction.__getitem__, (ga.name,))
copyreg.pickle(GameAction, pickle_gameaction)

RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def frame_hash(obs):
    f = np.array(obs.frame, dtype=np.uint8)
    return hashlib.md5(f.tobytes()).hexdigest()


def make_arc():
    """Create an Arcade instance."""
    return arc_agi.Arcade()


def get_game_info(arc_inst, game_key):
    games = arc_inst.get_environments()
    return next(g for g in games if game_key in g.game_id.lower())


def verify_chain(arc_inst, game_id, raw_actions):
    """Verify a full chain of actions. Returns (success, obs, state)."""
    env = arc_inst.make(game_id)
    obs = env.reset()
    for act in raw_actions:
        if isinstance(act, tuple):
            obs = env.step(act[0], data=act[1])
        else:
            obs = env.step(act)
        if obs is None:
            return False, None, None
        if obs.state == GameState.WIN:
            return True, obs, 'WIN'
    return obs.state == GameState.WIN, obs, obs.state.name if obs else None


def serialize_actions(raw_actions):
    """Convert action list to serializable format."""
    result = []
    for act in raw_actions:
        if isinstance(act, tuple):
            ga, data = act
            result.append(data['y'] * 64 + data['x'])
        else:
            result.append(act.value - 1)  # 0-indexed KB
    return result


def save_result(game_key, result):
    """Save fullchain result to JSON."""
    out_path = os.path.join(RESULTS_DIR, f'{game_key}_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'  Saved: {out_path}')
    return out_path


# ============================================================
# TR87 SOLVER — Pattern matching puzzle
# ============================================================
def solve_tr87(arc_inst, verbose=True):
    """
    TR87: pattern matching. Cycle adjustable sprites to match targets via rules.
    ACTION3/4 = move cursor left/right
    ACTION1/2 = cycle sprite type backward/forward (mod 7)
    """
    info = get_game_info(arc_inst, 'tr87')
    env = arc_inst.make(info.game_id)
    obs = env.reset()
    game = env._game

    all_actions = []
    per_level = {}
    n_levels = len(info.baseline_actions)

    if verbose:
        print(f'\n{"="*70}')
        print(f'SOLVING TR87 ({n_levels} levels, baseline={info.baseline_actions})')
        print(f'{"="*70}')

    for level_idx in range(n_levels):
        if verbose:
            print(f'\n--- Level {level_idx + 1}/{n_levels} ---')

        # Access game state
        game = env._game
        alter_rules = game.current_level.get_data("alter_rules")
        tree_trans = game.current_level.get_data("tree_translation")

        if verbose:
            print(f'  alter_rules={alter_rules}, tree_translation={tree_trans}')

        if not alter_rules:
            # Simple mode: cycle each ztgmtnnufb sprite to match rule target
            level_actions = solve_tr87_simple_level(game, verbose)
        else:
            # Complex mode: alter_rules changes how groups work
            level_actions = solve_tr87_alter_level(game, verbose)

        if level_actions is None:
            if verbose:
                print(f'  FAILED to solve analytically, trying BFS...')
            level_actions = solve_tr87_bfs(env, all_actions, verbose)

        if level_actions is None:
            if verbose:
                print(f'  FAILED level {level_idx + 1}')
            break

        if verbose:
            print(f'  Solution: {len(level_actions)} actions')

        # Execute actions
        for act in level_actions:
            obs = env.step(act)

        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs.state == GameState.WIN:
            if verbose:
                print(f'\n  GAME COMPLETE after level {level_idx + 1}!')
            break

    return build_result('tr87', all_actions, per_level, info)


def solve_tr87_simple_level(game, verbose=True):
    """Solve a simple tr87 level (no alter_rules)."""
    target_row = game.zvojhrjxxm
    adjust_row = game.ztgmtnnufb
    rules = game.cifzvbcuwqe
    cursor_idx = game.qvtymdcqear_index

    # Build rule lookup: left_name -> right_names (handle multi-sprite rules)
    # Rules may chain: A->B, B->C means target A ultimately needs adjust C
    rule_map = {}  # left_name_tuple -> right_name_list
    for left, right in rules:
        key = tuple(s.name for s in left)
        val = [s.name for s in right]
        rule_map[key] = val

    # Follow rule chains until we get to adjust row names
    # adjust row sprites have names like C*, B*, etc.
    adjust_prefixes = set()
    for s in adjust_row:
        # Get prefix (everything except last digit)
        adjust_prefixes.add(s.name[:-1])

    def resolve_chain(target_names):
        """Follow rules to get final adjust names."""
        current = target_names
        for _ in range(10):  # max chain depth
            # Check if current names match adjust row prefix
            if current and current[0][:-1] in adjust_prefixes:
                return current

            # Try to resolve through rules
            resolved = []
            i = 0
            any_resolved = False
            while i < len(current):
                matched = False
                for rule_len in range(min(len(current) - i, 5), 0, -1):
                    key = tuple(current[i:i+rule_len])
                    if key in rule_map:
                        resolved.extend(rule_map[key])
                        i += rule_len
                        matched = True
                        any_resolved = True
                        break
                if not matched:
                    resolved.append(current[i])
                    i += 1

            if not any_resolved:
                return resolved  # Can't resolve further
            current = resolved

        return current

    # Build needed names by matching target row through rules
    needed_names = []
    t_idx = 0
    while t_idx < len(target_row):
        matched = False
        # Try matching multiple target sprites to a rule
        for match_len in range(min(len(target_row) - t_idx, 5), 0, -1):
            target_chunk = [target_row[t_idx + k].name for k in range(match_len)]
            resolved = resolve_chain(target_chunk)
            # Check if resolved names match adjust prefix
            if resolved and resolved[0][:-1] in adjust_prefixes:
                needed_names.extend(resolved)
                t_idx += match_len
                matched = True
                break
        if not matched:
            if verbose:
                print(f'  No rule chain for target at position {t_idx}: {target_row[t_idx].name}')
            return None

    if len(needed_names) != len(adjust_row):
        if verbose:
            print(f'  Mismatch: {len(needed_names)} needed vs {len(adjust_row)} adjust sprites')
        return None

    if verbose:
        for i in range(len(adjust_row)):
            print(f'  Pos {i}: have={adjust_row[i].name[-2:]}, need={needed_names[i][-2:]}')

    # Build action sequence
    actions = []
    for i in range(len(adjust_row)):
        # Move cursor to position i
        while cursor_idx < i:
            actions.append(GameAction.ACTION4)
            cursor_idx += 1
        while cursor_idx > i:
            actions.append(GameAction.ACTION3)
            cursor_idx -= 1

        # Cycle sprite to match
        current_num = int(adjust_row[i].name[-1])
        needed_num = int(needed_names[i][-1])

        if current_num == needed_num:
            continue

        fwd = (needed_num - current_num) % 7
        bwd = (current_num - needed_num) % 7

        if fwd <= bwd:
            for _ in range(fwd):
                actions.append(GameAction.ACTION2)
        else:
            for _ in range(bwd):
                actions.append(GameAction.ACTION1)

    return actions


def solve_tr87_alter_level(game, verbose=True):
    """Solve an alter_rules tr87 level."""
    # In alter_rules mode:
    # - cifzvbcuwqe groups are pairs of (left_group, right_group)
    # - cursor navigates ALL group elements (flattened from all rules)
    # - ACTION1/2 cycles ALL sprites in the current group simultaneously

    rules = game.cifzvbcuwqe
    target_row = game.zvojhrjxxm
    adjust_row = game.ztgmtnnufb  # Not directly used in alter_rules
    cursor_idx = game.qvtymdcqear_index

    # Flatten all groups
    all_groups = [grp for rule in rules for grp in rule]
    n_groups = len(all_groups)

    if verbose:
        print(f'  alter_rules: {n_groups} groups')
        for i, grp in enumerate(all_groups):
            print(f'    group {i}: {[(s.name[-2:], s.rotation) for s in grp]}')

    # The win condition matches target_row[i..i+k] to rule.left names
    # and corresponding adjust positions to rule.right names
    # For alter_rules, ACTION1/2 cycles ALL sprites in the selected group

    # We need to figure out what each group needs to be cycled to
    # The target_row determines which rules to use
    # The adjust_row (ztgmtnnufb) values must match rule.right values

    # Strategy: BFS on this level (since it's complex)
    # But first try analytical if possible

    # For each rule: left must match a subsequence of target_row
    # right must match corresponding subsequence of ztgmtnnufb
    # In alter_rules, we can cycle any group's sprites

    # Let me just figure out what each group's target sprite type should be
    # by checking what the win condition requires

    # Actually, let me just use BFS for alter_rules levels since they're complex
    return None  # Fall through to BFS


def solve_tr87_bfs(env, prefix_actions, verbose=True):
    """BFS solver for tr87 using deepcopy."""
    game = env._game
    actions = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4]

    # Get initial state hash
    def get_state_key(g):
        """Extract minimal state for dedup."""
        # For tr87: cursor index + sprite names
        parts = [str(g.qvtymdcqear_index)]
        if g.current_level.get_data("alter_rules"):
            all_groups = [grp for rule in g.cifzvbcuwqe for grp in rule]
            for grp in all_groups:
                for s in grp:
                    parts.append(s.name[-2:])
        else:
            for s in g.ztgmtnnufb:
                parts.append(s.name[-2:])
        return '|'.join(parts)

    init_key = get_state_key(game)

    # BFS using deepcopy
    queue = deque()
    queue.append(([], game))
    visited = {init_key}
    states = 0
    t0 = time.time()

    while queue:
        seq, g_state = queue.popleft()

        if len(seq) >= 80:
            continue

        for a_idx, act in enumerate(actions):
            g_copy = copy.deepcopy(g_state)
            action_input = ActionInput(id=act, data={})
            result = g_copy.perform_action(action_input, raw=True)
            states += 1

            if states % 5000 == 0 and verbose:
                elapsed = time.time() - t0
                print(f'  BFS: {states} states, queue={len(queue)}, depth={len(seq)+1}, {elapsed:.1f}s')

            if result and result.state == GameState.WIN:
                if verbose:
                    print(f'  WIN at depth {len(seq)+1}! States: {states}')
                return [actions[i] for i in (seq + [a_idx])]

            if result and result.state == GameState.GAME_OVER:
                continue

            key = get_state_key(g_copy)
            if key not in visited:
                visited.add(key)
                queue.append((seq + [a_idx], g_copy))

            if time.time() - t0 > 120:
                if verbose:
                    print(f'  Time limit')
                return None

    return None


# ============================================================
# RE86 SOLVER — Sprite matching puzzle
# ============================================================
def solve_re86(arc_inst, verbose=True):
    """Solve all RE86 levels analytically."""
    info = get_game_info(arc_inst, 're86')
    env = arc_inst.make(info.game_id)
    obs = env.reset()

    if verbose:
        print(f'\n{"="*70}')
        print(f'SOLVING RE86 ({len(info.baseline_actions)} levels, baseline={info.baseline_actions})')
        print(f'{"="*70}')

    all_actions = []
    per_level = {}

    for level_idx in range(len(info.baseline_actions)):
        if verbose:
            print(f'\n--- Level {level_idx + 1} ---')

        level_actions = solve_level_bfs_deepcopy(env, verbose=verbose, max_depth=60, time_limit=180)

        if level_actions is None:
            if verbose:
                print(f'  FAILED level {level_idx + 1}')
            break

        for act in level_actions:
            obs = env.step(act)
        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if verbose:
            print(f'  Solution: {len(level_actions)} actions')

        if obs.state == GameState.WIN:
            if verbose:
                print(f'  GAME COMPLETE!')
            break

    return build_result('re86', all_actions, per_level, info)


def solve_level_bfs_deepcopy(env, verbose=True, max_depth=60, time_limit=120,
                              kb_actions=None, click_positions=None):
    """
    Generic BFS solver using deepcopy of game state.
    Works for any game type. Much faster than replay-from-scratch.
    """
    game = env._game

    if kb_actions is None:
        kb_actions = [a for a in env.action_space if a != GameAction.ACTION6]

    # Build action list
    action_set = list(kb_actions)
    if click_positions:
        for x, y in click_positions:
            action_set.append(('click', x, y))

    def get_frame(g):
        return np.array(g.get_pixels(0, 0, g.camera.width, g.camera.height), dtype=np.uint8)

    init_frame = get_frame(game)
    init_hash = hashlib.md5(init_frame.tobytes()).hexdigest()

    queue = deque()
    queue.append(([], game))
    visited = {init_hash}
    states = 0
    t0 = time.time()

    while queue:
        seq, g_state = queue.popleft()

        if len(seq) >= max_depth:
            continue

        for a_idx, act in enumerate(action_set):
            g_copy = copy.deepcopy(g_state)

            if isinstance(act, tuple) and act[0] == 'click':
                _, x, y = act
                action_input = ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y})
            else:
                action_input = ActionInput(id=act, data={})

            result = g_copy.perform_action(action_input, raw=True)
            states += 1

            if states % 5000 == 0 and verbose:
                elapsed = time.time() - t0
                print(f'  BFS: {states} states, queue={len(queue)}, depth={len(seq)+1}, '
                      f'visited={len(visited)}, {elapsed:.1f}s')

            if result and result.state == GameState.WIN:
                if verbose:
                    elapsed = time.time() - t0
                    print(f'  WIN at depth {len(seq)+1}! States: {states}, {elapsed:.1f}s')
                # Convert to actual actions
                solution = []
                for idx in seq + [a_idx]:
                    a = action_set[idx]
                    if isinstance(a, tuple) and a[0] == 'click':
                        solution.append((GameAction.ACTION6, {'x': a[1], 'y': a[2]}))
                    else:
                        solution.append(a)
                return solution

            if result and result.state == GameState.GAME_OVER:
                continue

            f = get_frame(g_copy)
            h = hashlib.md5(f.tobytes()).hexdigest()
            if h not in visited:
                visited.add(h)
                queue.append((seq + [a_idx], g_copy))

            if time.time() - t0 > time_limit:
                if verbose:
                    print(f'  Time limit ({time_limit}s), states={states}, visited={len(visited)}')
                return None

    if verbose:
        print(f'  BFS exhausted: {states} states, {len(visited)} unique')
    return None


# ============================================================
# CLICK-BASED SOLVER (tn36, lp85)
# ============================================================
def find_active_clicks_dc(env, grid_step=2, verbose=True):
    """Find active click positions using deepcopy."""
    game = env._game

    def get_frame(g):
        return np.array(g.get_pixels(0, 0, g.camera.width, g.camera.height), dtype=np.uint8)

    base_frame = get_frame(game)
    base_hash = hashlib.md5(base_frame.tobytes()).hexdigest()

    active = []
    unique_hashes = {}

    for y in range(0, 64, grid_step):
        for x in range(0, 64, grid_step):
            g_copy = copy.deepcopy(game)
            action_input = ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y})
            result = g_copy.perform_action(action_input, raw=True)

            if result is None:
                continue

            if result.state == GameState.WIN:
                if verbose:
                    print(f'  Single click WIN at ({x},{y})!')
                return [('win', x, y)]

            f = get_frame(g_copy)
            h = hashlib.md5(f.tobytes()).hexdigest()

            if h != base_hash and h not in unique_hashes:
                diff = int(np.abs(f.astype(float) - base_frame.astype(float)).sum())
                unique_hashes[h] = (x, y, diff)
                active.append((x, y, diff))

    if verbose:
        print(f'  Found {len(active)} unique active clicks (grid={grid_step})')

    return active


def solve_click_game(arc_inst, game_key, verbose=True):
    """Solve a click-only game level by level."""
    info = get_game_info(arc_inst, game_key)
    env = arc_inst.make(info.game_id)
    obs = env.reset()
    n_levels = len(info.baseline_actions)

    if verbose:
        print(f'\n{"="*70}')
        print(f'SOLVING {game_key.upper()} ({n_levels} levels, baseline={info.baseline_actions})')
        print(f'{"="*70}')

    all_actions = []
    per_level = {}

    for level_idx in range(n_levels):
        if verbose:
            print(f'\n--- Level {level_idx + 1}/{n_levels} ---')

        # Find active click positions
        active = find_active_clicks_dc(env, grid_step=2, verbose=verbose)

        if not active:
            if verbose:
                print(f'  No active clicks found!')
            # Try grid_step=1
            active = find_active_clicks_dc(env, grid_step=1, verbose=verbose)

        if not active:
            if verbose:
                print(f'  FAILED level {level_idx + 1} - no active clicks')
            break

        if active[0][0] == 'win':
            _, x, y = active[0]
            level_actions = [(GameAction.ACTION6, {'x': x, 'y': y})]
        else:
            # Get unique click positions and sort by effect size
            click_positions = [(x, y) for x, y, d in sorted(active, key=lambda t: t[2], reverse=True)]

            # Limit positions for tractable BFS
            max_clicks = 30
            if len(click_positions) > max_clicks:
                click_positions = click_positions[:max_clicks]

            baseline = info.baseline_actions[level_idx]
            max_d = min(15, max(3, int(baseline * 1.2)))

            if verbose:
                print(f'  Using {len(click_positions)} click positions, max_depth={max_d}')

            level_actions = solve_level_bfs_deepcopy(
                env, verbose=verbose, max_depth=max_d, time_limit=180,
                kb_actions=[], click_positions=click_positions
            )

        if level_actions is None:
            if verbose:
                print(f'  FAILED level {level_idx + 1}')
            break

        if verbose:
            print(f'  Solution: {len(level_actions)} actions')

        for act in level_actions:
            if isinstance(act, tuple):
                obs = env.step(act[0], data=act[1])
            else:
                obs = env.step(act)

        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs.state == GameState.WIN:
            if verbose:
                print(f'  GAME COMPLETE!')
            break

    return build_result(game_key, all_actions, per_level, info)


# ============================================================
# KB+CLICK SOLVER (sk48, ar25)
# ============================================================
def solve_kb_click_game(arc_inst, game_key, verbose=True):
    """Solve a keyboard+click game level by level."""
    info = get_game_info(arc_inst, game_key)
    env = arc_inst.make(info.game_id)
    obs = env.reset()
    n_levels = len(info.baseline_actions)

    if verbose:
        print(f'\n{"="*70}')
        print(f'SOLVING {game_key.upper()} ({n_levels} levels, baseline={info.baseline_actions})')
        print(f'{"="*70}')

    all_actions = []
    per_level = {}

    for level_idx in range(n_levels):
        if verbose:
            print(f'\n--- Level {level_idx + 1}/{n_levels} ---')

        baseline = info.baseline_actions[level_idx]

        # First try KB-only BFS
        kb_actions = [a for a in env.action_space if a != GameAction.ACTION6]
        max_d = min(40, max(8, int(baseline * 1.5)))

        level_actions = solve_level_bfs_deepcopy(
            env, verbose=verbose, max_depth=max_d, time_limit=90,
            kb_actions=kb_actions
        )

        if level_actions is None and GameAction.ACTION6 in env.action_space:
            if verbose:
                print(f'  KB-only failed, trying with clicks...')
            # Find active clicks
            active = find_active_clicks_dc(env, grid_step=2, verbose=verbose)
            if active:
                if active[0][0] == 'win':
                    _, x, y = active[0]
                    level_actions = [(GameAction.ACTION6, {'x': x, 'y': y})]
                else:
                    clicks = [(x, y) for x, y, d in sorted(active, key=lambda t: t[2], reverse=True)[:20]]
                    level_actions = solve_level_bfs_deepcopy(
                        env, verbose=verbose, max_depth=min(20, baseline),
                        time_limit=120, kb_actions=kb_actions, click_positions=clicks
                    )

        if level_actions is None:
            if verbose:
                print(f'  FAILED level {level_idx + 1}')
            break

        if verbose:
            print(f'  Solution: {len(level_actions)} actions')

        for act in level_actions:
            if isinstance(act, tuple):
                obs = env.step(act[0], data=act[1])
            else:
                obs = env.step(act)

        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs.state == GameState.WIN:
            if verbose:
                print(f'  GAME COMPLETE!')
            break

    return build_result(game_key, all_actions, per_level, info)


# ============================================================
# UTILITIES
# ============================================================
def build_result(game_key, all_actions, per_level, info):
    """Build result dictionary."""
    total = sum(per_level.values())
    return {
        'game': game_key,
        'source': 'analytical_solver (solve_6games_fullchain)',
        'type': 'analytical',
        'total_actions': total,
        'max_level': len(per_level),
        'n_levels': len(info.baseline_actions),
        'per_level': per_level,
        'baseline': info.baseline_actions,
        'all_actions': serialize_actions(all_actions),
        'raw_actions_repr': [repr_action(a) for a in all_actions],
    }


def repr_action(act):
    """Readable representation of an action."""
    if isinstance(act, tuple):
        ga, data = act
        return f'CLICK({data["x"]},{data["y"]})'
    return act.name


def main():
    games = sys.argv[1:] if len(sys.argv) > 1 else ['tr87', 're86', 'tn36', 'lp85', 'sk48', 'ar25']

    arc = make_arc()

    for gid in games:
        t0 = time.time()
        try:
            if gid == 'tr87':
                result = solve_tr87(arc, verbose=True)
            elif gid in ('tn36', 'lp85'):
                result = solve_click_game(arc, gid, verbose=True)
            elif gid in ('sk48', 'ar25'):
                result = solve_kb_click_game(arc, gid, verbose=True)
            elif gid == 're86':
                result = solve_re86(arc, verbose=True)
            else:
                print(f'Unknown game: {gid}')
                continue

            result['elapsed'] = round(time.time() - t0, 2)
            save_result(gid, result)

            print(f'\n  Summary: {result["max_level"]}/{result["n_levels"]} levels, '
                  f'{result["total_actions"]} actions, {result["elapsed"]}s')
            print(f'  Per level: {result["per_level"]}')
            print(f'  Baseline: {result["baseline"]}')

        except Exception as e:
            print(f'ERROR solving {gid}: {e}')
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
