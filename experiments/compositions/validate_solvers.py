"""
Solver Validation — Step 1268 prerequisite.

Replays each game's fullchain prescription on the current game version.
Reports which games still reach all expected levels.

Usage: PYTHONUTF8=1 python validate_solvers.py
"""
import sys, os, json, time
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/archive')

PDIR = 'B:/M/the-search/experiments/results/prescriptions'

# Map each game to its prescription file and action field name
GAME_PRESCRIPTIONS = {
    'ft09': ('ft09_fullchain.json',  'all_actions',   6),
    'ls20': ('ls20_fullchain.json',  'all_actions',   7),
    'vc33': ('vc33_analytical.json', 'all_actions',   7),
    'tr87': ('tr87_fullchain.json',  'all_actions',   6),
    'sp80': ('sp80_fullchain.json',  'sequence',      6),
    'sb26': ('sb26_fullchain.json',  'all_actions',   8),
    'tu93': ('tu93_fullchain.json',  'all_actions',   9),
    'cn04': ('cn04_fullchain.json',  'sequence',      5),
    'cd82': ('cd82_fullchain.json',  'sequence',      6),
    'lp85': ('lp85_full_seq.json',   'full_sequence', 8),
}


def make_game(game_name: str):
    try:
        import arcagi3
        return arcagi3.make(game_name.upper())
    except ImportError:
        import util_arcagi3
        return util_arcagi3.make(game_name.upper())


def load_sequence(game: str) -> tuple:
    """Returns (actions, expected_levels) or (None, 0) if not found."""
    fname, field, expected_levels = GAME_PRESCRIPTIONS[game]
    path = os.path.join(PDIR, fname)
    if not os.path.exists(path):
        return None, expected_levels
    with open(path) as f:
        d = json.load(f)
    actions = d.get(field, [])
    return actions, expected_levels


def validate_game(game: str, seed: int = 1) -> dict:
    """Replay prescription sequence, report levels reached."""
    actions, expected_levels = load_sequence(game)
    if not actions:
        return {
            'game': game, 'status': 'NO_SEQUENCE',
            'expected_levels': expected_levels,
            'max_level_reached': 0, 'actions_replayed': 0,
            'passed': False,
        }

    try:
        env = make_game(game)
    except Exception as e:
        return {
            'game': game, 'status': f'ENV_FAIL: {e}',
            'expected_levels': expected_levels,
            'max_level_reached': 0, 'actions_replayed': 0,
            'passed': False,
        }

    obs = env.reset(seed=seed)
    max_level = 0
    level = 0
    fresh_episode = True
    actions_taken = 0
    level_transitions = {}  # level -> action_index when first reached

    t_start = time.time()
    for i, action in enumerate(actions):
        if obs is None:
            obs = env.reset(seed=seed)
            fresh_episode = True
            continue

        action_int = int(action) % env.n_actions
        obs, reward, done, info = env.step(action_int)
        actions_taken += 1

        if fresh_episode:
            fresh_episode = False
            continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if cl > level:
            level = cl
            if cl > max_level:
                max_level = cl
                level_transitions[cl] = actions_taken

        if done:
            obs = env.reset(seed=seed)
            fresh_episode = True
            level = 0

    elapsed = time.time() - t_start
    passed = max_level >= expected_levels - 1  # reached final level (0-indexed: level n-1 = n levels solved)

    return {
        'game': game,
        'status': 'OK',
        'expected_levels': expected_levels,
        'max_level_reached': max_level,
        'level_transitions': level_transitions,
        'actions_replayed': actions_taken,
        'total_actions_in_seq': len(actions),
        'elapsed_seconds': round(elapsed, 2),
        'passed': passed,
    }


def main():
    print("=" * 65)
    print("SOLVER VALIDATION — replaying prescriptions on current game versions")
    print("=" * 65)
    print()

    results = []
    passed = []
    failed = []

    for game in GAME_PRESCRIPTIONS:
        print(f"  {game.upper()} ...", end='', flush=True)
        r = validate_game(game)
        results.append(r)
        status = 'PASS' if r['passed'] else 'FAIL'
        lvl = r['max_level_reached']
        exp = r['expected_levels']
        transitions = r.get('level_transitions', {})
        trans_str = ' '.join(f"L{k}@{v}" for k, v in sorted(transitions.items())) if transitions else 'none'
        print(f" {status} — reached L{lvl}/{exp-1} | {trans_str} | {r['elapsed_seconds']:.1f}s")
        (passed if r['passed'] else failed).append(game)

    print()
    print("=" * 65)
    print(f"PASSED ({len(passed)}/10): {passed}")
    print(f"FAILED ({len(failed)}/10): {failed}")
    print("=" * 65)

    out_path = os.path.join(
        'B:/M/the-search/experiments/compositions',
        'solver_validation_results.json'
    )
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'passed': passed, 'failed': failed}, f, indent=2)
    print(f"\nResults: {out_path}")


if __name__ == '__main__':
    main()
