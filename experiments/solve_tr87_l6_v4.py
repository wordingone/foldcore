"""
Solve TR87 Level 6 — full constraint satisfaction.
Enumerate all 7^12 group delta configurations (pruned by constraint propagation).
"""
import sys, json, time, copy, copyreg, itertools
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState, ActionInput

def pickle_gameaction(ga):
    return (GameAction.__getitem__, (ga.name,))
copyreg.pickle(GameAction, pickle_gameaction)

with open('B:/M/the-search/experiments/results/prescriptions/tr87_fullchain.json') as f:
    existing = json.load(f)

ACTION_MAP = {0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3, 3: GameAction.ACTION4}

def cycle_name(name, delta):
    base = name[:-1]
    num = int(name[-1])
    new_num = (num + delta - 1) % 7 + 1
    return base + str(new_num)

def play_to_level(env, target_level):
    obs = env.reset()
    for act_idx in existing['all_actions']:
        obs = env.step(ACTION_MAP[act_idx])
        if obs is None: return None, None
        if obs.levels_completed >= target_level:
            return obs, env
    return obs, env


def check_win(initial_groups, group_deltas, target_names, adjust_names, rules_structure, tree_translation=True):
    """
    Simulate the win condition check with given group deltas.

    initial_groups: list of list of name strings
    group_deltas: list of ints (one per group)
    target_names: list of name strings (fixed)
    adjust_names: list of name strings (fixed)
    rules_structure: list of (left_group_idx, right_group_idx) pairs
    """
    # Apply deltas to get current group names
    current_groups = []
    for gid, grp in enumerate(initial_groups):
        d = group_deltas[gid]
        current_groups.append([cycle_name(n, d) for n in grp])

    # Build rules from current groups
    rules = []
    for left_gid, right_gid in rules_structure:
        rules.append((current_groups[left_gid], current_groups[right_gid]))

    # Simulate bsqsshqpox
    t_idx = 0
    a_idx = 0

    while t_idx < len(target_names):
        matched = False
        for left_names, right_names in rules:
            if t_idx + len(left_names) > len(target_names):
                continue
            if not all(target_names[t_idx + i] == left_names[i] for i in range(len(left_names))):
                continue

            if tree_translation:
                # For each sprite in right, find a rule whose left[0] matches
                final_right = []
                intermediates = []
                ok = True
                for rname in right_names:
                    found = False
                    for sub_left, sub_right in rules:
                        if sub_left[0] == rname:
                            final_right.extend(sub_right)
                            intermediates.append(rname)
                            found = True
                            break
                    if not found:
                        ok = False
                        break
                if not ok:
                    continue
            else:
                final_right = right_names

            if a_idx + len(final_right) > len(adjust_names):
                return False
            if not all(adjust_names[a_idx + i] == final_right[i] for i in range(len(final_right))):
                return False

            t_idx += len(left_names)
            a_idx += len(final_right)
            matched = True
            break

        if not matched:
            return False

    return a_idx == len(adjust_names)


def main():
    arc = arc_agi.Arcade()
    info = next(g for g in arc.get_environments() if 'tr87' in g.game_id.lower())
    env = arc.make(info.game_id)

    print("Playing L1-L5...")
    obs, env = play_to_level(env, 5)
    print(f"Reached level {obs.levels_completed + 1}")

    game = env._game
    rules = game.cifzvbcuwqe

    target_names = [s.name for s in game.zvojhrjxxm]
    adjust_names = [s.name for s in game.ztgmtnnufb]

    initial_groups = []
    rules_structure = []
    for i, (left, right) in enumerate(rules):
        left_gid = len(initial_groups)
        initial_groups.append([s.name for s in left])
        right_gid = len(initial_groups)
        initial_groups.append([s.name for s in right])
        rules_structure.append((left_gid, right_gid))

    n_groups = len(initial_groups)
    print(f"\nTarget: {target_names}")
    print(f"Adjust: {adjust_names}")
    print(f"Groups: {n_groups}, rules: {len(rules_structure)}")
    for i, grp in enumerate(initial_groups):
        print(f"  G{i}: {grp}")

    # Smart search: enumerate only left-group deltas for A-rules (3 groups: 0, 4, 8)
    # and B-rules used as sub-rules (3 groups: 2, 6, 10)
    # The right-group deltas (1, 3, 5, 7, 9, 11) follow from constraints.

    # Actually, let's be smarter. Only 6 "left" groups matter (0,2,4,6,8,10).
    # 7^6 = 117649 — very manageable.
    # For each left-group configuration, the right-group deltas don't affect the
    # matching of left sides. They only affect the final adjust row match.
    # So we can check left-side matching first, then enumerate right-group deltas.

    # Even simpler: 7^12 = 13.8B is too much, but we can prune heavily.
    # Let's enumerate left-group deltas (7^6 = 117649), then for each that has
    # valid left-side matching, check if right-group deltas can satisfy adjust row.

    # For left matching: A-rules (groups 0, 4, 8) must have their left match target.
    # We know the valid deltas already.
    # For sub-rule matching: B-rules (groups 2, 6, 10) must have their left match
    # the intermediate B-names that come from A-rules' right sides.

    # The intermediate B-names depend on A-rules' right-group deltas (groups 1, 5, 9).
    # So we need to enumerate right-group deltas for A-rules too.

    # Let me just enumerate all 7^12 but with heavy pruning.
    # Better approach: enumerate just the 6 left groups + verify.

    # Actually, let's think about it differently. There are 12 groups.
    # Groups 0,4,8 (A-rule lefts): must cycle to match target. 3 valid deltas each.
    # Groups 1,5,9 (A-rule rights): 7 values each.
    # Groups 2,6,10 (B-rule lefts): must match intermediate B-names. Depends on groups 1,5,9.
    # Groups 3,7,11 (B-rule rights): must match adjust row. Determined by other choices.

    # So: enumerate Groups 0,4,8 deltas (3*3*3 = 27), Groups 1,5,9 deltas (7*7*7 = 343),
    # then for each combo, determine required Groups 2,6,10 and 3,7,11.
    # Total: 27 * 343 = 9261 combinations to check.

    # Valid deltas for A-rule left groups:
    # Group 0 (A3): needs to match A7, A6, or A1 (target positions)
    # But we need 3 rules covering 3 target positions. Each A-rule covers 1 position.
    # Since same rule can't be reused with different left deltas, we need 3 DIFFERENT rules.

    # Rules 0, 2, 4 are A-rules (groups 0, 4, 8).
    # We need to assign each to a target position (3! = 6 permutations).

    # For a given assignment, each A-rule's left delta is determined.
    # Then the right-side names (after cycling right groups) must flow through sub-rules.

    # Let me enumerate properly.
    t0 = time.time()
    solutions = []

    # Target: [A7, A6, A1]
    a_rules = [0, 2, 4]  # Rule indices (left groups 0, 4, 8)

    for perm in itertools.permutations(range(3)):
        # perm[i] = which A-rule handles target position i
        # Check that each rule's left group can cycle to match target[i]
        a_deltas = {}  # left_group_idx -> delta
        ok = True
        for t_pos in range(3):
            rule_idx = a_rules[perm[t_pos]]
            left_gid = rule_idx * 2
            left_names = initial_groups[left_gid]

            # left has 1 sprite. Find delta so cycle_name(left[0], d) = target[t_pos]
            # target[0]=A7, target[1]=A6, target[2]=A1
            t_name = target_names[t_pos]
            found = False
            for d in range(7):
                if cycle_name(left_names[0], d) == t_name:
                    if left_gid in a_deltas and a_deltas[left_gid] != d:
                        ok = False
                        break
                    a_deltas[left_gid] = d
                    found = True
                    break
            if not found or not ok:
                ok = False
                break

        if not ok:
            continue

        # Now enumerate right-group deltas for A-rules
        right_gids = [a_rules[perm[i]] * 2 + 1 for i in range(3)]

        for r_deltas in itertools.product(range(7), repeat=3):
            # right_gids[i] gets delta r_deltas[i]
            right_delta_map = {}
            for i in range(3):
                right_delta_map[right_gids[i]] = r_deltas[i]

            # Compute intermediate B-names from A-rules' right sides
            # For each target position, the A-rule's right side (after cycling) gives B-names
            # These B-names must match sub-rules' left sides
            # The sub-rules' right sides (after cycling) must match adjust row

            # Collect all intermediate B-names in order
            intermediate_b = []
            for t_pos in range(3):
                rule_idx = a_rules[perm[t_pos]]
                right_gid = rule_idx * 2 + 1
                right_names = initial_groups[right_gid]
                rd = right_delta_map[right_gid]
                cycled_right = [cycle_name(n, rd) for n in right_names]
                intermediate_b.extend(cycled_right)

            # Now: intermediate_b has 6 B-names.
            # For tree_translation, each B-name must match some B-rule's left[0].
            # The B-rule's right side (after cycling) must match the corresponding adjust entry.

            # B-rules: 1, 3, 5 (left groups 2, 6, 10)
            b_rules = [(1, 2, 3), (3, 6, 7), (5, 10, 11)]  # (rule_idx, left_gid, right_gid)

            # For each intermediate B-name, find which B-rule can match it and what deltas are needed
            b_group_deltas = {}
            all_ok = True

            for ib_idx, bname in enumerate(intermediate_b):
                adj_name = adjust_names[ib_idx]
                found = False

                for br_idx, br_left_gid, br_right_gid in b_rules:
                    br_left_names = initial_groups[br_left_gid]  # 1 sprite
                    br_right_names = initial_groups[br_right_gid]  # 1 sprite

                    # Find delta for left match
                    for ld in range(7):
                        if cycle_name(br_left_names[0], ld) == bname:
                            # Check consistency
                            if br_left_gid in b_group_deltas and b_group_deltas[br_left_gid] != ld:
                                continue  # Inconsistent

                            # Find delta for right to match adjust
                            for rd in range(7):
                                if cycle_name(br_right_names[0], rd) == adj_name:
                                    if br_right_gid in b_group_deltas and b_group_deltas[br_right_gid] != rd:
                                        continue

                                    # Check this is fully consistent
                                    test_deltas = dict(b_group_deltas)
                                    test_deltas[br_left_gid] = ld
                                    test_deltas[br_right_gid] = rd
                                    found = True

                                    # But we need to continue and check ALL remaining assignments
                                    # This is getting recursive... let me just try this and continue

                                    b_group_deltas[br_left_gid] = ld
                                    b_group_deltas[br_right_gid] = rd
                                    break
                            if found:
                                break
                    if found:
                        break

                if not found:
                    all_ok = False
                    break

            if all_ok:
                # Build full delta map
                full_deltas = [0] * n_groups
                for gid, d in a_deltas.items():
                    full_deltas[gid] = d
                for gid, d in right_delta_map.items():
                    full_deltas[gid] = d
                for gid, d in b_group_deltas.items():
                    full_deltas[gid] = d

                # Verify with full win check
                if check_win(initial_groups, full_deltas, target_names, adjust_names, rules_structure):
                    solutions.append(full_deltas)
                    print(f"  SOLUTION: deltas={full_deltas}")

    elapsed = time.time() - t0
    print(f"\nSearch complete: {len(solutions)} solutions in {elapsed:.1f}s")

    if not solutions:
        # The greedy assignment above may miss solutions where the same B-rule
        # is used for multiple B-names with different deltas (impossible since same group).
        # Let me try a cleaner recursive approach.
        print("\nTrying recursive search...")
        solutions = recursive_search(initial_groups, target_names, adjust_names, rules_structure)

    if not solutions:
        print("No solution found!")
        return

    # Pick solution with minimum total cycling
    best = min(solutions, key=lambda d: sum(d))
    print(f"\nBest solution: {best}")
    print(f"Total cycling: {sum(best)}")

    # Convert to action sequence
    actions = build_action_sequence(best, n_groups)
    print(f"Action sequence: {len(actions)} actions")
    print(f"Actions: {[a.name for a in actions]}")

    # Verify
    verify_and_save(arc, info, actions, best)


def recursive_search(initial_groups, target_names, adjust_names, rules_structure):
    """Full recursive search for valid delta configurations."""
    n_groups = len(initial_groups)

    # Rules: 0,1,2,3,4,5
    # A-rules: 0,2,4 (left groups 0,4,8; right groups 1,5,9)
    # B-rules: 1,3,5 (left groups 2,6,10; right groups 3,7,11)

    solutions = []
    t0 = time.time()
    count = [0]

    def search(t_pos, a_idx, deltas_fixed, adjust_remaining):
        """
        Recursive search.
        t_pos: current position in target_names
        a_idx: current position in adjust_names
        deltas_fixed: dict of group_idx -> delta (already committed)
        adjust_remaining: adjust_names[a_idx:]
        """
        count[0] += 1
        if count[0] % 100000 == 0:
            print(f"  ... {count[0]} nodes, {time.time()-t0:.1f}s")

        if t_pos >= len(target_names):
            if a_idx >= len(adjust_names):
                return [dict(deltas_fixed)]
            return []

        results = []

        # Try each A-rule to match target[t_pos]
        for a_rule_idx in [0, 2, 4]:
            left_gid = a_rule_idx * 2
            right_gid = a_rule_idx * 2 + 1
            left_names = initial_groups[left_gid]

            # Find delta for left group to match target[t_pos]
            t_name = target_names[t_pos]
            for ld in range(7):
                if cycle_name(left_names[0], ld) != t_name:
                    continue

                # Check consistency
                if left_gid in deltas_fixed and deltas_fixed[left_gid] != ld:
                    continue

                # Try all right deltas
                right_names = initial_groups[right_gid]
                for rd in range(7):
                    if right_gid in deltas_fixed and deltas_fixed[right_gid] != rd:
                        continue

                    cycled_right = [cycle_name(n, rd) for n in right_names]

                    # Tree translation: each cycled_right name must match some B-rule's left[0]
                    # and that B-rule's right must match adjust_names[a_idx + offset]

                    # Try to assign sub-rules for each right-side sprite
                    sub_results = assign_subrules(
                        cycled_right, 0, a_idx, deltas_fixed, left_gid, ld, right_gid, rd,
                        initial_groups, adjust_names
                    )

                    for (new_deltas, new_a_idx) in sub_results:
                        more = search(t_pos + len(left_names), new_a_idx, new_deltas, adjust_names[new_a_idx:])
                        results.extend(more)
                        if results:
                            return results  # Early exit on first solution

        return results

    def assign_subrules(cycled_right, cr_idx, a_idx, deltas_fixed, parent_left_gid, parent_ld,
                        parent_right_gid, parent_rd, initial_groups, adjust_names):
        """Recursively assign sub-rules for each B-name in cycled_right."""
        if cr_idx >= len(cycled_right):
            new_deltas = dict(deltas_fixed)
            new_deltas[parent_left_gid] = parent_ld
            new_deltas[parent_right_gid] = parent_rd
            return [(new_deltas, a_idx)]

        bname = cycled_right[cr_idx]
        results = []

        for b_rule_idx in [1, 3, 5]:
            b_left_gid = b_rule_idx * 2
            b_right_gid = b_rule_idx * 2 + 1
            b_left_names = initial_groups[b_left_gid]
            b_right_names = initial_groups[b_right_gid]

            for bld in range(7):
                if cycle_name(b_left_names[0], bld) != bname:
                    continue

                # Check consistency
                if b_left_gid in deltas_fixed and deltas_fixed[b_left_gid] != bld:
                    continue
                if b_left_gid == parent_left_gid and parent_ld != bld:
                    continue

                # Find right delta to match adjust
                if a_idx >= len(adjust_names):
                    continue

                for brd in range(7):
                    if b_right_gid in deltas_fixed and deltas_fixed[b_right_gid] != brd:
                        continue

                    cycled_b_right = [cycle_name(n, brd) for n in b_right_names]
                    if a_idx + len(cycled_b_right) > len(adjust_names):
                        continue
                    if not all(cycled_b_right[i] == adjust_names[a_idx + i] for i in range(len(cycled_b_right))):
                        continue

                    # Consistent! Continue to next cr_idx
                    temp_deltas = dict(deltas_fixed)
                    temp_deltas[parent_left_gid] = parent_ld
                    temp_deltas[parent_right_gid] = parent_rd
                    temp_deltas[b_left_gid] = bld
                    temp_deltas[b_right_gid] = brd

                    sub = assign_subrules(
                        cycled_right, cr_idx + 1, a_idx + len(cycled_b_right),
                        temp_deltas, parent_left_gid, parent_ld, parent_right_gid, parent_rd,
                        initial_groups, adjust_names
                    )
                    results.extend(sub)
                    if results:
                        return results  # Early exit

        return results

    result = search(0, 0, {}, adjust_names)
    print(f"  Recursive search: {count[0]} nodes")
    return result


def build_action_sequence(deltas, n_groups):
    """Convert group deltas to action sequence."""
    actions = []
    cursor = 0

    for gid in range(n_groups):
        d = deltas[gid] if isinstance(deltas, list) else deltas.get(gid, 0)
        if d == 0:
            continue

        # Move cursor to group gid
        fwd = (gid - cursor) % n_groups
        bwd = (cursor - gid) % n_groups
        if fwd <= bwd:
            for _ in range(fwd):
                actions.append(GameAction.ACTION4)
            cursor = gid
        else:
            for _ in range(bwd):
                actions.append(GameAction.ACTION3)
            cursor = gid

        # Cycle
        fwd_cycles = d
        bwd_cycles = 7 - d
        if fwd_cycles <= bwd_cycles:
            for _ in range(fwd_cycles):
                actions.append(GameAction.ACTION2)
        else:
            for _ in range(bwd_cycles):
                actions.append(GameAction.ACTION1)

    return actions


def verify_and_save(arc, info, actions, deltas):
    """Verify solution and save fullchain."""
    print("\n=== VERIFICATION (3x) ===")
    all_good = True
    for run in range(3):
        env = arc.make(info.game_id)
        obs, env = play_to_level(env, 5)
        for act in actions:
            obs = env.step(act)
            if obs is None:
                print(f"  Run {run}: NULL obs!")
                all_good = False
                break
        if obs is not None:
            success = obs.state == GameState.WIN
            print(f"  Run {run}: state={obs.state.name}, levels={obs.levels_completed} {'OK' if success else 'FAIL'}")
            if not success:
                all_good = False

    if all_good:
        action_indices = [a.value - 1 for a in actions]
        full_indices = existing['all_actions'] + action_indices
        per_level = dict(existing['per_level'])
        per_level['L6'] = len(actions)

        fullchain = {
            'game': 'tr87',
            'source': 'analytical_solver (solve_tr87_l6_v4)',
            'type': 'analytical',
            'total_actions': len(full_indices),
            'max_level': 6,
            'n_levels': 6,
            'per_level': per_level,
            'baseline': list(info.baseline_actions),
            'all_actions': full_indices,
            'raw_actions_repr': [ACTION_MAP[a].name for a in full_indices],
        }

        out_path = 'B:/M/the-search/experiments/results/prescriptions/tr87_fullchain.json'
        with open(out_path, 'w') as f:
            json.dump(fullchain, f, indent=2)
        print(f"\nSaved: {out_path}")
        print(f"Total: {len(full_indices)} actions across 6 levels (baseline total={sum(info.baseline_actions)})")
        for k, v in sorted(per_level.items()):
            bl = info.baseline_actions[int(k[1:])-1]
            print(f"  {k}: {v} actions (baseline: {bl})")
    else:
        print("\nVERIFICATION FAILED")


if __name__ == '__main__':
    main()
