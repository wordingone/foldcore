"""
SU15 fast solver. For each level:
1. Get initial fruit positions from source
2. Generate smart candidate clicks (midpoints of same-size pairs, positions along paths to goals)
3. Use greedy search with game engine verification
4. State dedup to avoid re-exploring

Key optimization: only try ~50-100 carefully chosen positions per step instead of thousands.
"""

import os
import json
import logging
import math
from collections import defaultdict
from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo
from arcengine import GameAction


def create_game():
    info = EnvironmentInfo(
        game_id='su15',
        local_dir='environment_files/su15/4c352900',
        class_name='Su15'
    )
    logger = logging.getLogger('su15')
    logger.setLevel(logging.WARNING)
    return LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)


def encode_action(x, y):
    return y * 64 + x + 7


def get_state(game):
    """Get game state."""
    g = game._game
    fruits = []
    for sprite in g.hmeulfxgy:
        size = g.amnmgwpkeb.get(sprite, 0)
        fruits.append((sprite.x, sprite.y, size))
    enemies = []
    for sprite in g.peiiyyzum:
        enemies.append((sprite.x, sprite.y))
    remaining = g.step_counter_ui.current_steps
    return sorted(fruits), enemies, remaining


def state_hash(fruits):
    return tuple(sorted(fruits))


def smart_candidates(fruits, goals, radius=8):
    """Generate smart click candidates based on current fruit state."""
    candidates = set()

    # 1. Midpoints between same-size fruit pairs (for merging)
    by_size = defaultdict(list)
    for fx, fy, fs in fruits:
        by_size[fs].append((fx, fy))

    for size, positions in by_size.items():
        if len(positions) < 2:
            continue
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                if dist <= 2 * radius:  # Both might be in range
                    # Try midpoint and nearby
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    for dx in range(-2, 3, 2):
                        for dy in range(-2, 3, 2):
                            x, y = mx + dx, my + dy
                            if 0 <= x <= 63 and 10 <= y <= 63:
                                candidates.add((x, y))
                elif dist <= 4 * radius:
                    # Pull one toward the other
                    for t in [0.3, 0.5, 0.7]:
                        x = int(x1 + (x2 - x1) * t)
                        y = int(y1 + (y2 - y1) * t)
                        if 0 <= x <= 63 and 10 <= y <= 63:
                            candidates.add((x, y))

    # 2. Points along path from each fruit to nearest goal (for moving to goal)
    for fx, fy, fs in fruits:
        for gx, gy in goals:
            gcx, gcy = gx + 4, gy + 4  # Goal center
            dist = math.sqrt((gcx - fx)**2 + (gcy - fy)**2)
            if dist > 0:
                # Points 5-15 pixels from fruit toward goal
                for d in [5, 8, 10, 12, 15]:
                    if d > dist:
                        d = dist
                    x = int(fx + (gcx - fx) * d / dist)
                    y = int(fy + (gcy - fy) * d / dist)
                    if 0 <= x <= 63 and 10 <= y <= 63:
                        candidates.add((x, y))

    # 3. On/near each goal (for final placement)
    for gx, gy in goals:
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                x, y = gx + 4 + dx, gy + 4 + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    candidates.add((x, y))

    # 4. On/near each fruit (for pulling other fruits to it)
    for fx, fy, fs in fruits:
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                x, y = fx + dx, fy + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    candidates.add((x, y))

    return sorted(candidates)


def solve_level_greedy(prefix_clicks, level_idx, step_limit):
    """Greedy solver: at each step, try all smart candidates and pick the best."""
    target_lc = level_idx + 1

    # Initial state
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})

    fruits, enemies, remaining = get_state(game)
    goals = [(s.x, s.y) for s in game._game.rqdsgrklq]
    goal_req = game._game.reqbygadvzmjired

    print(f"  Fruits: {fruits}")
    print(f"  Goals: {goals}")
    print(f"  Goal req: {goal_req}")
    print(f"  Steps: {remaining}")

    clicks = []
    visited_states = set()

    for step in range(step_limit):
        candidates = smart_candidates(fruits, goals)
        if not candidates:
            print(f"  Step {step+1}: No candidates!")
            break

        best_click = None
        best_score = -float('inf')
        best_state = None
        solved = False

        for cx, cy in candidates:
            # Replay from scratch
            game = create_game()
            game.reset()
            for x, y in prefix_clicks:
                game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            for x, y in clicks:
                game.step(GameAction.ACTION6, data={'x': x, 'y': y})

            result = game.step(GameAction.ACTION6, data={'x': cx, 'y': cy})

            if result.state == 'GAME_OVER':
                continue

            if result.levels_completed >= target_lc:
                clicks.append((cx, cy))
                print(f"  Step {step+1}: SOLVED! Click ({cx},{cy})")
                return clicks

            new_fruits, new_enemies, new_remaining = get_state(game)
            sh = state_hash(new_fruits)
            if sh in visited_states:
                continue

            score = compute_score(new_fruits, goals, goal_req, fruits)

            if score > best_score:
                best_score = score
                best_click = (cx, cy)
                best_state = (new_fruits, new_enemies, new_remaining, sh)

        if best_click is None:
            print(f"  Step {step+1}: No improving clicks found!")
            break

        clicks.append(best_click)
        fruits, enemies, remaining, sh = best_state
        visited_states.add(sh)
        print(f"  Step {step+1}: Click ({best_click[0]},{best_click[1]}) -> {len(fruits)} fruits, remaining={remaining}, score={best_score:.1f}")

        if remaining <= 0:
            print("  Out of steps!")
            break

    return None


def compute_score(fruits, goals, goal_req, prev_fruits):
    """Score state quality."""
    if not fruits:
        return -10000

    # Parse targets
    targets = []
    if goal_req:
        if isinstance(goal_req[0], (list, tuple)):
            for item in goal_req:
                targets.append((str(item[0]), int(item[1])))
        else:
            targets.append((str(goal_req[0]), int(goal_req[1])))

    score = 0

    # Reward merges
    merge_count = len(prev_fruits) - len(fruits)
    score += merge_count * 200

    # Reward higher total fruit mass
    total_size = sum(s for _, _, s in fruits)
    prev_total = sum(s for _, _, s in prev_fruits)
    score += (total_size - prev_total) * 100

    # Reward matching target sizes
    for target_str, target_count in targets:
        if target_str in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            continue
        target_size = int(target_str)
        # Count fruits of this size
        matching = sum(1 for _, _, s in fruits if s == target_size)
        # Count how many are on goals
        on_goal = 0
        for fx, fy, fs in fruits:
            if fs == target_size:
                for gx, gy in goals:
                    if gx <= fx + fs <= gx + 8 and gy <= fy + fs <= gy + 8:
                        on_goal += 1
                        break
        score += min(matching, target_count) * 100
        score += on_goal * 500

    # Reward proximity to goals
    for fx, fy, fs in fruits:
        min_dist = float('inf')
        for gx, gy in goals:
            gcx, gcy = gx + 4, gy + 4
            dist = math.sqrt((fx - gcx)**2 + (fy - gcy)**2)
            min_dist = min(min_dist, dist)
        score -= min_dist * 2

    return score


def main():
    l1_clicks = [(8,54),(13,49),(19,43),(25,37),(30,32),(36,26),(42,20),(47,15)]
    l2_clicks = [(39,38),(17,39),(15,56),(48,55),(32,38),(25,38),(23,38),(41,55),(35,55),(21,55),(27,55),(27,48),(25,43),(24,40),(28,35),(32,32),(33,28)]

    all_level_clicks = {0: l1_clicks, 1: l2_clicks}

    for level_idx in range(2, 9):
        prefix = []
        for i in range(level_idx):
            prefix.extend(all_level_clicks[i])

        print(f"\n{'='*60}")
        print(f"SOLVING Level {level_idx + 1}")
        print(f"{'='*60}")

        step_limits = [32, 32, 48, 48, 32, 32, 32, 48, 48]
        solution = solve_level_greedy(prefix, level_idx, step_limits[level_idx])

        if solution:
            all_level_clicks[level_idx] = solution
        else:
            print(f"  FAILED Level {level_idx + 1}")
            break

    # Verify full chain
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    game = create_game()
    game.reset()
    for level_idx in sorted(all_level_clicks.keys()):
        clicks = all_level_clicks[level_idx]
        for x, y in clicks:
            result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            if result.state == 'GAME_OVER':
                print(f"Level {level_idx+1}: GAME OVER")
                break
        else:
            print(f"Level {level_idx+1}: OK ({len(clicks)} clicks, lc={result.levels_completed})")
            continue
        break

    # Save
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_level_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_level_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    output = {
        "game": "su15",
        "source": "greedy_solver",
        "type": "analytical",
        "total_actions": len(all_action_ids),
        "max_level": max(all_level_clicks.keys()) + 1,
        "all_actions": all_action_ids,
    }
    for i in sorted(level_action_ids.keys()):
        output[f"l{i+1}_actions"] = level_action_ids[i]

    outpath = "experiments/results/prescriptions/su15_fullchain.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
