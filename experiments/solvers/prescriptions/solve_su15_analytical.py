"""
SU15 analytical solver for levels 3-9.
Designs click sequences based on fruit positions, merge requirements, and goal locations.
Tests each solution with the actual game engine.

Mechanics:
- Click at (cx,cy): creates vacuum that pulls all fruits within radius 8
- Pull: each of 4 animation steps moves fruit by min(4, distance) toward click point
- After 4 steps: check for overlapping same-size fruits -> merge into next size
- Fruit sizes: 0(1px) -> 1(2px) -> 2(3px) -> 3(4px) -> 4(5px) -> 5(7px) -> 6(8px) -> 7(9px) -> 8(10px)
- Two sprites overlap if their bounding boxes share any pixel
- Win: specific fruit sizes on goal zones

Enemies:
- Chase nearest fruit, can destroy size-0 fruits on contact
- Need to account for enemy interference

Key dimensions:
- Sprite sizes: size-N is (N+1)x(N+1) pixels, except: 0=1, 1=2, 2=3, 3=4, 4=5, 5=7, 6=8, 7=9, 8=10
- Goal sprite (avvxfurrqu): 9x9 circle
- Vacuum radius: 8 pixels
- Pull speed: 4 pixels/step
- Animation steps: 4 per click
- Max pull distance per click: 4*4 = 16 pixels (but each step is capped at min(4, remaining_dist))
"""

import os
import json
import logging
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


def play_clicks(game, clicks):
    """Play a sequence of (x,y) clicks, return final result."""
    for x, y in clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return result, False
    return result, True


def get_fruit_state(game):
    """Get current fruit positions and sizes."""
    g = game._game
    fruits = []
    for sprite in g.hmeulfxgy:
        size = g.amnmgwpkeb.get(sprite, 0)
        fruits.append((sprite.x, sprite.y, size))
    return sorted(fruits)


def get_goal_state(game):
    """Get goal positions."""
    g = game._game
    goals = []
    for sprite in g.rqdsgrklq:
        goals.append((sprite.x, sprite.y))
    return goals


def test_solution(prefix_clicks, level_clicks, level_idx):
    """Test a solution by replaying from scratch."""
    game = create_game()
    game.reset()

    # Play prefix
    for x, y in prefix_clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return False, "GAME_OVER during prefix"

    target_lc = level_idx + 1

    # Play level solution
    for i, (x, y) in enumerate(level_clicks):
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return False, f"GAME_OVER at click {i+1} ({x},{y})"
        if result.levels_completed >= target_lc:
            return True, f"Solved after {i+1} clicks!"

    return False, f"Level not completed after {len(level_clicks)} clicks. lc={result.levels_completed}"


def observe_after_click(prefix_clicks, level_clicks_so_far, click):
    """Play all clicks up to and including 'click', return fruit state."""
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    for x, y in level_clicks_so_far:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    if click:
        game.step(GameAction.ACTION6, data={'x': click[0], 'y': click[1]})
    fruits = get_fruit_state(game)
    goals = get_goal_state(game)
    g = game._game
    remaining = g.step_counter_ui.current_steps
    return fruits, goals, remaining


def iterative_solve(prefix_clicks, level_idx, max_clicks):
    """
    Iteratively build a solution by trying clicks and observing results.
    Greedy: at each step, try all candidate positions and pick the one that
    makes the most progress toward the goal.
    """
    target_lc = level_idx + 1

    # Get initial state
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})

    initial_fruits = get_fruit_state(game)
    goals = get_goal_state(game)
    g = game._game
    goal_req = g.reqbygadvzmjired

    print(f"  Initial fruits: {initial_fruits}")
    print(f"  Goals at: {goals}")
    print(f"  Goal requirement: {goal_req}")
    print(f"  Steps available: {g.step_counter_ui.current_steps}")

    # Build candidate click positions
    candidates = set()
    for fx, fy, fs in initial_fruits:
        for dx in range(-12, 13, 1):
            for dy in range(-12, 13, 1):
                x, y = fx + dx, fy + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    candidates.add((x, y))
    for gx, gy in goals:
        for dx in range(-16, 17, 1):
            for dy in range(-16, 17, 1):
                x, y = gx + dx, gy + dy
                if 0 <= x <= 63 and 10 <= y <= 63:
                    candidates.add((x, y))

    # Reduce to grid for tractability
    coarse_candidates = set()
    for x, y in candidates:
        if x % 2 == 0 and y % 2 == 0:
            coarse_candidates.add((x, y))
    candidates = sorted(coarse_candidates)

    print(f"  Candidate positions: {len(candidates)}")

    solution = []
    for click_num in range(max_clicks):
        game = create_game()
        game.reset()
        for x, y in prefix_clicks:
            game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        for x, y in solution:
            game.step(GameAction.ACTION6, data={'x': x, 'y': y})

        current_fruits = get_fruit_state(game)
        remaining = game._game.step_counter_ui.current_steps

        if remaining <= 0:
            print(f"  Out of steps after {click_num} clicks")
            break

        # Try each candidate and score the result
        best_score = -float('inf')
        best_click = None
        best_fruits = None

        for cx, cy in candidates:
            test_game = create_game()
            test_game.reset()
            for x, y in prefix_clicks:
                test_game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            for x, y in solution:
                test_game.step(GameAction.ACTION6, data={'x': x, 'y': y})

            result = test_game.step(GameAction.ACTION6, data={'x': cx, 'y': cy})

            if result.state == 'GAME_OVER':
                continue

            if result.levels_completed >= target_lc:
                solution.append((cx, cy))
                print(f"  SOLVED at click {click_num + 1}! Click ({cx},{cy})")
                return solution

            # Score: reward merges (fewer fruits of same size = good),
            # reward fruits near goals, penalize fruits far from goals
            new_fruits = get_fruit_state(test_game)
            score = score_state(new_fruits, goals, goal_req)

            if score > best_score:
                best_score = score
                best_click = (cx, cy)
                best_fruits = new_fruits

        if best_click is None:
            print(f"  No valid clicks remaining at step {click_num + 1}")
            break

        solution.append(best_click)
        print(f"  Click {click_num + 1}: ({best_click[0]},{best_click[1]}) -> fruits={best_fruits}, score={best_score:.2f}")

        # Update candidates based on new fruit positions
        if best_fruits:
            new_candidates = set()
            for fx, fy, fs in best_fruits:
                for dx in range(-12, 13, 2):
                    for dy in range(-12, 13, 2):
                        x, y = fx + dx, fy + dy
                        if 0 <= x <= 63 and 10 <= y <= 63:
                            new_candidates.add((x, y))
            for gx, gy in goals:
                for dx in range(-16, 17, 2):
                    for dy in range(-16, 17, 2):
                        x, y = gx + dx, gy + dy
                        if 0 <= x <= 63 and 10 <= y <= 63:
                            new_candidates.add((x, y))
            candidates = sorted(new_candidates)

    return None


def score_state(fruits, goals, goal_req):
    """Score a fruit state based on progress toward the goal requirement."""
    if not fruits:
        return -1000  # All fruits destroyed = bad

    # Parse goal requirement
    # goal_req can be: [size, count] or [[size, count], [size, count], ...]
    if not goal_req:
        return 0

    targets = []
    if isinstance(goal_req[0], (list, tuple)):
        for item in goal_req:
            targets.append((str(item[0]), int(item[1])))
    else:
        targets.append((str(goal_req[0]), int(goal_req[1])))

    score = 0

    # Reward: total fruit "mass" (higher sizes are better, means merges happened)
    for fx, fy, fs in fruits:
        score += fs * 10  # Reward higher sizes

    # Reward: fewer total fruits (means more merging)
    score -= len(fruits) * 5

    # Reward: fruits close to any goal
    for fx, fy, fs in fruits:
        min_dist = float('inf')
        for gx, gy in goals:
            # Goal is 9x9, center at (gx+4, gy+4)
            dist = abs(fx - gx - 4) + abs(fy - gy - 4)
            min_dist = min(min_dist, dist)
        score -= min_dist * 0.5  # Penalize distance from goals

    # Big reward: if a fruit of target size is on a goal
    for target_size_str, target_count in targets:
        # Check for enemy types
        if target_size_str in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            continue  # Enemy goals handled differently
        target_size = int(target_size_str)
        matching = sum(1 for fx, fy, fs in fruits if fs == target_size)
        on_goal = 0
        for fx, fy, fs in fruits:
            if fs == target_size:
                for gx, gy in goals:
                    # Check if fruit center is within goal bounds
                    if gx <= fx <= gx + 8 and gy <= fy <= gy + 8:
                        on_goal += 1
                        break
        # Reward having the right size fruits
        score += min(matching, target_count) * 50
        # Extra reward for fruits on goals
        score += on_goal * 200

    return score


def main():
    # Known L1 and L2 solutions
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

        # Get step limit from level data
        step_limits = [32, 32, 48, 48, 32, 32, 32, 48, 48]
        max_clicks = step_limits[level_idx]

        solution = iterative_solve(prefix, level_idx, max_clicks)

        if solution is not None:
            all_level_clicks[level_idx] = solution
            # Verify
            ok, msg = test_solution(prefix, solution, level_idx)
            print(f"  Verification: {msg}")
            if not ok:
                print(f"  WARNING: Solution did not verify!")
        else:
            print(f"  FAILED to solve level {level_idx + 1}")
            # Try to continue with a manual/designed solution
            manual = design_manual_solution(prefix, level_idx)
            if manual:
                all_level_clicks[level_idx] = manual
                ok, msg = test_solution(prefix, manual, level_idx)
                print(f"  Manual verification: {msg}")
            else:
                break

    # Save results
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_level_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_level_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    output = {
        "game": "su15",
        "source": "analytical_solver",
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
    print(f"Total actions: {len(all_action_ids)}")


def design_manual_solution(prefix_clicks, level_idx):
    """Design manual solutions for levels that greedy can't solve."""
    return None


if __name__ == "__main__":
    main()
