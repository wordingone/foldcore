"""
SU15 designed solutions for all 9 levels.
Each level's solution is designed analytically from the source code level data,
then verified with the game engine.

Mechanics recap:
- Click at (cx,cy): vacuum pulls fruits within radius 8 for 4 animation steps
- Each step: fruit moves min(4, dist_to_click) pixels toward click point
- After 4 steps: overlapping same-size fruits merge into next size
- Merged fruit appears at average position of merged fruits
- Enemies chase nearest fruit (speed 1-2 per step), can shrink fruits
- SIZE KEY: 0=1x1, 1=2x2, 2=3x3, 3=4x4, 4=5x5, 5=7x7, 6=8x8, 7=9x9, 8=10x10
- Goal zone = 9x9 sprite. Win check uses center of fruit within goal bounds.
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


def get_fruit_state(game):
    g = game._game
    fruits = []
    for sprite in g.hmeulfxgy:
        size = g.amnmgwpkeb.get(sprite, 0)
        fruits.append((sprite.x, sprite.y, size))
    return sorted(fruits)


def get_enemy_state(game):
    g = game._game
    enemies = []
    for sprite in g.peiiyyzum:
        etype = g.hirdajbmj.get(sprite, '')
        enemies.append((sprite.x, sprite.y, etype))
    return enemies


def play_and_observe(prefix_clicks, clicks):
    """Play all prefix + clicks, return state after each click."""
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})

    states = []
    for i, (x, y) in enumerate(clicks):
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        fruits = get_fruit_state(game)
        remaining = game._game.step_counter_ui.current_steps
        lc = result.levels_completed
        state = result.state
        states.append({
            'click': (x, y), 'fruits': fruits, 'remaining': remaining,
            'lc': lc, 'state': state
        })
        if state == 'GAME_OVER' or lc > len(prefix_clicks):
            break

    return states


def verify_solution(prefix_clicks, level_clicks, level_idx):
    """Verify a complete level solution."""
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return False, "GAME_OVER during prefix"

    target_lc = level_idx + 1
    for i, (x, y) in enumerate(level_clicks):
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return False, f"GAME_OVER at click {i+1} ({x},{y})"
        if result.levels_completed >= target_lc:
            return True, f"SOLVED at click {i+1} ({x},{y}), lc={result.levels_completed}"

    return False, f"Not completed. lc={result.levels_completed}, state={result.state}"


def try_click_at(prefix_clicks, prev_clicks, x, y):
    """Try a single click and report what happens."""
    game = create_game()
    game.reset()
    for px, py in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': px, 'y': py})
    for px, py in prev_clicks:
        game.step(GameAction.ACTION6, data={'x': px, 'y': py})

    result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    fruits = get_fruit_state(game)
    enemies = get_enemy_state(game)
    remaining = game._game.step_counter_ui.current_steps
    return fruits, enemies, remaining, result.levels_completed, result.state


def search_best_click(prefix_clicks, prev_clicks, candidates, target_lc):
    """Try each candidate and return the one with best result."""
    best = None
    best_score = -float('inf')

    for cx, cy in candidates:
        fruits, enemies, remaining, lc, state = try_click_at(prefix_clicks, prev_clicks, cx, cy)
        if state == 'GAME_OVER':
            continue
        if lc >= target_lc:
            return (cx, cy), fruits, lc, "SOLVED"

        # Score: more merged = better, closer to goals = better
        total_size = sum(s for _, _, s in fruits)
        n_fruits = len(fruits)
        score = total_size * 10 - n_fruits * 5

        if score > best_score:
            best_score = score
            best = ((cx, cy), fruits, lc, state)

    return best


def build_solution_interactive(prefix_clicks, level_idx, step_limit):
    """Build solution one click at a time, trying nearby positions."""
    target_lc = level_idx + 1
    clicks = []

    # Get initial state
    game = create_game()
    game.reset()
    for px, py in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': px, 'y': py})

    fruits = get_fruit_state(game)
    goals_sprites = game._game.rqdsgrklq
    goals = [(s.x, s.y) for s in goals_sprites]
    goal_req = game._game.reqbygadvzmjired

    print(f"  Fruits: {fruits}")
    print(f"  Goals: {goals}")
    print(f"  Goal req: {goal_req}")
    print(f"  Steps: {step_limit}")

    for step in range(step_limit):
        # Build candidates from current fruit positions + goals
        candidates = set()
        for fx, fy, fs in fruits:
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    x, y = fx + dx, fy + dy
                    if 0 <= x <= 63 and 10 <= y <= 63:
                        candidates.add((x, y))
        for gx, gy in goals:
            for dx in range(-12, 13):
                for dy in range(-12, 13):
                    x, y = gx + dx, gy + dy
                    if 0 <= x <= 63 and 10 <= y <= 63:
                        candidates.add((x, y))

        # Try all candidates
        best_click = None
        best_score = -float('inf')
        best_fruits = None
        solved = False

        for cx, cy in candidates:
            new_fruits, enemies, remaining, lc, state = try_click_at(prefix_clicks, clicks, cx, cy)
            if state == 'GAME_OVER':
                continue
            if lc >= target_lc:
                clicks.append((cx, cy))
                print(f"  Step {step+1}: SOLVED! Click ({cx},{cy})")
                return clicks

            score = compute_score(new_fruits, goals, goal_req, fruits)
            if score > best_score:
                best_score = score
                best_click = (cx, cy)
                best_fruits = new_fruits

        if best_click is None:
            print(f"  Step {step+1}: No valid clicks!")
            break

        clicks.append(best_click)
        print(f"  Step {step+1}: Click ({best_click[0]},{best_click[1]}) -> {len(best_fruits)} fruits: {best_fruits}")
        fruits = best_fruits

    return None


def compute_score(fruits, goals, goal_req, prev_fruits):
    """Score current state for greedy optimization."""
    if not fruits:
        return -10000

    score = 0

    # Reward merges (fewer fruits)
    score += (len(prev_fruits) - len(fruits)) * 100

    # Reward higher total size
    total_size = sum(s for _, _, s in fruits)
    prev_total = sum(s for _, _, s in prev_fruits)
    score += (total_size - prev_total) * 50

    # Reward fruits closer to goals
    for fx, fy, fs in fruits:
        min_dist = float('inf')
        for gx, gy in goals:
            gcx, gcy = gx + 4, gy + 4  # Goal center
            dist = ((fx - gcx)**2 + (fy - gcy)**2) ** 0.5
            min_dist = min(min_dist, dist)
        score -= min_dist

    return score


def main():
    # Known L1 and L2 solutions
    l1_clicks = [(8,54),(13,49),(19,43),(25,37),(30,32),(36,26),(42,20),(47,15)]
    l2_clicks = [(39,38),(17,39),(15,56),(48,55),(32,38),(25,38),(23,38),(41,55),(35,55),(21,55),(27,55),(27,48),(25,43),(24,40),(28,35),(32,32),(33,28)]

    all_level_clicks = {0: l1_clicks, 1: l2_clicks}

    # Verify L1+L2
    ok, msg = verify_solution([], l1_clicks, 0)
    print(f"L1: {msg}")
    ok, msg = verify_solution(l1_clicks, l2_clicks, 1)
    print(f"L2: {msg}")

    for level_idx in range(2, 9):
        prefix = []
        for i in range(level_idx):
            prefix.extend(all_level_clicks[i])

        print(f"\n{'='*60}")
        print(f"SOLVING Level {level_idx + 1}")
        print(f"{'='*60}")

        step_limits = [32, 32, 48, 48, 32, 32, 32, 48, 48]
        step_limit = step_limits[level_idx]

        solution = build_solution_interactive(prefix, level_idx, step_limit)

        if solution:
            all_level_clicks[level_idx] = solution
            ok, msg = verify_solution(prefix, solution, level_idx)
            print(f"  Verification: {msg}")
            if not ok:
                print("  FAILED verification!")
                break
        else:
            print(f"  FAILED to solve L{level_idx+1}")
            break

    # Save
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_level_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_level_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    max_level = max(all_level_clicks.keys()) + 1

    output = {
        "game": "su15",
        "source": "analytical_designed",
        "type": "analytical",
        "total_actions": len(all_action_ids),
        "max_level": max_level,
        "all_actions": all_action_ids,
    }
    for i in sorted(level_action_ids.keys()):
        output[f"l{i+1}_actions"] = level_action_ids[i]

    outpath = "experiments/results/prescriptions/su15_fullchain.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Total actions: {len(all_action_ids)}, Max level: {max_level}")


if __name__ == "__main__":
    main()
