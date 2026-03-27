"""
SU15 beam search solver using game engine.
Key optimization: don't replay from scratch each time.
Instead, save/restore game state using the ACTION7 (undo) mechanism
or just accept the replay cost with a focused beam.

Strategy:
1. Use a very tight beam (width=3-5)
2. Smart candidate selection (~20-30 positions per step)
3. Focus on the merge plan: merge pairs first, then move to goals
"""

import os
import json
import math
import logging
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
    """Get fruit state from game."""
    g = game._game
    fruits = []
    for sprite in g.hmeulfxgy:
        size = g.amnmgwpkeb.get(sprite, 0)
        fruits.append((sprite.x, sprite.y, size))
    remaining = g.step_counter_ui.current_steps
    goals = [(s.x, s.y) for s in g.rqdsgrklq]
    goal_req = g.reqbygadvzmjired
    return sorted(fruits), remaining, goals, goal_req


def replay_and_try(prefix_clicks, extra_clicks, test_click):
    """Replay full game state and try a click. Returns (fruits, remaining, lc, state)."""
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    for x, y in extra_clicks:
        result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
        if result.state == 'GAME_OVER':
            return None, 0, -1, 'GAME_OVER'

    result = game.step(GameAction.ACTION6, data={'x': test_click[0], 'y': test_click[1]})
    if result.state == 'GAME_OVER':
        return None, 0, result.levels_completed, 'GAME_OVER'

    fruits, remaining, goals, goal_req = get_state(game)
    return fruits, remaining, result.levels_completed, result.state


def smart_candidates(fruits, goals):
    """Generate ~30-50 smart click candidates."""
    candidates = set()

    by_size = defaultdict(list)
    for fx, fy, fs in fruits:
        by_size[fs].append((fx, fy))

    # 1. Between same-size pairs (merging)
    for size, positions in by_size.items():
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                x1, y1 = positions[i]
                x2, y2 = positions[j]
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                candidates.add((mx, my))
                # Also try near each fruit of the pair
                candidates.add((x1, y1))
                candidates.add((x2, y2))

    # 2. Path from fruit to nearest goal
    for fx, fy, fs in fruits:
        for gx, gy in goals:
            gcx, gcy = gx + 4, gy + 4
            dx, dy = gcx - fx, gcy - fy
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 0:
                for d in [4, 6, 8]:
                    if d < dist:
                        x = int(fx + dx * d / dist)
                        y = int(fy + dy * d / dist)
                        if 0 <= x <= 63 and 10 <= y < 63:
                            candidates.add((x, y))

    # 3. On each goal center
    for gx, gy in goals:
        candidates.add((gx + 4, gy + 4))
        candidates.add((gx + 2, gy + 2))
        candidates.add((gx + 6, gy + 6))

    # 4. Near each fruit
    for fx, fy, fs in fruits:
        for dx in [-4, 0, 4]:
            for dy in [-4, 0, 4]:
                x, y = fx + dx, fy + dy
                if 0 <= x <= 63 and 10 <= y < 63:
                    candidates.add((x, y))

    return sorted(candidates)


def score_state(fruits, goals, goal_req, prev_n_fruits):
    """Score a state for beam search ranking."""
    if not fruits:
        return -10000

    score = 0

    # Merges happened
    score += (prev_n_fruits - len(fruits)) * 300

    # Total mass
    total = sum(s for _, _, s in fruits)
    score += total * 30

    # Fewer fruits
    score -= len(fruits) * 15

    # Parse targets
    targets = []
    if goal_req:
        if isinstance(goal_req[0], (list, tuple)):
            for item in goal_req:
                targets.append((str(item[0]), int(item[1])))
        else:
            targets.append((str(goal_req[0]), int(goal_req[1])))

    for ts, tc in targets:
        if ts in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            continue
        target_size = int(ts)
        matching = sum(1 for _, _, s in fruits if s == target_size)
        score += min(matching, tc) * 100

        # On goal
        for fx, fy, fs in fruits:
            if fs == target_size:
                for gx, gy in goals:
                    if gx <= fx <= gx + 8 and gy <= fy <= gy + 8:
                        score += 600

    # Distance to goals
    for fx, fy, fs in fruits:
        min_d = min(math.sqrt((fx - gx - 4)**2 + (fy - gy - 4)**2) for gx, gy in goals)
        score -= min_d * 2

    return score


def solve_level_beam(prefix_clicks, level_idx, beam_width=3, step_limit=None):
    """Beam search: keep top beam_width partial solutions at each step."""
    if step_limit is None:
        step_limits = [32, 32, 48, 48, 32, 32, 32, 48, 48]
        step_limit = step_limits[level_idx]

    target_lc = level_idx + 1

    # Get initial state
    game = create_game()
    game.reset()
    for x, y in prefix_clicks:
        game.step(GameAction.ACTION6, data={'x': x, 'y': y})
    fruits, remaining, goals, goal_req = get_state(game)

    print(f"  Fruits: {fruits}")
    print(f"  Goals: {goals}")
    print(f"  Goal req: {goal_req}")
    print(f"  Steps: {remaining}")
    print(f"  Beam width: {beam_width}")

    # Initial beam: single empty path
    beam = [(0, [], fruits)]  # (neg_score, clicks, last_known_fruits)

    for step in range(step_limit):
        new_beam = []

        for neg_score, clicks, last_fruits in beam:
            candidates = smart_candidates(last_fruits, goals)

            for cx, cy in candidates:
                new_fruits, new_remaining, lc, state = replay_and_try(prefix_clicks, clicks, (cx, cy))

                if state == 'GAME_OVER' or new_fruits is None:
                    continue

                if lc >= target_lc:
                    final_clicks = clicks + [(cx, cy)]
                    print(f"  SOLVED at step {step+1} ({len(final_clicks)} clicks)")
                    return final_clicks

                s = score_state(new_fruits, goals, goal_req, len(last_fruits))
                new_beam.append((-s, clicks + [(cx, cy)], new_fruits))

        if not new_beam:
            print(f"  Beam empty at step {step+1}")
            break

        # Keep top beam_width
        new_beam.sort()
        beam = new_beam[:beam_width]

        # Report best
        best_score = -beam[0][0]
        best_fruits = beam[0][2]
        best_click = beam[0][1][-1] if beam[0][1] else None
        if best_click:
            print(f"  Step {step+1}: best click ({best_click[0]},{best_click[1]}) -> {len(best_fruits)} fruits, score={best_score:.0f}, beam={len(beam)}")

    return None


def main():
    l1_clicks = [(8,54),(13,49),(19,43),(25,37),(30,32),(36,26),(42,20),(47,15)]
    l2_clicks = [(39,38),(17,39),(15,56),(48,55),(32,38),(25,38),(23,38),(41,55),(35,55),(21,55),(27,55),(27,48),(25,43),(24,40),(28,35),(32,32),(33,28)]

    all_clicks = {0: l1_clicks, 1: l2_clicks}

    for level_idx in range(2, 9):
        prefix = []
        for i in range(level_idx):
            prefix.extend(all_clicks[i])

        print(f"\n{'='*60}")
        print(f"Level {level_idx + 1} (prefix={len(prefix)} clicks)")
        print(f"{'='*60}")

        # Try increasing beam width
        for bw in [3, 5, 10]:
            print(f"\n  Trying beam_width={bw}")
            solution = solve_level_beam(prefix, level_idx, beam_width=bw)
            if solution:
                all_clicks[level_idx] = solution
                break
        else:
            print(f"  FAILED Level {level_idx + 1}")
            break

    # Verify
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")
    game = create_game()
    game.reset()
    for level_idx in sorted(all_clicks.keys()):
        clicks = all_clicks[level_idx]
        for i, (x, y) in enumerate(clicks):
            result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            if result.state == 'GAME_OVER':
                print(f"L{level_idx+1}: GAME OVER at click {i+1}")
                break
        else:
            if result.levels_completed > level_idx:
                print(f"L{level_idx+1}: PASS ({len(clicks)} clicks)")
            else:
                print(f"L{level_idx+1}: NOT COMPLETE (lc={result.levels_completed})")
            continue
        break

    # Save
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    output = {
        "game": "su15",
        "source": "beam_search_solver",
        "type": "analytical",
        "total_actions": len(all_action_ids),
        "max_level": max(all_clicks.keys()) + 1,
        "all_actions": all_action_ids,
    }
    for i in sorted(level_action_ids.keys()):
        output[f"l{i+1}_actions"] = level_action_ids[i]

    outpath = "experiments/results/prescriptions/su15_fullchain.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")
    print(f"Max level solved: {max(all_clicks.keys()) + 1}")


if __name__ == "__main__":
    main()
