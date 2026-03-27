"""
SU15 solver with lightweight simulator.
Simulates vacuum/merge mechanics without the full game engine.
Uses game engine only for final verification.

MECHANICS (from source analysis):
- Click at (cx,cy): vacuum pulls fruits within radius=8
- Animation: 4 steps, each step moves fruit by min(4, remaining_dist) toward click
- After animation: check overlapping same-size fruits -> merge into next size at average pos
- Enemy types: "enemy"(speed=1), "enemy2"(speed=1), "enemy3"(speed=2)
- Enemies chase nearest fruit and can shrink/destroy fruits on collision
- Boundaries: y must be >= 10 (gnexwlqinp) and < 63 (ncfmodluov)
- Grid: 64x64

WIN CONDITION (kouxmshyjy):
- For each target (size, count) in goal_req:
  - Count fruits of that size whose CENTER falls within a goal sprite's bounds
  - Must have exactly the right count
"""

import os
import json
import math
import logging
from collections import defaultdict
from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo
from arcengine import GameAction

# Fruit sprite sizes: name -> (width, height)
FRUIT_SIZES = {
    0: (1, 1),   # "0" sprite
    1: (2, 2),   # "1" sprite
    2: (3, 3),   # "2" sprite
    3: (4, 4),   # "3" sprite
    4: (5, 5),   # "4" sprite
    5: (7, 7),   # "5" sprite
    6: (8, 8),   # "6" sprite
    7: (9, 9),   # "7" sprite
    8: (10, 10), # "8" sprite
}

VACUUM_RADIUS = 8
PULL_SPEED = 4
ANIM_STEPS = 4
Y_MIN = 10
Y_MAX = 63
GRID_W = 64
GRID_H = 64


class Fruit:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.size = size

    @property
    def w(self):
        return FRUIT_SIZES[self.size][0]

    @property
    def h(self):
        return FRUIT_SIZES[self.size][1]

    @property
    def cx(self):
        return self.x + self.w // 2

    @property
    def cy(self):
        return self.y + self.h // 2

    def overlaps(self, other):
        """Check if two sprites' bounding boxes overlap."""
        if self.x >= other.x + other.w or other.x >= self.x + self.w:
            return False
        if self.y >= other.y + other.h or other.y >= self.y + self.h:
            return False
        return True

    def clamp(self):
        """Clamp position to valid bounds."""
        self.x = max(0, min(GRID_W - self.w, self.x))
        self.y = max(Y_MIN, min(GRID_H - self.h, self.y))

    def copy(self):
        return Fruit(self.x, self.y, self.size)

    def __repr__(self):
        return f"F({self.x},{self.y},s{self.size})"


class Enemy:
    def __init__(self, x, y, speed=1):
        self.x = x
        self.y = y
        self.speed = speed

    def copy(self):
        return Enemy(self.x, self.y, self.speed)


class Goal:
    def __init__(self, x, y, w=9, h=9):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains_point(self, px, py):
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


class SU15Sim:
    """Lightweight SU15 level simulator."""

    def __init__(self, fruits, goals, goal_req, enemies=None, steps=32):
        self.fruits = [f.copy() for f in fruits]
        self.goals = goals
        self.goal_req = goal_req
        self.enemies = [e.copy() for e in (enemies or [])]
        self.steps = steps

    def copy(self):
        sim = SU15Sim(
            [f.copy() for f in self.fruits],
            self.goals,
            self.goal_req,
            [e.copy() for e in self.enemies],
            self.steps
        )
        return sim

    def click(self, cx, cy):
        """Simulate a click at (cx, cy)."""
        if self.steps <= 0:
            return False
        if cy < Y_MIN or cy >= Y_MAX:
            return True  # Click outside play area, no effect but still valid

        self.steps -= 1

        # Find fruits in range
        in_range = []
        for f in self.fruits:
            dist = math.sqrt((f.cx - cx)**2 + (f.cy - cy)**2)
            if dist <= VACUUM_RADIUS:
                in_range.append(f)

        # Animate: 4 steps of pulling
        for step in range(ANIM_STEPS):
            for f in in_range:
                dx = cx - f.cx
                dy = cy - f.cy
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    move = min(PULL_SPEED, dist)
                    nx = f.x + int(round(dx * move / dist))
                    ny = f.y + int(round(dy * move / dist))
                    # Actually the code moves the sprite position, not center
                    # Let me re-examine: the code computes (sprite_center - click) and moves sprite.x,y
                    # move_x = min(4, abs(dx)) * sign(dx), same for y
                    move_x = 0
                    move_y = 0
                    if dx > 0:
                        move_x = min(PULL_SPEED, dx)
                    elif dx < 0:
                        move_x = max(-PULL_SPEED, dx)
                    if dy > 0:
                        move_y = min(PULL_SPEED, dy)
                    elif dy < 0:
                        move_y = max(-PULL_SPEED, dy)

                    f.x += move_x
                    f.y += move_y
                    f.clamp()

            # Enemy movement during vacuum (enemies chase nearest fruit)
            for e in self.enemies:
                if not self.fruits:
                    break
                nearest = min(self.fruits, key=lambda f: (f.cx - e.x)**2 + (f.cy - e.y)**2)
                dx = 1 if nearest.cx > e.x else (-1 if nearest.cx < e.x else 0)
                dy = 1 if nearest.cy > e.y else (-1 if nearest.cy < e.y else 0)
                e.x += dx * e.speed
                e.y += dy * e.speed

        # After animation: merge check
        self._merge()

        # Check enemy-fruit collisions
        self._enemy_collisions()

        # Check win
        return True

    def _merge(self):
        """Check for overlapping same-size fruits and merge them."""
        merged = True
        while merged:
            merged = False
            n = len(self.fruits)
            for i in range(n):
                for j in range(i+1, n):
                    if self.fruits[i].size == self.fruits[j].size and self.fruits[i].overlaps(self.fruits[j]):
                        # Merge: create new fruit at average position, next size
                        f1, f2 = self.fruits[i], self.fruits[j]
                        new_size = f1.size + 1
                        if new_size > 8:
                            # Max size reached, remove both
                            self.fruits = [f for k, f in enumerate(self.fruits) if k != i and k != j]
                        else:
                            avg_cx = (f1.cx + f2.cx) // 2
                            avg_cy = (f1.cy + f2.cy) // 2
                            nw, nh = FRUIT_SIZES[new_size]
                            nx = avg_cx - nw // 2
                            ny = avg_cy - nh // 2
                            new_fruit = Fruit(nx, ny, new_size)
                            new_fruit.clamp()
                            self.fruits = [f for k, f in enumerate(self.fruits) if k != i and k != j]
                            self.fruits.append(new_fruit)
                        merged = True
                        break
                if merged:
                    break

    def _enemy_collisions(self):
        """Check enemy-fruit collisions (simplified)."""
        # In the real game, enemies shrink fruits by 1 size, size-0 get destroyed
        # For simplicity, we'll ignore enemy collisions in the sim
        # The real game engine verification will catch any issues
        pass

    def check_win(self):
        """Check if goal requirement is met."""
        if not self.goal_req:
            return False

        targets = []
        if isinstance(self.goal_req[0], (list, tuple)):
            for item in self.goal_req:
                targets.append((str(item[0]), int(item[1])))
        else:
            targets.append((str(self.goal_req[0]), int(self.goal_req[1])))

        for target_str, target_count in targets:
            if target_str in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
                # Enemy type targets - skip for now
                count = 0
                # Would need enemy-on-goal logic
                # For now assume if we get fruits right, enemies handle themselves
                continue

            target_size = int(target_str)
            count = 0
            for f in self.fruits:
                if f.size == target_size:
                    for g in self.goals:
                        if g.contains_point(f.cx, f.cy):
                            count += 1
                            break

            if count != target_count:
                return False

        return True

    def state_hash(self):
        parts = [(f.x, f.y, f.size) for f in self.fruits]
        parts.sort()
        return tuple(parts)

    def score(self):
        """Heuristic score for greedy search."""
        sc = 0

        # Parse targets
        targets = []
        if self.goal_req:
            if isinstance(self.goal_req[0], (list, tuple)):
                for item in self.goal_req:
                    targets.append((str(item[0]), int(item[1])))
            else:
                targets.append((str(self.goal_req[0]), int(self.goal_req[1])))

        # Reward: total fruit mass (higher = more merging done)
        total_mass = sum(f.size for f in self.fruits)
        sc += total_mass * 20

        # Reward: fewer fruits (= more merging)
        sc -= len(self.fruits) * 10

        # Reward: target-size fruits on goals
        for ts, tc in targets:
            if ts in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
                continue
            target_size = int(ts)
            on_goal = 0
            near_goal = 0
            for f in self.fruits:
                if f.size == target_size:
                    for g in self.goals:
                        if g.contains_point(f.cx, f.cy):
                            on_goal += 1
                            break
                        else:
                            dist = math.sqrt((f.cx - g.x - 4)**2 + (f.cy - g.y - 4)**2)
                            if dist < 20:
                                near_goal += 1
            sc += on_goal * 500
            sc += near_goal * 50
            # Having the right number of target-size fruits
            matching = sum(1 for f in self.fruits if f.size == target_size)
            sc += min(matching, tc) * 100

        # Penalty: fruit distance from nearest goal
        for f in self.fruits:
            min_dist = min(math.sqrt((f.cx - g.x - 4)**2 + (f.cy - g.y - 4)**2) for g in self.goals) if self.goals else 0
            sc -= min_dist * 2

        return sc


def get_level_data(level_idx):
    """Get level data from source code."""
    levels = [
        # Level 1
        {
            'fruits': [(3,58,2)],
            'goals': [(44,11)],
            'goal_req': [2, 1],
            'enemies': [],
            'steps': 32
        },
        # Level 2
        {
            'fruits': [(41,37,0),(18,37,0),(37,40,0),(16,41,0),(14,55,0),(16,57,0),(49,54,0),(47,56,0)],
            'goals': [(29,23)],
            'goal_req': [3, 1],
            'enemies': [],
            'steps': 32
        },
        # Level 3
        {
            'fruits': [(55,23,0),(61,23,0),(31,22,0),(31,15,0),(12,23,0),(8,28,0),
                        (46,22,1),(30,32,1),(18,16,1)],
            'goals': [(5,46),(19,46)],
            'goal_req': [[3, 1], [2, 1]],
            'enemies': [],
            'steps': 48
        },
        # Level 4
        {
            'fruits': [(5,26,0),(11,26,0),(31,27,0),(36,29,0),(33,47,0),(30,51,0),(12,47,0),(8,41,0)],
            'goals': [(1,53)],
            'goal_req': [3, 1],
            'enemies': [(52,19,1)],
            'steps': 48
        },
        # Level 5
        {
            'fruits': [(58,59,0),(44,53,0),(3,60,0),(14,54,0),
                        (14,28,1),(53,26,1),(6,25,1),(42,26,1)],
            'goals': [(28,11)],
            'goal_req': [3, 1],
            'enemies': [(4,37,1),(46,37,1)],
            'steps': 32
        },
        # Level 6 - one size-5 fruit, split by enemy+key mechanic?
        {
            'fruits': [(33,32,5)],
            'goals': [(2,12),(52,53)],
            'goal_req': [[3, 1], ["vnjbdkorwc", 1]],
            'enemies': [(16,34,1)],
            'steps': 32
        },
        # Level 7
        {
            'fruits': [(9,25,1),(20,35,1),(6,35,1),(30,37,1),(51,46,5)],
            'goals': [(19,13),(40,18)],
            'goal_req': [3, 2],
            'enemies': [(12,51,1),(52,56,1)],
            'steps': 32
        },
        # Level 8
        {
            'fruits': [(13,42,3),(3,40,3),(20,24,5)],
            'goals': [(52,15),(3,15),(52,51),(3,51)],
            'goal_req': [[4, 2], ["yckgseirmu", 1]],
            'enemies': [(43,31,1),(29,53,1),(47,48,1)],
            'steps': 48
        },
        # Level 9
        {
            'fruits': [(18,46,1),(23,52,1),(35,48,5)],
            'goals': [(7,37),(49,51),(7,51)],
            'goal_req': [[4, 1], ["vptxjilzzk", 1], [2, 1]],
            'enemies': [(51,13,1),(14,12,1),(15,22,1),(54,33,1)],
            'steps': 48
        },
    ]
    return levels[level_idx]


def create_sim(level_idx):
    """Create simulator from level data."""
    data = get_level_data(level_idx)
    fruits = [Fruit(x, y, s) for x, y, s in data['fruits']]
    goals = [Goal(x, y) for x, y in data['goals']]
    enemies = [Enemy(x, y, spd) for x, y, spd in data.get('enemies', [])]
    return SU15Sim(fruits, goals, data['goal_req'], enemies, data['steps'])


def greedy_solve(level_idx, max_attempts=3):
    """Greedy solver using lightweight sim."""
    sim = create_sim(level_idx)
    data = get_level_data(level_idx)

    print(f"  Fruits: {sim.fruits}")
    print(f"  Goals: {[(g.x,g.y) for g in sim.goals]}")
    print(f"  Goal req: {sim.goal_req}")
    print(f"  Steps: {sim.steps}")

    best_solution = None
    best_final_score = -float('inf')

    for attempt in range(max_attempts):
        solution = greedy_solve_attempt(level_idx, attempt)
        if solution is not None:
            # Verify with sim
            sim = create_sim(level_idx)
            for cx, cy in solution:
                sim.click(cx, cy)
            if sim.check_win():
                return solution
            score = sim.score()
            if score > best_final_score:
                best_final_score = score
                best_solution = solution

    return best_solution  # May not win - needs engine verification


def greedy_solve_attempt(level_idx, attempt=0):
    """Single greedy attempt."""
    sim = create_sim(level_idx)
    solution = []

    for step in range(sim.steps):
        if sim.steps <= 0:
            break

        candidates = smart_candidates_sim(sim)

        best_click = None
        best_score = -float('inf')
        best_sim = None

        for cx, cy in candidates:
            test = sim.copy()
            test.click(cx, cy)

            if test.check_win():
                solution.append((cx, cy))
                return solution

            score = test.score()
            # Add some randomization for different attempts
            if attempt > 0:
                import random
                random.seed(hash((cx, cy, step, attempt)))
                score += random.gauss(0, 10)

            if score > best_score:
                best_score = score
                best_click = (cx, cy)
                best_sim = test

        if best_click is None:
            break

        solution.append(best_click)
        sim = best_sim

        if step < 5 or (step + 1) % 5 == 0:
            print(f"  Step {step+1}: ({best_click[0]},{best_click[1]}) -> {len(sim.fruits)} fruits, score={best_score:.0f}")

    return solution


def smart_candidates_sim(sim):
    """Generate smart candidates from sim state."""
    candidates = set()

    # Between same-size pairs
    by_size = defaultdict(list)
    for f in sim.fruits:
        by_size[f.size].append(f)

    for size, fs in by_size.items():
        for i in range(len(fs)):
            for j in range(i+1, len(fs)):
                f1, f2 = fs[i], fs[j]
                dist = math.sqrt((f1.cx - f2.cx)**2 + (f1.cy - f2.cy)**2)
                if dist <= 2 * VACUUM_RADIUS:
                    mx = (f1.cx + f2.cx) // 2
                    my = (f1.cy + f2.cy) // 2
                    for dx in range(-3, 4, 3):
                        for dy in range(-3, 4, 3):
                            x, y = mx + dx, my + dy
                            if 0 <= x <= 63 and Y_MIN <= y < Y_MAX:
                                candidates.add((x, y))
                elif dist <= 3 * VACUUM_RADIUS:
                    # Pull one toward the other - click near the farther one
                    for t in [0.25, 0.5, 0.75]:
                        x = int(f1.cx + (f2.cx - f1.cx) * t)
                        y = int(f1.cy + (f2.cy - f1.cy) * t)
                        if 0 <= x <= 63 and Y_MIN <= y < Y_MAX:
                            candidates.add((x, y))

    # Near each fruit
    for f in sim.fruits:
        for dx in range(-6, 7, 3):
            for dy in range(-6, 7, 3):
                x, y = f.cx + dx, f.cy + dy
                if 0 <= x <= 63 and Y_MIN <= y < Y_MAX:
                    candidates.add((x, y))

    # Path from fruits to goals
    for f in sim.fruits:
        for g in sim.goals:
            gcx, gcy = g.x + 4, g.y + 4
            dist = math.sqrt((f.cx - gcx)**2 + (f.cy - gcy)**2)
            if dist > 0:
                for d in [4, 6, 8, 12, 16]:
                    if d > dist:
                        break
                    x = int(f.cx + (gcx - f.cx) * d / dist)
                    y = int(f.cy + (gcy - f.cy) * d / dist)
                    if 0 <= x <= 63 and Y_MIN <= y < Y_MAX:
                        candidates.add((x, y))

    # On goals
    for g in sim.goals:
        for dx in range(-4, 5, 2):
            for dy in range(-4, 5, 2):
                x, y = g.x + 4 + dx, g.y + 4 + dy
                if 0 <= x <= 63 and Y_MIN <= y < Y_MAX:
                    candidates.add((x, y))

    return sorted(candidates)


def create_engine():
    info = EnvironmentInfo(
        game_id='su15',
        local_dir='environment_files/su15/4c352900',
        class_name='Su15'
    )
    logger = logging.getLogger('su15')
    logger.setLevel(logging.WARNING)
    return LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)


def verify_with_engine(all_clicks_by_level):
    """Verify solutions with actual game engine."""
    game = create_engine()
    game.reset()

    for level_idx in sorted(all_clicks_by_level.keys()):
        clicks = all_clicks_by_level[level_idx]
        for i, (x, y) in enumerate(clicks):
            result = game.step(GameAction.ACTION6, data={'x': x, 'y': y})
            if result.state == 'GAME_OVER':
                return level_idx, i, "GAME_OVER"
            if result.levels_completed > level_idx:
                break
        else:
            if result.levels_completed <= level_idx:
                return level_idx, len(clicks), f"NOT_COMPLETE lc={result.levels_completed}"
        print(f"  L{level_idx+1}: PASS ({len(clicks)} clicks, lc={result.levels_completed})")

    return None  # All passed


def main():
    l1_clicks = [(8,54),(13,49),(19,43),(25,37),(30,32),(36,26),(42,20),(47,15)]
    l2_clicks = [(39,38),(17,39),(15,56),(48,55),(32,38),(25,38),(23,38),(41,55),(35,55),(21,55),(27,55),(27,48),(25,43),(24,40),(28,35),(32,32),(33,28)]

    all_clicks = {0: l1_clicks, 1: l2_clicks}

    for level_idx in range(2, 9):
        print(f"\n{'='*60}")
        print(f"Level {level_idx + 1}")
        print(f"{'='*60}")

        solution = greedy_solve(level_idx)
        if solution:
            all_clicks[level_idx] = solution
        else:
            print(f"  FAILED L{level_idx+1}")
            break

    # Verify with engine
    print(f"\n{'='*60}")
    print("ENGINE VERIFICATION")
    print(f"{'='*60}")

    fail = verify_with_engine(all_clicks)
    if fail:
        level_idx, click_idx, msg = fail
        print(f"  FAILED at L{level_idx+1} click {click_idx+1}: {msg}")
    else:
        print("  ALL LEVELS VERIFIED!")

    # Save
    all_action_ids = []
    level_action_ids = {}
    for i in sorted(all_clicks.keys()):
        acts = [encode_action(x, y) for x, y in all_clicks[i]]
        level_action_ids[i] = acts
        all_action_ids.extend(acts)

    output = {
        "game": "su15",
        "source": "sim_greedy_solver",
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
    print(f"Max level: {max(all_clicks.keys()) + 1}")


if __name__ == "__main__":
    main()
