"""
Analytical solver for RE86 — all 8 levels.
Each level: position cross/diamond/X-shaped sprites so that
specific target pixel positions get the right colors.
"""
import sys, json, os, time
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState

STEP = 3  # ilmaurgzng = 3
RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def solve_level(game):
    """
    Solve one RE86 level analytically.
    Handles both single-color-per-sprite and multi-sprite same-color cases.
    Returns list of GameAction values, or None.
    """
    movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
    targets = game.current_level.get_sprites_by_tag("vzuwsebntu")

    if not targets:
        return None

    target = targets[0]
    tp = target.pixels

    # Find target positions grouped by color
    color_targets = {}
    for r in range(tp.shape[0]):
        for c in range(tp.shape[1]):
            v = int(tp[r, c])
            if v != -1 and v != 4:
                if v not in color_targets:
                    color_targets[v] = []
                color_targets[v].append((c, r))

    # Group sprites by color
    color_to_sprites = {}
    for s in movable:
        unique = np.unique(s.pixels[s.pixels != -1])
        sprite_color = int([c for c in unique if c != 0][0]) if len([c for c in unique if c != 0]) > 0 else None
        if sprite_color is not None:
            if sprite_color not in color_to_sprites:
                color_to_sprites[sprite_color] = []
            color_to_sprites[sprite_color].append(s)

    # For each color, find sprite positions that collectively cover all targets
    sprite_solutions = {}  # sprite -> (target_x, target_y)

    for color, target_pts in color_targets.items():
        sprites_for_color = color_to_sprites.get(color, [])
        if not sprites_for_color:
            print(f'    WARNING: no sprite for color {color}')
            continue

        if len(sprites_for_color) == 1:
            # Single sprite covers all targets
            s = sprites_for_color[0]
            valid = find_valid_position(s, target_pts, color)
            if not valid:
                print(f'    No single valid position for color {color}!')
                return None
            best = min(valid, key=lambda p: abs(p[0] - s.x) + abs(p[1] - s.y))
            sprite_solutions[s.name] = (s, best, color)
        else:
            # Multiple sprites need to collectively cover all targets
            result = solve_multi_sprite_covering(sprites_for_color, target_pts, color)
            if result is None:
                print(f'    Multi-sprite covering failed for color {color}!')
                return None
            for s, pos in result:
                sprite_solutions[s.name] = (s, pos, color)

    # Build action sequence
    active_sprite = None
    for s in movable:
        h = s.height // 2
        w = s.width // 2
        if s.pixels[h, w] == 0:
            active_sprite = s
            break

    if active_sprite is None:
        return None

    # Process sprites: active first, then others in order
    to_process = [s for s in movable if s.name in sprite_solutions]
    active_first = [s for s in to_process if s == active_sprite]
    others = [s for s in to_process if s != active_sprite]
    to_process = active_first + others

    actions = []
    current_active = active_sprite

    for s in to_process:
        _, (target_x, target_y), color = sprite_solutions[s.name]

        # Switch to this sprite if not active
        if s != current_active:
            current_idx = movable.index(current_active)
            target_idx = movable.index(s)
            n_switches = (target_idx - current_idx) % len(movable)
            if n_switches == 0:
                n_switches = len(movable)
            for _ in range(n_switches):
                actions.append(GameAction.ACTION5)
            current_active = s

        # Move to target position
        dx = target_x - s.x
        dy = target_y - s.y

        n_right = dx // STEP if dx > 0 else 0
        n_left = -dx // STEP if dx < 0 else 0
        n_down = dy // STEP if dy > 0 else 0
        n_up = -dy // STEP if dy < 0 else 0

        actions.extend([GameAction.ACTION4] * n_right)
        actions.extend([GameAction.ACTION3] * n_left)
        actions.extend([GameAction.ACTION2] * n_down)
        actions.extend([GameAction.ACTION1] * n_up)

    return actions


def solve_multi_sprite_covering(sprites, target_pts, color):
    """
    Given multiple sprites of the same color and target points,
    find positions for each sprite that collectively cover all targets.
    Returns list of (sprite, (sx, sy)) or None.
    """
    target_set = set(target_pts)
    n_targets = len(target_set)
    n_sprites = len(sprites)

    # For each sprite, find which targets each valid position covers
    sprite_coverages = []
    for s in sprites:
        coverages = []  # list of (position, set_of_covered_targets)
        sx_cur, sy_cur = s.x, s.y

        for sy in range(-s.height + 1, 64):
            if (sy - sy_cur) % STEP != 0:
                continue
            for sx in range(-s.width + 1, 64):
                if (sx - sx_cur) % STEP != 0:
                    continue

                cx = sx + s.width // 2
                cy = sy + s.height // 2
                if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                    continue

                covered = set()
                for tx, ty in target_set:
                    row = ty - sy
                    col = tx - sx
                    if 0 <= row < s.height and 0 <= col < s.width:
                        pixel = int(s.pixels[row, col])
                        if pixel != -1 and (pixel == color or pixel == 0):
                            covered.add((tx, ty))

                if covered:
                    dist = abs(sx - sx_cur) + abs(sy - sy_cur)
                    coverages.append(((sx, sy), covered, dist))

        sprite_coverages.append(coverages)

    print(f'    Multi-sprite covering: {n_sprites} sprites, {n_targets} targets')
    for i, covs in enumerate(sprite_coverages):
        max_cover = max(len(c[1]) for c in covs) if covs else 0
        print(f'      Sprite {i} ({sprites[i].name}): {len(covs)} positions, max_cover={max_cover}')

    # Greedy set cover with backtracking
    # For small n_sprites (2-4), try all combinations
    if n_sprites <= 4:
        return brute_force_cover(sprites, sprite_coverages, target_set)

    return None


def brute_force_cover(sprites, sprite_coverages, target_set):
    """
    Try all combinations of sprite positions to find minimum-cost covering.
    """
    n = len(sprites)
    best = None
    best_cost = float('inf')

    # For efficiency, limit each sprite's candidate positions
    # Keep only positions that cover at least 1 uncovered target
    MAX_POSITIONS = 200

    limited = []
    for covs in sprite_coverages:
        # Sort by coverage (desc) then distance (asc)
        sorted_covs = sorted(covs, key=lambda c: (-len(c[1]), c[2]))[:MAX_POSITIONS]
        limited.append(sorted_covs)

    def search(sprite_idx, remaining, assignments, total_cost):
        nonlocal best, best_cost

        if not remaining:
            if total_cost < best_cost:
                best_cost = total_cost
                best = list(assignments)
            return

        if sprite_idx >= n:
            return

        # Try each position for this sprite
        for pos, covered, dist in limited[sprite_idx]:
            new_remaining = remaining - covered
            new_cost = total_cost + dist
            if new_cost >= best_cost:
                continue
            assignments.append((sprites[sprite_idx], pos))
            search(sprite_idx + 1, new_remaining, assignments, new_cost)
            assignments.pop()

        # Also try skipping this sprite (it stays in place)
        search(sprite_idx + 1, remaining, assignments, total_cost)

    search(0, target_set, [], 0)

    return best


def find_valid_position(sprite, target_pts, color):
    """
    Find all sprite positions where all target points fall on non-transparent, non-zero pixels.
    Positions must be reachable (aligned to step size 3 from current position).
    """
    sx_cur, sy_cur = sprite.x, sprite.y
    valid = []

    # Search space: sprite must be positioned so all targets are inside
    # Canvas is 64x64, sprite can go off-edge (center must be in bounds)

    for sy in range(-sprite.height + 1, 64):
        # Check alignment with step grid
        if (sy - sy_cur) % STEP != 0:
            continue

        for sx in range(-sprite.width + 1, 64):
            if (sx - sx_cur) % STEP != 0:
                continue

            # Check center is in bounds
            cx = sx + sprite.width // 2
            cy = sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue

            # Check all target points
            all_ok = True
            for tx, ty in target_pts:
                row = ty - sy
                col = tx - sx
                if row < 0 or row >= sprite.height or col < 0 or col >= sprite.width:
                    all_ok = False
                    break
                pixel = int(sprite.pixels[row, col])
                if pixel == -1:  # transparent
                    all_ok = False
                    break
                # Pixel should be the right color or the active marker (0)
                if pixel != color and pixel != 0:
                    all_ok = False
                    break

            if all_ok:
                valid.append((sx, sy))

    return valid


def main():
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    info = next(g for g in games if 're86' in g.game_id.lower())

    env = arc.make(info.game_id)
    obs = env.reset()
    n_levels = len(info.baseline_actions)

    print(f'SOLVING RE86 ({n_levels} levels, baseline={info.baseline_actions})')

    all_actions = []
    per_level = {}

    for level_idx in range(n_levels):
        print(f'\n--- Level {level_idx + 1}/{n_levels} ---')

        game = env._game
        level_actions = solve_level(game)

        if level_actions is None:
            print(f'  FAILED')
            break

        print(f'  Solution: {len(level_actions)} actions (baseline: {info.baseline_actions[level_idx]})')

        # Execute
        for a in level_actions:
            obs = env.step(a)

        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs and obs.state == GameState.WIN:
            print(f'  GAME COMPLETE!')
            break

        print(f'  State: {obs.state.name}')

    # Verify chain
    env2 = arc.make(info.game_id)
    obs2 = env2.reset()
    for a in all_actions:
        obs2 = env2.step(a)

    final_state = obs2.state.name if obs2 else 'None'
    print(f'\nVerification: state={final_state}')

    # Save
    serialized = [a.value - 1 for a in all_actions]
    result = {
        'game': 're86',
        'source': 'analytical_solver (solve_re86_analytical)',
        'type': 'analytical',
        'total_actions': sum(per_level.values()),
        'max_level': len(per_level),
        'n_levels': n_levels,
        'per_level': per_level,
        'baseline': info.baseline_actions,
        'all_actions': serialized,
    }

    out_path = os.path.join(RESULTS_DIR, 're86_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')

    return result


if __name__ == '__main__':
    main()
