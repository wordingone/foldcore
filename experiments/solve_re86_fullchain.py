"""
RE86 full chain solver — all 8 levels.
L1-L3: simple sprite positioning (no zones).
L4-L8: color-change zones + sprite positioning with zone-safe movement.

Each level: position cross/diamond/X sprites to cover target pixels.
L4+: sprites may need to be recolored by moving them through color-change zones.
"""
import sys, json, os, time, itertools
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState

STEP = 3  # movement step size
RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def get_sprite_color(sprite):
    """Get the main color of a sprite (excluding active marker 0 and transparent -1)."""
    vals = sprite.pixels[(sprite.pixels != -1) & (sprite.pixels != 0)]
    return int(vals[0]) if len(vals) > 0 else None


def is_active(sprite):
    """Check if sprite is the currently active one (center pixel = 0)."""
    h, w = sprite.height // 2, sprite.width // 2
    return int(sprite.pixels[h, w]) == 0


def get_target_colors(target_sprite):
    """Extract target pixels grouped by color."""
    tp = target_sprite.pixels
    color_targets = {}
    for r in range(tp.shape[0]):
        for c in range(tp.shape[1]):
            v = int(tp[r, c])
            if v != -1 and v != 4:
                if v not in color_targets:
                    color_targets[v] = []
                color_targets[v].append((c, r))
    return color_targets


def get_sprite_pixels(sprite):
    """Get list of (row, col) offsets for non-transparent pixels."""
    pixels = []
    for r in range(sprite.height):
        for c in range(sprite.width):
            if sprite.pixels[r, c] != -1:
                pixels.append((r, c))
    return pixels


def sprite_overlaps_zone(sprite_x, sprite_y, sprite_pixels, zone):
    """Check if sprite at given position has non-transparent pixels overlapping with zone."""
    zx, zy, zw, zh = zone.x, zone.y, zone.width, zone.height
    for r, c in sprite_pixels:
        px = sprite_x + c
        py = sprite_y + r
        if zx <= px < zx + zw and zy <= py < zy + zh:
            # Check zone pixel is also non-transparent
            zr, zc = py - zy, px - zx
            if zone.pixels[zr, zc] != -1:
                return True
    return False


def find_zone_safe_path(sx, sy, tx, ty, sprite, zones, target_color, allow_final_zone=None):
    """
    Find a sequence of moves from (sx,sy) to (tx,ty) that avoids zone collisions.
    Returns list of GameAction or None if no safe path found.

    allow_final_zone: if set, allow collision with this specific zone at the destination.
    """
    sprite_pixels = get_sprite_pixels(sprite)

    dx = tx - sx
    dy = ty - sy

    n_right = max(0, dx // STEP)
    n_left = max(0, -dx // STEP)
    n_down = max(0, dy // STEP)
    n_up = max(0, -dy // STEP)

    # Build dangerous zones list (zones with different color than target)
    danger_zones = [z for z in zones if int(z.pixels[1, 1]) != target_color]

    if not danger_zones:
        actions = []
        actions.extend([GameAction.ACTION4] * n_right)
        actions.extend([GameAction.ACTION3] * n_left)
        actions.extend([GameAction.ACTION2] * n_down)
        actions.extend([GameAction.ACTION1] * n_up)
        return actions

    def check_path(actions):
        """Check if path is zone-safe. Only checks intermediate positions, allows destination."""
        cx, cy = sx, sy
        for i, a in enumerate(actions):
            if a == GameAction.ACTION1: cy -= STEP
            elif a == GameAction.ACTION2: cy += STEP
            elif a == GameAction.ACTION3: cx -= STEP
            elif a == GameAction.ACTION4: cx += STEP

            is_final = (i == len(actions) - 1)
            for z in danger_zones:
                # At destination, allow the target zone
                if is_final and allow_final_zone and z == allow_final_zone:
                    continue
                if sprite_overlaps_zone(cx, cy, sprite_pixels, z):
                    return False
        return True

    # Try many movement orderings
    orderings = [
        [GameAction.ACTION4] * n_right + [GameAction.ACTION3] * n_left +
        [GameAction.ACTION2] * n_down + [GameAction.ACTION1] * n_up,

        [GameAction.ACTION2] * n_down + [GameAction.ACTION1] * n_up +
        [GameAction.ACTION4] * n_right + [GameAction.ACTION3] * n_left,

        [GameAction.ACTION1] * n_up + [GameAction.ACTION3] * n_left +
        [GameAction.ACTION2] * n_down + [GameAction.ACTION4] * n_right,

        [GameAction.ACTION1] * n_up + [GameAction.ACTION4] * n_right +
        [GameAction.ACTION2] * n_down + [GameAction.ACTION3] * n_left,

        [GameAction.ACTION2] * n_down + [GameAction.ACTION3] * n_left +
        [GameAction.ACTION1] * n_up + [GameAction.ACTION4] * n_right,

        [GameAction.ACTION2] * n_down + [GameAction.ACTION4] * n_right +
        [GameAction.ACTION1] * n_up + [GameAction.ACTION3] * n_left,

        [GameAction.ACTION3] * n_left + [GameAction.ACTION1] * n_up +
        [GameAction.ACTION4] * n_right + [GameAction.ACTION2] * n_down,

        [GameAction.ACTION4] * n_right + [GameAction.ACTION2] * n_down +
        [GameAction.ACTION3] * n_left + [GameAction.ACTION1] * n_up,

        [GameAction.ACTION3] * n_left + [GameAction.ACTION2] * n_down +
        [GameAction.ACTION4] * n_right + [GameAction.ACTION1] * n_up,

        [GameAction.ACTION3] * n_left + [GameAction.ACTION2] * n_down +
        [GameAction.ACTION1] * n_up + [GameAction.ACTION4] * n_right,

        [GameAction.ACTION1] * n_up + [GameAction.ACTION3] * n_left +
        [GameAction.ACTION4] * n_right + [GameAction.ACTION2] * n_down,

        [GameAction.ACTION3] * n_left + [GameAction.ACTION1] * n_up +
        [GameAction.ACTION2] * n_down + [GameAction.ACTION4] * n_right,
    ]

    for actions in orderings:
        if check_path(actions):
            return actions

    # Try interleaved moves
    h_moves = [GameAction.ACTION4] * n_right + [GameAction.ACTION3] * n_left
    v_moves = [GameAction.ACTION2] * n_down + [GameAction.ACTION1] * n_up

    for first_v in range(len(v_moves) + 1):
        actions = v_moves[:first_v] + h_moves + v_moves[first_v:]
        if check_path(actions):
            return actions

    for first_h in range(len(h_moves) + 1):
        actions = h_moves[:first_h] + v_moves + h_moves[first_h:]
        if check_path(actions):
            return actions

    # Reverse h_moves and try again
    h_moves_rev = [GameAction.ACTION3] * n_left + [GameAction.ACTION4] * n_right
    v_moves_rev = [GameAction.ACTION1] * n_up + [GameAction.ACTION2] * n_down

    for first_v in range(len(v_moves_rev) + 1):
        actions = v_moves_rev[:first_v] + h_moves_rev + v_moves_rev[first_v:]
        if check_path(actions):
            return actions

    for first_h in range(len(h_moves_rev) + 1):
        actions = h_moves_rev[:first_h] + v_moves_rev + h_moves_rev[first_h:]
        if check_path(actions):
            return actions

    print(f'    WARNING: No safe path found from ({sx},{sy}) to ({tx},{ty})!')
    return None


def find_recolor_path(sprite, zone, zones, all_movable):
    """
    Find a path to move sprite to overlap with zone (triggering recolor).
    Must avoid other zones with different colors during movement.
    The destination position CAN overlap the target zone (that's the point).
    Returns list of GameAction or None.
    """
    sprite_pixels = get_sprite_pixels(sprite)
    zone_color = int(zone.pixels[1, 1])
    sprite_color = get_sprite_color(sprite)

    # Dangerous zones: zones that are NOT our target zone AND have a different color
    danger_zones = [z for z in zones if z != zone and int(z.pixels[1, 1]) != sprite_color]

    best_path = None
    best_cost = float('inf')

    # Search positions near the zone where sprite overlaps it
    for sy in range(max(-sprite.height + 1, zone.y - sprite.height),
                     min(64, zone.y + zone.height + 1)):
        if (sy - sprite.y) % STEP != 0:
            continue
        for sx in range(max(-sprite.width + 1, zone.x - sprite.width),
                        min(64, zone.x + zone.width + 1)):
            if (sx - sprite.x) % STEP != 0:
                continue

            # Check center is in bounds
            cx = sx + sprite.width // 2
            cy = sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue

            # Check overlap with target zone
            if not sprite_overlaps_zone(sx, sy, sprite_pixels, zone):
                continue

            # Check that at this position, sprite doesn't overlap OTHER dangerous zones
            hits_danger = False
            for dz in danger_zones:
                if sprite_overlaps_zone(sx, sy, sprite_pixels, dz):
                    hits_danger = True
                    break
            if hits_danger:
                continue

            # Find safe path - allow the target zone overlap at destination
            cost = abs(sx - sprite.x) + abs(sy - sprite.y)
            if cost >= best_cost:
                continue

            # The path should avoid ALL zones except the target zone at the final step
            path = find_zone_safe_path(sprite.x, sprite.y, sx, sy, sprite, zones,
                                       sprite_color, allow_final_zone=zone)
            if path is not None:
                best_path = path
                best_cost = cost

    return best_path


def find_valid_positions(sprite, target_pts, color, step=STEP):
    """
    Find all sprite positions where all target points are covered by non-transparent pixels.
    """
    sx_cur, sy_cur = sprite.x, sprite.y
    valid = []

    for sy in range(-sprite.height + 1, 64):
        if (sy - sy_cur) % step != 0:
            continue
        for sx in range(-sprite.width + 1, 64):
            if (sx - sx_cur) % step != 0:
                continue

            cx = sx + sprite.width // 2
            cy = sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue

            all_ok = True
            for tx, ty in target_pts:
                row = ty - sy
                col = tx - sx
                if row < 0 or row >= sprite.height or col < 0 or col >= sprite.width:
                    all_ok = False
                    break
                pixel = int(sprite.pixels[row, col])
                if pixel == -1:
                    all_ok = False
                    break
                if pixel != color and pixel != 0:
                    all_ok = False
                    break

            if all_ok:
                valid.append((sx, sy))

    return valid


def solve_level_with_zones(game, env):
    """
    Solve one RE86 level, handling color-change zones.
    Returns list of GameAction values, or None.
    """
    movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
    targets = game.current_level.get_sprites_by_tag("vzuwsebntu")
    zones = game.current_level.get_sprites_by_tag("ozhohpbjxz")

    if not targets:
        return None

    target = targets[0]
    color_targets = get_target_colors(target)

    if not color_targets:
        return None

    # Determine which colors are available and which are needed
    sprite_colors = {}
    for s in movable:
        sprite_colors[s.name] = get_sprite_color(s)

    available_colors = set(sprite_colors.values()) - {None}
    needed_colors = set(color_targets.keys())
    missing_colors = needed_colors - available_colors

    actions = []

    if not zones or not missing_colors:
        # No zones needed - use simple positioning
        return solve_level_simple(game)

    print(f'    Need zone changes for colors: {missing_colors}')

    # Find zone for each missing color
    zone_map = {}  # color -> zone sprite
    for z in zones:
        zone_map[int(z.pixels[1, 1])] = z

    # Determine which sprite to recolor for each missing color
    # Strategy: assign sprites to needed colors, preferring sprites
    # that don't already match a needed color

    # First, identify sprites already matching needed colors
    already_matched = {}
    unmatched_sprites = []
    for s in movable:
        sc = sprite_colors[s.name]
        if sc in needed_colors and sc not in already_matched:
            already_matched[sc] = s
        else:
            unmatched_sprites.append(s)

    # Assign unmatched sprites to missing colors
    recolor_plan = []
    remaining_missing = list(missing_colors)
    for mc in remaining_missing:
        if mc not in zone_map:
            print(f'    ERROR: no zone for color {mc}')
            return None
        if not unmatched_sprites:
            # Need to recolor a sprite that's already matched to another color
            # Pick one that has the most alternatives
            print(f'    ERROR: no sprite available for recolor to {mc}')
            return None
        sprite_to_recolor = unmatched_sprites.pop(0)
        recolor_plan.append((sprite_to_recolor, zone_map[mc], mc))

    # Find which sprite is currently active
    active_sprite = None
    for s in movable:
        if is_active(s):
            active_sprite = s
            break

    if active_sprite is None:
        return None

    # Execute recolor plan
    for sprite_to_recolor, zone, target_color in recolor_plan:
        print(f'    Recoloring {sprite_to_recolor.name} from {sprite_colors[sprite_to_recolor.name]} to {target_color}')

        # Switch to this sprite if not active
        if sprite_to_recolor != active_sprite:
            # Calculate switches needed
            current_idx = movable.index(active_sprite)
            target_idx = movable.index(sprite_to_recolor)
            n_switches = (target_idx - current_idx) % len(movable)
            if n_switches == 0:
                n_switches = len(movable)
            for _ in range(n_switches):
                actions.append(GameAction.ACTION5)
            active_sprite = sprite_to_recolor

        # Find path to zone
        recolor_path = find_recolor_path(sprite_to_recolor, zone, zones, movable)
        if recolor_path is None:
            print(f'    ERROR: no recolor path found for {sprite_to_recolor.name}')
            return None

        actions.extend(recolor_path)
        sprite_colors[sprite_to_recolor.name] = target_color

    # Execute all recolor actions
    for a in actions:
        env.step(a)

    # Verify recoloring
    for s in movable:
        actual_color = get_sprite_color(s)
        print(f'    {s.name}: color={actual_color} at ({s.x},{s.y})')

    # Now find target positions for all sprites
    # Group target points by color
    # For each color, find which sprite(s) can cover those points

    # Update sprite colors after recoloring
    color_to_sprites = {}
    for s in movable:
        sc = get_sprite_color(s)
        if sc is not None:
            if sc not in color_to_sprites:
                color_to_sprites[sc] = []
            color_to_sprites[sc].append(s)

    # Find positions for each sprite
    sprite_positions = {}  # sprite_name -> (target_x, target_y)

    for color, target_pts in color_targets.items():
        sprites_for_color = color_to_sprites.get(color, [])
        if not sprites_for_color:
            print(f'    ERROR: no sprite for color {color} after recoloring')
            return None

        if len(sprites_for_color) == 1:
            s = sprites_for_color[0]
            valid = find_valid_positions(s, target_pts, color)
            if not valid:
                print(f'    No valid position for {s.name} color {color}')
                return None
            best = min(valid, key=lambda p: abs(p[0] - s.x) + abs(p[1] - s.y))
            sprite_positions[s.name] = (s, best)
        else:
            # Multi-sprite covering
            result = solve_multi_sprite(sprites_for_color, target_pts, color)
            if result is None:
                print(f'    Multi-sprite covering failed for color {color}')
                return None
            for s, pos in result:
                sprite_positions[s.name] = (s, pos)

    # Move sprites to target positions with zone avoidance
    # Process active sprite first, then others
    active_sprite = None
    for s in movable:
        if is_active(s):
            active_sprite = s
            break

    to_process = [s for s in movable if s.name in sprite_positions]
    active_first = [s for s in to_process if s == active_sprite]
    others = [s for s in to_process if s != active_sprite]
    to_process = active_first + others

    position_actions = []
    current_active = active_sprite

    for s in to_process:
        _, (target_x, target_y) = sprite_positions[s.name]
        target_color = get_sprite_color(s)

        # Switch to this sprite if needed
        if s != current_active:
            current_idx = movable.index(current_active)
            target_idx = movable.index(s)
            n_switches = (target_idx - current_idx) % len(movable)
            if n_switches == 0:
                n_switches = len(movable)
            for _ in range(n_switches):
                position_actions.append(GameAction.ACTION5)
            current_active = s

        # Find zone-safe path to target position
        path = find_zone_safe_path(s.x, s.y, target_x, target_y, s, zones, target_color)
        if path is None:
            print(f'    No safe path for {s.name} from ({s.x},{s.y}) to ({target_x},{target_y})')
            return None
        position_actions.extend(path)

    # Execute position actions
    for a in position_actions:
        env.step(a)

    actions.extend(position_actions)
    return actions


def solve_multi_sprite(sprites, target_pts, color):
    """Multi-sprite covering - find positions for each sprite to cover all targets."""
    target_set = set(target_pts)

    # For each sprite, find which targets each valid position covers
    sprite_coverages = []
    for s in sprites:
        coverages = []
        for pos in find_valid_positions_partial(s, target_pts, color):
            sx, sy, covered = pos
            dist = abs(sx - s.x) + abs(sy - s.y)
            coverages.append(((sx, sy), covered, dist))
        sprite_coverages.append(coverages)

    # Brute force search (sprites are usually 2-4)
    n = len(sprites)
    best = None
    best_cost = float('inf')

    MAX_POSITIONS = 200
    limited = []
    for covs in sprite_coverages:
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
        for pos, covered, dist in limited[sprite_idx]:
            new_remaining = remaining - covered
            new_cost = total_cost + dist
            if new_cost >= best_cost:
                continue
            assignments.append((sprites[sprite_idx], pos))
            search(sprite_idx + 1, new_remaining, assignments, new_cost)
            assignments.pop()
        search(sprite_idx + 1, remaining, assignments, total_cost)

    search(0, target_set, [], 0)
    return best


def find_valid_positions_partial(sprite, target_pts, color):
    """Find positions that cover at least one target point."""
    sx_cur, sy_cur = sprite.x, sprite.y
    results = []
    target_set = set(target_pts)

    for sy in range(-sprite.height + 1, 64):
        if (sy - sy_cur) % STEP != 0:
            continue
        for sx in range(-sprite.width + 1, 64):
            if (sx - sx_cur) % STEP != 0:
                continue
            cx = sx + sprite.width // 2
            cy = sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue

            covered = set()
            for tx, ty in target_set:
                row = ty - sy
                col = tx - sx
                if 0 <= row < sprite.height and 0 <= col < sprite.width:
                    pixel = int(sprite.pixels[row, col])
                    if pixel != -1 and (pixel == color or pixel == 0):
                        covered.add((tx, ty))

            if covered:
                results.append((sx, sy, covered))

    return results


def solve_level_simple(game):
    """Solve a level without zone handling (for L1-L3 type levels)."""
    movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
    targets = game.current_level.get_sprites_by_tag("vzuwsebntu")

    if not targets:
        return None

    target = targets[0]
    color_targets = get_target_colors(target)

    # Group sprites by color
    color_to_sprites = {}
    for s in movable:
        sc = get_sprite_color(s)
        if sc is not None:
            if sc not in color_to_sprites:
                color_to_sprites[sc] = []
            color_to_sprites[sc].append(s)

    # Find positions
    sprite_solutions = {}

    for color, target_pts in color_targets.items():
        sprites_for_color = color_to_sprites.get(color, [])
        if not sprites_for_color:
            print(f'    WARNING: no sprite for color {color}')
            continue

        if len(sprites_for_color) == 1:
            s = sprites_for_color[0]
            valid = find_valid_positions(s, target_pts, color)
            if not valid:
                return None
            best = min(valid, key=lambda p: abs(p[0] - s.x) + abs(p[1] - s.y))
            sprite_solutions[s.name] = (s, best, color)
        else:
            result = solve_multi_sprite(sprites_for_color, target_pts, color)
            if result is None:
                return None
            for s, pos in result:
                sprite_solutions[s.name] = (s, pos, get_sprite_color(s))

    # Build action sequence
    active_sprite = None
    for s in movable:
        if is_active(s):
            active_sprite = s
            break

    if active_sprite is None:
        return None

    to_process = [s for s in movable if s.name in sprite_solutions]
    active_first = [s for s in to_process if s == active_sprite]
    others = [s for s in to_process if s != active_sprite]
    to_process = active_first + others

    actions = []
    current_active = active_sprite

    for s in to_process:
        _, (target_x, target_y), color = sprite_solutions[s.name]

        if s != current_active:
            current_idx = movable.index(current_active)
            target_idx = movable.index(s)
            n_switches = (target_idx - current_idx) % len(movable)
            if n_switches == 0:
                n_switches = len(movable)
            for _ in range(n_switches):
                actions.append(GameAction.ACTION5)
            current_active = s

        dx = target_x - s.x
        dy = target_y - s.y

        actions.extend([GameAction.ACTION4] * (dx // STEP if dx > 0 else 0))
        actions.extend([GameAction.ACTION3] * (-dx // STEP if dx < 0 else 0))
        actions.extend([GameAction.ACTION2] * (dy // STEP if dy > 0 else 0))
        actions.extend([GameAction.ACTION1] * (-dy // STEP if dy < 0 else 0))

    return actions


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
        zones = game.current_level.get_sprites_by_tag("ozhohpbjxz")

        if zones:
            # Use zone-aware solver
            level_actions = solve_level_with_zones(game, env)
            if level_actions is None:
                print(f'  FAILED (zone solver)')
                # Try simple solver as fallback
                level_actions = solve_level_simple(game)
                if level_actions is None:
                    print(f'  FAILED (simple solver too)')
                    break
                for a in level_actions:
                    obs = env.step(a)
            # Actions already executed by zone solver
        else:
            level_actions = solve_level_simple(game)
            if level_actions is None:
                print(f'  FAILED')
                break
            for a in level_actions:
                obs = env.step(a)

        print(f'  Solution: {len(level_actions)} actions (baseline: {info.baseline_actions[level_idx]})')

        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs and obs.state == GameState.WIN:
            print(f'  GAME COMPLETE!')
            break

        print(f'  State: {obs.state.name}, score: {game._score}')

    # Verify chain
    env2 = arc.make(info.game_id)
    obs2 = env2.reset()
    for a in all_actions:
        obs2 = env2.step(a)

    final_state = obs2.state.name if obs2 else 'None'
    final_score = env2._game._score
    print(f'\nVerification: state={final_state}, score={final_score}')

    # Save
    serialized = [a.value - 1 for a in all_actions]
    result = {
        'game': 're86',
        'source': 'analytical_solver (solve_re86_fullchain)',
        'type': 'analytical',
        'total_actions': sum(per_level.values()),
        'max_level': len(per_level),
        'n_levels': n_levels,
        'per_level': per_level,
        'baseline': info.baseline_actions,
        'all_actions': serialized,
        'verified_state': final_state,
        'verified_score': final_score,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 're86_fullchain.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved: {out_path}')

    return result


if __name__ == '__main__':
    main()
