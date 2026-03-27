"""
Analytical solver for RE86 — all 8 levels.
Each level: position cross/diamond/X-shaped sprites so that
specific target pixel positions get the right colors.

L4+ add color zones: sprites change color when entering zones.
"""
import sys, json, os, time
import numpy as np
import logging
logging.disable(logging.INFO)

sys.path.insert(0, 'B:/M/the-search')
import arc_agi
from arcengine import GameAction, GameState
from itertools import permutations, combinations

STEP = 3
RESULTS_DIR = 'B:/M/the-search/experiments/results/prescriptions'


def get_sprite_color(sprite):
    vals = sprite.pixels[(sprite.pixels != -1) & (sprite.pixels != 0)]
    return int(vals[0]) if len(vals) > 0 else 0


def is_active(sprite):
    h, w = sprite.height // 2, sprite.width // 2
    return int(sprite.pixels[h, w]) == 0


def get_target_colors_and_points(game):
    targets = game.current_level.get_sprites_by_tag("vzuwsebntu")
    if not targets:
        return {}
    t = targets[0]
    tp = t.pixels
    ct = {}
    for r in range(tp.shape[0]):
        for c in range(tp.shape[1]):
            v = int(tp[r, c])
            if v != -1 and v != 4:
                ct.setdefault(v, []).append((c + t.x, r + t.y))
    return ct


def find_valid_positions(sprite, target_pts, ref_x, ref_y):
    valid = []
    for sy in range(-sprite.height + 1, 64):
        if (sy - ref_y) % STEP != 0:
            continue
        for sx in range(-sprite.width + 1, 64):
            if (sx - ref_x) % STEP != 0:
                continue
            cx, cy = sx + sprite.width // 2, sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue
            ok = True
            for tx, ty in target_pts:
                row, col = ty - sy, tx - sx
                if row < 0 or row >= sprite.height or col < 0 or col >= sprite.width:
                    ok = False; break
                if int(sprite.pixels[row, col]) == -1:
                    ok = False; break
            if ok:
                valid.append((sx, sy))
    return valid


def find_covering_positions(sprite, target_pts, ref_x, ref_y):
    """Find positions and which targets each covers."""
    results = []
    target_set = set(target_pts)
    for sy in range(-sprite.height + 1, 64):
        if (sy - ref_y) % STEP != 0:
            continue
        for sx in range(-sprite.width + 1, 64):
            if (sx - ref_x) % STEP != 0:
                continue
            cx, cy = sx + sprite.width // 2, sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue
            covered = set()
            for tx, ty in target_set:
                row, col = ty - sy, tx - sx
                if 0 <= row < sprite.height and 0 <= col < sprite.width:
                    if int(sprite.pixels[row, col]) != -1:
                        covered.add((tx, ty))
            if covered:
                dist = abs(sx - ref_x) + abs(sy - ref_y)
                results.append(((sx, sy), covered, dist))
    return results


def sprite_overlaps_zone(sx, sy, sprite, zone):
    for r in range(sprite.height):
        for c in range(sprite.width):
            if sprite.pixels[r, c] == -1:
                continue
            px, py = sx + c, sy + r
            if zone.x <= px < zone.x + zone.width and zone.y <= py < zone.y + zone.height:
                return True
    return False


def find_zone_entry(sprite, zone, all_zones):
    sx_cur, sy_cur = sprite.x, sprite.y
    best, best_dist = None, float('inf')
    for sy in range(-sprite.height + 1, 64):
        if (sy - sy_cur) % STEP != 0:
            continue
        for sx in range(-sprite.width + 1, 64):
            if (sx - sx_cur) % STEP != 0:
                continue
            cx, cy = sx + sprite.width // 2, sy + sprite.height // 2
            if cx < 0 or cy < 0 or cx >= 64 or cy >= 64:
                continue
            if not sprite_overlaps_zone(sx, sy, sprite, zone):
                continue
            if any(sprite_overlaps_zone(sx, sy, sprite, z) for z in all_zones if z is not zone):
                continue
            dist = abs(sx - sx_cur) + abs(sy - sy_cur)
            if dist < best_dist:
                best_dist = dist
                best = (sx, sy)
    return best


def gen_move(from_x, from_y, to_x, to_y):
    dx, dy = to_x - from_x, to_y - from_y
    acts = []
    if dx > 0: acts.extend([GameAction.ACTION4] * (dx // STEP))
    elif dx < 0: acts.extend([GameAction.ACTION3] * (-dx // STEP))
    if dy > 0: acts.extend([GameAction.ACTION2] * (dy // STEP))
    elif dy < 0: acts.extend([GameAction.ACTION1] * (-dy // STEP))
    return acts


def gen_move_safe(from_x, from_y, to_x, to_y, sprite, avoid_zones, vert_first=False):
    """Generate moves avoiding zones. Try both orderings."""
    dx, dy = to_x - from_x, to_y - from_y
    h_acts = ([GameAction.ACTION4] * (dx // STEP) if dx > 0 else
              [GameAction.ACTION3] * (-dx // STEP) if dx < 0 else [])
    v_acts = ([GameAction.ACTION2] * (dy // STEP) if dy > 0 else
              [GameAction.ACTION1] * (-dy // STEP) if dy < 0 else [])

    orders = [(v_acts + h_acts, 'VH'), (h_acts + v_acts, 'HV')]
    if not vert_first:
        orders = orders[::-1]

    for acts, name in orders:
        cx, cy = from_x, from_y
        safe = True
        for a in acts:
            if a == GameAction.ACTION1: cy -= STEP
            elif a == GameAction.ACTION2: cy += STEP
            elif a == GameAction.ACTION3: cx -= STEP
            elif a == GameAction.ACTION4: cx += STEP
            for z in avoid_zones:
                if sprite_overlaps_zone(cx, cy, sprite, z):
                    safe = False; break
            if not safe: break
        if safe:
            return acts, True

    # Neither safe — return vert-first as default
    return (v_acts + h_acts if vert_first else h_acts + v_acts), False


def solve_level_simple(game):
    """Solve a level where sprites already match target colors."""
    movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
    color_targets = get_target_colors_and_points(game)

    by_color = {}
    for s in movable:
        c = get_sprite_color(s)
        by_color.setdefault(c, []).append(s)

    solutions = {}
    for color, pts in color_targets.items():
        sprites_for = by_color.get(color, [])
        if not sprites_for:
            continue
        if len(sprites_for) == 1:
            s = sprites_for[0]
            valid = find_valid_positions(s, pts, s.x, s.y)
            if not valid:
                return None
            best = min(valid, key=lambda p: abs(p[0] - s.x) + abs(p[1] - s.y))
            solutions[s.name] = (s, best, color)
        else:
            result = solve_multi_cover(sprites_for, pts, color)
            if result is None:
                return None
            for s, pos in result:
                solutions[s.name] = (s, pos, color)

    active = next((s for s in movable if is_active(s)), None)
    if not active:
        return None

    ordered = ([s for s in movable if s.name in solutions and s == active] +
               [s for s in movable if s.name in solutions and s != active])

    actions, cur = [], active
    for s in ordered:
        _, (tx, ty), _ = solutions[s.name]
        if s != cur:
            ci, ti = movable.index(cur), movable.index(s)
            n_sw = (ti - ci) % len(movable) or len(movable)
            actions.extend([GameAction.ACTION5] * n_sw)
            cur = s
        actions.extend(gen_move(s.x, s.y, tx, ty))
    return actions


def solve_multi_cover(sprites_list, target_pts, color):
    """Find positions for multiple sprites to collectively cover all targets."""
    target_set = set(target_pts)
    n = len(sprites_list)
    coverages = []
    for s in sprites_list:
        covs = find_covering_positions(s, target_pts, s.x, s.y)
        # Filter to those matching color
        filtered = []
        for pos, covered, dist in covs:
            # Check pixels at covered positions have correct color
            filtered.append((pos, covered, dist))
        coverages.append(sorted(filtered, key=lambda c: (-len(c[1]), c[2]))[:200])

    best, best_cost = None, float('inf')

    def search(si, rem, asgn, cost):
        nonlocal best, best_cost
        if not rem:
            if cost < best_cost:
                best_cost = cost
                best = list(asgn)
            return
        if si >= n:
            return
        for pos, covered, dist in coverages[si]:
            nc = cost + dist
            if nc >= best_cost:
                continue
            asgn.append((sprites_list[si], pos))
            search(si + 1, rem - covered, asgn, nc)
            asgn.pop()
        search(si + 1, rem, asgn, cost)

    search(0, target_set, [], 0)
    return best


def solve_level_zones(env, game):
    """Solve a level with color zones using simulation."""
    movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
    color_targets = get_target_colors_and_points(game)
    zones = game.current_level.get_sprites_by_tag("ozhohpbjxz")
    initial_level = game.level_index

    zone_by_color = {}
    for z in zones:
        zc = int(z.pixels[1, 1]) if z.height > 1 else int(z.pixels[0, 0])
        zone_by_color.setdefault(zc, []).append(z)

    needed_colors = list(color_targets.keys())
    n_sprites = len(movable)

    # Find the best assignment: each sprite gets one target color
    # Multiple sprites can be assigned to the same color (for multi-sprite covering)
    # We need to cover ALL target points for EACH needed color

    best_plan = None
    best_cost = float('inf')

    # Generate all possible color assignments for sprites
    # Each sprite can be assigned to any needed color
    def gen_assignments(idx, current):
        if idx == n_sprites:
            # Check all needed colors are assigned at least one sprite
            assigned_colors = set(current.values())
            if set(needed_colors).issubset(assigned_colors):
                yield dict(current)
            return
        for nc in needed_colors:
            current[idx] = nc
            yield from gen_assignments(idx + 1, current)
        # Also allow NOT assigning this sprite (it stays put)
        # Only if we have enough sprites
        yield from gen_assignments(idx + 1, current)

    for assignment in gen_assignments(0, {}):
        # Group sprites by assigned color
        groups = {}
        for si, tc in assignment.items():
            groups.setdefault(tc, []).append(si)

        # Check feasibility and compute cost
        plan = []
        total_cost = 0
        feasible = True

        for tc in needed_colors:
            sprite_indices = groups.get(tc, [])
            if not sprite_indices:
                feasible = False
                break

            pts = color_targets[tc]

            if len(sprite_indices) == 1:
                si = sprite_indices[0]
                sprite = movable[si]
                if tc not in zone_by_color:
                    feasible = False
                    break

                found = False
                for zone in zone_by_color[tc]:
                    entry = find_zone_entry(sprite, zone, zones)
                    if entry is None:
                        continue
                    valid = find_valid_positions(sprite, pts, entry[0], entry[1])
                    if not valid:
                        continue
                    bt = min(valid, key=lambda p: abs(p[0] - entry[0]) + abs(p[1] - entry[1]))
                    cost = (abs(entry[0] - sprite.x) + abs(entry[1] - sprite.y) +
                            abs(bt[0] - entry[0]) + abs(bt[1] - entry[1]))
                    plan.append({'si': si, 'color': tc, 'zone': zone,
                                 'entry': entry, 'target': bt, 'multi': False})
                    total_cost += cost
                    found = True
                    break

                if not found:
                    feasible = False
                    break
            else:
                # Multi-sprite covering for this color
                sprites = [movable[si] for si in sprite_indices]

                # For each sprite, find zone entry and possible positions
                sprite_entries = []
                for si in sprite_indices:
                    sprite = movable[si]
                    if tc not in zone_by_color:
                        feasible = False
                        break
                    found_entry = False
                    for zone in zone_by_color[tc]:
                        entry = find_zone_entry(sprite, zone, zones)
                        if entry is not None:
                            sprite_entries.append((si, sprite, zone, entry))
                            found_entry = True
                            break
                    if not found_entry:
                        feasible = False
                        break

                if not feasible:
                    break

                # Find multi-sprite covering positions
                target_set = set(pts)
                coverages = []
                for si, sprite, zone, entry in sprite_entries:
                    covs = find_covering_positions(sprite, pts, entry[0], entry[1])
                    sorted_covs = sorted(covs, key=lambda c: (-len(c[1]), c[2]))[:200]
                    coverages.append(sorted_covs)

                # Search for covering
                n_s = len(sprite_entries)
                found_cover = [None]
                cover_cost = [float('inf')]

                def search_cover(idx, remaining, asgn, cost):
                    if not remaining:
                        if cost < cover_cost[0]:
                            cover_cost[0] = cost
                            found_cover[0] = list(asgn)
                        return
                    if idx >= n_s:
                        return
                    for pos, covered, dist in coverages[idx]:
                        nc = cost + dist
                        if nc >= cover_cost[0]:
                            continue
                        asgn.append((idx, pos))
                        search_cover(idx + 1, remaining - covered, asgn, nc)
                        asgn.pop()
                    search_cover(idx + 1, remaining, asgn, cost)

                search_cover(0, target_set, [], 0)

                if found_cover[0] is None:
                    feasible = False
                    break

                for cover_idx, target_pos in found_cover[0]:
                    si, sprite, zone, entry = sprite_entries[cover_idx]
                    cost = (abs(entry[0] - sprite.x) + abs(entry[1] - sprite.y) +
                            abs(target_pos[0] - entry[0]) + abs(target_pos[1] - entry[1]))
                    plan.append({'si': si, 'color': tc, 'zone': zone,
                                 'entry': entry, 'target': target_pos, 'multi': True})
                    total_cost += cost

        if feasible and total_cost < best_cost:
            best_cost = total_cost
            best_plan = list(plan)

    if best_plan is None:
        print(f'    No feasible zone assignment!')
        return None, None

    # Execute plan
    active = next((s for s in movable if is_active(s)), None)
    if not active:
        return None, None

    # Sort: process active sprite first, then others
    active_idx = movable.index(active)
    best_plan.sort(key=lambda p: (0 if p['si'] == active_idx else 1))

    all_actions = []
    cur_active = active
    obs = None

    for item in best_plan:
        sprite = movable[item['si']]
        target_color = item['color']
        target_zone = item['zone']
        entry_pos = item['entry']

        # Switch if needed
        if sprite != cur_active:
            ci, ti = movable.index(cur_active), movable.index(sprite)
            n_sw = (ti - ci) % len(movable) or len(movable)
            for _ in range(n_sw):
                obs = env.step(GameAction.ACTION5)
                all_actions.append(GameAction.ACTION5)
            cur_active = sprite

        # Move to zone entry (avoid other zones)
        avoid = [z for z in zones if z is not target_zone]
        acts, _ = gen_move_safe(sprite.x, sprite.y, entry_pos[0], entry_pos[1],
                                sprite, avoid, vert_first=True)
        for a in acts:
            obs = env.step(a)
            all_actions.append(a)

        # Handle color change animation
        if game.zrermyobpw is not None:
            for _ in range(200):
                obs = env.step(GameAction.ACTION1)
                all_actions.append(GameAction.ACTION1)
                if game.zrermyobpw is None:
                    break

        # Check color
        if get_sprite_color(sprite) != target_color:
            print(f'    Color fail: {sprite.name} is {get_sprite_color(sprite)}, wanted {target_color}')
            return None, None

        # Find safe path to target position
        # Recalculate valid positions from current position
        valid = find_valid_positions(sprite, color_targets[target_color],
                                     sprite.x, sprite.y)
        if item['multi']:
            # For multi-sprite covering, use pre-computed target
            # but verify grid alignment
            target_pos = item['target']
            if (target_pos[0] - sprite.x) % STEP != 0 or (target_pos[1] - sprite.y) % STEP != 0:
                # Grid misalignment — recalculate
                covs = find_covering_positions(sprite, color_targets[target_color],
                                                sprite.x, sprite.y)
                if not covs:
                    print(f'    No covering position after zone change!')
                    return None, None
                target_pos = min(covs, key=lambda c: c[2])[0]
        else:
            if not valid:
                print(f'    No valid pos for {sprite.name}')
                return None, None
            target_pos = min(valid, key=lambda p: abs(p[0] - sprite.x) + abs(p[1] - sprite.y))

        # Move to target, avoiding zones
        acts, safe = gen_move_safe(sprite.x, sprite.y, target_pos[0], target_pos[1],
                                    sprite, avoid, vert_first=True)
        for a in acts:
            obs = env.step(a)
            all_actions.append(a)

        # If we hit a zone during move, handle color change animation
        if game.zrermyobpw is not None:
            for _ in range(200):
                obs = env.step(GameAction.ACTION1)
                all_actions.append(GameAction.ACTION1)
                if game.zrermyobpw is None:
                    break

        # Check if level advanced
        if game.level_index != initial_level:
            break

    return all_actions, obs


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
        if game.level_index != level_idx:
            print(f'  ERROR: expected level {level_idx} but at {game.level_index}')
            break

        movable = game.current_level.get_sprites_by_tag("vfaeucgcyr")
        color_targets = get_target_colors_and_points(game)
        sprite_colors = set(get_sprite_color(s) for s in movable)
        needs_zones = not set(color_targets.keys()).issubset(sprite_colors)

        if needs_zones:
            level_actions, obs_after = solve_level_zones(env, game)
            if level_actions is not None:
                obs = obs_after
        else:
            level_actions = solve_level_simple(game)
            if level_actions is not None:
                for a in level_actions:
                    obs = env.step(a)

        if level_actions is None:
            print(f'  FAILED')
            break

        print(f'  Solution: {len(level_actions)} actions (baseline: {info.baseline_actions[level_idx]})')
        all_actions.extend(level_actions)
        per_level[f'L{level_idx + 1}'] = len(level_actions)

        if obs and obs.state == GameState.WIN:
            print(f'  GAME COMPLETE!')
            break

        state = obs.state.name if obs else 'None'
        print(f'  State: {state}, level_index: {game.level_index}')

    # Verify
    env2 = arc.make(info.game_id)
    obs2 = env2.reset()
    for a in all_actions:
        obs2 = env2.step(a)
    final_state = obs2.state.name if obs2 else 'None'
    print(f'\nVerification: state={final_state}')

    serialized = [a.value - 1 for a in all_actions]
    result = {
        'game': 're86',
        'source': 'analytical_solver',
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
