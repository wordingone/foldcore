"""
Offline solver for FT09 level solutions.
Computes the minimum clicks to satisfy cgj() for each level.
Reads sprite data directly from ft09.py.
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/ft09/0d8bbf25')
from ft09 import sprites as SPRITES, levels as LEVELS

# cgj neighbor offsets: (dx, dy) for pixels[row][col]
NEIGHBOR_OFFSETS = {
    (0, 0): (-4, -4),
    (0, 1): (0, -4),
    (0, 2): (4, -4),
    (1, 0): (-4, 0),
    (1, 2): (4, 0),
    (2, 0): (-4, 4),
    (2, 1): (0, 4),
    (2, 2): (4, 4),
}


def solve_level(level_idx):
    level = LEVELS[level_idx]
    name = level.name
    gqb = level.get_data("cwU") or [9, 8]
    gqb = [int(c) for c in gqb]
    initial_color = gqb[0]
    nRq = 0  # placeholder, set per sprite below

    # Map of wall positions to their sprites
    hkx_walls = {}  # (x,y) -> Hkx/NTi sprite
    for sp in level.get_sprites_by_tag("Hkx"):
        hkx_walls[(sp.x, sp.y)] = sp
    for sp in level.get_sprites_by_tag("NTi"):
        hkx_walls[(sp.x, sp.y)] = sp

    # bsT sprites (non-wall)
    all_bst = level.get_sprites_by_tag("bsT")
    bst_sprites = [sp for sp in all_bst if "Hkx" not in sp.tags and "NTi" not in sp.tags]

    # For each wall, determine required color
    required = {}  # (x,y) -> required_color (or "any_not_X")
    conflicts = []

    for etf in bst_sprites:
        nRq = int(etf.pixels[1][1])
        for (row, col), (dx, dy) in NEIGHBOR_OFFSETS.items():
            nx, ny = etf.x + dx, etf.y + dy
            if (nx, ny) not in hkx_walls:
                continue
            pixel_val = int(etf.pixels[row][col])
            need_eq = (pixel_val == 0)  # HJd: pixel==0 means wall must equal nRq
            if need_eq:
                req = int(nRq)
            else:
                # wall must NOT equal nRq; with gqb any other color
                req = ('not', int(nRq))

            if (nx, ny) in required:
                old = required[(nx, ny)]
                # Check consistency
                if isinstance(old, int) and isinstance(req, int) and old != req:
                    conflicts.append(f"CONFLICT at ({nx},{ny}): need {old} and {req}")
                elif isinstance(old, tuple) and isinstance(req, int):
                    if req == old[1]:  # need not-X but also need X
                        conflicts.append(f"CONFLICT at ({nx},{ny}): need not-{old[1]} and {req}")
                    else:
                        required[(nx, ny)] = req  # specific wins over "not X"
                elif isinstance(old, int) and isinstance(req, tuple):
                    if old == req[1]:  # need X but also need not-X
                        conflicts.append(f"CONFLICT at ({nx},{ny}): need {old} and not-{req[1]}")
                    # else old is fine (specific color != nRq satisfies not-nRq)
                # Two "not" constraints: OK as long as at least one color satisfies both
            else:
                required[(nx, ny)] = req

    # Compute clicks needed for each wall
    clicks = []
    for (wx, wy), req in required.items():
        curr = initial_color
        if isinstance(req, int):
            target = req
        else:
            # "not nRq" - pick first gqb color that isn't nRq and satisfies initial
            # initial might already satisfy
            not_val = req[1]
            if curr != not_val:
                continue  # already satisfied, no click needed
            # Need to click to a different color
            target = next(c for c in gqb if c != not_val)

        if curr == target:
            continue  # already correct
        # Count clicks to cycle from curr to target
        idx_curr = gqb.index(curr)
        idx_target = gqb.index(target)
        n_clicks = (idx_target - idx_curr) % len(gqb)
        for _ in range(n_clicks):
            clicks.append((wx * 2, wy * 2))  # display coord = grid * 2

    return name, gqb, bst_sprites, conflicts, clicks


def main():
    for i, level in enumerate(LEVELS):
        name, gqb, bst_sprites, conflicts, clicks = solve_level(i)
        bst_names = [sp.name for sp in bst_sprites]
        print(f"\nLevel {i} ({name}): gqb={gqb}, bsT={bst_names}")
        if conflicts:
            print(f"  CONFLICTS: {conflicts}")
        else:
            print(f"  Required clicks ({len(clicks)}): {sorted(set(clicks))}")
            dup = [c for c in clicks if clicks.count(c) > 1]
            if dup:
                print(f"  Multi-click walls: {sorted(set(dup))}")


if __name__ == "__main__":
    main()
