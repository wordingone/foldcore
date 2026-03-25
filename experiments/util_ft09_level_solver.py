"""
Offline solver for FT09 level solutions.
Computes the minimum clicks to satisfy cgj() for each level.
Reads sprite data directly from ft09.py.

Handles two wall types:
  Hkx: standard wall, clicking cycles only itself (uses level irw = center-only)
  NTi: spread wall, clicking cycles itself + neighbors where pixel==6

For levels with NTi walls (L5, L6): uses GF(2) Gaussian elimination (Lights-Out solver).
For levels without NTi walls (L1-L4): uses direct single-wall click computation.
"""
import sys
import numpy as np
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

# NTi spread: GBS offsets for each (row, col) position
_GBS = [
    [(-1, -1), (0, -1), (1, -1)],
    [(-1, 0),  (0, 0),  (1, 0)],
    [(-1, 1),  (0, 1),  (1, 1)],
]


def _gauss_elim_gf2(M, b):
    """Gaussian elimination over GF(2). Returns solution x or None if no solution."""
    n_rows, n_cols = M.shape
    A = np.hstack([M.copy() % 2, b.reshape(-1, 1) % 2]).astype(np.int32)
    pivot_rows = []
    row = 0
    for col in range(n_cols):
        pivot = None
        for r in range(row, n_rows):
            if A[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        A[[row, pivot]] = A[[pivot, row]]
        for r in range(n_rows):
            if r != row and A[r, col] == 1:
                A[r] = (A[r] + A[row]) % 2
        pivot_rows.append((row, col))
        row += 1
    for r in range(row, n_rows):
        if A[r, -1] == 1:
            return None
    x = np.zeros(n_cols, dtype=np.int32)
    for pr, pc in pivot_rows:
        x[pc] = A[pr, -1]
    return x


def _solve_nti_level(level_idx):
    """GF(2) Lights-Out solver for levels with NTi spread walls."""
    level = LEVELS[level_idx]
    name = level.name
    gqb = level.get_data("cwU") or [9, 8]
    gqb = [int(c) for c in gqb]
    mod = len(gqb)
    initial_color = gqb[0]

    # Build wall map with spread info
    walls = {}
    for sp in level.get_sprites_by_tag("Hkx"):
        walls[(sp.x, sp.y)] = {'type': 'Hkx', 'spread': []}
    for sp in level.get_sprites_by_tag("NTi"):
        spread = []
        for j in range(3):
            for i in range(3):
                if int(sp.pixels[j][i]) == 6:
                    ybc, lga = _GBS[j][i]
                    spread.append((sp.x + ybc * 4, sp.y + lga * 4))
        walls[(sp.x, sp.y)] = {'type': 'NTi', 'spread': spread}

    wall_pos = sorted(walls.keys())
    for i, pos in enumerate(wall_pos):
        walls[pos]['idx'] = i

    N = len(wall_pos)
    M = np.zeros((N, N), dtype=np.int32)
    for j, bpos in enumerate(wall_pos):
        M[walls[bpos]['idx'], j] = 1
        for npos in walls[bpos]['spread']:
            if npos in walls:
                M[walls[npos]['idx'], j] += 1
    M = M % mod

    # Build requirements from bsT sprites
    bst_all = level.get_sprites_by_tag("bsT")
    bst_nonwall = [sp for sp in bst_all if "Hkx" not in sp.tags and "NTi" not in sp.tags]
    required = {}
    conflicts = []

    for etf in bst_nonwall:
        nRq_val = int(etf.pixels[1][1])
        if nRq_val not in gqb:
            continue
        nRq_idx = gqb.index(nRq_val)
        for (row, col), (dx, dy) in NEIGHBOR_OFFSETS.items():
            nx, ny = etf.x + dx, etf.y + dy
            if (nx, ny) not in walls:
                continue
            pix_val = int(etf.pixels[row][col])
            need_eq = (pix_val == 0)
            if need_eq:
                req_idx = nRq_idx
            else:
                # "must not equal nRq"
                init_idx = 0
                if init_idx != nRq_idx:
                    req_idx = init_idx  # initial already satisfies != nRq
                else:
                    req_idx = 1 % mod  # need to change
            if (nx, ny) in required:
                if required[(nx, ny)] != req_idx:
                    conflicts.append(f"CONFLICT at ({nx},{ny})")
            else:
                required[(nx, ny)] = req_idx

    if conflicts:
        return name, gqb, bst_nonwall, conflicts, []

    t = np.zeros(N, dtype=np.int32)
    for pos, req_idx in required.items():
        t[walls[pos]['idx']] = req_idx

    x = _gauss_elim_gf2(M, t)
    if x is None:
        return name, gqb, bst_nonwall, ["NO_SOLUTION"], []

    clicks = []
    for j, cnt in enumerate(x):
        if cnt > 0:
            bpos = wall_pos[j]
            clicks.append((bpos[0] * 2, bpos[1] * 2))
    return name, gqb, bst_nonwall, conflicts, clicks


def solve_level(level_idx):
    """Solve FT09 level. Dispatches to GF(2) solver if NTi walls present."""
    level = LEVELS[level_idx]
    # Dispatch to NTi/Lights-Out solver if any NTi walls exist
    if level.get_sprites_by_tag("NTi"):
        return _solve_nti_level(level_idx)

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
