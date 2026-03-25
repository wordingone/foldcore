"""
Diagnostic: FT09 Lights-Out solver for L5 and L6 (NTi walls with spread patterns).

For levels with NTi walls, clicking one wall cycles itself + spread neighbors.
This is a Lights-Out system over Z/|gqb|.
For |gqb|==2 → GF(2) Gaussian elimination.
"""
import sys
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/ft09/0d8bbf25')
import os
os.environ['PYTHONUTF8'] = '1'

import numpy as np
from ft09 import levels as LEVELS

# ---- Build click-effect model from static level data ----

GBS = [
    [(-1, -1), (0, -1), (1, -1)],
    [(-1, 0),  (0, 0),  (1, 0)],
    [(-1, 1),  (0, 1),  (1, 1)],
]

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


def gauss_elim_gf2(M, b):
    """Gaussian elimination over GF(2). Returns solution x or None if no solution."""
    n_rows, n_cols = M.shape
    A = np.hstack([M.copy() % 2, b.reshape(-1, 1) % 2]).astype(np.int32)
    pivot_rows = []
    col = 0
    row = 0
    for col in range(n_cols):
        # Find pivot
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

    # Check consistency
    for r in range(row, n_rows):
        if A[r, -1] == 1:
            return None  # No solution

    # Back-substitute
    x = np.zeros(n_cols, dtype=np.int32)
    for pr, pc in pivot_rows:
        x[pc] = A[pr, -1]
    return x


def solve_level_gf2(level_idx):
    """Solve FT09 level using GF(2) Lights-Out when NTi walls present."""
    level = LEVELS[level_idx]
    name = level.name
    gqb = level.get_data("cwU") or [9, 8]
    gqb = [int(c) for c in gqb]
    mod = len(gqb)
    initial_color = gqb[0]

    # Build wall map
    walls = {}  # (x,y) -> {'type': 'Hkx'/'NTi', 'idx': int, 'init_color': int, 'spread': list}
    wall_pos = []

    for sp in level.get_sprites_by_tag("Hkx"):
        pos = (sp.x, sp.y)
        walls[pos] = {'type': 'Hkx', 'spread': []}
    for sp in level.get_sprites_by_tag("NTi"):
        pos = (sp.x, sp.y)
        # Determine spread from NTi pixels (pixels==6 -> spread to that neighbor)
        spread = []
        for j in range(3):
            for i in range(3):
                pix = int(sp.pixels[j][i])
                if pix == 6:
                    ybc, lga = GBS[j][i]
                    neighbor = (sp.x + ybc * 4, sp.y + lga * 4)
                    spread.append(neighbor)
        walls[pos] = {'type': 'NTi', 'spread': spread}

    # Assign indices
    wall_pos = sorted(walls.keys())
    for i, pos in enumerate(wall_pos):
        walls[pos]['idx'] = i
        walls[pos]['init_color'] = initial_color  # after gqb[0] remapping

    N = len(wall_pos)

    # Build click-effect matrix M[N x N] over Z/mod
    # M[i][j] = how many times wall i changes when button j is pressed once
    M = np.zeros((N, N), dtype=np.int32)
    for j, bpos in enumerate(wall_pos):
        wall = walls[bpos]
        # Button j always cycles itself
        M[walls[bpos]['idx'], j] = 1
        # Plus all spread neighbors that exist as walls
        for npos in wall['spread']:
            if npos in walls:
                M[walls[npos]['idx'], j] += 1

    M = M % mod

    # Build target vector from cgj() requirements
    bst_all = level.get_sprites_by_tag("bsT")
    bst_nonwall = [sp for sp in bst_all if "Hkx" not in sp.tags and "NTi" not in sp.tags]

    required = {}  # wall_pos -> required color index (0..mod-1), where index 0 = gqb[0]
    conflicts = []

    for etf in bst_nonwall:
        nRq_val = int(etf.pixels[1][1])
        # nRq_val is the raw pixel value from template, NOT remapped
        # For L5/L6, nRq_val should be in gqb
        if nRq_val not in gqb:
            print(f"  WARNING: nRq_val={nRq_val} not in gqb={gqb} for bsT at ({etf.x},{etf.y})")
            continue
        nRq_idx = gqb.index(nRq_val)

        for (row, col), (dx, dy) in NEIGHBOR_OFFSETS.items():
            nx, ny = etf.x + dx, etf.y + dy
            if (nx, ny) not in walls:
                continue
            pix_val = int(etf.pixels[row][col])
            need_eq = (pix_val == 0)
            if need_eq:
                req_idx = nRq_idx  # must equal gqb[nRq_idx]
            else:
                # Must NOT equal nRq - with 2 colors: must be the other color
                if mod == 2:
                    req_idx = (nRq_idx + 1) % mod
                else:
                    # With >2 colors: initial_color_idx is 0, if 0 != nRq_idx then already OK
                    init_idx = 0  # all start at gqb[0] = index 0
                    if init_idx != nRq_idx:
                        req_idx = init_idx  # already satisfied (any non-nRq color is fine, keep initial)
                    else:
                        req_idx = (nRq_idx + 1) % mod  # need to click once to get off nRq

            if (nx, ny) in required:
                if required[(nx, ny)] != req_idx:
                    conflicts.append(f"CONFLICT at ({nx},{ny}): need color_idx {required[(nx,ny)]} and {req_idx}")
            else:
                required[(nx, ny)] = req_idx

    if conflicts:
        print(f"  CONFLICTS: {conflicts}")
        return name, gqb, conflicts, []

    # Build target delta vector (how many clicks does each wall need, mod gqb)
    # init_idx = 0 for all walls (all start at gqb[0])
    t = np.zeros(N, dtype=np.int32)
    for pos, req_idx in required.items():
        wi = walls[pos]['idx']
        t[wi] = req_idx  # target color index (0 = initial, 1 = one click, etc.)

    print(f"L{level_idx+1} ({name}): gqb={gqb}, N={N}, mod={mod}")
    print(f"  Walls needing change: {[(wall_pos[i], gqb[t[i]]) for i in range(N) if t[i] != 0]}")

    if mod == 2:
        x = gauss_elim_gf2(M, t)
        if x is None:
            print(f"  No GF(2) solution!")
            return name, gqb, ["NO_SOLUTION"], []
        clicks = []
        for j, cnt in enumerate(x):
            if cnt > 0:
                bpos = wall_pos[j]
                clicks.append((bpos[0] * 2, bpos[1] * 2))  # display coords
        print(f"  Solution ({len(clicks)} clicks): {sorted(set(clicks))}")
        # Verify
        state = np.zeros(N, dtype=np.int32)
        for cx_d, cy_d in clicks:
            bpos = (cx_d // 2, cy_d // 2)
            j = walls[bpos]['idx']
            for i in range(N):
                state[i] = (state[i] + M[i, j]) % mod
        match = np.all(state == t)
        print(f"  Verify: {'PASS' if match else 'FAIL (state={state}, target={t})'}")
        return name, gqb, conflicts, clicks
    else:
        # For |gqb|>2 and no NTi: diagonal system → trivial
        clicks = []
        for wi, pos in enumerate(wall_pos):
            req = t[wi]
            # How many clicks to get from 0 to req? = req (since each click adds 1 mod mod)
            for _ in range(int(req)):
                clicks.append((pos[0] * 2, pos[1] * 2))
        print(f"  Solution ({len(clicks)} clicks): {sorted(set(clicks))}")
        return name, gqb, conflicts, clicks


# ---- Run for all levels ----
for i in range(6):
    name, gqb, conflicts, clicks = solve_level_gf2(i)
    if not conflicts:
        print(f"  -> {len(clicks)} total clicks, {len(set(clicks))} distinct positions\n")
    else:
        print()
