"""
SP80 Abstract Model Solver — solve ALL 6 levels by modeling the game as
pure Python state transitions. NO API calls during search.

The game is a liquid flow puzzle:
- "change" mode: move horizontal/vertical bars and L-shaped pieces around
- "spill" mode (ACTION5): liquid flows from sources, interacts with bars/pieces
- Win condition: ALL receptacles filled AND no liquid hits the floor

The abstract model replicates the game's step() liquid flow logic exactly.
BFS on abstract state (bar positions) should be 50K+ states/sec.

Action encoding: 0-6 = keyboard, 7+ = click (7 + y*64 + x)
"""
import sys
import os
import json
import time
import hashlib
import numpy as np
from collections import deque
from copy import deepcopy

os.chdir('B:/M/the-search')

# ============================================================
# SPRITE DEFINITIONS (from sp80.py source)
# ============================================================

# Pixel values: -1 = transparent, 4 = source marker, others = solid/colored
SPRITE_PIXEL_DATA = {
    'adbrqflmwi': [[8,8,8,4,8,8,8]],        # 7x1 bar with source at col 3
    'jgfvrvnkaz': [[8,8,8,8,8]],             # 5x1 bar
    'mdhkebfsmg': [[8],[8],[8],[8]],          # 1x4 vertical bar
    'nvzozwqarf': [[8,8,8,8,8,8,8,8]],       # 8x1 bar
    'odioorqnkn': [[8,8,8,8,8,8]],           # 6x1 bar
    'trurgcakbj': [[8,8,8,8]],               # 4x1 bar
    'uihgaxtzkm': [[8,8,8,8,8,8,8]],         # 7x1 bar (not clickable)
    'untfxhpddv': [[8,8,8]],                 # 3x1 bar
    'zgsbadjnjn': [[8,8,4,8,8]],             # 5x1 bar with source at col 2
    'qwsmjdrvqj': [[15,-1],[15,15]],          # 2x2 L-piece (top-right transparent)
    'vkwijvqdla': [[-1,15],[15,15]],           # 2x2 L-piece (top-left transparent)
    'xsrqllccpx': [[11,-1,11],[11,11,11]],    # 3x2 receptacle (top-center transparent)
    'syaipsfndp': [[4]],                      # 1x1 source marker
    'nkrtlkykwe': [[6]],                      # 1x1 liquid
}

SPRITE_TAGS = {
    'adbrqflmwi': {'ksmzdcblcz', 'syaipsfndp'},
    'jgfvrvnkaz': {'ksmzdcblcz', 'sys_click'},
    'mdhkebfsmg': {'ksmzdcblcz', 'sys_click'},
    'nvzozwqarf': {'ksmzdcblcz', 'sys_click'},
    'odioorqnkn': {'ksmzdcblcz', 'sys_click'},
    'trurgcakbj': {'ksmzdcblcz', 'sys_click'},
    'uihgaxtzkm': {'ksmzdcblcz'},
    'untfxhpddv': {'ksmzdcblcz', 'sys_click'},
    'zgsbadjnjn': {'ksmzdcblcz', 'syaipsfndp', 'sys_click'},
    'qwsmjdrvqj': {'hfjpeygkxy', 'sys_click'},
    'vkwijvqdla': {'hfjpeygkxy', 'sys_click'},
    'xsrqllccpx': {'xsrqllccpx'},
    'syaipsfndp': {'syaipsfndp'},
    'nkrtlkykwe': {'nkrtlkykwe'},
    'uzunfxpwmd': {'uzunfxpwmd'},
    'nadtnzkesz': set(),
    'ttkatugvbk': set(),
    'uzvelihpxo': set(),
}

# Container walls — hollow rectangles
CONTAINER_DIMS = {
    'nadtnzkesz': (34, 33),
    'ttkatugvbk': (22, 22),
    'uzvelihpxo': (18, 18),
}


def build_solid_pixels(name, rotation=0):
    """Build set of (local_x, local_y) solid pixel offsets for a sprite.
    Handles rotation. Returns (solid_set, width, height, source_offsets)."""
    if name in CONTAINER_DIMS:
        w, h = CONTAINER_DIMS[name]
        solids = set()
        for y in range(h):
            for x in range(w):
                if x == 0 or x == w-1 or y == 0 or y == h-1:
                    solids.add((x, y))
        return solids, w, h, []

    if name == 'uzunfxpwmd':
        # 32-wide, 1-tall floor
        solids = {(x, 0) for x in range(32)}
        if rotation == 0:
            return solids, 32, 1, []
        elif rotation == 90:
            # Rotated 90 CW: becomes 1-wide, 32-tall
            new_solids = set()
            for (x, y) in solids:
                # 90 CW: (x,y) -> (h-1-y, x) where h=1
                new_solids.add((1-1-y, x))
            return new_solids, 1, 32, []
        elif rotation == 180:
            new_solids = {(31-x, 0) for (x, y) in solids}
            return new_solids, 32, 1, []
        elif rotation == 270:
            new_solids = set()
            for (x, y) in solids:
                new_solids.add((y, 31-x))
            return new_solids, 1, 32, []

    if name not in SPRITE_PIXEL_DATA:
        return set(), 0, 0, []

    raw = SPRITE_PIXEL_DATA[name]
    orig_h = len(raw)
    orig_w = len(raw[0]) if raw else 0

    solids = set()
    sources = []
    for row_idx, row in enumerate(raw):
        for col_idx, val in enumerate(row):
            if val >= 0:
                solids.add((col_idx, row_idx))
            if val == 4:
                sources.append((col_idx, row_idx))

    if rotation == 0:
        return solids, orig_w, orig_h, sources

    # Apply rotation
    new_solids = set()
    new_sources = []
    for (x, y) in solids:
        if rotation == 90:
            nx, ny = orig_h - 1 - y, x
        elif rotation == 180:
            nx, ny = orig_w - 1 - x, orig_h - 1 - y
        elif rotation == 270:
            nx, ny = y, orig_w - 1 - x
        else:
            nx, ny = x, y
        new_solids.add((nx, ny))

    for (x, y) in sources:
        if rotation == 90:
            nx, ny = orig_h - 1 - y, x
        elif rotation == 180:
            nx, ny = orig_w - 1 - x, orig_h - 1 - y
        elif rotation == 270:
            nx, ny = y, orig_w - 1 - x
        else:
            nx, ny = x, y
        new_sources.append((nx, ny))

    if rotation in (90, 270):
        return new_solids, orig_h, orig_w, new_sources
    return new_solids, orig_w, orig_h, new_sources


# ============================================================
# LEVEL DEFINITIONS
# ============================================================

LEVELS = [
    # Level 1 (index 0)
    {
        'grid_size': (16, 16),
        'steps': 30,
        'rotation': 0,
        'sprites': [
            ('jgfvrvnkaz', 3, 4, 0),
            ('nkrtlkykwe', 9, 1, 0),
            ('syaipsfndp', 9, 0, 0),
            ('uzunfxpwmd', 0, 15, 0),
            ('uzvelihpxo', -1, -1, 0),
            ('xsrqllccpx', 4, 13, 0),
            ('xsrqllccpx', 10, 13, 0),
        ],
    },
    # Level 2 (index 1)
    {
        'grid_size': (16, 16),
        'steps': 45,
        'rotation': 180,
        'sprites': [
            ('jgfvrvnkaz', 6, 6, 0),
            ('nkrtlkykwe', 5, 1, 0),
            ('syaipsfndp', 5, 0, 0),
            ('untfxhpddv', 6, 9, 0),
            ('untfxhpddv', 11, 11, 0),
            ('uzunfxpwmd', 0, 15, 0),
            ('uzvelihpxo', -1, -1, 0),
            ('xsrqllccpx', 2, 13, 0),
            ('xsrqllccpx', 6, 13, 0),
            ('xsrqllccpx', 10, 13, 0),
        ],
    },
    # Level 3 (index 2)
    {
        'grid_size': (16, 16),
        'steps': 100,
        'rotation': 180,
        'sprites': [
            ('jgfvrvnkaz', 1, 8, 0),
            ('nkrtlkykwe', 1, 1, 0),
            ('nkrtlkykwe', 14, 1, 0),
            ('nkrtlkykwe', 6, 1, 0),
            ('odioorqnkn', 8, 7, 0),
            ('odioorqnkn', 1, 5, 0),
            ('syaipsfndp', 1, 0, 0),
            ('syaipsfndp', 14, 0, 0),
            ('syaipsfndp', 6, 0, 0),
            ('trurgcakbj', 10, 10, 0),
            ('uzunfxpwmd', 0, 15, 0),
            ('uzvelihpxo', -1, -1, 0),
            ('xsrqllccpx', 1, 13, 0),
            ('xsrqllccpx', 12, 13, 0),
            ('xsrqllccpx', 7, 13, 0),
        ],
    },
    # Level 4 (index 3)
    {
        'grid_size': (20, 20),
        'steps': 120,
        'rotation': 0,
        'sprites': [
            ('adbrqflmwi', 2, 9, 0),
            ('jgfvrvnkaz', 12, 5, 0),
            ('jgfvrvnkaz', 5, 5, 0),
            ('nkrtlkykwe', 7, 1, 0),
            ('syaipsfndp', 7, 0, 0),
            ('trurgcakbj', 12, 13, 0),
            ('trurgcakbj', 14, 10, 0),
            ('ttkatugvbk', -1, -1, 0),
            ('uzunfxpwmd', 0, 19, 0),
            ('xsrqllccpx', 2, 17, 0),
            ('xsrqllccpx', 16, 17, 0),
            ('xsrqllccpx', 8, 17, 0),
            ('xsrqllccpx', 12, 17, 0),
        ],
    },
    # Level 5 (index 4)
    {
        'grid_size': (20, 20),
        'steps': 100,
        'rotation': 180,
        'sprites': [
            ('jgfvrvnkaz', 2, 9, 0),
            ('nkrtlkykwe', 5, 1, 0),
            ('nkrtlkykwe', 13, 1, 0),
            ('qwsmjdrvqj', 8, 5, 0),
            ('syaipsfndp', 5, 0, 0),
            ('syaipsfndp', 13, 0, 0),
            ('trurgcakbj', 7, 13, 0),
            ('ttkatugvbk', -1, -1, 0),
            ('untfxhpddv', 11, 9, 0),
            ('uzunfxpwmd', 0, 19, 0),
            ('uzunfxpwmd', 19, 0, 90),
            ('uzunfxpwmd', -1, 0, 90),
            ('xsrqllccpx', 17, 6, 270),
            ('xsrqllccpx', 2, 17, 0),
            ('xsrqllccpx', 6, 17, 0),
            ('xsrqllccpx', 12, 17, 0),
        ],
    },
    # Level 6 (index 5)
    {
        'grid_size': (20, 20),
        'steps': 120,
        'rotation': 0,
        'sprites': [
            ('mdhkebfsmg', 14, 4, 0),
            ('nkrtlkykwe', 9, 1, 0),
            ('qwsmjdrvqj', 9, 5, 0),
            ('syaipsfndp', 9, 0, 0),
            ('ttkatugvbk', -1, -1, 0),
            ('uzunfxpwmd', 0, 19, 0),
            ('uzunfxpwmd', 19, 0, 90),
            ('uzunfxpwmd', 0, 0, 90),
            ('vkwijvqdla', 9, 14, 0),
            ('xsrqllccpx', 17, 9, 270),
            ('xsrqllccpx', 8, 17, 0),
            ('xsrqllccpx', 1, 11, 90),
            ('xsrqllccpx', 1, 6, 90),
            ('zgsbadjnjn', 7, 10, 0),
        ],
    },
]


class SpriteInfo:
    """Precomputed sprite info for abstract model."""
    __slots__ = ['name', 'init_x', 'init_y', 'w', 'h', 'tags', 'solid_offsets',
                 'source_offsets', 'rotation', 'is_moveable', 'moveable_idx']

    def __init__(self, name, x, y, rotation=0):
        self.name = name
        self.init_x = x
        self.init_y = y
        self.rotation = rotation
        self.tags = SPRITE_TAGS.get(name, set())

        solids, w, h, sources = build_solid_pixels(name, rotation)
        self.solid_offsets = frozenset(solids)
        self.w = w
        self.h = h
        self.source_offsets = sources

        self.is_moveable = ('ksmzdcblcz' in self.tags or 'hfjpeygkxy' in self.tags)
        self.moveable_idx = -1  # set later


class AbstractSP80:
    """Abstract model of SP80 game. Replicates game logic without arcengine."""

    def __init__(self, level_idx):
        level = LEVELS[level_idx]
        self.grid_w, self.grid_h = level['grid_size']
        self.max_steps = level['steps']
        self.rotation_k = level['rotation'] // 90 % 4

        # Parse all sprites
        self.all_sprites = []
        self.moveable = []
        self.receptacles = []
        self.initial_liquids = []
        self.source_sprites = []  # sprites with syaipsfndp tag

        for name, x, y, rot in level['sprites']:
            si = SpriteInfo(name, x, y, rot)
            self.all_sprites.append(si)

            if si.is_moveable:
                si.moveable_idx = len(self.moveable)
                self.moveable.append(si)
            elif 'xsrqllccpx' in si.tags:
                self.receptacles.append(si)
            elif 'nkrtlkykwe' in si.tags:
                self.initial_liquids.append(si)

            if 'syaipsfndp' in si.tags:
                self.source_sprites.append(si)

        # Precompute static (non-moveable) sprite pixel positions
        self.static_sprites = []  # (name, tags, global_pixel_set)
        for si in self.all_sprites:
            if si.is_moveable or 'nkrtlkykwe' in si.tags:
                continue
            if 'syaipsfndp' in si.tags and 'ksmzdcblcz' not in si.tags:
                continue  # standalone source markers, handled separately
            if 'xsrqllccpx' in si.tags:
                continue  # receptacles handled separately
            globals_set = set()
            for (lx, ly) in si.solid_offsets:
                globals_set.add((si.init_x + lx, si.init_y + ly))
            self.static_sprites.append((si.name, si.tags, globals_set, si))

        # Precompute receptacle global pixels (they don't move)
        self.receptacle_globals = []
        for ri, r in enumerate(self.receptacles):
            gset = set()
            for (lx, ly) in r.solid_offsets:
                gset.add((r.init_x + lx, r.init_y + ly))
            self.receptacle_globals.append((ri, gset, r))

        # Build static grid for fast spill simulation
        self._build_static_grid()

    def _build_static_grid(self):
        """Pre-compute a grid encoding of static sprites.
        Grid values: 0=empty, 1=wall, 2=floor(uzunfxpwmd), 100+ri=receptacle ri.
        This is called once during __init__ and reused for all spill simulations.
        """
        # Grid offset: shift all coords so minimum is at (0,0)
        # Find bounds
        all_coords = set()
        for name, tags, gset, si in self.static_sprites:
            all_coords |= gset
        for ri, gset, r in self.receptacle_globals:
            all_coords |= gset

        if not all_coords:
            self._grid_ox = 0
            self._grid_oy = 0
            self._grid_w = self.grid_w + 4
            self._grid_h = self.grid_h + 4
        else:
            min_x = min(x for x, y in all_coords) - 2
            min_y = min(y for x, y in all_coords) - 2
            max_x = max(x for x, y in all_coords) + 2
            max_y = max(y for x, y in all_coords) + 2
            self._grid_ox = min_x
            self._grid_oy = min_y
            self._grid_w = max_x - min_x + 1
            self._grid_h = max_y - min_y + 1

        # Use numpy array for fast lookup
        # Values: 0=empty, 1=wall, 2=floor, 100+ri=receptacle
        grid = np.zeros((self._grid_h, self._grid_w), dtype=np.int16)

        for name, tags, gset, si in self.static_sprites:
            val = 2 if 'uzunfxpwmd' in tags else 1
            for (x, y) in gset:
                gx, gy = x - self._grid_ox, y - self._grid_oy
                if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
                    grid[gy, gx] = val

        for ri, gset, r in self.receptacle_globals:
            for (x, y) in gset:
                gx, gy = x - self._grid_ox, y - self._grid_oy
                if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
                    if grid[gy, gx] == 0:
                        grid[gy, gx] = 100 + ri

        self._static_grid = grid

        # Pre-compute bar pixel offsets and tags for fast grid stamping
        self._bar_grid_vals = []
        for i, s in enumerate(self.moveable):
            if 'ksmzdcblcz' in s.tags:
                val = 50 + i  # bar value: 50+i
            elif 'hfjpeygkxy' in s.tags:
                val = 70 + i  # L-piece value: 70+i
            else:
                val = 50 + i
            self._bar_grid_vals.append((val, list(s.solid_offsets)))

    def _build_sprite_map(self, positions):
        """Build pixel map using pre-computed static grid + bar overlay.
        Returns numpy grid array.
        """
        grid = self._static_grid.copy()

        for i, (val, offsets) in enumerate(self._bar_grid_vals):
            sx, sy = positions[i]
            for (lx, ly) in offsets:
                gx = sx + lx - self._grid_ox
                gy = sy + ly - self._grid_oy
                if 0 <= gx < self._grid_w and 0 <= gy < self._grid_h:
                    if grid[gy, gx] == 0:
                        grid[gy, gx] = val

        return grid

    def _simulate_spill(self, positions):
        """Simulate the liquid spill. Returns (all_receptacles_filled, hit_floor).
        Win condition: all_filled AND NOT hit_floor.

        Uses numpy grid for fast pixel lookup.
        Grid values: 0=empty, 1=wall, 2=floor, 50+i=bar_i, 70+i=L-piece_i, 100+ri=receptacle_ri.
        """
        grid = self._build_sprite_map(positions)
        ox, oy = self._grid_ox, self._grid_oy
        gw, gh = self._grid_w, self._grid_h

        # Initialize liquid drops
        active_drops = []
        for liq in self.initial_liquids:
            active_drops.append((liq.init_x, liq.init_y, 0, 1))

        for si in self.source_sprites:
            if si.is_moveable:
                sx, sy = positions[si.moveable_idx]
            else:
                sx, sy = si.init_x, si.init_y

            for (src_lx, src_ly) in si.source_offsets:
                src_gx = sx + src_lx
                src_gy = sy + src_ly
                bx, by = src_gx, src_gy + 1

                has_liquid_below = False
                for liq in self.initial_liquids:
                    if liq.init_x == bx and liq.init_y == by:
                        has_liquid_below = True
                        break

                if not has_liquid_below:
                    bgx, bgy = bx - ox, by - oy
                    if 0 <= bgx < gw and 0 <= bgy < gh:
                        if grid[bgy, bgx] == 0:
                            active_drops.append((bx, by, 0, 1))

        liquid_set = set()
        for (x, y, dx, dy) in active_drops:
            liquid_set.add((x, y))

        filled_receptacles = set()
        hit_floor = False
        n_recept = len(self.receptacles)

        for _ in range(2000):
            if not active_drops:
                break

            next_drops = []

            for (x, y, dx, dy) in active_drops:
                nx, ny = x + dx, y + dy
                ngx, ngy = nx - ox, ny - oy

                # Check grid value at target
                if 0 <= ngx < gw and 0 <= ngy < gh:
                    val = grid[ngy, ngx]
                else:
                    val = -1  # out of bounds = wall

                # Also check if target is liquid
                if val == 0 and (nx, ny) in liquid_set:
                    val = 3  # treat as liquid

                if val == 0:
                    # Empty - flow there
                    liquid_set.add((nx, ny))
                    next_drops.append((nx, ny, dx, dy))

                elif val == 3:
                    # Liquid - propagate through
                    next_drops.append((nx, ny, dx, dy))

                elif 50 <= val < 70:
                    # Bar (ksmzdcblcz) - spread sideways
                    if dy != 0:
                        sides = [(-1, 0), (1, 0)]
                    else:
                        sides = [(0, -1), (0, 1)]
                    for (sdx, sdy) in sides:
                        sx2, sy2 = x + sdx, y + sdy
                        sgx, sgy = sx2 - ox, sy2 - oy
                        if (sx2, sy2) not in liquid_set:
                            if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                liquid_set.add((sx2, sy2))
                                next_drops.append((sx2, sy2, dx, dy))

                elif val >= 100:
                    # Receptacle - check if both adj are same receptacle
                    ri = val - 100
                    if dy != 0:
                        a1x, a1y = x - 1, y
                        a2x, a2y = x + 1, y
                    else:
                        a1x, a1y = x, y - 1
                        a2x, a2y = x, y + 1

                    a1gx, a1gy = a1x - ox, a1y - oy
                    a2gx, a2gy = a2x - ox, a2y - oy

                    a1_same = (0 <= a1gx < gw and 0 <= a1gy < gh and grid[a1gy, a1gx] == val)
                    a2_same = (0 <= a2gx < gw and 0 <= a2gy < gh and grid[a2gy, a2gx] == val)

                    if a1_same and a2_same:
                        filled_receptacles.add(ri)
                    else:
                        if dy != 0:
                            sides = [(-1, 0), (1, 0)]
                        else:
                            sides = [(0, -1), (0, 1)]
                        for (sdx, sdy) in sides:
                            sx2, sy2 = x + sdx, y + sdy
                            sgx, sgy = sx2 - ox, sy2 - oy
                            if (sx2, sy2) not in liquid_set:
                                if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                    liquid_set.add((sx2, sy2))
                                    next_drops.append((sx2, sy2, dx, dy))

                elif 70 <= val < 100:
                    # L-piece (hfjpeygkxy) - redirect
                    bar_i = val  # the L-piece index value
                    if dy != 0:
                        a1x, a1y = x - 1, y
                        a2x, a2y = x + 1, y
                    else:
                        a1x, a1y = x, y - 1
                        a2x, a2y = x, y + 1

                    a1gx, a1gy = a1x - ox, a1y - oy
                    a2gx, a2gy = a2x - ox, a2y - oy

                    a1_same = (0 <= a1gx < gw and 0 <= a1gy < gh and grid[a1gy, a1gx] == bar_i)
                    a2_same = (0 <= a2gx < gw and 0 <= a2gy < gh and grid[a2gy, a2gx] == bar_i)

                    if a1_same and not a2_same:
                        ndx, ndy = dy, -dx
                        sx2, sy2 = x + ndx, y + ndy
                        sgx, sgy = sx2 - ox, sy2 - oy
                        if (sx2, sy2) not in liquid_set:
                            if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                liquid_set.add((sx2, sy2))
                                next_drops.append((sx2, sy2, ndx, ndy))
                    elif a2_same and not a1_same:
                        ndx, ndy = -dy, dx
                        sx2, sy2 = x + ndx, y + ndy
                        sgx, sgy = sx2 - ox, sy2 - oy
                        if (sx2, sy2) not in liquid_set:
                            if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                liquid_set.add((sx2, sy2))
                                next_drops.append((sx2, sy2, ndx, ndy))
                    else:
                        if dy != 0:
                            sides = [(-1, 0), (1, 0)]
                        else:
                            sides = [(0, -1), (0, 1)]
                        for (sdx, sdy) in sides:
                            sx2, sy2 = x + sdx, y + sdy
                            sgx, sgy = sx2 - ox, sy2 - oy
                            if (sx2, sy2) not in liquid_set:
                                if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                    liquid_set.add((sx2, sy2))
                                    next_drops.append((sx2, sy2, dx, dy))

                elif val == 2:
                    # Floor - hit_floor
                    hit_floor = True

                elif val == 1 or val == -1:
                    # Wall - spread sideways
                    if dy != 0:
                        sides = [(-1, 0), (1, 0)]
                    else:
                        sides = [(0, -1), (0, 1)]
                    for (sdx, sdy) in sides:
                        sx2, sy2 = x + sdx, y + sdy
                        sgx, sgy = sx2 - ox, sy2 - oy
                        if (sx2, sy2) not in liquid_set:
                            if 0 <= sgx < gw and 0 <= sgy < gh and grid[sgy, sgx] == 0:
                                liquid_set.add((sx2, sy2))
                                next_drops.append((sx2, sy2, dx, dy))

            active_drops = next_drops

        all_filled = len(filled_receptacles) == n_recept
        return all_filled, hit_floor

    def _can_place(self, sprite_idx, new_x, new_y, positions):
        """Check if bar can be placed at new position. Mirrors aqltiyljgy()."""
        s = self.moveable[sprite_idx]
        if new_y < 3:
            return False
        # Check overlap with receptacles (with 1px margin)
        for ri, gset, r in self.receptacle_globals:
            rx, ry, rw, rh = r.init_x, r.init_y, r.w, r.h
            if (new_x < rx + rw + 1 and new_x + s.w > rx - 1 and
                new_y < ry + rh + 1 and new_y + s.h > ry - 1):
                return False
        return True

    def _check_move(self, sprite_idx, dx, dy, positions):
        """Check if sprite can move by (dx, dy). Returns new positions or None.
        Mirrors the try_move_sprite + collision logic.
        """
        s = self.moveable[sprite_idx]
        sx, sy = positions[sprite_idx]
        new_x, new_y = sx + dx, sy + dy

        if not self._can_place(sprite_idx, new_x, new_y, positions):
            return None

        new_solids = {(new_x + lx, new_y + ly) for (lx, ly) in s.solid_offsets}

        # Check collision with static sprites
        for name, tags, gset, si in self.static_sprites:
            if new_solids & gset:
                return None  # blocked by static

        # Check collision with other moveables
        all_bar_collisions = True
        has_collision = False
        for j, other in enumerate(self.moveable):
            if j == sprite_idx:
                continue
            ox, oy = positions[j]
            other_solids = {(ox + lx, oy + ly) for (lx, ly) in other.solid_offsets}
            if new_solids & other_solids:
                has_collision = True
                if not ('ksmzdcblcz' in other.tags or 'hfjpeygkxy' in other.tags):
                    all_bar_collisions = False
                    break

        # Check collision with receptacles
        for ri, gset, r in self.receptacle_globals:
            if new_solids & gset:
                has_collision = True
                all_bar_collisions = False
                break

        if has_collision and not all_bar_collisions:
            return None  # blocked by non-bar

        # Move succeeds (either no collision or only bar collisions)
        new_positions = list(positions)
        new_positions[sprite_idx] = (new_x, new_y)
        return tuple(new_positions)


def solve_level_bfs(level_idx, max_depth=30, max_states=10000000, time_limit=300, min_spill_depth=0):
    """BFS solver using abstract model.

    Uses two-phase approach:
    Phase 1: BFS on positions only (fast, no spill testing)
    Phase 2: Test spill at each position in BFS order

    For small levels (few bars): test spill at every position.
    For large levels: defer spill testing using min_spill_depth.
    """
    model = AbstractSP80(level_idx)
    n_bars = len(model.moveable)
    n_recept = len(model.receptacles)

    positions = tuple((s.init_x, s.init_y) for s in model.moveable)
    DIR_VECTORS = [(0,-1), (0,1), (-1,0), (1,0)]

    k = model.rotation_k
    print(f"  BFS L{level_idx+1}: {n_bars} bars, {n_recept} receptacles, grid={model.grid_w}x{model.grid_h}, rot={k*90}")
    for i, s in enumerate(model.moveable):
        print(f"    Bar {i}: {s.name} at ({s.init_x},{s.init_y}) {s.w}x{s.h}")
    for i, r in enumerate(model.receptacles):
        print(f"    Receptacle {i}: at ({r.init_x},{r.init_y}) {r.w}x{r.h} rot={r.rotation}")

    # Check initial position
    all_filled, hit_floor = model._simulate_spill(positions)
    if all_filled and not hit_floor:
        return [('spill',)], model

    # BFS with depth-layered processing
    # At each depth, first expand all nodes, then test spills on the new frontier
    current_layer = [(positions, [])]
    visited = {positions}
    spill_cache = {positions: (all_filled, hit_floor)}

    t0 = time.time()
    explored = 0
    spill_checks = 1

    for depth in range(max_depth):
        if time.time() - t0 > time_limit:
            el = time.time() - t0
            print(f"    TIMEOUT at depth {depth} ({el:.0f}s, explored={explored}, spills={spill_checks})")
            return None, model

        # Generate next layer (movement only, no spill testing)
        next_layer = []

        for positions, actions in current_layer:
            for bar_idx in range(n_bars):
                for d in range(4):
                    dx, dy = DIR_VECTORS[d]
                    new_positions = model._check_move(bar_idx, dx, dy, positions)
                    if new_positions is None:
                        continue
                    if new_positions in visited:
                        continue
                    visited.add(new_positions)
                    explored += 1
                    next_layer.append((new_positions, actions + [(bar_idx, d)]))

        if not next_layer:
            print(f"    EXHAUSTED at depth {depth+1} ({explored})")
            return None, model

        el = time.time() - t0
        rate = explored / max(el, 0.01)
        print(f"    d={depth+1} new={len(next_layer)} total={len(visited)} t={el:.1f}s ({rate:.0f}/s)", end='')

        # Test spills on new frontier
        if depth + 1 >= min_spill_depth:
            wins = []
            for new_positions, new_actions in next_layer:
                if new_positions not in spill_cache:
                    af, hf = model._simulate_spill(new_positions)
                    spill_cache[new_positions] = (af, hf)
                    spill_checks += 1
                else:
                    af, hf = spill_cache[new_positions]

                if af and not hf:
                    wins.append((new_positions, new_actions))

                if time.time() - t0 > time_limit:
                    break

            print(f" sp={spill_checks} wins={len(wins)}")

            if wins:
                # Pick shortest action sequence
                best = min(wins, key=lambda x: len(x[1]))
                el = time.time() - t0
                result = best[1] + [('spill',)]
                print(f"    SOLVED! depth={len(result)}, explored={explored}, spills={spill_checks}, time={el:.1f}s")
                return result, model
        else:
            print(f" (spill deferred)")

        if len(visited) >= max_states:
            print(f"    STATE LIMIT ({max_states})")
            return None, model

        current_layer = next_layer

    print(f"    MAX DEPTH ({max_depth})")
    return None, model


def abstract_to_game_actions(abstract_actions, model):
    """Convert abstract actions to game action encoding.

    Abstract actions are tuples:
    - (bar_idx, direction): move bar in direction (0=up,1=down,2=left,3=right in grid space)
    - ('spill',): trigger spill

    Game actions: 0-4=ACTION1-5, 7+=click

    We need to:
    1. Track which bar is currently selected
    2. Insert click actions to select the right bar when needed
    3. Map grid-space directions to game actions (accounting for rotation)
    """
    game_actions = []

    # Track bar positions for click coordinate computation
    positions = list((s.init_x, s.init_y) for s in model.moveable)

    # Initial selection: closest bar to origin
    dists = [(positions[i][0]**2 + positions[i][1]**2, i) for i in range(len(model.moveable))]
    dists.sort()
    selected = dists[0][1] if dists else -1

    k = model.rotation_k

    # Map grid-space direction to game action (ACTION1-4)
    # Grid: 0=up(dy=-1), 1=down(dy=1), 2=left(dx=-1), 3=right(dx=1)
    # Game after remap: ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right
    # So grid direction d maps to game action d (0-3) AFTER remap.
    # But we need the PRE-remap game action.
    # The game's remap: wdxitozphu[k] maps input -> effective
    # We need: effective -> input (inverse = qxlcnqsvsf)
    INVERSE_REMAP = {
        0: {0: 0, 1: 1, 2: 2, 3: 3},
        1: {0: 2, 1: 3, 2: 1, 3: 0},  # qxlcnqsvsf[1]: A1->A3, A2->A4, A3->A2, A4->A1
        2: {0: 1, 1: 0, 2: 3, 3: 2},  # qxlcnqsvsf[2]: A1->A2, A2->A1, A3->A4, A4->A3
        3: {0: 3, 1: 2, 2: 0, 3: 1},  # qxlcnqsvsf[3]: A1->A4, A2->A3, A3->A1, A4->A2
    }
    inv_remap = INVERSE_REMAP[k]

    DIR_VECTORS = [(0,-1), (0,1), (-1,0), (1,0)]

    def make_click(bar_idx):
        """Generate click action to select bar bar_idx at its current position."""
        s = model.moveable[bar_idx]
        bx, by = positions[bar_idx]

        # Grid center of bar
        gx = bx + s.w // 2
        gy = by + s.h // 2

        # Grid to display
        if model.grid_w <= 16 and model.grid_h <= 16:
            scale, off_x, off_y = 4, 0, 0
        else:
            scale, off_x, off_y = 3, 2, 2

        disp_x = gx * scale + off_x
        disp_y = gy * scale + off_y

        # Apply inverse of game rotation to display coords
        if k == 1:
            disp_x, disp_y = disp_y, 63 - disp_x
        elif k == 2:
            disp_x, disp_y = 63 - disp_x, 63 - disp_y
        elif k == 3:
            disp_x, disp_y = 63 - disp_y, disp_x

        disp_x = max(0, min(63, disp_x))
        disp_y = max(0, min(63, disp_y))
        return 7 + disp_y * 64 + disp_x

    for action in abstract_actions:
        if action[0] == 'spill':
            game_actions.append(4)
            # After spill, re-select closest bar to origin
            dists2 = [(positions[i][0]**2 + positions[i][1]**2, i) for i in range(len(model.moveable))]
            dists2.sort()
            selected = dists2[0][1] if dists2 else -1
        else:
            bar_idx, direction = action

            # Select bar if not already selected
            if bar_idx != selected:
                game_actions.append(make_click(bar_idx))
                selected = bar_idx

            # Move action: map grid direction to game action
            game_action = inv_remap[direction]
            game_actions.append(game_action)

            # Update position
            dx, dy = DIR_VECTORS[direction]
            ox, oy = positions[bar_idx]
            positions[bar_idx] = (ox + dx, oy + dy)

    return game_actions


def verify_with_game(game_actions):
    """Verify actions against real game engine."""
    sp80_path = 'B:/M/the-search/environment_files/sp80/0ee2d095'
    if sp80_path not in sys.path:
        sys.path.insert(0, sp80_path)
    from sp80 import Sp80
    from arcengine import GameAction, ActionInput, GameState

    ARCAGI3_TO_GA = {
        0: GameAction.ACTION1, 1: GameAction.ACTION2, 2: GameAction.ACTION3,
        3: GameAction.ACTION4, 4: GameAction.ACTION5, 5: GameAction.ACTION6,
        6: GameAction.ACTION7,
    }

    g = Sp80()
    g.full_reset()

    levels_completed = 0

    for i, action in enumerate(game_actions):
        if action < 7:
            ga = ARCAGI3_TO_GA[action]
            if ga == GameAction.ACTION6:
                ai = ActionInput(id=GameAction.ACTION6, data={'x': 0, 'y': 0})
            else:
                ai = ActionInput(id=ga, data={})
        else:
            ci = action - 7
            ai = ActionInput(id=GameAction.ACTION6, data={'x': ci % 64, 'y': ci // 64})

        try:
            r = g.perform_action(ai, raw=True)
        except Exception as e:
            print(f"    Error at action {i}: {e}")
            break

        if r is None:
            continue

        if r.levels_completed > levels_completed:
            levels_completed = r.levels_completed
            print(f"    Level {levels_completed} completed at action {i} (of {len(game_actions)})")

        if r.state == GameState.GAME_OVER:
            print(f"    GAME OVER at action {i}")
            break
        if r.state == GameState.WIN:
            print(f"    WIN at action {i}!")
            break

    return levels_completed


# ============================================================
# MAIN: Solve all 6 levels
# ============================================================

if __name__ == '__main__':
    print("="*60)
    print("SP80 Abstract Model Solver v2")
    print("="*60)

    t0_total = time.time()
    all_game_actions = []
    max_level_reached = 0

    for level_idx in range(6):
        print(f"\n{'='*50}")
        print(f"LEVEL {level_idx + 1}")
        print(f"{'='*50}")

        t0 = time.time()
        # Larger levels need more depth and deferred spill testing
        if level_idx <= 2:
            msd = 0  # test spill at every depth
            md = 25
        else:
            msd = 5  # defer spill testing for first 5 depths
            md = 20

        sol, model = solve_level_bfs(
            level_idx,
            max_depth=md,
            max_states=20000000,
            time_limit=240,
            min_spill_depth=msd
        )
        elapsed = time.time() - t0

        if sol is not None:
            game_actions = abstract_to_game_actions(sol, model)
            print(f"\n  Abstract: {sol} ({len(sol)} moves)")
            print(f"  Game:     {game_actions}")
            print(f"  Time:     {elapsed:.1f}s")
            all_game_actions.extend(game_actions)
            max_level_reached = level_idx + 1
        else:
            print(f"\n  UNSOLVED ({elapsed:.1f}s)")
            break

    total_elapsed = time.time() - t0_total

    print(f"\n{'='*50}")
    print(f"SUMMARY: L1-L{max_level_reached} solved, {len(all_game_actions)} total actions")
    print(f"Total time: {total_elapsed:.1f}s")

    # Verify with real game
    if all_game_actions:
        print(f"\nVerifying with real game engine...")
        verified = verify_with_game(all_game_actions)
        print(f"  Verified: {verified} levels completed")

        if verified == max_level_reached:
            # Save
            results = {
                'game': 'sp80',
                'method': 'abstract_model_bfs',
                'max_level': verified,
                'total_actions': len(all_game_actions),
                'sequence': all_game_actions,
                'levels': {},
            }
            out_path = 'B:/M/the-search/experiments/results/prescriptions/sp80_fullchain.json'
            with open(out_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved to {out_path}")
        else:
            print(f"\n  VERIFICATION MISMATCH: expected {max_level_reached}, got {verified}")
            print(f"  Sequence: {all_game_actions}")
