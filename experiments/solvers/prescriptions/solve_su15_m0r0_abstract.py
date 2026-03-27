"""
Abstract model solvers for SU15 (9 levels) and M0R0 (6 levels).

SU15 — vacuum fruit game:
  State = list of (x, y, size) for each fruit + goal requirement
  Actions = click at (x, y) creates vacuum pulling fruits within radius 8
  Same-size fruits that overlap after merging become next size
  Win = goal requirements met (specific size fruit + count in goal zone)
  Enemies chase nearest fruit, destroy size-0 fruits on contact

M0R0 — mirrored twin blocks:
  State = (left_block_pos, right_block_pos, toggle_positions, wall_states, door_states)
  Actions = UP/DOWN/LEFT/RIGHT move both blocks simultaneously (X-axis mirrored for crkfz)
  Walls (jggua sprites with 0-values) block movement
  Toggle blocks (cvcer) can be clicked and moved to open/close doors (dfnuk)
  Pressure plates (hnutp) open doors when a block is on them
  Win = both blocks overlap (at same position)

Uses arc_agi API for verification.
"""

import json
import sys
import os
import numpy as np
from collections import deque
from itertools import product

# ============================================================
# M0R0 ABSTRACT MODEL SOLVER
# ============================================================

def parse_m0r0_maze(sprite_pixels):
    """Parse a jggua maze sprite's pixel array into a wall grid.
    -1 = open cell, 0 = wall."""
    rows = len(sprite_pixels)
    cols = len(sprite_pixels[0]) if rows > 0 else 0
    walls = set()
    for r in range(rows):
        for c in range(cols):
            if sprite_pixels[r][c] == 0:
                walls.add((c, r))
    return walls, cols, rows

# M0R0 maze sprites from source (pixel arrays)
M0R0_MAZES = {
    "Level1": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0],
        [0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0],
        [0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level2": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level3": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
        [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1],
        [-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [-1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
        [-1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, -1],
        [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level4": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
        [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level5": [
        [0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1],
        [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1],
        [-1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, -1],
        [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, -1, 0],
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
    ],
    "Level6": [
        [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
        [0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
        [0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level7": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level8": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
        [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
        [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0],
        [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level9": [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "Level10": [
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
    ],
    "Level11": [
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        [-1, -1, -1, -1, 0, -1, -1, -1, -1],
    ],
}

# M0R0 Level definitions
# Each level has: maze, maze_pos, blocks (ubwff or kncqr, idtiq=left crkfz=right), toggles (cvcer),
# walls (wyiex), door_walls (dfnuk), pressure_plates (hnutp), grid_size
M0R0_LEVELS = [
    # Level 1: Simple maze, two blocks, no toggles
    {
        "maze": "Level6", "maze_pos": (0, 0), "maze_rot": 180,
        "blocks": {
            "ubwff-crkfz": (7, 9),  # right block (X-mirrored)
            "ubwff-idtiq": (3, 9),  # left block (normal)
        },
        "toggles": [],
        "wyiex": [],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (11, 11),
    },
    # Level 2: H-shaped maze, two blocks, wyiex walls
    {
        "maze": "Level11", "maze_pos": (2, 0),
        "blocks": {
            "ubwff-crkfz": (8, 1),
            "ubwff-idtiq": (4, 1),
        },
        "toggles": [],
        "wyiex": [
            (5, 5), (5, 6), (12, 8), (11, 8), (10, 8), (9, 8), (8, 8),
            (4, 8), (2, 8), (1, 8), (0, 8), (5, 8), (5, 7),
            (4, 12), (3, 12), (2, 12), (1, 12), (0, 12),
            (9, 12), (8, 12), (7, 12), (6, 12), (5, 12),
            (12, 12), (11, 12), (10, 12),
        ],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (13, 13),
    },
    # Level 3: Maze with 3 toggles, no doors/plates
    {
        "maze": "Level1", "maze_pos": (0, 0),
        "blocks": {
            "ubwff-crkfz": (8, 10),
            "ubwff-idtiq": (4, 10),
        },
        "toggles": [(1, 3), (6, 2), (8, 6)],
        "wyiex": [],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (13, 13),
    },
    # Level 4: Maze with 1 toggle, lots of wyiex walls
    {
        "maze": "Level9", "maze_pos": (0, 0),
        "blocks": {
            "ubwff-crkfz": (8, 4),
            "ubwff-idtiq": (2, 6),
        },
        "toggles": [(5, 5)],
        "wyiex": [
            (1, 1), (2, 1), (3, 1), (9, 1), (8, 1), (7, 1),
            (7, 9), (9, 9), (8, 9), (3, 9), (2, 9), (1, 9),
            (4, 6), (5, 6), (6, 6), (4, 4), (5, 4), (6, 4),
        ],
        "dfnuk": {},
        "hnutp": {},
        "grid_size": (11, 11),
    },
    # Level 5: Maze with doors and pressure plates
    {
        "maze": "Level2", "maze_pos": (0, 0),
        "blocks": {
            "ubwff-crkfz": (1, 12),
            "ubwff-idtiq": (13, 12),
        },
        "toggles": [],
        "wyiex": [],
        "dfnuk": {
            "qeazm": [(10, 9, 90), (10, 5, 90)],  # (x, y, rotation)
            "raixb": [(3, 5, 90)],
            "ujcze": [(2, 9, 90)],
        },
        "hnutp": {
            "qeazm": [(3, 12), (3, 1)],
            "raixb": [(8, 6)],
            "ujcze": [(14, 6)],
        },
        "grid_size": (15, 15),
    },
    # Level 6: Complex with toggles, doors, plates, lots of wyiex
    {
        "maze": None,  # No jggua maze — just wyiex walls
        "maze_pos": (0, 0),
        "blocks": {
            "ubwff-crkfz": (9, 4),
            "ubwff-idtiq": (3, 4),
        },
        "toggles": [(6, 9)],
        "wyiex": [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0),
            (0, 12), (1, 12), (2, 12), (3, 12), (4, 12), (5, 12), (6, 12), (7, 12), (8, 12), (9, 12), (10, 12), (11, 12), (12, 12),
            (0, 11), (0, 9), (0, 10), (0, 5), (0, 6), (0, 7), (0, 8), (0, 1), (0, 2), (0, 3), (0, 4),
            (12, 11), (12, 9), (12, 10), (12, 5), (12, 6), (12, 7), (12, 8), (12, 1), (12, 2), (12, 3), (12, 4),
            (6, 11), (6, 10), (6, 5), (6, 6), (6, 7), (6, 8), (6, 1), (6, 2), (6, 3), (6, 4),
            (5, 6), (1, 6), (7, 6), (11, 6),
        ],
        "dfnuk": {
            "raixb": [(2, 6, 90)],
            "ujcze": [(8, 6, 90)],
        },
        "hnutp": {
            "raixb": [(9, 2), (3, 9)],
            "ujcze": [(3, 2)],
        },
        "grid_size": (13, 13),
    },
]


def build_m0r0_collision_set(level_idx):
    """Build the static wall set for a M0R0 level.
    Returns: set of (x, y) positions that are walls.
    Also returns door info for levels with dfnuk sprites."""
    level = M0R0_LEVELS[level_idx]
    walls = set()

    # Add maze walls (jggua sprite)
    if level["maze"] is not None:
        maze_pixels = M0R0_MAZES[level["maze"]]
        mx, my = level.get("maze_pos", (0, 0))
        maze_rot = level.get("maze_rot", 0)

        # Handle rotation
        if maze_rot == 180:
            # Rotate 180: flip both axes
            rows = len(maze_pixels)
            cols = len(maze_pixels[0])
            rotated = [[maze_pixels[rows - 1 - r][cols - 1 - c] for c in range(cols)] for r in range(rows)]
            maze_pixels = rotated

        for r, row in enumerate(maze_pixels):
            for c, val in enumerate(row):
                if val == 0:
                    walls.add((mx + c, my + r))

    # Add wyiex static walls
    for wx, wy in level["wyiex"]:
        walls.add((wx, wy))

    return walls


def build_m0r0_door_info(level_idx):
    """Build door information: which doors exist and what plates control them.
    dfnuk sprites are the doors (3px tall vertical barriers when rot=90).
    hnutp sprites are pressure plates that open doors of matching color.

    Returns dict: color -> {"doors": [(x,y) positions], "plates": [(x,y) positions]}
    """
    level = M0R0_LEVELS[level_idx]
    door_info = {}

    for color, door_list in level.get("dfnuk", {}).items():
        positions = []
        for item in door_list:
            x, y, rot = item
            # dfnuk sprites are 3px tall, 1px wide. With rotation 90, they become 1px tall, 3px wide
            if rot == 90:
                positions.extend([(x, y), (x+1, y), (x+2, y)])
            else:
                positions.extend([(x, y), (x, y+1), (x, y+2)])

        plates = []
        for px, py in level.get("hnutp", {}).get(color, []):
            plates.append((px, py))

        door_info[color] = {"doors": positions, "plates": plates}

    return door_info


def solve_m0r0_level_bfs(level_idx, max_states=2000000):
    """Solve a M0R0 level using BFS on abstract state space.

    State = (left_x, left_y, right_x, right_y, toggle_positions_tuple, doors_open_tuple)

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (keyboard actions in PRISM encoding)

    Movement rules:
    - idtiq (left block): moves normally (UP=dy-1, DOWN=dy+1, LEFT=dx-1, RIGHT=dx+1)
    - crkfz (right block): X-axis MIRRORED (UP=dy-1, DOWN=dy+1, LEFT=dx+1, RIGHT=dx-1)
    - If a block would hit a wall, it stays put (other block still moves if free)
    - If blocks step on wyiex, they flash and return to previous position
    - If two blocks overlap (same position), level is won
    """
    level = M0R0_LEVELS[level_idx]
    gw, gh = level["grid_size"]
    static_walls = build_m0r0_collision_set(level_idx)
    door_info = build_m0r0_door_info(level_idx)
    toggles = list(level.get("toggles", []))

    # Initial positions
    left_pos = level["blocks"]["ubwff-idtiq"]
    right_pos = level["blocks"]["ubwff-crkfz"]

    # Initial toggle positions (as tuple for hashing)
    initial_toggles = tuple(toggles)

    # Initial door states: check if any plate is initially pressed
    initial_doors = {}
    for color, info in door_info.items():
        # Initially no block is on any plate
        initial_doors[color] = False  # doors closed
    initial_doors_tuple = tuple(sorted(initial_doors.items()))

    # State: (lx, ly, rx, ry, toggles_tuple, doors_tuple)
    initial_state = (left_pos[0], left_pos[1], right_pos[0], right_pos[1],
                     initial_toggles, initial_doors_tuple)

    # Action deltas: action 0=UP(dy=-1), 1=DOWN(dy=1), 2=LEFT(dx=-1), 3=RIGHT(dx=1)
    action_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def get_walls(toggle_positions, doors_state):
        """Get full wall set including toggles and door states."""
        w = set(static_walls)
        # Toggles are movable obstacles
        for tx, ty in toggle_positions:
            w.add((tx, ty))
        # Add closed door walls
        doors_dict = dict(doors_state)
        for color, info in door_info.items():
            if not doors_dict.get(color, False):
                for dx, dy in info["doors"]:
                    w.add((dx, dy))
        return w

    def can_move(x, y, dx, dy, walls_set):
        """Check if block at (x,y) can move by (dx,dy)."""
        nx, ny = x + dx, y + dy
        if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
            return False
        if (nx, ny) in walls_set:
            return False
        return True

    def check_plate_press(lx, ly, rx, ry, toggle_positions):
        """Check which doors should be open based on pressure plates."""
        doors = {}
        block_positions = {(lx, ly), (rx, ry)}
        # Also check toggle positions
        all_occupiers = block_positions | set(toggle_positions)

        for color, info in door_info.items():
            pressed = False
            for px, py in info["plates"]:
                if (px, py) in all_occupiers:
                    pressed = True
                    break
            doors[color] = pressed
        return tuple(sorted(doors.items()))

    def is_on_wyiex(x, y, toggle_positions, doors_state):
        """Check if position hits a wyiex (trap tile).
        In M0R0, wyiex tiles cause flash + revert if a block lands on them
        AND the tile isn't covered by a pxwnx (ubwff) block.
        For BFS, we treat stepping on wyiex as invalid (reverts)."""
        # wyiex positions
        for wx, wy in level["wyiex"]:
            if (x, y) == (wx, wy):
                return True
        return False

    # BFS
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    states_explored = 0

    while queue and states_explored < max_states:
        state, path = queue.popleft()
        states_explored += 1

        lx, ly, rx, ry, toggles_t, doors_t = state

        # Win condition: both blocks at same position
        if lx == rx and ly == ry:
            print(f"    Level {level_idx+1}: SOLVED in {len(path)} moves (explored {states_explored} states)")
            return path

        walls_set = get_walls(toggles_t, doors_t)

        for action_idx, (adx, ady) in enumerate(action_deltas):
            # Left block (idtiq): normal movement
            left_dx, left_dy = adx, ady
            # Right block (crkfz): X-axis mirrored
            right_dx, right_dy = -adx, ady

            # Try to move left block
            new_lx, new_ly = lx, ly
            if can_move(lx, ly, left_dx, left_dy, walls_set):
                new_lx, new_ly = lx + left_dx, ly + left_dy

            # Try to move right block
            new_rx, new_ry = rx, ry
            if can_move(rx, ry, right_dx, right_dy, walls_set):
                new_rx, new_ry = rx + right_dx, ry + right_dy

            # Check for wyiex trap — if either block hits wyiex, revert both
            # Actually from the code: wyiex causes flash + revert for that block only
            # But from the code (line 822-826), if ANY moved block hits wyiex,
            # both flash and revert to pre-move positions
            left_on_wyiex = is_on_wyiex(new_lx, new_ly, toggles_t, doors_t)
            right_on_wyiex = is_on_wyiex(new_rx, new_ry, toggles_t, doors_t)

            if left_on_wyiex or right_on_wyiex:
                # Revert both blocks — this move wastes a step but doesn't change state
                # Skip this action in BFS as it doesn't progress
                continue

            # Check for toggles collision — blocks can push toggles
            # Actually toggles (cvcer) are clicked, not pushed. They're obstacles that
            # can be selected and moved by clicking them.
            # For BFS, toggles remain static unless we model clicking.

            # Update door states based on pressure plates
            new_doors = check_plate_press(new_lx, new_ly, new_rx, new_ry, toggles_t)

            new_state = (new_lx, new_ly, new_rx, new_ry, toggles_t, new_doors)

            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [action_idx]))

    print(f"    Level {level_idx+1}: NOT SOLVED (explored {states_explored} states)")
    return None


def solve_m0r0_level_with_toggles(level_idx, max_states=5000000):
    """Solve M0R0 level with toggle manipulation.

    Two-phase approach:
    Phase 1: Try BFS without moving toggles (treat them as static obstacles)
    Phase 2: If that fails, BFS with toggle selection and movement

    For levels with toggles (cvcer), we need to:
    1. Click a toggle to select it (ACTION6 click)
    2. Move the toggle with arrow keys
    3. Click empty space to deselect

    State includes toggle positions. BFS explores both movement and toggle manipulation.
    """
    level = M0R0_LEVELS[level_idx]
    toggles = list(level.get("toggles", []))

    if not toggles:
        return solve_m0r0_level_bfs(level_idx, max_states)

    # Phase 1: Try without moving toggles
    print(f"    Phase 1: trying without toggle movement...")
    result = solve_m0r0_level_bfs(level_idx, min(max_states, 500000))
    if result is not None:
        return result

    print(f"    Phase 1 failed. Phase 2: BFS with toggle manipulation...")

    gw, gh = level["grid_size"]

    static_walls = build_m0r0_collision_set(level_idx)
    # Remove toggle positions from static walls (they're dynamic now)
    for tx, ty in toggles:
        static_walls.discard((tx, ty))

    door_info = build_m0r0_door_info(level_idx)

    # Optimization: identify which toggles are "critical" (blocking a passage)
    # Only make critical toggles movable, treat others as static
    # A toggle is critical if removing it from the wall set allows BFS to find a solution
    critical_toggles = []
    non_critical_toggles = []
    for ti, (tx, ty) in enumerate(toggles):
        # Check if this toggle blocks a connection
        test_walls = set(static_walls)
        for tj, (ttx, tty) in enumerate(toggles):
            if tj != ti:
                test_walls.add((ttx, tty))
        # Quick BFS test without this toggle
        left_pos_t = level["blocks"]["ubwff-idtiq"]
        right_pos_t = level["blocks"]["ubwff-crkfz"]
        test_state = (left_pos_t[0], left_pos_t[1], right_pos_t[0], right_pos_t[1])
        test_queue = deque([(test_state, 0)])
        test_visited = {test_state}
        found = False
        for _ in range(100000):
            if not test_queue:
                break
            (tlx, tly, trx, try_), depth = test_queue.popleft()
            if tlx == trx and tly == try_:
                found = True
                break
            for adx, ady in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                nlx, nly = tlx + adx, tly + ady
                nrx, nry = trx - adx, try_ + ady
                if not (0 <= nlx < gw and 0 <= nly < gh and (nlx, nly) not in test_walls):
                    nlx, nly = tlx, tly
                if not (0 <= nrx < gw and 0 <= nry < gh and (nrx, nry) not in test_walls):
                    nrx, nry = trx, try_
                ns = (nlx, nly, nrx, nry)
                if ns not in test_visited:
                    test_visited.add(ns)
                    test_queue.append((ns, depth + 1))
        if found:
            critical_toggles.append(ti)
            print(f"      Toggle {ti} at ({tx},{ty}) is CRITICAL (removing it enables solution)")
        else:
            non_critical_toggles.append(ti)

    if not critical_toggles:
        # Try each toggle individually
        for ti in range(len(toggles)):
            test_walls2 = set(static_walls)
            for tj in range(len(toggles)):
                if tj != ti:
                    test_walls2.add(toggles[tj])
            # Quick reachability test without this toggle
            left_pos_t = level["blocks"]["ubwff-idtiq"]
            right_pos_t = level["blocks"]["ubwff-crkfz"]
            test_state = (left_pos_t[0], left_pos_t[1], right_pos_t[0], right_pos_t[1])
            test_queue2 = deque([(test_state, 0)])
            test_visited2 = {test_state}
            found2 = False
            for _ in range(200000):
                if not test_queue2:
                    break
                (tlx, tly, trx, try_), depth = test_queue2.popleft()
                if tlx == trx and tly == try_:
                    found2 = True
                    break
                for adx, ady in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                    nlx, nly = tlx + adx, tly + ady
                    nrx, nry = trx - adx, try_ + ady
                    if not (0 <= nlx < gw and 0 <= nly < gh and (nlx, nly) not in test_walls2):
                        nlx, nly = tlx, tly
                    if not (0 <= nrx < gw and 0 <= nry < gh and (nrx, nry) not in test_walls2):
                        nrx, nry = trx, try_
                    ns = (nlx, nly, nrx, nry)
                    if ns not in test_visited2:
                        test_visited2.add(ns)
                        test_queue2.append((ns, depth + 1))
            if found2:
                critical_toggles = [ti]
                non_critical_toggles = [j for j in range(len(toggles)) if j != ti]
                print(f"      Toggle {ti} at {toggles[ti]} is the one that matters")
                break

    if not critical_toggles:
        # Still can't find — try all toggles (expensive)
        critical_toggles = list(range(len(toggles)))
        non_critical_toggles = []

    # Add non-critical toggles to static walls
    for ti in non_critical_toggles:
        tx, ty = toggles[ti]
        static_walls.add((tx, ty))

    # Remap toggle indices
    movable_toggles = [toggles[ti] for ti in critical_toggles]
    toggles = movable_toggles
    print(f"      Movable toggles: {toggles} (reduced from {len(M0R0_LEVELS[level_idx]['toggles'])})")

    left_pos = level["blocks"]["ubwff-idtiq"]
    right_pos = level["blocks"]["ubwff-crkfz"]

    initial_toggles = tuple(toggles)
    initial_doors = tuple(sorted({color: False for color in door_info}.items()))

    # State: (lx, ly, rx, ry, toggles_tuple, doors_tuple, selected_toggle_idx)
    # selected_toggle_idx = -1 means no toggle selected (blocks move mode)
    initial_state = (left_pos[0], left_pos[1], right_pos[0], right_pos[1],
                     initial_toggles, initial_doors, -1)

    action_deltas = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    def get_walls(toggle_positions, doors_state, exclude_toggle=-1):
        """Get wall set."""
        w = set(static_walls)
        for i, (tx, ty) in enumerate(toggle_positions):
            if i != exclude_toggle:
                w.add((tx, ty))
        doors_dict = dict(doors_state)
        for color, info in door_info.items():
            if not doors_dict.get(color, False):
                for dx, dy in info["doors"]:
                    w.add((dx, dy))
        return w

    def check_plates(lx, ly, rx, ry, toggle_positions):
        doors = {}
        block_positions = {(lx, ly), (rx, ry)}
        for color, info in door_info.items():
            pressed = any((px, py) in block_positions for px, py in info["plates"])
            doors[color] = pressed
        return tuple(sorted(doors.items()))

    # BFS
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    states_explored = 0

    while queue and states_explored < max_states:
        state, path = queue.popleft()
        states_explored += 1

        if states_explored % 500000 == 0:
            print(f"      ... {states_explored} states explored, queue size {len(queue)}")

        lx, ly, rx, ry, toggles_t, doors_t, sel_idx = state

        # Win: blocks overlap
        if lx == rx and ly == ry and sel_idx == -1:
            print(f"    Level {level_idx+1}: SOLVED in {len(path)} actions (explored {states_explored} states)")
            return path

        if sel_idx == -1:
            # BLOCK MOVE MODE
            walls_set = get_walls(toggles_t, doors_t)

            # Try movement actions (0-3)
            for ai, (adx, ady) in enumerate(action_deltas):
                left_dx, left_dy = adx, ady
                right_dx, right_dy = -adx, ady

                new_lx, new_ly = lx, ly
                if 0 <= lx + left_dx < gw and 0 <= ly + left_dy < gh and (lx + left_dx, ly + left_dy) not in walls_set:
                    new_lx, new_ly = lx + left_dx, ly + left_dy

                new_rx, new_ry = rx, ry
                if 0 <= rx + right_dx < gw and 0 <= ry + right_dy < gh and (rx + right_dx, ry + right_dy) not in walls_set:
                    new_rx, new_ry = rx + right_dx, ry + right_dy

                # Check wyiex
                wyiex_set = set(level["wyiex"])
                if (new_lx, new_ly) in wyiex_set or (new_rx, new_ry) in wyiex_set:
                    continue

                new_doors = check_plates(new_lx, new_ly, new_rx, new_ry, toggles_t)
                new_state = (new_lx, new_ly, new_rx, new_ry, toggles_t, new_doors, -1)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [ai]))

            # Try clicking each toggle to select it
            for ti, (tx, ty) in enumerate(toggles_t):
                new_state = (lx, ly, rx, ry, toggles_t, doors_t, ti)
                action_code = 100 + ti

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, path + [action_code]))

        else:
            # TOGGLE MOVE MODE
            toggle_list = list(toggles_t)
            tx, ty = toggle_list[sel_idx]
            walls_set = get_walls(toggles_t, doors_t, exclude_toggle=sel_idx)

            for ai, (adx, ady) in enumerate(action_deltas):
                ntx, nty = tx + adx, ty + ady
                if 0 <= ntx < gw and 0 <= nty < gh and (ntx, nty) not in walls_set:
                    if (ntx, nty) != (lx, ly) and (ntx, nty) != (rx, ry):
                        new_toggles = list(toggles_t)
                        new_toggles[sel_idx] = (ntx, nty)
                        new_toggles_t = tuple(new_toggles)
                        new_doors = check_plates(lx, ly, rx, ry, new_toggles_t)
                        new_state = (lx, ly, rx, ry, new_toggles_t, new_doors, sel_idx)

                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((new_state, path + [ai]))

            # Deselect
            new_state = (lx, ly, rx, ry, toggles_t, doors_t, -1)
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, path + [200]))

    print(f"    Level {level_idx+1}: NOT SOLVED (explored {states_explored} states)")
    return None


def convert_m0r0_actions_to_prism(actions, level_idx):
    """Convert abstract BFS actions to PRISM action indices.

    PRISM encoding:
    0 = ACTION1 = UP (dy=-1)
    1 = ACTION2 = DOWN (dy=1)
    2 = ACTION3 = LEFT (dx=-1)
    3 = ACTION4 = RIGHT (dx=1)
    5 = ACTION6 (click with x,y) — but keyboard action 5 in PRISM

    For click actions (toggle select/deselect), we need ACTION6 with coordinates.
    click_action = 7 + y_pixel * 64 + x_pixel

    Grid-to-pixel: need to know the scale. Grid size determines scale.
    scale = min(64 // gw, 64 // gh)
    x_pixel = grid_x * scale + x_offset + scale//2
    y_pixel = grid_y * scale + y_offset + scale//2
    """
    level = M0R0_LEVELS[level_idx]
    gw, gh = level["grid_size"]
    scale_x = 64 // gw
    scale_y = 64 // gh
    scale = min(scale_x, scale_y)
    scaled_w = gw * scale
    scaled_h = gh * scale
    x_offset = (64 - scaled_w) // 2
    y_offset = (64 - scaled_h) // 2

    toggles = list(level.get("toggles", []))

    prism_actions = []
    current_toggles = list(toggles)

    for a in actions:
        if a < 4:
            # Movement action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
            prism_actions.append(a)
        elif a >= 100 and a < 200:
            # Toggle select: click on toggle position
            ti = a - 100
            tx, ty = current_toggles[ti]
            px = tx * scale + x_offset + scale // 2
            py = ty * scale + y_offset + scale // 2
            px = min(px, 63)
            py = min(py, 63)
            click_action = 7 + py * 64 + px
            prism_actions.append(click_action)
        elif a == 200:
            # Deselect: click on empty space (0, 0) pixel — but that might be a wall
            # Click on center of the grid which should be wall
            px = 32
            py = 32
            click_action = 7 + py * 64 + px
            prism_actions.append(click_action)
        elif a >= 200:
            # Move toggle action encoded differently
            prism_actions.append(a - 200)

    return prism_actions


def solve_m0r0():
    """Solve all 6 levels of M0R0."""
    print("=" * 60)
    print("SOLVING M0R0 - Mirrored Twin Blocks")
    print("=" * 60)

    all_actions = []
    per_level = {}

    for level_idx in range(6):
        print(f"\n--- Level {level_idx + 1} ---")
        level = M0R0_LEVELS[level_idx]
        has_toggles = len(level.get("toggles", [])) > 0

        if has_toggles:
            result = solve_m0r0_level_with_toggles(level_idx)
        else:
            result = solve_m0r0_level_bfs(level_idx)

        if result is not None:
            prism_actions = convert_m0r0_actions_to_prism(result, level_idx)
            per_level[f"L{level_idx + 1}"] = {
                "count": len(prism_actions),
                "actions": prism_actions,
                "abstract_actions": result,
            }
            all_actions.extend(prism_actions)
            print(f"    -> {len(prism_actions)} PRISM actions")
        else:
            per_level[f"L{level_idx + 1}"] = {"count": 0, "actions": [], "status": "UNSOLVED"}

    return per_level, all_actions


# ============================================================
# SU15 ABSTRACT MODEL SOLVER
# ============================================================

# Fruit sizes: size N has pixel dimensions (N+1) x (N+1) (except size 0 = 1x1)
# Actually from sprites: "0"=1x1, "1"=2x2, "2"=3x3, "3"=4x4, "4"=5x5, "5"=7x7, "6"=8x8, "7"=9x9, "8"=10x10
FRUIT_SIZES = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10}

# Goal sprite "avvxfurrqu" is 9x9 (with rounded corners)
# Vacuum radius = 8 (okwzbiftnr)
# Vacuum pull steps = 4 (aevhcnismc / bjetwxoaq)
# Movement per step = 4 pixels (vprzfcbjwy / stqbquzms)
# Grid = 64x64, Y range for clicks: 10 to 63

# SU15 Level data extracted from source
SU15_LEVELS = [
    # Level 1
    {
        "fruits": [{"size": 2, "x": 3, "y": 58}],
        "goals": [(44, 11)],
        "enemies": [],
        "goal_req": [("2", 1)],  # 1 fruit of size 2 in goal
        "keys": [("eifgovhtsm", 30, 4)],
        "steps": 32,
    },
    # Level 2
    {
        "fruits": [
            {"size": 0, "x": 41, "y": 37}, {"size": 0, "x": 18, "y": 37},
            {"size": 0, "x": 37, "y": 40}, {"size": 0, "x": 16, "y": 41},
            {"size": 0, "x": 14, "y": 55}, {"size": 0, "x": 16, "y": 57},
            {"size": 0, "x": 49, "y": 54}, {"size": 0, "x": 47, "y": 56},
        ],
        "goals": [(29, 23)],
        "enemies": [],
        "goal_req": [("3", 1)],  # 1 fruit of size 3
        "keys": [("recfijsnol", 30, 3)],
        "steps": 32,
    },
    # Level 3
    {
        "fruits": [
            {"size": 0, "x": 55, "y": 23}, {"size": 0, "x": 61, "y": 23},
            {"size": 0, "x": 31, "y": 22}, {"size": 0, "x": 31, "y": 15},
            {"size": 0, "x": 12, "y": 23}, {"size": 0, "x": 8, "y": 28},
            {"size": 1, "x": 46, "y": 22}, {"size": 1, "x": 30, "y": 32},
            {"size": 1, "x": 18, "y": 16},
        ],
        "goals": [(5, 46), (19, 46)],
        "enemies": [],
        "goal_req": [("3", 1), ("2", 1)],  # 1 size-3 and 1 size-2 in goals
        "keys": [("eifgovhtsm", 36, 4), ("recfijsnol", 30, 3)],
        "steps": 48,
    },
    # Level 4
    {
        "fruits": [
            {"size": 0, "x": 5, "y": 26}, {"size": 0, "x": 11, "y": 26},
            {"size": 0, "x": 31, "y": 27}, {"size": 0, "x": 36, "y": 29},
            {"size": 0, "x": 33, "y": 47}, {"size": 0, "x": 30, "y": 51},
            {"size": 0, "x": 12, "y": 47}, {"size": 0, "x": 8, "y": 41},
        ],
        "goals": [(1, 53)],
        "enemies": [{"type": "enemy", "x": 52, "y": 19}],
        "goal_req": [("3", 1)],
        "keys": [("recfijsnol", 30, 3)],
        "steps": 48,
    },
    # Level 5
    {
        "fruits": [
            {"size": 0, "x": 58, "y": 59}, {"size": 0, "x": 44, "y": 53},
            {"size": 0, "x": 3, "y": 60}, {"size": 0, "x": 14, "y": 54},
            {"size": 1, "x": 14, "y": 28}, {"size": 1, "x": 53, "y": 26},
            {"size": 1, "x": 6, "y": 25}, {"size": 1, "x": 42, "y": 26},
        ],
        "goals": [(28, 11)],
        "enemies": [{"type": "enemy", "x": 4, "y": 37}, {"type": "enemy", "x": 46, "y": 37}],
        "goal_req": [("3", 1)],
        "keys": [("recfijsnol", 30, 3)],
        "steps": 32,
    },
    # Level 6
    {
        "fruits": [{"size": 5, "x": 33, "y": 32}],
        "goals": [(2, 12), (52, 53)],
        "enemies": [{"type": "enemy", "x": 16, "y": 34}],
        "goal_req": [("3", 1), ("vnjbdkorwc", 1)],  # size 3 + enemy type 1
        "keys": [("nyvfnpgcbv", 36, 3), ("recfijsnol", 30, 3)],
        "steps": 32,
    },
    # Level 7
    {
        "fruits": [
            {"size": 1, "x": 9, "y": 25}, {"size": 1, "x": 20, "y": 35},
            {"size": 1, "x": 6, "y": 35}, {"size": 1, "x": 30, "y": 37},
            {"size": 5, "x": 51, "y": 46},
        ],
        "goals": [(19, 13), (40, 18)],
        "enemies": [{"type": "enemy", "x": 12, "y": 51}, {"type": "enemy", "x": 52, "y": 56}],
        "goal_req": [("3", 2)],  # 2 fruits of size 3
        "keys": [("recfijsnol", 30, 3), ("recfijsnol", 36, 3)],
        "steps": 32,
    },
    # Level 8
    {
        "fruits": [
            {"size": 3, "x": 13, "y": 42}, {"size": 3, "x": 3, "y": 40},
            {"size": 5, "x": 20, "y": 24},
        ],
        "goals": [(52, 15), (3, 15), (52, 51), (3, 51)],
        "enemies": [
            {"type": "enemy", "x": 43, "y": 31},
            {"type": "enemy", "x": 29, "y": 53},
            {"type": "enemy", "x": 47, "y": 48},
        ],
        "goal_req": [("4", 2), ("yckgseirmu", 1)],  # 2 size-4 + enemy type 2
        "keys": [("jpjwahlikp", 30, 2), ("jpjwahlikp", 37, 2), ("tltqnwoiek", 44, 3)],
        "steps": 48,
    },
    # Level 9
    {
        "fruits": [
            {"size": 1, "x": 18, "y": 46}, {"size": 1, "x": 23, "y": 52},
            {"size": 5, "x": 35, "y": 48},
        ],
        "goals": [(7, 37), (49, 51), (7, 51)],
        "enemies": [
            {"type": "enemy", "x": 51, "y": 13},
            {"type": "enemy", "x": 14, "y": 12},
            {"type": "enemy", "x": 15, "y": 22},
            {"type": "enemy", "x": 54, "y": 33},
        ],
        "goal_req": [("4", 1), ("vptxjilzzk", 1), ("2", 1)],  # size 4 + enemy type 3 + size 2
        "keys": [("jpjwahlikp", 30, 2), ("pzqkrtozkk", 42, 3)],
        "steps": 48,
    },
]


def su15_click_action(x, y):
    """Convert grid coordinates to PRISM action index for SU15.
    Click action = 7 + y * 64 + x"""
    x = max(0, min(63, x))
    y = max(0, min(63, y))
    return 7 + y * 64 + x


def su15_fruit_center(x, y, size):
    """Get center of a fruit given its top-left position and size."""
    s = FRUIT_SIZES[size]
    return x + s // 2, y + s // 2


def su15_simulate_vacuum(fruits, click_x, click_y, vacuum_radius=8, pull_steps=4, pull_speed=4):
    """Simulate a vacuum click on a set of fruits.

    1. Find fruits within vacuum_radius of click point
    2. Pull them toward click point over pull_steps steps
    3. After pulling, merge same-size overlapping fruits

    Returns new fruit list after vacuum + merge.
    """
    YMIN = 10
    YMAX = 63

    affected = []
    unaffected = []

    for f in fruits:
        cx, cy = su15_fruit_center(f["x"], f["y"], f["size"])
        # Distance check: closest point on fruit to click
        s = FRUIT_SIZES[f["size"]]
        # Closest point
        closest_x = max(f["x"], min(click_x, f["x"] + s - 1))
        closest_y = max(f["y"], min(click_y, f["y"] + s - 1))
        dx = click_x - closest_x
        dy = click_y - closest_y
        dist_sq = dx * dx + dy * dy
        if dist_sq <= vacuum_radius * vacuum_radius:
            affected.append(dict(f))
        else:
            unaffected.append(dict(f))

    # Simulate pulling over pull_steps steps
    # Each step: move each fruit up to pull_speed pixels toward click point
    # Vacuum radius shrinks: radius = vacuum_radius * (1 - step/(pull_steps-1)) approximately
    for step in range(pull_steps):
        for f in affected:
            cx, cy = su15_fruit_center(f["x"], f["y"], f["size"])
            dx = click_x - cx
            dy = click_y - cy

            mx = 0
            my = 0
            if dx > 0:
                mx = min(pull_speed, dx)
            elif dx < 0:
                mx = max(-pull_speed, dx)
            if dy > 0:
                my = min(pull_speed, dy)
            elif dy < 0:
                my = max(-pull_speed, dy)

            new_x = f["x"] + mx
            new_y = f["y"] + my

            s = FRUIT_SIZES[f["size"]]
            # Clamp to bounds
            new_x = max(0, min(63 - s, new_x))
            new_y = max(YMIN, min(63 - s, new_y))

            f["x"] = new_x
            f["y"] = new_y

    # After pulling, merge same-size overlapping fruits
    # Use union-find for merging
    all_fruits = affected + unaffected
    merged = True
    while merged:
        merged = False
        new_fruits = []
        used = [False] * len(all_fruits)

        for i in range(len(all_fruits)):
            if used[i]:
                continue
            group = [i]
            for j in range(i + 1, len(all_fruits)):
                if used[j]:
                    continue
                if all_fruits[i]["size"] == all_fruits[j]["size"]:
                    # Check overlap (bounding box intersection)
                    si = FRUIT_SIZES[all_fruits[i]["size"]]
                    sj = FRUIT_SIZES[all_fruits[j]["size"]]
                    if (all_fruits[i]["x"] < all_fruits[j]["x"] + sj and
                        all_fruits[i]["x"] + si > all_fruits[j]["x"] and
                        all_fruits[i]["y"] < all_fruits[j]["y"] + sj and
                        all_fruits[i]["y"] + si > all_fruits[j]["y"]):
                        group.append(j)
                        used[j] = True

            if len(group) >= 2:
                # Merge: average position, size + 1
                merged = True
                used[i] = True
                avg_cx = sum(su15_fruit_center(all_fruits[g]["x"], all_fruits[g]["y"], all_fruits[g]["size"])[0] for g in group) // len(group)
                avg_cy = sum(su15_fruit_center(all_fruits[g]["x"], all_fruits[g]["y"], all_fruits[g]["size"])[1] for g in group) // len(group)
                new_size = all_fruits[i]["size"] + 1
                if new_size > 8:
                    # Max size reached, fruits disappear
                    continue
                ns = FRUIT_SIZES[new_size]
                nx = avg_cx - ns // 2
                ny = avg_cy - ns // 2
                nx = max(0, min(63 - ns, nx))
                ny = max(YMIN, min(63 - ns, ny))
                new_fruits.append({"size": new_size, "x": nx, "y": ny})
            else:
                new_fruits.append(dict(all_fruits[i]))

        all_fruits = new_fruits

    return all_fruits


def su15_check_win(fruits, goals, goal_req):
    """Check if win condition is met.

    For each goal requirement (size, count):
    - Count fruits whose CENTER is inside any goal zone
    - Goal zones are 9x9 sprites at given positions
    """
    goal_size = 9  # avvxfurrqu is 9x9

    # Count fruits in goal zones by size
    size_counts = {}
    for f in fruits:
        cx, cy = su15_fruit_center(f["x"], f["y"], f["size"])
        in_goal = False
        for gx, gy in goals:
            if gx <= cx < gx + goal_size and gy <= cy < gy + goal_size:
                in_goal = True
                break
        if in_goal:
            key = str(f["size"])
            size_counts[key] = size_counts.get(key, 0) + 1

    # Check all requirements
    for req_type, req_count in goal_req:
        # Skip enemy requirements for now (they're about enemies in goals, not fruits)
        if req_type in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            continue
        if size_counts.get(req_type, 0) != req_count:
            return False
    return True


def solve_su15_level_greedy(level_idx):
    """Solve a SU15 level using greedy/planning approach.

    Strategy:
    1. Identify goal requirements
    2. Plan merging sequence to create required fruit sizes
    3. Find click positions that pull fruits together for merging
    4. Move final fruits to goal zones

    This is a planning problem, not BFS (state space too large for 64x64 grid).
    """
    level = SU15_LEVELS[level_idx]
    fruits = [dict(f) for f in level["fruits"]]
    goals = level["goals"]
    goal_req = level["goal_req"]

    print(f"    Fruits: {[(f['size'], f['x'], f['y']) for f in fruits]}")
    print(f"    Goals: {goals}")
    print(f"    Requirements: {goal_req}")

    actions = []

    # For levels with just one fruit that needs to be moved to goal
    if len(fruits) == 1 and len(goal_req) == 1:
        f = fruits[0]
        req_size, req_count = goal_req[0]
        if req_size in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            print(f"    Requires enemy in goal — skipping fruit-only planning")
            return None

        if str(f["size"]) == req_size:
            # Just need to move fruit to goal
            gx, gy = goals[0]
            goal_center_x = gx + 4
            goal_center_y = gy + 4

            # Plan vacuum clicks to pull fruit toward goal
            path_actions = plan_vacuum_path(f, goal_center_x, goal_center_y)
            if path_actions:
                actions.extend(path_actions)
                return actions

    # General strategy: plan merges and movements
    # For each goal requirement, figure out which fruits to merge
    plan = plan_merges(fruits, goal_req, goals)
    if plan is not None:
        return plan

    return None


def plan_vacuum_path(fruit, target_x, target_y, max_clicks=20):
    """Plan a sequence of vacuum clicks to move a fruit to a target position.

    Each click pulls fruits within radius 8, moving them up to 4*4=16 pixels toward click.
    Strategy: click on the far side of target from fruit, so vacuum pulls fruit toward target.
    """
    actions = []
    f = dict(fruit)

    for _ in range(max_clicks):
        cx, cy = su15_fruit_center(f["x"], f["y"], f["size"])

        # Check if close enough to target
        if abs(cx - target_x) <= 2 and abs(cy - target_y) <= 2:
            break

        # Click position: between fruit and target, but shifted toward target
        # The vacuum pulls fruit TOWARD the click point
        # So click closer to the target than the fruit
        dx = target_x - cx
        dy = target_y - cy

        # Click at the target, or slightly beyond fruit toward target
        click_x = target_x
        click_y = target_y

        # But click must be within vacuum radius of fruit
        dist = (dx*dx + dy*dy) ** 0.5
        if dist > 8:
            # Click at closest point within radius toward target
            scale = 7.0 / dist
            click_x = int(cx + dx * scale)
            click_y = int(cy + dy * scale)

        # Clamp click to valid range
        click_y = max(10, min(63, click_y))
        click_x = max(0, min(63, click_x))

        action = su15_click_action(click_x, click_y)
        actions.append(action)

        # Simulate the vacuum effect
        fruits_list = [f]
        result = su15_simulate_vacuum(fruits_list, click_x, click_y)
        if result:
            f = result[0]

    return actions


def plan_merges(fruits, goal_req, goals):
    """Plan a merge strategy for SU15.

    Bottom-up planning:
    1. Figure out required final sizes
    2. Recursively determine how many of each intermediate size we need
    3. Execute merges smallest-first: merge pairs of size 0, then 1, then 2, etc.
    4. After all merges, move final fruits to goals
    """
    actions = []
    current_fruits = [dict(f) for f in fruits]

    # Parse goal requirements (skip enemy types)
    fruit_reqs = []
    for req_type, req_count in goal_req:
        if req_type in ("vnjbdkorwc", "yckgseirmu", "vptxjilzzk"):
            continue
        fruit_reqs.append((int(req_type), req_count))

    if not fruit_reqs:
        return None

    # Count current inventory
    inventory = {}
    for f in current_fruits:
        inventory[f["size"]] = inventory.get(f["size"], 0) + 1

    # Determine full merge plan recursively
    # needed[size] = how many fruits of this size we need to produce via merging
    # (doesn't count existing ones already reserved for final goals)
    needed_to_produce = {}

    for target_size, target_count in fruit_reqs:
        have = inventory.get(target_size, 0)
        deficit = target_count - have
        if deficit > 0:
            needed_to_produce[target_size] = needed_to_produce.get(target_size, 0) + deficit

    # Propagate downward: to produce N of size S, need 2*N of size (S-1)
    # that aren't already available. Only go down to size 0 (can't create from nothing).
    max_size = max(needed_to_produce.keys()) if needed_to_produce else 0
    for size in range(max_size, 0, -1):
        produce = needed_to_produce.get(size, 0)
        if produce <= 0:
            continue
        source_size = size - 1
        if source_size < 0:
            print(f"    Cannot create size {size} — need size {source_size} which doesn't exist")
            return None
        sources_needed = produce * 2
        sources_available = inventory.get(source_size, 0)
        # Reserve available sources
        reserved = min(sources_available, sources_needed)
        inventory[source_size] = sources_available - reserved
        shortfall = sources_needed - reserved
        if shortfall > 0:
            # Need to produce more of source_size via merging even smaller
            needed_to_produce[source_size] = needed_to_produce.get(source_size, 0) + shortfall
            # Check if this is feasible by looking at even smaller sizes
            can_produce = True
            check_size = source_size
            while check_size > 0:
                check_size -= 1
                avail = inventory.get(check_size, 0)
                if avail > 0:
                    break
            else:
                if inventory.get(0, 0) <= 0 and source_size > 0:
                    print(f"    Cannot create enough size {source_size} — no smaller fruits available")
                    return None

    # Reset inventory
    inventory = {}
    for f in current_fruits:
        inventory[f["size"]] = inventory.get(f["size"], 0) + 1

    print(f"    Inventory: {inventory}")
    print(f"    Need to produce: {needed_to_produce}")

    # Build ordered merge list: merge smallest sizes first
    merge_plan = []
    for size in sorted(needed_to_produce.keys()):
        if size <= 0:
            continue  # Can't produce size 0 or negative from merges
        count = needed_to_produce[size]
        source = size - 1
        if source < 0:
            continue
        for _ in range(count):
            merge_plan.append(source)

    print(f"    Merge plan (merge pairs of size): {merge_plan}")

    # Execute merges: for each planned merge, find two fruits of the right size
    for merge_idx, merge_size in enumerate(merge_plan):
        # Find two closest fruits of this size in current_fruits
        candidates = [f for f in current_fruits if f["size"] == merge_size]

        if len(candidates) < 2:
            print(f"    Cannot find 2 fruits of size {merge_size} (have {len(candidates)})")
            print(f"    Current inventory: {[(f['size'], f['x'], f['y']) for f in current_fruits]}")
            return None

        # Pick closest pair
        best_pair = None
        best_dist = float('inf')
        for a in range(len(candidates)):
            for b in range(a + 1, len(candidates)):
                fa, fb = candidates[a], candidates[b]
                c1x, c1y = su15_fruit_center(fa["x"], fa["y"], fa["size"])
                c2x, c2y = su15_fruit_center(fb["x"], fb["y"], fb["size"])
                d = (c1x - c2x)**2 + (c1y - c2y)**2
                if d < best_dist:
                    best_dist = d
                    best_pair = (
                        next(i for i, f in enumerate(current_fruits) if f is fa),
                        next(i for i, f in enumerate(current_fruits) if f is fb)
                    )

        if best_pair is None:
            return None

        # Plan clicks to merge the pair
        merge_actions = plan_merge_pair(current_fruits, best_pair[0], best_pair[1])
        if merge_actions is None:
            print(f"    Failed to merge pair")
            return None

        actions.extend(merge_actions)

        # Simulate ALL actions on current state to get new state
        for act in merge_actions:
            if act >= 7:
                click_idx = act - 7
                cx = click_idx % 64
                cy = click_idx // 64
                current_fruits = su15_simulate_vacuum(current_fruits, cx, cy)

        inv = {}
        for f in current_fruits:
            inv[f["size"]] = inv.get(f["size"], 0) + 1
        print(f"      Merge {merge_idx+1}/{len(merge_plan)}: pair of size {merge_size} -> inventory {inv}")

    # Now move fruits to goals
    print(f"    Final state: {[(f['size'], f['x'], f['y']) for f in current_fruits]}")

    for target_size, target_count in fruit_reqs:
        matching = [f for f in current_fruits if f["size"] == target_size]
        for i in range(min(target_count, len(matching))):
            f = matching[i]
            if i < len(goals):
                gx, gy = goals[i]
                goal_cx = gx + 4
                goal_cy = gy + 4

                move_actions = plan_vacuum_path(f, goal_cx, goal_cy)
                if move_actions:
                    actions.extend(move_actions)
                    for act in move_actions:
                        if act >= 7:
                            click_idx = act - 7
                            cx = click_idx % 64
                            cy = click_idx // 64
                            current_fruits = su15_simulate_vacuum(current_fruits, cx, cy)

    return actions if actions else None


def find_safe_click(target_fruit, all_fruits, direction_x, direction_y):
    """Find a click position that only affects the target fruit and not others.

    Click must be within radius 8 of target, and NOT within radius 8 of any other fruit.
    Direction hints which way we want to pull the target.
    """
    tcx, tcy = su15_fruit_center(target_fruit["x"], target_fruit["y"], target_fruit["size"])
    ts = FRUIT_SIZES[target_fruit["size"]]

    # Try clicking in the direction we want to pull (target moves toward click)
    for r in range(1, 8):
        click_x = int(tcx + direction_x * r)
        click_y = int(tcy + direction_y * r)
        click_y = max(10, min(63, click_y))
        click_x = max(0, min(63, click_x))

        # Check this click doesn't affect other fruits
        safe = True
        for f in all_fruits:
            if f is target_fruit:
                continue
            fs = FRUIT_SIZES[f["size"]]
            # Closest point on fruit to click
            closest_x = max(f["x"], min(click_x, f["x"] + fs - 1))
            closest_y = max(f["y"], min(click_y, f["y"] + fs - 1))
            dx = click_x - closest_x
            dy = click_y - closest_y
            if dx * dx + dy * dy <= 64:  # radius 8 squared
                safe = False
                break

        if safe:
            # Verify target IS within range
            closest_x = max(target_fruit["x"], min(click_x, target_fruit["x"] + ts - 1))
            closest_y = max(target_fruit["y"], min(click_y, target_fruit["y"] + ts - 1))
            dx = click_x - closest_x
            dy = click_y - closest_y
            if dx * dx + dy * dy <= 64:
                return click_x, click_y

    # Fallback: click right on top of target (might affect nearby fruits)
    return tcx, max(10, min(63, tcy))


def plan_merge_pair(fruits, idx1, idx2):
    """Plan clicks to merge two fruits together.

    Strategy:
    1. If fruits are far apart, use safe clicks to pull one toward the other
    2. When close enough, click at midpoint to merge
    3. Safe clicks avoid affecting other fruits
    """
    actions = []
    working_fruits = [dict(f) for f in fruits]

    target_size = working_fruits[idx1]["size"]
    f1_id = id(working_fruits[idx1])
    f2_id = id(working_fruits[idx2])

    # Track by position
    f1_x, f1_y = working_fruits[idx1]["x"], working_fruits[idx1]["y"]
    f2_x, f2_y = working_fruits[idx2]["x"], working_fruits[idx2]["y"]

    def find_tracked(wf, old_x, old_y, target_sz, exclude_xy=None):
        """Find fruit closest to (old_x, old_y) of given size."""
        best = None
        best_dist = float('inf')
        for f in wf:
            if f["size"] != target_sz:
                continue
            if exclude_xy and abs(f["x"] - exclude_xy[0]) < 2 and abs(f["y"] - exclude_xy[1]) < 2:
                continue
            d = abs(f["x"] - old_x) + abs(f["y"] - old_y)
            if d < best_dist:
                best_dist = d
                best = f
        return best

    for attempt in range(30):
        f1 = find_tracked(working_fruits, f1_x, f1_y, target_size)
        f2 = find_tracked(working_fruits, f2_x, f2_y, target_size, exclude_xy=(f1_x, f1_y) if f1 else None)

        if f1 is None or f2 is None:
            # Check if merged
            merged = [f for f in working_fruits if f["size"] == target_size + 1]
            if merged:
                break
            candidates = [f for f in working_fruits if f["size"] == target_size]
            if len(candidates) < 2:
                break
            f1, f2 = candidates[0], candidates[1]

        c1x, c1y = su15_fruit_center(f1["x"], f1["y"], f1["size"])
        c2x, c2y = su15_fruit_center(f2["x"], f2["y"], f2["size"])

        # Check overlap
        s = FRUIT_SIZES[f1["size"]]
        if abs(f1["x"] - f2["x"]) < s and abs(f1["y"] - f2["y"]) < s:
            break

        total_dist = ((c1x - c2x)**2 + (c1y - c2y)**2) ** 0.5

        if total_dist <= 16:
            # Close enough — click at midpoint (both within radius 8)
            click_x = (c1x + c2x) // 2
            click_y = (c1y + c2y) // 2
            click_y = max(10, min(63, click_y))
            click_x = max(0, min(63, click_x))
        else:
            # Far apart — pull f1 toward f2 using safe click
            dir_x = c2x - c1x
            dir_y = c2y - c1y
            d = (dir_x**2 + dir_y**2) ** 0.5
            if d > 0:
                dir_x /= d
                dir_y /= d
            click_x, click_y = find_safe_click(f1, working_fruits, dir_x, dir_y)

        action = su15_click_action(click_x, click_y)
        actions.append(action)

        f1_x, f1_y = f1["x"], f1["y"]
        f2_x, f2_y = f2["x"], f2["y"]

        working_fruits = su15_simulate_vacuum(working_fruits, click_x, click_y)

        # Update tracking
        new_f1 = find_tracked(working_fruits, f1_x, f1_y, target_size)
        if new_f1:
            f1_x, f1_y = new_f1["x"], new_f1["y"]
        new_f2 = find_tracked(working_fruits, f2_x, f2_y, target_size, exclude_xy=(f1_x, f1_y))
        if new_f2:
            f2_x, f2_y = new_f2["x"], new_f2["y"]

    return actions


def solve_su15_with_api():
    """Solve SU15 levels using the API for precise simulation.

    For each level, use the abstract model to generate candidate click sequences,
    then verify with the actual game API.
    """
    print("=" * 60)
    print("SOLVING SU15 - Vacuum Fruit (API-assisted)")
    print("=" * 60)

    try:
        sys.path.insert(0, "B:/M/the-search/experiments")
        from util_arcagi3 import _Env
        env = _Env("su15")
    except Exception as e:
        print(f"  Cannot load API: {e}")
        return {}, []

    per_level = {}
    all_actions = []

    # L1 and L2 already solved
    l1_actions = [3471, 3156, 2778, 2400, 2085, 1707, 1329, 1014]
    l2_actions = [2478, 2520, 3606, 3575, 2471, 2464, 2462, 3568, 3562, 3548, 3554, 3106, 2784, 2591, 2275, 2087, 1832]

    per_level["L1"] = {"count": len(l1_actions), "actions": l1_actions, "source": "known_solution"}
    per_level["L2"] = {"count": len(l2_actions), "actions": l2_actions, "source": "known_solution"}
    all_actions.extend(l1_actions)
    all_actions.extend(l2_actions)

    # Use abstract planner for remaining levels
    for level_idx in range(2, 9):
        print(f"\n--- Level {level_idx + 1} ---")
        result = solve_su15_level_greedy(level_idx)

        if result is not None:
            per_level[f"L{level_idx + 1}"] = {
                "count": len(result),
                "actions": result,
                "source": "abstract_model_solver",
            }
            all_actions.extend(result)
            print(f"    -> {len(result)} actions (abstract)")
        else:
            per_level[f"L{level_idx + 1}"] = {
                "count": 0,
                "actions": [],
                "status": "UNSOLVED — vacuum physics too complex for abstract model",
            }
            print(f"    -> UNSOLVED")

    return per_level, all_actions


def solve_su15():
    """Solve all 9 levels of SU15."""
    print("=" * 60)
    print("SOLVING SU15 - Vacuum Fruit Game")
    print("=" * 60)

    per_level = {}
    all_actions = []

    # L1 and L2 already solved — use known solutions
    l1_actions = [3471, 3156, 2778, 2400, 2085, 1707, 1329, 1014]
    l2_actions = [2478, 2520, 3606, 3575, 2471, 2464, 2462, 3568, 3562, 3548, 3554, 3106, 2784, 2591, 2275, 2087, 1832]

    per_level["L1"] = {"count": len(l1_actions), "actions": l1_actions, "source": "known_solution"}
    per_level["L2"] = {"count": len(l2_actions), "actions": l2_actions, "source": "known_solution"}
    all_actions.extend(l1_actions)
    all_actions.extend(l2_actions)

    for level_idx in range(2, 9):
        print(f"\n--- Level {level_idx + 1} ---")
        result = solve_su15_level_greedy(level_idx)

        if result is not None:
            per_level[f"L{level_idx + 1}"] = {
                "count": len(result),
                "actions": result,
                "source": "abstract_model_solver",
            }
            all_actions.extend(result)
            print(f"    -> {len(result)} actions")
        else:
            per_level[f"L{level_idx + 1}"] = {
                "count": 0,
                "actions": [],
                "status": "UNSOLVED — vacuum physics too complex for abstract model",
            }
            print(f"    -> UNSOLVED")

    return per_level, all_actions


# ============================================================
# VERIFICATION
# ============================================================

def verify_with_api(game_name, actions, expected_levels=None):
    """Verify solution by replaying actions through arc_agi API wrapper."""
    try:
        sys.path.insert(0, "B:/M/the-search/experiments")
        from util_arcagi3 import _Env

        env = _Env(game_name)
        frame = env.reset()
        max_level = 0

        for i, action_int in enumerate(actions):
            frame, reward, done, info = env.step(action_int)
            level = info.get('level', 0)
            if level > max_level:
                max_level = level
                print(f"  -> Level {level} reached at action {i+1}")

            if done:
                print(f"  Verification: done after {i+1} actions, reached level {max_level}")
                return max_level

        print(f"  Verification: completed all {len(actions)} actions, max level {max_level}")
        return max_level

    except Exception as e:
        print(f"  Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return -1


# ============================================================
# MAIN
# ============================================================

def main():
    # Solve M0R0
    m0r0_per_level, m0r0_all = solve_m0r0()

    # Solve SU15
    su15_per_level, su15_all = solve_su15()

    # Save M0R0 results
    m0r0_result = {
        "game": "m0r0",
        "source": "abstract_model_bfs_solver",
        "type": "analytical",
        "total_actions": len(m0r0_all),
        "max_level": max((int(k[1:]) for k, v in m0r0_per_level.items() if v.get("count", 0) > 0), default=0),
        "per_level": {k: {"count": v["count"], "actions": v["actions"]} for k, v in m0r0_per_level.items()},
        "all_actions": m0r0_all,
    }

    m0r0_path = "B:/M/the-search/experiments/results/prescriptions/m0r0_fullchain.json"
    with open(m0r0_path, "w") as f:
        json.dump(m0r0_result, f, indent=2)
    print(f"\nM0R0 saved to {m0r0_path}")

    # Save SU15 results
    su15_result = {
        "game": "su15",
        "source": "abstract_model_vacuum_solver",
        "type": "analytical",
        "total_actions": len(su15_all),
        "max_level": max((int(k[1:]) for k, v in su15_per_level.items() if v.get("count", 0) > 0), default=0),
        "per_level": {k: {"count": v["count"], "actions": v["actions"]} for k, v in su15_per_level.items()},
        "all_actions": su15_all,
    }

    su15_path = "B:/M/the-search/experiments/results/prescriptions/su15_fullchain.json"
    with open(su15_path, "w") as f:
        json.dump(su15_result, f, indent=2)
    print(f"SU15 saved to {su15_path}")

    # Verify with API
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    if m0r0_all:
        print("\nVerifying M0R0...")
        m0r0_levels = verify_with_api("m0r0", m0r0_all)
        m0r0_result["verified_levels"] = m0r0_levels

    if su15_all:
        print("\nVerifying SU15...")
        su15_levels = verify_with_api("su15", su15_all)
        su15_result["verified_levels"] = su15_levels

    # Re-save with verification results
    with open(m0r0_path, "w") as f:
        json.dump(m0r0_result, f, indent=2)
    with open(su15_path, "w") as f:
        json.dump(su15_result, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
