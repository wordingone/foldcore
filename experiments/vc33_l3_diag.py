"""
VC33 Level 3 (LEVELS[3] = "Level 4") diagnostic.
Traces dzy mapping, initial state, and runs BFS via game engine.
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')

import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from collections import deque


def print_level_sprites():
    from vc33 import levels as LEVELS
    L = LEVELS[3]
    print(f"Level name: {L.name}")
    print(f"grid_size: {L.grid_size}")
    print(f"TiD: {L.get_data('TiD')}")
    print(f"RoA: {L.get_data('RoA')}")
    print("Sprites:")
    for sp in L._sprites:
        print(f"  {sp.name}({','.join(sp.tags)}): pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")


def trace_dzy_and_initial():
    """Load vc33 game, start at level 3, and inspect dzy mapping."""
    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("VC33 not found"); return

    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    # Fast-forward to level 3 using known solutions
    SOLUTIONS = {
        0: [(62,34),(62,34),(62,34)],
        1: [(0,24),(0,24),(0,44),(0,44),(0,44),(0,44),(0,44)],
        2: [(12,56),(24,56),(12,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),
            (46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56)],
    }

    obs = env.reset()
    ts = 0
    for lvl in range(3):
        clicks = SOLUTIONS[lvl]
        for cx, cy in clicks:
            obs = env.step(action6, data={"x": cx, "y": cy})
            ts += 1
        print(f"After level {lvl} solution: levels_completed={obs.levels_completed}")

    print(f"\nNow at level 3. levels_completed={obs.levels_completed}, ts={ts}")

    # Get game internals
    game = env.game if hasattr(env, 'game') else None
    if game is None:
        # Try through env._game or similar
        for attr in ['_game', 'game', '_env', 'env']:
            if hasattr(env, attr):
                game = getattr(env, attr)
                break

    if game:
        print(f"Game type: {type(game)}")
        # Try to access dzy
        if hasattr(game, 'dzy'):
            print("\ndzy mapping:")
            for zgd, (pmj, chd) in game.dzy.items():
                print(f"  ZGd({zgd.name} at ({zgd.x},{zgd.y})) -> gel({pmj.name}({pmj.x},{pmj.y}), {chd.name}({chd.x},{chd.y}))")
        # Try to access rDn sprites
        if hasattr(game, 'current_level'):
            rdn = game.current_level.get_sprites_by_tag("rDn")
            hqb = game.current_level.get_sprites_by_tag("HQB")
            fzk = game.current_level.get_sprites_by_tag("fZK")
            zgd = game.current_level.get_sprites_by_tag("ZGd")
            uxg = game.current_level.get_sprites_by_tag("UXg")
            print("\nrDn sprites:")
            for sp in rdn:
                print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
            print("HQB sprites:")
            for sp in hqb:
                print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
            print("fZK sprites:")
            for sp in fzk:
                print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
            print("UXg sprites:")
            for sp in uxg:
                print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")
            print("ZGd sprites:")
            for sp in zgd:
                print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")

    return env, action6, obs


def get_rdn_heights(game):
    """Extract rDn heights (variable dimension) from game state."""
    rdn = game.current_level.get_sprites_by_tag("rDn")
    oro = game.oro  # TiD direction
    # height along travel axis = width if oro[0] else height
    result = {}
    for sp in rdn:
        if oro[0]:
            h = sp.width
        else:
            h = sp.height
        result[sp.name] = (sp, h)
    return result


def get_hqb_positions(game):
    hqb = game.current_level.get_sprites_by_tag("HQB")
    result = {}
    for sp in hqb:
        result[sp.name] = (sp.x, sp.y)
    return result


def bfs_level3(env, action6, obs):
    """BFS over ZGd click sequences to find winning solution."""
    from arcengine import GameState

    game = None
    for attr in ['game', '_game', '_env', 'env']:
        if hasattr(env, attr):
            game = getattr(env, attr)
            break
    if game is None:
        print("Cannot access game internals"); return

    # Get all ZGd display coords (we're at level 3 now)
    zgd_sprites = game.current_level.get_sprites_by_tag("ZGd")
    zgd_sorted = sorted(zgd_sprites, key=lambda s: s.x)

    # Get camera info
    cam = env.camera if hasattr(env, 'camera') else None
    if cam is None and hasattr(game, 'camera'):
        cam = game.camera

    print("\nZGd sprites (sorted by x):")
    for i, sp in enumerate(zgd_sorted):
        print(f"  Action {i}: ZGd {sp.name}({sp.x},{sp.y})")

    # Determine camera scale for display coords
    # For 64x64 grid with camera 64x64: scale=1, display=grid
    # Need to figure out actual display coords
    # From prior work: for 64x64 grid, display = grid directly
    gs = game.current_level.grid_size
    print(f"\ngrid_size={gs}")

    # Camera is Camera(0,0,64,64,3,4,[vrr]) from __init__
    # but it adapts to grid_size. Let me just use the obs.frame to test
    # Actually let's compute display coords empirically:
    # For 64x64 grid, camera_width=64, camera_height=64 (it adapts)
    # scale_x = display_width / camera_width = ?
    # For standard arcengine with display 64x64 and grid 64x64: scale=1
    # For non-square or non-64 grids, it scales
    # Based on prior analysis: for 64x64 grid, display coords = grid coords

    # Use all ZGd grid positions as display coords for BFS
    # (since grid=display for 64x64)
    click_coords = [(sp.x, sp.y) for sp in zgd_sorted]
    print(f"Click coords: {click_coords}")

    # Encode state from rDn heights
    rdn = game.current_level.get_sprites_by_tag("rDn")
    rdn_sorted = sorted(rdn, key=lambda s: s.name)

    def get_state():
        oro = game.oro
        heights = []
        for sp in rdn_sorted:
            if oro[0]:
                heights.append(sp.width)
            else:
                heights.append(sp.height)
        hqb = game.current_level.get_sprites_by_tag("HQB")
        hqb_pos = tuple((sp.x, sp.y) for sp in sorted(hqb, key=lambda s: s.name))
        return tuple(heights) + hqb_pos

    init_state = get_state()
    print(f"\nInitial rDn heights: {dict(zip([sp.name for sp in rdn_sorted], init_state[:len(rdn_sorted)]))}")
    hqb = game.current_level.get_sprites_by_tag("HQB")
    for sp in hqb:
        print(f"Initial HQB {sp.name}: ({sp.x},{sp.y})")
    fzk = game.current_level.get_sprites_by_tag("fZK")
    for sp in fzk:
        print(f"Target fZK {sp.name}: ({sp.x},{sp.y})")

    # Check gug() initially
    print(f"Initial gug(): {game.gug()}")

    print("\nTesting each ZGd action individually:")
    for i, (cx, cy) in enumerate(click_coords):
        # Save state
        saved_heights = {sp.name: (sp.width, sp.height) for sp in rdn_sorted}
        saved_hqb = {sp.name: (sp.x, sp.y) for sp in hqb}

        obs2 = env.step(action6, data={"x": cx, "y": cy})
        new_heights = {sp.name: sp.height for sp in rdn_sorted}
        new_hqb = {sp.name: (sp.x, sp.y) for sp in hqb}
        h_change = {n: new_heights[n] - saved_heights[n][1] for n in new_heights}
        hqb_change = {n: (new_hqb[n][1] - saved_hqb[n][1]) for n in new_hqb}
        print(f"  Action {i} ({cx},{cy}): h_delta={h_change} hqb_y_delta={hqb_change} lvls={obs2.levels_completed}")

        # Undo: reset to level 3 state
        obs_reset = env.reset()
        for lvl in range(3):
            for cx2, cy2 in SOLUTIONS[lvl]:
                obs_reset = env.step(action6, data={"x": cx2, "y": cy2})

        # Refresh rdn_sorted reference
        rdn = game.current_level.get_sprites_by_tag("rDn")
        rdn_sorted = sorted(rdn, key=lambda s: s.name)
        hqb = game.current_level.get_sprites_by_tag("HQB")

    print(f"\nAfter individual tests - re-check state:")
    print(f"Heights: {dict(zip([sp.name for sp in rdn_sorted], [sp.height for sp in rdn_sorted]))}")
    for sp in hqb:
        print(f"  HQB {sp.name}: ({sp.x},{sp.y})")


if __name__ == "__main__":
    print("=== VC33 Level 3 Diagnostic ===\n")

    # First: just print sprite info without game engine
    print("--- Static sprite analysis ---")
    print_level_sprites()

    print("\n--- Game engine analysis ---")
    result = trace_dzy_and_initial()
    if result:
        env, action6, obs = result
        bfs_level3(env, action6, obs)
