"""
VC33 Level 3 BFS solver.
Uses game engine directly to find winning click sequence.
Explores both ZGd lever clicks and zHk switch clicks.
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')

import logging
logging.getLogger().setLevel(logging.WARNING)

import numpy as np
from collections import deque
import copy

# Known solutions for levels 0-2
SOLUTIONS_PREV = {
    0: [(62,34),(62,34),(62,34)],
    1: [(0,24),(0,24),(0,44),(0,44),(0,44),(0,44),(0,44)],
    2: [(12,56),(24,56),(12,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),
        (46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56)],
}


def get_clickables(game):
    """Get all clickable display positions for current level."""
    zgd = game.current_level.get_sprites_by_tag("ZGd")
    zhk = game.current_level.get_sprites_by_tag("zHk")
    clickables = []
    for sp in sorted(zgd, key=lambda s: s.x):
        clickables.append((sp.x, sp.y, "ZGd", sp.name))
    for sp in sorted(zhk, key=lambda s: (sp.x, sp.y)):
        clickables.append((sp.x, sp.y, "zHk", sp.name))
    return clickables


def get_state(game):
    """Encode current game state as tuple."""
    rdn = game.current_level.get_sprites_by_tag("rDn")
    hqb = game.current_level.get_sprites_by_tag("HQB")
    oro = game.oro

    state = []
    for sp in sorted(rdn, key=lambda s: s.name):
        if oro[0]:
            state.append(sp.width)
        else:
            state.append(sp.height)
        state.append(sp.y)  # track position too
    for sp in sorted(hqb, key=lambda s: s.name):
        state.append(sp.x)
        state.append(sp.y)
    return tuple(state)


def reset_to_level3(env, action6):
    """Reset and fast-forward to level 3."""
    obs = env.reset()
    for lvl in range(3):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={"x": cx, "y": cy})
    return obs


def apply_click(env, action6, cx, cy, game):
    """Apply a click and return (obs, levels_before, levels_after)."""
    lvls_before = game.current_level  # not reliable across resets
    obs = env.step(action6, data={"x": cx, "y": cy})
    return obs


def bfs_solve(env, action6):
    """BFS to find winning sequence. Returns list of (cx,cy) clicks."""
    from arcengine import GameState

    game = None
    for attr in ['game', '_game']:
        if hasattr(env, attr):
            game = getattr(env, attr)
            break
    if game is None:
        print("ERROR: Cannot access game internals")
        return None

    # Get initial state
    obs = reset_to_level3(env, action6)

    if obs.state == GameState.WIN:
        print("Already at WIN state after L3 reset??")
        return []

    init_state = get_state(game)
    print(f"Initial state: {init_state}")

    # Get all clickable coords
    clickables_raw = get_clickables(game)
    print(f"Clickable positions:")
    for cx, cy, tag, name in clickables_raw:
        print(f"  ({cx},{cy}) type={tag} name={name}")

    # We need display coords. For 64x64 grid, display = grid
    # But click must land ON the sprite. Use grid positions directly.
    click_coords = [(cx, cy) for cx, cy, tag, name in clickables_raw]
    print(f"\nClick coords: {click_coords}")
    print(f"Total actions: {len(click_coords)}")

    # BFS over click sequences (budget=25)
    # State = game state tuple
    # Queue: (state_tuple, click_sequence)
    # We use the game engine to step forward

    # Since we can't easily copy game state without resetting,
    # use iterative deepening DFS or just try DFS with pruning

    # Alternative: enumerate states greedily based on physics model
    # Actually: let's do BFS up to depth 30 using state hashing

    MAX_DEPTH = 30

    # BFS with replay: to reach a state, replay from scratch
    # This is slow but correct. For 30 depth with 8 actions, too slow.

    # Instead: model the state analytically
    # State = (BfR.y, BfR.h, cGJ.y, cGJ.h, JYf.y, JYf.h, mZh.y, mZh.h, Oqo.y, Oqo.h, Ubu.x, Ubu.y)

    rdn_names = sorted([sp.name for sp in game.current_level.get_sprites_by_tag("rDn")])
    hqb_names = sorted([sp.name for sp in game.current_level.get_sprites_by_tag("HQB")])

    print(f"\nrDn names: {rdn_names}")
    print(f"HQB names: {hqb_names}")

    # Print initial rDn state
    for sp in sorted(game.current_level.get_sprites_by_tag("rDn"), key=lambda s: s.name):
        print(f"  {sp.name}: y={sp.y}, h={sp.height}, w={sp.width}")
    for sp in sorted(game.current_level.get_sprites_by_tag("HQB"), key=lambda s: s.name):
        print(f"  HQB {sp.name}: x={sp.x}, y={sp.y}")

    # Test what each click does on initial state
    print("\nTesting clicks on initial state:")
    for i, (cx, cy) in enumerate(click_coords):
        obs_test = reset_to_level3(env, action6)

        # Apply click
        obs2 = env.step(action6, data={"x": cx, "y": cy})

        rdn_after = {}
        hqb_after = {}
        for sp in game.current_level.get_sprites_by_tag("rDn"):
            rdn_after[sp.name] = (sp.y, sp.height)
        for sp in game.current_level.get_sprites_by_tag("HQB"):
            hqb_after[sp.name] = (sp.x, sp.y)

        print(f"  Click {i} ({cx},{cy}): win={obs2.state == GameState.WIN} lvls={obs2.levels_completed}")

        # Get initial for comparison
        obs_init = reset_to_level3(env, action6)
        rdn_init = {}
        hqb_init = {}
        for sp in game.current_level.get_sprites_by_tag("rDn"):
            rdn_init[sp.name] = (sp.y, sp.height)
        for sp in game.current_level.get_sprites_by_tag("HQB"):
            hqb_init[sp.name] = (sp.x, sp.y)

        rdn_changed = {n: (rdn_init[n], rdn_after[n]) for n in rdn_after if rdn_after[n] != rdn_init.get(n)}
        hqb_changed = {n: (hqb_init[n], hqb_after[n]) for n in hqb_after if hqb_after[n] != hqb_init.get(n)}
        if rdn_changed:
            print(f"    rDn changes: {rdn_changed}")
        if hqb_changed:
            print(f"    HQB changes: {hqb_changed}")

    return None


def main():
    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("VC33 not found"); return

    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    print("=== VC33 Level 3 BFS Solver ===\n")
    bfs_solve(env, action6)


if __name__ == "__main__":
    main()
