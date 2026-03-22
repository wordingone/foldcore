"""
Test the analytical solution for VC33 Level 3 (LEVELS[3] = "Level 4").

Solution plan:
Phase 1: Move Ubu from JYf to Oqo
  - 2x gel(JYf,Oqo) → JYf.y=55, Oqo.y=55, Ubu.y=49
  - Switch (12,43) → Ubu moves to Oqo (x=18)
  [+ some no-ops for animation]
Phase 2: Lower Oqo/Ubu to y=40
  - 3x gel(JYf,Oqo) → Oqo.y=46, Ubu.y=40
Phase 3: Get BfR.y=46
  - 2x gel(cGJ,BfR) → BfR.y=49
  - 1x gel(mZh,cGJ) → refill cGJ
  - 1x gel(cGJ,BfR) → BfR.y=46
Phase 4: Move Ubu from Oqo to BfR
  - Switch (27,34) → Ubu on BfR (x=33)
  [+ some no-ops for animation]
Phase 5: Lower BfR/Ubu to reach Ubu.y=25
  - 5x [gel(mZh,cGJ) + gel(cGJ,BfR)]
  → Ubu.y = 40 - 5*3 = 25 ✓
"""
import sys
sys.path.insert(0, 'B:/M/the-search/experiments/environment_files/vc33/9851e02b')
sys.path.insert(0, 'B:/M/the-search/experiments')

import logging
logging.getLogger().setLevel(logging.WARNING)

SOLUTIONS_PREV = {
    0: [(62,34),(62,34),(62,34)],
    1: [(0,24),(0,24),(0,44),(0,44),(0,44),(0,44),(0,44)],
    2: [(12,56),(24,56),(12,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),(34,56),(24,56),(12,56),
        (46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56),(46,56)],
}

# Click indices and display coords for Level 3 actions:
# C0 = (9,61)  = gel(Oqo→JYf) [Ubu up by 3]
# C1 = (15,61) = gel(JYf→Oqo) [Ubu down by 3 when on JYf; when Ubu on Oqo, Oqo moves up]
# C2 = (39,61) = gel(cGJ→BfR) [BfR up; cGJ down]
# C3 = (45,61) = gel(BfR→cGJ) [BfR down; cGJ up]
# C4 = (51,61) = gel(mZh→cGJ) [cGJ up; mZh down]
# C5 = (57,61) = gel(cGJ→mZh) [cGJ down; mZh up]
# C6 = (27,34) = zHk switch (Oqo→BfR)
# C7 = (12,43) = zHk switch (JYf→Oqo)

C0=(9,61); C1=(15,61); C2=(39,61); C3=(45,61); C4=(51,61); C5=(57,61); C6=(27,34); C7=(12,43)

# No-op position (off screen or unused area)
NOP = (32, 32)  # middle of grid - no sprite there hopefully

# Solution with 1 no-op per switch (animation completes in 1 step)
SOLUTION_CLICKS = (
    # Phase 1: 2x gel(JYf→Oqo) to align at y=55
    [C1, C1] +
    # Switch: JYf→Oqo; 1 no-op for animation
    [C7] + [NOP]*1 +
    # Phase 2: 3x gel(JYf→Oqo) to lower Oqo to y=46
    [C1, C1, C1] +
    # Phase 3: get BfR.y=46
    [C2, C2, C4, C2] +
    # Phase 4: switch Oqo→BfR; 1 no-op for animation
    [C6] + [NOP]*1 +
    # Phase 5: 5x [gel(mZh→cGJ) + gel(cGJ→BfR)]
    [C4, C2] * 5
)


def print_state(game, label=""):
    rdn = {sp.name: (sp.y, sp.height)
           for sp in game.current_level.get_sprites_by_tag("rDn")}
    hqb = {sp.name: (sp.x, sp.y)
           for sp in game.current_level.get_sprites_by_tag("HQB")}
    print(f"  [{label}] BfR={rdn.get('BfR')} cGJ={rdn.get('cGJ')} "
          f"JYf={rdn.get('JYf')} mZh={rdn.get('mZh')} Oqo={rdn.get('Oqo')} "
          f"Ubu={hqb.get('Ubu')}")


def test_solution():
    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("VC33 not found"); return False

    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    # Fast-forward to level 3
    obs = env.reset()
    for lvl in range(3):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={"x": cx, "y": cy})

    game = getattr(env, 'game', None)
    if game is None:
        game = getattr(env, '_game', None)
    print(f"Level 3 start: levels_completed={obs.levels_completed}")
    print_state(game, "initial")

    # Apply solution
    total_clicks = 0
    for i, (cx, cy) in enumerate(SOLUTION_CLICKS):
        obs = env.step(action6, data={"x": cx, "y": cy})
        total_clicks += 1

        if obs.state == GameState.WIN:
            print(f"\nWIN at click {i} (total {total_clicks} clicks)!")
            print(f"levels_completed = {obs.levels_completed}")
            return True

        if obs.state == GameState.GAME_OVER:
            print(f"\nGAME OVER at click {i}!")
            print_state(game, f"click {i}")
            return False

        # Print state at key checkpoints
        if i in [1, 2, 22, 25, 26, 30, 34, 35, 55]:
            print_state(game, f"click {i} ({cx},{cy})")

    print(f"\nNo WIN after {total_clicks} clicks.")
    print_state(game, "final")
    return False


def test_shorter_nop():
    """Test with fewer no-ops to find minimum wait needed."""
    import arc_agi
    from arcengine import GameState

    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    game = getattr(env, 'game', None) or getattr(env, '_game', None)

    for n_nop in [1, 5, 10, 15, 20]:
        obs = env.reset()
        for lvl in range(3):
            for cx, cy in SOLUTIONS_PREV[lvl]:
                obs = env.step(action6, data={"x": cx, "y": cy})

        # Phase 1: 2 gel(JYf,Oqo)
        for _ in range(2):
            obs = env.step(action6, data={"x": C1[0], "y": C1[1]})

        # Switch click
        obs = env.step(action6, data={"x": C7[0], "y": C7[1]})

        # n_nop no-ops
        for _ in range(n_nop):
            obs = env.step(action6, data={"x": NOP[0], "y": NOP[1]})

        # Check if Ubu moved
        ubu = game.current_level.get_sprites_by_tag("HQB")[0]
        print(f"  n_nop={n_nop}: Ubu.x={ubu.x} (expected ~18 for Oqo)")
        if ubu.x == 18:
            print(f"    Animation complete with {n_nop} no-ops!")
            break


if __name__ == "__main__":
    print("=== Test animation no-ops needed ===")
    test_shorter_nop()

    print("\n=== Test full solution ===")
    result = test_solution()
    print(f"\nResult: {'WIN' if result else 'FAIL'}")

    if not result:
        print("\nDebugging: trace through solution manually...")
