"""
VC33 Level 6 (LEVELS[5]) solution test.
TiD=[-3,0], lia()=False, horizontal negative direction.

BFS solution (20 actions + 2 NOPs):
GA, GD, GD, GD, S1, NOP, GB, GB, GC, GC, GC, GC, GC, GC, S2, NOP, GD, GD, GD, GD, GD, GD

Actions:
  GA=(0,27)  -- gel(JHJ->egw or similar)
  GB=(0,33)  -- gel(uct->JHJ or similar)
  GC=(24,33) -- gel(uct->egw or similar)
  GD=(24,27) -- gel(egw->JHJ or similar)
  S1=(6,30)  -- switch 1
  S2=(30,30) -- switch 2
  NOP=(60,60) -- safe no-op (far from all sprites, grid is 64x64)

Win condition: Ubu on egw, utq(egw)=48
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
    3: [(15,61),(15,61),(12,43),(32,32),(15,61),(15,61),(15,61),
        (39,61),(39,61),(51,61),(39,61),(27,34),(32,32),
        (51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61),(51,61),(39,61)],
    4: [(61,17),(61,17),(61,17),(61,17),(61,35),(61,35),(61,35),(61,35),(61,35),(61,52),(61,52),(25,49),(32,32),
        (61,29),(61,29),(61,29),(61,52),(61,52),(40,32),(32,32),
        (61,17),(61,17),(61,17),(61,17),(28,14),(32,32),
        (61,11),(61,11),(61,11),(61,11),(40,32),(32,32),
        (61,11),(61,35),(61,35),(61,35),(61,46),(61,46),(25,49),(32,32),
        (61,29),(61,11),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52),(61,52)],
}

GA=(0,27); GB=(0,33); GC=(24,33); GD=(24,27)
S1=(6,30); S2=(30,30)
NOP=(60,60)  # SAFE: far from all sprites in 64x64 grid

# BFS solution: 20 actions + 2 NOPs = 22 clicks total
SOLUTION_L5 = (
    [GA, GD, GD, GD] +         # initial gel moves
    [S1, NOP] +                  # switch 1 + safe NOP
    [GB, GB] +                   # gel moves
    [GC, GC, GC, GC, GC, GC] +  # gel moves
    [S2, NOP] +                  # switch 2 + safe NOP
    [GD, GD, GD, GD, GD, GD]    # final gel moves
)


def print_state(game, label=""):
    rdn = {sp.name: (sp.x, sp.width)
           for sp in game.current_level.get_sprites_by_tag("rDn")}
    hqb = {sp.name: (sp.x, sp.y)
           for sp in game.current_level.get_sprites_by_tag("HQB")}
    print(f"  [{label}] rDn={rdn} HQB={hqb}")


def advance_to_level(env, action6, target_level):
    obs = env.reset()
    for lvl in range(target_level):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={"x": cx, "y": cy})
    return obs


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
    game = getattr(env, 'game', None) or getattr(env, '_game', None)

    obs = advance_to_level(env, action6, 5)
    print(f"Level 5 start: levels_completed={obs.levels_completed}")
    print_state(game, "initial")

    # Check krt conditions at start
    zhk = game.current_level.get_sprites_by_tag("zHk")
    for sp in zhk:
        print(f"  Switch {sp.name} at ({sp.x},{sp.y}) w={sp.width} h={sp.height}")
        print(f"    krt={game.krt(sp)}")

    for i, (cx, cy) in enumerate(SOLUTION_L5):
        obs = env.step(action6, data={"x": cx, "y": cy})

        if obs.state == GameState.WIN:
            print(f"  FULL WIN at click {i}!")
            return True
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at click {i}!")
            print_state(game, f"click {i}")
            return False
        if obs.levels_completed > 5:
            print(f"  Level advanced at click {i}! levels={obs.levels_completed}")
            return True

        # Print at each step
        print_state(game, f"[{i}]({cx},{cy})")

    print(f"No level advance after {len(SOLUTION_L5)} clicks")
    print_state(game, "final")
    return False


if __name__ == "__main__":
    print("=== Test VC33 Level 5 (LEVELS[5]) Solution ===")
    result = test_solution()
    print(f"\nResult: {'LEVEL ADVANCE' if result else 'FAIL'}")
