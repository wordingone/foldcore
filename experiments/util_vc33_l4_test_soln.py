"""
Test Level 4 (LEVELS[4]) solution.
Strategy:
- A1 = (61,17) = gel(bUo→UWO): cni.x increases by 3 per click
- A5 = (61,52) = gel(SnP→Xfy): Ubu.x decreases by 3 per click
- A3 = (61,35) = gel(UWO→SnP): refills SnP

Solution:
1. A1×2: cni.x = 10→13→16 ✓
2. A5×6: Ubu.x = 37→19, SnP depleted
3. A3, A5: refill+pump → Ubu.x=16
4. A3, A5: → Ubu.x=13
5. A3, A5: → Ubu.x=10 ✓

Total: 2+6+2+2+2 = 14 clicks
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
}

A1 = (61,17); A3 = (61,35); A5 = (61,52)

SOLUTION_L4 = (
    [A1, A1] +                    # cni.x: 10→16
    [A5]*6 +                      # Ubu.x: 37→19, SnP depleted
    [A3, A5] +                    # refill+pump: Ubu.x=16
    [A3, A5] +                    # Ubu.x=13
    [A3, A5]                      # Ubu.x=10 ✓
)


def print_state(game, label=""):
    from arcengine import GameState
    rdn = {sp.name: (sp.x, sp.width)
           for sp in game.current_level.get_sprites_by_tag("rDn")}
    hqb = {sp.name: (sp.x, sp.y)
           for sp in game.current_level.get_sprites_by_tag("HQB")}
    fzk = {sp.name: (sp.x, sp.y)
           for sp in game.current_level.get_sprites_by_tag("fZK")}
    print(f"  [{label}] rDn.x={rdn}  HQB={hqb}")


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
    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]
    game = getattr(env, 'game', None) or getattr(env, '_game', None)

    obs = advance_to_level(env, action6, 4)
    print(f"Level 4 start: levels_completed={obs.levels_completed}")
    print_state(game, "initial")

    for i, (cx, cy) in enumerate(SOLUTION_L4):
        obs = env.step(action6, data={"x": cx, "y": cy})

        if obs.state == GameState.WIN:
            print(f"  FULL WIN at click {i}!")
            return True
        if obs.state == GameState.GAME_OVER:
            print(f"  GAME OVER at click {i}!")
            print_state(game, f"click {i}")
            return False
        if obs.levels_completed > 4:
            print(f"  Level advanced at click {i}! levels={obs.levels_completed}")
            print_state(game, f"after L4 advance")
            return True  # level 4 complete!

        if i in [1, 7, 9, 11, 13]:
            print_state(game, f"click {i} ({cx},{cy})")

    print(f"No level advance after {len(SOLUTION_L4)} clicks")
    print_state(game, "final")
    return False


if __name__ == "__main__":
    print("=== Test VC33 Level 4 Solution ===")
    result = test_solution()
    print(f"\nResult: {'LEVEL ADVANCE' if result else 'FAIL'}")
