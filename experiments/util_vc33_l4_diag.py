"""
VC33 Level 4 (LEVELS[4] = "Level 5") diagnostic.
TiD=[3,0] means horizontal movement (X direction).
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


def advance_to_level(env, action6, target_level):
    obs = env.reset()
    for lvl in range(target_level):
        for cx, cy in SOLUTIONS_PREV[lvl]:
            obs = env.step(action6, data={"x": cx, "y": cy})
    print(f"At level {target_level}: levels_completed={obs.levels_completed}")
    return obs


def inspect_level(game, label=""):
    print(f"\n--- {label} ---")
    rdn = game.current_level.get_sprites_by_tag("rDn")
    hqb = game.current_level.get_sprites_by_tag("HQB")
    fzk = game.current_level.get_sprites_by_tag("fZK")
    uxg = game.current_level.get_sprites_by_tag("UXg")
    zgd = game.current_level.get_sprites_by_tag("ZGd")
    zhk = game.current_level.get_sprites_by_tag("zHk")
    oro = game.oro

    print(f"TiD/oro: {oro}")
    print(f"lia(): {game.lia()}")
    print(f"grid_size: {game.current_level.grid_size}")

    print(f"rDn ({len(rdn)}):")
    for sp in sorted(rdn, key=lambda s: s.name):
        h = sp.width if oro[0] else sp.height
        ebl = sp.x if oro[0] else sp.y
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}, "
              f"ebl={ebl}, jqo_h={h}")

    print(f"HQB:")
    for sp in hqb:
        ebl = sp.x if oro[0] else sp.y
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}, ebl={ebl}")

    print(f"fZK:")
    for sp in fzk:
        ebl = sp.x if oro[0] else sp.y
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}, ebl={ebl}")

    print(f"UXg:")
    for sp in sorted(uxg, key=lambda s: s.name):
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")

    print(f"ZGd ({len(zgd)}):")
    for sp in sorted(zgd, key=lambda s: (s.x, s.y)):
        print(f"  {sp.name}: pos=({sp.x},{sp.y})")

    print(f"zHk ({len(zhk)}):")
    for sp in sorted(zhk, key=lambda s: (sp.x, sp.y)):
        print(f"  {sp.name}: pos=({sp.x},{sp.y}), w={sp.width}, h={sp.height}")

    print(f"\ndzy mapping:")
    for zgd_sp, (pmj, chd) in game.dzy.items():
        print(f"  ZGd({zgd_sp.name} at ({zgd_sp.x},{zgd_sp.y})) "
              f"-> gel({pmj.name}({pmj.x},{pmj.y}), {chd.name}({chd.x},{chd.y}))")

    print(f"\nSuo and mud for each rDn:")
    for sp in sorted(rdn, key=lambda s: s.name):
        suo = game.suo(sp)
        mud = game.mud(sp)
        ebl = sp.x if oro[0] else sp.y
        print(f"  {sp.name}(ebl={ebl}): suo={[s.name for s in suo]}, mud={mud}")

    print(f"\ngug(): {game.gug()}")


def test_each_click(game, env, action6, obs_start, clicks):
    print(f"\nTesting each click:")
    rdn_before = {sp.name: (sp.x, sp.y, sp.width, sp.height)
                  for sp in game.current_level.get_sprites_by_tag("rDn")}
    hqb_before = {sp.name: (sp.x, sp.y)
                  for sp in game.current_level.get_sprites_by_tag("HQB")}

    for i, (cx, cy) in enumerate(clicks):
        obs = advance_to_level(env, action6, 4)

        rdn_b = {sp.name: (sp.x, sp.y, sp.width, sp.height)
                 for sp in game.current_level.get_sprites_by_tag("rDn")}
        hqb_b = {sp.name: (sp.x, sp.y)
                 for sp in game.current_level.get_sprites_by_tag("HQB")}

        obs2 = env.step(action6, data={"x": cx, "y": cy})

        rdn_a = {sp.name: (sp.x, sp.y, sp.width, sp.height)
                 for sp in game.current_level.get_sprites_by_tag("rDn") if sp}
        hqb_a = {sp.name: (sp.x, sp.y)
                 for sp in game.current_level.get_sprites_by_tag("HQB")}

        rdn_delta = {n: (rdn_b[n], rdn_a.get(n)) for n in rdn_b if rdn_a.get(n) != rdn_b[n]}
        hqb_delta = {n: (hqb_b[n], hqb_a.get(n)) for n in hqb_b if hqb_a.get(n) != hqb_b[n]}

        from arcengine import GameState
        win = obs2.state == GameState.WIN
        print(f"  Click {i} ({cx},{cy}): lvls={obs2.levels_completed} win={win}")
        if rdn_delta:
            print(f"    rDn: {rdn_delta}")
        if hqb_delta:
            print(f"    HQB: {hqb_delta}")


def main():
    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    env = arc.make(vc33.game_id)
    action6 = env.action_space[0]

    obs = advance_to_level(env, action6, 4)
    game = getattr(env, 'game', None) or getattr(env, '_game', None)

    inspect_level(game, "Level 4 (LEVELS[4])")

    # Get all clickable sprites
    zgd = game.current_level.get_sprites_by_tag("ZGd")
    zhk = game.current_level.get_sprites_by_tag("zHk")
    clicks = [(sp.x, sp.y) for sp in sorted(zgd, key=lambda s: (s.x, s.y))]
    clicks += [(sp.x, sp.y) for sp in sorted(zhk, key=lambda s: (s.x, s.y))]
    print(f"\nAll click coords: {clicks}")

    test_each_click(game, env, action6, obs, clicks)


if __name__ == "__main__":
    main()
