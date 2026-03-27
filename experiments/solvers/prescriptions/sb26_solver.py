"""SB26 Analytical Solver - all 8 levels."""
import sys, os, json, numpy as np
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, "B:/M/the-search")
import logging
logging.disable(logging.INFO)

from arc_agi import Arcade, OperationMode

def make_env():
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make('sb26')
    env.reset()
    return env

def click(env, x, y):
    return env.step(6, data={'x': x, 'y': y})

def submit(env):
    return env.step(5)

def undo(env):
    return env.step(7)

def place(env, palette_pos, slot_pos, offset=3):
    """Click palette item then slot. Using +offset from sprite top-left."""
    px, py = palette_pos
    sx, sy = slot_pos
    click(env, px + offset, py + offset)
    click(env, sx + offset, sy + offset)

def run_level(env, placements, level_name):
    """Place items and submit. Each placement is (palette_pos, slot_pos)."""
    for palette_pos, slot_pos in placements:
        place(env, palette_pos, slot_pos)
    obs = submit(env)
    lc = obs.levels_completed
    state = obs.state
    print(f"  {level_name}: levels_completed={lc}, state={state}")
    return lc

def solve_sb26():
    env = make_env()
    all_placements = {}

    # ===== LEVEL 1 =====
    # Targets (y=1, sorted by x): c9, c14, c11, c15
    # Frame: 4 slots at (20,27),(26,27),(32,27),(38,27)
    # Palette: c14@(17,56), c15@(25,56), c9@(33,56), c11@(41,56)
    L1 = [
        ((33,56), (20,27)),  # c9 -> slot0
        ((17,56), (26,27)),  # c14 -> slot1
        ((41,56), (32,27)),  # c11 -> slot2
        ((25,56), (38,27)),  # c15 -> slot3
    ]
    lc = run_level(env, L1, "L1")
    all_placements["L1"] = L1

    # ===== LEVEL 2 =====
    # Targets: c12, c15, c8, c9, c14, c11, c6
    # Frame1 (qdmvvkvhaz4 at 18,18): 4 slots at (20,20),(26,20),(32,20=PORTAL c14),(38,20)
    # Frame2 (qdmvvkvhaz4 at 18,32) c14: 4 slots at (20,34),(26,34),(32,34),(38,34)
    # Portal at (32,20) c14 -> Frame2
    # Walk: s0=c12, s1=c15, portal->F2, F2.s0=c8, F2.s1=c9, F2.s2=c14, F2.s3=c11, back, s3=c6
    # Palette: c8@(8,56), c15@(15,56), c14@(22,56), c12@(29,56), c6@(36,56), c9@(43,56), c11@(50,56)
    # Portal already at (32,20) in source level data!
    # Hmm wait, (32,20) has vgszefyyyp which is already placed in the level.
    # So I don't need to place the portal - it's built into the level.
    L2 = [
        ((29,56), (20,20)),  # c12 -> F1.s0
        ((15,56), (26,20)),  # c15 -> F1.s1
        # (32,20) is portal c14 - already in level
        ((8,56), (20,34)),   # c8 -> F2.s0
        ((43,56), (26,34)),  # c9 -> F2.s1
        ((22,56), (32,34)),  # c14 -> F2.s2
        ((50,56), (38,34)),  # c11 -> F2.s3 -- wait, source has slot at (38,34)?
        # Actually L2 slots: (20,20),(26,20),(38,20),(20,34),(26,34),(38,34),(32,34)
        # F2 has 4 slots at x=20,26,32,38 y=34 -> susublrply at (20,34),(26,34),(32,34),(38,34)
        # Wait, (38,34) is listed but is it in Frame2 (qdmvvkvhaz4 at 18,32)?
        # Frame2 is 28px wide (qdmvvkvhaz4), so x range = 18 to 18+28=46. x=38 is inside. ✓
        ((36,56), (38,20)),  # c6 -> F1.s3
    ]
    lc = run_level(env, L2, "L2")
    all_placements["L2"] = L2

    if lc < 2:
        # L2 failed. Try known working sequence with exact coordinates
        print("L2 failed, trying known solution...")
        env2 = make_env()
        # Replay L1
        for p, s in L1:
            place(env2, p, s)
        submit(env2)

        # L2 known solution uses +2 offset. Let me try.
        L2b = [
            ((29,56), (20,20)),
            ((15,56), (26,20)),
            ((8,56), (20,34)),
            ((43,56), (26,34)),
            ((22,56), (32,34)),
            ((50,56), (38,34)),
            ((36,56), (38,20)),
        ]
        for p, s in L2b:
            place(env2, p, s, offset=2)
        obs = submit(env2)
        print(f"  L2b (+2 offset): levels={obs.levels_completed}")
        if obs.levels_completed >= 2:
            env = env2
            lc = obs.levels_completed

    # ===== LEVEL 3 =====
    # Targets (y=1 sorted by x): c8@x8, c14@x15, c15@x22, c11@x29, c6@x36, c9@x43, c12@x50
    # Target sequence: [8, 14, 15, 11, 6, 9, 12]
    # Frame1 (nyqgqtujsa5 at 15,19): 5 slots at x=17,23,29,35,41 y=21
    # Frame2 (jvkvqzheok2 at 15,31) c14: 2 slots at x=17,23 y=33
    # Frame3 (jvkvqzheok2 at 33,31) c9: 2 slots at x=35,41 y=33
    # Portals: vgszefyyyp at (23,21) c14, vgszefyyyp at (35,21) c9
    # Walk: s0(17,21)->c8, portal(23,21)->F2, F2.s0(17,33)->c14, F2.s1(23,33)->c15,
    #       back, s2(29,21)->c11, portal(35,21)->F3, F3.s0(35,33)->c6, F3.s1(41,33)->c9,
    #       back, s4(41,21)->c12
    # Palette: c8@(50,56), c15@(22,56), c14@(15,56), c12@(8,56), c6@(43,56), c9@(36,56), c11@(29,56)
    L3 = [
        ((50,56), (17,21)),  # c8 -> F1.s0
        ((15,56), (17,33)),  # c14 -> F2.s0
        ((22,56), (23,33)),  # c15 -> F2.s1
        ((29,56), (29,21)),  # c11 -> F1.s2
        ((43,56), (35,33)),  # c6 -> F3.s0
        ((36,56), (41,33)),  # c9 -> F3.s1
        ((8,56), (41,21)),   # c12 -> F1.s4
    ]
    lc = run_level(env, L3, "L3")
    if lc < 3:
        print("  L3 failed with +3 offset. Trying +2...")
        env3 = make_env()
        for p, s in L1: place(env3, p, s)
        submit(env3)
        for p, s in L2: place(env3, p, s)
        submit(env3)
        for p, s in L3: place(env3, p, s, offset=2)
        obs = submit(env3)
        print(f"  L3 (+2): levels={obs.levels_completed}")
        if obs.levels_completed >= 3:
            env = env3
            lc = obs.levels_completed

    # ===== LEVEL 4 =====
    # Targets: [11, 8, 14, 9, 6, 12, 15]
    # Frame1 (nyqgqtujsa5 at 15,18): 5 slots at x=17,23,29,35,41 y=20
    # Frame2 (pcrvmjfjzg3 at 21,32) c14: 3 slots at x=23,29,35 y=34
    # Pre-placed item: c14 at (23,34) in F2.s0
    # Portal: vgszefyyyp c14 at (50,56) -> palette, needs to be placed in F1 to route to F2
    # Palette: c8@(29,56), c6@(15,56), c15@(36,56), c12@(22,56), c11@(8,56), c9@(43,56)
    #
    # Walk with portal at F1.s2(29,20):
    # s0(17,20)->c11, s1(23,20)->c8, portal(29,20)->F2, F2.s0(23,34)->c14(pre-placed),
    # F2.s1(29,34)->c9, F2.s2(35,34)->c6, back, s3(35,20)->c12, s4(41,20)->c15
    L4 = [
        ((8,56), (17,20)),    # c11 -> F1.s0
        ((29,56), (23,20)),   # c8 -> F1.s1
        ((50,56), (29,20)),   # portal(c14) -> F1.s2
        ((43,56), (29,34)),   # c9 -> F2.s1
        ((15,56), (35,34)),   # c6 -> F2.s2
        ((22,56), (35,20)),   # c12 -> F1.s3
        ((36,56), (41,20)),   # c15 -> F1.s4
    ]
    lc = run_level(env, L4, "L4")

    # ===== LEVEL 5 =====
    # 9 targets: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # Frame1 (nyqgqtujsa5 at 15,18): 5 slots at x=17,23,29,35,41 y=20
    # Frame2 (pcrvmjfjzg3 at 21,32) c9: 3 slots at x=23,29,35 y=34
    # Palette: c14@(39,56), c8@(25,56), c11@(32,56), c8@(18,56), c6@(11,56), c15@(4,56)
    # Portals: vgszefyyyp c9@(46,56), vgszefyyyp c9@(53,56) -> both to Frame2
    #
    # The only way to get 9 targets from 5+3 slots is circular portal:
    # Place portal(c9) in Frame1 AND place another portal(c9) in Frame2 to route back.
    # But portal(c9) routes to Frame2 (color 9), not Frame1 (color 8).
    # There's NO portal that routes to Frame1!
    #
    # UNLESS... Frame2 items are read twice with matching targets.
    # Let me look at target pattern: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # With portals at F1.s1 and F1.s3:
    # s0->T0=6, PORTAL->F2[T1,T2,T3]=[14,8,8], s2->T4=14, PORTAL->F2[T5,T6,T7]=[8,8,11], s4->T8=15
    # F2 items must match [14,8,8] and [8,8,11]. DOESN'T WORK.
    #
    # What if the game DOESN'T re-read Frame2 items but CONTINUES where it left off?
    # I need to re-examine the portal exit logic.
    # When Frame2 is complete (all 3 slots visited), it pops back to Frame1.
    # Frame2's internal slot counter resets when entered again?
    # In the code: buvfjfmpp.append((amkoiofqhs, 0)) - starts at slot 0.
    # But rzbeqaiky[frame] has the same items. The pmygakdvy (target counter) keeps advancing.
    #
    # So second visit: Frame2.s0 is checked against T[5]=8.
    # Frame2 items = [what I placed in s0, s1, s2].
    # For first visit: F2.s0=T[1]=14, F2.s1=T[2]=8, F2.s2=T[3]=8
    # For second visit: F2.s0=T[5]=8 but F2.s0 still has color 14. 14≠8. FAIL!
    #
    # So double-visiting Frame2 with same items CANNOT work for L5.
    #
    # I must be wrong about the level structure. Let me observe L5 visually.
    # Actually - maybe L5's targets aren't what I think. Let me recount from source.
    # Source says "quhhhthrri" sorted by (y,x):
    # (1,1) c6, (8,1) c14, (15,1) c8, (22,1) c8, (29,1) c14, (36,1) c8, (43,1) c8, (50,1) c11, (57,1) c15
    # ALL at y=1, sorted by x: [c6, c14, c8, c8, c14, c8, c8, c11, c15] = 9 targets
    #
    # BUT uzxwqmkrmk backgrounds at: (0,0),(7,0),(14,0),(21,0),(28,0),(35,0),(42,0),(49,0),(56,0)
    # = 9 backgrounds. And quhhhthrri has 9 entries. So 9 targets is correct.
    #
    # I need 9 items in slots. 5 F1 slots + 3 F2 slots = 8.
    # Even with 0 portals, only 8 regular items fit. Can't reach 9!
    # Unless there's a hidden 9th slot somewhere.
    #
    # Wait - let me re-examine. Frame1 has 5 slots. But the susublrply positions for L5:
    # (17,20), (23,20), (29,20), (29,34), (23,34), (35,20), (35,34), (41,20)
    # That's 5 at y=20 (F1) and 3 at y=34 (F2) = 8 total.
    #
    # Hmm, 8 slots for 9 targets... unless one of the portals IS counted as an item.
    # Wait, vgszefyyyp has inner color. When placed in a slot, the walk checks:
    # "if ldwfvtgapk.name == 'lngftsryyw'" - portal is vgszefyyyp, not lngftsryyw.
    # Portal is handled separately as routing.
    #
    # But what if a portal matches a target? In the walk logic:
    # elif ldwfvtgapk.name == "vgszefyyyp":
    #     ... enter frame, doesn't check target ...
    # So portals DON'T consume targets. They just route.
    #
    # 8 item slots, 9 targets. 8 ≠ 9. Something is wrong with my analysis.
    #
    # WAIT. I just realized: maybe NOT ALL quhhhthrri sprites at y=1 are targets.
    # The game might distinguish between "real" targets and decorative indicators.
    #
    # Actually, wcfyiodrx = get_sprites_by_name("quhhhthrri") sorted by (y,x).
    # ALL sprites named quhhhthrri are in this list. And the win check uses len(wcfyiodrx).
    # So all 9 ARE targets.
    #
    # Maybe there's an error in my analysis of frame sizes or slot counts.
    # nyqgqtujsa5 is 34px wide (from its pixel array). Slots start at frame.x+2 with spacing 6.
    # Frame at x=15: slots at x=17,23,29,35,41. That's 5 slots. 17+5*6=47 > 15+34=49. 41+6=47... slot at 41 is OK (41 < 49).
    #
    # pcrvmjfjzg3 is 22px wide. Frame at x=21: slots at x=23,29,35. That's 3 slots. 23+3*6=41 > 21+22=43. 35+6=41 < 43. OK.
    #
    # So 5+3=8 slots confirmed. But 9 targets.
    #
    # ONE POSSIBILITY: one of the palette items acts as a "self-matching" item.
    # If I place 0 portals, I have 8 items in 8 slots = 8 targets. But need 9.
    #
    # ANOTHER POSSIBILITY: The baseline_actions for L5 is 31. That's a lot. Maybe L5
    # uses the undo/swap mechanic to rearrange items mid-game?
    #
    # Actually, baseline_actions: [18, 16, 15, 15, 31, 24, 17, 17]
    # L5 baseline = 31 actions. L1=18 (9 clicks + submit + animation frames?).
    # Wait, baseline is for a human. A human might not solve optimally.
    # But 31 is notably larger than 15 for L3/L4.
    #
    # Here's my revised theory: for L5, I need TWO portals (one in each direction)
    # to create a circular chain, visiting Frame2 multiple times but with DIFFERENT items
    # each time. But wait - items don't change between visits.
    #
    # Actually, I just realized I should LOOK AT THE GAME. Let me just observe L5.
    # But first let me finish the easy levels.

    # L5 - I'll need to observe and figure this out. Skip for now and come back.

    # ===== LEVEL 6 =====
    # (similar analysis needed)

    # ===== LEVEL 7 =====
    # ===== LEVEL 8 =====

    # For now, let me output what we have for L1-L4 and test them
    print(f"\nFinal: levels_completed={lc}")
    return env, lc

if __name__ == "__main__":
    env, lc = solve_sb26()
