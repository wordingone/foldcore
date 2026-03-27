"""SB26 Full Analytical Solver - all 8 levels."""
import sys, os, json, numpy as np
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, "B:/M/the-search")
import logging; logging.disable(logging.INFO)
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

def place(env, palette_pos, slot_pos, offset=3):
    """Click palette item center, then slot center."""
    px, py = palette_pos
    sx, sy = slot_pos
    click(env, px + offset, py + offset)
    click(env, sx + offset, sy + offset)

def swap(env, pos1, pos2, offset=3):
    """Swap two items by clicking both."""
    x1, y1 = pos1
    x2, y2 = pos2
    click(env, x1 + offset, y1 + offset)
    click(env, x2 + offset, y2 + offset)

def run_level(env, placements, level_name):
    for palette_pos, slot_pos in placements:
        place(env, palette_pos, slot_pos)
    obs = submit(env)
    lc = obs.levels_completed
    print(f"  {level_name}: levels_completed={lc}")
    return lc

def solve():
    env = make_env()
    total_actions = []
    level_data = {}

    def encode_click(x, y):
        return 7 + (y+3)*64 + (x+3)

    def record_place(palette, slot):
        total_actions.append(encode_click(*palette))
        total_actions.append(encode_click(*slot))

    def record_submit():
        total_actions.append(4)

    def record_swap(pos1, pos2):
        total_actions.append(encode_click(*pos1))
        total_actions.append(encode_click(*pos2))

    # ===== L1 =====
    L1 = [((33,56),(20,27)), ((17,56),(26,27)), ((41,56),(32,27)), ((25,56),(38,27))]
    lc = run_level(env, L1, "L1")
    for p,s in L1: record_place(p,s)
    record_submit()
    l1_acts = len(total_actions)

    # ===== L2 =====
    L2 = [((29,56),(20,20)), ((15,56),(26,20)), ((8,56),(20,34)),
          ((43,56),(26,34)), ((22,56),(32,34)), ((50,56),(38,34)), ((36,56),(38,20))]
    lc = run_level(env, L2, "L2")
    for p,s in L2: record_place(p,s)
    record_submit()
    l2_acts = len(total_actions) - l1_acts

    # ===== L3 =====
    L3 = [((50,56),(17,21)), ((15,56),(17,33)), ((22,56),(23,33)),
          ((29,56),(29,21)), ((43,56),(35,33)), ((36,56),(41,33)), ((8,56),(41,21))]
    lc = run_level(env, L3, "L3")
    for p,s in L3: record_place(p,s)
    record_submit()
    l3_acts = len(total_actions) - l1_acts - l2_acts

    # ===== L4 =====
    L4 = [((8,56),(17,20)), ((29,56),(23,20)), ((50,56),(29,20)),
          ((43,56),(29,34)), ((15,56),(35,34)), ((22,56),(35,20)), ((36,56),(41,20))]
    lc = run_level(env, L4, "L4")
    for p,s in L4: record_place(p,s)
    record_submit()
    l4_acts = len(total_actions) - l1_acts - l2_acts - l3_acts

    if lc < 4:
        print("ERROR: L1-L4 failed!")
        return

    # ===== L5 =====
    # 9 targets: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # Frame1 5 slots: (17,20),(23,20),(29,20),(35,20),(41,20)
    # Frame2 (color 9) 3 slots: (23,34),(29,34),(35,34)
    # Palette: c15@(4,56), c6@(11,56), c8@(18,56), c8@(25,56), c11@(32,56), c14@(39,56)
    # Portals: c9@(46,56), c9@(53,56)
    #
    # Strategy: 2 submits with portals at F1.s1(23,20) and F1.s3(35,20)
    # First submit: fill T0-T3. F2=[c14,c8,c8].
    # Items: s0=c6, s1=portal, s2=c11(wrong for T4), s3=portal, s4=c15
    # F2: s0=c14, s1=c8, s2=c8
    # Walk: s0->T0=6✓, portal->F2[T1=14✓,T2=8✓,T3=8✓], s2->T4=14 vs c11✗ ERROR
    # Result: T0-T3 filled, T4-T8 unfilled.
    #
    # Swap: F2.s0(23,34)[c14] <-> s2(29,20)[c11] => F2.s0=c11, s2=c14
    # Swap: F2.s0(23,34)[c11] <-> F2.s2(35,34)[c8] => F2=[c8,c8,c11]
    #
    # Second submit walk:
    # s0: T0 filled -> skip
    # portal->F2: T1,T2,T3 filled -> skip all
    # s2: T4 unfilled, c14 -> T4=14✓
    # portal->F2: T5=8 vs c8✓, T6=8 vs c8✓, T7=11 vs c11✓
    # s4: T8=15 vs c15✓ -> WIN!

    print("  L5: two-submit strategy...")
    l5_start = len(total_actions)

    # First placement
    L5_place = [
        ((11,56), (17,20)),   # c6 -> F1.s0
        ((46,56), (23,20)),   # portal -> F1.s1
        ((32,56), (29,20)),   # c11 -> F1.s2 (intentionally wrong)
        ((53,56), (35,20)),   # portal -> F1.s3
        ((4,56), (41,20)),    # c15 -> F1.s4
        ((39,56), (23,34)),   # c14 -> F2.s0
        ((25,56), (29,34)),   # c8 -> F2.s1
        ((18,56), (35,34)),   # c8 -> F2.s2
    ]
    for p, s in L5_place:
        place(env, p, s)
        record_place(p, s)

    # First submit (will fail on T4 but fill T0-T3)
    obs = submit(env)
    record_submit()
    print(f"    First submit: levels={obs.levels_completed}")

    # Wait for error animation to complete
    # The submit triggers animation; we need the step() calls to process it
    # Actually, the arc_agi step function should handle all animation frames internally
    # Let me check if the game state is ready for more actions

    # Swaps
    # Swap s2(29,20)[c11] with F2.s0(23,34)[c14]
    swap(env, (29,20), (23,34))
    record_swap((29,20), (23,34))

    # Swap F2.s0(23,34)[c11] with F2.s2(35,34)[c8]
    swap(env, (23,34), (35,34))
    record_swap((23,34), (35,34))

    # Second submit
    obs = submit(env)
    record_submit()
    lc = obs.levels_completed
    print(f"    Second submit: levels={lc}")
    l5_acts = len(total_actions) - l5_start

    if lc < 5:
        print("  L5 FAILED!")
        # Debug: observe frame
        frame = np.array(obs.frame[0])
        print(f"  Frame state: {obs.state}")
        return

    # ===== L6 =====
    # Targets (y=1 sorted by x): c9@1, c11@8, c11@15, c12@22, c15@29, c15@36, c14@43, c6@50, c6@57
    # Target: [9, 11, 11, 12, 15, 15, 14, 6, 6]
    # 9 targets!
    #
    # Frames (sorted y,x):
    # pcrvmjfjzg3 at (8,18) c8: 3 slots at x=10,16,22, y=20
    # pcrvmjfjzg3 at (34,18) c14: 3 slots at x=36,42,48, y=20
    # pcrvmjfjzg3 at (8,32) c9: 3 slots at x=10,16,22, y=34
    # pcrvmjfjzg3 at (34,32) c12: 3 slots at x=36,42,48, y=34
    #
    # 4 frames × 3 slots = 12 slots! 9 targets. Plenty of room.
    # But only 9 palette items needed. Let me check.
    #
    # Palette: c15@(36,56), c6@(29,56), c11@(22,56), c15@(15,56), c6@(8,56), c11@(1,56)
    # Items in frames: c14@(36,20) in F2.s0, c9@(10,34) in F3.s0, c12@(36,34) in F4.s0
    # 6 palette + 3 pre-placed = 9 items
    #
    # Portals: c14@(43,56), c9@(50,56), c12@(57,56) -> 3 portals on palette
    # 9 items + 3 portals = 12 = 12 slots ✓
    #
    # Walk order through frames:
    # Frame1 (8,18) c8 is the root (sorted first by y,x)
    # F1 slots: s0(10,20), s1(16,20), s2(22,20)
    # F1 has pre-placed item c14 at (36,20)? No, that's in Frame2 (34,18).
    # Actually c14 is an lngftsryyw at (36,20). Frame2 is at (34,18), slots at x=36,42,48 y=20.
    # So c14 at (36,20) = Frame2.slot0! Pre-placed.
    #
    # Portals: c14@(43,56)->Frame2, c9@(50,56)->Frame3, c12@(57,56)->Frame4
    #
    # F1 has susublrply at (10,20),(16,20),(22,20) and other slots
    # Also susublrply at (48,20),(22,34),(48,34) in other frames
    # And items already in frames: c14@(36,20), c9@(10,34), c12@(36,34)
    # These items have sys_click REMOVED (y <= 53).
    #
    # F1 slots: s0(10,20), s1(16,20), s2(22,20)
    # F2 (34,18) c14 slots: s0(36,20)=c14_preplaced, s1(42,20)=slot, s2(48,20)=slot
    # F3 (8,32) c9 slots: s0(10,34)=c9_preplaced, s1(16,34)=slot, s2(22,34)=slot
    # F4 (34,32) c12 slots: s0(36,34)=c12_preplaced, s1(42,34)=slot, s2(48,34)=slot
    #
    # Walk: F1.s0->T0=9, F1.s1->T1=11, F1.s2->T2=11(or portal?)
    # What's in F1 slots? Need to check L6 source more carefully.
    # L6 source items NOT at y=56:
    # lngftsryyw at (36,20) c14 -> Frame2.s0
    # lngftsryyw at (10,34) c9 -> Frame3.s0
    # lngftsryyw at (36,34) c12 -> Frame4.s0
    # These have sys_click removed (y ≤ 53).
    #
    # L6 portals: vgszefyyyp at (43,56) c14, (50,56) c9, (57,56) c12
    # All on palette row (y=56 > 53), so sys_click active.
    #
    # L6 palette items: c15@(36,56), c6@(29,56), c11@(22,56), c15@(15,56), c6@(8,56), c11@(1,56)
    # 6 items.
    #
    # So I have: 6 palette items + 3 palette portals + 3 pre-placed items = 12 pieces, 12 slots.
    # Each frame has 3 slots. With portals, the walk traverses multiple frames.
    #
    # Walk through F1 with portals in certain slots:
    # Place portals in F1 to route to F2, F3, F4.
    # F1.s0 = palette item, F1.s1 = portal(c14)->F2, F1.s2 = portal(c9)->F3?
    # But I also need portal(c12)->F4 somewhere.
    # F1 only has 3 slots. If 2 are portals, 1 is item. But I need to visit 3 sub-frames.
    # With nested portals: portal in F1 -> F2, portal in F2 -> F3?
    # No, portals route by color. Portal(c14) -> F2(c14). Portal in F2 could route to F3 if F2 has a portal(c9).
    # But sub-frame portals aren't placed yet - they need to be in the frame's slot positions.
    #
    # Actually, with 4 frames and chained portals:
    # F1.s0 -> T0
    # F1.s1 = portal(c14) -> F2
    #   F2.s0 -> T1 (pre-placed c14)
    #   F2.s1 = portal(c9) -> F3
    #     F3.s0 -> T2 (pre-placed c9)
    #     F3.s1 -> T3
    #     F3.s2 = portal(c12) -> F4
    #       F4.s0 -> T4 (pre-placed c12)
    #       F4.s1 -> T5
    #       F4.s2 -> T6
    #     Back to F3
    #   Back to F2
    #   F2.s2 -> T7
    # Back to F1
    # F1.s2 -> T8
    # Total: 9 targets! ✓
    #
    # But wait, the pre-placed items (c14, c9, c12) need to match their targets.
    # T = [9, 11, 11, 12, 15, 15, 14, 6, 6]
    # T1=11 but F2.s0 has c14! 11≠14.
    #
    # Hmm. The pre-placed items have specific colors that may not match.
    # I need to figure out which walk order makes the pre-placed colors match.
    #
    # Pre-placed: F2.s0=c14, F3.s0=c9, F4.s0=c12
    # Targets: [9, 11, 11, 12, 15, 15, 14, 6, 6]
    # c14 needs to match target 14 -> T6=14 ✓
    # c9 needs to match target 9 -> T0=9 ✓
    # c12 needs to match target 12 -> T3=12 ✓
    #
    # So I need: F3.s0 = T0=9 (c9 pre-placed ✓)
    # But F3 is accessed via portal. How can F3.s0 match T0?
    # T0 is the FIRST target. F3 is accessed through nested portals.
    # Unless F3 is the ROOT frame...
    #
    # Wait, qaagahahj = frames sorted by (y,x):
    # F1 at (8,18) -> (y=18, x=8)
    # F2 at (34,18) -> (y=18, x=34)
    # F3 at (8,32) -> (y=32, x=8)
    # F4 at (34,32) -> (y=32, x=34)
    # Sorted by (y,x): F1(18,8), F2(18,34), F3(32,8), F4(32,34)
    # Root frame = F1 at (8,18). Walk starts here.
    #
    # I need a walk order where pre-placed items land on matching targets.
    # c14=target 14 at position 6, c9=target 9 at position 0, c12=target 12 at position 3.
    #
    # Can F3 (with c9 pre-placed) be the first frame visited?
    # Only if there's a portal from F1.s0 to F3.
    # Place portal(c9) at F1.s0. Walk: F1.s0 = PORTAL -> F3.
    # F3.s0 (pre-placed c9) -> T0=9 ✓!
    #
    # Then from F3, continue with F3.s1, F3.s2.
    # Place portal(c12) at F3.s2. Walk: F3.s1->T1, F3.s2=PORTAL->F4.
    # F4.s0 (pre-placed c12) -> ?. At this point, targets consumed: T0(via F3.s0), T1(via F3.s1).
    # Next target = T2=11. F4.s0=c12, T2=11. 12≠11!
    #
    # What if F3.s1 is also a portal? portal(c12) at F3.s1?
    # Walk: F1.s0=PORTAL(c9)->F3, F3.s0=c9->T0=9✓, F3.s1=PORTAL(c12)->F4
    # F4.s0=c12->T1=11. 12≠11!
    #
    # Hmm. The pre-placed items dictate where each frame can appear in the walk.
    # c9 must match T0=9 or any position with target 9. Only T0=9.
    # c14 must match any position with target 14. T6=14.
    # c12 must match any position with target 12. T3=12.
    #
    # So F3 (c9) must be at T0, F4 (c12) must be at T3, F2 (c14) must be at T6.
    #
    # Walk positions:
    # T0: F3.s0 (c9=9) ✓
    # T1: ?
    # T2: ?
    # T3: F4.s0 (c12=12) ✓
    # T4: ?
    # T5: ?
    # T6: F2.s0 (c14=14) ✓
    # T7: ?
    # T8: ?
    #
    # I need a walk order that places F3.s0 at T0, F4.s0 at T3, F2.s0 at T6.
    #
    # One option: F1.s0 = PORTAL(c9) -> F3
    #   F3: s0=c9->T0, s1->T1=11, s2->T2=11
    #   But then after F3, we're back at F1.s1.
    #   F1.s1 -> T3=12? I need F4.s0=c12 at T3.
    #   F1.s1 = PORTAL(c12) -> F4? That works!
    #   F4: s0=c12->T3=12✓, s1->T4=15, s2->T5=15
    #   Back to F1.s2.
    #   F1.s2 = PORTAL(c14) -> F2?
    #   F2: s0=c14->T6=14✓, s1->T7=6, s2->T8=6
    #   Back to F1. No more slots. Done!
    #   Total: 3 + 3 + 3 = 9 targets. ✓!
    #
    # But F1 has ALL 3 slots as portals. No items in F1 itself!
    # That's fine - all items go into sub-frames.
    #
    # Remaining items to place:
    # F3.s1: T1=11 -> c11 from palette (22,56) or (1,56)
    # F3.s2: T2=11 -> c11 from palette
    # F4.s1: T4=15 -> c15 from palette (36,56) or (15,56)
    # F4.s2: T5=15 -> c15 from palette
    # F2.s1: T7=6 -> c6 from palette (29,56) or (8,56)
    # F2.s2: T8=6 -> c6 from palette
    #
    # F1: s0=portal(c9), s1=portal(c12), s2=portal(c14)
    # F3: s0=c9(pre-placed), s1=c11, s2=c11
    # F4: s0=c12(pre-placed), s1=c15, s2=c15
    # F2: s0=c14(pre-placed), s1=c6, s2=c6
    #
    # Palette items used: c11@(22,56), c11@(1,56), c15@(36,56), c15@(15,56), c6@(29,56), c6@(8,56)
    # All 6 palette items used! ✓
    # Portals used: c9@(46,56), c12@(57,56), c14@(43,56)
    # All 3 portals used! ✓

    print("  L6: three-portal nested strategy...")
    l6_start = len(total_actions)

    # F1 slots at (10,20),(16,20),(22,20)
    # F1.s0 = portal(c9)@(46,56) -> slot(10,20)
    # F1.s1 = portal(c12)@(57,56) -> slot(16,20)
    # F1.s2 = portal(c14)@(43,56) -> slot(22,20)
    # F3 slots at (10,34),(16,34),(22,34)
    # F3.s1 = c11@(1,56) -> slot(16,34)
    # F3.s2 = c11@(22,56) -> slot(22,34)
    # F4 slots at (36,34),(42,34),(48,34)
    # F4.s1 = c15@(15,56) -> slot(42,34)
    # F4.s2 = c15@(36,56) -> slot(48,34)
    # F2 slots at (36,20),(42,20),(48,20)
    # F2.s1 = c6@(8,56) -> slot(42,20)
    # F2.s2 = c6@(29,56) -> slot(48,20)

    L6 = [
        ((46,56), (10,20)),   # portal(c9) -> F1.s0
        ((57,56), (16,20)),   # portal(c12) -> F1.s1
        ((43,56), (22,20)),   # portal(c14) -> F1.s2
        ((1,56), (16,34)),    # c11 -> F3.s1
        ((22,56), (22,34)),   # c11 -> F3.s2
        ((15,56), (42,34)),   # c15 -> F4.s1
        ((36,56), (48,34)),   # c15 -> F4.s2
        ((8,56), (42,20)),    # c6 -> F2.s1
        ((29,56), (48,20)),   # c6 -> F2.s2
    ]
    for p, s in L6:
        place(env, p, s)
        record_place(p, s)
    obs = submit(env)
    record_submit()
    lc = obs.levels_completed
    print(f"    L6: levels={lc}")
    l6_acts = len(total_actions) - l6_start

    # ===== L7 =====
    # Targets (y=1 sorted x): c8@8, c9@15, c14@22, c11@29, c14@36, c9@43, c8@50
    # Target: [8, 9, 14, 11, 14, 9, 8]
    # 7 targets
    #
    # Frames (sorted y,x):
    # pcrvmjfjzg3 at (21,12) c8: 3 slots at x=23,29,35, y=14
    # pcrvmjfjzg3 at (21,25) c9: 3 slots at x=23,29,35, y=27
    # pcrvmjfjzg3 at (21,38) c14: 3 slots at x=23,29,35, y=40
    #
    # 3 frames × 3 slots = 9 slots. 7 targets.
    #
    # Item in frame: c11@(29,40) in F3.s1
    # Portals: c14@(46,56), c9@(53,56)
    # Palette: c14@(39,56), c14@(25,56), c8@(18,56), c9@(11,56), c8@(4,56), c9@(32,56)
    # Wait let me recheck...
    # L7 source items:
    # lngftsryyw at (39,56) c14, (25,56) c14, (18,56) c8, (29,40) c11, (11,56) c9, (4,56) c8, (32,56) c9
    # That's: palette c14@(39,56), c14@(25,56), c8@(18,56), c9@(11,56), c8@(4,56), c9@(32,56) = 6 palette
    # Pre-placed: c11@(29,40) = F3.s1
    # Portals: c14@(46,56), c9@(53,56) = 2 palette portals
    # Total: 6 + 1 + 2 = 9 = 9 slots ✓
    #
    # Root frame = F1 at (21,12). Walk starts here.
    # Pre-placed: F3.s1(29,40)=c11. c11 must match which target? T3=11. ✓
    #
    # Walk order: need F3.s1 at T3 position.
    # F1(c8): s0->T0, s1->T1, s2->T2 (3 targets)
    # If F1.s1 = portal(c9) -> F2:
    #   F2: s0->T1, s1->T2, s2->T3...
    # Hmm this gets complex. Let me think about it differently.
    #
    # c11 pre-placed in F3.s1 must match T3=11. So F3.s1 must be checked at T3.
    # That means: before F3.s1, exactly 3 targets have been checked (T0,T1,T2).
    # F3.s0 checked at some point before F3.s1.
    # Total items before F3.s1: at least 1 (F3.s0) + items from F1 and portal paths.
    #
    # One approach: F1.s0->T0, F1.s1=portal(c14)->F3
    # F3: s0->T1, s1(pre-placed c11)->T2=14. 11≠14! Bad.
    #
    # Another: F1.s0->T0, F1.s1->T1, F1.s2=portal(c14)->F3
    # F3: s0->T2=14, s1(c11)->T3=11✓, s2->T4=14
    # Back to F1. No more slots.
    # Then we've checked T0-T4 = 5 targets. Need T5,T6 too.
    # F1 has no more slots. Where do T5,T6 go?
    #
    # Need portal from F3 to F2 for more nesting.
    # F3.s2 = portal(c9)->F2. F2: s0->T5, s1->T6, s2->T7? But only 7 targets total.
    #
    # Wait, with this walk:
    # F1.s0->T0=8, F1.s1->T1=9, F1.s2=portal(c14)->F3
    #   F3.s0->T2=14, F3.s1(c11)->T3=11✓, F3.s2=portal(c9)->F2
    #     F2.s0->T4=14, F2.s1->T5=9, F2.s2->T6=8
    #     Back to F3. F3 done. Back to F1. F1 done.
    # Total: 7 targets. ✓!
    #
    # Items to place:
    # F1.s0(23,14): c8 -> T0=8. Palette c8@(4,56) or (18,56)
    # F1.s1(29,14): c9 -> T1=9. Palette c9@(11,56) or (32,56)
    # F1.s2(35,14): portal(c14) -> from palette (46,56)
    # F3.s0(23,40): c14 -> T2=14. Palette c14@(39,56) or (25,56)
    # F3.s1(29,40): c11 PRE-PLACED -> T3=11 ✓
    # F3.s2(35,40): portal(c9) -> from palette (53,56)
    # F2.s0(23,27): c14 -> T4=14. Palette c14@(25,56) or (39,56)
    # F2.s1(29,27): c9 -> T5=9. Palette c9@(32,56) or (11,56)
    # F2.s2(35,27): c8 -> T6=8. Palette c8@(18,56) or (4,56)
    #
    # Item assignment:
    # c8@(4,56) -> F1.s0(23,14)
    # c9@(11,56) -> F1.s1(29,14)
    # portal(c14)@(46,56) -> F1.s2(35,14)
    # c14@(39,56) -> F3.s0(23,40)
    # (F3.s1 pre-placed)
    # portal(c9)@(53,56) -> F3.s2(35,40)
    # c14@(25,56) -> F2.s0(23,27)
    # c9@(32,56) -> F2.s1(29,27)
    # c8@(18,56) -> F2.s2(35,27)

    print("  L7: nested portal strategy...")
    l7_start = len(total_actions)

    L7 = [
        ((4,56), (23,14)),    # c8 -> F1.s0
        ((11,56), (29,14)),   # c9 -> F1.s1
        ((46,56), (35,14)),   # portal(c14) -> F1.s2
        ((39,56), (23,40)),   # c14 -> F3.s0
        ((53,56), (35,40)),   # portal(c9) -> F3.s2
        ((25,56), (23,27)),   # c14 -> F2.s0
        ((32,56), (29,27)),   # c9 -> F2.s1
        ((18,56), (35,27)),   # c8 -> F2.s2
    ]
    for p, s in L7:
        place(env, p, s)
        record_place(p, s)
    obs = submit(env)
    record_submit()
    lc = obs.levels_completed
    print(f"    L7: levels={lc}")
    l7_acts = len(total_actions) - l7_start

    # ===== L8 =====
    # Targets sorted by (y,x):
    # (11,1):c8, (18,1):c11, (25,1):c12, (32,1):c9, (39,1):c14, (46,1):c15
    # (11,8):c8, (18,8):c11, (25,8):c12, (32,8):c9, (39,8):c14, (46,8):c15
    # 12 targets: [8,11,12,9,14,15, 8,11,12,9,14,15]
    #
    # Frames (sorted y,x):
    # qdmvvkvhaz4 at (18,22) c8: 4 slots at x=20,26,32,38 y=24
    # qdmvvkvhaz4 at (18,36) c9: 4 slots at x=20,26,32,38 y=38
    #
    # 2 frames × 4 slots = 8 slots. 12 targets.
    # Palette: c14@(39,56), c8@(25,56), c9@(32,56), c12@(18,56), c11@(11,56), c15@(4,56)
    # Portals: c8@(46,56), c9@(53,56)
    # 6 items + 2 portals = 8 = 8 slots ✓
    #
    # Circular portal: portal(c9) in F1 last slot, portal(c8) in F2 last slot
    # Walk: F1.s0->T0=8, F1.s1->T1=11, F1.s2->T2=12,
    #        F1.s3=portal(c9)->F2,
    #        F2.s0->T3=9, F2.s1->T4=14, F2.s2->T5=15,
    #        F2.s3=portal(c8)->F1,
    #        F1.s0->T6=8(same item!), F1.s1->T7=11(same!), F1.s2->T8=12(same!),
    #        F1.s3=portal(c9)->F2,
    #        F2.s0->T9=9(same!), F2.s1->T10=14(same!), F2.s2->T11=15(same!) -> WIN!
    #
    # T = [8,11,12,9,14,15, 8,11,12,9,14,15]
    # Period = 6 = 3 F1 items + 3 F2 items. Repeated twice. ✓
    #
    # Items:
    # F1.s0(20,24): c8@(25,56)
    # F1.s1(26,24): c11@(11,56)
    # F1.s2(32,24): c12@(18,56)
    # F1.s3(38,24): portal(c9)@(53,56)
    # F2.s0(20,38): c9@(32,56)
    # F2.s1(26,38): c14@(39,56)
    # F2.s2(32,38): c15@(4,56)
    # F2.s3(38,38): portal(c8)@(46,56)

    print("  L8: circular portal strategy...")
    l8_start = len(total_actions)

    L8 = [
        ((25,56), (20,24)),   # c8 -> F1.s0
        ((11,56), (26,24)),   # c11 -> F1.s1
        ((18,56), (32,24)),   # c12 -> F1.s2
        ((53,56), (38,24)),   # portal(c9) -> F1.s3
        ((32,56), (20,38)),   # c9 -> F2.s0
        ((39,56), (26,38)),   # c14 -> F2.s1
        ((4,56), (32,38)),    # c15 -> F2.s2
        ((46,56), (38,38)),   # portal(c8) -> F2.s3
    ]
    for p, s in L8:
        place(env, p, s)
        record_place(p, s)
    obs = submit(env)
    record_submit()
    lc = obs.levels_completed
    print(f"    L8: levels={lc}")
    l8_acts = len(total_actions) - l8_start

    print(f"\n=== FINAL: levels_completed={lc}, state={obs.state} ===")
    print(f"Total actions: {len(total_actions)}")

    # Save solution
    result = {
        "game": "sb26",
        "source": "analytical_solver",
        "type": "fullchain",
        "version_hash": "7fbdac44",
        "total_actions": len(total_actions),
        "max_level": lc,
        "all_actions": total_actions,
        "per_level": {
            "L1": l1_acts,
            "L2": l2_acts + l1_acts,  # cumulative
        }
    }

    out_path = "B:/M/the-search/experiments/results/prescriptions/sb26_fullchain.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

    return lc

if __name__ == "__main__":
    lc = solve()
