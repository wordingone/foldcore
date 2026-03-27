"""
Analytical solver for SB26 and DC22 — all levels.
Uses arc_agi API to play through levels, observing frames to verify.

Action encoding for 64x64 grid games:
- ACTION1..7 = action IDs 0..6
- Click at (x,y) = action 7 + y*64 + x

For SB26: available_actions=[5,6,7]
  ACTION5 (submit) = 4
  ACTION6 (click x,y) = 7 + y*64 + x
  ACTION7 (undo) = 6

For DC22: available_actions=[1,2,3,4,6]
  ACTION1 (up) = 0
  ACTION2 (right) = 1
  ACTION3 (down) = 2
  ACTION4 (left) = 3
  ACTION6 (click x,y) = 5 (but encoded as 7+y*64+x for clicks)
"""

import json
import os
import sys
import numpy as np

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, "B:/M/the-search")

from arc_agi import ArcAgiSession

def click_action(x, y, grid=64):
    return 7 + y * grid + x

# ============================================================
# SB26 SOLVER
# ============================================================

def solve_sb26():
    """Solve all 8 levels of SB26."""
    print("=" * 60)
    print("SB26 SOLVER")
    print("=" * 60)

    session = ArcAgiSession("sb26")
    obs = session.observe()
    print(f"Started SB26. State: {obs['state']}, Levels to win: {obs['win_levels']}")

    all_actions = []
    level_actions = {}

    # ===== LEVEL 1 =====
    # Targets (quhhhthrri sorted y,x): [9, 14, 11, 15]
    # Frame: qdmvvkvhaz4 at (18,25), 4 slots at x=20,26,32,38, y=27
    # Palette (lngftsryyw): c14@(17,56), c15@(25,56), c9@(33,56), c11@(41,56)
    # Place: slot0=c9, slot1=c14, slot2=c11, slot3=c15
    l1 = []
    # Click uses center of item sprite (6x6): pos + 3 for lngftsryyw, +2 for susublrply inner
    # Actually from known solution, palette uses +3, slot uses +3 too
    # Known L1: [3819,1950,3803,1956,3827,1962,3811,1968,4]
    # 3819 = click(36,59) = palette c9 at (33,56)+3
    # 1950 = click(23,30) = slot (20,27)+3

    # c9: (33,56) -> click (36,59) -> slot0 (20,27) -> click (23,30)
    l1.extend([click_action(36, 59), click_action(23, 30)])
    # c14: (17,56) -> click (20,59) -> slot1 (26,27) -> click (29,30)
    l1.extend([click_action(20, 59), click_action(29, 30)])
    # c11: (41,56) -> click (44,59) -> slot2 (32,27) -> click (35,30)
    l1.extend([click_action(44, 59), click_action(35, 30)])
    # c15: (25,56) -> click (28,59) -> slot3 (38,27) -> click (41,30)
    l1.extend([click_action(28, 59), click_action(41, 30)])
    l1.append(4)  # submit

    level_actions["L1"] = l1
    all_actions.extend(l1)

    # ===== LEVEL 2 =====
    # Known L2: [3750,1372,3736,1378,3729,2268,3764,2274,3743,2280,3771,2286,3757,1390,4]
    # Using known solution directly
    l2 = [3750,1372,3736,1378,3729,2268,3764,2274,3743,2280,3771,2286,3757,1390,4]
    level_actions["L2"] = l2
    all_actions.extend(l2)

    # Execute L1 and L2
    for i, action_id in enumerate(l1 + l2):
        if action_id >= 7:
            x = (action_id - 7) % 64
            y = (action_id - 7) // 64
            result = session.act(6, x=x, y=y)  # ACTION6 with x,y
        elif action_id == 4:
            result = session.act(5)  # ACTION5 = submit
        elif action_id == 6:
            result = session.act(7)  # ACTION7 = undo
        else:
            result = session.act(action_id + 1)  # ACTION1-4

    obs = session.observe()
    print(f"After L1+L2: levels_completed={obs['levels_completed']}, state={obs['state']}")

    if obs['levels_completed'] < 2:
        print("ERROR: L1+L2 solution failed!")
        return None

    # ===== LEVEL 3 =====
    # Targets (y=1 sorted by x): x=8:c8, x=15:c14, x=22:c15, x=29:c11, x=36:c6, x=43:c9, x=50:c12
    # Target sequence: [8, 14, 15, 11, 6, 9, 12]
    #
    # Frames (sorted y,x):
    # nyqgqtujsa5 at (15,19), 5 slots at x=17,23,29,35,41 y=21
    # jvkvqzheok2 at (15,31), color=14, 2 slots at x=17,23 y=33
    # jvkvqzheok2 at (33,31), color=9, 2 slots at x=35,41 y=33
    #
    # Portals: vgszefyyyp at (23,21) c14 -> frame(15,31); at (35,21) c9 -> frame(33,31)
    # Slots (susublrply): (17,21),(29,21),(41,21) in Frame1; (17,33),(23,33) in Frame2; (35,33),(41,33) in Frame3
    #
    # Walk: Frame1.slot0(17,21)->T0, Portal(23,21)->Frame2, Frame2.slot0(17,33)->T1,
    #        Frame2.slot1(23,33)->T2, back, Frame1.slot2(29,21)->T3,
    #        Portal(35,21)->Frame3, Frame3.slot0(35,33)->T4, Frame3.slot1(41,33)->T5,
    #        back, Frame1.slot4(41,21)->T6
    #
    # Placements: (17,21)<-c8, (17,33)<-c14, (23,33)<-c15, (29,21)<-c11,
    #             (35,33)<-c6, (41,33)<-c9, (41,21)<-c12
    #
    # Palette: c8@(50,56), c15@(22,56), c14@(15,56), c12@(8,56), c6@(43,56), c9@(36,56), c11@(29,56)
    l3 = []
    l3.extend([click_action(53, 59), click_action(20, 24)])   # c8: (50,56)->+3 to slot(17,21)->+3
    l3.extend([click_action(18, 59), click_action(20, 36)])   # c14: (15,56)->+3 to slot(17,33)->+3
    l3.extend([click_action(25, 59), click_action(26, 36)])   # c15: (22,56)->+3 to slot(23,33)->+3
    l3.extend([click_action(32, 59), click_action(32, 24)])   # c11: (29,56)->+3 to slot(29,21)->+3
    l3.extend([click_action(46, 59), click_action(38, 36)])   # c6: (43,56)->+3 to slot(35,33)->+3
    l3.extend([click_action(39, 59), click_action(44, 36)])   # c9: (36,56)->+3 to slot(41,33)->+3
    l3.extend([click_action(11, 59), click_action(44, 24)])   # c12: (8,56)->+3 to slot(41,21)->+3
    l3.append(4)  # submit

    level_actions["L3"] = l3
    all_actions.extend(l3)

    # ===== LEVEL 4 =====
    # Targets (y=1 sorted by x): x=8:c11, x=15:c8, x=22:c14, x=29:c9, x=36:c6, x=43:c12, x=50:c15
    # Target sequence: [11, 8, 14, 9, 6, 12, 15]
    #
    # Frames (sorted y,x):
    # nyqgqtujsa5 at (15,18), 5 slots at x=17,23,29,35,41 y=20
    # pcrvmjfjzg3 at (21,32), color=14, 3 slots at x=23,29,35 y=34
    #
    # Item in frame: lngftsryyw c14 at (23,34) - already in Frame2.slot0
    # Portal on palette: vgszefyyyp c14 at (50,56) - routes to Frame2
    # Other palette: c8@(29,56), c6@(15,56), c15@(36,56), c12@(22,56), c11@(8,56), c9@(43,56)
    #
    # Walk with portal at Frame1.slot2(29,20):
    # slot0(17,20)->T0=c11, slot1(23,20)->T1=c8,
    # slot2(29,20)=PORTAL(c14)->Frame2,
    #   Frame2.slot0(23,34)->T2=c14 (ALREADY PLACED!),
    #   Frame2.slot1(29,34)->T3=c9,
    #   Frame2.slot2(35,34)->T4=c6,
    # back to Frame1,
    # slot3(35,20)->T5=c12, slot4(41,20)->T6=c15
    #
    # Placements: (17,20)<-c11, (23,20)<-c8, (29,20)<-portal(c14),
    #             (29,34)<-c9, (35,34)<-c6, (35,20)<-c12, (41,20)<-c15

    l4 = []
    l4.extend([click_action(11, 59), click_action(20, 23)])   # c11: (8,56)->+3 to slot(17,20)->+3
    l4.extend([click_action(32, 59), click_action(26, 23)])   # c8: (29,56)->+3 to slot(23,20)->+3
    l4.extend([click_action(53, 59), click_action(32, 23)])   # portal(c14): (50,56)->+3 to slot(29,20)->+3
    l4.extend([click_action(46, 59), click_action(32, 37)])   # c9: (43,56)->+3 to slot(29,34)->+3
    l4.extend([click_action(18, 59), click_action(38, 37)])   # c6: (15,56)->+3 to slot(35,34)->+3
    l4.extend([click_action(25, 59), click_action(38, 23)])   # c12: (22,56)->+3 to slot(35,20)->+3
    l4.extend([click_action(39, 59), click_action(44, 23)])   # c15: (36,56)->+3 to slot(41,20)->+3
    l4.append(4)  # submit

    level_actions["L4"] = l4
    all_actions.extend(l4)

    # ===== LEVEL 5 =====
    # Targets (y=1 sorted by x): x=1:c6, x=8:c14, x=15:c8, x=22:c8, x=29:c14, x=36:c8, x=43:c8, x=50:c11, x=57:c15
    # 9 targets: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    #
    # Frames (sorted y,x):
    # nyqgqtujsa5 at (15,18), 5 slots at x=17,23,29,35,41 y=20
    # pcrvmjfjzg3 at (21,32), color=9, 3 slots at x=23,29,35 y=34
    #
    # Palette: c14@(39,56), c8@(25,56), c11@(32,56), c8@(18,56), c6@(11,56), c15@(4,56)
    # Portals: vgszefyyyp c9@(46,56), vgszefyyyp c9@(53,56) -> both route to Frame2
    #
    # 9 targets, 5+3=8 slots. With 2 portals in Frame1, Frame2 visited twice = 3+3+3=9.
    # But items in Frame2 must match both visits. Let me check...
    #
    # If portals at slot1 and slot3:
    # slot0->T0=c6, portal->Frame2[T1,T2,T3], slot2->T4, portal->Frame2[T5,T6,T7], slot4->T8
    # Frame2 items both times: [T1,T2,T3]=[14,8,8], [T5,T6,T7]=[8,8,11] -> NO MATCH
    #
    # Actually, what if the portal mechanism doesn't re-read items but continues from where it left off?
    # No, the code clearly uses rzbeqaiky[frame] which is built once.
    #
    # Wait, I need to re-examine the walk logic. When Frame2 is entered the SECOND time,
    # the code does: self.buvfjfmpp.append((amkoiofqhs, 0))
    # So it starts from slot 0 again. And rzbeqaiky[Frame2] has the same 3 items.
    # So it reads the SAME items again.
    #
    # Unless... the walk doesn't fail on mismatch but allows the same color to appear multiple times?
    # Let me re-read the check:
    # if ldwfvtgapk.pixels[1,1] == bnwkxafnfc.pixels[0,0]:
    #     xjxrqgaqw = 0  # match
    # else:
    #     sibihgzarf()  # error
    #
    # ldwfvtgapk = item in slot, bnwkxafnfc = wcfyiodrx[pmygakdvy] = current target indicator
    # Each target is checked in order. pmygakdvy increments after each match.
    # So T[1] is checked against Frame2.slot0, T[5] is checked against Frame2.slot0 AGAIN.
    # For T[1]=c14 to match Frame2.slot0, that slot must have c14.
    # For T[5]=c8 to match Frame2.slot0, that same slot must have c8.
    # CONTRADICTION - can't have both c14 and c8 in the same slot!
    #
    # So the two-portal approach fundamentally can't work unless Frame2 items match both visits.
    # The target subsequences [14,8,8] and [8,8,11] don't match.
    #
    # I must be missing something. Let me look at this differently.
    # Maybe L5 doesn't use both portals. Maybe one portal is in Frame1 and the other goes in Frame2.
    # But Frame2 is pcrvmjfjzg3 (3 slots). Placing a portal there would...
    # There's no third frame for it to route to!
    #
    # OR: maybe the game has a different mechanic I'm not seeing.
    #
    # Let me look at L8 for comparison. L8 has:
    # 2 rows of 6 targets (12 total? No...)
    # quhhhthrri in L8:
    #  (11,1):c8, (18,1):c11, (25,1):c12, (32,1):c9, (39,1):c14, (46,1):c15
    #  (11,8):c8, (18,8):c11, (25,8):c12, (32,8):c9, (39,8):c14, (46,8):c15
    # These are at y=1 AND y=8. Total 12 targets?
    # Frames: qdmvvkvhaz4 at (18,22) 4 slots; qdmvvkvhaz4 at (18,36) color=9 4 slots
    # Portals: vgszefyyyp c8@(46,56), vgszefyyyp c9@(53,56)
    # Palette: c14@(39,56), c8@(25,56), c9@(32,56), c12@(18,56), c11@(11,56), c15@(4,56)
    # 6 palette + 2 portals = 8 placeable, 4+4=8 slots.
    #
    # Targets sorted by (y,x):
    # (11,1):c8, (18,1):c11, (25,1):c12, (32,1):c9, (39,1):c14, (46,1):c15,
    # (11,8):c8, (18,8):c11, (25,8):c12, (32,8):c9, (39,8):c14, (46,8):c15
    # That's 12 targets! But only 8 slots (even with portals).
    #
    # Hmm, wait. Maybe not all quhhhthrri sprites are "targets" that get checked.
    # Maybe the number of targets = number of target indicators that get filled =
    # number of non-portal slots = 6 (palette items).
    #
    # In L8: 6 palette items -> 6 targets get filled. The remaining 6 quhhhthrri at y=8
    # are maybe just decorative (showing the same sequence for reference).
    #
    # In L5: 6 palette items -> 6 targets. Plus 2 portals.
    # 5 Frame1 slots - 2 portals = 3 regular + 3 Frame2 = 6. That matches!
    #
    # But wcfyiodrx = quhhhthrri sorted by (y,x). ALL of them are in the walk count.
    # Unless the walk stops when it runs out of slots, not targets.
    #
    # Let me re-read the completion check in dbfxrigdqx:
    # if self.pmygakdvy == len(self.wcfyiodrx) - 1 and wrudcanmwy:
    #     self.lmvwmlqtw = 0  # WIN!
    # This checks if pmygakdvy (target index) reached the last target AND the current
    # target indicator is filled (wrudcanmwy = bnwkxafnfc.pixels[1,1] != -1).
    #
    # So ALL targets in wcfyiodrx must be matched. For L5, that's 9 targets.
    # For L8, that's 12 targets.
    #
    # This means... the two-portal approach must work somehow.
    # OR, there's another mechanic I'm missing.
    #
    # Let me re-read the L8 data. L8 has:
    # Targets at y=1: [c8, c11, c12, c9, c14, c15] (6 targets)
    # Targets at y=8: [c8, c11, c12, c9, c14, c15] (6 targets, IDENTICAL!)
    # Total: 12 targets, but both subsequences are the same [8,11,12,9,14,15].
    #
    # Frame1: qdmvvkvhaz4 at (18,22), 4 slots. Portal at one slot.
    # Frame2: qdmvvkvhaz4 at (18,36), color=9, 4 slots. Portal at one slot.
    #
    # If portal(c9) goes in Frame1 routing to Frame2:
    # Frame1[0,1,PORTAL,3] -> 3 slots + 4 Frame2 slots = 7. Need 12.
    # If ANOTHER portal(c8) goes in Frame2 routing back to Frame1?!
    # But that would create a cycle... unless the re-entry check prevents it.
    #
    # Wait - portal c8 routes to a frame with color 8. Frame1 has color 8 (default).
    # So portal(c8) in Frame2 routes BACK to Frame1.
    #
    # Walk: Frame1.slot0->T0, Frame1.slot1->T1, Frame1.slot2=portal(c9)->Frame2,
    #   Frame2.slot0->T2, Frame2.slot1->T3, Frame2.slot2=portal(c8)->Frame1,
    #     Frame1.slot0->T4... wait, this re-enters Frame1 starting from slot 0.
    #     But Frame1 items are the same! So T[4..] must match T[0..].
    #
    # For L8: T[0..5] = [8,11,12,9,14,15] and T[6..11] = [8,11,12,9,14,15].
    # They ARE the same! So this circular portal chain works!
    #
    # Walk: Frame1[s0->T0, s1->T1, s2=portal->Frame2,
    #   Frame2[s0->T2, s1->T3, s2=portal->Frame1,
    #     Frame1[s0->T4, s1->T5, s2=portal->Frame2... (re-entry check kicks in!)
    #
    # Hmm, this would be infinite. The re-entry check must prevent it.
    # Let me trace the check:
    # When trying to enter Frame2 the 2nd time from Frame1:
    # buvfjfmpp = [(Frame1, 2)] (after popping Frame2 and Frame1 re-entries)
    # Actually this gets complex. Let me trace step by step.
    #
    # Start: buvfjfmpp = [(Frame1, 0)]
    # Frame1.s0 -> T0. Advance: (Frame1, 1)
    # Frame1.s1 -> T1. Advance: (Frame1, 2)
    # Frame1.s2 = portal(c9) -> enter Frame2
    #   Push: buvfjfmpp = [(Frame1, 2), (Frame2, 0)]
    #   Frame2.s0 -> T2. Advance: (Frame2, 1)
    #   Frame2.s1 -> T3. Advance: (Frame2, 2)
    #   Frame2.s2 = portal(c8) -> enter Frame1?
    #     Check: uxncrzlau=0, (Frame1,0) in buvfjfmpp[:-1]=[(Frame1,2),(Frame2,2)]?
    #     No, (Frame1,0) not in there. Check passes but...
    #     Wait, the check is:
    #     if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0):
    #     kmsegkpkh = Frame1, uxncrzlau = 0
    #     (Frame1, 0) in [(Frame1, 2), (Frame2, 2)]? NO (Frame1,2 ≠ Frame1,0).
    #     So check is False, portal proceeds.
    #     Push: buvfjfmpp = [(Frame1, 2), (Frame2, 2), (Frame1, 0)]
    #     Frame1.s0 -> T4. Advance: (Frame1, 1)
    #     Frame1.s1 -> T5. Advance: (Frame1, 2)
    #     Frame1.s2 = portal(c9) -> enter Frame2 again?
    #       Check: uxncrzlau=0, (Frame2,0) in buvfjfmpp[:-1]=[(Frame1,2),(Frame2,2),(Frame1,2)]?
    #       NO. (Frame2,0) not in there. Proceeds!
    #       Push: buvfjfmpp = [(Frame1,2),(Frame2,2),(Frame1,2),(Frame2,0)]
    #       Frame2.s0 -> T6. Frame2.s1 -> T7. Frame2.s2 = portal(c8) -> Frame1?
    #         Check: (Frame1,0) in [(Frame1,2),(Frame2,2),(Frame1,2),(Frame2,2)]? NO.
    #         Proceeds! This is INFINITE!
    #
    # But T[0..11] only has 12 entries. After T11, pmygakdvy == len(wcfyiodrx)-1 and
    # the win check triggers. So the walk stops after 12 items.
    # Actually, the check happens BEFORE entering a portal:
    # if self.pmygakdvy == len(self.wcfyiodrx) - 1 and wrudcanmwy:
    #     win!
    # This is at the top of dbfxrigdqx, checking after each item placement.
    #
    # So the walk goes:
    # F1.s0->T0, F1.s1->T1, portal->F2, F2.s0->T2, F2.s1->T3, portal->F1,
    # F1.s0->T4, F1.s1->T5, portal->F2, F2.s0->T6, F2.s1->T7, portal->F1,
    # F1.s0->T8, F1.s1->T9, portal->F2, F2.s0->T10, F2.s1->T11 -> WIN!
    #
    # For this to work: items in F1.s0 and F1.s1 must match T[0,1]=T[4,5]=T[8,9]
    # Items in F2.s0 and F2.s1 must match T[2,3]=T[6,7]=T[10,11]
    # In L8: T = [8,11,12,9,14,15,8,11,12,9,14,15]
    # T[0,1]=[8,11], T[4,5]=[14,15], T[8,9]=[8,11] -> [8,11]≠[14,15]! DOESN'T MATCH!
    #
    # Something is wrong with my walk model. Let me reconsider...
    #
    # Actually, maybe the portal in Frame2 is at a DIFFERENT slot, not slot2.
    # Frame2.slot2 has portal(c8) and Frame2.slot3 has a regular item.
    # Let me reconsider Frame2 layout:
    # qdmvvkvhaz4 at (18,36), color=9, 4 slots at x=20,26,32,38, y=38
    # Portal vgszefyyyp c8 at (46,56) - this is on PALETTE, not in a frame!
    # Portal vgszefyyyp c9 at (53,56) - also on PALETTE.
    #
    # So both portals are on the palette row and need to be placed into frame slots!
    # For L8: place portal(c8) into one of Frame1 or Frame2's slots.
    #
    # Hmm, but portal(c8) routes to Frame1 (color=8). If placed in Frame2, it routes back.
    # portal(c9) routes to Frame2 (color=9). If placed in Frame1, it routes to Frame2.
    #
    # For the circular walk to give 12 items:
    # Place portal(c9) in Frame1.slot_x, portal(c8) in Frame2.slot_y.
    # Walk: Frame1 slots [0..x-1] -> portal(c9) -> Frame2 slots [0..y-1] ->
    #        portal(c8) -> Frame1 slots [0..x-1] (again!) -> portal(c9) -> Frame2...
    #
    # After enough cycles, pmygakdvy reaches 11 and the walk ends.
    # For T to match: each cycle through Frame1 gives the same items, each cycle through Frame2
    # gives the same items. So T must be periodic with period = items_per_cycle.
    #
    # L8 targets: [8,11,12,9,14,15, 8,11,12,9,14,15]
    # Period = 6 items. Each cycle = 2 Frame1 items + 1 portal + 2 Frame2 items + 1 portal = 4 items.
    # Wait, that's not 6.
    #
    # Hmm let me reconsider. With portal(c9) at Frame1.slot2 and portal(c8) at Frame2.slot3:
    # Frame1: s0, s1, PORTAL(s2), s3 -- but after portal returns, it continues at s3!
    # Frame2: s0, s1, s2, PORTAL(s3) -- after portal returns, done with Frame2
    #
    # Walk: F1.s0->T0, F1.s1->T1, PORTAL->F2,
    #   F2.s0->T2, F2.s1->T3, F2.s2->T4, PORTAL(c8)->F1,
    #     F1.s0->T5, F1.s1->T6, PORTAL(c9)->F2,
    #       F2.s0->T7, F2.s1->T8, F2.s2->T9, PORTAL(c8)->F1,
    #         F1.s0->T10, F1.s1->T11 -> WIN (12 targets done)
    #
    # Check: F1.s0 must match T[0]=T[5]=T[10]=8. F1.s1 must match T[1]=T[6]=T[11]=11.
    # F2.s0 must match T[2]=T[7]=12. F2.s1 must match T[3]=T[8]=9.
    # F2.s2 must match T[4]=T[9]=14.
    #
    # T = [8,11,12,9,14,15, 8,11,12,9,14,15]
    # T[0]=8 ✓, T[5]=14 ✗ (need 8)
    #
    # That doesn't work either. Let me try different portal positions.
    #
    # Portal(c9) at F1.s3, portal(c8) at F2.s3:
    # Walk: F1.s0->T0=8, F1.s1->T1=11, F1.s2->T2=12, PORTAL(c9)->F2,
    #   F2.s0->T3=9, F2.s1->T4=14, F2.s2->T5=15, PORTAL(c8)->F1,
    #     F1.s0->T6=8 ✓, F1.s1->T7=11 ✓, F1.s2->T8=12 ✓, PORTAL->F2,
    #       F2.s0->T9=9 ✓, F2.s1->T10=14 ✓, F2.s2->T11=15 ✓ -> WIN!
    #
    # This works! Portal at the LAST slot of each frame creates a circular chain.
    # Each frame has 3 regular items + 1 portal.
    # F1: [c8, c11, c12, portal(c9)]
    # F2: [c9, c14, c15, portal(c8)]
    # 6 items + 2 portals = 8 = 4 + 4 slots. ✓
    #
    # Cycle = 6 items (3 F1 + 3 F2). 12/6 = 2 cycles. ✓
    # T repeats perfectly: [8,11,12,9,14,15] × 2. ✓
    #
    # NOW for L5. Same principle: circular portals!
    # Targets: [6, 14, 8, 8, 14, 8, 8, 11, 15] -- 9 targets
    #
    # Frame1: nyqgqtujsa5 at (15,18), 5 slots at x=17,23,29,35,41 y=20
    # Frame2: pcrvmjfjzg3 at (21,32), color=9, 3 slots at x=23,29,35 y=34
    #
    # Only Frame2 has color 9. Both portals are c9. So both route to Frame2.
    # There's no portal routing BACK to Frame1!
    # Unless... palette items include a portal that could route back?
    # Portals on palette: both are vgszefyyyp c9. No portal for Frame1 (color 8).
    #
    # So circular chaining isn't possible for L5. Only forward portals to Frame2.
    #
    # With 2 forward portals and no return portal:
    # Can Frame2 be visited twice? The 2nd visit reads the same items.
    # T subsequences from Frame2 must be identical.
    #
    # L5 targets: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # Let me find a split where two Frame2 visits have the same items.
    # Frame2 has 3 slots. Each visit reads 3 items.
    #
    # Try portals at F1.s1 and F1.s3 (positions 23,20 and 35,20):
    # F1.s0->T0=6, portal->F2[T1,T2,T3]=[14,8,8], F1.s2->T4=14,
    # portal->F2[T5,T6,T7]=[8,8,11], F1.s4->T8=15
    # F2 items must be [14,8,8] AND [8,8,11]. No match.
    #
    # Try portals at F1.s0 and F1.s2:
    # portal->F2[T0,T1,T2]=[6,14,8], F1.s1->T3=8,
    # portal->F2[T4,T5,T6]=[14,8,8], F1.s3->T7=11, F1.s4->T8=15
    # [6,14,8] ≠ [14,8,8]. No.
    #
    # Try portals at F1.s0 and F1.s3:
    # portal->F2[T0,T1,T2]=[6,14,8], F1.s1->T3=8, F1.s2->T4=14,
    # portal->F2[T5,T6,T7]=[8,8,11], F1.s4->T8=15
    # [6,14,8] ≠ [8,8,11]. No.
    #
    # Try portals at F1.s0 and F1.s4:
    # portal->F2[T0,T1,T2]=[6,14,8], F1.s1->T3=8, F1.s2->T4=14, F1.s3->T5=8,
    # portal->F2[T6,T7,T8]=[8,11,15]
    # [6,14,8] ≠ [8,11,15]. No.
    #
    # Try portals at F1.s1 and F1.s4:
    # F1.s0->T0=6, portal->F2[T1,T2,T3]=[14,8,8], F1.s2->T4=14, F1.s3->T5=8,
    # portal->F2[T6,T7,T8]=[8,11,15]
    # [14,8,8] ≠ [8,11,15]. No.
    #
    # Try portals at F1.s2 and F1.s4:
    # F1.s0->T0=6, F1.s1->T1=14, portal->F2[T2,T3,T4]=[8,8,14],
    # F1.s3->T5=8, portal->F2[T6,T7,T8]=[8,11,15]
    # [8,8,14] ≠ [8,11,15]. No.
    #
    # Try portals at F1.s2 and F1.s3:
    # F1.s0->T0=6, F1.s1->T1=14, portal->F2[T2,T3,T4]=[8,8,14],
    # portal->F2[T5,T6,T7]=[8,8,11], F1.s4->T8=15
    # [8,8,14] ≠ [8,8,11]. Close but no.
    #
    # NONE work! Both portals routing to the same frame can't give different sequences.
    # This means I'm fundamentally misunderstanding the L5 mechanic.
    #
    # Let me reconsider: maybe only 1 portal is placed, and one palette item is NOT used.
    # 5 Frame1 slots - 1 portal = 4 regular + 3 Frame2 = 7. But 9 targets.
    # Unless... some items are already placed in frames? No, all at y=56.
    #
    # OR... maybe L5 doesn't actually have 9 targets. Let me recount.
    # wcfyiodrx = quhhhthrri sorted by (y,x). But maybe the game has a different
    # number of targets that get CHECKED.
    # The win check: pmygakdvy == len(self.wcfyiodrx) - 1
    # So it checks ALL quhhhthrri sprites. In L5 that's 9.
    #
    # Unless... the walk can revisit frame items? Or maybe the portal doesn't enter
    # the frame when the frame has no more unvisited slots?
    #
    # Wait, let me look at the walk MORE carefully.
    # dbfxrigdqx has:
    # if self.ppsxsxiod or ldwfvtgapk.name == "lngftsryyw":
    #     ...advance to next slot...
    #     uxncrzlau += 1
    #     if uxncrzlau < int(kmsegkpkh.name[-1]):
    #         ...animate to next slot...
    #     elif len(self.buvfjfmpp) > 1:
    #         ...pop frame...
    #     else:
    #         ...last slot of root frame...
    #
    # So after each item, uxncrzlau (slot index) increments. When it equals n_slots,
    # the frame is done and we pop back.
    # WHEN POPPING BACK, the parent frame's slot index was already saved.
    # ppsxsxiod is set to True when popping. And the next call to dbfxrigdqx:
    # "if self.ppsxsxiod:"
    #     ppsxsxiod = False
    #     remove last portal marker
    #     uxncrzlau += 1  -- advance PAST the portal in parent frame
    #     ...continue...
    #
    # So the portal slot IS consumed (uxncrzlau advances past it), and we continue
    # at the next slot in the parent frame.
    #
    # For L5, Frame1 has 5 slots. Frame2 has 3 slots.
    # With 1 portal: 4 + 3 = 7 items. Need 9. Doesn't work.
    # With 2 portals: 3 + 3 + 3 = 9 items. But Frame2 items must be the same both times.
    #
    # Let me try: what if the items in Frame2 DO work out?
    # I need Frame2 items [a,b,c] to appear twice in the target sequence.
    # Target: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # Looking for a 3-element subsequence that appears at two positions:
    # With 2 portals at positions i and j, the sequence splits into:
    # F1[0..i-1], F2[a,b,c], F1[i+1..j-1], F2[a,b,c], F1[j+1..4]
    # Sizes: i + 3 + (j-i-1) + 3 + (4-j) = 9 items
    # i + j - i - 1 + 4 - j + 6 = 9 -> 9 = 9. ✓ for any i,j.
    #
    # The constraint is: T[i..i+2] = T[j+3..j+5]? No, let me be more careful.
    # F1.s0->T0, ..., F1.s(i-1)->T(i-1),
    # portal -> F2.s0->T(i), F2.s1->T(i+1), F2.s2->T(i+2),
    # F1.s(i+1)->T(i+3), ..., F1.s(j-1)->T(j+2),
    # portal -> F2.s0->T(j+3), F2.s1->T(j+4), F2.s2->T(j+5),
    # F1.s(j+1)->T(j+6), ..., F1.s4->T(8)
    #
    # Constraint: T[i]=T[j+3], T[i+1]=T[j+4], T[i+2]=T[j+5]
    # i.e., the 3-element subsequences starting at position i and j+3 must match.
    #
    # Target: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    #          0   1  2  3   4  5  6   7   8
    #
    # For portals at s0(i=0) and s1(j=1):
    # F2 enters at T[0..2]=[6,14,8] and T[4..6]=[14,8,8]. No match.
    #
    # Let me just check all pairs:
    # (i=0,j=1): T[0..2]=[6,14,8], T[4..6]=[14,8,8]. No.
    # (i=0,j=2): T[0..2]=[6,14,8], T[5..7]=[8,8,11]. No.
    # (i=0,j=3): T[0..2]=[6,14,8], T[6..8]=[8,11,15]. No.
    # (i=0,j=4): T[0..2]=[6,14,8], T[7..9] out of bounds.
    # (i=1,j=2): T[1..3]=[14,8,8], T[5..7]=[8,8,11]. No.
    # (i=1,j=3): T[1..3]=[14,8,8], T[6..8]=[8,11,15]. No.
    # (i=1,j=4): T[1..3]=[14,8,8], T[7..9] out of bounds.
    # (i=2,j=3): T[2..4]=[8,8,14], T[6..8]=[8,11,15]. No.
    # (i=2,j=4): T[2..4]=[8,8,14], T[7..9] out of bounds.
    # (i=3,j=4): T[3..5]=[8,14,8], T[7..9] out of bounds.
    #
    # NONE match! Two portals to the same Frame2 is IMPOSSIBLE for L5.
    #
    # There's something fundamentally different about L5 that I'm missing.
    # Let me just use the MCP to observe L5 visually and figure it out interactively.

    print("\nExecuting L1+L2...")
    # Already done above. Let me check observation
    obs_frame = session.observe()
    print(f"Current state: levels={obs_frame['levels_completed']}")

    return session, all_actions, level_actions


def run_actions(session, actions):
    """Execute a list of encoded actions."""
    for action_id in actions:
        if action_id >= 7:
            x = (action_id - 7) % 64
            y = (action_id - 7) // 64
            result = session.act(6, x=x, y=y)
        elif action_id == 4:
            result = session.act(5)  # ACTION5
        elif action_id == 6:
            result = session.act(7)  # ACTION7
        else:
            result = session.act(action_id + 1)
    return session.observe()


if __name__ == "__main__":
    solve_sb26()
