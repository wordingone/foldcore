"""
SB26 Full Chain Solver — ALL 8 levels solved analytically from source code.

SB26 is a color-sorting puzzle:
- Target sequence (quhhhthrri sprites sorted by y,x) = colors to match
- Frames (pkpgflvjel tagged) contain slots (susublrply) and can hold portals (vgszefyyyp)
- Palette items (lngftsryyw at y>53) are placed into slots by clicking item then slot
- Portals route evaluation into sub-frames matching the portal's inner color
- ACTION5 (submit) checks: walks frames depth-first following portals, matching slot colors to target sequence
- WIN = all target colors matched in walk order

Action encoding: click(x,y) = 7 + y*64 + x, ACTION5=4
Click offset: sprite position + 2 for both palette items and slots (hit the inner area)
"""

import json
import sys
import os

os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, "B:/M/the-search")

def click(x, y):
    """Encode a click at sprite position (x,y) using +2 offset."""
    return 7 + (y + 2) * 64 + (x + 2)

SUBMIT = 4

# ============================================================
# LEVEL DATA — parsed from sb26.py source code
# ============================================================
# For each level:
#   targets: quhhhthrri sprites sorted by (y, x), extract color
#   frames: pkpgflvjel sprites sorted by (y, x), name[-1] = slot count, color from pixels[0,0] or remap
#   palette: lngftsryyw sprites at y>53 (clickable), with their color
#   portals: vgszefyyyp sprites (can be in palette or pre-placed in frames)
#   pre_placed: lngftsryyw sprites at y<=53 (already in frames, not clickable)
#   slots: susublrply sprites (empty positions in frames)

# Frame type -> slot count (from sprite name last char)
# jvkvqzheok1=1, jvkvqzheok2=2, pcrvmjfjzg3=3, qdmvvkvhaz4=4,
# nyqgqtujsa5=5, wbkmnqvtxh6=6, zzssdzqbbr7=7

EVRMZYFOPO = 53  # energy bar y position (items below this = palette)
KOJDUUMCAP = 6   # slot spacing

def frame_slot_positions(frame_x, frame_y, n_slots):
    """Compute slot positions within a frame."""
    return [(frame_x + 2 + i * KOJDUUMCAP, frame_y + 2) for i in range(n_slots)]

def solve_level(level_num, targets, frames, palette_items, portal_items, pre_placed_items, walk_order):
    """
    Given the walk order (list of (slot_x, slot_y, needed_color, source_type)),
    generate the action sequence.

    walk_order: list of tuples:
      - ('place', palette_pos, slot_pos) for regular items
      - ('portal', portal_pos, slot_pos) for portal items
      - ('skip',) for pre-placed items (no action needed)
    """
    actions = []
    for instruction in walk_order:
        if instruction[0] == 'skip':
            continue
        elif instruction[0] in ('place', 'portal'):
            src_x, src_y = instruction[1]
            dst_x, dst_y = instruction[2]
            actions.append(click(src_x, src_y))
            actions.append(click(dst_x, dst_y))
    actions.append(SUBMIT)
    return actions


def solve_all():
    """Solve all 8 levels of SB26."""

    results = {}
    all_actions = []

    # ==================== LEVEL 1 ====================
    # Targets (y=1, sorted by x): x=18:c9, x=25:c14, x=32:c11, x=39:c15
    # Sequence: [9, 14, 11, 15]
    # Frame: qdmvvkvhaz4 at (18,25), 4 slots at (20,27),(26,27),(32,27),(38,27)
    # Palette: c14@(17,56), c15@(25,56), c9@(33,56), c11@(41,56)
    # No portals
    # Walk: slot0=c9, slot1=c14, slot2=c11, slot3=c15

    l1 = []
    l1.append(click(33, 56)); l1.append(click(20, 27))  # c9 -> slot0
    l1.append(click(17, 56)); l1.append(click(26, 27))  # c14 -> slot1
    l1.append(click(41, 56)); l1.append(click(32, 27))  # c11 -> slot2
    l1.append(click(25, 56)); l1.append(click(38, 27))  # c15 -> slot3
    l1.append(SUBMIT)

    print(f"L1: {len(l1)} actions")
    results["L1"] = {"actions": l1, "n_actions": len(l1)}
    all_actions.extend(l1)

    # ==================== LEVEL 2 ====================
    # Targets (y=1, sorted by x): x=8:c12, x=15:c15, x=22:c8, x=29:c9, x=36:c14, x=43:c11, x=50:c6
    # Sequence: [12, 15, 8, 9, 14, 11, 6]
    # Frames sorted by (y,x):
    #   F1: qdmvvkvhaz4 at (18,18), c8, 4 slots at (20,20),(26,20),(32,20),(38,20)
    #   F2: qdmvvkvhaz4 at (18,32), c14, 4 slots at (20,34),(26,34),(32,34),(38,34)
    # Portal: vgszefyyyp c14 at (32,20) -> routes to F2 (color 14)
    # Palette: c8@(8,56), c15@(15,56), c14@(22,56), c12@(29,56), c6@(36,56), c9@(43,56), c11@(50,56)
    # Slots: (20,20),(26,20),(38,20) in F1; (20,34),(26,34),(32,34),(38,34) in F2
    #
    # Walk: F1.s0(20,20)->c12, F1.s1(26,20)->c15, F1.s2=PORTAL(14)->F2,
    #   F2.s0(20,34)->c8, F2.s1(26,34)->c9, F2.s2(32,34)->c14, F2.s3(38,34)->c11,
    #   back, F1.s3(38,20)->c6

    l2 = []
    l2.append(click(29, 56)); l2.append(click(20, 20))  # c12 -> F1.s0
    l2.append(click(15, 56)); l2.append(click(26, 20))  # c15 -> F1.s1
    l2.append(click(8, 56));  l2.append(click(20, 34))  # c8  -> F2.s0
    l2.append(click(43, 56)); l2.append(click(26, 34))  # c9  -> F2.s1
    l2.append(click(22, 56)); l2.append(click(32, 34))  # c14 -> F2.s2
    l2.append(click(50, 56)); l2.append(click(38, 34))  # c11 -> F2.s3
    l2.append(click(36, 56)); l2.append(click(38, 20))  # c6  -> F1.s3
    l2.append(SUBMIT)

    print(f"L2: {len(l2)} actions")
    results["L2"] = {"actions": l2, "n_actions": len(l2)}
    all_actions.extend(l2)

    # ==================== LEVEL 3 ====================
    # Targets (y=1, sorted by x): x=8:c8, x=15:c14, x=22:c15, x=29:c11, x=36:c6, x=43:c9, x=50:c12
    # Note: quhhhthrri sorted by (y,x) — all at y=1
    # From source lines 438-444:
    #   (15,1)c14, (22,1)c15, (29,1)c11, (36,1)c6, (43,1)c9, (8,1)c8, (50,1)c12
    # Sorted by (y,x): (8,1)c8, (15,1)c14, (22,1)c15, (29,1)c11, (36,1)c6, (43,1)c9, (50,1)c12
    # Sequence: [8, 14, 15, 11, 6, 9, 12]
    #
    # Frames sorted by (y,x):
    #   F1: nyqgqtujsa5 at (15,19), c8, 5 slots at (17,21),(23,21),(29,21),(35,21),(41,21)
    #   F2: jvkvqzheok2 at (15,31), c14, 2 slots at (17,33),(23,33)
    #   F3: jvkvqzheok2 at (33,31), c9, 2 slots at (35,33),(41,33)
    #
    # Items at F1 slot positions:
    #   (17,21)=susublrply, (23,21)=vgszefyyyp c14, (29,21)=susublrply,
    #   (35,21)=vgszefyyyp c9, (41,21)=susublrply
    # Portals at (23,21) c14 and (35,21) c9 are PRE-PLACED in the frame, not on palette
    #
    # Walk: F1.s0(17,21)->c8, F1.s1=PORTAL(14)->F2,
    #   F2.s0(17,33)->c14, F2.s1(23,33)->c15, back,
    #   F1.s2(29,21)->c11, F1.s3=PORTAL(9)->F3,
    #   F3.s0(35,33)->c6, F3.s1(41,33)->c9, back,
    #   F1.s4(41,21)->c12
    #
    # Palette: c8@(50,56), c15@(22,56), c14@(15,56), c12@(8,56), c6@(43,56), c9@(36,56), c11@(29,56)

    l3 = []
    l3.append(click(50, 56)); l3.append(click(17, 21))  # c8  -> F1.s0
    l3.append(click(15, 56)); l3.append(click(17, 33))  # c14 -> F2.s0
    l3.append(click(22, 56)); l3.append(click(23, 33))  # c15 -> F2.s1
    l3.append(click(29, 56)); l3.append(click(29, 21))  # c11 -> F1.s2
    l3.append(click(43, 56)); l3.append(click(35, 33))  # c6  -> F3.s0
    l3.append(click(36, 56)); l3.append(click(41, 33))  # c9  -> F3.s1
    l3.append(click(8, 56));  l3.append(click(41, 21))  # c12 -> F1.s4
    l3.append(SUBMIT)
    print(f"L3: {len(l3)} actions")
    results["L3"] = {"actions": l3, "n_actions": len(l3)}
    all_actions.extend(l3)

    # ==================== LEVEL 4 ====================
    # Targets (y=1, sorted by x):
    # From source lines 477-483:
    #   (8,1)c11, (15,1)c8, (22,1)c14, (29,1)c9, (36,1)c6, (43,1)c12, (50,1)c15
    # Sequence: [11, 8, 14, 9, 6, 12, 15]
    #
    # Frames sorted by (y,x):
    #   F1: nyqgqtujsa5 at (15,18), c8, 5 slots at (17,20),(23,20),(29,20),(35,20),(41,20)
    #   F2: pcrvmjfjzg3 at (21,32), c14, 3 slots at (23,34),(29,34),(35,34)
    #
    # Portal on palette: vgszefyyyp c14 at (50,56)
    # Pre-placed item: lngftsryyw c14 at (23,34) — already in F2.s0
    #
    # The portal must go into F1 to route to F2. Where in the walk should it go?
    # Target[2] = 14, which is the first F2 item (c14 already at F2.s0).
    # So portal goes at F1.s2 (29,20).
    #
    # Walk: F1.s0(17,20)->c11, F1.s1(23,20)->c8, F1.s2=PORTAL(14)->F2,
    #   F2.s0(23,34)->c14(pre-placed), F2.s1(29,34)->c9, F2.s2(35,34)->c6, back,
    #   F1.s3(35,20)->c12, F1.s4(41,20)->c15
    #
    # Palette: c8@(29,56), c6@(15,56), c15@(36,56), c12@(22,56), c11@(8,56), c9@(43,56)
    # Portal: c14@(50,56)

    l4 = []
    l4.append(click(8, 56));  l4.append(click(17, 20))  # c11 -> F1.s0
    l4.append(click(29, 56)); l4.append(click(23, 20))  # c8  -> F1.s1
    l4.append(click(50, 56)); l4.append(click(29, 20))  # portal(14) -> F1.s2
    l4.append(click(43, 56)); l4.append(click(29, 34))  # c9  -> F2.s1
    l4.append(click(15, 56)); l4.append(click(35, 34))  # c6  -> F2.s2
    l4.append(click(22, 56)); l4.append(click(35, 20))  # c12 -> F1.s3
    l4.append(click(36, 56)); l4.append(click(41, 20))  # c15 -> F1.s4
    l4.append(SUBMIT)
    print(f"L4: {len(l4)} actions")
    results["L4"] = {"actions": l4, "n_actions": len(l4)}
    all_actions.extend(l4)

    # ==================== LEVEL 5 ====================
    # Targets (y=1, sorted by x):
    # From source lines 514-522:
    #   (1,1)c6, (8,1)c14, (15,1)c8, (22,1)c8, (29,1)c14, (36,1)c8, (43,1)c8, (50,1)c11, (57,1)c15
    # Sequence: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # 9 targets!
    #
    # Frames sorted by (y,x):
    #   F1: nyqgqtujsa5 at (15,18), c8, 5 slots at (17,20),(23,20),(29,20),(35,20),(41,20)
    #   F2: pcrvmjfjzg3 at (21,32), c9, 3 slots at (23,34),(29,34),(35,34)
    #
    # Portals on palette: vgszefyyyp c9@(46,56), vgszefyyyp c9@(53,56)
    # Palette: c14@(39,56), c8@(25,56), c11@(32,56), c8@(18,56), c6@(11,56), c15@(4,56)
    #
    # Both portals route to F2 (color 9). Need to place both in F1.
    # 3 regular F1 slots + 3 F2 visit1 + 3 F2 visit2 = 9 targets.
    # F2 items are the same both visits, so T[visit1] must equal T[visit2].
    #
    # Portals at slot1 and slot2:
    #   s0->T[0]=6, PORTAL->F2[T[1]=14,T[2]=8,T[3]=8],
    #   PORTAL->F2[T[4]=14,T[5]=8,T[6]=8],
    #   s3->T[7]=11, s4->T[8]=15
    # F2 visit1: [14,8,8] == F2 visit2: [14,8,8] ✓ MATCH!
    #
    # F2 items: slot0=c14, slot1=c8, slot2=c8
    # F1 regular: slot0=c6, slot3=c11, slot4=c15

    l5 = []
    l5.append(click(11, 56)); l5.append(click(17, 20))  # c6  -> F1.s0
    l5.append(click(46, 56)); l5.append(click(23, 20))  # portal(9) -> F1.s1
    l5.append(click(53, 56)); l5.append(click(29, 20))  # portal(9) -> F1.s2
    l5.append(click(39, 56)); l5.append(click(23, 34))  # c14 -> F2.s0
    l5.append(click(25, 56)); l5.append(click(29, 34))  # c8  -> F2.s1
    l5.append(click(18, 56)); l5.append(click(35, 34))  # c8  -> F2.s2
    l5.append(click(32, 56)); l5.append(click(35, 20))  # c11 -> F1.s3
    l5.append(click(4, 56));  l5.append(click(41, 20))  # c15 -> F1.s4
    l5.append(SUBMIT)
    print(f"L5: {len(l5)} actions")
    results["L5"] = {"actions": l5, "n_actions": len(l5)}
    all_actions.extend(l5)

    # ==================== LEVEL 6 ====================
    # Targets (y=1, sorted by x):
    # From source lines 562-570:
    #   (1,1)c9, (8,1)c11, (15,1)c11, (22,1)c12, (29,1)c15, (36,1)c15, (43,1)c14, (50,1)c6, (57,1)c6
    # Sequence: [9, 11, 11, 12, 15, 15, 14, 6, 6]
    # 9 targets!
    #
    # Frames sorted by (y,x):
    # From source lines 558-561:
    #   F1: pcrvmjfjzg3 at (8,18), c8, 3 slots at (10,20),(16,20),(22,20)
    #   F2: pcrvmjfjzg3 at (34,18), c14, 3 slots at (36,20),(42,20),(48,20)
    #   F3: pcrvmjfjzg3 at (8,32), c9, 3 slots at (10,34),(16,34),(22,34)
    #   F4: pcrvmjfjzg3 at (34,32), c12, 3 slots at (36,34),(42,34),(48,34)
    #
    # Items in frames (lngftsryyw NOT on palette):
    #   c14 at (36,20) — in F2.s0 (y=20 <= 53)
    #   c9  at (10,34) — in F3.s0 (y=34 <= 53)
    #   c12 at (36,34) — in F4.s0 (y=34 <= 53)
    #
    # Portals on palette (at y=56):
    #   vgszefyyyp c14 at (43,56)
    #   vgszefyyyp c9  at (50,56)
    #   vgszefyyyp c12 at (57,56)
    #
    # Palette items (lngftsryyw at y=56):
    #   c15@(36,56), c6@(29,56), c11@(22,56), c15@(15,56), c6@(8,56), c11@(1,56)
    #
    # Slots (susublrply):
    #   (10,20),(16,20),(22,20): F1 slots
    #   (48,20): F2.s2 (F2 has 3 slots, s0=pre-placed c14, s1=?, s2=?)
    #   (22,34): F3.s2
    #   (48,34): F4.s2
    #   (42,20): F2.s1
    #   (42,34): F4.s1
    #   (16,34): F3.s1
    #
    # F1 has 3 slots: (10,20),(16,20),(22,20) — all plain susublrply
    # F2 has 3 slots: (36,20)=pre-placed c14, (42,20)=slot, (48,20)=slot
    # F3 has 3 slots: (10,34)=pre-placed c9, (16,34)=slot, (22,34)=slot
    # F4 has 3 slots: (36,34)=pre-placed c12, (42,34)=slot, (48,34)=slot
    #
    # The 3 portals must go into F1's 3 slots (which are the only plain slots in the main frame).
    # Portal c14 -> F2, Portal c9 -> F3, Portal c12 -> F4
    #
    # Walk through F1 with portals:
    # F1.s0=PORTAL(c14)->F2: F2.s0=c14(pre), F2.s1=?, F2.s2=?
    # F1.s1=PORTAL(c9)->F3: F3.s0=c9(pre), F3.s1=?, F3.s2=?
    # F1.s2=PORTAL(c12)->F4: F4.s0=c12(pre), F4.s1=?, F4.s2=?
    #
    # Target sequence: [9, 11, 11, 12, 15, 15, 14, 6, 6]
    # BUT wait — portals don't consume targets. So:
    # F1.s0=PORTAL->F2[T0,T1,T2]=9,11,11 → F2 needs [c9,c11,c11]?
    # No! F2 pre-placed = c14, which must match T0=9? 14≠9. WRONG.
    #
    # Let me reconsider portal placement order. Maybe portals go at specific F1 slots.
    # The walk always starts at F1's first slot. Each F1 slot either routes via portal or is a regular item.
    #
    # Since F1 has ONLY 3 slots and ALL need portals (the other frames have pre-placed items),
    # the portals MUST go at F1.s0, F1.s1, F1.s2.
    # But which portal where?
    #
    # Try: F1.s0=portal(c9)->F3, F1.s1=portal(c14)->F2, F1.s2=portal(c12)->F4
    # Walk: F3[T0,T1,T2]=[9,11,11], F2[T3,T4,T5]=[12,15,15], F4[T6,T7,T8]=[14,6,6]
    # F3: s0=c9(pre) must match T0=9 ✓, s1=c11, s2=c11
    # F2: s0=c14(pre) must match T3=12? 14≠12. NO.
    #
    # Try: F1.s0=portal(c9)->F3, F1.s1=portal(c12)->F4, F1.s2=portal(c14)->F2
    # Walk: F3[9,11,11], F4[12,15,15], F2[14,6,6]
    # F3: s0=c9 match T0=9 ✓, s1=c11, s2=c11 ✓
    # F4: s0=c12 match T3=12 ✓, s1=c15, s2=c15 ✓
    # F2: s0=c14 match T6=14 ✓, s1=c6, s2=c6 ✓
    # ALL MATCH!
    #
    # Placements:
    # F1.s0 (10,20) <- portal(c9) from (50,56)
    # F1.s1 (16,20) <- portal(c12) from (57,56)
    # F1.s2 (22,20) <- portal(c14) from (43,56)
    # F3.s1 (16,34) <- c11 from palette
    # F3.s2 (22,34) <- c11 from palette
    # F4.s1 (42,34) <- c15 from palette
    # F4.s2 (48,34) <- c15 from palette
    # F2.s1 (42,20) <- c6 from palette
    # F2.s2 (48,20) <- c6 from palette
    #
    # Palette: c15@(36,56), c6@(29,56), c11@(22,56), c15@(15,56), c6@(8,56), c11@(1,56)

    l6 = []
    l6.append(click(50, 56)); l6.append(click(10, 20))  # portal(c9)  -> F1.s0
    l6.append(click(57, 56)); l6.append(click(16, 20))  # portal(c12) -> F1.s1
    l6.append(click(43, 56)); l6.append(click(22, 20))  # portal(c14) -> F1.s2
    l6.append(click(22, 56)); l6.append(click(16, 34))  # c11 -> F3.s1
    l6.append(click(1, 56));  l6.append(click(22, 34))  # c11 -> F3.s2
    l6.append(click(36, 56)); l6.append(click(42, 34))  # c15 -> F4.s1
    l6.append(click(15, 56)); l6.append(click(48, 34))  # c15 -> F4.s2
    l6.append(click(29, 56)); l6.append(click(42, 20))  # c6  -> F2.s1
    l6.append(click(8, 56));  l6.append(click(48, 20))  # c6  -> F2.s2
    l6.append(SUBMIT)
    print(f"L6: {len(l6)} actions")
    results["L6"] = {"actions": l6, "n_actions": len(l6)}
    all_actions.extend(l6)

    # ==================== LEVEL 7 ====================
    # Targets (y=1, sorted by x):
    # From source lines 609-615:
    #   (8,1)c8, (15,1)c9, (22,1)c14, (29,1)c11, (36,1)c14, (43,1)c9, (50,1)c8
    # Sequence: [8, 9, 14, 11, 14, 9, 8]
    # 7 targets
    #
    # Frames sorted by (y,x):
    # From source lines 606-608:
    #   F1: pcrvmjfjzg3 at (21,12), c8, 3 slots at (23,14),(29,14),(35,14)
    #   F2: pcrvmjfjzg3 at (21,25), c9, 3 slots at (23,27),(29,27),(35,27)
    #   F3: pcrvmjfjzg3 at (21,38), c14, 3 slots at (23,40),(29,40),(35,40)
    #
    # Items in frames:
    #   lngftsryyw c11 at (29,40) — in F3 at slot1 position (y=40 <= 53)
    # Portals on palette:
    #   vgszefyyyp c14 at (46,56)
    #   vgszefyyyp c9  at (53,56)
    #
    # Palette items:
    #   c14@(39,56), c14@(25,56) — wait, re-check:
    #   lngftsryyw at: (39,56)c14, (25,56)c14? No, let me re-read source.
    #
    # From source lines 599-605:
    #   lngftsryyw at (39,56) c14 — palette
    #   lngftsryyw at (25,56) c14 — palette
    #   lngftsryyw at (18,56) c8  — palette
    #   lngftsryyw at (29,40) c11 — in frame! (y=40 <= 53)
    #   lngftsryyw at (11,56) c9  — palette
    #   lngftsryyw at (4,56)  c8  — palette
    #   lngftsryyw at (32,56) c9  — palette
    #
    # So palette items: c14@(39,56), c14@(25,56), c8@(18,56), c9@(11,56), c8@(4,56), c9@(32,56)
    # Pre-placed: c11@(29,40) in F3.s1
    # Portals: c14@(46,56), c9@(53,56)
    #
    # Slots:
    #   (23,14),(29,14),(35,14) — F1
    #   (23,27),(29,27),(35,27) — F2
    #   (23,40),(35,40) — F3 (s0 and s2; s1 has c11 pre-placed)
    #   Also (29,40) position has the lngftsryyw c11
    #
    # F1 (c8, 3 slots): all empty
    # F2 (c9, 3 slots): all empty
    # F3 (c14, 3 slots): s0=(23,40) empty, s1=(29,40) c11 pre-placed, s2=(35,40) empty
    #
    # Portals must route to sub-frames. Portal c9 routes to F2, portal c14 routes to F3.
    # We need to place portals somewhere in F1 (the first frame in the walk).
    #
    # Target: [8, 9, 14, 11, 14, 9, 8]
    # 7 targets, F1 has 3 slots. Two portals + 1 regular item in F1?
    # 1 regular F1 + 3 F2 + 3 F3 = 7.
    # BUT portal c9 and c14 are BOTH on the palette (at y=56).
    # Need to place them in F1 slots.
    #
    # Try portals at F1.s0 and F1.s1:
    # F1.s0=PORTAL(c9)->F2[T0,T1,T2], F1.s1=PORTAL(c14)->F3[T3,T4,T5], F1.s2->T6
    # F2: T[0..2]=[8,9,14] need items c8, c9, c14
    # F3: T[3..5]=[11,14,9] need items c11, c14, c9
    # F3.s1 is pre-placed c11. T3=11 matches slot1? No — slot0 maps to T3, slot1 to T4, slot2 to T5.
    # F3 walk: s0->T3=11, s1->T4=14, s2->T5=9
    # F3.s0 needs c11 but s1 is pre-placed c11. Wrong slot!
    # F3.s1(pre-placed c11) would match T4=14? No, 11≠14.
    #
    # Try portals at F1.s0 and F1.s2:
    # F1.s0=PORTAL(c9)->F2[T0,T1,T2]=[8,9,14], F1.s1->T3=11, F1.s2=PORTAL(c14)->F3[T4,T5,T6]=[14,9,8]
    # F2 needs: s0=c8, s1=c9, s2=c14
    # F3 needs: s0=c14, s1=c9, s2=c8. BUT F3.s1=c11(pre-placed). c11≠c9. NO.
    #
    # Try portals at F1.s1 and F1.s2:
    # F1.s0->T0=8, F1.s1=PORTAL(c9)->F2[T1,T2,T3]=[9,14,11], F1.s2=PORTAL(c14)->F3[T4,T5,T6]=[14,9,8]
    # F2 needs: s0=c9, s1=c14, s2=c11
    # F3 needs: s0=c14, s1=c9, s2=c8. F3.s1=c11(pre-placed). c11≠c9. NO.
    #
    # Try portals at F1.s1 and F1.s0, switching which portal where:
    # F1.s0=PORTAL(c14)->F3[T0,T1,T2]=[8,9,14], F1.s1=PORTAL(c9)->F2[T3,T4,T5]=[11,14,9], F1.s2->T6=8
    # F3 needs: s0=c8, s1=c9, s2=c14. F3.s1=c11(pre-placed). c11≠c9. NO.
    #
    # F1.s0=PORTAL(c14)->F3[T0,T1,T2], F1.s1->T3, F1.s2=PORTAL(c9)->F2[T4,T5,T6]
    # F3: T[0..2]=[8,9,14]. s0=c8, s1=c9? But s1=c11(pre-placed). 11≠9. NO.
    #
    # Hmm. The pre-placed c11 at F3.s1 is always a problem.
    # F3 walk always goes s0, s1, s2. s1=c11 is fixed.
    # So we need targets at F3 positions to have T[F3_start+1] = 11.
    #
    # With portals at different F1 positions, F3's targets are:
    # If portal to F3 at F1.s0: F3 gets T[0..2]=[8,9,14]. T[1]=9≠11. NO.
    # If portal to F3 at F1.s1: F3 gets T[1+offset..]. Let me think about offsets more carefully.
    #
    # Portal to F3 at F1.s0 (and portal to F2 at F1.s1):
    #   F3[T0,T1,T2], F2[T3,T4,T5], s2=T6
    #   F3 needs T1=11? T1=9. NO.
    #
    # Portal to F3 at F1.s0 (and portal to F2 at F1.s2):
    #   F3[T0,T1,T2], s1=T3, F2[T4,T5,T6]
    #   F3 needs T1=11? T1=9. NO.
    #
    # Portal to F3 at F1.s1 (and portal to F2 at F1.s0):
    #   F2[T0,T1,T2], F3[T3,T4,T5], s2=T6
    #   F3 needs T4=11? T4=14. NO.
    #
    # Portal to F3 at F1.s1 (and portal to F2 at F1.s2):
    #   s0=T0, F3[T1,T2,T3], F2[T4,T5,T6]
    #   F3 needs T2=11? T2=14. NO.
    #
    # Portal to F3 at F1.s2 (and portal to F2 at F1.s0):
    #   F2[T0,T1,T2], s1=T3, F3[T4,T5,T6]
    #   F3 needs T5=11? T5=9. NO.
    #
    # Portal to F3 at F1.s2 (and portal to F2 at F1.s1):
    #   s0=T0, F2[T1,T2,T3], F3[T4,T5,T6]
    #   F3 needs T5=11? T5=9. NO.
    #
    # NONE work because T has 11 only at position 3, and no portal placement puts
    # F3's second slot (where c11 is) at target position 3 with F3 being a 3-slot frame.
    #
    # Wait — maybe F3 has the c11 at a different slot than I think.
    # F3: pcrvmjfjzg3 at (21,38), c14, 3 slots
    # Slot positions: x = 21+2+i*6 = 23, 29, 35 at y=38+2=40
    # Pre-placed c11 at position (29,40) -> that's slot1 (x=29). ✓
    #
    # Hmm. 11 is at target[3] and F3.s1.
    # For F3.s1 to match target[3], F3 must start at target[2]:
    # F3[T2,T3,T4] = F3[14,11,14] -> s0=c14, s1=c11(pre), s2=c14
    #
    # How to get F3 starting at T2? Need 2 targets consumed before F3.
    # Option: F1.s0->T0=8, F1.s1=portal->F3[T1,T2,T3]=[9,14,11]
    # T1=9, T2=14, T3=11. F3: s0=c9, s1=c14(pre=c11)?? No, s1=c11, and T2=14. 11≠14.
    # That doesn't work either. The issue is that in F3, the walk goes s0,s1,s2 and
    # s1 is always c11. So we need the middle target for F3 to be 11.
    #
    # Portal to F3 starting at target_idx i: F3 maps to T[i], T[i+1], T[i+2]
    # Need T[i+1] = 11. Looking at targets: [8,9,14,11,14,9,8]
    # T[i+1]=11 when i+1=3, so i=2.
    # So F3 must start at target index 2.
    #
    # To have 2 targets before F3, either:
    # a) F1.s0->T0, F1.s1->T1, F1.s2=PORTAL->F3[T2,T3,T4]
    #    But then we need another portal somewhere for F2, and F1 has no more slots.
    #    Unless one of F1's regular slots has a portal... F1 only has 3 slots total.
    #    With portal at s2, s0 and s1 are regular.
    #    But where does F2 portal go? F2 has 3 empty slots.
    #    Wait — maybe F2 also needs a portal?
    #
    # b) F1.s0=PORTAL->F2[T0,T1,T2]=[8,9,14], back to F1, F1.s1=PORTAL->F3[T3,T4,T5]=[11,14,9]
    #    F3.s1 needs T4=14. But s1=c11(pre-placed). 11≠14. NO.
    #
    # c) What if F2 has a portal to F3? NESTED portals!
    #    F1.s0=PORTAL->F2, in F2 there's a portal to F3.
    #    F2.s0->T0, F2.s1=PORTAL->F3[T1,T2,T3]=[9,14,11]
    #    F3.s1=c11 needs T2=14. 11≠14. NO.
    #    F2.s0=PORTAL->F3[T0,T1,T2]=[8,9,14], F2.s1->T3=11, F2.s2->T4=14
    #    F3 needs T1=11? T1=9≠11. NO.
    #
    # d) What if F1 has portal to F2, and F2 has portal to F3 at the right position?
    #    F1.s0->T0=8, F1.s1=PORTAL(c9)->F2[F2 walk], F1.s2->T6=8
    #    F2 walk with portal at s1:
    #    F2.s0->T1=9, F2.s1=PORTAL(c14)->F3[T2,T3,T4]=[14,11,14], F2.s2->T5=9
    #    F3: s0=c14->T2=14 ✓, s1=c11(pre)->T3=11 ✓, s2=c14->T4=14 ✓
    #    F2: s0=c9->T1=9 ✓, s2=c9->T5=9 ✓
    #    F1: s0=c8->T0=8 ✓, s2=c8->T6=8 ✓
    #    ALL MATCH!!!
    #
    # So the solution uses NESTED portals:
    # F1 (c8, 3 slots): s0=c8, s1=portal(c9)->F2, s2=c8
    # F2 (c9, 3 slots): s0=c9, s1=portal(c14)->F3, s2=c9
    # F3 (c14, 3 slots): s0=c14, s1=c11(pre-placed), s2=c14
    #
    # Placements needed:
    # F1.s0 (23,14) <- c8 from (4,56) or (18,56)
    # F1.s1 (29,14) <- portal(c9) from (53,56)
    # F1.s2 (35,14) <- c8 from (18,56) or (4,56)
    # F2.s0 (23,27) <- c9 from (11,56) or (32,56)
    # F2.s1 (29,27) <- portal(c14) from (46,56)
    # F2.s2 (35,27) <- c9 from (32,56) or (11,56)
    # F3.s0 (23,40) <- c14 from (39,56) or (25,56)
    # F3.s1 (29,40) <- c11 (ALREADY PLACED)
    # F3.s2 (35,40) <- c14 from (25,56) or (39,56)

    l7 = []
    l7.append(click(4, 56));  l7.append(click(23, 14))  # c8  -> F1.s0
    l7.append(click(53, 56)); l7.append(click(29, 14))  # portal(c9) -> F1.s1
    l7.append(click(18, 56)); l7.append(click(35, 14))  # c8  -> F1.s2
    l7.append(click(11, 56)); l7.append(click(23, 27))  # c9  -> F2.s0
    l7.append(click(46, 56)); l7.append(click(29, 27))  # portal(c14) -> F2.s1
    l7.append(click(32, 56)); l7.append(click(35, 27))  # c9  -> F2.s2
    l7.append(click(39, 56)); l7.append(click(23, 40))  # c14 -> F3.s0
    # F3.s1 = c11 already placed, skip
    l7.append(click(25, 56)); l7.append(click(35, 40))  # c14 -> F3.s2
    l7.append(SUBMIT)
    print(f"L7: {len(l7)} actions")
    results["L7"] = {"actions": l7, "n_actions": len(l7)}
    all_actions.extend(l7)

    # ==================== LEVEL 8 ====================
    # Targets:
    # From source lines 648-659:
    #   Row 1 (y=1): (11,1)c8, (18,1)c11, (25,1)c12, (32,1)c9, (39,1)c14, (46,1)c15
    #   Row 2 (y=8): (11,8)c8, (18,8)c11, (25,8)c12, (32,8)c9, (39,8)c14, (46,8)c15
    # Sorted by (y,x): first all y=1 by x, then all y=8 by x
    # Sequence: [8, 11, 12, 9, 14, 15, 8, 11, 12, 9, 14, 15]
    # 12 targets!
    #
    # Frames sorted by (y,x):
    #   F1: qdmvvkvhaz4 at (18,22), c8, 4 slots at (20,24),(26,24),(32,24),(38,24)
    #   F2: qdmvvkvhaz4 at (18,36), c9, 4 slots at (20,38),(26,38),(32,38),(38,38)
    #
    # Palette items (y=56):
    #   c14@(39,56), c8@(25,56), c9@(32,56), c12@(18,56), c11@(11,56), c15@(4,56)
    #
    # Portals on palette:
    #   vgszefyyyp c8 at (46,56) — routes to F1 (color 8)? No, F1 IS color 8!
    #   Wait: vgszefyyyp c8 at (46,56) routes to the frame with color 8 = F1.
    #   vgszefyyyp c9 at (53,56) — routes to F2 (color 9)
    #
    # Wait, portals routing to F1 from F1 would be a self-loop? No — the portal could be
    # placed in F2, creating nested evaluation. Let me think.
    #
    # 12 targets, F1 has 4 slots, F2 has 4 slots. Total = 8 slots.
    # Need 12 targets from 8 slots. So some frames must be visited multiple times.
    #
    # Target: [8,11,12,9,14,15, 8,11,12,9,14,15]
    # The pattern repeats! First 6 = last 6.
    # So if F1 is visited twice (via portal) and F2 is visited twice, each with 3 slots...
    # No, each frame has 4 slots.
    #
    # With portals in F1 routing to F2 and portals in F2 routing to F1 (mutual recursion):
    # F1.s0->T0, F1.s1=PORTAL(c9)->F2, F2.s0->T1, F2.s1=PORTAL(c8)->F1???
    # That would be an infinite loop. Let me check the anti-loop mechanism.
    #
    # dbfxrigdqx line 976:
    # if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0)
    # This checks if we're at slot 0 of a frame that's already in the stack and the parent was also at slot 0.
    #
    # Actually, let me re-read this condition:
    # kmsegkpkh = the frame we're about to enter (via portal)
    # uxncrzlau = 0 (always entering at slot 0)
    # Check: (frame, 0) in buvfjfmpp[:-1] AND buvfjfmpp[-2][1] == 0
    # This triggers an error if:
    # 1. The target frame is already in the stack
    # 2. AND the entry above it in the stack was at slot 0
    #
    # For F1->portal->F2->portal->F1:
    # buvfjfmpp = [(F1, 1), (F2, 1)]. Pushing (F1, 0).
    # Check: (F1, 0) in [(F1, 1)]? F1 is the same frame but slot 1, not 0. (F1,0) ≠ (F1,1). So FALSE.
    # So the portal to F1 IS allowed! No self-loop detected.
    #
    # The walk would be:
    # F1.s0, F1.s1=PORTAL->F2, F2.s0, F2.s1=PORTAL->F1, F1.s0, F1.s1, ...(?)
    # Wait, when we re-enter F1 via a portal from F2, we walk F1 from slot 0 again.
    # F1's items haven't changed. So we get the same slots.
    # But then F1.s1 is still a portal, and we'd enter F2 again... infinite loop!
    #
    # Let me check if the anti-loop catches this on the second round.
    # After re-entering F1 and walking to F1.s1 (portal) again:
    # buvfjfmpp = [(F1, 1), (F2, ?), (F1, 1)]
    # About to push (F2, 0).
    # Check: (F2, 0) in buvfjfmpp[:-1]? buvfjfmpp[:-1] = [(F1,1), (F2,?)].
    # If F2 was popped already... hmm, let me trace more carefully.
    #
    # Actually, wait. When we enter F2 via portal and walk through F2, after F2's last slot,
    # we don't necessarily re-enter F1. Let me think about the flow differently.
    #
    # Maybe the answer for L8 is simpler than mutual recursion. Let me look at L8's structure:
    # Target repeats: [8,11,12,9,14,15] twice.
    # F1 (4 slots), F2 (4 slots).
    #
    # What if F1 has 2 portals to F2, and each portal visit reads F2 with 3 items?
    # 2 portals in F1 + 2 regular F1 slots + 3*2 F2 slots = 2+6=8 ≠ 12.
    #
    # What if F1 has 3 portals to F2?
    # 1 regular + 3*3 F2 = 10. Still not 12.
    #
    # What if both F1 and F2 have portals to each other?
    # F1: s0=item, s1=portal(F2), s2=item, s3=portal(F2)
    # F2: s0=item, s1=portal(F1), s2=item, s3=portal(F1)
    # Walk: F1.s0, portal->F2, F2.s0, portal->F1, F1.s0, F1.s1, F1.s2, F1.s3...
    # This gets complicated. Let me think about what items there are.
    #
    # Actually, there are only 6 palette items and 2 portals. That's 8 things for 8 slots.
    # One portal goes in F1, one in F2. Or both in same frame.
    #
    # portal c8 -> routes to F1 (border color 8)
    # portal c9 -> routes to F2 (border color 9)
    #
    # If portal c9 goes in F1 and portal c8 goes in F2:
    # F1 has 3 regular items + 1 portal(F2)
    # F2 has 3 regular items + 1 portal(F1)
    # Walk: depends on where portals are placed.
    #
    # Let me try F1.s1=portal(c9)->F2 and F2.s1=portal(c8)->F1:
    # Stack: [(F1, 0)]
    # F1.s0->T0=8
    # F1.s1=portal->F2. Stack: [(F1, 1), (F2, 0)]
    #   F2.s0->T1=11
    #   F2.s1=portal->F1. Stack: [(F1, 1), (F2, 1), (F1, 0)]
    #     Check: (F1,0) in [(F1,1),(F2,1)]? No — (F1,0) ≠ (F1,1) and ≠ (F2,1).
    #     AND buvfjfmpp[-2][1] == 0? buvfjfmpp[-2] = (F2,1), [1]=1 ≠ 0. FALSE.
    #     Portal enters F1 again!
    #     F1.s0->T2=12
    #     F1.s1=portal->F2. Stack: [(F1,1),(F2,1),(F1,1),(F2,0)]
    #       Check: (F2,0) in [(F1,1),(F2,1),(F1,1)]? Yes! (F2,1) has same frame.
    #       Wait — (F2,0) vs (F2,1): same frame but different slot. So FALSE.
    #       Hmm, the check is: (kmsegkpkh, uxncrzlau) in buvfjfmpp[:-1]
    #       Looking for EXACT tuple (F2, 0). In stack: (F2,1). (F2,0) ≠ (F2,1). FALSE.
    #       AND buvfjfmpp[-2][1] == 0? buvfjfmpp[-2]=(F1,1), [1]=1≠0. FALSE.
    #       Portal enters F2 again!
    #       This is infinite recursion!
    #
    # Hmm, that can't be right. The game must prevent this somehow.
    # Let me re-read the anti-loop check:
    # Line 976: if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0)
    # uxncrzlau here is not the entry slot — it's the current slot in the CALLING frame.
    # Wait no, let me re-read. At the point where this code runs:
    # kmsegkpkh, uxncrzlau = self.buvfjfmpp[-1]  # current frame and slot
    # ldwfvtgapk = self.rzbeqaiky[kmsegkpkh][uxncrzlau]  # item at current slot
    # The item is a portal (vgszefyyyp).
    # Then: if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1]
    # This checks if current position (current_frame, 0) is already in the stack.
    # It's checking if the CURRENT frame at slot 0 is already a parent in the recursion.
    # And the second condition: buvfjfmpp[-2][1] == 0 — the parent's slot is also 0.
    #
    # So this only triggers for specific slot-0 situations. It won't catch deeper recursion.
    #
    # Maybe there IS no infinite loop because the anti-loop check catches it in practice.
    # Let me trace more carefully for F1.s1=portal(c9) and F2.s1=portal(c8):
    #
    # Step 1: Walk starts. buvfjfmpp = [(F1, 0)]
    # F1.s0 -> check item. Regular item. T0 matched. Advance to s1.
    # buvfjfmpp = [(F1, 1)]
    # Step 2: F1.s1 = portal. uxncrzlau = 1, so uxncrzlau==0 is FALSE. No anti-loop.
    # Enter F2. buvfjfmpp = [(F1, 1), (F2, 0)]
    # Step 3: F2.s0 -> regular. T1 matched. Advance to s1.
    # buvfjfmpp = [(F1, 1), (F2, 1)]
    # Step 4: F2.s1 = portal. uxncrzlau = 1, so uxncrzlau==0 is FALSE. No anti-loop.
    # Enter F1. buvfjfmpp = [(F1, 1), (F2, 1), (F1, 0)]
    # Step 5: F1.s0 -> regular. T2 matched. Advance to s1.
    # buvfjfmpp = [(F1, 1), (F2, 1), (F1, 1)]
    # Step 6: F1.s1 = portal. uxncrzlau = 1. uxncrzlau==0 is FALSE. No anti-loop.
    # Enter F2. buvfjfmpp = [(F1,1),(F2,1),(F1,1),(F2,0)]
    # Step 7: F2.s0 -> regular. T3 matched. Advance to s1.
    # buvfjfmpp = [(F1,1),(F2,1),(F1,1),(F2,1)]
    # Step 8: F2.s1 = portal. uxncrzlau = 1. FALSE. No anti-loop.
    # Enter F1. buvfjfmpp = [(F1,1),(F2,1),(F1,1),(F2,1),(F1,0)]
    # This will loop forever!
    #
    # So mutual recursion with portals at slot 1 would be infinite. The game must have
    # a different structure.
    #
    # Let me try portals at s0 instead:
    # F1.s0=portal(c9)->F2, F2.s0=portal(c8)->F1:
    # buvfjfmpp = [(F1, 0)]
    # F1.s0 = portal at slot 0. uxncrzlau = 0.
    # Check: uxncrzlau==0 TRUE. (F1,0) in buvfjfmpp[:-1]? buvfjfmpp[:-1] = []. FALSE.
    # Enter F2. buvfjfmpp = [(F1,0),(F2,0)]
    # F2.s0 = portal at slot 0. uxncrzlau = 0.
    # Check: uxncrzlau==0 TRUE. (F2,0) in buvfjfmpp[:-1]? buvfjfmpp[:-1] = [(F1,0)]. FALSE.
    # AND buvfjfmpp[-2][1] == 0? buvfjfmpp[-2]=(F1,0), [1]=0. TRUE.
    # All conditions TRUE -> sibihgzarf() ERROR! Anti-loop CATCHES this!
    #
    # Interesting. So the anti-loop specifically catches: entering a frame at slot 0 when the
    # grandparent (parent's parent) was also at slot 0. This prevents F1->F2->F1 at slot 0.
    #
    # What about F1.s0=portal->F2 and F2.s2=portal->F1?
    # buvfjfmpp = [(F1, 0)]
    # F1.s0=portal. uxncrzlau=0. (F1,0) in []? FALSE. Enter F2.
    # buvfjfmpp = [(F1,0),(F2,0)]
    # F2.s0 = regular. T0. Advance.
    # buvfjfmpp = [(F1,0),(F2,1)]. F2.s1 = regular. T1. Advance.
    # buvfjfmpp = [(F1,0),(F2,2)]. F2.s2 = portal(c8)->F1.
    # uxncrzlau=2. uxncrzlau==0? FALSE. No anti-loop. Enter F1.
    # buvfjfmpp = [(F1,0),(F2,2),(F1,0)]
    # F1.s0=portal. uxncrzlau=0. (F1,0) in [(F1,0),(F2,2)]? YES!
    # AND buvfjfmpp[-2][1]==0? buvfjfmpp[-2]=(F2,2), [1]=2. FALSE.
    # Overall condition FALSE (because third part fails). Portal enters!
    # buvfjfmpp = [(F1,0),(F2,2),(F1,0),(F2,0)]
    # F2.s0=regular. T2. Advance.
    # buvfjfmpp = [(F1,0),(F2,2),(F1,0),(F2,1)]. F2.s1=regular. T3. Advance.
    # buvfjfmpp = [(F1,0),(F2,2),(F1,0),(F2,2)]. F2.s2=portal.
    # uxncrzlau=2. uxncrzlau==0? FALSE. Enter F1.
    # buvfjfmpp = [(F1,0),(F2,2),(F1,0),(F2,2),(F1,0)]
    # F1.s0=portal. uxncrzlau=0. (F1,0) in [first 4]? YES.
    # buvfjfmpp[-2]=(F2,2), [1]=2≠0. FALSE. Enter again!
    # Infinite loop again!
    #
    # OK, so mutual recursion only terminates when both portals are at slot 0. Otherwise infinite.
    # But at slot 0, the anti-loop catches it immediately, preventing even one nested visit.
    #
    # Let me reconsider. Maybe L8 doesn't use mutual recursion. Maybe one portal goes in F1
    # and the other portal is ALSO placed in F1 (both routing to... F2 and F1 respectively).
    #
    # portal c9 -> F2 (placed in F1)
    # portal c8 -> F1 (placed in... F2? Then it's mutual recursion again)
    #
    # Actually, if portal c8 is placed in F2, it routes BACK to F1. That's mutual recursion.
    # If portal c8 is placed in F1, it routes to F1... which would be checked immediately.
    # uxncrzlau=0 check for self-loop: (F1,0) in buvfjfmpp[:-1]? Depends on position.
    #
    # Wait — maybe I should step back. Let me reconsider:
    # Maybe portal c8 doesn't go to F1. Maybe the frame matching uses a DIFFERENT attribute.
    #
    # Re-reading the portal matching code (line 984):
    # amkoiofqhs = next(frame for frame in qaagahahj if frame.pixels[0,0] == ldwfvtgapk.pixels[1,1])
    # portal inner color (pixels[1,1]) matches frame border color (pixels[0,0]).
    #
    # For L8:
    # F1: qdmvvkvhaz4 at (18,22), default color = 8 (border pixels[0,0] = 8)
    # F2: qdmvvkvhaz4 at (18,36), color_remap(None, 9) -> pixels[0,0] = 9
    #
    # Portal c8 at (46,56): inner color = pixels[1,1] = 8. Routes to F1 (color 8).
    # Portal c9 at (53,56): inner color = pixels[1,1] = 9. Routes to F2 (color 9).
    #
    # Hmm, but what if portal c8 is used to create a SELF-REFERENCING portal in F1?
    # If placed in F1, it would try to enter F1 itself.
    #
    # Wait, actually — the qaagahahj iteration is `next(frame for frame in self.qaagahahj if ...)`.
    # If a portal with color 8 is in F1, it finds F1 (since F1 has pixels[0,0]=8).
    # Then it pushes (F1, 0) onto buvfjfmpp.
    # If this is the first visit, the check fails and we enter F1 recursively.
    # This IS a self-reference!
    #
    # Let me trace F1.s0=regular, F1.s1=portal(c9)->F2, F1.s2=portal(c8)->F1, F1.s3=regular:
    # buvfjfmpp = [(F1, 0)]
    # F1.s0 -> T0=8. Advance.
    # buvfjfmpp = [(F1, 1)]. F1.s1=portal(c9)->F2. uxncrzlau=1≠0. Enter F2.
    # buvfjfmpp = [(F1,1),(F2,0)].
    # F2 has 4 slots, all regular: s0->T1, s1->T2, s2->T3, s3->T4.
    # After F2 done, pop. buvfjfmpp = [(F1,1)]. ppsxsxiod=True. Advance to s2.
    # buvfjfmpp = [(F1,2)]. F1.s2=portal(c8)->F1. uxncrzlau=2≠0. Enter F1.
    # buvfjfmpp = [(F1,2),(F1,0)].
    # F1.s0->T5. Advance.
    # buvfjfmpp = [(F1,2),(F1,1)]. F1.s1=portal(c9)->F2. uxncrzlau=1≠0. Enter F2.
    # buvfjfmpp = [(F1,2),(F1,1),(F2,0)].
    # F2: s0->T6, s1->T7, s2->T8, s3->T9. Pop.
    # buvfjfmpp = [(F1,2),(F1,1)]. ppsxsxiod=True. Advance to s2.
    # buvfjfmpp = [(F1,2),(F1,2)]. F1.s2=portal(c8)->F1. uxncrzlau=2≠0. Enter F1.
    # buvfjfmpp = [(F1,2),(F1,2),(F1,0)].
    # Check: uxncrzlau=0 TRUE. (F1,0) in [(F1,2),(F1,2)]? No, (F1,0)≠(F1,2). FALSE.
    # AND buvfjfmpp[-2]=(F1,2), [1]=2≠0. FALSE.
    # Enter F1 again! This loops.
    #
    # Hmm, infinite loop again. The anti-loop only catches specific cases.
    #
    # But if F2 also has a portal somewhere, maybe the walk terminates differently...
    #
    # Actually, maybe I should think about this differently. With 12 targets and the pattern
    # [8,11,12,9,14,15] repeating, maybe the solution uses a simpler structure:
    # F1 contains a portal to F2. F2 has no portal. But F1's portal is visited twice
    # because F1 itself is entered twice (once directly, once via self-referencing portal).
    #
    # Or maybe: F1 has 2 portals to F2 and no self-referencing portal.
    # But then: 2 regular F1 slots + 2 * 4 F2 slots = 2 + 8 = 10 ≠ 12.
    # Or 1 regular + 3 portals to F2: 1 + 3*4 = 13. Too many.
    #
    # Actually, maybe there's a cleverer structure. Let me look at the target pattern:
    # [8, 11, 12, 9, 14, 15, 8, 11, 12, 9, 14, 15]
    # Two identical blocks of 6.
    #
    # With 2 frames of 4 slots each = 8 slots. 12 targets.
    # Visited once each = 8. Need 4 more from revisiting.
    #
    # What if F1 has 1 portal to F2, and F2 has 1 portal back to F1, but the anti-loop
    # terminates after one bounce? Let me try specific positions.
    #
    # F1: s0=portal(c9)->F2, s1=regular, s2=regular, s3=regular
    # F2: s0=portal(c8)->F1, s1=regular, s2=regular, s3=regular
    # Walk:
    # (F1,0): portal->F2. buvfjfmpp = [(F1,0),(F2,0)]
    # (F2,0): portal->F1. uxncrzlau=0. (F2,0) in [(F1,0)]? No. AND buvfjfmpp[-2]=(F1,0),[1]=0==0 TRUE.
    # Overall: uxncrzlau==0 TRUE, (F2,0) in [(F1,0)] FALSE. So overall FALSE. Hmm wait.
    # The condition is:
    # uxncrzlau == 0 AND (kmsegkpkh, uxncrzlau) in buvfjfmpp[:-1] AND buvfjfmpp[-2][1] == 0
    # All three must be TRUE for the error.
    # uxncrzlau = 0 ✓
    # (kmsegkpkh, 0) = (F1, 0). Is (F1,0) in buvfjfmpp[:-1] = [(F1,0)]? YES ✓
    # buvfjfmpp[-2] = buvfjfmpp[0] = (F1,0). [1] = 0 == 0. YES ✓
    # ERROR! Anti-loop catches it.
    #
    # So F1.s0->F2.s0->F1 is caught. But F1.s0->F2.sN->F1 where N>0?
    #
    # F1: s0=portal(c9)->F2. F2: s0=regular, s1=regular, s2=regular, s3=portal(c8)->F1.
    # Walk:
    # (F1,0): portal. uxncrzlau=0. (F1,0) in []? FALSE. Enter F2.
    # buvfjfmpp = [(F1,0),(F2,0)]
    # (F2,0): regular. T0. Advance.
    # (F2,1): regular. T1. Advance.
    # (F2,2): regular. T2. Advance.
    # (F2,3): portal(c8)->F1. uxncrzlau=3≠0. FALSE. Enter F1.
    # buvfjfmpp = [(F1,0),(F2,3),(F1,0)]
    # (F1,0): portal. uxncrzlau=0. (F1,0) in [(F1,0),(F2,3)]? YES.
    # buvfjfmpp[-2]=(F2,3). [1]=3≠0. FALSE. Overall FALSE!
    # Enter F2 AGAIN. buvfjfmpp = [(F1,0),(F2,3),(F1,0),(F2,0)]
    # (F2,0): regular. T3. Advance. (F2,1): T4. (F2,2): T5.
    # (F2,3): portal. uxncrzlau=3≠0. Enter F1.
    # buvfjfmpp = [(F1,0),(F2,3),(F1,0),(F2,3),(F1,0)]
    # (F1,0): portal. uxncrzlau=0. (F1,0) in [(F1,0),(F2,3),(F1,0),(F2,3)]? YES.
    # buvfjfmpp[-2]=(F2,3). [1]=3≠0. FALSE. Enter F2 AGAIN!
    # Still infinite!
    #
    # The only way to prevent infinite loops is to have the portal at slot 0 of a
    # frame that's already in the stack when the parent was also at slot 0.
    #
    # OK, I think the key insight for L8 is that the portal c8 routes to F1 BUT
    # it's NOT placed in any frame. Instead, maybe it stays on the palette.
    # And maybe F1 has two portals to F2, but the target pattern allows it.
    #
    # Target: [8,11,12,9,14,15, 8,11,12,9,14,15] — two identical blocks of 6.
    # If F1 has 2 portals to F2, placed at s1 and s3:
    # F1.s0->T0=8, portal->F2[T1,T2,T3,T4]=[11,12,9,14],
    # F1.s2->T5=15, portal->F2[T6,T7,T8,T9]=[8,11,12,9],
    # F1.s3... wait, the portal IS at s3. After portal, s3 is consumed.
    # Hmm, I need to recount.
    #
    # F1 has 4 slots. If portals at s1 and s3:
    # s0->T0=8
    # s1=portal(c9)->F2: F2[T1,T2,T3,T4]=[11,12,9,14]
    # s2->T5=15
    # s3=portal(c9)->F2: F2[T6,T7,T8,T9]=[8,11,12,9]
    # After s3: F1 done, but targets remain: T10=14, T11=15
    # F1 is the only top-level frame, so this is a fail.
    #
    # Hmm, that only gives 10 targets from 2+4+4=10. Need 12.
    #
    # What about 3 portals in F1?
    # F1: s0=regular, s1=portal, s2=portal, s3=portal
    # 1 regular + 3*4=12... but 1+12=13≠12.
    #
    # F1: s0=portal, s1=portal, s2=portal, s3=regular
    # 3*4+1=13≠12.
    #
    # What if the portal(c8) is placed in F2, routing to F1, creating a bounce?
    # F1.s0=portal(c9)->F2.
    # F2: s0=regular, s1=regular, s2=regular, s3=portal(c8)->F1.
    #
    # Walk:
    # (F1,0): portal. Enter F2. [(F1,0),(F2,0)]
    # F2.s0->T0=8. F2.s1->T1=11. F2.s2->T2=12. F2.s3=portal->F1.
    # uxncrzlau=3≠0. Enter F1. [(F1,0),(F2,3),(F1,0)]
    # F1.s0=portal. uxncrzlau=0. (F1,0) in [(F1,0),(F2,3)]? YES.
    # buvfjfmpp[-2]=(F2,3),[1]=3≠0. FALSE. Enter F2!
    # [(F1,0),(F2,3),(F1,0),(F2,0)]
    # F2.s0->T3=9. F2.s1->T4=14. F2.s2->T5=15. F2.s3=portal->F1.
    # uxncrzlau=3≠0. Enter F1.
    # [(F1,0),(F2,3),(F1,0),(F2,3),(F1,0)]
    # F1.s0=portal. uxncrzlau=0. (F1,0) in first 4? YES (F1,0) at index 0.
    # buvfjfmpp[-2]=(F2,3),[1]=3≠0. FALSE. Enter F2 AGAIN!
    # Infinite loop.
    #
    # The anti-loop only catches it when buvfjfmpp[-2][1]==0, which means the
    # grandparent was at slot 0. This never happens when bouncing between
    # non-zero portal slots.
    #
    # SO the game might actually have a finite walk if the total targets are consumed
    # and the WIN condition triggers (line 926: pmygakdvy == len(wcfyiodrx)-1 and filled).
    # When all 12 targets are filled, the next dbfxrigdqx call checks wrudcanmwy
    # (line 919: target already filled) and sees it's the last target -> WIN!
    #
    # YES! The win condition doesn't need the walk to "finish" — it just needs all
    # targets to be filled. Once pmygakdvy reaches 11 (last target) and wrudcanmwy is true,
    # it wins, even if the walk would have continued infinitely.
    #
    # So the pattern works:
    # F1.s0=portal(c9)->F2.
    # F2: s0=item, s1=item, s2=item, s3=portal(c8)->F1.
    # Walk: F2.s0->T0, F2.s1->T1, F2.s2->T2, portal->F1,
    #   F1.s0=portal->F2, F2.s0->T3, F2.s1->T4, F2.s2->T5, portal->F1,
    #   F1.s0=portal->F2, F2.s0->T6, F2.s1->T7, F2.s2->T8, portal->F1,
    #   F1.s0=portal->F2, F2.s0->T9, F2.s1->T10, F2.s2->T11 WIN!
    #   (pmygakdvy=11 = last target, wrudcanmwy=True -> next_level)
    #
    # But wait, each F2 visit reads the SAME 3 items. So targets must repeat:
    # T[0..2] = T[3..5] = T[6..8] = T[9..11]
    # [8,11,12] = [9,14,15] = [8,11,12] = [9,14,15]
    # 8≠9! NOT EQUAL!
    #
    # The targets are [8,11,12,9,14,15,8,11,12,9,14,15].
    # Blocks of 3: [8,11,12], [9,14,15], [8,11,12], [9,14,15]
    # Alternating! So F2 can't have all 3 the same each visit.
    #
    # Unless F2 items change between visits... which they don't.
    #
    # Hmm. What if the bounce is between two different frames, each handling one block?
    # F1 has portal to F2 at s0, F2 has portal to F1 at s3.
    # F1 has 3 regular items for [8,11,12] and portal at s0.
    # Wait, F1.s0 IS the portal. So F1's regular items are at s1, s2, s3.
    # But s3 can't be regular AND a portal.
    #
    # Let me try: F1.s3=portal(c9)->F2, F2.s0=portal(c8)->F1.
    # Walk:
    # (F1,0): F1.s0=regular->T0. Advance.
    # (F1,1): s1->T1. (F1,2): s2->T2. (F1,3): portal->F2.
    # [(F1,3),(F2,0)]
    # F2.s0=portal(c8)->F1. uxncrzlau=0. (F2,0) in [(F1,3)]? NO.
    # AND buvfjfmpp[-2]=(F1,3),[1]=3≠0. FALSE. Overall FALSE. Enter F1.
    # [(F1,3),(F2,0),(F1,0)]
    # F1.s0->T3. F1.s1->T4. F1.s2->T5. F1.s3=portal->F2.
    # uxncrzlau=3≠0. Enter F2.
    # [(F1,3),(F2,0),(F1,3),(F2,0)]
    # F2.s0=portal. uxncrzlau=0. (F2,0) in [(F1,3),(F2,0),(F1,3)]? YES (index 1).
    # buvfjfmpp[-2]=(F1,3),[1]=3≠0. FALSE. Overall FALSE. Enter F1.
    # [(F1,3),(F2,0),(F1,3),(F2,0),(F1,0)]
    # F1.s0->T6. F1.s1->T7. F1.s2->T8. F1.s3=portal->F2.
    # Enter F2. [(F1,3),(F2,0),(F1,3),(F2,0),(F1,3),(F2,0)]
    # F2.s0=portal. uxncrzlau=0. (F2,0) in stack? YES.
    # buvfjfmpp[-2]=(F1,3),[1]=3≠0. FALSE. Enter F1.
    # F1.s0->T9. F1.s1->T10. F1.s2->T11. WIN!
    # (pmygakdvy=11, wrudcanmwy checks if T11 is already filled. After T11 is matched,
    # the fill animation runs, then advance. At that point pmygakdvy would be 11,
    # and on the next check wrudcanmwy = True -> WIN!)
    #
    # Actually wait — let me re-check when WIN triggers.
    # The win check at line 926: if self.pmygakdvy == len(self.wcfyiodrx) - 1 and wrudcanmwy:
    # This checks BEFORE processing the current item. It means: if the last target is already
    # filled (from a previous match), AND we're looking at the last target, WIN.
    #
    # So the flow is:
    # - Match T11 (last target). Fill animation plays. xjxrqgaqw runs.
    # - After fill, modqnpqfi=5, then dbfxrigdqx called again.
    # - At this point, the walk continues at F1.s3 (portal).
    # - pmygakdvy is now pointing past all targets? No, let me re-check.
    #
    # Actually, when a match happens (line 922: xjxrqgaqw = 0), the fill animation runs.
    # Then modqnpqfi = 5 (line 817), then dbfxrigdqx is called again.
    # At that point, pmygakdvy still points to the same target (11), and wrudcanmwy is True
    # (the target was just filled).
    # So line 926: pmygakdvy == 11 (len-1) and wrudcanmwy=True -> WIN!
    #
    # So the walk for this layout is:
    # F1 (3 regular + portal at s3) gets visited 4 times, F2 (portal at s0) is just a bounce pad.
    # Each F1 visit provides 3 targets. 4 * 3 = 12. ✓
    # F1 items must match all target blocks: [8,11,12] and [9,14,15].
    # But F1 items are the SAME each visit! [8,11,12] ≠ [9,14,15]. WRONG!
    #
    # Ugh. Same problem.
    #
    # Wait — what if both F1 and F2 have regular items AND portals?
    # F1: s0=regular, s1=regular, s2=regular, s3=portal(c9)->F2
    # F2: s0=regular, s1=regular, s2=regular, s3=portal(c8)->F1
    #
    # Walk:
    # F1.s0->T0=8, F1.s1->T1=11, F1.s2->T2=12, F1.s3=portal->F2
    # F2.s0->T3=9, F2.s1->T4=14, F2.s2->T5=15, F2.s3=portal->F1
    # F1.s0->T6=8, F1.s1->T7=11, F1.s2->T8=12, F1.s3=portal->F2
    # F2.s0->T9=9, F2.s1->T10=14, F2.s2->T11=15.
    # pmygakdvy=11, wrudcanmwy=True -> WIN!
    #
    # F1 items: s0=c8, s1=c11, s2=c12. Always [8,11,12]. Matches T[0,1,2] and T[6,7,8]. ✓
    # F2 items: s0=c9, s1=c14, s2=c15. Always [9,14,15]. Matches T[3,4,5] and T[9,10,11]. ✓
    # PERFECT MATCH!
    #
    # But wait — does the anti-loop actually allow this?
    # Let me trace the full anti-loop checks:
    #
    # Visit 1:
    # F1.s3=portal(c9). uxncrzlau=3≠0. No check. Enter F2.
    # buvfjfmpp = [(F1,3),(F2,0)]
    #
    # F2.s3=portal(c8)->F1. uxncrzlau=3≠0. No check. Enter F1.
    # buvfjfmpp = [(F1,3),(F2,3),(F1,0)]
    #
    # Visit 2:
    # F1.s3=portal(c9). uxncrzlau=3≠0. No check. Enter F2.
    # buvfjfmpp = [(F1,3),(F2,3),(F1,3),(F2,0)]
    #
    # F2.s3=portal(c8)->F1. uxncrzlau=3≠0. No check. Enter F1.
    # buvfjfmpp = [(F1,3),(F2,3),(F1,3),(F2,3),(F1,0)]
    #
    # Visit 3:
    # F1.s0->T6. s1->T7. s2->T8. s3=portal. Enter F2.
    # buvfjfmpp = [(F1,3),(F2,3),(F1,3),(F2,3),(F1,3),(F2,0)]
    # F2.s0->T9. s1->T10. s2->T11. WIN before reaching s3!
    #
    # T11 is the last target (index 11 = len-1). After it's matched,
    # dbfxrigdqx is called again. pmygakdvy=11, wrudcanmwy=True -> next_level! ✓
    #
    # So the solution is:
    # F1 (4 slots): s0=c8, s1=c11, s2=c12, s3=portal(c9)
    # F2 (4 slots): s0=c9, s1=c14, s2=c15, s3=portal(c8)
    #
    # Placements:
    # F1 slots at (20,24),(26,24),(32,24),(38,24)
    # F2 slots at (20,38),(26,38),(32,38),(38,38)
    # Palette: c14@(39,56), c8@(25,56), c9@(32,56), c12@(18,56), c11@(11,56), c15@(4,56)
    # Portal c8@(46,56), portal c9@(53,56)
    #
    # F1.s0 (20,24) <- c8 from (25,56)
    # F1.s1 (26,24) <- c11 from (11,56)
    # F1.s2 (32,24) <- c12 from (18,56)
    # F1.s3 (38,24) <- portal(c9) from (53,56)
    # F2.s0 (20,38) <- c9 from (32,56)
    # F2.s1 (26,38) <- c14 from (39,56)
    # F2.s2 (32,38) <- c15 from (4,56)
    # F2.s3 (38,38) <- portal(c8) from (46,56)

    l8 = []
    l8.append(click(25, 56)); l8.append(click(20, 24))  # c8  -> F1.s0
    l8.append(click(11, 56)); l8.append(click(26, 24))  # c11 -> F1.s1
    l8.append(click(18, 56)); l8.append(click(32, 24))  # c12 -> F1.s2
    l8.append(click(53, 56)); l8.append(click(38, 24))  # portal(c9) -> F1.s3
    l8.append(click(32, 56)); l8.append(click(20, 38))  # c9  -> F2.s0
    l8.append(click(39, 56)); l8.append(click(26, 38))  # c14 -> F2.s1
    l8.append(click(4, 56));  l8.append(click(32, 38))  # c15 -> F2.s2
    l8.append(click(46, 56)); l8.append(click(38, 38))  # portal(c8) -> F2.s3
    l8.append(SUBMIT)
    print(f"L8: {len(l8)} actions")
    results["L8"] = {"actions": l8, "n_actions": len(l8)}
    all_actions.extend(l8)

    # ==================== SUMMARY ====================
    total = sum(len(results[f"L{i}"]["actions"]) for i in range(1, 9))
    print(f"\nTotal actions across all 8 levels: {total}")

    return results, all_actions


def verify_with_game():
    """Run the solutions against the actual game."""
    import logging
    from arc_agi import LocalEnvironmentWrapper, EnvironmentInfo

    results, all_actions = solve_all()

    print("\n" + "=" * 60)
    print("VERIFICATION against actual game")
    print("=" * 60)

    info = EnvironmentInfo(
        game_id='sb26',
        local_dir='environment_files/sb26/7fbdac44',
        class_name='Sb26'
    )
    logger = logging.getLogger('sb26')
    logger.setLevel(logging.WARNING)
    s = LocalEnvironmentWrapper(info, logger, scorecard_id='test', seed=0)
    s.reset()

    def do_action(session, action_id):
        """Execute an encoded action: click(x,y)=7+y*64+x, submit=4, undo=6."""
        if action_id >= 7:
            x = (action_id - 7) % 64
            y = (action_id - 7) // 64
            return session.step(6, data={'x': x, 'y': y})
        elif action_id == 4:
            return session.step(5)  # submit
        elif action_id == 6:
            return session.step(7)  # undo
        else:
            return session.step(action_id + 1)

    levels_done = 0
    for level in range(1, 9):
        key = f"L{level}"
        actions = results[key]["actions"]
        for a in actions:
            obs = do_action(s, a)

        levels_done = obs.levels_completed
        state = str(obs.state)
        passed = levels_done >= level
        status = "PASS" if passed else "FAIL"
        print(f"  L{level}: {len(actions):2d} actions -> levels_completed={levels_done}, state={state} [{status}]")

        if not passed:
            print(f"  FAILED at L{level}! Stopping.")
            break

    # Save results
    output = {
        "game": "sb26",
        "source": "analytical_abstract_model_solver",
        "type": "fullchain",
        "max_level": levels_done,
        "total_actions": sum(len(results[f"L{i}"]["actions"]) for i in range(1, 9)),
    }
    for i in range(1, 9):
        key = f"L{i}"
        output[f"l{i}_actions"] = results[key]["actions"]
        output[f"l{i}_n_actions"] = results[key]["n_actions"]
    output["all_actions"] = all_actions

    outpath = "B:/M/the-search/experiments/results/prescriptions/sb26_fullchain.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {outpath}")

    return levels_done


if __name__ == "__main__":
    levels = verify_with_game()
    if levels >= 8:
        print("\n*** ALL 8 LEVELS SOLVED! ***")
    else:
        print(f"\n*** Solved {levels}/8 levels ***")
