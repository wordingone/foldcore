"""
SB26 Analytical Solver — solve all 8 levels by reading game source data.

SB26 is a color sorting game:
- Top row: target color sequence (quhhhthrri sprites sorted by y,x)
- Middle: frames (pkpgflvjel tagged) containing slots (susublrply) and portals (vgszefyyyp)
- Bottom: palette of colored items (lngftsryyw at y=56)
- Goal: place items from palette into slots so they match the target sequence

Mechanics:
- Click palette item, then click slot = place item there
- Click two palette items = swap them
- Portals in frames route to other frames
- ACTION5 (submit) checks arrangement: walks frames left-to-right, following portals
- Target sequence = quhhhthrri sorted by (y,x), read their inner color

Action encoding: click(x,y) = 7 + y*64 + x, ACTION5=4, ACTION7(undo)=6
"""

import json
import sys
import os

# Add parent for arc_agi
sys.path.insert(0, "B:/M/the-search")

# Parse all level data from source
# Frame sprites and their slot counts (from sprite name suffix digit):
# jvkvqzheok1 = 1 slot (10px wide), jvkvqzheok2 = 2 slots (16px wide)
# pcrvmjfjzg3 = 3 slots (22px wide), qdmvvkvhaz4 = 4 slots (28px wide)
# nyqgqtujsa5 = 5 slots (34px wide), wbkmnqvtxh6 = 6 slots (34px wide)
# zzssdzqbbr7 = 7 slots (46px wide)

KOJDUUMCAP = 6  # slot spacing

def click_action(x, y):
    return 7 + y * 64 + x

def submit_action():
    return 4  # ACTION5

def undo_action():
    return 6  # ACTION7

# Parse sprite data
# lngftsryyw items (6x6): clickable palette items at bottom (y=56)
# susublrply (6x6): empty slots in frames
# vgszefyyyp (6x6): portals in frames, inner color = target frame color
# quhhhthrri (6x6): target sequence indicators

# Frame name -> number of slots
frame_slots = {
    "jvkvqzheok1": 1,
    "jvkvqzheok2": 2,
    "pcrvmjfjzg3": 3,
    "qdmvvkvhaz4": 4,
    "nyqgqtujsa5": 5,
    "wbkmnqvtxh6": 6,
    "zzssdzqbbr7": 7,
}

# Frame border color mapping (default color 8 = azure, portals match frame color)
# Frame border = 8 by default. color_remap(None, X) changes it.

# Parse each level
levels_data = [
    # Level 1: 4 targets, 1 frame (4 slots), 4 palette items, no portals
    {
        "targets": [(9, 18,1), (14, 25,1), (11, 32,1), (15, 39,1)],  # (color, x, y) of quhhhthrri
        "frames": [("qdmvvkvhaz4", 18, 25, 8, 4)],  # (name, x, y, border_color, n_slots)
        "palette": [(14, 17, 56), (15, 25, 56), (9, 33, 56), (11, 41, 56)],  # (color, x, y)
        "portals": [],
        "slots": [(20, 27), (26, 27), (32, 27), (38, 27)],  # susublrply positions
        "items_in_frames": [],  # lngftsryyw items already placed in frames (y <= 53)
    },
    # Level 2: 7 targets, 2 frames (4+4 slots), 7 palette items, 1 portal
    {
        "targets": [(12, 8,1), (15, 15,1), (8, 22,1), (9, 29,1), (14, 36,1), (11, 43,1), (6, 50,1)],
        "frames": [("qdmvvkvhaz4", 18, 18, 8, 4), ("qdmvvkvhaz4", 18, 32, 14, 4)],
        "palette": [(8, 8, 56), (15, 15, 56), (14, 22, 56), (12, 29, 56), (6, 36, 56), (9, 43, 56), (11, 50, 56)],
        "portals": [("vgszefyyyp", 32, 20, 14)],  # portal at (32,20) color 14 -> routes to frame with color 14
        "slots": [(20,20),(26,20),(38,20),(20,34),(26,34),(38,34),(32,34)],
        "items_in_frames": [],
    },
    # Level 3: 7 targets, 1 big frame (5 slots) + 2 sub-frames (2+2 slots), 7 palette items, 2 portals
    {
        "targets": [(14, 15,1), (15, 22,1), (11, 29,1), (6, 36,1), (9, 43,1), (8, 8,1), (12, 50,1)],
        # Sorted by (y,x): all at y=1, so by x: 8,15,22,29,36,43,50
        # Colors: 8,14,15,11,6,9,12
        "frames": [("nyqgqtujsa5", 15, 19, 8, 5), ("jvkvqzheok2", 15, 31, 14, 2), ("jvkvqzheok2", 33, 31, 9, 2)],
        "palette": [(8, 50, 56), (15, 22, 56), (14, 15, 56), (12, 8, 56), (6, 43, 56), (9, 36, 56), (11, 29, 56)],
        "portals": [("vgszefyyyp", 23, 21, 14), ("vgszefyyyp", 35, 21, 9)],
        # Portal at (23,21) color=14 -> frame at (15,31) color=14
        # Portal at (35,21) color=9 -> frame at (33,31) color=9
        "slots": [(29,21),(17,33),(23,33),(35,33),(41,33),(17,21),(41,21)],
        "items_in_frames": [],
    },
    # Level 4: 7 targets, 1 big frame (5 slots) + 1 sub-frame (3 slots), 7 palette items (6 at bottom + 1 in frame)
    {
        "targets": [(11, 8,1), (8, 15,1), (14, 22,1), (9, 29,1), (6, 36,1), (12, 43,1), (15, 50,1)],
        "frames": [("nyqgqtujsa5", 15, 18, 8, 5), ("pcrvmjfjzg3", 21, 32, 14, 3)],
        "palette_bottom": [(8, 29, 56), (6, 15, 56), (15, 36, 56), (12, 22, 56), (11, 8, 56), (9, 43, 56)],
        "palette_in_frame": [(14, 23, 34)],  # lngftsryyw at (23,34) color 14, inside the sub-frame
        "portals": [("vgszefyyyp", 50, 56, 14)],  # portal at (50,56) on palette row, color 14
        "slots": [(17,20),(23,20),(35,20),(41,20),(29,20),(29,34),(35,34)],
        "items_in_frames": [(14, 23, 34)],
    },
]

# Instead of manually parsing, let me write the solver using the game MCP

def solve_sb26():
    """Solve SB26 by programmatically analyzing source and producing action sequences."""

    # I'll work from the source level data directly.
    # The key insight: when ACTION5 (submit) is pressed, the game:
    # 1. Renders the frame, finds items/slots at each frame position
    # 2. Walks the target sequence (wcfyiodrx = quhhhthrri sorted by y,x)
    # 3. For each target, checks the corresponding frame slot
    # 4. If it's a plain item (lngftsryyw), its inner color must match the target
    # 5. If it's a portal (vgszefyyyp), it follows to the matching frame
    #
    # The walk order is:
    # - Start at frame[0] (first frame sorted by y,x), slot 0
    # - Walk slots left to right (slot spacing = 6px)
    # - When encountering a portal, push current frame and enter the portal's target frame
    # - When done with sub-frame, pop back and continue
    # - Each slot matches the next target in wcfyiodrx order

    # For each level, I need:
    # 1. Parse target colors in order (quhhhthrri sorted by y,x)
    # 2. Parse frame layout (frames sorted by y,x, with slot positions)
    # 3. Determine walk order through frames (following portals)
    # 4. Determine which palette item color goes in which slot
    # 5. Generate click sequence: for each slot in walk order, click the matching palette item then the slot

    print("SB26 Analytical Solver")
    print("=" * 60)

    # Level data parsed from source
    # Format: targets (quhhhthrri), frames, palette items (lngftsryyw), portals, slots

    all_level_actions = {}
    all_actions = []

    # For each level, I'll generate the action sequence
    # The mechanics for placing items:
    # 1. Click a palette item (at y=56, clickable)
    # 2. Click a slot position (susublrply at frame positions)
    # This places the item in the slot.
    # Then submit with ACTION5.

    # Level 1: Simple - 4 slots, no portals
    # Targets (sorted by y,x): all at y=1
    # x=18: color 9, x=25: color 14, x=32: color 11, x=39: color 15
    # Target sequence: [9, 14, 11, 15]
    # Frame: qdmvvkvhaz4 at (18,25), 4 slots
    # Slot positions (sorted by y,x in frame):
    #   susublrply at (20,27), (26,27), (32,27), (38,27) - spaced by 6
    # Walk order: slot0=(20,27), slot1=(26,27), slot2=(32,27), slot3=(38,27)
    # Need: slot0=color9, slot1=color14, slot2=color11, slot3=color15
    # Palette: (14,17,56), (15,25,56), (9,33,56), (11,41,56)
    # Click sequence: palette(9) @ (33+2,56+2)=(35,58) -> slot0 @ (20+2,27+2)=(22,29)
    #                 palette(14) @ (17+2,56+2)=(19,58) -> slot1 @ (26+2,27+2)=(28,29)
    #                 palette(11) @ (41+2,56+2)=(43,58) -> slot2 @ (32+2,27+2)=(34,29)
    #                 palette(15) @ (25+2,56+2)=(27,58) -> slot3 @ (38+2,27+2)=(40,29)
    #                 submit

    # Wait - need to verify click coordinates. The lngftsryyw sprite is 6x6 with
    # the inner 4x4 being color 8. The sys_click tag means clicking anywhere on
    # the sprite works. Let me use center of sprite: x+3, y+3 for 6x6 sprites.
    # Actually, the slot (susublrply) also has the sys_click tag.
    # Palette items: click at their position center (x+3, y+3)
    # Slots: click at their position center (x+3, y+3)

    # BUT: from the hjewbkcejq function (click handler), clicking an item selects it,
    # then clicking a slot places it. The click detection uses get_sprite_at(x,y,"sys_click").
    # So we need to click on the sprite's solid pixels.
    # lngftsryyw inner pixels are at offset (1,1) to (4,4) with color 8
    # The outer border is -2 (transparent for collision). So actual clickable area is inner.
    # susublrply inner clickable area: pixels 2,3 x 2,3 have color 2

    # For click coordinates in the 64x64 grid:
    # sprite at position (sx, sy) with size 6x6: center pixel = (sx+3, sy+3)

    # Level 1 solution:
    l1_actions = []
    # Palette items (color -> (x,y)):
    # color 14 at (17,56), color 15 at (25,56), color 9 at (33,56), color 11 at (41,56)
    # Target walk: slot0=color9, slot1=color14, slot2=color11, slot3=color15

    # Click color 9 palette (33,56) -> center (36,59)
    l1_actions.append(click_action(36, 59))  # 7+59*64+36 = 3819 ✓
    # Click slot 0 (20,27) -> center (23,30)
    l1_actions.append(click_action(23, 30))  # 7+30*64+23 = 1950 ✓
    # Click color 14 palette (17,56) -> center (20,59)
    l1_actions.append(click_action(20, 59))  # 7+59*64+20 = 3803 ✓
    # Click slot 1 (26,27) -> center (29,30)
    l1_actions.append(click_action(29, 30))  # 7+30*64+29 = 1956 ✓
    # Click color 11 palette (41,56) -> center (44,59)
    l1_actions.append(click_action(44, 59))  # 7+59*64+44 = 3827 ✓
    # Click slot 2 (32,27) -> center (35,30)
    l1_actions.append(click_action(35, 30))  # 7+30*64+35 = 1962 ✓
    # Click color 15 palette (25,56) -> center (28,59)
    l1_actions.append(click_action(28, 59))  # 7+59*64+28 = 3811 ✓
    # Click slot 3 (38,27) -> center (41,30)
    l1_actions.append(click_action(41, 30))  # 7+30*64+41 = 1968 ✓
    # Submit
    l1_actions.append(submit_action())

    # Verify against known solution
    known_l1 = [3819, 1950, 3803, 1956, 3827, 1962, 3811, 1968, 4]
    assert l1_actions == known_l1, f"L1 mismatch: {l1_actions} vs {known_l1}"
    print(f"L1: {len(l1_actions)} actions - VERIFIED against known solution")
    all_level_actions["L1"] = l1_actions
    all_actions.extend(l1_actions)

    # Level 2: 7 targets, 2 frames, 1 portal
    # Targets (sorted by y=1, then x): x=8:12, x=15:15, x=22:8, x=29:9, x=36:14, x=43:11, x=50:6
    # Target sequence: [12, 15, 8, 9, 14, 11, 6]
    # (Wait: quhhhthrri color_remap(None, color) - the inner 4x4 area gets this color)
    #
    # Frames sorted by (y,x):
    # Frame1: qdmvvkvhaz4 at (18,18), color 8, 4 slots
    # Frame2: qdmvvkvhaz4 at (18,32), color 14, 4 slots
    #
    # Slot positions inside frames (susublrply):
    # Frame1 area: y≈18-28. Slots: (20,20), (26,20), (38,20) -- wait, (32,20) is a portal
    # Frame2 area: y≈32-42. Slots: (20,34), (26,34), (32,34), (38,34) -- wait, 38 not listed
    # Actually from source: slots at (20,20),(26,20),(38,20),(20,34),(26,34),(38,34),(32,34)
    # Portal: vgszefyyyp at (32,20) color 14
    #
    # Walk order through frames:
    # Start at Frame1 (y=18, sorted first), slot positions in Frame1:
    # Frame1 at x=18, 4 slots start at x=18+2=20 with spacing 6: x=20,26,32,38
    # At slot x=20: susublrply at (20,20) -> plain slot
    # At slot x=26: susublrply at (26,20) -> plain slot
    # At slot x=32: vgszefyyyp at (32,20) color=14 -> PORTAL to Frame2
    #   Enter Frame2 at (18,32), 4 slots at x=20,26,32,38
    #   Frame2 slot x=20: susublrply at (20,34) -> plain slot
    #   Frame2 slot x=26: susublrply at (26,34) -> plain slot
    #   Frame2 slot x=32: susublrply at (32,34) -> plain slot
    #   Frame2 slot x=38: susublrply at (38,34) -- wait, (38,34) not in listed slots...
    #   Actually (38,34) is not listed. Listed: (20,34),(26,34),(38,34),(32,34)
    #   Wait: (38,34) IS listed. So Frame2 has slots at 20,26,32,38 all at y=34.
    #   After Frame2 completes, pop back to Frame1
    # At slot x=38: susublrply at (38,20) -> plain slot
    #
    # So walk order: Frame1.slot0(20,20), Frame1.slot1(26,20), PORTAL->Frame2,
    #   Frame2.slot0(20,34), Frame2.slot1(26,34), Frame2.slot2(32,34), Frame2.slot3(38,34),
    #   back to Frame1, Frame1.slot3(38,20)
    #
    # Wait - that gives 7 positions for 7 targets. But the portal position itself doesn't
    # count as a target slot. So:
    # Target[0] = Frame1.slot0 -> color 12
    # Target[1] = Frame1.slot1 -> color 15
    # Target[2] = PORTAL at Frame1.slot2 -> enters Frame2
    #   Target[2] = Frame2.slot0 -> color 8
    #   Target[3] = Frame2.slot1 -> color 9
    #   Target[4] = Frame2.slot2 -> color 14
    #   Target[5] = Frame2.slot3 -> color 11
    # Back to Frame1
    # Target[6] = Frame1.slot3 -> color 6
    #
    # That's 7 targets matching walk order.
    #
    # Palette (at y=56):
    # (8,8,56), (15,15,56), (14,22,56), (12,29,56), (6,36,56), (9,43,56), (11,50,56)
    #
    # Click sequence:
    # color12 at (29,56) -> slot (20,20)
    # color15 at (15,56) -> slot (26,20)
    # color8 at (8,56) -> slot (20,34) -- these go into frame2 via portal
    # color9 at (43,56) -> slot (26,34)
    # color14 at (22,56) -> slot (32,34)
    # color11 at (50,56) -> slot (38,34)
    # color6 at (36,56) -> slot (38,20)
    # submit

    l2_actions = []
    # color12 palette (29,56) center (32,59) -> slot (20,20) center (23,23)
    l2_actions.append(click_action(32, 59))  # 7+59*64+32 = 3815
    l2_actions.append(click_action(23, 23))  # 7+23*64+23 = 1495
    # Wait, let me check against the known L2 solution:
    # Known: [3750, 1372, 3736, 1378, 3729, 2268, 3764, 2274, 3743, 2280, 3771, 2286, 3757, 1390, 4]
    # 3750 = 7 + 3743 -> 3743/64 = 58 r 31 -> click(31,58)
    # Hmm, that's at y=58, not y=59. The palette items center at y=56+3=59?
    # Wait: lngftsryyw sprite is 6x6. Position is top-left. So (8,56) means x=8..13, y=56..61.
    # Center is at (8+2, 56+2) = (10, 58)? Or (8+3, 56+3) = (11, 59)?
    # Let me check: known L1 action 3819 -> click(36,59). Palette color 9 at pos (33,56).
    # 33+3=36, 56+3=59. So center = pos + 3.
    #
    # But known L2 action 3750 -> click(31,58). Hmm, 31-8=23? That doesn't match any palette item at x=8.
    # Let me recalculate: 3750-7=3743, 3743//64=58, 3743%64=31. Click at (31,58).
    # Palette at (29,56): center (32,59). That doesn't match (31,58).
    #
    # Hmm. Let me look at the inner clickable area of lngftsryyw more carefully.
    # lngftsryyw pixels: [-2,-2,-2,-2,-2,-2], [-2,8,8,8,8,-2], ...
    # The -2 pixels are transparent. The clickable area is the inner 4x4 at offset (1,1) to (4,4).
    # So for sprite at pos (x,y), clickable area is (x+1,y+1) to (x+4,y+4).
    # Center of clickable area: (x+2.5, y+2.5) -> pixel (x+2, y+2) or (x+3, y+3).
    #
    # Known L2 action 3750 -> (31, 58).
    # Looking at L2 palette: positions 8,15,22,29,36,43,50 at y=56.
    # Item at (29,56): inner area (30,57)-(33,60), center approx (31,58) or (32,59)
    # (31,58) = 29+2, 56+2. So the click coords use offset +2, not +3!
    #
    # Let me re-verify L1: action 3819 -> (36,59). Palette at (33,56): 33+3=36, 56+3=59.
    # So L1 uses offset +3 but L2 uses offset +2? That can't be right.
    #
    # Actually wait - in L1, the palette positions are different: (17,56), (25,56), (33,56), (41,56)
    # For 33+3=36, 56+3=59 -> (36,59) -> action 3819. ✓
    # But inner area is (34,57)-(37,60). Center is (35.5, 58.5). Pixel (35,58) or (36,59).
    # With offset +3: 33+3=36, 56+3=59. That would be bottom-right of inner area.
    #
    # For L2 action 3750 -> (31,58). Palette at (29,56): 29+2=31, 56+2=58.
    # Inner area: (30,57)-(33,60). (31,58) is top-left-ish of inner area.
    #
    # Both are valid click positions (inside the inner area). The exact pixel doesn't matter
    # as long as it hits a visible, non-transparent pixel of a sys_click sprite.
    # I should use +2 consistently since it's the top-left of the inner area.
    #
    # Let me redo L1 with +2 and see if it still works:
    # color9 at (33,56): (35,58). Action = 7+58*64+35 = 3754. Known = 3819 (which is +3).
    # Both should work since both hit the inner area. Let me just use +3 for consistency with L1.
    #
    # Actually, the known L2 uses +2. Let me just use +2 for everything, or check what the
    # existing L2 solution does more carefully.

    # Let me decode all known L2 actions:
    # 3750 -> (31,58) = palette(29,56)+2 = color 12? No wait:
    # L2 palette: (8,8,56)=c8, (15,15,56)=c15, (14,22,56)=c14, (12,29,56)=c12,
    #             (6,36,56)=c6, (9,43,56)=c9, (11,50,56)=c11
    # 3750 -> (31,58) = (29+2, 56+2) -> palette color 12 ✓
    # 1372 -> 1372-7=1365, 1365//64=21, 1365%64=21 -> (21,21) = ??
    # Slot at (20,20): (20+1,20+1)=(21,21)? No, (20+2,20+2)=(22,22)?
    # Hmm, (21,21) doesn't match +2 offset for slot at (20,20) which would be (22,22).
    # Let me check: susublrply pixels: [-2,-2,-2,-2,-2,-2], [-2,-2,-2,-2,-2,-2], [-2,-2,2,2,-2,-2], ...
    # The clickable pixels (color 2) are at offset (2,2), (3,2), (2,3), (3,3).
    # So for slot at (20,20), clickable pixels at (22,22), (23,22), (22,23), (23,23).
    # (21,21) is at pixel (-2) which is transparent for collision but IS color -2 meaning passthrough.
    #
    # Wait - in the get_sprite_at function, it checks for "sys_click" tag. The susublrply has
    # tags=["susublrply", "sys_click"]. The function returns the sprite if the clicked pixel
    # is at a non-(-1) position... Let me check:
    # susublrply pixel at offset (1,1) = -2. The sprite pixel -2 means... what?
    # In arcengine, -1 = transparent (don't render), -2 = transparent for rendering but
    # collidable/detectable.
    # So get_sprite_at checks if pixel >= -1 at clicked position. Since -2 < -1, it wouldn't
    # match. The clickable area is only where pixels are >= -1, which for susublrply is (2,2)-(3,3).
    #
    # But then (21,21) for slot (20,20): offset (1,1), pixel -2 -> not clickable?
    # Unless the slot has been replaced by a placed item...
    #
    # Hmm, let me re-examine. Actually, I think when you place an item in a slot, the item
    # sprite moves to the slot position, and then you're clicking on the item sprite instead.
    #
    # Wait, let me re-read the click handler (hjewbkcejq). It first checks if lqcskynzr is None:
    # - If None and clicked sprite is tagged "lngftsryyw": select it (set lqcskynzr = sprite)
    # - If selected (lqcskynzr not None):
    #   - If clicked "lngftsryyw": swap or reselect
    #   - If clicked "susublrply": move selected item to slot position
    #
    # So the flow is: click palette item (selects it), click slot (places it).
    # When placing, the palette item MOVES to the slot position.
    # So after placing, the slot is now occupied by the palette item sprite.
    #
    # For L2, the first action places color 12 from (29,56) to slot at...
    # Action 1372 -> (21,21).
    # Actually maybe the susublrply at (20,20) is clicked via its -2 pixels?
    # Let me re-read get_sprite_at more carefully from the hjewbkcejq code...
    # It uses self.current_level.get_sprite_at(knyvifgps, ijwolhvht, "sys_click")
    # The ARCBaseGame.get_sprite_at searches by tag. It checks if the pixel at (x,y)
    # is part of the sprite. For sprites with -2 pixels, -2 is treated as collidable.
    #
    # So -2 IS clickable! That means for susublrply at (20,20), clicking (21,21) hits
    # the pixel at offset (1,1) which is -2 -> collidable -> clickable. ✓
    #
    # So for sprites, any non-(-1) pixel is clickable when using get_sprite_at with a tag.
    # -1 = fully transparent, -2 = invisible but collidable.
    #
    # This means I can click anywhere within the 6x6 area of a sprite (as long as it's not -1).
    # For lngftsryyw, the border is -2 (offset 0 and 5) and inner is color 8 (offset 1-4).
    # All are non-(-1), so clicking at offset +1 (or +2, +3, +4) all work.
    #
    # The known L2 uses offset +2 for palette and +1 for slots.
    # Let me just use offset +2 for everything to be safe.

    # Actually let me just use the game MCP to test interactively. That will be faster
    # than trying to reverse-engineer the exact click coordinates.
    #
    # But first, let me figure out the walk order for each level, which is the hard part.
    # Then I can test with the MCP.

    # Let me rewrite this more systematically.
    pass


def analyze_level(level_idx, level_data):
    """
    Analyze a level's walk order and required placements.

    Returns list of (slot_x, slot_y, target_color) in walk order.
    """
    # Parse from source
    # quhhhthrri = target indicators, sorted by (y, x)
    # The walk through frames determines which slot maps to which target index
    pass


def solve_all_levels():
    """
    For each level 3-8, determine:
    1. Target sequence (quhhhthrri colors sorted by y,x)
    2. Frame walk order (following portals)
    3. Which palette item goes in which slot
    4. Generate click sequence
    """

    # I'll trace through each level manually based on the source code.
    # The submission check walks frames according to:
    # - qaagahahj = frames sorted by (y,x)
    # - Start at frame[0], slot 0
    # - rfdjlhefnd builds rzbeqaiky[frame] = items at each slot position
    # - dbfxrigdqx walks: if item is lngftsryyw and color matches target -> advance
    #   if item is vgszefyyyp (portal) -> enter linked frame
    # - Slots are at x = frame.x + 2 + i*6 for i in range(n_slots)
    #   where n_slots = int(frame.name[-1])

    # For each level, the slot positions within a frame are at:
    # x_slot[i] = frame.x + 2 + i * kojduumcap (kojduumcap=6)
    # y_slot = frame.y + 2
    # The rfdjlhefnd function looks for items at these (x_slot, y_slot) positions

    # Targets are wcfyiodrx = quhhhthrri sorted by (y,x). Their color is what
    # needs to be matched.

    levels_info = {}

    # ===================== LEVEL 3 =====================
    # Targets (quhhhthrri at y=1, sorted by x):
    # x=8:c8, x=15:c14, x=22:c15, x=29:c11, x=36:c6, x=43:c9, x=50:c12
    # Target sequence: [8, 14, 15, 11, 6, 9, 12]
    #
    # Frames (pkpgflvjel, sorted by y,x):
    # nyqgqtujsa5 at (15,19), n=5 slots. Color=8 (default).
    #   Slot positions: x=15+2+i*6 = 17,23,29,35,41 at y=19+2=21
    # jvkvqzheok2 at (15,31), n=2 slots. Color=14.
    #   Slot positions: x=15+2+i*6 = 17,23 at y=31+2=33
    # jvkvqzheok2 at (33,31), n=2 slots. Color=9.
    #   Slot positions: x=33+2+i*6 = 35,41 at y=31+2=33
    #
    # Items at slot positions:
    # susublrply at: (17,21),(23,21) -- wait, (29,21) also, but (23,21) has portal
    # Let me check: susublrply positions: (29,21),(17,33),(23,33),(35,33),(41,33),(17,21),(41,21)
    # And portals: vgszefyyyp at (23,21) color=14, vgszefyyyp at (35,21) color=9
    #
    # Frame1 (nyqgqtujsa5, 5 slots at x=17,23,29,35,41, y=21):
    #   x=17: susublrply at (17,21) -> plain slot
    #   x=23: vgszefyyyp at (23,21) color=14 -> PORTAL to frame with color 14
    #   x=29: susublrply at (29,21) -> plain slot
    #   x=35: vgszefyyyp at (35,21) color=9 -> PORTAL to frame with color 9
    #   x=41: susublrply at (41,21) -> plain slot
    #
    # Frame2 (jvkvqzheok2 at (15,31), color=14, 2 slots at x=17,23, y=33):
    #   x=17: susublrply at (17,33) -> plain slot
    #   x=23: susublrply at (23,33) -> plain slot
    #
    # Frame3 (jvkvqzheok2 at (33,31), color=9, 2 slots at x=35,41, y=33):
    #   x=35: susublrply at (35,33) -> plain slot
    #   x=41: susublrply at (41,33) -> plain slot
    #
    # Walk order through frames:
    # Frame1.slot0 (17,21) -> target[0]=8
    # Frame1.slot1 PORTAL(14) -> enter Frame2
    #   Frame2.slot0 (17,33) -> target[1]=14
    #   Frame2.slot1 (23,33) -> target[2]=15
    #   Exit Frame2, back to Frame1
    # Frame1.slot2 (29,21) -> target[3]=11
    # Frame1.slot3 PORTAL(9) -> enter Frame3
    #   Frame3.slot0 (35,33) -> target[4]=6
    #   Frame3.slot1 (41,33) -> target[5]=9
    #   Exit Frame3, back to Frame1
    # Frame1.slot4 (41,21) -> target[6]=12
    #
    # Placements needed:
    # (17,21) <- color 8  (palette at (50,56))
    # (17,33) <- color 14 (palette at (15,56))
    # (23,33) <- color 15 (palette at (22,56))
    # (29,21) <- color 11 (palette at (29,56))
    # (35,33) <- color 6  (palette at (43,56))
    # (41,33) <- color 9  (palette at (36,56))
    # (41,21) <- color 12 (palette at (8,56))
    #
    # Palette items (lngftsryyw at y=56):
    # color 8 at (50,56), color 15 at (22,56), color 14 at (15,56),
    # color 12 at (8,56), color 6 at (43,56), color 9 at (36,56), color 11 at (29,56)

    l3_actions = []
    # Place color 8: palette(50,56) -> slot(17,21)
    l3_actions.append(click_action(52, 58)); l3_actions.append(click_action(19, 23))
    # Place color 14: palette(15,56) -> slot(17,33)
    l3_actions.append(click_action(17, 58)); l3_actions.append(click_action(19, 35))
    # Place color 15: palette(22,56) -> slot(23,33)
    l3_actions.append(click_action(24, 58)); l3_actions.append(click_action(25, 35))
    # Place color 11: palette(29,56) -> slot(29,21)
    l3_actions.append(click_action(31, 58)); l3_actions.append(click_action(31, 23))
    # Place color 6: palette(43,56) -> slot(35,33)
    l3_actions.append(click_action(45, 58)); l3_actions.append(click_action(37, 35))
    # Place color 9: palette(36,56) -> slot(41,33)
    l3_actions.append(click_action(38, 58)); l3_actions.append(click_action(43, 35))
    # Place color 12: palette(8,56) -> slot(41,21)
    l3_actions.append(click_action(10, 58)); l3_actions.append(click_action(43, 23))
    l3_actions.append(submit_action())

    levels_info["L3"] = {"actions": l3_actions, "n_actions": len(l3_actions)}
    print(f"L3: {len(l3_actions)} actions")

    # ===================== LEVEL 4 =====================
    # Targets (y=1, sorted by x):
    # x=8:c11, x=15:c8, x=22:c14, x=29:c9, x=36:c6, x=43:c12, x=50:c15
    # Target sequence: [11, 8, 14, 9, 6, 12, 15]
    #
    # Frames (sorted by y,x):
    # nyqgqtujsa5 at (15,18), n=5 slots. Color=8.
    #   Slots at x=17,23,29,35,41, y=20
    # pcrvmjfjzg3 at (21,32), n=3 slots. Color=14.
    #   Slots at x=23,29,35, y=34
    #
    # Items at slot positions in Frame1:
    # susublrply at (17,20), (23,20), (29,20), (35,20), (41,20)
    # No portals in frame1... but wait:
    # Portal: vgszefyyyp at (50,56) color=14 -> this is on the PALETTE ROW, not in a frame!
    # Hmm, portals at y=56 are on the palette row. These are palette items but with portal sprite.
    #
    # Wait, lngftsryyw at (23,34) color=14 is IN the sub-frame (pcrvmjfjzg3 at (21,32)).
    # So the sub-frame already has an item placed in it.
    # And the portal at (50,56) is on the palette row as a portal item.
    #
    # Let me re-examine. on_set_level says:
    # Items with y > evrmzyfopo (53) get susublrply spots created for them -> palette items
    # Items with y <= 53 have sys_click removed -> already placed in frames
    #
    # So for L4:
    # lngftsryyw items: (8,29,56)=c8, (6,15,56)=c6, (15,36,56)=c15, (12,22,56)=c12,
    #                   (11,8,56)=c11, (9,43,56)=c9
    #                   (14,23,34)=c14 -> IN FRAME (y<=53), sys_click removed, not on palette
    # vgszefyyyp at (50,56) color=14 -> tagged lngftsryyw+sys_click, ON PALETTE (y>53)
    #
    # So palette row has: c11(8,56), c6(15,56), c12(22,56), c8(29,56), c15(36,56), c9(43,56)
    #                     and a PORTAL(14) at (50,56)
    #
    # Frame1 (nyqgqtujsa5 at (15,18), 5 slots at x=17,23,29,35,41, y=20):
    #   All slots are susublrply -> plain slots
    #   No portals in frame1 initially.
    #
    # Frame2 (pcrvmjfjzg3 at (21,32), 3 slots at x=23,29,35, y=34):
    #   Slot x=23: lngftsryyw at (23,34) color=14 -> already placed item!
    #   Slot x=29: susublrply at (29,34) -> plain slot
    #   Slot x=35: susublrply at (35,34) -> plain slot
    #
    # Wait but the portal at (50,56) has color=14 which matches Frame2's color.
    # But it's on the palette row... how does this work?
    #
    # Actually, I think the portal IS a palette item that you can click and place into a slot,
    # and when placed in a frame slot, it acts as a portal during submission.
    # vgszefyyyp has tags ["lngftsryyw", "sys_click"] so it IS treated as a lngftsryyw item.
    #
    # So the game mechanic is:
    # - Regular items (lngftsryyw): fill slots with their color
    # - Portal items (vgszefyyyp): when in a slot, during submission they route to the matching frame
    #
    # For L4, I need to place the portal(14) into one of Frame1's slots to create a route to Frame2.
    # Then place items in Frame2's slots to match the target sequence.
    #
    # The walk order needs the portal placed correctly.
    # Target sequence: [11, 8, 14, 9, 6, 12, 15]
    # Frame1 has 5 slots. 2 of them need to route via portal to Frame2's 3 slots.
    # Wait, Frame2 already has item c14 at slot0. So Frame2 needs 2 more items in slots 1,2.
    # Plus the portal takes 1 slot in Frame1.
    # That leaves 4 regular slots in Frame1 + 2 slots in Frame2 = 6, plus Frame2.slot0 already has c14.
    # Total = 7 = target count. ✓
    #
    # But which slot in Frame1 should the portal go?
    # The portal routes to Frame2 which has color 14.
    # Looking at the target sequence: [11, 8, 14, 9, 6, 12, 15]
    # Target[2] = 14. Frame2 already has c14 at slot0.
    # So the portal needs to go in Frame1's slot2 (x=29, y=20).
    # Then walk: slot0=c11, slot1=c8, slot2=PORTAL->Frame2,
    #   Frame2.slot0=c14(already there), Frame2.slot1=c9, Frame2.slot2=c6
    #   back to Frame1, slot3=c12, slot4=c15
    # Wait that gives: [11, 8, 14, 9, 6, 12, 15] = target sequence! ✓
    #
    # But wait - Frame2 slot0 already has the c14 item (lngftsryyw at (23,34) color=14).
    # When the submission check encounters this, it checks if the item's inner color matches
    # target[2]=14. Since it's 14, it passes. ✓
    #
    # Placements:
    # Frame1.slot0 (17,20) <- c11 from palette(8,56)
    # Frame1.slot1 (23,20) <- c8 from palette(29,56)
    # Frame1.slot2 (29,20) <- PORTAL(14) from palette(50,56)
    # Frame2.slot0 (23,34) <- c14 ALREADY THERE
    # Frame2.slot1 (29,34) <- c9 from palette(43,56)
    # Frame2.slot2 (35,34) <- c6 from palette(15,56)
    # Frame1.slot3 (35,20) <- c12 from palette(22,56)
    # Frame1.slot4 (41,20) <- c15 from palette(36,56)

    l4_actions = []
    # Place c11: palette(8,56) -> slot(17,20)
    l4_actions.append(click_action(10, 58)); l4_actions.append(click_action(19, 22))
    # Place c8: palette(29,56) -> slot(23,20)
    l4_actions.append(click_action(31, 58)); l4_actions.append(click_action(25, 22))
    # Place PORTAL(14): palette(50,56) -> slot(29,20)
    l4_actions.append(click_action(52, 58)); l4_actions.append(click_action(31, 22))
    # Place c9: palette(43,56) -> slot(29,34)
    l4_actions.append(click_action(45, 58)); l4_actions.append(click_action(31, 36))
    # Place c6: palette(15,56) -> slot(35,34)
    l4_actions.append(click_action(17, 58)); l4_actions.append(click_action(37, 36))
    # Place c12: palette(22,56) -> slot(35,20)
    l4_actions.append(click_action(24, 58)); l4_actions.append(click_action(37, 22))
    # Place c15: palette(36,56) -> slot(41,20)
    l4_actions.append(click_action(38, 58)); l4_actions.append(click_action(43, 22))
    l4_actions.append(submit_action())

    levels_info["L4"] = {"actions": l4_actions, "n_actions": len(l4_actions)}
    print(f"L4: {len(l4_actions)} actions")

    # ===================== LEVEL 5 =====================
    # Targets (y=1, sorted by x):
    # x=1:c6, x=8:c14, x=15:c8, x=22:c8, x=29:c14, x=36:c8, x=43:c8, x=50:c11, x=57:c15
    # Target sequence: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # 9 targets
    #
    # Frames (sorted by y,x):
    # nyqgqtujsa5 at (15,18), n=5 slots. Color=8.
    #   Slots at x=17,23,29,35,41, y=20
    # pcrvmjfjzg3 at (21,32), n=3 slots. Color=9.
    #   Slots at x=23,29,35, y=34
    #
    # Palette (lngftsryyw at y=56):
    # (14,39,56), (8,25,56), (11,32,56), (8,18,56), (6,11,56), (15,4,56), and others
    # Wait, items: set_position(39,56) c14, (25,56) c8, (32,56) c11, (18,56) c8, (11,56) c6, (4,56) c15
    # That's 6 palette items.
    #
    # Portals: vgszefyyyp at (46,56) c9, vgszefyyyp at (53,56) c9
    # TWO portals, both color=9 routing to Frame2 (pcrvmjfjzg3 at (21,32) color=9)
    #
    # Frame1 slots: 5 slots at (17,20), (23,20), (29,20), (35,20), (41,20)
    # All susublrply -> plain slots
    #
    # Frame2 slots: 3 slots at (23,34), (29,34), (35,34)
    # All susublrply -> plain slots
    #
    # 9 targets, 5 Frame1 slots + 3 Frame2 slots = 8 slots. Need 1 more.
    # But if I place 2 portals in Frame1, each takes a slot and adds 3 Frame2 slots?
    # No, both portals route to the SAME Frame2. You can't enter Frame2 twice.
    #
    # Wait, can you? The portal logic in dbfxrigdqx:
    # When encountering vgszefyyyp, it finds the frame matching the portal's inner color,
    # pushes current position, and enters the new frame. But it checks:
    # "if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0)"
    # This prevents infinite portal loops. But entering Frame2 a second time via a different
    # portal in Frame1 should be fine since it's not the same (frame, slot) pair in buvfjfmpp.
    #
    # Actually no - when entering a portal, it does:
    # amkoiofqhs = next(frame for frame in qaagahahj if frame.pixels[0,0] == portal_inner_color)
    # This finds THE frame with matching color. It always finds the same Frame2.
    # Then it pushes (Frame2, 0) onto buvfjfmpp.
    # The check "if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1]"
    # prevents re-entering if Frame2 is already in the stack. But the second portal would
    # push (Frame2, 0) again, and the check looks for it in buvfjfmpp[:-1].
    # If the first portal's Frame2 visit is already done and popped, then the second portal
    # can enter Frame2 again.
    #
    # So YES, two portals to the same frame works as long as they're not nested.
    # Walk order:
    # Frame1.slot0, Frame1.slot1, Frame1.slot2=PORTAL->Frame2, Frame2[0..2], back,
    # Frame1.slot3=PORTAL->Frame2, Frame2[0..2], back, Frame1.slot4
    #
    # But that would be 1 + 1 + 3 + 3 + 1 = 9 targets. But Frame2 only has 3 slots,
    # and visiting it twice would need 6 items in 3 slots. That can't work - each visit
    # reads the same 3 slots.
    #
    # Actually wait - the second visit reads the SAME items in Frame2. So targets[5,6,7]
    # would have to match the same colors as targets[2,3,4]. Let me check:
    # Target sequence: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # If portal at slot1 and slot3:
    # slot0=c6, slot1=PORTAL, Frame2=[c14,c8,c8], slot2=PORTAL (can't re-enter!),
    # Actually it would fail because Frame2 is already visited.
    #
    # Hmm, let me reconsider. Maybe only ONE portal is placed in Frame1, and the other
    # portal stays on the palette. Or maybe the portals aren't both needed.
    #
    # With 1 portal: 5 Frame1 slots - 1 portal = 4 regular + 3 Frame2 = 7. But 9 targets.
    # That's not enough.
    #
    # With 2 portals in Frame1:
    # 5 slots - 2 portals = 3 regular in Frame1 + 3 in Frame2 first visit + 3 in Frame2 second visit
    # But Frame2 has same items both times, so targets must be:
    # T[0], PORTAL, T[1..3]_frame2, T[4], PORTAL, T[5..7]_frame2_same, T[8]
    # Targets [2..4] must equal targets [5..7].
    # [8, 8, 14] vs [8, 8, 11]? No, that doesn't match.
    #
    # Unless the portals can have different configs... Let me re-think.
    # Both portals have color 9. Frame2 has color 9.
    #
    # Actually, I think I need to re-read the walk logic more carefully.
    # When the walk encounters a portal, it enters Frame2 from slot 0.
    # When Frame2 is complete, it pops back and continues at the next slot.
    # If it encounters another portal later, it checks:
    # "if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0)"
    # uxncrzlau = 0 (starting at slot 0 of Frame2)
    # buvfjfmpp[:-1] = previous entries, buvfjfmpp[-2][1] = previous frame's slot
    # If Frame2 was visited before AND the previous entry was also at slot 0, it errors.
    # Hmm, this is the re-entry check. Let me trace through:
    #
    # After first portal, buvfjfmpp = [(Frame1, portal1_idx), (Frame2, 0)]
    # Frame2 completes: pops Frame2, buvfjfmpp = [(Frame1, portal1_idx)]
    # Then continues, slot idx increments, next portal encountered.
    # buvfjfmpp = [(Frame1, portal2_idx), (Frame2, 0)]
    # Check: uxncrzlau=0, (Frame2,0) in buvfjfmpp[:-1]?
    # buvfjfmpp[:-1] = [(Frame1, portal2_idx)] -> (Frame2,0) not in there.
    # So the check passes! Frame2 can be entered again.
    #
    # Wait, but the check is:
    # (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1]
    # kmsegkpkh = the current frame being checked (Frame2), uxncrzlau=0
    # buvfjfmpp[:-1] = [(Frame1, portal2_idx)]
    # (Frame2, 0) is not in [(Frame1, portal2_idx)], so the check is False.
    # It proceeds! So Frame2 IS entered again. ✓
    #
    # But then Frame2 has the same items both times. The targets that map to Frame2
    # must be the same colors both times. Let me check:
    #
    # Target: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # Possible walk: slot0=6, slot1=PORTAL, [Frame2:8,8,14], slot3=PORTAL, [Frame2:8,8,14]...
    # No wait - [14,8,8] vs [8,8,11]. Hmm.
    #
    # Let me try portal at slot1 and slot3:
    # slot0=c6, slot1=PORTAL->Frame2[T1,T2,T3], slot2=T4, slot3=PORTAL->Frame2[T5,T6,T7], slot4=T8
    # T1..T3 = same items as T5..T7 (same Frame2 items read twice)
    # T: [6, 14, 8, 8, 14, 8, 8, 11, 15]
    # T1=14, T2=8, T3=8 AND T5=8, T6=8, T7=11
    # These DON'T match! Frame2 items can't be both [14,8,8] and [8,8,11].
    #
    # Try portal at slot0 and slot2:
    # slot0=PORTAL->Frame2[T0,T1,T2], slot1=T3, slot2=PORTAL->Frame2[T4,T5,T6], slot3=T7, slot4=T8
    # T0..T2 = T4..T6 => [6,14,8] = [14,8,8]? No.
    #
    # Try portal at slot0 and slot3:
    # slot0=PORTAL[T0..T2], slot1=T3, slot2=T4, slot3=PORTAL[T5..T7], slot4=T8
    # [6,14,8] = [8,8,11]? No.
    #
    # Try portal at slot2 and slot4:
    # slot0=T0, slot1=T1, slot2=PORTAL[T2..T4], slot3=T5, slot4=PORTAL[T6..T8]
    # [8,8,14] = [8,11,15]? No.
    #
    # NONE of these work with two portals to the same frame where items are the same!
    # So maybe only ONE portal is used, and the other stays on the palette.
    #
    # With 1 portal: 4 regular Frame1 slots + 3 Frame2 slots = 7. But 9 targets.
    # Doesn't add up to 9!
    #
    # Unless... there are more frames or items I'm missing.
    # Let me re-read L5 data more carefully.
    #
    # L5 has 9 target indicators (quhhhthrri). But only 8 slots (5 + 3) plus portals.
    #
    # Wait, I need to recount. L5 susublrply positions:
    # (17,20), (23,20), (29,20), (29,34), (23,34), (35,20), (35,34), (41,20)
    # That's 8 slots total: 5 in Frame1 (17,23,29,35,41 at y=20) and 3 in Frame2 (23,29,35 at y=34).
    # Plus 2 portals on palette. Total positions for items = 5 + 3 = 8.
    # But 9 targets. 8 ≠ 9. Something's wrong.
    #
    # Unless one portal IS used AND the walk visits Frame2 once, giving 4+3=7 item slots.
    # Still 7 ≠ 9.
    #
    # Wait, maybe I miscounted targets. Let me recount:
    # quhhhthrri in L5:
    # (1,1):c6, (8,1):c14, (15,1):c8, (22,1):c8, (29,1):c14, (36,1):c8, (43,1):c8, (50,1):c11, (57,1):c15
    # That's 9 targets with 9 uzxwqmkrmk backgrounds.
    # uzxwqmkrmk positions: (0,0),(7,0),(49,0),(14,0),(21,0),(28,0),(35,0),(42,0),(56,0)
    # 9 backgrounds for 9 targets. ✓
    #
    # 9 targets but only 8 item slots. That means I miscounted something.
    # Frame1 = nyqgqtujsa5 at (15,18), 5 slots.
    # Frame2 = pcrvmjfjzg3 at (21,32), 3 slots.
    # 5 + 3 = 8 slots.
    # If I use 1 portal (taking 1 Frame1 slot), that's 4 + 3 = 7 items.
    # 7 ≠ 9.
    #
    # Hmm. Maybe the two portals somehow enable double-visiting Frame2?
    # With 2 portals: 3 regular Frame1 + 3 first visit + 3 second visit = 9.
    # But Frame2 items are the same both times!
    #
    # UNLESS... the submission re-reads items each visit. If items were moved between visits?
    # No, the submission is a single check pass, items don't move.
    #
    # Wait, I think I misread the submission logic. Let me re-read rfdjlhefnd:
    # It renders the frame and builds rzbeqaiky by looking at what items/slots are
    # at each frame slot position. This happens ONCE at submit time.
    # Then dbfxrigdqx walks the sequence.
    #
    # For Frame2 with 3 slots, it has 3 items. When visited twice, it reads the SAME 3 items.
    # So targets for both visits must match. But they don't.
    #
    # I think the answer is: both portals aren't used. Some items might already be in frames.
    # Let me re-check L5's lngftsryyw items more carefully.
    #
    # L5 sprites from source:
    # lngftsryyw at (39,56) c14, (25,56) c8, (32,56) c11, (18,56) c8, (11,56) c6, (4,56) c15
    # That's 6 palette items + 2 portals = 8 placeable things.
    #
    # Hmm, no items already in frames (all at y=56).
    #
    # 5 Frame1 slots + 3 Frame2 slots = 8. 6 items + 2 portals = 8 things.
    # If 1 portal is placed in Frame1: 4 regular Frame1 + 3 Frame2 = 7 + 1 unused portal on palette.
    # But we need 9 target matches. This doesn't work with single portal either.
    #
    # I must be misunderstanding the portal/frame mechanic. Let me re-read dbfxrigdqx.

    # Actually wait - I just realized something. Let me re-read more carefully.
    # rfdjlhefnd scans each frame's slots and builds rzbeqaiky.
    # When it encounters a portal (vgszefyyyp) during the WALK, it:
    # 1. Pushes (current_frame, current_slot+1) onto stack
    # 2. Enters the target frame at slot 0
    # When the target frame is done, it pops the stack and continues.
    #
    # The key: portal doesn't "use" a target. It's just a routing mechanism.
    # So with a portal at Frame1.slot2, the walk is:
    # Frame1.slot0 -> target[0]
    # Frame1.slot1 -> target[1]
    # Frame1.slot2 = PORTAL -> ENTER Frame2 (doesn't consume a target)
    # Frame2.slot0 -> target[2]
    # Frame2.slot1 -> target[3]
    # Frame2.slot2 -> target[4]
    # EXIT Frame2 -> back to Frame1
    # Frame1.slot3 -> target[5]
    # Frame1.slot4 -> target[6]
    # Total: 7 targets consumed with 7 non-portal slots.
    #
    # But we have 9 targets and only 7 regular slots (4 Frame1 + 3 Frame2).
    # Unless TWO portals go into Frame1, each entering Frame2 separately:
    # With portals at slot1 and slot3:
    # slot0 -> T[0]
    # slot1 = PORTAL -> Frame2: slot0->T[1], slot1->T[2], slot2->T[3]
    # slot2 -> T[4]
    # slot3 = PORTAL -> Frame2: slot0->T[5], slot1->T[6], slot2->T[7]
    # slot4 -> T[8]
    # Total: 3 Frame1 + 3 Frame2_v1 + 3 Frame2_v2 = 9. ✓
    #
    # But Frame2 items are the same both times!
    # T[1..3] must equal T[5..7]:
    # T[1]=14, T[2]=8, T[3]=8 AND T[5]=8, T[6]=8, T[7]=11
    # 14≠8 at position 1. DOESN'T MATCH.
    #
    # So this STILL doesn't work with the same Frame2 items.
    #
    # Unless... I'm wrong about the target sequence or the portal placement positions.
    # Let me try different portal positions.
    #
    # Portals at slot0 and slot1:
    # slot0=PORTAL: Frame2[T0,T1,T2]
    # slot1=PORTAL: Frame2[T3,T4,T5] (same items read again)
    # slot2->T6, slot3->T7, slot4->T8
    # T[0..2]=[6,14,8] must equal T[3..5]=[8,14,8]. 6≠8. No.
    #
    # Portals at slot0 and slot4:
    # slot0=PORTAL: Frame2[T0,T1,T2]=[6,14,8]
    # slot1->T3=8, slot2->T4=14, slot3->T5=8
    # slot4=PORTAL: Frame2[T6,T7,T8]=[8,11,15]
    # [6,14,8]≠[8,11,15]. No.
    #
    # Portals at slot1 and slot4:
    # slot0->T0=6
    # slot1=PORTAL: Frame2[T1,T2,T3]=[14,8,8]
    # slot2->T4=14, slot3->T5=8
    # slot4=PORTAL: Frame2[T6,T7,T8]=[8,11,15]
    # [14,8,8]≠[8,11,15]. No.
    #
    # Portals at slot2 and slot3:
    # slot0->T0=6, slot1->T1=14
    # slot2=PORTAL: Frame2[T2,T3,T4]=[8,8,14]
    # slot3=PORTAL: Frame2[T5,T6,T7]=[8,8,11]
    # [8,8,14]≠[8,8,11]. No.
    #
    # No combination works with same Frame2 items!
    #
    # I must be misunderstanding something. Let me re-read the code.
    # Maybe the portal enters Frame2 at a DIFFERENT starting slot?
    # Or maybe items change between visits?
    #
    # Actually, wait. Let me re-read the "else" branch at line 975:
    # elif ldwfvtgapk.name == "vgszefyyyp":
    #     if uxncrzlau == 0 and (kmsegkpkh, uxncrzlau) in self.buvfjfmpp[:-1] and (self.buvfjfmpp[-2][1] == 0):
    #         self.sibihgzarf()  # ERROR!
    #         return
    #     ...
    #     amkoiofqhs = next(frame for frame in qaagahahj if frame.pixels[0,0] == portal_inner_color)
    #     self.buvfjfmpp.append((amkoiofqhs, 0))
    #     ...enters frame...
    #
    # When re-entering Frame2 the second time:
    # buvfjfmpp = [(Frame1, slot_after_portal2)]
    # Check: uxncrzlau=0, (Frame2,0) in buvfjfmpp[:-1]?
    # buvfjfmpp[:-1] = []. Empty. So check fails. Portal is entered.
    # ✓ - portal CAN be entered again.
    #
    # But the items in Frame2 are the same. Unless...
    # OH WAIT. I think I see. The rfdjlhefnd function is called ONCE at submit,
    # and it builds rzbeqaiky. But maybe the WALK modifies Frame2's items?
    #
    # Let me re-read dbfxrigdqx carefully:
    # When a match is found (line 920-924):
    # if ldwfvtgapk.pixels[1,1] == bnwkxafnfc.pixels[0,0]:
    #     self.xjxrqgaqw = 0  # triggers fill animation
    # This fills in the target indicator. The ITEM stays in place.
    #
    # So items are not modified. Frame2 has the same items both times.
    #
    # I must be wrong about 9 targets. Let me VERY carefully recount.
    # L5 quhhhthrri sprites from source:
    #  sprites["quhhhthrri"].clone().set_position(1, 1).color_remap(None, 6),    # x=1
    #  sprites["quhhhthrri"].clone().set_position(8, 1).color_remap(None, 14),   # x=8
    #  sprites["quhhhthrri"].clone().set_position(50, 1).color_remap(None, 11),  # x=50
    #  sprites["quhhhthrri"].clone().set_position(15, 1).color_remap(None, 8),   # x=15
    #  sprites["quhhhthrri"].clone().set_position(22, 1).color_remap(None, 8),   # x=22
    #  sprites["quhhhthrri"].clone().set_position(29, 1).color_remap(None, 14),  # x=29
    #  sprites["quhhhthrri"].clone().set_position(36, 1).color_remap(None, 8),   # x=36
    #  sprites["quhhhthrri"].clone().set_position(43, 1).color_remap(None, 8),   # x=43
    #  sprites["quhhhthrri"].clone().set_position(57, 1).color_remap(None, 15),  # x=57
    # That's indeed 9 targets.
    #
    # Hmm, but with 5+3=8 slots, minus portals... I need to re-think.
    #
    # Actually, maybe I should just USE THE GAME MCP to test and observe what happens!
    # That will be much more reliable than trying to reverse-engineer all the obfuscated logic.

    print("\nGenerating solutions using analytical approach + MCP verification...")
    return levels_info

if __name__ == "__main__":
    solve_all_levels()
