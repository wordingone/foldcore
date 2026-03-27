"""Play SB26 level by level, observing frames and testing actions."""
import sys, os, json, numpy as np
os.environ["PYTHONUTF8"] = "1"
sys.path.insert(0, "B:/M/the-search")
from arc_agi import ArcAgiSession

def click(x, y): return 7 + y * 64 + x
SUBMIT = 4
UNDO = 6

def act(s, action_id):
    if action_id >= 7:
        x = (action_id - 7) % 64; y = (action_id - 7) // 64
        return s.act(6, x=x, y=y)
    elif action_id == 4:
        return s.act(5)
    elif action_id == 6:
        return s.act(7)
    return s.act(action_id + 1)

def run_actions(s, actions):
    for a in actions:
        r = act(s, a)
    return s.observe()

def observe_frame(s):
    """Get the current frame as numpy array and analyze it."""
    obs = s.observe()
    frame = np.array(obs.get('frame', []))
    return obs, frame

# L1 solution (verified)
L1 = [click(36,59), click(23,30), click(20,59), click(29,30),
      click(44,59), click(35,30), click(28,59), click(41,30), SUBMIT]

# L2 solution (verified)
L2 = [3750,1372,3736,1378,3729,2268,3764,2274,3743,2280,3771,2286,3757,1390,4]

s = ArcAgiSession("sb26")
print("SB26 started")

# Run L1
print("Running L1...")
obs = run_actions(s, L1)
print(f"After L1: levels={obs['levels_completed']}")

# Run L2
print("Running L2...")
obs = run_actions(s, L2)
print(f"After L2: levels={obs['levels_completed']}")

# Now at L3 - observe
print("\n=== LEVEL 3 ===")
obs = s.observe()
print(f"State: {obs['state']}, levels_completed: {obs['levels_completed']}")

# L3 analysis (from source):
# Targets: [8, 14, 15, 11, 6, 9, 12]
# Frame1 (nyqgqtujsa5 at 15,19): 5 slots at x=17,23,29,35,41 y=21
# Frame2 (jvkvqzheok2 at 15,31) c14: 2 slots at x=17,23 y=33
# Frame3 (jvkvqzheok2 at 33,31) c9: 2 slots at x=35,41 y=33
# Portal at (23,21) c14 -> Frame2
# Portal at (35,21) c9 -> Frame3
# Slots: (17,21), (29,21), (41,21) in F1; (17,33),(23,33) in F2; (35,33),(41,33) in F3
# Walk: F1.s0(17,21)->c8, portal(23,21)->F2, F2.s0(17,33)->c14, F2.s1(23,33)->c15,
#       back, F1.s2(29,21)->c11, portal(35,21)->F3, F3.s0(35,33)->c6, F3.s1(41,33)->c9,
#       back, F1.s4(41,21)->c12
# Palette: c8@(50,56), c15@(22,56), c14@(15,56), c12@(8,56), c6@(43,56), c9@(36,56), c11@(29,56)

# Use center of sprite (+3) for both palette and slots
# Palette click = (x+3, y+3), Slot click = (x+3, y+3)
L3 = [
    click(53, 59), click(20, 24),  # c8@(50,56) -> slot(17,21)
    click(18, 59), click(20, 36),  # c14@(15,56) -> slot(17,33)
    click(25, 59), click(26, 36),  # c15@(22,56) -> slot(23,33)
    click(32, 59), click(32, 24),  # c11@(29,56) -> slot(29,21)
    click(46, 59), click(38, 36),  # c6@(43,56) -> slot(35,33)
    click(39, 59), click(44, 36),  # c9@(36,56) -> slot(41,33)
    click(11, 59), click(44, 24),  # c12@(8,56) -> slot(41,21)
    SUBMIT
]

print("Running L3...")
obs = run_actions(s, L3)
print(f"After L3: levels={obs['levels_completed']}, state={obs['state']}")

if obs['levels_completed'] < 3:
    print("L3 FAILED! Let me debug...")
    # Try with +2 offset instead
    s2 = ArcAgiSession("sb26")
    run_actions(s2, L1 + L2)

    L3b = [
        click(52, 58), click(19, 23),  # c8@(50,56)+2 -> slot(17,21)+2
        click(17, 58), click(19, 35),  # c14@(15,56)+2 -> slot(17,33)+2
        click(24, 58), click(25, 35),  # c15@(22,56)+2 -> slot(23,33)+2
        click(31, 58), click(31, 23),  # c11@(29,56)+2 -> slot(29,21)+2
        click(45, 58), click(37, 35),  # c6@(43,56)+2 -> slot(35,33)+2
        click(38, 58), click(43, 35),  # c9@(36,56)+2 -> slot(41,33)+2
        click(10, 58), click(43, 23),  # c12@(8,56)+2 -> slot(41,21)+2
        SUBMIT
    ]
    print("Trying L3 with +2 offset...")
    obs = run_actions(s2, L3b)
    print(f"After L3b: levels={obs['levels_completed']}, state={obs['state']}")
    if obs['levels_completed'] >= 3:
        L3 = L3b
        s = s2
    else:
        print("L3b also failed! Need to investigate.")
        sys.exit(1)

# ===== LEVEL 4 =====
print("\n=== LEVEL 4 ===")
# Targets: [11, 8, 14, 9, 6, 12, 15]
# Frame1 (nyqgqtujsa5 at 15,18): 5 slots at x=17,23,29,35,41 y=20
# Frame2 (pcrvmjfjzg3 at 21,32) c14: 3 slots at x=23,29,35 y=34
# Item in Frame2: c14 at (23,34) (already placed in slot0)
# Portal on palette: vgszefyyyp c14 at (50,56)
# Other palette: c8@(29,56), c6@(15,56), c15@(36,56), c12@(22,56), c11@(8,56), c9@(43,56)
#
# Walk with portal at F1.s2(29,20):
# s0(17,20)->c11, s1(23,20)->c8, portal->F2, F2.s0(23,34)->c14(pre-placed),
# F2.s1(29,34)->c9, F2.s2(35,34)->c6, back, s3(35,20)->c12, s4(41,20)->c15

# Using +2 for click offset (confirmed from L2 known solution)
off = 2
L4 = [
    click(8+off, 56+off), click(17+off, 20+off),   # c11 -> slot(17,20)
    click(29+off, 56+off), click(23+off, 20+off),   # c8 -> slot(23,20)
    click(50+off, 56+off), click(29+off, 20+off),   # portal(c14) -> slot(29,20)
    click(43+off, 56+off), click(29+off, 34+off),   # c9 -> slot(29,34)
    click(15+off, 56+off), click(35+off, 34+off),   # c6 -> slot(35,34)
    click(22+off, 56+off), click(35+off, 20+off),   # c12 -> slot(35,20)
    click(36+off, 56+off), click(41+off, 20+off),   # c15 -> slot(41,20)
    SUBMIT
]

print("Running L4...")
obs = run_actions(s, L4)
print(f"After L4: levels={obs['levels_completed']}, state={obs['state']}")

# ===== LEVEL 5 =====
print("\n=== LEVEL 5 ===")
# This is the tricky one with 9 targets and circular portals
# After more analysis: I think both portals go to Frame2 and the walk visits twice.
# BUT the items must match. Since they can't, maybe I'm wrong about the target count.
#
# Actually, let me reconsider: maybe the SAME item can satisfy different targets
# because the walk doesn't "consume" items, just checks them.
# The item stays in the slot, and different targets check it.
# Each target just needs the slot color to match. The slot is read, not consumed.
# So when Frame2 is visited twice, the same 3 items match 3 targets each time,
# as long as target colors match the same Frame2 items.
#
# For this to work: targets at portal1+0..2 must EQUAL targets at portal2+0..2.
# Target: [6, 14, 8, 8, 14, 8, 8, 11, 15]
#
# Need two consecutive 3-element blocks from targets that match.
# Check: [14,8,8] at pos 1-3 and [8,8,11] at pos 4-6? No.
# Hmm... what if the portals aren't in Frame1?
#
# Actually wait — what if Level 5's Frame1 doesn't have 5 slots worth of items,
# but some slots are pre-filled? No, all items are at y=56.
#
# What if Level 5 uses NESTED portals? Frame1 has a portal to Frame2, and Frame2
# has its own portal back? But Frame2 is pcrvmjfjzg3 (3 slots) and its color is 9.
# There are 2 portals of color 9 on the palette. One goes in Frame1, one goes in Frame2?
# But Frame2 portal color 9 would route to... itself (Frame2 color=9)? That's a self-loop.
#
# Or maybe Frame2 portal routes to Frame1 if Frame1 has color 9? No, Frame1 is color 8.
#
# OK let me just try playing L5 interactively to figure it out.
# First, let me see if we're at L5.

if obs['levels_completed'] >= 4:
    print("At L5, observing...")
    # I'll save what I know and continue later
else:
    print(f"Not at L5 yet, only {obs['levels_completed']} levels done")

# Save progress so far
all_acts = L1 + L2 + L3 + L4
print(f"\nTotal actions so far: {len(all_acts)}")
print(f"Levels completed: {obs['levels_completed']}")
