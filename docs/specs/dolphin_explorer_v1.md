# Full Graph Explorer Spec (dolphin-equivalent)

Status: DRAFT — send after forward nav results (mail 4075)
Send if: K ≤ 5 (forward nav alone doesn't beat argmin baseline)
Reference: github.com/dolphin-in-a-coma/arc-agi-3-just-explore

## Constitutional Audit

| Rule | Check | Pass? |
|------|-------|-------|
| R1 | No external loss/reward. Segmentation uses color clustering (internal). Priority groups are perceptual priors, not rewards. | YES |
| R2 | Graph growth IS the computation. No separate optimizer. Transitions recorded as byproduct of acting. | YES |
| R3 | Graph grows with experience. Behavior changes as graph structure changes (new frontiers, new paths). | YES |
| R4 | Second-exposure testable: does graph from try1 help try2? | YES (testable) |
| R5 | Game is ground truth. | YES |
| R6 | Each component removable: test without segmentation, without masking, without BFS. | YES (testable) |

## Mechanism

### 1. Frame Processor (exact constants from dolphin source)

**Connected component segmentation:**
- Flood-fill BFS, 4-connectivity, on the 64x64 frame (values 0-15, ARC palette)
- Per segment: bounding_box (x1,y1,x2,y2), color (int), area (px count), is_rectangle (bool)
- Twin detection: segments with identical area, rectangle status, AND color

**Status bar detection and masking:**
- Check each segment: is bounding box fully within 3px of any screen edge?
- If on edge AND aspect ratio >= 5:1 → status bar line → mask
- If on edge AND aspect ratio < 5:1 → check for twins on same edge
  - If >= 3 twins (same area + color + rectangle) → status bar dots → mask all
  - If < 3 twins → NOT a status bar → keep
- Masked segments set to color 16

**Frame hashing:**
- Hash the MASKED frame (status bars → color 16)
- Pack two 4-bit pixels per byte (values 0-15)
- Blake2B 128-bit digest with shape embedded in person tag
- Same game state always produces same hash regardless of step counter

### 2. Action Space (from dolphin source)

**KB actions:** Use `available_actions` from game — typically ACTION1-ACTION5. Add to G0 (highest priority). Dolphin maps ACTION1-5 as KB; ACTION6 = click.

**Click actions (when ACTION6 in available_actions):**
- Each segment = one click action (click random pixel within segment)
- NOT a grid — actual object-based clicking
- Priority groups (exact dolphin thresholds):
  - **Salient colors:** {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  - **Non-salient:** {0, 1, 2, 3, 4, 5}
  - **Medium width:** 2-32 px in BOTH x AND y dimensions
  - G0: salient AND medium → "likely buttons"
  - G1: medium but NOT salient → "medium non-colored elements"
  - G2: salient but NOT medium → "too small/large colored elements"
  - G3: NOT status_bar AND none of above → "everything else"
  - G4: status_bar (color 16) → lowest priority

**Typical action count:** 5 KB + 10-25 segments = 15-30 total (vs our 7 KB-only or 71 with 8x8 grid)

### 3. Graph Explorer

**Data structures:**
- Nodes: dict[frame_hash → NodeInfo]
- Edges: (node, action_idx) → target_node, with result/distance/group metadata
- Adjacency + REVERSE adjacency for backward BFS
- Frontier: set of nodes with untested actions in active group

**Distance computation (backward BFS from frontier):**
1. Set all frontier nodes to distance 0
2. BFS backward through reverse edges
3. Each node gets distance = shortest path to nearest frontier node
4. Each edge stores distance to its target's nearest frontier

**Action selection:**
- At frontier (node has untested actions): pick random untested action from highest active group
- Not at frontier: pick edge with lowest distance to frontier (navigate toward frontier)

**Priority group advancement:**
- When current node has distance=INFINITY (no reachable frontier in current group)
- Advance active_group += 1
- Recalculate frontier with new group's untested actions
- Rebuild distances

**Transition recording:**
- On each action: record (node, action, result, target_node)
- Update adjacency + reverse adjacency
- Recalculate frontier and distances if new node discovered

### 4. Game Loop

```
While steps < 50K and time < 5 min:
    frame = observe()
    masked_frame = mask_status_bars(segment(frame))
    node = hash(masked_frame)

    if node not in graph:
        segments = segment(masked_frame)
        actions = KB_actions + segment_click_actions(segments)
        add_node(node, actions, priority_groups)

    if node has untested actions in active group:
        action = random untested action from highest group
    else:
        next_hop = get_next_hop(node)  # lowest-distance edge
        action = edge_to(next_hop)

    result = env.step(action)
    record_test(node, action, result)

    if win: done
    if game_over: reset (don't clear graph)
```

### 5. Kill Criteria

K ≤ 5 on API (25 games, 5-min budget) → doesn't beat state-argmin baseline → KILL
K > 5 → CONTINUE (iterate: tune priority heuristics, add macro-actions)

Note: at 5-min budget with 60ms API latency, ~5K steps per game. The priority system matters MORE at low step counts — smart exploration > random exploration. dolphin's 30/52 levels proves the approach works. If K≤5 at 5K steps, try 50K steps (matching baseline conditions) before killing.

### 6. Forward Test

Most likely failure mode: segmentation errors (wrong segments, wrong priorities) leading to wasted actions on non-interactive regions. This is NEW — never tested. Different from counting (argmin) and navigation (forward nav). Passes forward test.

### 7. Wiring

```
observation (64x64x16)
    → flood-fill CC segmentation → segments[]
    → status bar detection → mask
    → mask frame → hash → node lookup
    → graph explorer: choose action (untested at frontier / navigate to frontier)
    → env.step(action)
    → record transition in graph
    → loop
```
