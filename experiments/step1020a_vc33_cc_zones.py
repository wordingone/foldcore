"""
Step 1020a — VC33 CC Zone Ablation

Hypothesis: Zone identity is sufficient for VC33 — exact pixel coords are purgable.

Method: Replace hardcoded VC33 click coords with Connected-Component zone centers
discovered from the actual game frame. CC detection: find contiguous same-color
regions in the 64x64 game frame, compute their centroids. Map each prescribed click
to the nearest CC centroid.

Three conditions:
  A) Exact: canonical prescribed coords (baseline)
  B) CC zones: nearest CC centroid from initial frame
  C) Grid-8 snap: 8-pixel grid snap (same as 1020 Condition B — verified KILL)

KILL: <7/7 on B → CC zone discovery insufficient (exact coords required)
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from collections import deque

import arcagi3
from arcengine import GameAction, GameState

# ─── VC33 solutions from step0610 (verified working) ───
GA=(0,27); GB=(0,33); GC=(24,33); GD=(24,27)
S1=(6,30); S2=(30,30); NOP_L5=(60,60)
SOLN_5 = [GA,GD,GD,GD,S1,NOP_L5,GB,GB,GC,GC,GC,GC,GC,GC,S2,NOP_L5,GD,GD,GD,GD,GD,GD]
SOLN_6_EXACT = [
    (24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),
    (24,32),(22,38),
    (20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),(20,8),
    (42,8),(40,16),
    (24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),(24,8),
    (38,32),(38,32),(42,8),(22,38),(40,38),
    (20,32),(20,32),(20,32),(20,32),(20,32),(20,32),
    (42,8),(42,8),(42,8),(42,8),
]
PRESCRIPTIONS = {
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
    5: SOLN_5,
    6: SOLN_6_EXACT,
}

# ─── CC zone detection ───

def find_cc_centroids(frame, min_size=4):
    """Find connected components in frame and return their centroids.

    frame: array, possibly (1, H, W) palette indices or (H, W, C) RGB
    Returns: list of (cx, cy, color_key, size) for each CC above min_size.
    Centroid (cx, cy) is in display coords (0-63 range matching click coords).
    """
    frame = np.array(frame)
    # Normalize to (H, W) with integer color per pixel
    # VC33 frames: (C, H, W) where C>=1 — use channel 0 as color key
    if frame.ndim == 3:
        # Could be (C, H, W) or (H, W, C)
        if frame.shape[0] <= 4 and frame.shape[1] >= 32:
            # (C, H, W) format — take channel 0
            img = frame[0].astype(np.int32)
        else:
            # (H, W, C) format — take channel 0
            img = frame[:, :, 0].astype(np.int32)
    elif frame.ndim == 2:
        img = frame.astype(np.int32)
    else:
        img = frame.reshape(-1, frame.shape[-1]).astype(np.int32)

    H, W = img.shape
    visited = np.zeros((H, W), dtype=bool)
    centroids = []

    for y0 in range(H):
        for x0 in range(W):
            if visited[y0, x0]:
                continue
            color = img[y0, x0]
            queue = deque([(y0, x0)])
            visited[y0, x0] = True
            pixels = [(y0, x0)]

            while queue:
                y, x = queue.popleft()
                for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx]:
                        if img[ny, nx] == color:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                            pixels.append((ny, nx))

            if len(pixels) >= min_size:
                # Centroid in DISPLAY coords: frame pixel (row, col) = display (col, row)
                # since display_to_grid(cx, cy) = (cx//2, cy//2) where cx=col, cy=row
                ccy = sum(p[0] for p in pixels) / len(pixels)  # avg row = display y
                ccx = sum(p[1] for p in pixels) / len(pixels)  # avg col = display x
                centroids.append((ccx, ccy, int(color), len(pixels)))

    return centroids


def snap_to_cc(cx, cy, centroids):
    """Snap display click (cx,cy) to nearest CC centroid."""
    if not centroids:
        return cx, cy
    best_dist = float('inf')
    best = (cx, cy)
    for ccx, ccy, _, _ in centroids:
        d = (cx - ccx)**2 + (cy - ccy)**2
        if d < best_dist:
            best_dist = d
            best = (int(round(ccx)), int(round(ccy)))
    return best


def snap_to_8grid(cx, cy):
    gx = round((cx - 4) / 8)
    gy = round((cy - 4) / 8)
    gx = max(0, min(7, gx))
    gy = max(0, min(7, gy))
    return gx * 8 + 4, gy * 8 + 4


# ─── Play VC33 ───

GA_LIST = list(GameAction)[1:]
ACTION6 = GA_LIST[5]
ACTION1 = GA_LIST[0]

env = arcagi3.make('VC33')
env.reset(seed=0)


def get_inner(): return env._env


def do_click(cx, cy):
    obs = get_inner().step(ACTION6, data={"x": cx, "y": cy})
    if obs is None:
        return True, 0, None
    done = obs.state in (GameState.GAME_OVER, GameState.WIN)
    lvl = obs.levels_completed - env._levels_offset
    return done, lvl, obs  # return full obs


# Get initial frames for each level (for CC detection)
print("=== Extracting CC zones from initial frames ===")
level_frames = {}
level_centroids = {}

# To get L_i frame, we need to advance to that level
env.reset(seed=0)
# Get initial frame via null ACTION1 step
obs_init = get_inner().step(ACTION1)
frame = np.array(obs_init.frame) if (obs_init is not None and obs_init.frame is not None) else None
if frame is not None:
    print(f"  Frame shape: {frame.shape}, dtype={frame.dtype}, range=[{frame.min()},{frame.max()}]")

for lvl_idx in range(7):
    # Capture frame at start of this level
    if frame is not None:
        level_frames[lvl_idx] = frame
        centroids = find_cc_centroids(frame, min_size=4)
        level_centroids[lvl_idx] = centroids
        print(f"  L{lvl_idx+1}: frame={frame.shape}, {len(centroids)} CCs (sizes: {sorted([c[3] for c in centroids], reverse=True)[:10]})")
    else:
        print(f"  L{lvl_idx+1}: no frame")
        level_centroids[lvl_idx] = []

    # Advance to next level using click actions
    if lvl_idx < 6:
        for cx, cy in PRESCRIPTIONS[lvl_idx]:
            _, lvl, obs_new = do_click(cx, cy)
            if obs_new is not None and getattr(obs_new, 'frame', None) is not None:
                frame = np.array(obs_new.frame)
            if lvl > lvl_idx:
                break


# Show CC snap mappings for each level
print("\n=== CC Zone Mappings (exact -> CC centroid) ===")
for lvl_idx in range(7):
    centroids = level_centroids[lvl_idx]
    distinct_clicks = sorted(set(PRESCRIPTIONS[lvl_idx]))
    mismatches = []
    for cx, cy in distinct_clicks:
        sc, sy = snap_to_cc(cx, cy, centroids)
        g_exact = (cx//2, cy//2)
        g_snap = (sc//2, sy//2)
        if g_exact != g_snap:
            dist = ((cx-sc)**2 + (cy-sy)**2)**0.5
            mismatches.append(f"({cx},{cy})->({sc},{sy}) dist={dist:.1f}: grid {g_exact}->{g_snap}")
    if mismatches:
        print(f"  L{lvl_idx+1}: {len(mismatches)} mismatches: {mismatches[:3]}{'...' if len(mismatches)>3 else ''}")
    else:
        print(f"  L{lvl_idx+1}: ALL {len(distinct_clicks)} positions hit correctly by CC snap")


def play_vc33_condition(click_strategy):
    """Play VC33 with a given click mapping strategy.
    click_strategy: function(lvl_idx, cx, cy) -> (actual_cx, actual_cy)
    """
    results = {}
    for lvl_idx in range(7):
        env.reset(seed=0)
        # Advance past previous levels using EXACT canonical clicks
        for prev_idx in range(lvl_idx):
            for cx, cy in PRESCRIPTIONS[prev_idx]:
                done_f, lvl, _ = do_click(cx, cy)
                if lvl > prev_idx:
                    break

        actual = get_inner()._game.level_index
        if actual != lvl_idx:
            results[lvl_idx] = {'done': False, 'steps': 0, 'error': f'at_level={actual}'}
            continue

        level_start = actual
        done_flag = False
        steps = 0
        for cx, cy in PRESCRIPTIONS[lvl_idx]:
            ac_x, ac_y = click_strategy(lvl_idx, cx, cy)
            done_f, lvl, _ = do_click(ac_x, ac_y)
            steps += 1
            if lvl > level_start:
                done_flag = True
                break
        results[lvl_idx] = {'done': done_flag, 'steps': steps}
    return results


# ─── Condition A: Exact ───
print("\n=== Condition A: Exact prescription ===")
results_A = play_vc33_condition(lambda lvl, cx, cy: (cx, cy))
for i, r in results_A.items():
    print(f"  L{i+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Condition B: CC zone centers ───
print("\n=== Condition B: CC zone center snap ===")
results_B = play_vc33_condition(
    lambda lvl, cx, cy: snap_to_cc(cx, cy, level_centroids[lvl])
)
for i, r in results_B.items():
    print(f"  L{i+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Condition C: Grid-8 snap ───
print("\n=== Condition C: Grid-8 snap ===")
results_C = play_vc33_condition(lambda lvl, cx, cy: snap_to_8grid(cx, cy))
for i, r in results_C.items():
    print(f"  L{i+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Summary ───
def s(r): return 'PASS' if r.get('done') else 'FAIL'
print("\n=== Summary ===")
print(f"{'Level':<8} {'Exact':>8} {'CC-zone':>10} {'Grid-8':>8}")
for i in range(7):
    print(f"  L{i+1}     {s(results_A[i]):>8} {s(results_B[i]):>10} {s(results_C[i]):>8}")

a_all = all(r['done'] for r in results_A.values())
b_all = all(r['done'] for r in results_B.values())
c_all = all(r['done'] for r in results_C.values())
print(f"\nVerdict:")
print(f"  A (exact): {'ALL PASS' if a_all else 'PARTIAL'}")
print(f"  B (CC-zone): {'ALL PASS — CC zones sufficient!' if b_all else 'KILL — CC zones insufficient, exact coords required'}")
print(f"  C (grid-8): {'ALL PASS' if c_all else 'KILL — grid-8 insufficient'}")

print("\nStep 1020a DONE")
