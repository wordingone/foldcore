"""
Step 1019a — FT09 Zone Ablation

R3 hypothesis: exact pixel coordinates are required, OR do coarser zones suffice?

Ablation question: replace exact prescription pixel clicks with CC-discovered zone centers.
If the substrate only discovers approximate zones (not exact walls), can it still solve?

Three conditions:
  A) Exact: click at exact prescription positions (wx*2, wy*2)
  B) Zone-8: snap to nearest 8-pixel grid (gx*8+4, gy*8+4) — standard action grid
  C) Zone-6: k-means k=6 cluster centers across all prescription positions

KILL criterion: if A passes but B or C fails -> coarser zones are insufficient
PASS criterion: if B or C passes -> coarser zones sufficient (substrate doesn't need exact coords)

FT09 mechanics confirmed:
  - display_to_grid(cx,cy) = (cx//2, cy//2) — 2px precision
  - Wall at (wx,wy) hit by any click in pixel range [wx*2, wx*2+1] x [wy*2, wy*2+1]
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np

import arcagi3
from util_ft09_level_solver import solve_level

# ─── Get prescription clicks per level ───
prescriptions = {}
for lvl_idx in range(6):
    name, gqb, bst, conflicts, clicks = solve_level(lvl_idx)
    prescriptions[lvl_idx] = clicks  # list of (cx,cy), may have repeats for multi-click walls

print("=== FT09 Prescription ===")
for lvl_idx, clicks in prescriptions.items():
    print(f"  L{lvl_idx+1}: {len(clicks)} clicks, {len(set(clicks))} distinct positions")
    print(f"    {sorted(set(clicks))}")

# ─── Zone approximation functions ───

def snap_to_8grid(cx, cy):
    """Snap to nearest 8-pixel grid center (gx*8+4, gy*8+4)."""
    gx = round((cx - 4) / 8)
    gy = round((cy - 4) / 8)
    gx = max(0, min(7, gx))
    gy = max(0, min(7, gy))
    return gx * 8 + 4, gy * 8 + 4

def compute_zone6_centers():
    """K-means k=6 on all prescription positions across all levels."""
    all_pos = []
    for clicks in prescriptions.values():
        all_pos.extend(set(clicks))
    all_pos = np.array(all_pos, dtype=float)

    # Simple k-means
    np.random.seed(42)
    centers = all_pos[np.random.choice(len(all_pos), 6, replace=False)]
    for _ in range(100):
        dists = np.array([[np.linalg.norm(p - c) for c in centers] for p in all_pos])
        assignments = dists.argmin(axis=1)
        new_centers = np.array([
            all_pos[assignments == k].mean(axis=0) if (assignments == k).any() else centers[k]
            for k in range(6)
        ])
        if np.allclose(centers, new_centers, atol=0.1):
            break
        centers = new_centers
    return centers

zone6_centers = compute_zone6_centers()
print(f"\nZone-6 centers (k-means k=6): {[tuple(int(x) for x in c) for c in zone6_centers]}")

def snap_to_zone6(cx, cy):
    """Snap to nearest Zone-6 cluster center."""
    pos = np.array([cx, cy], dtype=float)
    dists = [np.linalg.norm(pos - c) for c in zone6_centers]
    best = zone6_centers[np.argmin(dists)]
    return int(round(best[0])), int(round(best[1]))

# ─── Play FT09 with a given click strategy ───

def play_ft09(click_strategy, verbose=False):
    """Play FT09 prescription with the given click strategy.
    click_strategy: function(lvl_idx, exact_cx, exact_cy) -> (actual_cx, actual_cy)
    Returns: dict of {lvl_idx: {'done': bool, 'steps': int}}
    """
    from arcengine import GameAction, GameState
    GA_LIST = list(GameAction)[1:]  # ACTION1..ACTION7
    ACTION6 = GA_LIST[5]  # ACTION6
    ACTION1 = GA_LIST[0]  # ACTION1 (null)

    env = arcagi3.make('FT09')
    env.reset(seed=0)

    def get_inner():
        """Always return fresh reference (env._env may be recreated after reset)."""
        return env._env

    def do_click(cx, cy):
        obs = get_inner().step(ACTION6, data={"x": cx, "y": cy})
        if obs is None:
            return None, 0.0, True, {'level': 0}
        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        lvl = obs.levels_completed - env._levels_offset
        return obs.frame, 0.0, done, {'level': lvl}

    def do_null():
        obs = get_inner().step(ACTION1)
        if obs is None:
            return None, 0.0, True, {'level': 0}
        done = obs.state in (GameState.GAME_OVER, GameState.WIN)
        lvl = obs.levels_completed - env._levels_offset
        return obs.frame, 0.0, done, {'level': lvl}

    results = {}
    for lvl_idx in range(6):
        # Reset and advance through all previous levels
        env.reset(seed=0)

        for prev_idx in range(lvl_idx):
            prev_clicks = prescriptions[prev_idx]
            level_done = False
            for ec_x, ec_y in prev_clicks:
                ac_x, ac_y = click_strategy(prev_idx, ec_x, ec_y)
                _, _, _, info = do_click(ac_x, ac_y)
                if info.get('level', 0) > prev_idx:
                    level_done = True
                    break
            if not level_done:
                for _ in range(20):
                    _, _, _, info = do_null()
                    if info.get('level', 0) > prev_idx:
                        break

        # Check we're at the right level
        actual_level = get_inner()._game.level_index
        if actual_level != lvl_idx:
            results[lvl_idx] = {'done': False, 'steps': 0, 'error': f'at_level={actual_level}'}
            continue

        # Play the current level
        clicks = prescriptions[lvl_idx]
        level_start = actual_level
        steps = 0
        done_flag = False

        for ec_x, ec_y in clicks:
            ac_x, ac_y = click_strategy(lvl_idx, ec_x, ec_y)
            if verbose:
                grid = get_inner()._game.camera.display_to_grid(ac_x, ac_y)
                print(f"    L{lvl_idx+1}: exact=({ec_x},{ec_y}) -> approx=({ac_x},{ac_y}) -> grid={grid}")
            _, _, _, info = do_click(ac_x, ac_y)
            steps += 1
            if info.get('level', 0) > level_start:
                done_flag = True
                break

        # Extra null steps for animation/level detection
        if not done_flag:
            for _ in range(20):
                _, _, _, info = do_null()
                steps += 1
                if info.get('level', 0) > level_start:
                    done_flag = True
                    break

        results[lvl_idx] = {'done': done_flag, 'steps': steps}

    return results


# ─── Test Condition A: Exact ───
print("\n=== Condition A: Exact prescription clicks ===")
results_A = play_ft09(lambda lvl, cx, cy: (cx, cy), verbose=True)
for lvl_idx, r in results_A.items():
    status = 'PASS' if r['done'] else 'FAIL'
    print(f"  L{lvl_idx+1}: {status} ({r['steps']} steps) {r.get('error','')}")

# ─── Test Condition B: Zone-8 (8-pixel grid snap) ───
print("\n=== Condition B: Zone-8 (8-pixel grid snap) ===")
results_B = play_ft09(lambda lvl, cx, cy: snap_to_8grid(cx, cy))
for lvl_idx, r in results_B.items():
    status = 'PASS' if r['done'] else 'FAIL'
    print(f"  L{lvl_idx+1}: {status} ({r['steps']} steps) {r.get('error','')}")

# ─── Show Zone-8 mappings ───
print("\nZone-8 mappings (exact -> snapped):")
for lvl_idx in range(6):
    mismatches = []
    for cx, cy in set(prescriptions[lvl_idx]):
        sc, sy = snap_to_8grid(cx, cy)
        g_exact = (cx//2, cy//2)
        g_snapped = (sc//2, sy//2)
        if g_exact != g_snapped:
            mismatches.append(f"({cx},{cy})->({sc},{sy}): grid {g_exact}->{g_snapped} MISS")
    if mismatches:
        print(f"  L{lvl_idx+1} mismatches: {mismatches}")
    else:
        print(f"  L{lvl_idx+1}: all walls hit correctly by 8-pixel snap")

# ─── Test Condition C: Zone-6 (k-means cluster centers) ───
print("\n=== Condition C: Zone-6 (k-means k=6 cluster centers) ===")
results_C = play_ft09(lambda lvl, cx, cy: snap_to_zone6(cx, cy))
for lvl_idx, r in results_C.items():
    status = 'PASS' if r['done'] else 'FAIL'
    print(f"  L{lvl_idx+1}: {status} ({r['steps']} steps) {r.get('error','')}")

# ─── Summary ───
print("\n=== Summary ===")
print(f"{'Level':<8} {'Exact (A)':>10} {'Zone-8 (B)':>12} {'Zone-6 (C)':>12}")
for lvl_idx in range(6):
    def s(r): return 'PASS' if r.get('done') else 'FAIL'
    print(f"  L{lvl_idx+1}     {s(results_A[lvl_idx]):>10} {s(results_B[lvl_idx]):>12} {s(results_C[lvl_idx]):>12}")

a_all = all(r['done'] for r in results_A.values())
b_all = all(r['done'] for r in results_B.values())
c_all = all(r['done'] for r in results_C.values())
print(f"\nVerdict:")
print(f"  A (exact): {'ALL PASS' if a_all else 'PARTIAL'}")
print(f"  B (zone-8): {'ALL PASS' if b_all else 'PARTIAL/FAIL'}")
print(f"  C (zone-6): {'ALL PASS' if c_all else 'PARTIAL/FAIL'}")

if a_all and not b_all and not c_all:
    print("  KILL: exact coordinates required, zones insufficient")
elif a_all and b_all:
    print("  PASS: 8-pixel zone approximation sufficient")
    if c_all:
        print("  STRONG PASS: 6-zone approximation also sufficient")
else:
    print("  UNEXPECTED: baseline not fully passing — check prescription solver")

print("\nStep 1019a DONE")
