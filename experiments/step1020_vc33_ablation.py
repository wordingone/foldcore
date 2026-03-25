"""
Step 1020 — VC33 Prescription Ablation

Catalog all 7 VC33 level prescriptions (from step0610).
Test zone approximations and click order sensitivity.

R3 hypothesis: coarser zone approximations are insufficient (like FT09),
AND click order matters for VC33 (unlike FT09) because canal lock heights
are adjusted incrementally — position AND sequence both encode information.

VC33 mechanics:
  - display_to_grid(cx,cy) = (cx//2, cy//2) — same 2px precision as FT09
  - Each click adjusts canal lock height at that grid position
  - Win condition: all boats at required heights

Three zone conditions:
  A) Exact: canonical prescription clicks
  B) Zone-8: snap to nearest 8-pixel grid center
  C) Zone-6: k-means k=6 cluster centers

Plus: click ORDER test (shuffle) — VC33 expected to be order-dependent.

KILL criterion: if A passes but B and C fail -> exact coords required
PASS criterion: if B or C passes -> coarser zones sufficient
"""
import sys, os
sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import random
import numpy as np

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

print("=== VC33 Prescription ===")
for lvl_idx, clicks in PRESCRIPTIONS.items():
    distinct = len(set(clicks))
    print(f"  L{lvl_idx+1}: {len(clicks)} clicks, {distinct} distinct positions")

# ─── Zone approximation functions (same as FT09 ablation) ───

def snap_to_8grid(cx, cy):
    gx = round((cx - 4) / 8)
    gy = round((cy - 4) / 8)
    gx = max(0, min(7, gx))
    gy = max(0, min(7, gy))
    return gx * 8 + 4, gy * 8 + 4

def compute_zone6_centers():
    all_pos = []
    for clicks in PRESCRIPTIONS.values():
        all_pos.extend(set(clicks))
    all_pos = np.array(all_pos, dtype=float)
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
print(f"\nZone-6 centers: {[tuple(int(x) for x in c) for c in zone6_centers]}")

def snap_to_zone6(cx, cy):
    pos = np.array([cx, cy], dtype=float)
    dists = [np.linalg.norm(pos - c) for c in zone6_centers]
    best = zone6_centers[np.argmin(dists)]
    return int(round(best[0])), int(round(best[1]))

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
        return True, 0
    done = obs.state in (GameState.GAME_OVER, GameState.WIN)
    lvl = obs.levels_completed - env._levels_offset
    return done, lvl


def play_vc33(click_strategy, verbose=False):
    """Play VC33 with the given click strategy per level.
    click_strategy: function(lvl_idx, ordered_clicks) -> ordered_clicks (may reorder/snap)
    Returns: dict of {lvl_idx: {'done': bool, 'steps': int}}
    """
    results = {}
    for lvl_idx in range(7):
        env.reset(seed=0)

        # Advance past previous levels using EXACT canonical order (not subject to strategy)
        for prev_idx in range(lvl_idx):
            prev_clicks = PRESCRIPTIONS[prev_idx]
            level_done = False
            for cx, cy in prev_clicks:
                _, lvl = do_click(cx, cy)
                if lvl > prev_idx:
                    level_done = True
                    break
            if not level_done:
                # Some levels need extra steps
                for _ in range(10):
                    obs = get_inner().step(ACTION1)
                    if obs is None:
                        break
                    lvl = obs.levels_completed - env._levels_offset
                    if lvl > prev_idx:
                        break

        actual = get_inner()._game.level_index
        if actual != lvl_idx:
            results[lvl_idx] = {'done': False, 'steps': 0, 'error': f'at_level={actual}'}
            continue

        # Apply click strategy
        raw_clicks = PRESCRIPTIONS[lvl_idx]
        modified_clicks = click_strategy(lvl_idx, raw_clicks)

        level_start = actual
        done_flag = False
        steps = 0
        for cx, cy in modified_clicks:
            if verbose:
                g = get_inner()._game.camera.display_to_grid(cx, cy)
                print(f"    L{lvl_idx+1}: ({cx},{cy}) -> grid={g}")
            _, lvl = do_click(cx, cy)
            steps += 1
            if lvl > level_start:
                done_flag = True
                break

        results[lvl_idx] = {'done': done_flag, 'steps': steps}
    return results


# ─── Condition A: Exact ───
print("\n=== Condition A: Exact prescription clicks ===")
results_A = play_vc33(lambda lvl, clicks: clicks)
for lvl_idx, r in results_A.items():
    print(f"  L{lvl_idx+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Condition B: Zone-8 ───
print("\n=== Condition B: Zone-8 (8-pixel snap) ===")
results_B = play_vc33(lambda lvl, clicks: [snap_to_8grid(cx, cy) for cx, cy in clicks])
for lvl_idx, r in results_B.items():
    print(f"  L{lvl_idx+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Zone-8 mismatch analysis ───
print("\nZone-8 mappings (exact -> snapped):")
for lvl_idx in range(7):
    mismatches = []
    for cx, cy in set(PRESCRIPTIONS[lvl_idx]):
        sc, sy = snap_to_8grid(cx, cy)
        g_exact = (cx//2, cy//2)
        g_snapped = (sc//2, sy//2)
        if g_exact != g_snapped:
            mismatches.append(f"({cx},{cy})->({sc},{sy}): grid {g_exact}->{g_snapped} MISS")
    if mismatches:
        print(f"  L{lvl_idx+1}: {len(mismatches)} mismatches: {mismatches[:3]}{'...' if len(mismatches)>3 else ''}")
    else:
        print(f"  L{lvl_idx+1}: all positions hit correctly by 8-pixel snap")

# ─── Condition C: Zone-6 ───
print("\n=== Condition C: Zone-6 (k-means k=6) ===")
results_C = play_vc33(lambda lvl, clicks: [snap_to_zone6(cx, cy) for cx, cy in clicks])
for lvl_idx, r in results_C.items():
    print(f"  L{lvl_idx+1}: {'PASS' if r['done'] else 'FAIL'} ({r['steps']} steps) {r.get('error','')}")

# ─── Condition D: Order shuffle (20 shuffles per level) ───
N_SHUFFLES = 20
RANDOM_SEED = 42
rng = random.Random(RANDOM_SEED)
print(f"\n=== Condition D: Click order shuffle ({N_SHUFFLES} per level) ===")
order_results = {}
for lvl_idx in range(7):
    clicks = list(PRESCRIPTIONS[lvl_idx])
    passes = 0
    for _ in range(N_SHUFFLES):
        shuffled = clicks[:]
        rng.shuffle(shuffled)
        env.reset(seed=0)
        for prev in range(lvl_idx):
            for cx, cy in PRESCRIPTIONS[prev]:
                _, lvl = do_click(cx, cy)
                if lvl > prev: break
        actual = get_inner()._game.level_index
        if actual != lvl_idx:
            continue
        level_start = actual
        done_flag = False
        for cx, cy in shuffled:
            _, lvl = do_click(cx, cy)
            if lvl > level_start:
                done_flag = True
                break
        if done_flag:
            passes += 1
    order_results[lvl_idx] = passes
    verdict = 'ORDER-FREE' if passes == N_SHUFFLES else ('ORDER-DEPENDENT' if passes == 0 else 'PARTIAL')
    print(f"  L{lvl_idx+1}: {passes}/{N_SHUFFLES} shuffles pass -> {verdict}")

# ─── Summary ───
def s(r): return 'PASS' if r.get('done') else 'FAIL'

print("\n=== Summary ===")
print(f"{'Level':<8} {'Exact':>8} {'Zone-8':>8} {'Zone-6':>8} {'Order':>12}")
for lvl_idx in range(7):
    ord_v = f"{order_results[lvl_idx]}/{N_SHUFFLES}"
    print(f"  L{lvl_idx+1}     {s(results_A[lvl_idx]):>8} {s(results_B[lvl_idx]):>8} {s(results_C[lvl_idx]):>8} {ord_v:>12}")

a_all = all(r['done'] for r in results_A.values())
b_all = all(r['done'] for r in results_B.values())
c_all = all(r['done'] for r in results_C.values())
order_sensitive = any(v < N_SHUFFLES for v in order_results.values())

print(f"\nVerdicts:")
print(f"  Zone: {'KILL — exact coords required' if a_all and not b_all and not c_all else ('PASS — zones sufficient' if b_all else 'PARTIAL')}")
print(f"  Order: {'ORDER-DEPENDENT (prescription is a sequence)' if order_sensitive else 'ORDER-FREE (prescription is a set)'}")

print("\nStep 1020 DONE")
