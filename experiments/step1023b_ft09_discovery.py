"""
Step 1023b — Mode Map Discovery on FT09

DEBATE EXPERIMENT #2 (prosecution). Tests whether 576's mode map + CC discovery
generalizes to a DIFFERENT click game (FT09).

FT09 has different visual structure from VC33:
- 6 levels, 75 total clicks at exact 2px precision
- Hkx walls (L1-L4) and NTi walls with Lights-Out coupling (L5-L6)
- Clickable targets may be harder to detect (walls vs zones)

Method: Same as 1023 — random exploration + mode map + CC discovery.
No graph. Measure what clusters are discovered and whether any match
FT09's known click targets.

This is a WHERE discovery diagnostic — can the mechanism find FT09's
interaction targets?
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
MODE_WARMUP = 5000
MIN_CLUSTER = 2      # Smaller min for FT09 (walls may be thin)
MAX_CLUSTER = 100
MAX_STEPS = 50_000
TIME_CAP = 60
BURST = 5
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]

# FT09 L1 prescription (from D2): 4 clicks at (7,23)
FT09_L1_CLICKS = [(7, 23)] * 4

# ─── Mode map + CC (from 576) ───

def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]
    c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1

def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)

def find_isolated_clusters(mode_arr, min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if min_sz <= sz <= max_sz:
                ys, xs = np.where(region)
                clusters.append({
                    'cy': float(ys.mean()), 'cx': float(xs.mean()),
                    'color': int(color), 'size': sz,
                    'cy_int': int(round(ys.mean())),
                    'cx_int': int(round(xs.mean())),
                })
    return clusters

# ─── Run one seed ───

def run_seed(arc, game_id, seed):
    np.random.seed(seed * 1000)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]

    freq = np.zeros((64, 64, 16), dtype=np.int32)
    clusters = []
    phase = 'explore'
    nav_cluster = 0
    nav_burst = 0

    obs = env.reset()
    steps = 0
    levels = 0
    l1_step = None
    t0 = time.time()

    while steps < MAX_STEPS:
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset()
            nav_burst = 0
            continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset()
            nav_burst = 0
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            continue

        update_freq(freq, obs.frame)

        if steps == MODE_WARMUP and phase == 'explore':
            mode = compute_mode(freq)
            clusters = find_isolated_clusters(mode)
            if clusters:
                phase = 'navigate'
                # Sort by size (smaller = more likely interactive)
                clusters.sort(key=lambda c: c['size'])

        if phase == 'explore':
            a = int(np.random.randint(N_GRID))
            cx, cy = CLICK_GRID[a]
        else:
            if clusters:
                if nav_burst <= 0:
                    nav_cluster = (nav_cluster + 1) % len(clusters)
                    nav_burst = BURST
                cx = clusters[nav_cluster]['cx_int']
                cy = clusters[nav_cluster]['cy_int']
                nav_burst -= 1
            else:
                cx, cy = CLICK_GRID[0]

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1

        if obs is None:
            break
        if obs.levels_completed > lvls_before:
            levels = obs.levels_completed
            if l1_step is None:
                l1_step = steps

    elapsed = time.time() - t0
    cluster_info = [(c['cx_int'], c['cy_int'], c['color'], c['size']) for c in clusters]

    # Check if any cluster is near FT09 L1 target (7, 23)
    target_hits = []
    for c in clusters:
        dist = ((c['cx_int'] - 7)**2 + (c['cy_int'] - 23)**2)**0.5
        if dist < 5:
            target_hits.append((c['cx_int'], c['cy_int'], dist))

    return {
        'seed': seed, 'levels': levels, 'l1_step': l1_step,
        'steps': steps, 'elapsed': round(elapsed, 1),
        'n_clusters': len(clusters), 'clusters': cluster_info,
        'target_hits': target_hits, 'phase': phase,
    }

# ─── Main ───

def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP — FT09 not found")
        return

    print("=== Step 1023b: FT09 Mode Map Discovery ===")
    print(f"Game: {ft09.game_id}")
    print(f"FT09 L1 target: (7, 23), 4 clicks needed")
    print()

    N_SEEDS = 5
    results = []
    for seed in range(N_SEEDS):
        r = run_seed(arc, ft09.game_id, seed)
        status = f"L{r['levels']}@{r['l1_step']}" if r['levels'] > 0 else "FAIL"
        print(f"  s{seed}: {status:12s}  clusters={r['n_clusters']}  phase={r['phase']}  "
              f"steps={r['steps']}  {r['elapsed']}s")
        if r['clusters']:
            print(f"    clusters: {r['clusters'][:10]}{'...' if len(r['clusters'])>10 else ''}")
        if r['target_hits']:
            print(f"    TARGET HIT: {r['target_hits']}")
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    disc = sum(1 for r in results if r['n_clusters'] > 0)
    hits = sum(1 for r in results if r['target_hits'])

    print(f"\n=== Summary ===")
    print(f"  Wins: {wins}/{N_SEEDS}")
    print(f"  Seeds with clusters: {disc}/{N_SEEDS}")
    print(f"  Seeds with L1 target hit (<5px): {hits}/{N_SEEDS}")

    if wins > 0:
        print(f"  SIGNAL: Mode map discovers FT09 targets AND solves levels")
    elif hits > 0:
        print(f"  PARTIAL: Mode map finds targets but navigation doesn't solve")
    elif disc > 0:
        print(f"  DISCOVERY ONLY: Clusters found but none near FT09 targets")
    else:
        print(f"  KILL: No clusters detected. FT09 visual structure not stable enough for mode map.")

    print("\nStep 1023b DONE")

if __name__ == "__main__":
    main()
