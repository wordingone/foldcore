"""
Step 1035 — FT09: Spatial Correlation for Per-Wall Targets

D2-grounded: WHERE → HOW → SPATIAL PAIR → STATE COMPARE → ACT.
R3 hypothesis: The substrate discovers each wall's target color by
spatially pairing wall zones with nearby condition sprites (bsT zones).
Condition sprites don't respond to clicks (zero magnitude) but show
the target color for their paired wall.

Method:
  1. WHERE phase: Mode map discovers ALL zones (walls + sprites)
  2. HOW phase: Click each zone → classify as wall (high mag) or sprite (zero mag)
  3. SPATIAL PAIRING: For each wall, find nearest non-wall zone = condition sprite
  4. STATE COMPARISON: wall color vs paired sprite color. Different → click wall.
  5. ACT: Click only mismatched walls. Re-read state, repeat.
  6. On level transition: re-discover (new level = new layout).

FT09, 5 seeds, 120s cap.
Kill: L1 < 5/5 or L2 = 0/5.
Success: FT09 L2 = first FT09 L2+ from discovered interaction.
"""
import os, sys, time
os.environ['PYTHONUTF8'] = '1'
import logging
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
from scipy.ndimage import label as ndlabel
from collections import Counter

sys.path.insert(0, 'B:/M/the-search')
sys.path.insert(0, 'B:/M/the-search/experiments')

import arc_agi
from arcengine import GameAction, GameState

# ─── Constants ───
WHERE_STEPS = 3000
HOW_CLICKS = 5
REDISCOVER_STEPS = 1500
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 120
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4
MAG_THRESHOLD = 0.1
MAX_ACT_ROUNDS = 100

# ─── Mode map ───
def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r, c = np.arange(64)[:, None], np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1

def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)

def find_isolated_clusters(mode_arr):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        if not mask.any(): continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({
                    'cx_int': int(round(xs.mean())),
                    'cy_int': int(round(ys.mean())),
                    'color': int(color), 'size': sz
                })
    return clusters

def local_zone_change(frame_before, frame_after, cx, cy, radius=ZONE_RADIUS):
    a = np.array(frame_before[0], dtype=np.float32)
    b = np.array(frame_after[0], dtype=np.float32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    return float(np.abs(b[y0:y1, x0:x1] - a[y0:y1, x0:x1]).mean())

def read_zone_color(frame, cx, cy, radius=2):
    arr = np.array(frame[0], dtype=np.int32)
    y0, y1 = max(0, cy - radius), min(64, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(64, cx + radius + 1)
    patch = arr[y0:y1, x0:x1].flatten()
    if len(patch) == 0: return 0
    return Counter(patch.tolist()).most_common(1)[0][0]


# ─── Discovery: classify wall vs sprite ───
def discover(env, action6, obs, n_steps, rng):
    """WHERE+HOW. Returns walls (interactive), sprites (non-interactive), all clusters."""
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    steps = 0

    for _ in range(n_steps):
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue
        update_freq(freq, obs.frame)
        a = rng.randint(N_GRID)
        cx, cy = CLICK_GRID[a]
        obs = env.step(action6, data={"x": cx, "y": cy})
        steps += 1

    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)

    # HOW: measure local magnitude to classify wall vs sprite
    magnitudes = [0.0] * len(clusters)
    for zi, cl in enumerate(clusters):
        for _ in range(HOW_CLICKS):
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue
            frame_before = obs.frame
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1
            if obs and obs.frame and len(obs.frame) > 0:
                mag = local_zone_change(frame_before, obs.frame, cl['cx_int'], cl['cy_int'])
                magnitudes[zi] = max(magnitudes[zi], mag)

    walls = [(zi, clusters[zi]) for zi in range(len(clusters))
             if magnitudes[zi] > MAG_THRESHOLD]
    sprites = [(zi, clusters[zi]) for zi in range(len(clusters))
               if magnitudes[zi] <= MAG_THRESHOLD]

    return walls, sprites, clusters, magnitudes, obs, steps


# ─── Spatial pairing: each wall → nearest sprite ───
def pair_walls_to_sprites(walls, sprites):
    """For each wall, find the nearest sprite by Euclidean distance."""
    pairs = {}  # wall_zi → sprite_zi
    for wzi, wcl in walls:
        best_dist = float('inf')
        best_szi = None
        for szi, scl in sprites:
            dx = wcl['cx_int'] - scl['cx_int']
            dy = wcl['cy_int'] - scl['cy_int']
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_szi = szi
        if best_szi is not None:
            pairs[wzi] = (best_szi, best_dist)
    return pairs


# ─── Spatial-correlation action ───
def spatial_act(env, action6, obs, walls, sprites, clusters, rng,
                budget_steps, time_limit):
    """Compare each wall's color to its paired sprite's color. Click mismatched walls."""
    steps = 0
    max_levels = obs.levels_completed if obs else 0
    level_steps = {}
    rediscover_needed = False
    act_rounds = 0
    n_targeted = 0
    n_wasted = 0

    # Pair walls to sprites
    pairs = pair_walls_to_sprites(walls, sprites)

    while steps < budget_steps and time.time() < time_limit:
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue
        if not walls: break

        act_rounds += 1
        if act_rounds > MAX_ACT_ROUNDS:
            rediscover_needed = True
            break

        # STATE COMPARE: read wall color vs paired sprite color
        mismatched = []
        for wzi, wcl in walls:
            wall_color = read_zone_color(obs.frame, wcl['cx_int'], wcl['cy_int'])
            if wzi in pairs:
                szi, dist = pairs[wzi]
                sprite_cl = clusters[szi]
                sprite_color = read_zone_color(obs.frame, sprite_cl['cx_int'], sprite_cl['cy_int'])
                if wall_color != sprite_color:
                    mismatched.append((wzi, wcl))

        if not mismatched:
            # All walls match their paired sprites — should be solved
            # Try clicking first wall to check if we're actually done
            if walls:
                wzi, wcl = walls[0]
                levels_before = obs.levels_completed
                obs = env.step(action6, data={"x": wcl['cx_int'], "y": wcl['cy_int']})
                steps += 1
                n_wasted += 1
                if obs and obs.levels_completed > levels_before:
                    if obs.levels_completed > max_levels:
                        max_levels = obs.levels_completed
                        level_steps[max_levels] = steps
                    rediscover_needed = True
                    break
                # Undo: click again to toggle back
                obs = env.step(action6, data={"x": wcl['cx_int'], "y": wcl['cy_int']})
                steps += 1
            continue

        # ACT: click mismatched walls
        for wzi, wcl in mismatched:
            if steps >= budget_steps or time.time() >= time_limit: break
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            levels_before = obs.levels_completed
            obs = env.step(action6, data={"x": wcl['cx_int'], "y": wcl['cy_int']})
            steps += 1
            n_targeted += 1

            if obs is None: break
            if obs.levels_completed > levels_before:
                if obs.levels_completed > max_levels:
                    max_levels = obs.levels_completed
                    level_steps[max_levels] = steps
                rediscover_needed = True
                break

        if rediscover_needed: break

    return obs, steps, max_levels, level_steps, rediscover_needed, {
        'act_rounds': act_rounds,
        'targeted_clicks': n_targeted,
        'wasted_clicks': n_wasted,
        'n_pairs': len(pairs),
    }


# ─── Run ───
def run_seed(arc, game_id, seed):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    env = arc.make(game_id)
    action6 = list(GameAction)[6]
    obs = env.reset()
    total_steps = 0
    t0 = time.time()
    max_levels = 0
    level_steps = {}
    all_stats = []

    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        # Discover: classify walls vs sprites
        n_disc = WHERE_STEPS if total_steps == 0 else REDISCOVER_STEPS
        walls, sprites, clusters, mags, obs, s = discover(
            env, action6, obs, n_disc, rng)
        total_steps += s

        if obs and obs.levels_completed > max_levels:
            max_levels = obs.levels_completed

        if not walls:
            break

        # Pair + act
        budget = min(10000, MAX_STEPS - total_steps)
        obs, s, ml, ls, rediscover, stats = spatial_act(
            env, action6, obs, walls, sprites, clusters, rng,
            budget, t0 + TIME_CAP)
        total_steps += s
        stats['n_walls'] = len(walls)
        stats['n_sprites'] = len(sprites)
        if ml > max_levels:
            max_levels = ml
        level_steps.update(ls)
        all_stats.append(stats)

        if not rediscover:
            break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1),
        'stats': all_stats
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if not ft09:
        print("SKIP"); return

    print("=== Step 1035: FT09 Spatial Correlation for Per-Wall Targets ===")
    print(f"WHERE → HOW (wall vs sprite) → SPATIAL PAIR → STATE COMPARE → ACT")
    print(f"Wall=high mag (interactive), Sprite=zero mag (shows target color)")
    print(f"Game: {ft09.game_id}\n")

    results = []
    for seed in range(5):
        r = run_seed(arc, ft09.game_id, seed)
        stats_str = ""
        if r['stats']:
            s = r['stats'][0]
            stats_str = (f"  walls={s['n_walls']}  sprites={s['n_sprites']}  "
                        f"pairs={s['n_pairs']}  targeted={s['targeted_clicks']}  "
                        f"wasted={s['wasted_clicks']}")
        print(f"  s{seed}: L={r['max_levels']}  steps={r['total_steps']}{stats_str}  "
              f"{r['elapsed']}s")
        if r['level_steps']:
            for lvl, step in sorted(r['level_steps'].items()):
                print(f"    L{lvl} @ step {step}")
        results.append(r)

    l1 = sum(1 for r in results if r['max_levels'] >= 1)
    l2 = sum(1 for r in results if r['max_levels'] >= 2)
    l3 = sum(1 for r in results if r['max_levels'] >= 3)
    max_l = max(r['max_levels'] for r in results)
    print(f"\n  FT09: {l1}/5 L1, {l2}/5 L2, {l3}/5 L3+, max_level={max_l}")
    if l2 > 0:
        print(f"  SIGNAL: Spatial correlation produces FT09 L2+!")
    else:
        print(f"  FAIL: No L2+. Spatial pairing may be incorrect.")
    print("\nStep 1035 DONE")


if __name__ == "__main__":
    main()
