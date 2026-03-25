"""
Step 1034 — VC33 L3 Diagnostic: What does L3 require that L2 doesn't?

D2-grounded diagnostic. Play 1032 substrate (majority-vote state tracking) to L3.
At each level, log zone structure, magnitudes, causal matrix.
Compare L1/L2/L3 to understand what changes.

Method:
  1. Use 1032 substrate to reach L3 (it gets L2 reliably)
  2. At each level transition, log:
     - Number of discovered zones, positions
     - Response magnitudes per zone
     - Causal matrix (click zone A → measure change at zone B)
     - Zone colors and majority-vote target
  3. At L3: try click-ALL to see if targeting vs discovery is the bottleneck
  4. Compare L1/L2/L3 zone structures

VC33 only, 3 seeds, 5 min cap.
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
TIME_CAP = 300
N_GRID = 64
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
ZONE_RADIUS = 4
MAG_THRESHOLD = 0.1
MAX_ACT_ROUNDS = 100
CAUSAL_CLICKS = 3

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


# ─── Discovery with full diagnostics ───
def discover_with_diagnostics(env, action6, obs, n_steps, rng):
    """WHERE+HOW discovery. Returns interactive zones, all zones, magnitudes, causal matrix."""
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

    # HOW: measure local magnitude
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

    interactive = [(zi, clusters[zi]) for zi in range(len(clusters))
                   if magnitudes[zi] > MAG_THRESHOLD]

    # Causal matrix for interactive zones
    causal = np.zeros((len(interactive), len(interactive)), dtype=np.float32)
    for ai, (azi, acl) in enumerate(interactive):
        for _ in range(CAUSAL_CLICKS):
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue
            frame_before = obs.frame
            obs = env.step(action6, data={"x": acl['cx_int'], "y": acl['cy_int']})
            steps += 1
            if obs and obs.frame and len(obs.frame) > 0:
                for bi, (bzi, bcl) in enumerate(interactive):
                    change = local_zone_change(frame_before, obs.frame, bcl['cx_int'], bcl['cy_int'])
                    causal[ai, bi] = max(causal[ai, bi], change)

    return interactive, clusters, magnitudes, causal, obs, steps


# ─── State-aware action (1032 substrate) ───
def infer_target_and_act(env, action6, obs, interactive, rng, budget_steps, time_limit):
    steps = 0
    max_levels = obs.levels_completed if obs else 0
    level_steps = {}
    rediscover_needed = False
    act_rounds = 0

    while steps < budget_steps and time.time() < time_limit:
        if obs is None or obs.state == GameState.GAME_OVER:
            obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue
        if not interactive: break

        act_rounds += 1
        if act_rounds > MAX_ACT_ROUNDS:
            rediscover_needed = True
            break

        zone_colors = {}
        for zi, cl in interactive:
            zone_colors[zi] = read_zone_color(obs.frame, cl['cx_int'], cl['cy_int'])

        color_counts = Counter(zone_colors.values())
        target_color = color_counts.most_common(1)[0][0]
        mismatched = [(zi, cl) for zi, cl in interactive
                      if zone_colors.get(zi, target_color) != target_color]

        if not mismatched:
            if len(color_counts) > 1:
                alt_target = color_counts.most_common(2)[1][0]
                mismatched = [(zi, cl) for zi, cl in interactive
                              if zone_colors.get(zi, alt_target) != alt_target]
            if not mismatched:
                if interactive:
                    zi, cl = interactive[0]
                    if obs.frame:
                        levels_before = obs.levels_completed
                        obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                        steps += 1
                        if obs and obs.levels_completed > levels_before:
                            if obs.levels_completed > max_levels:
                                max_levels = obs.levels_completed
                                level_steps[max_levels] = steps
                            rediscover_needed = True
                            break
                continue

        for zi, cl in mismatched:
            if steps >= budget_steps or time.time() >= time_limit: break
            if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN):
                if obs is None or obs.state == GameState.GAME_OVER:
                    obs = env.reset()
                break
            if not obs.frame or len(obs.frame) == 0:
                obs = env.reset(); continue

            levels_before = obs.levels_completed
            obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
            steps += 1

            if obs is None: break
            if obs.levels_completed > levels_before:
                if obs.levels_completed > max_levels:
                    max_levels = obs.levels_completed
                    level_steps[max_levels] = steps
                rediscover_needed = True
                break

        if rediscover_needed: break

    return obs, steps, max_levels, level_steps, rediscover_needed


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
    level_diagnostics = {}  # level_num → diagnostic data

    while total_steps < MAX_STEPS and time.time() - t0 < TIME_CAP:
        current_level = obs.levels_completed if obs else 0

        # Discover with full diagnostics
        n_disc = WHERE_STEPS if total_steps == 0 else REDISCOVER_STEPS
        interactive, clusters, mags, causal, obs, s = discover_with_diagnostics(
            env, action6, obs, n_disc, rng)
        total_steps += s

        if obs and obs.levels_completed > max_levels:
            max_levels = obs.levels_completed

        # Log diagnostics for this level
        level_num = obs.levels_completed if obs else 0
        if level_num not in level_diagnostics:
            zone_colors = {}
            if obs and obs.frame:
                for zi, cl in interactive:
                    zone_colors[zi] = read_zone_color(obs.frame, cl['cx_int'], cl['cy_int'])

            color_counts = Counter(zone_colors.values()) if zone_colors else Counter()
            target = color_counts.most_common(1)[0][0] if color_counts else -1
            n_mismatched = sum(1 for zi, cl in interactive
                             if zone_colors.get(zi, target) != target) if zone_colors else 0

            level_diagnostics[level_num] = {
                'n_zones_total': len(clusters),
                'n_interactive': len(interactive),
                'positions': [(cl['cx_int'], cl['cy_int']) for _, cl in interactive],
                'magnitudes': [mags[zi] for zi, _ in interactive],
                'zone_colors': zone_colors,
                'target_color': target,
                'n_mismatched': n_mismatched,
                'n_color_classes': len(color_counts),
                'causal_matrix': causal.tolist() if len(causal) > 0 else [],
                'causal_max_offdiag': float(np.max(causal[~np.eye(len(causal), dtype=bool)])) if len(causal) > 1 else 0.0,
            }

            print(f"    Level {level_num}: {len(interactive)} interactive / {len(clusters)} total zones")
            print(f"      Positions: {level_diagnostics[level_num]['positions']}")
            print(f"      Magnitudes: {[round(m, 2) for m in level_diagnostics[level_num]['magnitudes']]}")
            print(f"      Colors: {dict(zone_colors)}  target={target}  mismatched={n_mismatched}")
            print(f"      Color classes: {dict(color_counts)}")
            if len(causal) > 1:
                print(f"      Causal max off-diag: {level_diagnostics[level_num]['causal_max_offdiag']:.3f}")
                # Print causal matrix compactly
                for row_i in range(min(len(causal), 8)):
                    row_str = " ".join(f"{v:.2f}" for v in causal[row_i][:8])
                    print(f"      Causal[{row_i}]: [{row_str}]")

        if not interactive:
            print(f"    Level {level_num}: NO interactive zones found — stuck")
            break

        # State-aware action (1032 substrate)
        budget = min(10000, MAX_STEPS - total_steps)
        obs, s, ml, ls, rediscover = infer_target_and_act(
            env, action6, obs, interactive, rng, budget, t0 + TIME_CAP)
        total_steps += s
        if ml > max_levels:
            max_levels = ml
        level_steps.update(ls)

        # If we're at L3 and stuck, try click-ALL as diagnostic
        if not rediscover and max_levels >= 2:
            print(f"    L{max_levels} stuck — trying click-ALL diagnostic...")
            if obs and obs.frame and obs.state not in (GameState.GAME_OVER, GameState.WIN):
                levels_before = obs.levels_completed
                for zi, cl in interactive:
                    if total_steps >= MAX_STEPS or time.time() - t0 >= TIME_CAP: break
                    if obs is None or obs.state in (GameState.GAME_OVER, GameState.WIN): break
                    if not obs.frame: break
                    obs = env.step(action6, data={"x": cl['cx_int'], "y": cl['cy_int']})
                    total_steps += 1
                if obs and obs.levels_completed > levels_before:
                    print(f"    click-ALL advanced! L{levels_before} → L{obs.levels_completed}")
                    print(f"    → Problem is TARGETING (discovery OK, but majority-vote target wrong)")
                    if obs.levels_completed > max_levels:
                        max_levels = obs.levels_completed
                        level_steps[max_levels] = total_steps
                    continue  # re-discover for next level
                else:
                    print(f"    click-ALL did NOT advance — problem may be DISCOVERY (missing zones or wrong zones)")
            break

        if not rediscover:
            break

    elapsed = time.time() - t0
    return {
        'seed': seed,
        'max_levels': max_levels,
        'level_steps': level_steps,
        'total_steps': total_steps,
        'elapsed': round(elapsed, 1),
        'diagnostics': level_diagnostics
    }


def main():
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if not vc33:
        print("SKIP"); return

    print("=== Step 1034: VC33 L3 Diagnostic ===")
    print(f"What does L3 require that L2 doesn't?")
    print(f"Using 1032 substrate (majority-vote) to reach L3, then diagnostic.")
    print(f"Game: {vc33.game_id}\n")

    for seed in range(3):
        print(f"\n--- Seed {seed} ---")
        r = run_seed(arc, vc33.game_id, seed)
        print(f"\n  Result: L={r['max_levels']}  steps={r['total_steps']}  {r['elapsed']}s")
        if r['level_steps']:
            for lvl, step in sorted(r['level_steps'].items()):
                print(f"    L{lvl} @ step {step}")
        print(f"  Levels diagnosed: {sorted(r['diagnostics'].keys())}")

    print("\nStep 1034 DONE")


if __name__ == "__main__":
    main()
