"""
Step 1019b — FT09 Click Order Ablation

R3 hypothesis: click order is irrelevant (each click changes one target wall,
final state is order-independent) — OR does click order matter due to cgj()
triggering early on intermediate states?

For Hkx-only levels (L1-L4): clicks are decoupled, order provably irrelevant.
For NTi levels (L5-L6): clicks are coupled (each NTi cycles 5 walls), but
the final combined state is the same regardless of order. cgj() may fire early.

Test: shuffle prescription clicks N=20 times per level, check pass rate.
PASS criterion: all shuffles pass for all levels -> order irrelevant
KILL criterion: some shuffle fails -> order-dependent (prescription is a sequence, not a set)
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
from util_ft09_level_solver import solve_level

N_SHUFFLES = 20
RANDOM_SEED = 42

# ─── Get prescription clicks per level ───
prescriptions = {}
for lvl_idx in range(6):
    name, gqb, bst, conflicts, clicks = solve_level(lvl_idx)
    prescriptions[lvl_idx] = clicks

print("=== FT09 Prescription ===")
for lvl_idx, clicks in prescriptions.items():
    print(f"  L{lvl_idx+1}: {len(clicks)} clicks, {len(set(clicks))} distinct")

# ─── Play FT09 with given clicks (as ordered list) ───

GA_LIST = list(GameAction)[1:]
ACTION6 = GA_LIST[5]
ACTION1 = GA_LIST[0]

env = arcagi3.make('FT09')
env.reset(seed=0)


def get_inner():
    return env._env


def do_click(cx, cy):
    obs = get_inner().step(ACTION6, data={"x": cx, "y": cy})
    if obs is None:
        return True, 0
    done = obs.state in (GameState.GAME_OVER, GameState.WIN)
    lvl = obs.levels_completed - env._levels_offset
    return done, lvl


def do_null():
    obs = get_inner().step(ACTION1)
    if obs is None:
        return True, 0
    done = obs.state in (GameState.GAME_OVER, GameState.WIN)
    lvl = obs.levels_completed - env._levels_offset
    return done, lvl


def play_level(lvl_idx, ordered_clicks):
    """Play a single level with the given click order. Returns (pass, steps_to_pass)."""
    env.reset(seed=0)

    # Advance through previous levels using canonical prescription order
    for prev_idx in range(lvl_idx):
        prev_clicks = prescriptions[prev_idx]
        level_done = False
        for cx, cy in prev_clicks:
            _, lvl = do_click(cx, cy)
            if lvl > prev_idx:
                level_done = True
                break
        if not level_done:
            for _ in range(20):
                _, lvl = do_null()
                if lvl > prev_idx:
                    break

    actual_level = get_inner()._game.level_index
    if actual_level != lvl_idx:
        return False, 0, f'at_level={actual_level}'

    level_start = actual_level
    done_flag = False
    steps = 0
    for cx, cy in ordered_clicks:
        _, lvl = do_click(cx, cy)
        steps += 1
        if lvl > level_start:
            done_flag = True
            break
    if not done_flag:
        for _ in range(20):
            _, lvl = do_null()
            steps += 1
            if lvl > level_start:
                done_flag = True
                break
    return done_flag, steps, ''


# ─── Test canonical order first ───
print(f"\n=== Condition A: Canonical order (baseline) ===")
rng = random.Random(RANDOM_SEED)
for lvl_idx in range(6):
    ok, steps, err = play_level(lvl_idx, prescriptions[lvl_idx])
    print(f"  L{lvl_idx+1}: {'PASS' if ok else 'FAIL'} ({steps} steps) {err}")

# ─── Test N shuffles per level ───
print(f"\n=== Condition B: {N_SHUFFLES} random shuffles per level ===")
all_pass = True
for lvl_idx in range(6):
    clicks = list(prescriptions[lvl_idx])
    passes = 0
    steps_list = []
    fail_examples = []
    for shuffle_i in range(N_SHUFFLES):
        shuffled = clicks[:]
        rng.shuffle(shuffled)
        ok, steps, err = play_level(lvl_idx, shuffled)
        if ok:
            passes += 1
            steps_list.append(steps)
        else:
            fail_examples.append((shuffle_i, err))
    if passes == N_SHUFFLES:
        avg_steps = sum(steps_list) / len(steps_list)
        min_steps = min(steps_list)
        print(f"  L{lvl_idx+1}: ALL {N_SHUFFLES}/{N_SHUFFLES} PASS (avg_steps={avg_steps:.1f}, min={min_steps})")
    else:
        all_pass = False
        print(f"  L{lvl_idx+1}: {passes}/{N_SHUFFLES} PASS, FAILURES: {fail_examples[:3]}")

# ─── Test: what is the minimum prefix that passes? ───
print(f"\n=== Minimum prefix test (canonical order) ===")
for lvl_idx in range(6):
    clicks = prescriptions[lvl_idx]
    n = len(clicks)
    # Binary search for minimum prefix
    lo, hi = 1, n
    while lo < hi:
        mid = (lo + hi) // 2
        ok, _, _ = play_level(lvl_idx, clicks[:mid])
        if ok:
            hi = mid
        else:
            lo = mid + 1
    ok, steps, _ = play_level(lvl_idx, clicks[:lo])
    print(f"  L{lvl_idx+1}: min prefix = {lo}/{n} clicks to pass")

# ─── Summary ───
print(f"\n=== Summary ===")
if all_pass:
    print(f"  PASS: click order irrelevant — prescription is a SET not a sequence")
    print(f"  Implication: substrate only needs to DISCOVER which walls to click,")
    print(f"  not the specific order.")
else:
    print(f"  KILL: some orderings fail — prescription is order-dependent")

print("\nStep 1019b DONE")
