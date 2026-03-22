#!/usr/bin/env python3
"""
Step 434 — Pure random walk on LS20. 10 seeds, 50K steps each.
No codebook, no substrate. Uniform random action selection.
Compare to substrate's ~26K for Level 1.
"""

import time, logging, random
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def run_seed(seed, arc, ls20, max_steps=50000):
    from arcengine import GameState
    env = arc.make(ls20.game_id); obs = env.reset()
    na = len(env.action_space)
    rng = random.Random(seed)

    ts = go = lvls = 0; unique = set(); t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset()
            if obs is None: break
            continue
        if obs.state == GameState.WIN:
            break

        pooled = np.array(obs.frame[0], dtype=np.float32)
        unique.add(hash(pooled.tobytes()))

        idx = rng.randint(0, na - 1)
        action = env.action_space[idx]
        data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0])
            cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed

    elapsed = time.time() - t0
    return {'seed': seed, 'levels': lvls, 'steps': ts, 'unique': len(unique),
            'go': go, 'elapsed': elapsed}


def main():
    import arc_agi
    print(f"Step 434: Pure random walk on LS20. 10 seeds, 50K each.", flush=True)

    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return

    results = []
    for seed in range(10):
        r = run_seed(seed, arc, ls20)
        results.append(r)
        nav = f"L{r['levels']}@{r['steps']}" if r['levels'] > 0 else "none"
        print(f"  Seed {seed}: {nav}  unique={r['unique']}  go={r['go']}  {r['elapsed']:.0f}s", flush=True)

    print(f"\n{'='*60}")
    print("STEP 434 — RANDOM WALK BASELINE")
    print(f"{'='*60}")

    navigated = [r for r in results if r['levels'] > 0]
    not_nav = [r for r in results if r['levels'] == 0]

    print(f"Navigated: {len(navigated)}/10 seeds")
    if navigated:
        steps = [r['steps'] for r in navigated]
        print(f"Steps to Level 1: {steps}")
        print(f"Mean: {np.mean(steps):.0f}  Median: {np.median(steps):.0f}")
    print(f"Did not navigate: {len(not_nav)}/10 seeds (at 50K)")

    avg_unique = np.mean([r['unique'] for r in results])
    print(f"Avg unique states: {avg_unique:.0f}")
    print(f"\nSubstrate baseline: ~26K steps, 60% reliability")
    if navigated:
        print(f"Random walk: {np.mean([r['steps'] for r in navigated]):.0f} steps, {len(navigated)*10}% reliability")
        if np.mean([r['steps'] for r in navigated]) < 30000:
            print("WARNING: random walk navigates at similar speed. Substrate adds nothing.")
    else:
        print("Random walk: 0/10 at 50K. Substrate is faster than random.")


if __name__ == '__main__':
    main()
