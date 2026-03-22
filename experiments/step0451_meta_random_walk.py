#!/usr/bin/env python3
"""
Step 451 — Pure random walk baseline. action = randint(4) every step.
No substrate. Tests if 1/10 at 10K is just random walk performance.
10K steps, 10 seeds (same as 449b).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def run_seed(arc, game_id, seed, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id); obs = env.reset()
    na = len(env.action_space)
    ts = go = lvls = 0; action_counts = [0]*na; level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        action_idx = int(np.random.randint(na))
        action_counts[action_idx] += 1
        action = env.action_space[action_idx]; data = {}
        if action.is_complex():
            arr = np.array(obs.frame[0]); cy, cx = divmod(int(np.argmax(arr)), 64)
            data = {"x": cx, "y": cy}
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
        if time.time() - t0 > 280: break
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    return {'seed': seed, 'levels': lvls, 'level_step': level_step, 'dom': dom}


def main():
    import arc_agi
    print("Step 451: Pure random walk. action=randint(4). 10K steps, 10 seeds.", flush=True)
    arc = arc_agi.Arcade(); games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time(); results = []
    for seed in range(10):
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=10000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:20s}  dom={r['dom']:.0f}%", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    print(f"\nStep 451: {len(wins)}/10  steps={sorted([r['level_step'] for r in wins])}"
          f"  elapsed={time.time()-t0:.0f}s", flush=True)
    if wins:
        print(f"Random walk also navigates: CA-graph may be pure random walk.", flush=True)
    else:
        print(f"Random walk fails: CA-graph 1/10 is above random baseline.", flush=True)


if __name__ == '__main__':
    main()
