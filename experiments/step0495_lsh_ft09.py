#!/usr/bin/env python3
"""
Step 495 — FT09 with raw 64x64 encoding + higher k. FT09 priority (flagged).
FT09: 32 visual states (confirmed by k-means, Step 476). 6 native actions.
avgpool16 collapsed FT09 to 1 cell (Steps 467/469). Try raw 64x64 + centered_enc.
k=16 and k=20 for finer discrimination.
5 seeds, 50K steps per config. Report unique cells + navigation.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 50_000
TIME_CAP = 35  # per seed


def raw_enc(frame):
    """Raw 64x64 centered (no pooling)."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.flatten()
    return x - x.mean()


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class LSHGraph:
    def __init__(self, k, obs_dim, n_actions=4, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, obs_dim).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()

    def step(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_config(arc, game_id, k, enc_fn, obs_dim, n_seeds, label):
    from arcengine import GameState
    print(f"\n  Config: {label} (k={k}, obs_dim={obs_dim})", flush=True)
    wins = 0
    all_cells = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = arc.make(game_id)
        na = len(env.action_space)
        g = LSHGraph(k=k, obs_dim=obs_dim, n_actions=na, seed=seed)
        obs = env.reset()
        ts = go = lvls = 0
        level_step = None
        t0 = time.time()
        while ts < MAX_STEPS:
            if obs is None: obs = env.reset(); continue
            if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
            x = enc_fn(obs.frame)
            action_idx = g.step(x)
            action = env.action_space[action_idx % na]
            data = {}
            if action.is_complex():
                arr = np.array(obs.frame[0])
                cy, cx = divmod(int(np.argmax(arr)), 64)
                data = {"x": cx, "y": cy}
            obs_before = obs.levels_completed
            obs = env.step(action, data=data)
            ts += 1
            if obs is None: break
            if obs.levels_completed > obs_before:
                lvls = obs.levels_completed
                if level_step is None: level_step = ts
            if time.time() - t0 > TIME_CAP: break
        status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
        print(f"    seed={seed}  {status:12s}  cells={len(g.cells_seen):5d}  "
              f"go={go}  {time.time()-t0:.0f}s", flush=True)
        if lvls > 0: wins += 1
        all_cells.append(len(g.cells_seen))
    print(f"  -> {wins}/{n_seeds}  cells={sorted(all_cells)}", flush=True)
    return wins, all_cells


def main():
    import arc_agi
    n_seeds = 5
    print(f"Step 495: FT09 raw64x64 + high-k LSH. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"FT09 has 6 native actions, 32 known visual states. avgpool16 collapses to 1 cell.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09: print("SKIP — FT09 not found"); return

    # First: check action space
    env0 = arc.make(ft09.game_id)
    na = len(env0.action_space)
    print(f"\nFT09 action space: {na} actions", flush=True)
    for i, a in enumerate(env0.action_space[:8]):
        print(f"  action[{i}]: {a}", flush=True)

    t0 = time.time()
    configs = [
        ("raw64_k12",  lambda f: raw_enc(f), 4096, 12),
        ("raw64_k16",  lambda f: raw_enc(f), 4096, 16),
        ("raw64_k20",  lambda f: raw_enc(f), 4096, 20),
        ("pool16_k12", lambda f: centered_enc(avgpool16(f)), 256, 12),
        ("pool16_k16", lambda f: centered_enc(avgpool16(f)), 256, 16),
    ]
    all_results = {}
    for label, enc_fn, obs_dim, k in configs:
        wins, cells = run_config(arc, ft09.game_id, k=k, enc_fn=enc_fn,
                                 obs_dim=obs_dim, n_seeds=n_seeds, label=label)
        all_results[label] = {'wins': wins, 'cells': cells}

    print(f"\n{'='*50}", flush=True)
    print("STEP 495 SUMMARY", flush=True)
    for label, r in all_results.items():
        print(f"  {label}: {r['wins']}/{n_seeds}  cells={r['cells']}", flush=True)

    best = max(all_results, key=lambda l: max(all_results[l]['cells']))
    best_cells = max(all_results[best]['cells'])
    print(f"\nVERDICT:", flush=True)
    if any(r['wins'] > 0 for r in all_results.values()):
        best_win = max(all_results, key=lambda l: all_results[l]['wins'])
        print(f"  FT09 NAVIGATES: {all_results[best_win]['wins']}/{n_seeds} with {best_win}!", flush=True)
    elif best_cells >= 30:
        print(f"  ENCODING WORKS: {best_cells} cells with {best}. FT09's 32 states discriminated.", flush=True)
        print(f"  Navigation fails — action selection problem, not encoding.", flush=True)
    elif best_cells >= 10:
        print(f"  PARTIAL DISCRIMINATION: {best_cells} cells. Some states distinguished.", flush=True)
    else:
        print(f"  STILL COLLAPSING: {best_cells} max cells. Raw 64x64 doesn't discriminate FT09.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
