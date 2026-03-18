#!/usr/bin/env python3
"""
Step 455 — LSH k=10 on FT09 (69-class click-space action).
Cross-game test. Codebook reached Level 1 at step 82 on FT09.
LSH prediction: MARGINAL at 10K (69 actions needs more edges per cell).
3 seeds, 10K steps. Extend to 50K if 0/3.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    return x - x.mean()


class LSHGraph:
    def __init__(self, k, n_actions, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.k = k
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)
        cell = self._hash(x)
        self.cells_seen.add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[cell] = d.get(cell, 0) + 1

        vc = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        mn = min(vc)
        cands = [a for a, c in enumerate(vc) if c == mn]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, k=10, max_steps=10000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)
    g = LSHGraph(k=k, n_actions=na, seed=seed)
    ts = go = lvls = 0
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)

        action_counts[action_idx % na] += 1
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

        if time.time() - t0 > 290: break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells_seen) / max(g.step_count, 1)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'n_actions': na, 'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    print("Step 455: LSH k=10 on FT09. 10K steps, 3 seeds.", flush=True)
    print("FT09: 69-class click-space. Codebook reached Level 1 at step 82.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP: FT09 not found"); return

    print(f"Found: {ft09.game_id}", flush=True)
    t_total = time.time()
    results = []

    for seed in [0, 1, 2]:
        r = run_seed(arc, ft09.game_id, seed=seed, k=10, max_steps=10000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}"
              f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%"
              f"  n_actions={r['n_actions']}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_ratio = sum(r['ratio'] for r in results) / len(results)

    print(f"\nStep 455: {len(wins)}/3 at 10K  steps={sorted([r['level_step'] for r in wins])}"
          f"  avg_cells={avg_cells:.0f}  ratio={avg_ratio:.3f}"
          f"  elapsed={time.time()-t_total:.0f}s", flush=True)

    # Extension to 50K if 0/3
    if len(wins) == 0:
        print(f"\nExtending to 50K (0/3 at 10K)...", flush=True)
        results50 = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ft09.game_id, seed=seed, k=10, max_steps=50000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}"
                  f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
            results50.append(r)
        wins50 = [r for r in results50 if r['levels'] > 0]
        print(f"FT09 50K: {len(wins50)}/3  steps={sorted([r['level_step'] for r in wins50])}"
              f"  elapsed={time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
