#!/usr/bin/env python3
"""
Step 454 — LSH reliability at 50K steps, 10 seeds.
Same k=10 LSH as Step 453. Gate question: does 3/10 at 30K extrapolate?
Compare directly to cosine graph (Step 445): 3/10 at 50K.
Prediction: 5-6/10 at 50K.
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
    def __init__(self, k, n_actions=4, seed=0):
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


def run_seed(arc, game_id, seed, k=10, n_actions=4, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    g = LSHGraph(k=k, n_actions=n_actions, seed=seed)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)
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
        'dom': dom, 'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    print("Step 454: LSH k=10 reliability — 50K steps, 10 seeds. Gate question.", flush=True)
    print("Compare to cosine graph Step 445: 3/10 at 50K.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    for seed in range(10):
        r = run_seed(arc, ls20.game_id, seed=seed, k=10, n_actions=4, max_steps=50000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:22s}  cells={r['unique_cells']:4d}"
              f"  ratio={r['ratio']:.3f}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    wins = [r for r in results if r['levels'] > 0]
    level_steps = sorted([r['level_step'] for r in wins])
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)

    print(f"\n{'='*60}", flush=True)
    print("STEP 454 RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Reliability: {len(wins)}/10", flush=True)
    print(f"Steps-to-level: {level_steps}", flush=True)
    if level_steps:
        print(f"Min: {min(level_steps)}  Median: {level_steps[len(level_steps)//2]}  Max: {max(level_steps)}", flush=True)
    print(f"Avg unique cells: {avg_cells:.0f}", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)

    print(f"\nComparison:", flush=True)
    print(f"  Cosine graph (Step 445): 3/10 at 50K  median ~19K", flush=True)
    print(f"  LSH k=10 (Step 454):     {len(wins)}/10 at 50K  median {level_steps[len(level_steps)//2] if level_steps else 'N/A'}", flush=True)


if __name__ == '__main__':
    main()
