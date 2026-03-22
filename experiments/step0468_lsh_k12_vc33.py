#!/usr/bin/env python3
"""
Step 468 — LSH k=12 on VC33. Cross-game test 2/3.
VC33: 1 action (click), timing game, 50-step visual cycle.
With 1 action, argmin is trivial — clicks every step.
Tests whether edge-count graph helps timing, or just random clicking.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)
K = 12

def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()

def centered_enc(x): return x - x.mean()

def signal_quality(edges, cells_seen, n_actions):
    qualities, totals = [], []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        totals.append(total)
        qualities.append((max(counts) - min(counts)) / total if total > 0 else 0.0)
    if not qualities: return 0.0, 0.0
    return sum(qualities) / len(qualities), sum(totals) / len(totals)

class LSHGraph:
    def __init__(self, k=K, n_actions=1, seed=0):
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

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[cell] = d.get(cell, 0) + 1
        visit_counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action

def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = LSHGraph(k=K, n_actions=na, seed=seed)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)
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
        if time.time() - t0 > 280: break
    sig_q, _ = signal_quality(g.edges, g.cells_seen, na)
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen) / (2**K),
        'n_actions': na, 'sig_q': sig_q, 'elapsed': time.time() - t0,
    }

def main():
    import arc_agi
    print(f"Step 468: LSH k={K} on VC33. 5 seeds x 50K.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    vc33 = next((g for g in games if 'vc33' in g.game_id.lower()), None)
    if not vc33: print("SKIP: VC33 not found"); return
    print(f"Game: {vc33.game_id}", flush=True)
    t_total = time.time()
    results = []
    for seed in range(5):
        r = run_seed(arc, vc33.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{2**K}"
              f"  occ={r['occupancy']:.4f}  sig_q={r['sig_q']:.3f}"
              f"  n_actions={r['n_actions']}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / 5
    print(f"\nReliability: {len(wins)}/5  avg_cells={avg_cells:.0f}  level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"\nVERDICT:", flush=True)
    if avg_cells < 5:
        print(f"  DEGENERATE ({avg_cells:.0f} cells). VC33 frames identical to LSH.", flush=True)
    elif len(wins) > 0:
        print(f"  VC33 NAVIGATES: {len(wins)}/5. Cross-game generality confirmed.", flush=True)
    else:
        print(f"  VC33 structure resolved ({avg_cells:.0f} cells) but 0 levels.", flush=True)
        print(f"  Timing game cannot be solved by edge-count argmin — as predicted.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)

if __name__ == '__main__':
    main()
