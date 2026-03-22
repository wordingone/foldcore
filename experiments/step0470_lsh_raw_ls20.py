#!/usr/bin/env python3
"""
Step 470 — Raw 64x64 k=16 LSH on LS20. No avgpool.
Tests whether LSH handles 4096D where codebook (Steps 377-389) failed.
Baseline: k=12 avgpool16 = 6/10 at 50K.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 16

def raw_enc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.flatten()
    return x - x.mean()

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
    def __init__(self, k, obs_dim, n_actions, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, obs_dim).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = n_actions
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()

    def step(self, x):
        self.step_count += 1
        bits = (self.H @ x > 0).astype(np.int64)
        cell = int(np.dot(bits, self.powers))
        self.cells_seen.add(cell)
        if self.prev_cell is not None and self.prev_action is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        visit_counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell; self.prev_action = action
        return action

def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = LSHGraph(k=K, obs_dim=4096, n_actions=na, seed=seed)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        x = raw_enc(obs.frame)
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
        if time.time() - t0 > 280: break
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen, na)
    return {'seed': seed, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen)/(2**K),
            'sig_q': sig_q, 'edge_mean': edge_mean, 'elapsed': time.time()-t0}

def main():
    import arc_agi
    print(f"Step 470: Raw 64x64 k={K} on LS20. 5 seeds x 50K.", flush=True)
    print(f"Baseline: avgpool16 k=12 = 6/10. Codebook raw 64x64 failed (Steps 377-389).", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(5):
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:5d}/{2**K}"
              f"  occ={r['occupancy']:.4f}  sig_q={r['sig_q']:.3f}"
              f"  edge_mean={r['edge_mean']:.1f}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / 5
    avg_occ = sum(r['occupancy'] for r in results) / 5
    avg_sig = sum(r['sig_q'] for r in results) / 5
    print(f"\n{len(wins)}/5  avg_cells={avg_cells:.0f}/{2**K}  avg_occ={avg_occ:.4f}  avg_sig_q={avg_sig:.3f}", flush=True)
    print(f"level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print("\nVERDICT:", flush=True)
    if avg_cells < 10:
        print(f"  DEGENERATE ({avg_cells:.0f} cells). Raw LS20 frames also collapse. avgpool16 is essential.", flush=True)
    elif len(wins) > 0:
        print(f"  LS20 NAVIGATES raw 4096D: {len(wins)}/5. LSH handles high-D — avgpool not required.", flush=True)
    else:
        print(f"  Structure resolved ({avg_cells:.0f} cells) but 0/5. Signal diluted in 4096D.", flush=True)
        print(f"  avgpool16 is not a codebook limitation — it's genuinely needed for signal focus.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)

if __name__ == '__main__':
    main()
