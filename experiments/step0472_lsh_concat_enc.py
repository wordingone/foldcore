#!/usr/bin/env python3
"""
Step 472 — Concatenated encoding (position + diff) + k=12 LSH on LS20.
Encoding: concat(centered_enc(avgpool16(obs)), centered_enc(avgpool16(obs-prev)))
512D total. Cells represent where agent is AND what just changed.
Baseline: k=12 raw frame = 6/10, diff-frame = 2/5 (Step 471).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12
OBS_DIM = 512  # 256 position + 256 diff


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
    def __init__(self, k=K, obs_dim=OBS_DIM, n_actions=4, seed=0):
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
        visit_counts = [sum(self.edges.get((cell, a), {}).values()) for a in range(self.n_actions)]
        min_c = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = LSHGraph(k=K, obs_dim=OBS_DIM, n_actions=na, seed=seed)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    prev_pool = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); prev_pool = None; continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); prev_pool = None; continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); prev_pool = None; continue
        pool = avgpool16(obs.frame)
        diff = np.zeros_like(pool) if prev_pool is None else pool - prev_pool
        x = np.concatenate([centered_enc(pool), centered_enc(diff)])
        action_idx = g.step(x)
        prev_pool = pool
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
            'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen) / (2**K),
            'sig_q': sig_q, 'edge_mean': edge_mean, 'elapsed': time.time() - t0}


def main():
    import arc_agi
    print(f"Step 472: Concat encoding (pos+diff) k={K} on LS20. 5 seeds x 50K.", flush=True)
    print(f"Encoding: concat(centered_enc(pool), centered_enc(diff)) = 512D", flush=True)
    print(f"Baseline: raw frame k=12 = 6/10 (Step 459), diff k=12 = 2/5 (Step 471)", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(5):
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{2**K}"
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
    if len(wins) > 3:
        print(f"  CONCAT HELPS: {len(wins)}/5 > baseline 3/5. New encoding to explore.", flush=True)
    elif len(wins) >= 3:
        print(f"  CONCAT NEUTRAL: {len(wins)}/5 ~ baseline. 512D doesn't hurt but doesn't help.", flush=True)
    elif len(wins) > 0:
        print(f"  CONCAT HURTS: {len(wins)}/5 < baseline. Extra dims dilute signal (U13 applies).", flush=True)
    else:
        print(f"  CONCAT KILLS: 0/5. Position alone is sufficient; diff dims are noise.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
