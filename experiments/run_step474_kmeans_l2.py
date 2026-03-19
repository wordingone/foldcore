#!/usr/bin/env python3
"""
Step 474 — L2 k-means centroid mapping + graph on LS20. New Family #8.
Warmup: collect 1000 obs → k-means (n=300) → freeze centroids.
Cell = argmin_i ||obs - centroid_i||_L2. Graph + edge-count argmin.
Baseline: LSH k=12 = 6/10 at 50K (Step 459).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
N_CLUSTERS = 300
WARMUP_STEPS = 1000


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


class KMeansGraph:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=4):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.centroids = None  # set after warmup
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._warmup_buf = []

    def warmup_done(self):
        return self.centroids is not None

    def collect(self, x):
        self._warmup_buf.append(x.copy())

    def fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._warmup_buf, dtype=np.float32)
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._warmup_buf = []

    def assign(self, x):
        # L2 nearest centroid
        diffs = self.centroids - x
        dists = np.sum(diffs * diffs, axis=1)
        return int(np.argmin(dists))

    def step(self, x):
        cell = self.assign(x)
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
    g = KMeansGraph(n_clusters=N_CLUSTERS, n_actions=na)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    warmup_done_at = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        x = centered_enc(avgpool16(obs.frame))
        if not g.warmup_done():
            g.collect(x)
            if len(g._warmup_buf) >= WARMUP_STEPS:
                g.fit()
                warmup_done_at = ts
            # During warmup: random action
            action_idx = int(np.random.randint(na))
        else:
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
            'unique_cells': len(g.cells_seen), 'occupancy': len(g.cells_seen) / N_CLUSTERS,
            'sig_q': sig_q, 'edge_mean': edge_mean,
            'warmup_done_at': warmup_done_at, 'elapsed': time.time() - t0}


def main():
    import arc_agi
    print(f"Step 474: L2 k-means (n={N_CLUSTERS}) + graph on LS20. 5 seeds x 50K.", flush=True)
    print(f"Warmup: {WARMUP_STEPS} obs -> k-means -> freeze. Cell = argmin L2 dist.", flush=True)
    print(f"Baseline (LSH k=12 Step 459): 6/10 at 50K, occ=0.083, sig_q=0.184", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t0 = time.time()
    results = []
    for seed in range(5):
        r = run_seed(arc, ls20.game_id, seed=seed)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:3d}/{N_CLUSTERS}"
              f"  occ={r['occupancy']:.3f}  sig_q={r['sig_q']:.3f}"
              f"  edge_mean={r['edge_mean']:.1f}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / 5
    avg_occ = sum(r['occupancy'] for r in results) / 5
    avg_sig = sum(r['sig_q'] for r in results) / 5
    print(f"\n{len(wins)}/5  avg_cells={avg_cells:.0f}/{N_CLUSTERS}  avg_occ={avg_occ:.3f}"
          f"  avg_sig_q={avg_sig:.3f}", flush=True)
    print(f"level_steps={sorted([r['level_step'] for r in wins])}", flush=True)
    print("\nVERDICT:", flush=True)
    if avg_cells < 10:
        print(f"  DEGENERATE ({avg_cells:.0f} cells). K-means collapsed — check cluster quality.", flush=True)
    elif len(wins) > 3:
        print(f"  L2 BEATS LSH: {len(wins)}/5. Data-aligned partitioning > random hyperplanes.", flush=True)
    elif len(wins) >= 2:
        print(f"  L2 COMPARABLE: {len(wins)}/5. Similar to LSH. Data alignment doesn't help.", flush=True)
    elif len(wins) > 0:
        print(f"  L2 WEAKER: {len(wins)}/5 < LSH 6/10. Random hyperplanes better than data-aligned.", flush=True)
    else:
        print(f"  L2 FAILS: 0/5. Structure exists ({avg_cells:.0f} cells) but no navigation.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
