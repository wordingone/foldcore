#!/usr/bin/env python3
"""
Step 476 — L2 k-means cross-game: FT09 (3 seeds) + VC33 (3 seeds).
Same architecture as 474/475. Tests if warmup captures FT09/VC33 structure.
Prediction: FT09 degenerate (frozen game), VC33 maybe 0-1/3 (50-state cycle).
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
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._warmup_buf = []

    def warmup_done(self): return self.centroids is not None

    def collect(self, x): self._warmup_buf.append(x.copy())

    def fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._warmup_buf, dtype=np.float32)
        # Use fewer clusters if warmup didn't get enough unique obs
        n = min(self.n_clusters, len(set(x.tobytes() for x in X)))
        n = max(n, 2)
        km = MiniBatchKMeans(n_clusters=n, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._warmup_buf = []

    def step(self, x):
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
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


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = KMeansGraph(n_clusters=N_CLUSTERS, n_actions=na)
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
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
    n_centroids = len(g.centroids) if g.centroids is not None else 0
    sig_q, _ = signal_quality(g.edges, g.cells_seen, na)
    return {'seed': seed, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen), 'n_centroids': n_centroids,
            'occupancy': len(g.cells_seen) / max(n_centroids, 1),
            'sig_q': sig_q, 'elapsed': time.time() - t0}


def run_game(arc, game_tag, game_id, n_seeds=3, max_steps=30000):
    print(f"\n--- {game_tag} ({n_seeds} seeds x {max_steps//1000}K) ---", flush=True)
    results = []
    for seed in range(n_seeds):
        r = run_seed(arc, game_id, seed=seed, max_steps=max_steps)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:3d}/{r['n_centroids']}"
              f"  occ={r['occupancy']:.3f}  sig_q={r['sig_q']:.3f}  {r['elapsed']:.0f}s", flush=True)
        results.append(r)
    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / n_seeds
    print(f"  -> {len(wins)}/{n_seeds}  avg_cells={avg_cells:.0f}/{results[0]['n_centroids']}", flush=True)
    return results


def main():
    import arc_agi
    print(f"Step 476: L2 k-means cross-game. FT09 + VC33, 3 seeds x 30K each.", flush=True)
    print(f"Prediction: FT09 degenerate (frozen game), VC33 0-1/3 (timing game).", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    t_total = time.time()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    vc33 = next((g for g in games if 'vc33' in g.game_id.lower()), None)
    ft09_results = run_game(arc, 'FT09', ft09.game_id) if ft09 else []
    vc33_results = run_game(arc, 'VC33', vc33.game_id) if vc33 else []
    print(f"\nVERDICT:", flush=True)
    ft_wins = [r for r in ft09_results if r['levels'] > 0]
    vc_wins = [r for r in vc33_results if r['levels'] > 0]
    ft_cells = sum(r['unique_cells'] for r in ft09_results) / max(len(ft09_results), 1)
    vc_cells = sum(r['unique_cells'] for r in vc33_results) / max(len(vc33_results), 1)
    print(f"  FT09: {len(ft_wins)}/{len(ft09_results)}  avg_cells={ft_cells:.0f}", flush=True)
    if ft_cells < 5:
        print(f"  FT09 DEGENERATE: game still frozen. k-means warmup collapses same as LSH.", flush=True)
    elif len(ft_wins) > 0:
        print(f"  FT09 NAVIGATES: k-means captures frozen-game structure.", flush=True)
    else:
        print(f"  FT09 structure captured ({ft_cells:.0f} cells) but no navigation.", flush=True)
    print(f"  VC33: {len(vc_wins)}/{len(vc33_results)}  avg_cells={vc_cells:.0f}", flush=True)
    if vc_cells < 5:
        print(f"  VC33 DEGENERATE: timing game collapses.", flush=True)
    elif len(vc_wins) > 0:
        print(f"  VC33 NAVIGATES: k-means captures timing cycle.", flush=True)
    else:
        print(f"  VC33 structure captured ({vc_cells:.0f} cells) but edge-count can't solve timing.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
