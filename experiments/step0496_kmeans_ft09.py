#!/usr/bin/env python3
"""
Step 496 — FT09 k-means graph. Step 476 confirmed k-means reaches 32 FT09 cells.
Step 495 confirmed 6 non-complex actions (ACTION1-6). Can argmin navigate?
Same KMeansGraph as LS20 Steps 474-475 but on FT09.
Sweep n_clusters={32, 64, 300} and warmup={500, 1000}.
5 seeds, 50K steps each (~10s/seed expected). Report cells, wins, sig_q.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 50_000
TIME_CAP = 40  # per seed


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraph:
    def __init__(self, n_clusters=300, n_actions=6, warmup=1000):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []

    def _fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._buf, dtype=np.float32)
        n = min(self.n_clusters, len(set(x.tobytes() for x in X)), len(X))
        n = max(n, 2)
        km = MiniBatchKMeans(n_clusters=n, random_state=42,
                             n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._buf = []

    def step(self, x):
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return int(np.random.randint(self.n_actions))
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
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


def sig_quality(edges, cells_seen, n_actions):
    qs = []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        if total > 0:
            qs.append((max(counts) - min(counts)) / total)
    return sum(qs) / len(qs) if qs else 0.0


def run_config(arc, game_id, n_clusters, warmup, n_seeds, label):
    from arcengine import GameState
    print(f"\n  Config: {label} (n={n_clusters}, warmup={warmup})", flush=True)
    wins = 0
    all_cells = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = arc.make(game_id)
        na = len(env.action_space)
        g = KMeansGraph(n_clusters=n_clusters, n_actions=na, warmup=warmup)
        obs = env.reset()
        ts = go = lvls = 0
        level_step = None
        t0 = time.time()
        while ts < MAX_STEPS:
            if obs is None: obs = env.reset(); continue
            if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
            if obs.state == GameState.WIN: break
            if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
            x = centered_enc(avgpool16(obs.frame))
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
        sq = sig_quality(g.edges, g.cells_seen, na)
        status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
        n_c = len(g.centroids) if g.centroids is not None else 0
        print(f"    seed={seed}  {status:12s}  cells={len(g.cells_seen):3d}/{n_c}"
              f"  go={go}  sq={sq:.3f}  {time.time()-t0:.0f}s", flush=True)
        if lvls > 0: wins += 1
        all_cells.append(len(g.cells_seen))
    print(f"  -> {wins}/{n_seeds}  cells={sorted(all_cells)}", flush=True)
    return wins, all_cells


def main():
    import arc_agi
    n_seeds = 5
    print(f"Step 496: FT09 k-means graph. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"FT09: 6 non-complex actions, 32 visual states (k-means confirmed Step 476).", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP — FT09 not found"); return

    t0 = time.time()
    configs = [
        ("km32_w500",  32,  500),
        ("km64_w500",  64,  500),
        ("km300_w500", 300, 500),
        ("km300_w1k",  300, 1000),
    ]
    all_results = {}
    for label, n_clusters, warmup in configs:
        wins, cells = run_config(arc, ft09.game_id, n_clusters=n_clusters,
                                 warmup=warmup, n_seeds=n_seeds, label=label)
        all_results[label] = {'wins': wins, 'cells': cells}

    print(f"\n{'='*50}", flush=True)
    print("STEP 496 SUMMARY", flush=True)
    for label, r in all_results.items():
        print(f"  {label}: {r['wins']}/{n_seeds}  cells={r['cells']}", flush=True)

    best_wins = max(r['wins'] for r in all_results.values())
    best_cells = max(max(r['cells']) for r in all_results.values())
    print(f"\nVERDICT:", flush=True)
    if best_wins > 0:
        best_label = max(all_results, key=lambda l: all_results[l]['wins'])
        print(f"  FT09 NAVIGATES: {all_results[best_label]['wins']}/{n_seeds} with {best_label}!", flush=True)
    elif best_cells >= 25:
        print(f"  MAPPING WORKS ({best_cells} cells, 32 expected). Navigation fails — action selection problem.", flush=True)
    elif best_cells >= 10:
        print(f"  PARTIAL ({best_cells} cells). Warmup too short or n_clusters too high.", flush=True)
    else:
        print(f"  COLLAPSE ({best_cells} cells). FT09 not discriminatable even by k-means.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
