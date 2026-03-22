#!/usr/bin/env python3
"""
Step 477 — Softmax action selection on L2 k-means. Temperature sweep.
action_probs = softmax(-edge_counts / T). T={0.1, 1.0, 10.0}.
Same k-means substrate as 474 (n=300, 1K warmup).
Baseline: argmin (T->0) = 5-6/10 on LS20.
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


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


class KMeansSoftmax:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=4, temperature=1.0):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.temperature = temperature
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
        km = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=42,
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
        counts = np.array([sum(self.edges.get((cell, a), {}).values())
                           for a in range(self.n_actions)], dtype=np.float32)
        probs = softmax(-counts / self.temperature)
        action = int(np.random.choice(self.n_actions, p=probs))
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, temperature, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = KMeansSoftmax(n_clusters=N_CLUSTERS, n_actions=na, temperature=temperature)
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
    return {'seed': seed, 'T': temperature, 'levels': lvls, 'level_step': level_step,
            'unique_cells': len(g.cells_seen), 'elapsed': time.time() - t0}


def main():
    import arc_agi
    temps = [0.1, 1.0, 10.0]
    n_seeds = 3  # 3 temps x 3 seeds = 9 runs ~4 min
    print(f"Step 477: Softmax T={temps} on L2 k-means. {n_seeds} seeds x 50K.", flush=True)
    print(f"Baseline (argmin T->0): 5-6/10 on LS20.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t_total = time.time()
    all_results = {}
    for T in temps:
        print(f"\n--- T={T} ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, temperature=T)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:3d}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)
        wins = [r for r in results if r['levels'] > 0]
        all_results[T] = {'wins': len(wins), 'level_steps': sorted([r['level_step'] for r in wins])}
        print(f"  -> {len(wins)}/{n_seeds}  steps={all_results[T]['level_steps']}", flush=True)
    print(f"\n{'='*50}", flush=True)
    print("STEP 477 SUMMARY", flush=True)
    for T in temps:
        r = all_results[T]
        print(f"  T={T}: {r['wins']}/{n_seeds}  steps={r['level_steps']}", flush=True)
    print("\nVERDICT:", flush=True)
    best_T = max(temps, key=lambda T: all_results[T]['wins'])
    best_wins = all_results[best_T]['wins']
    baseline = n_seeds * 6 // 10  # proportional baseline
    if best_wins > baseline:
        print(f"  T={best_T} BEATS ARGMIN: {best_wins}/{n_seeds}. Soft exploration helps.", flush=True)
    elif best_wins == all_results[0.1]['wins']:
        print(f"  ALL T EQUAL argmin: ceiling is not in action selection stochasticity.", flush=True)
    else:
        print(f"  T={best_T} best ({best_wins}/{n_seeds}). Marginal vs argmin baseline.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
