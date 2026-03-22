#!/usr/bin/env python3
"""
Step 497 — FT09 death penalty diagnostic.
Step 496: go=1302 deterministic across ALL configs. Deaths may be action-independent.
Test: if death-penalized argmin reduces go → deaths are avoidable by action selection.
If go stays ~1302 → deaths are action-independent (timer/mechanic, not action-caused).
Sweep penalty={10, 100, 1000, 10000}. 3 seeds, 30K steps, km32 (32 true states).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
MAX_STEPS = 30_000
TIME_CAP = 30  # per seed
WARMUP = 500
N_CLUSTERS = 32


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


class KMeansGraphDeathAware:
    def __init__(self, n_clusters=N_CLUSTERS, n_actions=6, warmup=WARMUP, death_penalty=0):
        self.n_clusters = n_clusters
        self.n_actions = n_actions
        self.warmup = warmup
        self.death_penalty = death_penalty
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

    def observe(self, x):
        """Get cell for current observation."""
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return None
        diffs = self.centroids - x
        return int(np.argmin(np.sum(diffs * diffs, axis=1)))

    def step(self, x):
        cell = self.observe(x)
        if cell is None:
            return int(np.random.randint(self.n_actions))
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

    def on_death(self):
        """Penalize the action that caused death."""
        if self.death_penalty > 0 and self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            # Add large count to 'death' pseudo-cell (-1) to inflate edge count
            d[-1] = d.get(-1, 0) + self.death_penalty
        self.prev_cell = None
        self.prev_action = None


def run_config(arc, game_id, death_penalty, n_seeds, label):
    from arcengine import GameState
    print(f"\n  Config: {label} (penalty={death_penalty})", flush=True)
    wins = 0
    all_go = []
    all_cells = []
    for seed in range(n_seeds):
        np.random.seed(seed)
        env = arc.make(game_id)
        na = len(env.action_space)
        g = KMeansGraphDeathAware(n_clusters=N_CLUSTERS, n_actions=na,
                                  warmup=WARMUP, death_penalty=death_penalty)
        obs = env.reset()
        ts = go = lvls = 0
        level_step = None
        t0 = time.time()
        while ts < MAX_STEPS:
            if obs is None: obs = env.reset(); continue
            if obs.state == GameState.GAME_OVER:
                go += 1
                g.on_death()
                obs = env.reset()
                continue
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
        n_c = len(g.centroids) if g.centroids is not None else 0
        status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
        print(f"    seed={seed}  {status:12s}  cells={len(g.cells_seen):2d}/{n_c}"
              f"  go={go}  {time.time()-t0:.0f}s", flush=True)
        if lvls > 0: wins += 1
        all_go.append(go)
        all_cells.append(len(g.cells_seen))
    avg_go = sum(all_go) / n_seeds
    print(f"  -> {wins}/{n_seeds}  avg_go={avg_go:.0f}  cells={sorted(all_cells)}", flush=True)
    return wins, all_go, all_cells


def main():
    import arc_agi
    n_seeds = 3
    print(f"Step 497: FT09 death penalty sweep. {n_seeds} seeds, {MAX_STEPS//1000}K steps.", flush=True)
    print(f"Baseline: go=1302 at 50K. Here baseline ~780 at 30K.", flush=True)
    print(f"Q: does penalty reduce go? If yes: deaths avoidable. If no: action-independent.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ft09 = next((g for g in games if 'ft09' in g.game_id.lower()), None)
    if not ft09:
        print("SKIP — FT09 not found"); return

    t0 = time.time()
    configs = [
        ("baseline",  0),
        ("pen10",     10),
        ("pen100",    100),
        ("pen1k",     1000),
        ("pen10k",    10000),
    ]
    all_results = {}
    for label, penalty in configs:
        wins, go_list, cells = run_config(arc, ft09.game_id, death_penalty=penalty,
                                          n_seeds=n_seeds, label=label)
        all_results[label] = {'wins': wins, 'go': go_list, 'cells': cells}

    print(f"\n{'='*50}", flush=True)
    print("STEP 497 SUMMARY", flush=True)
    baseline_go = sum(all_results['baseline']['go']) / n_seeds
    for label, r in all_results.items():
        avg_go = sum(r['go']) / n_seeds
        delta = avg_go - baseline_go
        flag = f"  ({delta:+.0f} vs baseline)" if label != 'baseline' else ""
        print(f"  {label:10s}: {r['wins']}/{n_seeds}  avg_go={avg_go:.0f}{flag}", flush=True)

    print(f"\nVERDICT:", flush=True)
    min_go = min(sum(r['go']) / n_seeds for r in all_results.values())
    go_reduction = (baseline_go - min_go) / baseline_go
    if go_reduction > 0.2:
        print(f"  DEATHS AVOIDABLE: {go_reduction*100:.0f}% reduction with penalty. Action selection matters.", flush=True)
        if any(r['wins'] > 0 for r in all_results.values()):
            print(f"  NAVIGATION ACHIEVED!", flush=True)
    elif go_reduction > 0.05:
        print(f"  PARTIAL AVOIDANCE: {go_reduction*100:.0f}% reduction. Some deaths are action-caused.", flush=True)
    else:
        print(f"  DEATHS ACTION-INDEPENDENT: <5% reduction. FT09 kills on timer/mechanic, not action.", flush=True)
        print(f"  Standard exploration cannot solve FT09. Game requires a fundamentally different approach.", flush=True)
    print(f"Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()
