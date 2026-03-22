#!/usr/bin/env python3
"""
Step 459 — LSH k=12 reliability: 10 seeds x 50K steps.
Same as Step 454 (k=10, 10 seeds, 50K) but with k=12.
Confirms whether k=12 anomaly (step-471 win, sig_q=0.236) was real.

Step 458 showed k=12: 2/3, sig_q=0.236 (vs k=10: 1/3, sig_q=0.116).
Hypothesis: k=12 is genuinely better. Prediction: 5-6/10 at 50K.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    return x - x.mean()


def signal_quality(edges, cells_seen, n_actions=4):
    qualities, totals = [], []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        totals.append(total)
        qualities.append((max(counts) - min(counts)) / total if total > 0 else 0.0)
    if not qualities:
        return 0.0, 0.0
    return sum(qualities) / len(qualities), sum(totals) / len(totals)


class LSHGraph:
    def __init__(self, k=12, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.k = k
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.n_actions = 4
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

        visit_counts = [
            sum(self.edges.get((cell, a), {}).values())
            for a in range(self.n_actions)
        ]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_cell = cell
        self.prev_action = action
        return action


def run_seed(arc, game_id, seed, k=12, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    g = LSHGraph(k=k, seed=seed)
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

        if time.time() - t0 > 280: break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells_seen) / max(g.step_count, 1)
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen)
    occupancy = len(g.cells_seen) / (2 ** k)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'occupancy': occupancy, 'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    k = 12
    n_seeds = 10
    print(f"Step 459: LSH k={k} reliability. {n_seeds} seeds x 50K steps.", flush=True)
    print(f"Confirming k=12 anomaly from Step 458 (sig_q=0.236, step-471 win).", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []

    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed, k=k, max_steps=50000)
        status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed:2d}  {status:22s}  cells={r['unique_cells']:4d}/{2**k}"
              f"  occ={r['occupancy']:.4f}  edge_mean={r['edge_mean']:.1f}"
              f"  sig_q={r['sig_q']:.3f}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s",
              flush=True)
        results.append(r)

    wins = [r for r in results if r['levels'] > 0]
    level_steps = sorted([r['level_step'] for r in wins])
    avg_cells = sum(r['unique_cells'] for r in results) / n_seeds
    avg_occ = sum(r['occupancy'] for r in results) / n_seeds
    avg_edge = sum(r['edge_mean'] for r in results) / n_seeds
    avg_sig = sum(r['sig_q'] for r in results) / n_seeds

    print(f"\n{'='*60}", flush=True)
    print(f"Step 459: k={k} RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Reliability: {len(wins)}/{n_seeds}", flush=True)
    print(f"Level steps: {level_steps}", flush=True)
    if level_steps:
        print(f"Min: {min(level_steps)}  Max: {max(level_steps)}  "
              f"Median: {sorted(level_steps)[len(level_steps)//2]}", flush=True)
    print(f"avg_cells={avg_cells:.0f}/{2**k}  avg_occ={avg_occ:.4f}", flush=True)
    print(f"avg_edge_mean={avg_edge:.1f}  avg_sig_q={avg_sig:.3f}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  k=10 (454):   4/10  sig_q~0.116", flush=True)
    print(f"  k=12 (458):   2/3   sig_q=0.236  (small sample)", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    if len(wins) >= 6:
        print(f"  k=12 CONFIRMED better: {len(wins)}/10 > k=10's 4/10.", flush=True)
        print(f"  k=12 is new LSH baseline.", flush=True)
    elif len(wins) >= 4:
        print(f"  k=12 SIMILAR to k=10 ({len(wins)}/10 vs 4/10). Anomaly was seed noise.", flush=True)
        print(f"  k is not the reliability lever.", flush=True)
    else:
        print(f"  k=12 WORSE than k=10 ({len(wins)}/10 < 4/10). Step 458 k=12 was outlier.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
