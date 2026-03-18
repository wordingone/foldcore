#!/usr/bin/env python3
"""
Step 458 — LSH k-sweep: k=8, 10, 12, 14.
Same as Step 453/454 (centered_enc, edge-count argmin). 50K steps.
Tests whether k=10 is optimal or if finer/coarser discrimination helps reliability.

10 seeds requested. 5-min cap: 10 seeds x 50K x 4 k-values ~ 15 min.
Running 3 seeds per k (4 k-values x 3 seeds x ~22s = ~270s < 5 min).

Diagnostics (LSH-appropriate):
- Navigation: levels + step counts
- Occupancy: occupied cells / max cells
- Mean edge count per occupied cell at 50K
- Signal quality: (max_edge - min_edge) / total_edges per cell, averaged
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    return x - x.mean()


def compute_signal_quality(edges, cells_seen, n_actions=4):
    """Avg (max_edge - min_edge) / total_edges over occupied cells."""
    qualities = []
    edge_totals = []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        edge_totals.append(total)
        if total > 0:
            qualities.append((max(counts) - min(counts)) / total)
        else:
            qualities.append(0.0)
    if not qualities:
        return 0.0, 0.0
    return sum(qualities) / len(qualities), sum(edge_totals) / len(edge_totals)


class LSHGraph:
    def __init__(self, k, seed=0):
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


def run_seed(arc, game_id, seed, k, max_steps=50000):
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
    signal_q, edge_mean = compute_signal_quality(g.edges, g.cells_seen)
    occupancy = len(g.cells_seen) / (2 ** k)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'signal_q': signal_q, 'edge_mean': edge_mean,
        'occupancy': occupancy, 'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    n_seeds = 3  # 10 seeds x 4 k-values x 50K ~ 15 min; 3 seeds ~ 4.5 min
    print(f"Step 458: LSH k-sweep k=8,10,12,14. 50K steps, {n_seeds} seeds per k.", flush=True)
    print("Diagnostics: occupancy, edge_mean, signal_quality, navigation.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    k_values = [8, 10, 12, 14]
    k_results = {}

    for k in k_values:
        max_cells = 2 ** k
        print(f"\n--- k={k} (max_cells={max_cells}) ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, k=k, max_steps=50000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{max_cells}"
                  f"  occ={r['occupancy']:.4f}  edge_mean={r['edge_mean']:.1f}"
                  f"  sig_q={r['signal_q']:.3f}  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s",
                  flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_cells = sum(r['unique_cells'] for r in results) / n_seeds
        avg_occ = sum(r['occupancy'] for r in results) / n_seeds
        avg_edge = sum(r['edge_mean'] for r in results) / n_seeds
        avg_sig = sum(r['signal_q'] for r in results) / n_seeds

        k_results[k] = {
            'wins': len(wins), 'avg_cells': avg_cells, 'avg_occ': avg_occ,
            'avg_edge': avg_edge, 'avg_sig': avg_sig,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/{n_seeds}  occ={avg_occ:.4f}  edge_mean={avg_edge:.1f}"
              f"  sig_q={avg_sig:.3f}  steps={k_results[k]['level_steps']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STEP 458 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'k':<5} {'MaxCells':<10} {'Wins':<8} {'Occupancy':<12} {'EdgeMean':<12}"
          f"{'SigQ':<8} {'Steps'}", flush=True)
    for k in k_values:
        rr = k_results[k]
        print(f"  k={k:<2}  {2**k:<10}  {rr['wins']}/{n_seeds:<5}  {rr['avg_occ']:.5f}     "
              f"{rr['avg_edge']:<12.1f}{rr['avg_sig']:<8.3f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  k=10 (454): 4/10 at 50K", flush=True)
    print(f"  k=10 (453): 3/10 at 30K", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    wins_by_k = {k: k_results[k]['wins'] for k in k_values}
    sig_by_k = {k: k_results[k]['avg_sig'] for k in k_values}
    best_k = max(wins_by_k, key=lambda k: wins_by_k[k])
    max_wins = wins_by_k[best_k]
    min_wins = min(wins_by_k.values())

    if max_wins - min_wins <= 1:
        print(f"  All k values similar ({min_wins}-{max_wins}/{n_seeds}). k is NOT the reliability lever.", flush=True)
        print(f"  Look elsewhere: preprocessing, edge initialization, action selection.", flush=True)
    else:
        print(f"  Best k={best_k} ({max_wins}/{n_seeds}). k IS a reliability lever.", flush=True)
        if best_k < 10:
            print(f"  Coarser cells help: lower k = more revisitation = better edge stats.", flush=True)
        elif best_k > 10:
            print(f"  Finer cells help: higher k = better discrimination = fresh argmin.", flush=True)

    # Signal quality trend
    sigs = [sig_by_k[k] for k in k_values]
    print(f"  Signal quality trend (k=8->14): {' '.join(f'{s:.3f}' for s in sigs)}", flush=True)
    if sigs[0] > sigs[-1] * 1.2:
        print(f"  Signal quality drops with k (sparse edges at high k confirmed).", flush=True)
    elif sigs[-1] > sigs[0] * 1.2:
        print(f"  Signal quality improves with k (finer cells give cleaner argmin).", flush=True)
    else:
        print(f"  Signal quality stable across k.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
