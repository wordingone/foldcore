#!/usr/bin/env python3
"""
Step 457 — Fixed LSH k=10, durability test at 200K steps.
Same as Step 453/454 (k=10, centered_enc, edge-count argmin).
Tests whether edge convergence is a practical problem at long timescales.

Diagnostics at 50K, 100K, 150K, 200K:
- Navigation events (levels found per segment)
- Edge count stats: min/max/mean per occupied cell
- Argmin signal quality: (max_edge - min_edge) / total_edges per cell
- dom%

10 seeds requested. 5-min cap: 10 seeds x 200K ~ 15 min. Running 3 seeds.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

SNAPSHOT_EVERY = 50000


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x):
    return x - x.mean()


def edge_stats(edges, cells_seen, n_actions=4):
    """Per-cell edge counts, argmin signal quality."""
    cell_total = []
    signal_quality = []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        cell_total.append(total)
        if total > 0:
            sq = (max(counts) - min(counts)) / total
        else:
            sq = 0.0
        signal_quality.append(sq)
    if not cell_total:
        return {'min': 0, 'max': 0, 'mean': 0, 'signal_quality': 0.0}
    return {
        'min': min(cell_total),
        'max': max(cell_total),
        'mean': sum(cell_total) / len(cell_total),
        'signal_quality': sum(signal_quality) / len(signal_quality),
    }


class LSHGraph:
    def __init__(self, k=10, seed=0):
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


def run_seed(arc, game_id, seed, max_steps=200000):
    from arcengine import GameState
    np.random.seed(seed)
    g = LSHGraph(k=10, seed=seed)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    action_counts = [0] * na
    level_steps = []
    t0 = time.time()

    # Per-segment tracking
    seg_size = SNAPSHOT_EVERY
    snapshots = {}  # step -> stats
    seg_levels = {}  # segment -> count
    next_snap = seg_size

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
            level_steps.append(ts)
            seg = ((ts - 1) // seg_size) * seg_size
            seg_levels[seg] = seg_levels.get(seg, 0) + 1

        # Snapshot at segment boundaries
        if ts == next_snap:
            dom = max(action_counts) / max(sum(action_counts), 1) * 100
            stats = edge_stats(g.edges, g.cells_seen)
            stats['cells'] = len(g.cells_seen)
            stats['dom'] = dom
            stats['elapsed'] = time.time() - t0
            snapshots[ts] = stats
            next_snap += seg_size

        if time.time() - t0 > 280: break

    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    ratio = len(g.cells_seen) / max(g.step_count, 1)

    return {
        'seed': seed, 'total_levels': lvls, 'level_steps': level_steps,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'elapsed': time.time() - t0,
        'snapshots': snapshots, 'seg_levels': seg_levels,
        'steps_run': ts,
    }


def main():
    import arc_agi
    print("Step 457: Fixed LSH k=10, 200K steps, 3 seeds (5-min cap).", flush=True)
    print("Testing edge convergence at long timescales.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    n_seeds = 3  # 10 seeds x 200K ~ 15 min; 3 seeds ~ 4.7 min

    for seed in range(n_seeds):
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=200000)
        print(f"\n  seed={seed}  levels={r['total_levels']}  steps_run={r['steps_run']}"
              f"  cells={r['unique_cells']}  ratio={r['ratio']:.4f}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        if r['level_steps']:
            print(f"  level_steps={r['level_steps']}", flush=True)
        print(f"  Segment levels: {r['seg_levels']}", flush=True)
        print(f"  Snapshots:", flush=True)
        for snap_step in sorted(r['snapshots']):
            s = r['snapshots'][snap_step]
            print(f"    step={snap_step:6d}  cells={s['cells']:4d}  "
                  f"edge_mean={s['mean']:.1f}  edge_max={s['max']}  "
                  f"signal_q={s['signal_quality']:.3f}  dom={s['dom']:.0f}%  "
                  f"t={s['elapsed']:.0f}s", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 457 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    # Aggregate segment-level navigation rates
    all_segs = set()
    for r in results:
        all_segs.update(r['seg_levels'].keys())
        all_segs.update(r['snapshots'].keys())

    print(f"\nNavigation rate by segment (levels across {n_seeds} seeds):", flush=True)
    for seg in sorted(all_segs):
        if seg in [s for r in results for s in r['snapshots']]:
            total_nav = sum(r['seg_levels'].get(seg, 0) for r in results)
            avg_sq = sum(r['snapshots'].get(seg, {}).get('signal_quality', 0) for r in results) / n_seeds
            avg_cells = sum(r['snapshots'].get(seg, {}).get('cells', 0) for r in results) / n_seeds
            print(f"  [{seg:6d}-{seg+50000:6d}]  nav={total_nav}/{n_seeds}  "
                  f"signal_q={avg_sq:.3f}  avg_cells={avg_cells:.0f}", flush=True)

    # Overall
    total_wins = sum(1 for r in results if r['total_levels'] > 0)
    all_level_steps = sorted([s for r in results for s in r['level_steps']])
    print(f"\nTotal: {total_wins}/{n_seeds} seeds navigated", flush=True)
    print(f"All level steps: {all_level_steps}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Fixed LSH k=10 (454): 4/10 at 50K", flush=True)
    print(f"  Random walk (451):    1/10 at 10K (step 1329)", flush=True)

    # Verdict
    segs_sorted = sorted(all_segs)
    if len(segs_sorted) >= 3:
        early_nav = sum(results[s].get('seg_levels', {}).get(0, 0) for s in range(n_seeds))
        # Compare early vs late signal quality
        early_sq = []
        late_sq = []
        for r in results:
            snaps = r['snapshots']
            if snaps:
                first_snap = min(snaps.keys())
                last_snap = max(snaps.keys())
                early_sq.append(snaps[first_snap]['signal_quality'])
                late_sq.append(snaps[last_snap]['signal_quality'])
        if early_sq and late_sq:
            avg_early = sum(early_sq) / len(early_sq)
            avg_late = sum(late_sq) / len(late_sq)
            drop = (avg_early - avg_late) / max(avg_early, 1e-9)
            print(f"\nSignal quality: early={avg_early:.3f}  late={avg_late:.3f}  "
                  f"drop={drop*100:.1f}%", flush=True)
            if drop > 0.3:
                print("VERDICT: Signal quality drops >30% -- convergence IS real at 200K.", flush=True)
                print("Growth mechanism needed (not exponential like Step 456).", flush=True)
            else:
                print("VERDICT: Signal quality stable -- convergence NOT practical problem at 200K.", flush=True)
                print("Fixed k=10 sufficient. U25 convergence concern = theoretical.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
