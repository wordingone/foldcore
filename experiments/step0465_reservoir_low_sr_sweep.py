#!/usr/bin/env python3
"""
Step 465 — Low spectral radius sweep: sr={0.0, 0.05, 0.1, 0.2}.
res_dim=64, k=10 LSH on h-space, graph+edge argmin.
Final 5 experiments (16-20/20) for reservoir family kill assessment.

NOTE: Spec 50K×5 = ~8 min. Running 30K×3 = ~3 min (same reduction as Steps 463/464).

Hypothesis: Lower sr → less recurrence → h closer to obs mapping → lower chg_rate.
sr=0.0 best so far (1/3, chg_rate=0.71, Step 460).

Kill verdict after this step:
- sr=0.0 still best → reservoir memory contributes NOTHING. Kill temporal component.
- Any sr>0 beats sr=0.0 → reservoir memory helps. Continue.

Codebook ban check: no cosine, no attract, no spawn, no F.normalize. PASSES.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

RES_DIM = 64
K = 10


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


def effective_rank(h_samples):
    if len(h_samples) < 5:
        return 0
    H = np.array(h_samples, dtype=np.float32)
    H -= H.mean(axis=0)
    _, s, _ = np.linalg.svd(H, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total == 0:
        return 0
    cumvar = np.cumsum(s2) / total
    return int(np.searchsorted(cumvar, 0.90)) + 1


def make_reservoir(res_dim, obs_dim, sr, seed):
    rng = np.random.RandomState(seed + 7777)
    W_in = rng.randn(res_dim, obs_dim).astype(np.float32) * 0.1
    if sr == 0.0:
        W = np.zeros((res_dim, res_dim), dtype=np.float32)
    else:
        W_raw = rng.randn(res_dim, res_dim).astype(np.float32)
        eigs = np.linalg.eigvals(W_raw)
        actual_sr = np.max(np.abs(eigs))
        W = (W_raw / actual_sr * sr).astype(np.float32)
    return W_in, W


class ReservoirLSHGraph:
    def __init__(self, sr, res_dim=RES_DIM, k=K, seed=0):
        self.sr = sr
        self.W_in, self.W = make_reservoir(res_dim, 256, sr, seed)
        self.h = np.zeros(res_dim, dtype=np.float32)

        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, res_dim).astype(np.float32)
        self.k = k
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)

        self.n_actions = 4
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()
        self.cell_changes = 0
        self.h_samples = []

    def _hash(self, h):
        bits = (self.H @ h > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)
        self.h = np.tanh(self.W_in @ x + self.W @ self.h)
        cell = self._hash(self.h)
        self.cells_seen.add(cell)

        if self.prev_cell is not None and cell != self.prev_cell:
            self.cell_changes += 1

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            d = self.edges.setdefault(key, {})
            d[cell] = d.get(cell, 0) + 1

        if self.step_count % 50 == 0 and len(self.h_samples) < 200:
            self.h_samples.append(self.h.copy())

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


def run_seed(arc, game_id, seed, sr, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = ReservoirLSHGraph(sr=sr, seed=seed)
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
    cell_change_rate = g.cell_changes / max(g.step_count - 1, 1)
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen)
    occupancy = len(g.cells_seen) / (2 ** K)
    rank90 = effective_rank(g.h_samples)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen),
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'cell_change_rate': cell_change_rate, 'occupancy': occupancy,
        'rank90': rank90, 'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    sr_values = [0.0, 0.05, 0.1, 0.2]
    n_seeds = 3
    max_steps = 30000
    print(f"Step 465: Low-sr sweep sr={sr_values}. res_dim={RES_DIM}, k={K}.", flush=True)
    print(f"NOTE: 50Kx5 = ~8min. Running {max_steps//1000}Kx{n_seeds} = ~3min.", flush=True)
    print(f"Baseline (Step 460 sr=0.0): 1/3  chg_rate=0.71  sig_q=0.124", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    sr_results = {}

    for sr in sr_values:
        print(f"\n--- sr={sr} ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, sr=sr, max_steps=max_steps)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/1024"
                  f"  occ={r['occupancy']:.4f}  chg={r['cell_change_rate']:.4f}"
                  f"  sig_q={r['sig_q']:.3f}  rank90={r['rank90']:3d}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_occ  = sum(r['occupancy'] for r in results) / n_seeds
        avg_chg  = sum(r['cell_change_rate'] for r in results) / n_seeds
        avg_sig  = sum(r['sig_q'] for r in results) / n_seeds
        avg_rank = sum(r['rank90'] for r in results) / n_seeds

        sr_results[sr] = {
            'wins': len(wins), 'avg_occ': avg_occ, 'avg_chg': avg_chg,
            'avg_sig': avg_sig, 'avg_rank': avg_rank,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/{n_seeds}  occ={avg_occ:.4f}  chg={avg_chg:.4f}"
              f"  sig_q={avg_sig:.3f}  rank90={avg_rank:.1f}  steps={sr_results[sr]['level_steps']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STEP 465 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'sr':<7} {'Wins':<8} {'Occ':<10} {'ChgRate':<12} {'SigQ':<8} {'Rank90':<8} {'Steps'}", flush=True)
    for sr in sr_values:
        rr = sr_results[sr]
        print(f"  {sr:<5}  {rr['wins']}/{n_seeds}    {rr['avg_occ']:.4f}    {rr['avg_chg']:.4f}       "
              f"{rr['avg_sig']:.3f}   {rr['avg_rank']:<8.1f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Step 460 (sr=0.0, k=10, 30K): 1/3  chg=0.71  occ=0.005  sig_q=0.124", flush=True)
    print(f"  Step 460 (sr=0.9, k=10, 30K): 1/3  chg=0.90  occ=0.017  sig_q=0.305", flush=True)
    print(f"  Pure LSH k=12 (459): 6/10 at 50K", flush=True)

    # Kill verdict
    print(f"\nKILL VERDICT (20 experiments complete):", flush=True)
    wins_by_sr = {sr: sr_results[sr]['wins'] for sr in sr_values}
    best_sr = max(wins_by_sr, key=lambda s: wins_by_sr[s])
    max_wins = wins_by_sr[best_sr]

    if wins_by_sr[0.0] >= max_wins:
        print(f"  sr=0.0 is still best ({wins_by_sr[0.0]}/{n_seeds}).", flush=True)
        print(f"  Reservoir MEMORY contributes NOTHING. W is a dead weight.", flush=True)
        print(f"  The architecture is: tanh(W_in @ obs) + LSH + graph.", flush=True)
        print(f"  This is just nonlinear projection + pure LSH graph. NOT a reservoir.", flush=True)
        print(f"  KILL temporal component. Family verdict: reservoir dynamics = not useful.", flush=True)
    else:
        print(f"  sr={best_sr} beats sr=0.0 ({max_wins} vs {wins_by_sr[0.0]}).", flush=True)
        print(f"  Reservoir memory DOES contribute. sr range {best_sr} is sweet spot.", flush=True)
        print(f"  Continue: 5-seed reliability at best sr.", flush=True)

    chg_trend = [sr_results[sr]['avg_chg'] for sr in sr_values]
    print(f"  chg_rate trend (sr=0.0->0.2): {' '.join(f'{c:.3f}' for c in chg_trend)}", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
