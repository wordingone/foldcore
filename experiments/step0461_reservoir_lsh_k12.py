#!/usr/bin/env python3
"""
Step 461 — Reservoir-LSH k=12: upgrade Step 460's k=10 hash to k=12.
h(t) = tanh(W_in @ centered_enc(obs) + sr * W @ h(t-1))
cell = LSH(h(t)) using k=12 hyperplanes on 64-dim h-space.
Graph + edge-count argmin (same as Steps 453+).

Hypothesis: k=12 gave LSH 6/10 vs k=10's 4/10 (Steps 458-459).
Same effect should apply when hashing reservoir output — fewer collisions,
cleaner argmin, better nav reliability.

Baseline (Step 460):
  sr=0.0: 1/3 at 30K  chg_rate=0.71  sig_q=0.124
  sr=0.9: 1/3 at 30K  chg_rate=0.90  sig_q=0.305

Run sr={0.0, 0.9} with k=12. 30K steps, 3 seeds per sr.
Compare: sig_q, chg_rate, occupancy, wins.

Codebook ban check: no cosine, no attract, no spawn, no F.normalize. PASSES.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

RES_DIM = 64
K = 12


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
        self.res_dim = res_dim
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


def run_seed(arc, game_id, seed, sr, k=K, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = ReservoirLSHGraph(sr=sr, k=k, seed=seed)
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
    cell_change_rate = g.cell_changes / max(g.step_count - 1, 1)
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen)
    occupancy = len(g.cells_seen) / (2 ** k)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'cell_change_rate': cell_change_rate, 'occupancy': occupancy,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    k = K
    print(f"Step 461: Reservoir-LSH k={k}. sr={{0.0, 0.9}}, 30K steps, 3 seeds.", flush=True)
    print(f"Baseline (Step 460 k=10): sr=0: 1/3, sig_q=0.124 | sr=0.9: 1/3, sig_q=0.305", flush=True)
    print(f"Hypothesis: k=12 reduces occupancy, improves sig_q, better nav reliability.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    sr_values = [0.0, 0.9]
    sr_results = {}

    for sr in sr_values:
        print(f"\n--- sr={sr} ---", flush=True)
        results = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ls20.game_id, seed=seed, sr=sr, k=k, max_steps=30000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{2**k}"
                  f"  occ={r['occupancy']:.4f}  chg_rate={r['cell_change_rate']:.4f}"
                  f"  sig_q={r['sig_q']:.3f}  edge_mean={r['edge_mean']:.1f}"
                  f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_cells = sum(r['unique_cells'] for r in results) / 3
        avg_occ = sum(r['occupancy'] for r in results) / 3
        avg_chg = sum(r['cell_change_rate'] for r in results) / 3
        avg_sig = sum(r['sig_q'] for r in results) / 3
        avg_edge = sum(r['edge_mean'] for r in results) / 3

        sr_results[sr] = {
            'wins': len(wins), 'avg_cells': avg_cells, 'avg_occ': avg_occ,
            'avg_chg': avg_chg, 'avg_sig': avg_sig, 'avg_edge': avg_edge,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/3  occ={avg_occ:.4f}  chg_rate={avg_chg:.4f}"
              f"  sig_q={avg_sig:.3f}  steps={sr_results[sr]['level_steps']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STEP 461 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'sr':<6} {'Wins':<8} {'Occupancy':<12} {'CellChgRate':<14} {'SigQ':<8} {'EdgeMean':<10} {'Steps'}", flush=True)
    for sr in sr_values:
        rr = sr_results[sr]
        print(f"  {sr:<4}  {rr['wins']}/3    {rr['avg_occ']:.5f}      {rr['avg_chg']:.4f}         "
              f"{rr['avg_sig']:.3f}   {rr['avg_edge']:<10.1f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines (Step 460 k=10):", flush=True)
    print(f"  sr=0.0: 1/3  chg_rate=0.71  ratio=0.005  sig_q=0.124", flush=True)
    print(f"  sr=0.9: 1/3  chg_rate=0.90  ratio=0.017  sig_q=0.305", flush=True)
    print(f"  Pure LSH k=12 (459): 6/10  sig_q=0.184  occ=0.083", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    wins_0 = sr_results[0.0]['wins']
    wins_9 = sr_results[0.9]['wins']
    sig_0 = sr_results[0.0]['avg_sig']
    sig_9 = sr_results[0.9]['avg_sig']
    occ_0 = sr_results[0.0]['avg_occ']
    occ_9 = sr_results[0.9]['avg_occ']

    if wins_0 + wins_9 >= 3:
        print(f"  k=12 IMPROVES Reservoir-LSH: {wins_0+wins_9}/6 total wins.", flush=True)
        if occ_0 < 0.08:
            print(f"  sr=0 occupancy={occ_0:.4f} < pure-LSH k=12 (0.083) — reservoir adds variability.", flush=True)
        if sig_9 > sr_results[0.0]['avg_sig'] * 1.2:
            print(f"  sr=0.9 higher sig_q ({sig_9:.3f} vs sr=0: {sig_0:.3f}) — dynamics help discrimination.", flush=True)
    else:
        print(f"  k=12 does NOT improve Reservoir-LSH: {wins_0+wins_9}/6 wins.", flush=True)
        print(f"  k is not the lever for reservoir. Look at res_dim or sr range.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
