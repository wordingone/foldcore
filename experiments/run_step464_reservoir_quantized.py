#!/usr/bin/env python3
"""
Step 464 — Reservoir output quantization: q={0.5, 0.1, 0.05}.
h_q = round(h / q) * q before LSH. Attacks chg_rate flooding root cause.
sr=0.9, res_dim=64, k=10, graph+edge argmin.

Baselines:
  Step 463 (no quant, res_dim=64, sr=0.9, k=10, 30K): 1/3  chg_rate=0.90  occ=49%
  Pure LSH k=10 (454): 4/10 at 50K  occ~0.5-1%  chg_rate~0.2%

Predictions:
  q=0.5: chg_rate ~10-20%  occ ~5-10%  nav 1-2/3
  q=0.1: chg_rate ~40-60%  occ ~20-30%  nav 0-1/3
  q=0.05: chg_rate ~70-80%  occ ~40%   nav 0/3

Codebook ban check: no cosine, no attract, no spawn, no F.normalize. PASSES.
Experiment 15/20 of reservoir family.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

RES_DIM = 64
K = 10
SR = 0.9


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
    W_raw = rng.randn(res_dim, res_dim).astype(np.float32)
    eigs = np.linalg.eigvals(W_raw)
    actual_sr = np.max(np.abs(eigs))
    W = (W_raw / actual_sr * sr).astype(np.float32)
    return W_in, W


class QuantizedReservoirLSHGraph:
    def __init__(self, q, res_dim=RES_DIM, sr=SR, k=K, seed=0):
        self.q = q
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

    def _quantize(self, h):
        return np.round(h / self.q) * self.q

    def _hash(self, h):
        bits = (self.H @ h > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, obs):
        self.step_count += 1
        x = centered_enc(obs)
        self.h = np.tanh(self.W_in @ x + self.W @ self.h)
        h_q = self._quantize(self.h)
        cell = self._hash(h_q)
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


def run_seed(arc, game_id, seed, q, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = QuantizedReservoirLSHGraph(q=q, seed=seed)
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
    occupancy = len(g.cells_seen) / (2 ** K)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'cell_change_rate': cell_change_rate, 'occupancy': occupancy,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    q_values = [0.5, 0.1, 0.05]
    n_seeds = 3
    max_steps = 30000
    print(f"Step 464: Quantized Reservoir-LSH. q={q_values}, sr={SR}, k={K}.", flush=True)
    print(f"Baseline (Step 463 no-quant): 1/3  chg_rate=0.90  occ=49%", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    q_results = {}

    for q in q_values:
        print(f"\n--- q={q} ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, q=q, max_steps=max_steps)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/1024"
                  f"  occ={r['occupancy']:.4f}  chg={r['cell_change_rate']:.4f}"
                  f"  sig_q={r['sig_q']:.3f}  edge_mean={r['edge_mean']:.1f}"
                  f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_occ = sum(r['occupancy'] for r in results) / n_seeds
        avg_chg = sum(r['cell_change_rate'] for r in results) / n_seeds
        avg_sig = sum(r['sig_q'] for r in results) / n_seeds
        avg_edge = sum(r['edge_mean'] for r in results) / n_seeds

        q_results[q] = {
            'wins': len(wins), 'avg_occ': avg_occ, 'avg_chg': avg_chg,
            'avg_sig': avg_sig, 'avg_edge': avg_edge,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/{n_seeds}  occ={avg_occ:.4f}  chg={avg_chg:.4f}"
              f"  sig_q={avg_sig:.3f}  edge_mean={avg_edge:.1f}  steps={q_results[q]['level_steps']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STEP 464 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'q':<8} {'Wins':<8} {'Occ':<10} {'ChgRate':<12} {'SigQ':<8} {'EdgeMean':<10} {'Steps'}", flush=True)
    for q in q_values:
        rr = q_results[q]
        print(f"  {q:<6}  {rr['wins']}/{n_seeds}    {rr['avg_occ']:.4f}    {rr['avg_chg']:.4f}       "
              f"{rr['avg_sig']:.3f}   {rr['avg_edge']:<10.1f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Step 463 (no quant, res_dim=64): 1/3  chg=0.90  occ=0.49  sig_q=0.305", flush=True)
    print(f"  Pure LSH k=10 (454):  4/10 at 50K  chg~0.002  occ~0.005", flush=True)
    print(f"  Pure LSH k=12 (459):  6/10 at 50K  chg~0.002  occ~0.083", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    wins_by_q = {q: q_results[q]['wins'] for q in q_values}
    max_wins = max(wins_by_q.values())
    best_q = max(wins_by_q, key=lambda q: wins_by_q[q])

    if max_wins == 0:
        print(f"  ALL q=0/3. Quantization does not help.", flush=True)
        print(f"  Reservoir temporal dynamics fundamentally incompatible with edge-count argmin.", flush=True)
        print(f"  Reservoir family approaches kill at ~{sum(wins_by_q.values())} total wins / 15 experiments.", flush=True)
    else:
        print(f"  Best q={best_q} ({max_wins}/3). Quantization IS a lever.", flush=True)
        chg_coarse = q_results[0.5]['avg_chg']
        chg_fine = q_results[0.05]['avg_chg']
        print(f"  chg_rate: q=0.5->{chg_coarse:.3f}, q=0.05->{chg_fine:.3f} (baseline 0.90).", flush=True)
        if q_results[0.5]['avg_occ'] < 0.10:
            print(f"  q=0.5 occ={q_results[0.5]['avg_occ']:.3f} — approaching pure-LSH regime.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
