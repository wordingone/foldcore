#!/usr/bin/env python3
"""
Step 460 — Reservoir-LSH hybrid: spectral radius sweep sr={0.0, 0.5, 0.9}.
h(t) = tanh(W_in @ centered_enc(obs) + sr * W @ h(t-1))
cell = LSH(h(t)) using k=10 hyperplanes on h-space (64-dim).
Graph + edge-count argmin (same as Steps 453+).

Tests whether reservoir dynamics can be compatible with graph mechanism
through spectral radius control.

Diagnostics (reservoir-appropriate):
- Temporal consistency: cell_change_rate (fraction of steps where cell changes)
- Occupied cells at 30K
- Signal quality (same as Steps 458-459)
- Navigation: levels + steps

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


def make_reservoir(res_dim, obs_dim, sr, seed):
    """Build W_in (res_dim x obs_dim) and W (res_dim x res_dim) scaled to spectral radius sr."""
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

        # LSH on h-space (res_dim-dim)
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
        self.cell_changes = 0  # steps where cell changed from prev

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
    ratio = len(g.cells_seen) / max(g.step_count, 1)
    cell_change_rate = g.cell_changes / max(g.step_count - 1, 1)
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen), 'ratio': ratio,
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'cell_change_rate': cell_change_rate,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    print("Step 460: Reservoir-LSH hybrid. sr sweep={0.0, 0.5, 0.9}.", flush=True)
    print("h(t) = tanh(W_in @ centered_enc(obs) + sr * W @ h(t-1)). k=10 on h-space.", flush=True)
    print("30K steps, 3 seeds per sr. Diagnostics: temporal consistency + signal quality.", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    sr_values = [0.0, 0.5, 0.9]
    sr_results = {}

    for sr in sr_values:
        print(f"\n--- sr={sr} ---", flush=True)
        results = []
        for seed in [0, 1, 2]:
            r = run_seed(arc, ls20.game_id, seed=seed, sr=sr, max_steps=30000)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}"
                  f"  ratio={r['ratio']:.4f}  chg_rate={r['cell_change_rate']:.4f}"
                  f"  edge_mean={r['edge_mean']:.1f}  sig_q={r['sig_q']:.3f}"
                  f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_cells = sum(r['unique_cells'] for r in results) / 3
        avg_ratio = sum(r['ratio'] for r in results) / 3
        avg_chg = sum(r['cell_change_rate'] for r in results) / 3
        avg_sig = sum(r['sig_q'] for r in results) / 3
        avg_edge = sum(r['edge_mean'] for r in results) / 3

        sr_results[sr] = {
            'wins': len(wins), 'avg_cells': avg_cells, 'avg_ratio': avg_ratio,
            'avg_chg': avg_chg, 'avg_sig': avg_sig, 'avg_edge': avg_edge,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/3  chg_rate={avg_chg:.4f}  ratio={avg_ratio:.4f}"
              f"  sig_q={avg_sig:.3f}  steps={sr_results[sr]['level_steps']}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("STEP 460 SUMMARY", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'sr':<6} {'Wins':<8} {'CellChgRate':<14} {'Ratio':<10} {'SigQ':<8} {'EdgeMean':<10} {'Steps'}", flush=True)
    for sr in sr_values:
        rr = sr_results[sr]
        print(f"  {sr:<4}  {rr['wins']}/3    {rr['avg_chg']:.4f}         {rr['avg_ratio']:.5f}   "
              f"{rr['avg_sig']:.3f}   {rr['avg_edge']:<10.1f} {rr['level_steps']}", flush=True)

    print(f"\nBaselines:", flush=True)
    print(f"  Pure LSH k=10 (454):  4/10 at 50K  chg_rate~?  ratio=0.001-0.005", flush=True)
    print(f"  ESN sr=0.9 (448):     0/3  at 50K  ratio=0.942", flush=True)

    # Verdict
    print(f"\nVERDICT:", flush=True)
    any_nav = any(rr['wins'] > 0 for rr in sr_results.values())
    if not any_nav:
        print(f"  ALL sr=0 -- reservoir output destroys local continuity. Architecture-level limitation.", flush=True)
        print(f"  Reservoir-LSH hybrid killed.", flush=True)
    else:
        best_sr = max(sr_values, key=lambda s: sr_results[s]['wins'])
        print(f"  Best sr={best_sr} ({sr_results[best_sr]['wins']}/3).", flush=True)
        chg_rate_0 = sr_results[0.0]['avg_chg']
        chg_rate_9 = sr_results[0.9]['avg_chg']
        if chg_rate_9 > chg_rate_0 * 2:
            print(f"  Temporal inconsistency confirmed: chg_rate 0->{chg_rate_0:.3f}, 0.9->{chg_rate_9:.3f}.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
