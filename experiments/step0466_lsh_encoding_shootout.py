#!/usr/bin/env python3
"""
Step 466 — Encoding shootout: centered_enc vs tanh(W_in @ centered_enc(obs)).
Both use k=12 LSH + graph + edge-count argmin.

A: centered_enc(obs) -> LSH k=12 [baseline: Step 459 = 6/10 at 50K]
B: tanh(W_in @ centered_enc(obs)) -> LSH k=12 (res_dim=64, sr=0.0 = no recurrence)

NOTE: 2 encodings × 10 seeds × 50K = ~9 min. Running 5 seeds × 50K = ~4.5 min.
Step 459 (10 seeds) already establishes A baseline. This adds B + 5-seed replication of A.

Kill: B <= A -> nonlinear projection adds nothing. centered_enc is optimal encoding.

Codebook ban check: PASSES.
"""
import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)

K = 12
RES_DIM = 64


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
    """Parametric encoding: 'linear' = centered_enc, 'nonlinear' = tanh(W_in @ enc)."""
    def __init__(self, encoding, k=K, seed=0, res_dim=RES_DIM):
        self.encoding = encoding
        rng_lsh = np.random.RandomState(seed + 9999)
        obs_dim = res_dim if encoding == 'nonlinear' else 256
        self.H = rng_lsh.randn(k, obs_dim).astype(np.float32)
        self.k = k
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)

        if encoding == 'nonlinear':
            rng_w = np.random.RandomState(seed + 7777)
            self.W_in = rng_w.randn(res_dim, 256).astype(np.float32) * 0.1

        self.n_actions = 4
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.step_count = 0
        self.cells_seen = set()
        self.cell_changes = 0

    def _encode(self, obs):
        x = centered_enc(obs)
        if self.encoding == 'nonlinear':
            return np.tanh(self.W_in @ x)
        return x

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, obs):
        self.step_count += 1
        x = self._encode(obs)
        cell = self._hash(x)
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


def run_seed(arc, game_id, seed, encoding, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    g = LSHGraph(encoding=encoding, seed=seed)
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
    chg_rate = g.cell_changes / max(g.step_count - 1, 1)
    sig_q, edge_mean = signal_quality(g.edges, g.cells_seen)
    occupancy = len(g.cells_seen) / (2 ** K)

    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(g.cells_seen),
        'dom': dom, 'sig_q': sig_q, 'edge_mean': edge_mean,
        'chg_rate': chg_rate, 'occupancy': occupancy,
        'elapsed': time.time() - t0,
    }


def main():
    import arc_agi
    encodings = ['linear', 'nonlinear']
    n_seeds = 5
    max_steps = 50000
    print(f"Step 466: Encoding shootout. k={K}, {n_seeds} seeds x {max_steps//1000}K steps.", flush=True)
    print(f"A=centered_enc (linear), B=tanh(W_in @ centered_enc) (nonlinear, res_dim={RES_DIM})", flush=True)
    print(f"Baseline (Step 459, 10 seeds): A=6/10  sig_q=0.184  occ=0.083", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP: LS20 not found"); return

    t_total = time.time()
    enc_results = {}

    for enc in encodings:
        label = 'A (linear)' if enc == 'linear' else 'B (nonlinear)'
        print(f"\n--- Encoding {label} ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, encoding=enc, max_steps=max_steps)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['unique_cells']:4d}/{2**K}"
                  f"  occ={r['occupancy']:.4f}  chg={r['chg_rate']:.4f}"
                  f"  sig_q={r['sig_q']:.3f}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)

        wins = [r for r in results if r['levels'] > 0]
        avg_occ = sum(r['occupancy'] for r in results) / n_seeds
        avg_chg = sum(r['chg_rate'] for r in results) / n_seeds
        avg_sig = sum(r['sig_q'] for r in results) / n_seeds

        enc_results[enc] = {
            'wins': len(wins), 'avg_occ': avg_occ, 'avg_chg': avg_chg,
            'avg_sig': avg_sig,
            'level_steps': sorted([r['level_step'] for r in wins]),
        }
        print(f"  -> {len(wins)}/{n_seeds}  occ={avg_occ:.4f}  chg={avg_chg:.4f}"
              f"  sig_q={avg_sig:.3f}  steps={enc_results[enc]['level_steps']}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 466 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    wins_A = enc_results['linear']['wins']
    wins_B = enc_results['nonlinear']['wins']
    print(f"A (centered_enc):   {wins_A}/{n_seeds}  occ={enc_results['linear']['avg_occ']:.4f}"
          f"  chg={enc_results['linear']['avg_chg']:.4f}  sig_q={enc_results['linear']['avg_sig']:.3f}", flush=True)
    print(f"B (tanh nonlinear): {wins_B}/{n_seeds}  occ={enc_results['nonlinear']['avg_occ']:.4f}"
          f"  chg={enc_results['nonlinear']['avg_chg']:.4f}  sig_q={enc_results['nonlinear']['avg_sig']:.3f}", flush=True)
    print(f"Step 459 reference: A=6/10  occ=0.083  sig_q=0.184", flush=True)

    print(f"\nVERDICT:", flush=True)
    if wins_B > wins_A:
        print(f"  B > A: nonlinear projection HELPS ({wins_B} vs {wins_A}/{n_seeds}).", flush=True)
        print(f"  tanh(W_in) adds useful structure. Worth exploring further.", flush=True)
    elif wins_B == wins_A:
        print(f"  B == A: nonlinear projection NEUTRAL ({wins_B} vs {wins_A}/{n_seeds}).", flush=True)
        print(f"  centered_enc is sufficient. Nonlinear projection adds no benefit.", flush=True)
    else:
        print(f"  B < A: nonlinear projection HURTS ({wins_B} vs {wins_A}/{n_seeds}).", flush=True)
        print(f"  centered_enc is better. Close the nonlinear projection thread.", flush=True)

    print(f"\nTotal elapsed: {time.time() - t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
