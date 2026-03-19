#!/usr/bin/env python3
"""
Step 473 — Multi-hash LSH (L tables, k=12 each) on LS20.
L independent hash tables. Action = argmin(sum of edge counts across tables).
L={1,3,5}. Baseline: L=1 = 6/10 at 50K (Step 459).
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)
K = 12


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def centered_enc(x): return x - x.mean()


def signal_quality(edges, cells_seen, n_actions):
    qualities, totals = [], []
    for c in cells_seen:
        counts = [sum(edges.get((c, a), {}).values()) for a in range(n_actions)]
        total = sum(counts)
        totals.append(total)
        qualities.append((max(counts) - min(counts)) / total if total > 0 else 0.0)
    if not qualities: return 0.0, 0.0
    return sum(qualities) / len(qualities), sum(totals) / len(totals)


class HashTable:
    """Single LSH table with its own random hyperplanes and graph."""
    def __init__(self, k, seed):
        rng = np.random.RandomState(seed)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.edges = {}
        self.cells_seen = set()
        self.prev_cell = None

    def hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def update(self, cell, prev_cell, prev_action):
        self.cells_seen.add(cell)
        if prev_cell is not None and prev_action is not None:
            d = self.edges.setdefault((prev_cell, prev_action), {})
            d[cell] = d.get(cell, 0) + 1

    def visit_counts(self, cell, n_actions):
        return [sum(self.edges.get((cell, a), {}).values()) for a in range(n_actions)]


class MultiHashGraph:
    def __init__(self, L, k=K, n_actions=4, seed=0):
        self.tables = [HashTable(k, seed=seed * 1000 + i * 37 + 9999) for i in range(L)]
        self.n_actions = n_actions
        self.prev_cells = [None] * L
        self.prev_action = None

    def step(self, x):
        cells = [t.hash(x) for t in self.tables]
        for i, (t, cell) in enumerate(zip(self.tables, cells)):
            t.update(cell, self.prev_cells[i], self.prev_action)

        # Sum visit counts across all tables
        combined = [0] * self.n_actions
        for i, (t, cell) in enumerate(zip(self.tables, cells)):
            vc = t.visit_counts(cell, self.n_actions)
            for a in range(self.n_actions):
                combined[a] += vc[a]

        min_c = min(combined)
        candidates = [a for a, c in enumerate(combined) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cells = cells
        self.prev_action = action
        return action

    def total_cells(self):
        return sum(len(t.cells_seen) for t in self.tables)

    def avg_sig_q(self, n_actions):
        qs = [signal_quality(t.edges, t.cells_seen, n_actions)[0] for t in self.tables]
        return sum(qs) / len(qs)


def run_seed(arc, game_id, seed, L, max_steps=50000):
    from arcengine import GameState
    np.random.seed(seed)
    env = arc.make(game_id)
    na = len(env.action_space)
    g = MultiHashGraph(L=L, k=K, n_actions=na, seed=seed)
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
    return {'seed': seed, 'L': L, 'levels': lvls, 'level_step': level_step,
            'total_cells': g.total_cells(), 'sig_q': g.avg_sig_q(na),
            'elapsed': time.time() - t0}


def main():
    import arc_agi
    L_values = [1, 3, 5]
    n_seeds = 5
    print(f"Step 473: Multi-hash LSH L={L_values} k={K} on LS20. {n_seeds} seeds x 50K.", flush=True)
    print(f"Baseline (L=1 Step 459): 6/10 at 50K.", flush=True)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20: print("SKIP"); return
    t_total = time.time()
    all_results = {}
    for L in L_values:
        print(f"\n--- L={L} ({L} hash table{'s' if L>1 else ''}) ---", flush=True)
        results = []
        for seed in range(n_seeds):
            r = run_seed(arc, ls20.game_id, seed=seed, L=L)
            status = f"LEVEL 1 at {r['level_step']}" if r['levels'] > 0 else "no level"
            print(f"  seed={seed}  {status:22s}  cells={r['total_cells']:5d}"
                  f"  sig_q={r['sig_q']:.3f}  {r['elapsed']:.0f}s", flush=True)
            results.append(r)
        wins = [r for r in results if r['levels'] > 0]
        avg_sig = sum(r['sig_q'] for r in results) / n_seeds
        all_results[L] = {'wins': len(wins), 'avg_sig': avg_sig,
                          'level_steps': sorted([r['level_step'] for r in wins])}
        print(f"  -> {len(wins)}/{n_seeds}  avg_sig_q={avg_sig:.3f}  steps={all_results[L]['level_steps']}", flush=True)
    print(f"\n{'='*50}", flush=True)
    print("STEP 473 SUMMARY", flush=True)
    for L in L_values:
        r = all_results[L]
        print(f"  L={L}: {r['wins']}/{n_seeds}  avg_sig_q={r['avg_sig']:.3f}  steps={r['level_steps']}", flush=True)
    print("\nVERDICT:", flush=True)
    w1 = all_results[1]['wins']
    w3 = all_results[3]['wins']
    w5 = all_results[5]['wins']
    if w3 > w1:
        print(f"  L=3 HELPS: {w3} vs {w1}/{n_seeds}. Multi-hash amplification works.", flush=True)
        print(f"  Test L=5 trend: {'improving' if w5 >= w3 else 'diminishing returns'}.", flush=True)
    elif w3 == w1:
        print(f"  L=3 NEUTRAL: {w3} vs {w1}/{n_seeds}. Extra tables add no benefit.", flush=True)
        print(f"  Seed luck is not in the hash function — one table is sufficient.", flush=True)
    else:
        print(f"  L=3 HURTS: {w3} vs {w1}/{n_seeds}. Multiple tables dilute the signal.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
