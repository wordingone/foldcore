"""
Step 600 -- Graph density at L1 transition.

Diagnostic: when the agent first reaches L1, how large is the graph?
  - cells: unique LSH nodes visited
  - edges: unique (node, action) pairs with at least one observation
  - avg_degree: edges / cells

Run argmin K=12, 20 seeds, 10K. Record graph state at first L1 event.
Also record graph state at step 10K (final) for comparison.

Provides empirical grounding for cover time analysis (Proposition 8).
"""
import numpy as np
import time
import sys

DIM = 256
N_A = 4
K = 12
MAX_STEPS = 10_000
TIME_CAP = 60
N_SEEDS = 20


def enc_vec(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def lsh_hash(x, H):
    bits = (H @ x > 0).astype(np.int64)
    return int(np.dot(bits, 1 << np.arange(len(bits))))


class Argmin:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}; self._pn = self._pa = self._cn = None
        self.cells = set(); self.total_deaths = 0

    def observe(self, frame):
        x = enc_vec(frame); n = lsh_hash(x, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; return action

    def graph_stats(self):
        n_cells = len(self.cells)
        n_edges = len(self.G)
        avg_degree = n_edges / n_cells if n_cells > 0 else 0
        return n_cells, n_edges, avg_degree

    def on_death(self): self.total_deaths += 1
    def on_reset(self): self._pn = None


def run_seed(mk, seed):
    env = mk(); sub = Argmin(seed=seed * 100 + 7)
    obs = env.reset(seed=seed); sub.on_reset()
    l1 = go = step = 0; prev_cl = 0; fresh = True; t0 = time.time()
    l1_stats = None

    while step < MAX_STEPS and time.time() - t0 < TIME_CAP:
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        sub.observe(obs); action = sub.act()
        obs, _, done, info = env.step(action); step += 1
        if done:
            sub.on_death(); obs = env.reset(seed=seed); sub.on_reset()
            prev_cl = 0; fresh = True; go += 1; continue
        cl = info.get('level', 0) if isinstance(info, dict) else 0
        if fresh: prev_cl = cl; fresh = False
        elif cl >= 1 and prev_cl < 1:
            l1 += 1
            if l1_stats is None:
                l1_stats = (step,) + sub.graph_stats()
        prev_cl = cl

    final_stats = (step,) + sub.graph_stats()
    return dict(l1=l1, l1_stats=l1_stats, final_stats=final_stats)


def main():
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    print(f"Step 600: Graph density at L1 transition", flush=True)
    print(f"  K={K} | {N_SEEDS} seeds x {MAX_STEPS} steps", flush=True)

    t0 = time.time()
    results = []
    for seed in range(N_SEEDS):
        r = run_seed(mk, seed)
        results.append(r)
        if r['l1_stats']:
            s, nc, ne, nd = r['l1_stats']
            print(f"  s{seed}: L1@{s} cells={nc} edges={ne} deg={nd:.1f}", flush=True)
        else:
            fs, nc, ne, nd = r['final_stats']
            print(f"  s{seed}: no L1 (final cells={nc} edges={ne})", flush=True)

    # Summarize L1 graph stats
    l1_data = [r['l1_stats'] for r in results if r['l1_stats'] is not None]
    final_data = [r['final_stats'] for r in results]
    wins = sum(1 for r in results if r['l1'] > 0)

    print(f"\n{'='*60}", flush=True)
    print(f"Step 600: Graph density at L1 ({wins}/{N_SEEDS} seeds reached L1)", flush=True)

    if l1_data:
        steps_at_l1 = [d[0] for d in l1_data]
        cells_at_l1 = [d[1] for d in l1_data]
        edges_at_l1 = [d[2] for d in l1_data]
        deg_at_l1   = [d[3] for d in l1_data]
        print(f"\n  At first L1 (n={len(l1_data)}):", flush=True)
        print(f"    step:    avg={np.mean(steps_at_l1):.0f}  min={min(steps_at_l1)}  max={max(steps_at_l1)}", flush=True)
        print(f"    cells:   avg={np.mean(cells_at_l1):.0f}  min={min(cells_at_l1)}  max={max(cells_at_l1)}", flush=True)
        print(f"    edges:   avg={np.mean(edges_at_l1):.0f}  min={min(edges_at_l1)}  max={max(edges_at_l1)}", flush=True)
        print(f"    avg_deg: avg={np.mean(deg_at_l1):.2f}", flush=True)

    cells_final = [d[1] for d in final_data]
    edges_final = [d[2] for d in final_data]
    print(f"\n  At step {MAX_STEPS} (all seeds):", flush=True)
    print(f"    cells: avg={np.mean(cells_final):.0f}  min={min(cells_final)}  max={max(cells_final)}", flush=True)
    print(f"    edges: avg={np.mean(edges_final):.0f}", flush=True)

    print(f"\n  Total elapsed: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
