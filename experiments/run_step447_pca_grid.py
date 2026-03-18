#!/usr/bin/env python3
"""
Step 447 — PCA Grid Graph on LS20.
Same as Step 446 but random projection replaced with PCA-based projection.

Warmup: collect 500 obs, compute top-8 PCs (frozen). Then percentile bins.
No cosine, no attract, no prototypes, no F.normalize. PCA computed once.

Ban checklist:
1. Cosine/attract? NO — PCA + bin lookup, no cosine matching
2. LVQ? NO — PCA + grid + graph (three different techniques)
3. Codebook + X? NO — no prototypes at all
4. Shared spatial engine? NO — no match->update->grow cycle

Distinguishes interpretations from 446b:
- If navigates: data-aligned projection recovers what cosine does. Random
  projection was the bottleneck. Graph family validated without codebook DNA.
- If 0/3 with ~2000 cells: adaptive placement (moving nodes toward data)
  is necessary, not just initial alignment. Navigation IS cosine-mediated.

Parameters: proj_dim=8, bins=4, warmup=500. LS20 30K steps, 3 seeds.
"""

import time, logging
import numpy as np

logging.getLogger().setLevel(logging.WARNING)


class PCAGridGraph:
    def __init__(self, proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256,
                 warmup_n=500):
        self.proj_dim = proj_dim
        self.bins_per_dim = bins_per_dim
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.warmup_n = warmup_n

        # PCA projection (computed during warmup, frozen after)
        self.P = None           # (proj_dim, obs_dim) top-K eigenvectors
        self.bin_edges = None   # (proj_dim, bins_per_dim - 1)
        self.warmup_buf = []    # raw 256D observations

        # Graph: edges only, no node positions
        self.edges = {}   # (cell_id, action) -> {next_cell: count}
        self.cells = set()

        self.prev_cell = None
        self.prev_action = None

    def compute_pca_and_bins(self):
        """Compute PCA from warmup buffer. Freeze P and bin_edges."""
        X = np.stack(self.warmup_buf)  # (warmup_n, obs_dim)
        # Center
        mean = X.mean(axis=0)
        Xc = X - mean
        self.mean = mean

        # Covariance eigenvectors via SVD (more numerically stable than eig)
        # U: (n, n), S: (min,), Vt: (obs_dim, obs_dim)
        # Top-k right singular vectors = top-k PCs
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.P = Vt[:self.proj_dim].astype(np.float32)  # (proj_dim, obs_dim)

        # Project all warmup obs and compute percentile bin edges
        projected = (Xc @ self.P.T)  # (n, proj_dim)
        edges = []
        for d in range(self.proj_dim):
            pcts = [100.0 * (i + 1) / self.bins_per_dim
                    for i in range(self.bins_per_dim - 1)]
            edges.append(np.percentile(projected[:, d], pcts))
        self.bin_edges = np.array(edges)  # (proj_dim, bins_per_dim-1)

    def project(self, obs):
        """obs: flat float32 (obs_dim,). Returns (proj_dim,) PCA-projected."""
        return (obs - self.mean) @ self.P.T

    def quantize(self, proj):
        """proj: (proj_dim,). Returns tuple of bin indices."""
        cell = []
        for d in range(self.proj_dim):
            idx = int(np.searchsorted(self.bin_edges[d], proj[d]))
            cell.append(idx)
        return tuple(cell)

    def step(self, obs):
        """obs: flat float32. Returns action index."""
        if self.bin_edges is None:
            self.warmup_buf.append(obs.copy())
            if len(self.warmup_buf) >= self.warmup_n:
                self.compute_pca_and_bins()
            return int(np.random.randint(self.n_actions))

        proj = self.project(obs)
        cell_id = self.quantize(proj)
        self.cells.add(cell_id)

        if self.prev_cell is not None and self.prev_action is not None:
            key = (self.prev_cell, self.prev_action)
            if key not in self.edges:
                self.edges[key] = {}
            self.edges[key][cell_id] = self.edges[key].get(cell_id, 0) + 1

        visit_counts = [
            sum(self.edges.get((cell_id, a), {}).values())
            for a in range(self.n_actions)
        ]
        min_count = min(visit_counts)
        candidates = [a for a, c in enumerate(visit_counts) if c == min_count]
        action = candidates[int(np.random.randint(len(candidates)))]

        self.prev_cell = cell_id
        self.prev_action = action
        return action


def avgpool16(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    return arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def run_structural_test():
    """Structural sanity check < 30s."""
    print("Structural test...", flush=True)
    t0 = time.time()

    g = PCAGridGraph(proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256)
    rng = np.random.RandomState(0)

    for i in range(600):
        obs = rng.randn(256).astype(np.float32)
        g.step(obs)

    assert g.bin_edges is not None, "R1: PCA/bins not computed after 600 warmup steps"
    assert g.P is not None, "R1: PCA projection matrix not set"
    assert g.P.shape == (8, 256), f"R1: P shape {g.P.shape}"
    assert g.bin_edges.shape == (8, 3), f"R1: bin_edges shape {g.bin_edges.shape}"

    actions = []
    for i in range(1000):
        obs = rng.randn(256).astype(np.float32)
        action = g.step(obs)
        actions.append(action)

    from collections import Counter
    counts = Counter(actions)
    dom = max(counts.values()) / len(actions)
    unique_a = len(counts)

    assert unique_a == 4, f"R2: Only {unique_a}/4 actions used"
    assert dom < 0.5, f"R3: dom={dom:.0%} > 50%"
    assert len(g.cells) > 0, "R4: No cells created"
    assert len(g.edges) > 0, "R5: No edges created"
    assert not hasattr(g, 'nodes'), "R6: .nodes attribute found (codebook DNA)"

    elapsed = time.time() - t0
    print(f"  PASS: cells={len(g.cells)}  edges={len(g.edges)}  dom={dom:.0%}  {elapsed:.1f}s",
          flush=True)
    return True


def run_seed(arc, game_id, seed, max_steps=30000):
    from arcengine import GameState
    np.random.seed(seed)
    g = PCAGridGraph(proj_dim=8, bins_per_dim=4, n_actions=4, obs_dim=256)
    env = arc.make(game_id)
    obs = env.reset()
    na = len(env.action_space)

    ts = go = lvls = 0
    unique_cells = set()
    action_counts = [0] * na
    level_step = None
    t0 = time.time()

    while ts < max_steps:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); continue

        pooled = avgpool16(obs.frame)
        action_idx = g.step(pooled)

        if g.bin_edges is not None:
            unique_cells.add(g.quantize(g.project(pooled)))

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
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
            print(f"  LEVEL {lvls} at step {ts}  unique_cells={len(unique_cells)}"
                  f"  cells={len(g.cells)}  edges={len(g.edges)}", flush=True)

        if time.time() - t0 > 280:
            break

    elapsed = time.time() - t0
    dom = max(action_counts) / max(sum(action_counts), 1) * 100
    return {
        'seed': seed, 'levels': lvls, 'level_step': level_step,
        'unique_cells': len(unique_cells), 'cells': len(g.cells),
        'edges': len(g.edges), 'dom': dom, 'elapsed': elapsed,
    }


def main():
    import arc_agi
    print("Step 447: PCA Grid Graph on LS20 (no cosine, no attract)", flush=True)
    print("PCA projection (data-aligned, frozen after warmup). 30K steps, 3 seeds.", flush=True)
    print(flush=True)

    if not run_structural_test():
        print("STRUCTURAL FAIL — stopping.", flush=True)
        return
    print(flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP: LS20 not found"); return

    t_total = time.time()
    results = []
    for seed in [0, 1, 2]:
        print(f"--- Seed {seed} ---", flush=True)
        r = run_seed(arc, ls20.game_id, seed=seed, max_steps=30000)
        status = f"LEVEL 1 at step {r['level_step']}" if r['levels'] > 0 else "no level"
        print(f"  seed={seed}  {status:26s}  unique_cells={r['unique_cells']:4d}"
              f"  cells={r['cells']:4d}  edges={r['edges']:5d}"
              f"  dom={r['dom']:.0f}%  {r['elapsed']:.0f}s", flush=True)
        results.append(r)

    print(f"\n{'='*60}", flush=True)
    print("STEP 447 FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)

    wins = [r for r in results if r['levels'] > 0]
    avg_cells = sum(r['unique_cells'] for r in results) / len(results)
    avg_dom = sum(r['dom'] for r in results) / len(results)

    print(f"Reliability: {len(wins)}/3", flush=True)
    if wins:
        print(f"Step-to-level: {sorted([r['level_step'] for r in wins])}", flush=True)
    print(f"Avg unique_cells: {avg_cells:.0f}", flush=True)
    print(f"Avg dom: {avg_dom:.0f}%", flush=True)
    print(f"Total elapsed: {time.time() - t_total:.0f}s", flush=True)
    print(flush=True)

    if wins:
        print("PCA NAVIGATES: data-aligned projection recovers cosine's contribution.", flush=True)
        print("Random projection was the bottleneck. Graph family validated.", flush=True)
    elif avg_dom <= 50 and avg_cells >= 50:
        print("0/3 at 30K with healthy dynamics (~2000 cells).", flush=True)
        print("Adaptive placement (moving nodes toward data) is necessary.", flush=True)
        print("Navigation is partially cosine-mediated. Graph family collapses toward codebook DNA.", flush=True)
    else:
        print("DEAD dynamics. Different failure mode.", flush=True)

    print(flush=True)
    print(f"Comparison:", flush=True)
    print(f"  Random 446b: 0/3 at 30K  unique_cells=1869  dom=25%", flush=True)
    print(f"  Cosine  442b: 1/3 at 30K  unique_obs=4461-5516  dom=25%", flush=True)


if __name__ == '__main__':
    main()
