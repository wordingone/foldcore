#!/usr/bin/env python3
"""
Step 515 -- K-means negative transfer: frozen CIFAR centroids on LS20.
Analog of Step 506 (codebook negative transfer).

Fit k-means n=300 on 1000 CIFAR-100 images. Freeze. Run LS20 50K steps.
Baseline: codebook Step 506 FAIL (0/1), dynamic growth Step 507 WIN@11170.
Prediction: 0/1 at 50K (same failure as codebook). Kill: LS20 WINS -> transfer is codebook-specific.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

N_CLUSTERS = 300
N_CIFAR_WARMUP = 1000
MAX_LS20_STEPS = 50_000
N_ARC_ACTIONS = 4
N_CIFAR_ACTIONS = 100


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class FrozenKMeansGraph:
    def __init__(self):
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self._mode = None
        self.cells_seen_per_mode = {}

    def fit(self, X_feats):
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(N_CLUSTERS, len(X_feats)),
                    random_state=42, n_init=3, max_iter=100)
        km.fit(X_feats)
        self.centroids = km.cluster_centers_.astype(np.float32)
        print(f"  k-means fit: {len(self.centroids)} centroids on {len(X_feats)} obs", flush=True)

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def step(self, x, n_actions):
        if self.centroids is None:
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
        self.cells_seen_per_mode.setdefault(self._mode, set()).add(cell)
        edges = self.edges.setdefault(self._mode, {})
        if self.prev_cell is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(edges.get((cell, a), {}).values()) for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_ls20(g, arc, game_id, max_steps=MAX_LS20_STEPS):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < max_steps:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    ls20_cells = g.cells_seen_per_mode.get('ls20', set())
    cifar_cells = g.cells_seen_per_mode.get('cifar', set())
    overlap = ls20_cells & cifar_cells
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  LS20: {status}  ls20_cells={len(ls20_cells)}/{N_CLUSTERS}  "
          f"overlap_with_cifar={len(overlap)}  go={go}  steps={ts}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls, level_step, len(ls20_cells), len(overlap)


def main():
    t_total = time.time()
    print("Step 515: K-means negative transfer (frozen CIFAR centroids -> LS20)", flush=True)
    print(f"n_clusters={N_CLUSTERS}  cifar_warmup={N_CIFAR_WARMUP}  max_ls20={MAX_LS20_STEPS//1000}K", flush=True)
    print(f"Analog: codebook Step 506 (FAIL). Prediction: FAIL (0/1).", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    # Build and fit k-means on CIFAR
    print(f"\nEncoding {N_CIFAR_WARMUP} CIFAR images...", flush=True)
    np.random.seed(0)
    feats = np.array([encode_cifar(X[i]) for i in range(N_CIFAR_WARMUP)], dtype=np.float32)

    g = FrozenKMeansGraph()
    g.set_mode('cifar')
    # Register CIFAR cells
    for i in range(N_CIFAR_WARMUP):
        g.step(feats[i], N_CIFAR_ACTIONS)

    print(f"Fitting k-means on CIFAR features...", flush=True)
    g2 = FrozenKMeansGraph()
    g2.fit(feats)
    # Register CIFAR cells on fitted g2
    g2.set_mode('cifar')
    for i in range(N_CIFAR_WARMUP):
        g2.step(feats[i], N_CIFAR_ACTIONS)
    cifar_cells = g2.cells_seen_per_mode.get('cifar', set())
    print(f"  CIFAR cells used: {len(cifar_cells)}/{N_CLUSTERS}", flush=True)

    print(f"\n--- LS20 ({MAX_LS20_STEPS//1000}K steps, frozen CIFAR centroids) ---", flush=True)
    lvls, level_step, n_ls20_cells, n_overlap = run_ls20(g2, arc, ls20.game_id)

    print(f"\n{'='*60}", flush=True)
    print("STEP 515 SUMMARY", flush=True)
    print(f"  Centroids: {N_CLUSTERS} (frozen from {N_CIFAR_WARMUP} CIFAR images)", flush=True)
    print(f"  CIFAR cells used: {len(cifar_cells)}/{N_CLUSTERS}", flush=True)
    print(f"  LS20 result: {'WIN@'+str(level_step) if lvls>0 else 'FAIL'}", flush=True)
    print(f"  LS20 cells: {n_ls20_cells}/{N_CLUSTERS}  overlap with CIFAR: {n_overlap}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if lvls > 0:
        print(f"  UNEXPECTED WIN: LS20 navigates with frozen CIFAR centroids.", flush=True)
        print(f"  Negative transfer is codebook-specific. K-means doesn't show it.", flush=True)
    else:
        print(f"  EXPECTED FAIL: LS20 cannot navigate with frozen CIFAR centroids.", flush=True)
        print(f"  Consistent with codebook Step 506. Negative transfer is universal.", flush=True)
        if n_overlap == 0:
            print(f"  Zero overlap: CIFAR and ARC frames occupy different centroid regions.", flush=True)
        else:
            print(f"  {n_overlap} shared cells: partial overlap, but insufficient for navigation.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
