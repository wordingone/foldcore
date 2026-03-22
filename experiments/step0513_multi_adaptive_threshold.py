#!/usr/bin/env python3
"""
Step 513 -- Domain-adaptive threshold via local density.

Spawn threshold = median pairwise L2 of K=10 nearest existing centroids * 0.5
Dense regions (ARC, L2~0.5) auto-calibrate fine threshold.
Sparse regions (CIFAR, L2~4.3) auto-calibrate coarse threshold.

Chain: CIFAR 1-pass -> LS20 5-min -> CIFAR 1-pass
Report: centroid count per domain, LS20 navigation, CIFAR NMI
Kill: 0/1 LS20 or NMI < 0.20
"""
import time
import numpy as np
import logging
logging.getLogger().setLevel(logging.WARNING)

K_LOCAL = 10         # nearest centroids for density estimate
DENSITY_SCALE = 0.5  # threshold = median_pairwise * scale
N_ARC_ACTIONS_LS20 = 4


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


class AdaptiveGraph:
    """Local-density adaptive spawn threshold. Separate edges per mode."""

    def __init__(self, k_local=K_LOCAL, density_scale=DENSITY_SCALE):
        self.k_local = k_local
        self.density_scale = density_scale
        self.centroids = None
        self.edges = {}      # mode -> {(cell, action): {next_cell: count}}
        self.prev_cell = None
        self.prev_action = None
        self._mode = None
        self.spawns = {}     # mode -> count
        self._thresholds_sampled = []  # for diagnostics

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _local_threshold(self, x):
        """Compute adaptive threshold from K nearest centroid pairwise distances."""
        n = len(self.centroids)
        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        k = min(self.k_local, n)
        k_idx = np.argpartition(dists, k-1)[:k] if k < n else np.arange(n)
        k_centroids = self.centroids[k_idx]
        # Pairwise L2 among k nearest
        if k <= 1:
            return dists[int(np.argmin(dists))] * self.density_scale
        pw_dists = []
        for i in range(k):
            for j in range(i+1, k):
                d = np.sqrt(np.sum((k_centroids[i] - k_centroids[j])**2))
                pw_dists.append(d)
        threshold = float(np.median(pw_dists)) * self.density_scale
        return max(threshold, 1e-6)

    def _spawn(self, x):
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1, -1)])
        self.spawns[self._mode] = self.spawns.get(self._mode, 0) + 1
        return len(self.centroids) - 1

    def step(self, x, n_actions):
        if self.centroids is None or len(self.centroids) < self.k_local:
            cell = self._spawn(x)
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

        threshold = self._local_threshold(x)
        self._thresholds_sampled.append(threshold)

        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nearest = int(np.argmin(dists))
        cell = self._spawn(x) if dists[nearest] > threshold else nearest

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

    @property
    def n_centroids(self):
        return len(self.centroids) if self.centroids is not None else 0


def run_cifar(g, X, y, label, record_assignments=False):
    g.set_mode('cifar')
    correct = 0
    assignments = np.zeros(len(X), dtype=np.int32) if record_assignments else None
    cifar_start = g.spawns.get('cifar', 0)
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = g.step(x, 100)
        if record_assignments:
            # find nearest centroid after step
            diffs = g.centroids - x
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            assignments[i] = int(np.argmin(dists))
        if a == int(y[i]):
            correct += 1
    acc = correct / len(X) * 100
    cifar_spawned = g.spawns.get('cifar', 0) - cifar_start
    thresh_info = ""
    if g._thresholds_sampled:
        recent = g._thresholds_sampled[-500:]
        thresh_info = f"  threshold_median={np.median(recent):.3f}"
    print(f"  {label}: acc={acc:.2f}%  centroids={g.n_centroids}  "
          f"spawns_cifar={g.spawns.get('cifar',0)}{thresh_info}  {time.time()-t0:.0f}s", flush=True)
    return acc, assignments


def cifar_nmi(assignments, labels, n_c):
    from sklearn.metrics import normalized_mutual_info_score
    # Only use labels for images that map to a CIFAR-range centroid
    nmi = normalized_mutual_info_score(labels, assignments, average_method='arithmetic')
    purities = []
    for c in range(n_c):
        mask = assignments == c
        if mask.sum() == 0:
            continue
        counts = np.bincount(labels[mask], minlength=100)
        purities.append(counts.max() / mask.sum())
    purity = float(np.mean(purities)) if purities else 0.0
    n_pure = sum(p > 0.5 for p in purities)
    return nmi, purity, n_pure


def run_ls20(g, arc, game_id, duration=300):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while time.time() - t0 < duration:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS_LS20)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    thresh_info = ""
    if g._thresholds_sampled:
        recent = g._thresholds_sampled[-500:]
        thresh_info = f"  threshold_median={np.median(recent):.3f}"
    print(f"  LS20: {status}  centroids={g.n_centroids}  "
          f"spawns_ls20={g.spawns.get('ls20',0)}  go={go}  steps={ts}{thresh_info}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


def main():
    t_total = time.time()
    print("Step 513: Domain-adaptive threshold via local density", flush=True)
    print(f"k_local={K_LOCAL}  density_scale={DENSITY_SCALE}", flush=True)

    print("\nLoading...", flush=True)
    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    print(f"  CIFAR {len(X)} images, LS20 ready", flush=True)

    np.random.seed(0)
    g = AdaptiveGraph(k_local=K_LOCAL, density_scale=DENSITY_SCALE)

    print("\n--- Phase 1: CIFAR (1-pass) ---", flush=True)
    acc1, asgn1 = run_cifar(g, X, y, "P1", record_assignments=True)
    n_cifar_centroids = g.spawns.get('cifar', 0)
    nmi1, pur1, npure1 = cifar_nmi(asgn1, y, g.n_centroids)
    print(f"  CIFAR NMI={nmi1:.4f}  purity={pur1:.4f}  n_pure>50%={npure1}/{g.n_centroids}", flush=True)

    print("\n--- Phase 2: LS20 (5-min) ---", flush=True)
    ls20_lvls, ls20_step = run_ls20(g, arc, ls20.game_id, 300)

    print("\n--- Phase 3: CIFAR (1-pass) ---", flush=True)
    acc3, asgn3 = run_cifar(g, X, y, "P3", record_assignments=True)
    # NMI for CIFAR images only (map to their nearest centroid post-chain)
    nmi3, pur3, npure3 = cifar_nmi(asgn3, y, g.n_centroids)
    print(f"  CIFAR NMI={nmi3:.4f}  purity={pur3:.4f}  n_pure>50%={npure3}/{g.n_centroids}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("STEP 513 SUMMARY", flush=True)
    print(f"  CIFAR centroids:   {n_cifar_centroids}", flush=True)
    print(f"  LS20 centroids:    {g.spawns.get('ls20', 0)}", flush=True)
    print(f"  Total centroids:   {g.n_centroids}", flush=True)
    print(f"  P1 CIFAR NMI:  {nmi1:.4f}  purity={pur1:.4f}", flush=True)
    print(f"  LS20 result:   {'WIN@'+str(ls20_step) if ls20_lvls > 0 else 'FAIL'}", flush=True)
    print(f"  P3 CIFAR NMI:  {nmi3:.4f}  purity={pur3:.4f}  (delta_acc={acc3-acc1:+.2f}pp)", flush=True)

    print(f"\nVERDICT:", flush=True)
    nav_ok = ls20_lvls > 0
    nmi_ok = nmi1 >= 0.20
    if nav_ok and nmi_ok:
        print(f"  PASS: LS20 navigates + NMI={nmi1:.4f} >= 0.20.", flush=True)
        print(f"  Adaptive threshold self-calibrates across domains.", flush=True)
    elif not nav_ok:
        print(f"  FAIL (navigation): LS20 did not navigate.", flush=True)
    else:
        print(f"  FAIL (classification): NMI={nmi1:.4f} < 0.20.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
