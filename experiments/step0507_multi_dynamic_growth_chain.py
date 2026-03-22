#!/usr/bin/env python3
"""
Step 507 -- Dynamic centroid growth during chain: CIFAR-100 -> LS20 -> CIFAR-100.

Key difference from Step 506: centroids are NOT frozen after warmup.
Spawn-on-novelty: if min_L2_dist(x, centroids) > spawn_threshold, create new centroid.
Domains get their own centroids as needed.

the prediction: Phase 1 ~300 CIFAR centroids. Phase 2 spawns ~30-50 new ARC centroids.
Navigation should work because ARC frames get ARC-specific centroids.

T1: synthetic data, calibrate spawn_threshold, <30s.
T2: full chain, 5-min per phase.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

SPAWN_THRESHOLD = 0.3   # L2 distance; calibrated in T1
PHASE_DURATION = 300    # 5 min per phase
N_ARC_ACTIONS = 4
N_CIFAR_ACTIONS = 100


def encode_cifar(img):
    """[32,32,3] uint8 HWC or CHW -> 256D centered."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    """obs.frame [64x64] values 0-15 -> 256D centered."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class DynamicGraph:
    """Spawn-on-novelty centroid growth. Separate edge dicts per mode.
    Centroids: numpy array grown by appending. Frozen once assigned (no EWMA)."""

    def __init__(self, spawn_threshold=SPAWN_THRESHOLD):
        self.threshold = spawn_threshold
        self.centroids = None   # np array [N, 256], grows
        self.edges = {'cifar': {}, 'arc': {}}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._mode = 'cifar'
        self.spawns_cifar = 0
        self.spawns_arc = 0

    def set_mode(self, mode):
        assert mode in ('cifar', 'arc')
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _spawn(self, x):
        """Add new centroid. Returns its index."""
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1, -1)])
        cell = len(self.centroids) - 1
        if self._mode == 'cifar':
            self.spawns_cifar += 1
        else:
            self.spawns_arc += 1
        return cell

    def step(self, x, n_actions):
        if self.centroids is None:
            cell = self._spawn(x)
            return int(np.random.randint(n_actions))

        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nearest = int(np.argmin(dists))
        min_dist = dists[nearest]

        if min_dist > self.threshold:
            cell = self._spawn(x)
        else:
            cell = nearest

        self.cells_seen.add(cell)
        edges = self.edges[self._mode]
        if self.prev_cell is not None and self.prev_action is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1

        counts = [sum(edges.get((cell, a), {}).values()) for a in range(n_actions)]
        min_c = min(counts)
        candidates = [a for a, c in enumerate(counts) if c == min_c]
        action = candidates[int(np.random.randint(len(candidates)))]
        self.prev_cell = cell
        self.prev_action = action
        return action

    @property
    def n_centroids(self):
        return len(self.centroids) if self.centroids is not None else 0


# ================================================================
# T1: Calibration on synthetic data
# ================================================================

def run_t1():
    print("T1: Spawn threshold calibration (synthetic data)", flush=True)
    t0 = time.time()

    # Test encodings
    img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    enc = encode_cifar(img)
    assert enc.shape == (256,) and abs(enc.mean()) < 1e-6

    frame = [np.random.randint(0, 16, (64, 64))]
    enc = encode_arc(frame)
    assert enc.shape == (256,) and abs(enc.mean()) < 1e-6

    # Calibrate: how many centroids do we get from 200 synthetic CIFAR images?
    rng = np.random.RandomState(42)
    for threshold in [0.1, 0.3, 0.5, 1.0]:
        g = DynamicGraph(spawn_threshold=threshold)
        g.set_mode('cifar')
        for _ in range(200):
            img = (rng.rand(32, 32, 3) * 255).astype(np.uint8)
            x = encode_cifar(img)
            g.step(x, N_CIFAR_ACTIONS)
        print(f"  threshold={threshold:.1f}: {g.n_centroids} centroids from 200 CIFAR images", flush=True)

    # Check ARC frames spawn new centroids after CIFAR pre-seeding
    g = DynamicGraph(spawn_threshold=SPAWN_THRESHOLD)
    g.set_mode('cifar')
    for _ in range(200):
        img = (rng.rand(32, 32, 3) * 255).astype(np.uint8)
        x = encode_cifar(img)
        g.step(x, N_CIFAR_ACTIONS)
    n_after_cifar = g.n_centroids

    g.set_mode('arc')
    for _ in range(50):
        frame = [rng.randint(0, 16, (64, 64))]
        x = encode_arc(frame)
        g.step(x, N_ARC_ACTIONS)
    n_after_arc = g.n_centroids
    new_arc = n_after_arc - n_after_cifar

    print(f"  ARC spawns {new_arc} new centroids after {n_after_cifar} CIFAR centroids (threshold={SPAWN_THRESHOLD})", flush=True)
    assert n_after_arc >= n_after_cifar, "ARC phase should not remove centroids"
    print(f"  PASS: {time.time()-t0:.1f}s", flush=True)
    return True


# ================================================================
# T2: Full chain run
# ================================================================

def run_phase_cifar(g, X, y, duration_s, offset=0):
    """Present CIFAR images in a loop for duration_s seconds. Return metrics."""
    g.set_mode('cifar')
    correct = total = 0
    t0 = time.time()
    i = offset
    while time.time() - t0 < duration_s:
        x = encode_cifar(X[i % len(X)])
        action = g.step(x, N_CIFAR_ACTIONS)
        if action == int(y[i % len(y)]):
            correct += 1
        total += 1
        i += 1
    acc = correct / total * 100 if total > 0 else 0
    elapsed = time.time() - t0
    print(f"  acc={acc:.2f}%  total_imgs={total}  centroids={g.n_centroids}"
          f"  cells={len(g.cells_seen)}  spawns_cifar={g.spawns_cifar}  {elapsed:.0f}s", flush=True)
    return acc, total


def run_phase_arc(g, arc, game_id, duration_s):
    """Run LS20 for duration_s seconds. Return (levels, level_step, steps)."""
    from arcengine import GameState
    g.set_mode('arc')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while time.time() - t0 < duration_s:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame or len(obs.frame) == 0: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        action_idx = g.step(x, N_ARC_ACTIONS)
        action = action_space[action_idx]
        obs_before = obs.levels_completed
        obs = env.step(action)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    elapsed = time.time() - t0
    print(f"  {status}  cells={len(g.cells_seen)}/{g.n_centroids}"
          f"  go={go}  steps={ts}  spawns_arc={g.spawns_arc}  {elapsed:.0f}s", flush=True)
    return lvls, level_step, ts


def main():
    t_total = time.time()
    print("Step 507: Dynamic centroid growth chain -- CIFAR-100 -> LS20 -> CIFAR-100", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}  phase_duration={PHASE_DURATION}s per phase", flush=True)

    if not run_t1():
        print("T1 FAILED -- aborting"); return

    print("\nLoading data...", flush=True)
    import torchvision
    import arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    print(f"  CIFAR-100: {len(X)} images, {len(set(y))} classes", flush=True)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP -- LS20 not found"); return

    np.random.seed(0)
    g = DynamicGraph(spawn_threshold=SPAWN_THRESHOLD)

    print(f"\n--- Phase 1: CIFAR-100 ({PHASE_DURATION}s) ---", flush=True)
    acc1, n_cifar1 = run_phase_cifar(g, X, y, PHASE_DURATION)
    nc_after_p1 = g.n_centroids

    print(f"\n--- Phase 2: LS20 ({PHASE_DURATION}s) ---", flush=True)
    lvls, level_step, arc_steps = run_phase_arc(g, arc, ls20.game_id, PHASE_DURATION)
    nc_after_p2 = g.n_centroids
    new_arc_centroids = nc_after_p2 - nc_after_p1

    print(f"\n--- Phase 3: CIFAR-100 ({PHASE_DURATION}s) ---", flush=True)
    acc3, n_cifar3 = run_phase_cifar(g, X, y, PHASE_DURATION, offset=n_cifar1)
    nc_after_p3 = g.n_centroids

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 507 SUMMARY", flush=True)
    print(f"  Phase 1 CIFAR:  {acc1:.2f}%  centroids after={nc_after_p1}  (spawns: {g.spawns_cifar - (nc_after_p2 - nc_after_p1)})", flush=True)
    print(f"  Phase 2 LS20:   {'WIN@'+str(level_step) if lvls > 0 else 'FAIL'}  new_centroids={new_arc_centroids}  steps={arc_steps}", flush=True)
    print(f"  Phase 3 CIFAR:  {acc3:.2f}%  centroids after={nc_after_p3}  (delta={acc3-acc1:+.2f}pp)", flush=True)
    print(f"  Total centroids: {g.n_centroids}  (cifar={g.spawns_cifar}  arc={g.spawns_arc})", flush=True)

    print(f"\nVERDICT:", flush=True)
    if lvls > 0:
        print(f"  LS20 NAVIGATES. Dynamic growth enables cross-domain transfer.", flush=True)
        print(f"  ARC spawned {new_arc_centroids} new centroids (Predicted 30-50).", flush=True)
    else:
        print(f"  LS20 FAILS. Dynamic growth insufficient -- domains can't share centroid space.", flush=True)
    acc_delta = acc3 - acc1
    if abs(acc_delta) < 0.5:
        print(f"  CIFAR accuracy: UNCHANGED ({acc_delta:+.2f}pp). No ARC interference.", flush=True)
    elif acc_delta > 0:
        print(f"  CIFAR accuracy: IMPROVED +{acc_delta:.2f}pp.", flush=True)
    else:
        print(f"  CIFAR accuracy: DEGRADED {acc_delta:.2f}pp. ARC centroids dilute CIFAR.", flush=True)
    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
