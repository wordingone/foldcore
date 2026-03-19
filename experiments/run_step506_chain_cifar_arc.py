#!/usr/bin/env python3
"""
Step 506 — Chain test: CIFAR-100 -> ARC-AGI-3 (LS20) -> CIFAR-100.
One k-means graph substrate, state persists across benchmark transitions.

Phase 1: 1000 CIFAR-100 test images. Track centroids built, classification accuracy.
Phase 2: WITHOUT resetting state, run LS20. 50K steps, 1 seed. Track navigation.
Phase 3: Back to CIFAR-100 (next 1000 images). Track accuracy with ARC-expanded state.

Encoding: both benchmarks -> 256D via avgpool-to-16x16 + centered_enc.
  CIFAR: 32x32 RGB -> grayscale -> 2x2 avgpool -> 16x16 = 256D
  ARC:   64x64 values 0-15 -> 4x4 avgpool -> 16x16 = 256D

T1 (synthetic, <30s): run before importing heavy deps.
T2 (real data): full chain run.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

N_CIFAR_P1 = 1000
N_CIFAR_P3 = 1000
MAX_ARC_STEPS = 50_000
TIME_CAP_ARC = 270   # 4.5 min
WARMUP = 500
N_CLUSTERS = 300
N_ARC_ACTIONS = 4
N_CIFAR_ACTIONS = 100


def encode_cifar(img):
    """img: np array [32,32,3] HWC or [3,32,32] CHW, uint8. Returns 256D centered."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    """frame: obs.frame (list/array 64x64, values 0-15). Returns 256D centered."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class ChainGraph:
    """k-means graph with separate edge dicts per benchmark mode.
    Centroids are shared (persist across all phases). Edges are mode-scoped."""

    def __init__(self, n_clusters=N_CLUSTERS, warmup=WARMUP):
        self.n_clusters = n_clusters
        self.warmup = warmup
        self.centroids = None
        self.edges = {'cifar': {}, 'arc': {}}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._buf = []
        self._mode = 'cifar'

    def set_mode(self, mode):
        assert mode in ('cifar', 'arc')
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _fit(self):
        from sklearn.cluster import MiniBatchKMeans
        X = np.array(self._buf, dtype=np.float32)
        n = min(self.n_clusters, len(set(x.tobytes() for x in X)), len(X))
        n = max(n, 2)
        km = MiniBatchKMeans(n_clusters=n, random_state=42, n_init=3, max_iter=100, batch_size=256)
        km.fit(X)
        self.centroids = km.cluster_centers_.astype(np.float32)
        self._buf = []

    def step(self, x, n_actions):
        if self.centroids is None:
            self._buf.append(x.copy())
            if len(self._buf) >= self.warmup:
                self._fit()
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        cell = int(np.argmin(np.sum(diffs * diffs, axis=1)))
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


# ============================================================
# T1: Structural test (synthetic data, no heavy deps)
# ============================================================

def run_t1():
    print("T1: Structural test (synthetic data)", flush=True)
    t0 = time.time()

    # Test encode_cifar
    img_rgb = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    enc = encode_cifar(img_rgb)
    assert enc.shape == (256,), f"encode_cifar shape {enc.shape}"
    assert abs(enc.mean()) < 1e-6, f"encode_cifar not centered: mean={enc.mean()}"

    # Test encode_arc (synthetic 64x64 frame as list-of-array)
    frame = [np.random.randint(0, 16, (64, 64)).tolist()]
    enc = encode_arc(frame)
    assert enc.shape == (256,), f"encode_arc shape {enc.shape}"
    assert abs(enc.mean()) < 1e-6, f"encode_arc not centered: mean={enc.mean()}"

    # Test ChainGraph: 600 CIFAR-like steps, then 200 ARC-like steps
    g = ChainGraph(n_clusters=50, warmup=200)
    g.set_mode('cifar')
    for _ in range(600):
        img = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
        x = encode_cifar(img)
        a = g.step(x, N_CIFAR_ACTIONS)
        assert 0 <= a < N_CIFAR_ACTIONS, f"CIFAR action out of range: {a}"
    assert g.centroids is not None, "Centroids not fit after 600 CIFAR steps (warmup=200)"
    n_c_after_cifar = len(g.centroids)

    g.set_mode('arc')
    for _ in range(200):
        frame = [np.random.randint(0, 16, (64, 64)).tolist()]
        x = encode_arc(frame)
        a = g.step(x, N_ARC_ACTIONS)
        assert 0 <= a < N_ARC_ACTIONS, f"ARC action out of range: {a}"
    n_c_after_arc = len(g.centroids)
    assert n_c_after_arc == n_c_after_cifar, "Centroids changed during ARC phase (should be frozen after fit)"
    assert len(g.cells_seen) > 0, "No cells seen during ARC phase"

    print(f"  PASS: encode_cifar/arc=256D centered. Centroids fit={n_c_after_cifar}. "
          f"Cells={len(g.cells_seen)}. CIFAR/ARC edges separate. {time.time()-t0:.1f}s", flush=True)
    return True


# ============================================================
# T2: Full chain run
# ============================================================

def run_phase1_cifar(g, X, y):
    g.set_mode('cifar')
    correct = 0
    n = min(N_CIFAR_P1, len(X))
    t0 = time.time()
    for i in range(n):
        x = encode_cifar(X[i])
        action = g.step(x, N_CIFAR_ACTIONS)
        if action == int(y[i]):
            correct += 1
    acc = correct / n * 100
    n_c = len(g.centroids) if g.centroids is not None else 0
    print(f"  acc={acc:.2f}%  centroids={n_c}  cells={len(g.cells_seen)}  {time.time()-t0:.0f}s", flush=True)
    return acc, n_c


def run_phase2_arc(g, arc, game_id):
    from arcengine import GameState
    g.set_mode('arc')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_ARC_STEPS and time.time() - t0 < TIME_CAP_ARC:
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
    n_c = len(g.centroids) if g.centroids is not None else 0
    status = f"WIN@{level_step}" if lvls > 0 else f"FAIL"
    print(f"  {status}  cells={len(g.cells_seen)}/{n_c}  go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls, level_step, n_c


def run_phase3_cifar(g, X, y, offset):
    g.set_mode('cifar')
    correct = 0
    n = min(N_CIFAR_P3, len(X) - offset)
    t0 = time.time()
    for i in range(offset, offset + n):
        x = encode_cifar(X[i])
        action = g.step(x, N_CIFAR_ACTIONS)
        if action == int(y[i]):
            correct += 1
    acc = correct / n * 100
    n_c = len(g.centroids) if g.centroids is not None else 0
    print(f"  acc={acc:.2f}%  centroids={n_c}  cells={len(g.cells_seen)}  {time.time()-t0:.0f}s", flush=True)
    return acc, n_c


def main():
    t_total = time.time()
    print("Step 506: Chain benchmark -- CIFAR-100 -> LS20 -> CIFAR-100", flush=True)
    print(f"n_clusters={N_CLUSTERS} warmup={WARMUP} arc_steps={MAX_ARC_STEPS//1000}K", flush=True)

    # T1
    if not run_t1():
        print("T1 FAILED — aborting"); return

    # Load deps
    print("\nLoading data...", flush=True)
    import torchvision
    ds = torchvision.datasets.CIFAR100('C:/Users/Admin/cifar100_data', train=False, download=True)
    X = np.array(ds.data)    # [10000, 32, 32, 3]
    y = np.array(ds.targets)
    print(f"  CIFAR-100: {len(X)} test images, {len(set(y))} classes", flush=True)

    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next((g for g in games if 'ls20' in g.game_id.lower()), None)
    if not ls20:
        print("SKIP — LS20 not found"); return

    np.random.seed(0)
    g = ChainGraph(n_clusters=N_CLUSTERS, warmup=WARMUP)

    print(f"\n--- Phase 1: CIFAR-100 ({N_CIFAR_P1} images) ---", flush=True)
    acc1, nc1 = run_phase1_cifar(g, X, y)

    print(f"\n--- Phase 2: LS20 ({MAX_ARC_STEPS//1000}K steps, {TIME_CAP_ARC}s cap) ---", flush=True)
    lvls, level_step, nc2 = run_phase2_arc(g, arc, ls20.game_id)

    print(f"\n--- Phase 3: CIFAR-100 ({N_CIFAR_P3} images, offset {N_CIFAR_P1}) ---", flush=True)
    acc3, nc3 = run_phase3_cifar(g, X, y, offset=N_CIFAR_P1)

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 506 SUMMARY", flush=True)
    print(f"  Phase 1 CIFAR:  {acc1:.2f}%  (chance=1.00%)", flush=True)
    print(f"  Phase 2 LS20:   {'WIN@'+str(level_step) if lvls > 0 else 'FAIL'}  centroids={nc2}", flush=True)
    print(f"  Phase 3 CIFAR:  {acc3:.2f}%  (delta={acc3-acc1:+.2f}pp vs Phase 1)", flush=True)

    print(f"\nVERDICT:", flush=True)
    if lvls > 0:
        print(f"  LS20 NAVIGATES with CIFAR-pretrained centroids.", flush=True)
    else:
        print(f"  LS20 FAILS in chain. CIFAR centroids may not map ARC frames usefully.", flush=True)
    acc_delta = acc3 - acc1
    if abs(acc_delta) < 0.5:
        print(f"  CIFAR accuracy: UNCHANGED ({acc_delta:+.2f}pp). No cross-task interference.", flush=True)
    elif acc_delta > 0:
        print(f"  CIFAR accuracy: IMPROVED +{acc_delta:.2f}pp. ARC state helps CIFAR.", flush=True)
    else:
        print(f"  CIFAR accuracy: DEGRADED {acc_delta:.2f}pp. ARC state hurts CIFAR.", flush=True)
    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
