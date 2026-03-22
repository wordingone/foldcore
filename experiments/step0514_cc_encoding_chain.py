#!/usr/bin/env python3
"""
Step 514 -- Connected-component encoding (Rudakov-style) on chain.
ARC: segment 64x64 into single-color CCs, build 128D CC feature vector.
CIFAR: keep avgpool16+centered (CC not applicable to RGB photos).

Chain: CIFAR 1-pass -> LS20 50K steps -> CIFAR 1-pass.
Baseline: avgpool16 chain (Step 508): WIN@11170 LS20.
Kill: LS20 fails (0/1).

CC features (per frame, 128D -> zero-pad 256D):
  For each of 16 colors: [n_cc/20, fraction, mean_r, mean_c, std_r, std_c, max_frac, n_large/10]
  = 16 * 8 = 128D, padded to 256D, then centered.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

SPAWN_THRESHOLD = 0.3
N_ARC_ACTIONS = 4
N_CIFAR_ACTIONS = 100
MAX_LS20_STEPS = 50_000


# ---- Encoders ----

def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc_cc(frame):
    """ARC 64x64 int-values-0-15 -> 256D CC feature, centered."""
    from scipy.ndimage import label
    arr = np.array(frame[0], dtype=np.int8)
    feats = np.zeros(128, dtype=np.float32)
    for c in range(16):
        base = c * 8
        mask = (arr == c)
        total = mask.sum()
        if total == 0:
            continue
        feats[base + 1] = float(total) / 4096.0          # fraction
        rows, cols = np.where(mask)
        feats[base + 2] = float(rows.mean()) / 64.0       # mean row
        feats[base + 3] = float(cols.mean()) / 64.0       # mean col
        feats[base + 4] = float(rows.std()) / 64.0 if total > 1 else 0.0
        feats[base + 5] = float(cols.std()) / 64.0 if total > 1 else 0.0
        labeled, n_cc = label(mask)
        feats[base + 0] = float(n_cc) / 20.0              # count
        if n_cc > 0:
            sizes = np.array([np.sum(labeled == i) for i in range(1, n_cc+1)])
            feats[base + 6] = float(sizes.max()) / 4096.0  # largest CC frac
            feats[base + 7] = float((sizes > 64).sum()) / 10.0  # n large CCs
    out = np.zeros(256, dtype=np.float32)
    out[:128] = feats
    return out - out.mean()


# ---- T1: timing + sanity ----

def run_t1():
    print("T1: CC encoding timing and sanity", flush=True)
    t0 = time.time()
    rng = np.random.RandomState(0)

    # CIFAR encode
    img = (rng.rand(32, 32, 3) * 255).astype(np.uint8)
    enc = encode_cifar(img)
    assert enc.shape == (256,) and abs(enc.mean()) < 1e-5

    # ARC CC encode
    frame = [rng.randint(0, 16, (64, 64))]
    enc = encode_arc_cc(frame)
    assert enc.shape == (256,) and abs(enc.mean()) < 1e-5

    # Time: 100 CC encodes
    frames = [[rng.randint(0, 16, (64, 64))] for _ in range(100)]
    t_enc = time.time()
    for f in frames:
        encode_arc_cc(f)
    ms_per = (time.time() - t_enc) / 100 * 1000
    est_50k = ms_per * 50000 / 1000
    print(f"  CC encode: {ms_per:.2f}ms/frame  est 50K steps: {est_50k:.0f}s", flush=True)

    # L2 stats for CC vs avgpool16
    from experiments.run_step514_cc_encoding_chain import encode_arc_cc as _cc
    cifar_encs = [encode_cifar((rng.rand(32, 32, 3) * 255).astype(np.uint8)) for _ in range(50)]
    arc_encs = [encode_arc_cc([rng.randint(0, 16, (64, 64))]) for _ in range(50)]
    def l2(a, b): return float(np.sqrt(np.sum((a-b)**2)))
    arc_dists = [l2(arc_encs[i], arc_encs[j]) for i in range(10) for j in range(i+1,10)]
    cifar_dists = [l2(cifar_encs[i], cifar_encs[j]) for i in range(10) for j in range(i+1,10)]
    print(f"  ARC CC L2 mean={np.mean(arc_dists):.3f}  CIFAR L2 mean={np.mean(cifar_dists):.3f}", flush=True)
    print(f"  PASS: {time.time()-t0:.1f}s", flush=True)
    return ms_per


# ---- DynamicGraph ----

class DynamicGraph:
    def __init__(self, threshold=SPAWN_THRESHOLD):
        self.threshold = threshold
        self.centroids = None
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self._mode = None
        self.spawns = {}

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _spawn(self, x):
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1, -1)])
        self.spawns[self._mode] = self.spawns.get(self._mode, 0) + 1
        return len(self.centroids) - 1

    def step(self, x, n_actions):
        if self.centroids is None:
            self._spawn(x)
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nearest = int(np.argmin(dists))
        cell = self._spawn(x) if dists[nearest] > self.threshold else nearest
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


# ---- Phase runners ----

def run_cifar(g, X, y, label):
    g.set_mode('cifar')
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = g.step(x, N_CIFAR_ACTIONS)
        if a == int(y[i]): correct += 1
    acc = correct / len(X) * 100
    print(f"  {label}: acc={acc:.2f}%  centroids={g.n_centroids}  "
          f"spawns_cifar={g.spawns.get('cifar',0)}  {time.time()-t0:.0f}s", flush=True)
    return acc


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
        x = encode_arc_cc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  LS20: {status}  centroids={g.n_centroids}  "
          f"spawns_ls20={g.spawns.get('ls20',0)}  go={go}  steps={ts}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


def main():
    t_total = time.time()
    print("Step 514: CC encoding chain CIFAR->LS20->CIFAR", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}  max_ls20_steps={MAX_LS20_STEPS//1000}K", flush=True)
    print(f"Baseline (Step 508 avgpool16): LS20 WIN@11170", flush=True)

    # T1: Timing check
    print("\nT1: CC timing...", flush=True)
    from scipy.ndimage import label as _check_label  # noqa: verify import
    rng = np.random.RandomState(0)
    enc = encode_cifar((rng.rand(32, 32, 3) * 255).astype(np.uint8))
    assert enc.shape == (256,)
    enc = encode_arc_cc([rng.randint(0, 16, (64, 64))])
    assert enc.shape == (256,)
    frames_test = [[rng.randint(0, 16, (64, 64))] for _ in range(100)]
    t0 = time.time()
    for f in frames_test:
        encode_arc_cc(f)
    ms_per = (time.time() - t0) / 100 * 1000
    est_steps_5min = int(300_000 / ms_per)
    print(f"  CC encode: {ms_per:.2f}ms/frame  est steps in 5min: {est_steps_5min//1000}K", flush=True)

    arc_encs = [encode_arc_cc([rng.randint(0, 16, (64, 64))]) for _ in range(20)]
    arc_dists = [float(np.sqrt(np.sum((arc_encs[i]-arc_encs[j])**2)))
                 for i in range(10) for j in range(i+1,10)]
    cifar_encs = [encode_cifar((rng.rand(32, 32, 3)*255).astype(np.uint8)) for _ in range(20)]
    cifar_dists = [float(np.sqrt(np.sum((cifar_encs[i]-cifar_encs[j])**2)))
                   for i in range(10) for j in range(i+1,10)]
    print(f"  ARC CC L2 mean={np.mean(arc_dists):.3f}  min={np.min(arc_dists):.3f}", flush=True)
    print(f"  CIFAR L2 mean={np.mean(cifar_dists):.3f}  min={np.min(cifar_dists):.3f}", flush=True)
    print(f"  T1 PASS: {time.time()-t_total:.1f}s", flush=True)

    print("\nLoading CIFAR-100...", flush=True)
    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    print(f"  CIFAR {len(X)} images, LS20 ready", flush=True)

    np.random.seed(0)
    g = DynamicGraph(threshold=SPAWN_THRESHOLD)

    print("\n--- Phase 1: CIFAR (1-pass, avgpool16) ---", flush=True)
    acc1 = run_cifar(g, X, y, "P1")

    print(f"\n--- Phase 2: LS20 ({MAX_LS20_STEPS//1000}K steps, CC encoding) ---", flush=True)
    ls20_lvls, ls20_step = run_ls20(g, arc, ls20.game_id)

    print("\n--- Phase 3: CIFAR (1-pass, avgpool16) ---", flush=True)
    acc3 = run_cifar(g, X, y, "P3")

    print(f"\n{'='*60}", flush=True)
    print("STEP 514 SUMMARY", flush=True)
    print(f"  CIFAR P1:  {acc1:.2f}%  (avgpool16)", flush=True)
    print(f"  LS20:      {'WIN@'+str(ls20_step) if ls20_lvls>0 else 'FAIL'}  (CC encoding)", flush=True)
    print(f"  CIFAR P3:  {acc3:.2f}%  (delta={acc3-acc1:+.2f}pp)", flush=True)
    print(f"  Total centroids: {g.n_centroids}  spawns: {g.spawns}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if ls20_lvls > 0:
        print(f"  CC ENCODING PASSES. LS20 navigates with CC features.", flush=True)
        print(f"  WIN@{ls20_step} vs avgpool16 baseline WIN@11170.", flush=True)
        if ls20_step < 11170:
            print(f"  CC faster than avgpool16 ({ls20_step} < 11170 steps).", flush=True)
        else:
            print(f"  CC slower than avgpool16 ({ls20_step} > 11170 steps).", flush=True)
    else:
        print(f"  KILL: CC encoding FAILS navigation. avgpool16 is better for ARC.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
