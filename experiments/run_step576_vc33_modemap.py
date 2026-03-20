"""
Step 576 -- VC33 mode map + isolated CC: autonomous zone discovery.

Cross-game test: does the same mode-map pipeline that found LS20 sprites
autonomously discover VC33's magic zones?

VC33 mechanics (Step 505): click-only game (ACTION6). Win requires clicking
specific zones. Magic positions at (62,26) and (62,34). Frame shows two
4x4 color-9 blocks at right edge (x=60-63, y=24-27 and y=32-35).

Pipeline:
1. Warmup: argmin over 64-click grid (8x8 stride), accumulate pixel freqs
2. Mode map: argmax pixel color per position
3. Isolated CC: find small connected components (size 2-60 pixels, color != bg)
4. Navigation: cycle through CC centers via ACTION6 click, no visited marker

Kill:  0/5 wins -> mode map doesn't generalize to VC33
Signal: >=3/5 -> mode map discovers VC33 zones the same way it found LS20 sprites

Step 505 baseline: 3/3 wins (prescribed zone positions, k-means)
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64       # 8x8 grid, stride 8
MODE_WARMUP = 5000  # frames before computing mode map
MODE_EVERY = 500    # recompute mode map every N frames
MIN_CLUSTER = 4     # min pixels in isolated CC
MAX_CLUSTER = 60    # max pixels in isolated CC
MAX_STEPS = 200_000
TIME_CAP = 60       # seconds per seed
BURST = 5           # clicks per cluster in sweep before moving on

# 8x8 click grid: positions at stride 8 (centers of 8x8 cells)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]
BG_COLOR = 0  # background color to exclude from CC detection


# ── encoding ──────────────────────────────────────────────────────────────────

def encode(frame, H):
    """frame[0] is 64x64 uint8 [0-15] -> avgpool16 -> center -> k-bit hash."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


# ── mode map + isolated CC ────────────────────────────────────────────────────

def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]
    c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1


def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)


def find_isolated_clusters(mode_arr, min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    clusters = []
    for color in range(1, 16):  # skip background (0)
        mask = (mode_arr == color)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if min_sz <= sz <= max_sz:
                ys, xs = np.where(region)
                clusters.append({
                    'cy': float(ys.mean()),
                    'cx': float(xs.mean()),
                    'color': int(color),
                    'size': sz,
                    'cy_int': int(round(ys.mean())),
                    'cx_int': int(round(xs.mean())),
                })
    return clusters


# ── substrate ─────────────────────────────────────────────────────────────────

class SubVC33:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.n_cells = set()

        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = None
        self.n_frames = 0
        self.clusters = []
        self.cluster_visits = {}  # cluster_idx -> visit count
        self.cluster_scores = {}  # cluster_idx -> level increments scored

        # Phase: 'explore' (argmin 64-click) or 'navigate' (cluster targets)
        self.phase = 'explore'
        self._nav_cluster = 0    # current cluster being swept
        self._nav_burst = 0      # clicks remaining in current burst

    def observe(self, frame):
        n = encode(frame, self.H)
        self.n_cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

        # Accumulate mode map
        update_freq(self.freq, frame)
        self.n_frames += 1
        if self.n_frames >= MODE_WARMUP and self.n_frames % MODE_EVERY == 0:
            self.mode = compute_mode(self.freq)
            new_clusters = find_isolated_clusters(self.mode)
            if len(new_clusters) > 0 and self.phase == 'explore':
                self.clusters = new_clusters
                self.phase = 'navigate'
                self.cluster_visits = {i: 0 for i in range(len(self.clusters))}
                self.cluster_scores = {i: 0 for i in range(len(self.clusters))}

    def act(self):
        if self.phase == 'explore':
            # Argmin over 64-click grid
            counts = [sum(self.G.get((self._cn, a), {}).values())
                      for a in range(N_CLICKS)]
            min_c = min(counts)
            candidates = [a for a, c in enumerate(counts) if c == min_c]
            a = candidates[int(np.random.randint(len(candidates)))]
            self._pn = self._cn
            self._pa = a
            cx, cy = CLICK_GRID[a]
        else:
            # Navigate: burst-sweep through clusters.
            # Click same cluster BURST times before moving on.
            # Interleaving different clusters resets puzzle state (VC33-specific).
            if self.clusters:
                if self._nav_burst <= 0:
                    # Move to next cluster
                    self._nav_cluster = (self._nav_cluster + 1) % len(self.clusters)
                    self._nav_burst = BURST
                a = self._nav_cluster
                self._nav_burst -= 1
                self.cluster_visits[a] = self.cluster_visits.get(a, 0) + 1
                cx = self.clusters[a]['cx_int']
                cy = self.clusters[a]['cy_int']
                self._pn = self._cn
                self._pa = a + N_CLICKS
            else:
                cx, cy = CLICK_GRID[0]
        return cx, cy

    def on_level(self, cluster_action=None):
        """Called when level increments during navigate phase."""
        if cluster_action is not None and self.phase == 'navigate':
            self.cluster_scores[cluster_action] = self.cluster_scores.get(cluster_action, 0) + 1

    def on_reset(self):
        self._pn = None
        self._nav_burst = 0  # restart burst on reset


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = SubVC33(seed=0)
    frame = [np.zeros((64, 64), dtype=np.uint8)]
    sub.observe(frame)
    cx, cy = sub.act()
    assert 0 <= cx < 64 and 0 <= cy < 64

    # Mode map detection test
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    # Inject isolated 4x4 block of color 9 at y=24-27, x=60-63
    f = np.zeros((64, 64), dtype=np.uint8)
    f[24:28, 60:64] = 9
    for _ in range(100):
        r = np.arange(64)[:, None]
        c = np.arange(64)[None, :]
        freq[r, c, f.reshape(64, 64)] += 1
    mode = compute_mode(freq)
    clusters = find_isolated_clusters(mode)
    cc9 = [c for c in clusters if c['color'] == 9]
    assert len(cc9) >= 1, f"Expected color-9 cluster, got {clusters}"
    assert abs(cc9[0]['cx'] - 61.5) < 2 and abs(cc9[0]['cy'] - 25.5) < 2
    print(f"T0 PASS: detected color-9 cluster at ({cc9[0]['cx']:.1f}, {cc9[0]['cy']:.1f})")


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    sub = SubVC33(seed=seed * 1000)
    obs = env.reset()

    ts = go = lvls = 0
    level_step = None
    last_cluster_action = None
    t0_s = time.time()

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset()
            sub.on_reset()
            continue
        if obs.state == GameState.GAME_OVER:
            go += 1
            obs = env.reset()
            sub.on_reset()
            continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset()
            sub.on_reset()
            continue

        sub.observe(obs.frame)
        cx, cy = sub.act()

        # Track which cluster action this was
        if sub.phase == 'navigate' and sub.clusters:
            last_cluster_action = min(sub.cluster_visits,
                                      key=lambda i: sub.cluster_visits.get(i, 0) + 1)

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1

        if obs is None:
            break
        if obs.levels_completed > lvls_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
                print(f"  s{seed} WIN@{ts} phase={sub.phase} clusters={len(sub.clusters)} "
                      f"go={go}", flush=True)
            sub.on_level(last_cluster_action if sub.phase == 'navigate' else None)
        if time.time() - t0_s > TIME_CAP:
            break

    # Report clusters found
    clusters_str = ""
    if sub.clusters:
        cluster_info = [(c['cx_int'], c['cy_int'], c['color'], c['size'])
                        for c in sub.clusters]
        clusters_str = str(cluster_info)

    elapsed = time.time() - t0_s
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  s{seed}: {status:12s}  cells={len(sub.n_cells):4d}  go={go}  "
          f"clusters={len(sub.clusters)}  steps={ts}  {elapsed:.0f}s", flush=True)
    if sub.clusters:
        print(f"    cluster positions: {clusters_str}", flush=True)
    return dict(seed=seed, levels=lvls, level_step=level_step,
                cells=len(sub.n_cells), go=go, steps=ts,
                n_clusters=len(sub.clusters),
                cluster_scores=dict(sub.cluster_scores))


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("SKIP -- VC33 not found")
        return
    print(f"VC33: {vc33.game_id}", flush=True)
    print(f"Mode map pipeline: warmup={MODE_WARMUP} frames, then isolated CC navigation.", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 290:
            print("TOTAL TIME CAP HIT")
            break
        r = run_seed(arc, vc33.game_id, seed)
        results.append(r)

    wins = sum(1 for r in results if r['levels'] > 0)
    any_clusters = sum(1 for r in results if r['n_clusters'] > 0)
    avg_cells = float(np.mean([r['cells'] for r in results])) if results else 0

    print(f"\n{'='*50}")
    print(f"STEP 576: {wins}/{len(results)} wins")
    print(f"  Seeds with clusters detected: {any_clusters}/{len(results)}")
    print(f"  avg_cells: {avg_cells:.0f}")
    print(f"  Step 505 baseline: 3/3 wins (prescribed zones)")

    if wins == 0 and any_clusters == 0:
        print("FAIL: Mode map found no clusters. VC33 visual structure not stable enough.")
    elif wins == 0 and any_clusters > 0:
        print("PARTIAL: Clusters detected but navigation failed. Wrong click positions?")
    elif wins >= 3:
        print(f"SUCCESS: {wins}/{len(results)}. Mode map autonomously discovered VC33 zones.")
    else:
        print(f"PARTIAL: {wins}/{len(results)}. Mode map works but less reliable than prescribed.")


if __name__ == "__main__":
    main()
