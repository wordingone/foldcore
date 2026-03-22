"""
Step 605 -- VC33 L2: argmin L0/L1 + mode-map CC for L2 targeting.

VC33 coordinate system:
  - Camera(0, 0, 64, 64, 3, 4): display_to_grid is 1:1 (pixel == grid cell)
  - Click at (cx, cy) directly targets grid cell (cx, cy)
  - Frame is 64x64

Level structure: multiple levels with grid sizes 32x32, 52x52, 64x64.
TiD = movement direction, RoA = energy budget per level.

Approach:
  - L0: argmin over 64-click 8x8 grid
  - L1: argmin + build mode map, find priority CCs, navigate to them
  - On level-up: reset nav state, keep accumulated mode maps

Kill:  L2=0/5 -> argmin + mode map insufficient for VC33 L2
Signal: L2>=3/5 -> push to L3 and LS20

Protocol: 5 seeds x 50K steps, 60s/seed
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64       # 8x8 grid, stride 8
MODE_WARMUP = 300
MODE_EVERY = 100
MIN_CLUSTER = 2
MAX_CLUSTER = 100
MAX_STEPS = 50_000
TIME_CAP = 60
BURST = 8
STATUS_MASK_TOP = 4  # mask top 4 rows (UI bar)

BRIGHT_COLORS = set(range(2, 15))
# 8x8 click grid over 64x64 frame at stride 8, center at 4
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1


def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)


def cluster_priority(cluster):
    bright = cluster['color'] in BRIGHT_COLORS
    small = cluster['size'] <= 20
    if small and bright: return 0
    if small:            return 1
    if bright:           return 2
    return 3


def find_priority_clusters(mode_arr, mask_top=STATUS_MASK_TOP,
                           min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        mask[:mask_top, :] = False
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if min_sz <= sz <= max_sz:
                ys, xs = np.where(region)
                clusters.append({
                    'cy_int': int(round(ys.mean())),
                    'cx_int': int(round(xs.mean())),
                    'color': int(color),
                    'size': sz,
                })
    clusters.sort(key=cluster_priority)
    return clusters


class SubVC33:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = None
        self.frames = 0
        self.game_level = 0
        self.clusters = []
        self._nav_cluster = 0
        self._nav_burst = 0

    def observe(self, frame):
        update_freq(self.freq, frame)
        self.frames += 1
        if self.game_level >= 1 and self.frames >= MODE_WARMUP and self.frames % MODE_EVERY == 0:
            self.mode = compute_mode(self.freq)
            cls = find_priority_clusters(self.mode)
            if cls and len(cls) > len(self.clusters):
                self.clusters = cls

        n = encode(frame, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        if self.game_level >= 1 and self.clusters:
            if self._nav_burst <= 0:
                self._nav_cluster = (self._nav_cluster + 1) % len(self.clusters)
                self._nav_burst = BURST
            c = self.clusters[self._nav_cluster]
            self._nav_burst -= 1
            # VC33: 1:1 click mapping, click directly at cluster center
            return c['cx_int'], c['cy_int']

        counts = [sum(self.G.get((self._cn, a), {}).values())
                  for a in range(N_CLICKS)]
        min_c = min(counts)
        candidates = [a for a, cc in enumerate(counts) if cc == min_c]
        a = candidates[int(np.random.randint(len(candidates)))]
        gy, gx = divmod(a, 8)
        cx, cy = CLICK_GRID[a]
        self._pn = self._cn
        self._pa = a
        return cx, cy

    def on_level_up(self, new_lvl):
        self.game_level = new_lvl
        self._nav_cluster = 0
        self._nav_burst = 0
        # Reset mode map for new level visual
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.frames = 0
        self.clusters = []

    def on_reset(self):
        self._pn = None
        self._nav_burst = 0
        self.game_level = 0


def t0():
    sub = SubVC33(seed=0)
    assert sub.game_level == 0
    sub.on_level_up(1)
    assert sub.game_level == 1
    sub.on_reset()
    assert sub.game_level == 0
    print("T0 PASS", flush=True)


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
    sub = SubVC33(seed=seed * 1000)
    obs = env.reset()

    ts = go = 0
    prev_lvls = 0
    l1_step = l2_step = None
    t_start = time.time()

    while ts < MAX_STEPS:
        if obs is None:
            obs = env.reset(); sub.on_reset(); prev_lvls = 0; continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); sub.on_reset(); prev_lvls = 0; continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
            obs = env.reset(); sub.on_reset(); prev_lvls = 0; continue

        sub.observe(obs.frame)
        cx, cy = sub.act()

        lvls_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1

        if obs is None:
            break

        if obs.levels_completed > lvls_before:
            new_lvl = obs.levels_completed
            if new_lvl >= 1 and l1_step is None:
                l1_step = ts
                print(f"  s{seed} L1@{ts} go={go} cells={len(sub.cells)}", flush=True)
            if new_lvl >= 2 and l2_step is None:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!! go={go} f={sub.frames} cls={len(sub.clusters)}",
                      flush=True)
            sub.on_level_up(new_lvl)
            prev_lvls = new_lvl

        if time.time() - t_start > TIME_CAP:
            print(f"  s{seed} cap@{ts} go={go} lvl={prev_lvls} cls={len(sub.clusters)}",
                  flush=True)
            break

    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  cells={len(sub.cells)}  go={go}  "
          f"lvl={prev_lvls}  cls={len(sub.clusters)}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, cells=len(sub.cells), go=go, ts=ts)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    vc33 = next((e for e in envs if 'vc33' in e.game_id.lower()), None)
    if vc33 is None:
        print("SKIP -- VC33 not found"); return

    print(f"Step 605: VC33 L2 -- argmin + mode-map CC", flush=True)
    print(f"  game={vc33.game_id}  K={K}  5 seeds x {MAX_STEPS} steps", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 295:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(arc, vc33.game_id, seed)
        results.append(r)

    l1_wins = sum(1 for r in results if r['l1'])
    l2_wins = sum(1 for r in results if r['l2'])

    print(f"\n{'='*60}", flush=True)
    print(f"Step 605: VC33 L2 (argmin + mode-map CC)", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    if l2_wins >= 3:
        print("  SIGNAL: Mode-map CC reaches L2.", flush=True)
    elif l2_wins > 0:
        print(f"  PARTIAL: {l2_wins}/{len(results)} L2.", flush=True)
    elif l1_wins >= 3:
        print("  L1 confirmed. L2 needs VC33 source analysis.", flush=True)
    else:
        print("  L1 not reached.", flush=True)


if __name__ == "__main__":
    main()
