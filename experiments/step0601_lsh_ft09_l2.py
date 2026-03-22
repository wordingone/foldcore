"""
Step 601 -- FT09 L2: level-aware mode map navigation.

FT09: L1 achieved trivially at step 575 (5 cells). Push to L2.

Pipeline:
  1. L0 phase: argmin over 64-click grid + accumulate l0 mode map
  2. When l0 mode map ready (>=150 frames): find isolated CCs, burst-navigate
  3. On L1 fire (levels_completed 0->1): game_level=1, reset clusters
  4. L1 phase: accumulate l1 mode map, find l1 CCs, burst-navigate to them
  5. On L2 fire: record success

Kill:  0/5 L2 at 10K -> autonomous mode-map nav doesn't transfer to L2.
Signal: >=3/5 L2 -> mode-map pipeline generalizes to multi-level FT09. VC33 follows.

Protocol: 5 seeds x 10K steps, 60s/seed
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64        # 8x8 grid, stride 8 (click only; no simple actions needed)
MODE_WARMUP = 150    # frames before first mode map computation
MODE_EVERY = 50      # recompute interval
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 10_000
TIME_CAP = 60
BURST = 5            # clicks per cluster before cycling to next

CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


# ── encoding ──────────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


# ── mode map ──────────────────────────────────────────────────────────────────

def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1


def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)


def find_isolated_clusters(mode_arr, min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    clusters = []
    for color in range(1, 16):
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
                    'cy_int': int(round(ys.mean())),
                    'cx_int': int(round(xs.mean())),
                    'color': int(color),
                    'size': sz,
                })
    return clusters


# ── substrate ─────────────────────────────────────────────────────────────────

class SubFT09:
    """Level-aware mode map substrate for FT09 multi-level push."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        # Dual mode maps: freq[0] for L0 game state, freq[1] for L1 game state
        self.freq = [np.zeros((64, 64, 16), dtype=np.int32),
                     np.zeros((64, 64, 16), dtype=np.int32)]
        self.mode_map = [None, None]
        self.n_frames = [0, 0]

        self.game_level = 0    # 0=L0, 1=L1 (seeking L2), 2+=won
        self.clusters = []
        self._nav_cluster = 0
        self._nav_burst = 0

    def _li(self):
        """Mode map index clamped to [0,1]."""
        return min(self.game_level, 1)

    def observe(self, frame):
        li = self._li()
        update_freq(self.freq[li], frame)
        self.n_frames[li] += 1

        if (self.n_frames[li] >= MODE_WARMUP and
                self.n_frames[li] % MODE_EVERY == 0):
            self.mode_map[li] = compute_mode(self.freq[li])
            if not self.clusters:
                new_cls = find_isolated_clusters(self.mode_map[li])
                if new_cls:
                    self.clusters = new_cls

        n = encode(frame, self.H)
        self.cells.add(n)
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        if self.clusters:
            if self._nav_burst <= 0:
                self._nav_cluster = (self._nav_cluster + 1) % len(self.clusters)
                self._nav_burst = BURST
            c = self.clusters[self._nav_cluster]
            self._nav_burst -= 1
            return c['cx_int'], c['cy_int']

        # Argmin over 64-click grid (exploration fallback)
        counts = [sum(self.G.get((self._cn, a), {}).values())
                  for a in range(N_CLICKS)]
        min_c = min(counts)
        candidates = [a for a, cc in enumerate(counts) if cc == min_c]
        a = candidates[int(np.random.randint(len(candidates)))]
        gy, gx = divmod(a, 8)
        cx, cy = gx * 8 + 4, gy * 8 + 4
        self._pn = self._cn
        self._pa = a
        return cx, cy

    def on_level_up(self, new_lvls_completed):
        """Called when levels_completed increments."""
        self.game_level = new_lvls_completed
        self.clusters = []
        self._nav_cluster = 0
        self._nav_burst = 0
        # If we already have a mode map for this new game state, use it
        li = self._li()
        if self.mode_map[li] is not None:
            cls = find_isolated_clusters(self.mode_map[li])
            if cls:
                self.clusters = cls

    def on_reset(self):
        """Called on GAME_OVER — level counter resets, mode maps persist."""
        self._pn = None
        self._nav_burst = 0
        self.game_level = 0
        # Restore L0 clusters from existing mode map if available
        if self.mode_map[0] is not None:
            self.clusters = find_isolated_clusters(self.mode_map[0])
        else:
            self.clusters = []
        self._nav_cluster = 0


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    H = np.random.RandomState(42).randn(K, DIM).astype(np.float32)
    frame = [np.random.randint(0, 16, (64, 64), dtype=np.uint8)]
    n = encode(frame, H)
    assert isinstance(n, int)

    # Inject known 4x4 color-7 block into mode map
    freq = np.zeros((64, 64, 16), dtype=np.int32)
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[20:24, 30:34] = 7
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    for _ in range(200):
        freq[r, c, arr] += 1
    mode = compute_mode(freq)
    cls = find_isolated_clusters(mode)
    c7 = [x for x in cls if x['color'] == 7]
    assert len(c7) >= 1, f"Expected color-7 cluster, got {cls}"
    assert abs(c7[0]['cx_int'] - 31) <= 2 and abs(c7[0]['cy_int'] - 21) <= 2
    print("T0 PASS", flush=True)


# ── experiment ────────────────────────────────────────────────────────────────

def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]    # ACTION6 click (only action)
    sub = SubFT09(seed=seed * 1000)
    obs = env.reset()

    ts = go = 0
    prev_lvls = 0
    l1_step = l2_step = None
    t0 = time.time()

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
            if l1_step is None and new_lvl >= 1:
                l1_step = ts
                print(f"  s{seed} L1@{ts} go={go} f0={sub.n_frames[0]} "
                      f"cls={len(sub.clusters)}", flush=True)
                sub.on_level_up(1)
            if l2_step is None and new_lvl >= 2:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!! f1={sub.n_frames[1]} "
                      f"cls={len(sub.clusters)}", flush=True)
                sub.on_level_up(2)
            prev_lvls = new_lvl

        if time.time() - t0 > TIME_CAP:
            print(f"  s{seed} cap@{ts} f0={sub.n_frames[0]} f1={sub.n_frames[1]}", flush=True)
            break

    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  cells={len(sub.cells)}  go={go}  "
          f"f0={sub.n_frames[0]}  f1={sub.n_frames[1]}  cls={len(sub.clusters)}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, cells=len(sub.cells),
                go=go, ts=ts, n_frames=sub.n_frames[:])


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found"); return

    print(f"Step 601: FT09 L2 push", flush=True)
    print(f"  game={ft09.game_id}  K={K}  5 seeds x {MAX_STEPS} steps", flush=True)

    results = []
    t_total = time.time()

    for seed in range(5):
        if time.time() - t_total > 295:
            print("TOTAL TIME CAP"); break
        print(f"\nseed {seed}:", flush=True)
        r = run_seed(arc, ft09.game_id, seed)
        results.append(r)

    l1_wins = sum(1 for r in results if r['l1'])
    l2_wins = sum(1 for r in results if r['l2'])

    print(f"\n{'='*60}", flush=True)
    print(f"Step 601: FT09 L2 push", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)

    if l2_wins >= 3:
        print("  SIGNAL: Mode-map pipeline reaches L2. Try VC33.", flush=True)
    elif l2_wins > 0:
        print(f"  PARTIAL: {l2_wins}/{len(results)} L2. Needs more budget or tuning.", flush=True)
    elif l1_wins >= 3:
        print("  L1 only. L2 state needs more frames or game has only 1 level.", flush=True)
    else:
        print("  FAIL: L1 not reached. Check action_space index.", flush=True)


if __name__ == "__main__":
    main()
