"""
Step 602 -- FT09 L2: argmin L0 + mode-map priority tiers for L1.

Step 601 root cause: mode-map clusters at 150 frames are wrong targets.
Step 575 proved: pure argmin over 64 clicks finds L1 at 50K steps.

Fix:
  - L0: pure argmin (NO mode map). Same mechanism that got L1.
  - On L1 fire: level-aware reset, start L1 mode map.
  - L1: argmin fallback + priority CC navigation once mode map ready.

Priority tiers (inspired by dolphin-in-a-coma, 3rd place, 17 levels):
  T1: size 4-16px, bright color (non-0, non-1, non-15) — likely buttons
  T2: size 4-16px, any non-BG color
  T3: size 16-60px, bright color
  T4: size 16-60px, any color
Status bar masking: exclude top 8 rows from mode map (common UI band).

Kill:  0/5 L2 -> argmin doesn't carry over to L2 game state.
Signal: >=3/5 L2 -> same mechanism generalizes across levels. VC33 follows.

Protocol: 5 seeds x 50K steps, 60s/seed
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64         # 8x8 grid, stride 8
MODE_WARMUP = 1000    # frames for L1 mode map (after L1 fires)
MODE_EVERY = 200
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 60
BURST = 5
STATUS_MASK_TOP = 8   # mask top 8 rows (UI status bar)

BRIGHT_COLORS = set(range(2, 15))   # colors 2-14 (non-BG-black, non-white)
CLICK_GRID = [(gx * 8 + 4, gy * 8 + 4) for gy in range(8) for gx in range(8)]


# ── encoding ──────────────────────────────────────────────────────────────────

def encode(frame, H):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    x -= x.mean()
    bits = (H @ x > 0).astype(np.uint8)
    return int(np.packbits(bits, bitorder='big').tobytes().hex(), 16)


# ── mode map + priority CC ────────────────────────────────────────────────────

def update_freq(freq_arr, frame):
    arr = np.array(frame[0], dtype=np.int32)
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    freq_arr[r, c, arr] += 1


def compute_mode(freq_arr):
    return np.argmax(freq_arr, axis=2).astype(np.int32)


def cluster_priority(cluster):
    """Lower = higher priority. T1=0, T2=1, T3=2, T4=3."""
    bright = cluster['color'] in BRIGHT_COLORS
    small = cluster['size'] <= 16
    if small and bright: return 0
    if small:            return 1
    if bright:           return 2
    return 3


def find_priority_clusters(mode_arr, mask_top=STATUS_MASK_TOP,
                           min_sz=MIN_CLUSTER, max_sz=MAX_CLUSTER):
    """Find isolated CCs, mask status bar, sort by priority tiers."""
    clusters = []
    for color in range(1, 16):
        mask = (mode_arr == color)
        mask[:mask_top, :] = False   # status bar mask
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


# ── substrate ─────────────────────────────────────────────────────────────────

class SubFT09:
    """Argmin L0 + priority-CC mode-map L1."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        # L1 mode map — built only after L1 fires
        self.l1_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l1_mode = None
        self.l1_frames = 0

        self.game_level = 0   # 0=L0, 1=L1 (seeking L2)
        self.clusters = []    # priority-sorted L1 CCs
        self._nav_cluster = 0
        self._nav_burst = 0

    def observe(self, frame):
        if self.game_level >= 1:
            update_freq(self.l1_freq, frame)
            self.l1_frames += 1
            if (self.l1_frames >= MODE_WARMUP and
                    self.l1_frames % MODE_EVERY == 0 and
                    not self.clusters):
                self.l1_mode = compute_mode(self.l1_freq)
                cls = find_priority_clusters(self.l1_mode)
                if cls:
                    self.clusters = cls
                    print(f"    L1 clusters found: {len(cls)} "
                          f"(T1={sum(1 for c in cls if cluster_priority(c)==0)} "
                          f"T2={sum(1 for c in cls if cluster_priority(c)==1)})",
                          flush=True)

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
            return c['cx_int'], c['cy_int']

        # Argmin over 64-click grid (L0 primary, L1 fallback before clusters ready)
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

    def on_level_up(self, new_lvls):
        self.game_level = new_lvls
        # Reset nav position, keep accumulated clusters if any
        self._nav_cluster = 0
        self._nav_burst = 0

    def on_reset(self):
        """GAME_OVER: levels_completed resets. Mode maps and clusters persist."""
        self._pn = None
        self._nav_burst = 0
        self.game_level = 0


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    H = np.random.RandomState(0).randn(K, DIM).astype(np.float32)
    frame = [np.random.randint(0, 16, (64, 64), dtype=np.uint8)]
    n = encode(frame, H)
    assert isinstance(n, int)

    freq = np.zeros((64, 64, 16), dtype=np.int32)
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[20:24, 30:34] = 7    # small bright block, should be T1
    r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
    for _ in range(200):
        freq[r, c, arr] += 1
    mode = compute_mode(freq)
    cls = find_priority_clusters(mode)
    c7 = [x for x in cls if x['color'] == 7]
    assert len(c7) >= 1, f"Expected color-7 T1 cluster: {cls}"
    assert cluster_priority(c7[0]) == 0, f"Expected T1, got {cluster_priority(c7[0])}"
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
                print(f"  s{seed} L1@{ts} go={go} cells={len(sub.cells)}", flush=True)
                sub.on_level_up(1)
            if l2_step is None and new_lvl >= 2:
                l2_step = ts
                print(f"  s{seed} L2@{ts}!! go={go} f1={sub.l1_frames} "
                      f"cls={len(sub.clusters)}", flush=True)
                sub.on_level_up(2)
            prev_lvls = new_lvl

        if time.time() - t0 > TIME_CAP:
            print(f"  s{seed} cap@{ts} go={go} f1={sub.l1_frames}", flush=True)
            break

    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  cells={len(sub.cells)}  go={go}  "
          f"f1={sub.l1_frames}  cls={len(sub.clusters)}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, cells=len(sub.cells),
                go=go, ts=ts, l1_frames=sub.l1_frames)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found"); return

    print(f"Step 602: FT09 L2 -- argmin L0 + priority-CC L1", flush=True)
    print(f"  game={ft09.game_id}  K={K}  5 seeds x {MAX_STEPS} steps", flush=True)
    print(f"  Priority tiers: T1=small+bright, T2=small, T3=large+bright, T4=large", flush=True)

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
    print(f"Step 602: FT09 L2", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    if l1_wins > 0:
        l1_steps = [r['l1'] for r in results if r['l1']]
        print(f"  L1 steps: avg={int(np.mean(l1_steps))} min={min(l1_steps)} max={max(l1_steps)}",
              flush=True)
    if l2_wins >= 3:
        print("  SIGNAL: Argmin generalizes to L2.", flush=True)
    elif l2_wins > 0:
        print(f"  PARTIAL: {l2_wins}/{len(results)} L2.", flush=True)
    elif l1_wins >= 3:
        print("  L1 confirmed. L2 state has different mechanism or needs more budget.", flush=True)
    else:
        print("  L1 not reached at 50K. FT09 needs different approach.", flush=True)


if __name__ == "__main__":
    main()
