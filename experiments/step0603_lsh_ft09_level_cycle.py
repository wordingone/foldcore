"""
Step 603 -- FT09 L2: fix level cycling (on_level_up every transition).

Step 602 bug: on_level_up(1) gated by `l1_step is None` — only fired ONCE.
After death in L1 state, game_level reset to 0, never re-entered L1 mode.
Result: f1=1061 for all seeds (1 L1 run then L0 forever).

Fix: call on_level_up for EVERY level transition (not just first).
Consequence: each game run cycles L0->L1->L2, agent gets many more L1 frames.

Other changes vs 602:
  - MODE_WARMUP=500 (faster cluster detection; many L1 cycles expected)
  - BURST=10 (more sustained clicking per cluster)
  - Count total L1 cycles to track cycling behavior

Kill:  L2=0/5 after many L1 cycles -> 20 T1 clusters are wrong targets.
Signal: L2>=1/5 -> cycling mechanism works. Push VC33.

Protocol: 5 seeds x 50K steps, 60s/seed
"""
import time
import logging
import numpy as np
from scipy.ndimage import label as ndlabel

logging.getLogger().setLevel(logging.WARNING)

K = 12
DIM = 256
N_CLICKS = 64
MODE_WARMUP = 500
MODE_EVERY = 100
MIN_CLUSTER = 4
MAX_CLUSTER = 60
MAX_STEPS = 50_000
TIME_CAP = 60
BURST = 10
STATUS_MASK_TOP = 8

BRIGHT_COLORS = set(range(2, 15))
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
    small = cluster['size'] <= 16
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


class SubFT09:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}
        self._pn = self._pa = self._cn = None
        self.cells = set()

        self.l1_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l1_mode = None
        self.l1_frames = 0

        self.game_level = 0
        self.l1_cycles = 0    # total L1 entries
        self.clusters = []
        self._nav_cluster = 0
        self._nav_burst = 0

    def observe(self, frame):
        if self.game_level >= 1:
            update_freq(self.l1_freq, frame)
            self.l1_frames += 1
            if (self.l1_frames >= MODE_WARMUP and
                    self.l1_frames % MODE_EVERY == 0):
                self.l1_mode = compute_mode(self.l1_freq)
                cls = find_priority_clusters(self.l1_mode)
                if cls and len(cls) > len(self.clusters):
                    self.clusters = cls  # only update if more clusters found

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
        if new_lvls == 1:
            self.l1_cycles += 1
        self._nav_cluster = 0
        self._nav_burst = 0

    def on_reset(self):
        self._pn = None
        self._nav_burst = 0
        self.game_level = 0


def t0():
    sub = SubFT09(seed=0)
    assert sub.game_level == 0
    sub.on_level_up(1); assert sub.game_level == 1; assert sub.l1_cycles == 1
    sub.on_reset();      assert sub.game_level == 0
    sub.on_level_up(1); assert sub.l1_cycles == 2   # second L1 cycle counts
    print("T0 PASS", flush=True)


def run_seed(arc, game_id, seed):
    from arcengine import GameState
    np.random.seed(seed)

    env = arc.make(game_id)
    action6 = env.action_space[0]
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
            # FIX: call on_level_up for EVERY transition (not just first)
            if new_lvl >= 1:
                if l1_step is None:
                    l1_step = ts
                    print(f"  s{seed} L1@{ts} go={go} cells={len(sub.cells)}", flush=True)
                sub.on_level_up(1)
            if new_lvl >= 2:
                if l2_step is None:
                    l2_step = ts
                    print(f"  s{seed} L2@{ts}!! cyc={sub.l1_cycles} f1={sub.l1_frames} "
                          f"cls={len(sub.clusters)}", flush=True)
                sub.on_level_up(2)
            prev_lvls = new_lvl

        if time.time() - t0 > TIME_CAP:
            print(f"  s{seed} cap@{ts} cyc={sub.l1_cycles} f1={sub.l1_frames}", flush=True)
            break

    status = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "---")
    print(f"  s{seed}: {status}  cells={len(sub.cells)}  go={go}  "
          f"cyc={sub.l1_cycles}  f1={sub.l1_frames}  cls={len(sub.clusters)}", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step, cells=len(sub.cells),
                go=go, ts=ts, l1_cycles=sub.l1_cycles, l1_frames=sub.l1_frames)


def main():
    t0()

    import arc_agi
    arc = arc_agi.Arcade()
    envs = arc.get_environments()
    ft09 = next((e for e in envs if 'ft09' in e.game_id.lower()), None)
    if ft09 is None:
        print("SKIP -- FT09 not found"); return

    print(f"Step 603: FT09 L2 -- level cycling fix", flush=True)
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
    avg_cyc = np.mean([r['l1_cycles'] for r in results]) if results else 0
    avg_f1 = np.mean([r['l1_frames'] for r in results]) if results else 0

    print(f"\n{'='*60}", flush=True)
    print(f"Step 603: FT09 L2 (level cycling fix)", flush=True)
    print(f"  L1: {l1_wins}/{len(results)}  L2: {l2_wins}/{len(results)}", flush=True)
    print(f"  avg L1 cycles: {avg_cyc:.1f}  avg L1 frames: {avg_f1:.0f}", flush=True)
    if l2_wins >= 3:
        print("  SIGNAL: Level cycling + priority CC reaches L2.", flush=True)
    elif l2_wins > 0:
        print(f"  PARTIAL: {l2_wins}/{len(results)} L2.", flush=True)
    elif l1_wins >= 3:
        print("  L1 cycling confirmed. 20 clusters still wrong? Need different click targets.", flush=True)
    else:
        print("  L1 not reached.", flush=True)


if __name__ == "__main__":
    main()
