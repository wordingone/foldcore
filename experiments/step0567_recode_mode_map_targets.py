"""
Step 567 — Mode map rare-color targeting. R1-compliant L2 attempt.

After mode stabilizes: find rare-color clusters (<5% coverage, 2-20px).
Navigate toward nearest unvisited cluster. Fallback to argmin.
Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.

Kill: rare_clusters=0 -> no signal. L1<2/5 -> targeting hurts.
Find: L2>0 -> FIRST EVER L2.
5-min cap. LS20. 5 seeds.
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel

N_A = 4
K = 16
FG_DIM = 4096
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05
MODE_EVERY = 200
WARMUP = 100          # frames before mode reliable
RARE_THRESH = 0.05    # color covering <5% of 64x64 = <205 pixels = rare
MIN_CLUSTER = 2
MAX_CLUSTER = 20
VISIT_DIST = 4        # pixels: target "reached" within this distance
REDETECT_EVERY = 500  # recompute targets every N steps


def dir_action(ty, tx, ay, ax):
    """0=UP, 1=DOWN, 2=LEFT, 3=RIGHT."""
    dy = ty - ay
    dx = tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    else:
        return 2 if dx < 0 else 3


def find_rare_clusters(mode_arr):
    """Find connected clusters of rare-color pixels in mode map.
    Returns list of {cy, cx, color, size}.
    """
    total = mode_arr.size
    rare_thresh_px = total * RARE_THRESH
    # Count pixels per color
    colors, counts = np.unique(mode_arr, return_counts=True)
    rare_colors = colors[counts < rare_thresh_px]
    clusters = []
    for color in rare_colors:
        mask = (mode_arr == color)
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if sz < MIN_CLUSTER or sz > MAX_CLUSTER:
                continue
            ys, xs = np.where(region)
            clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                             'color': int(color), 'size': sz})
    return clusters


class RecodeTargeted:
    def __init__(self, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, FG_DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self._last_visit = {}
        # Background model
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        # Targeting
        self.targets = []        # list of {cy, cx, color, size}
        self.visited = []        # centroids we've been near
        self.agent_yx = None
        self.prev_arr = None
        self._curr_arr = None
        self._steps_since_detect = REDETECT_EVERY  # force detect on first act
        # Stats
        self.target_actions = 0
        self.fb_actions = 0
        self.n_targets_found = 0

    def _update_bg(self, arr):
        r = np.arange(64)[:, None]
        c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1
        self.n_frames += 1
        if self.n_frames % MODE_EVERY == 0:
            self.mode = np.argmax(self.freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        return (arr != self.mode).astype(np.float32).flatten()

    def _base(self, x):
        return int(np.packbits(
            (self.H @ x > 0).astype(np.uint8), bitorder='big'
        ).tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref:
            n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32)
        self._update_bg(arr)
        # Update agent pos from frame diff
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
        self.prev_arr = arr.copy()
        self._curr_arr = arr
        # Encode
        if self.n_frames < WARMUP:
            x = arr.astype(np.float32).flatten() / 15.0
            x = x - x.mean()
        else:
            x = self._fg_enc(arr)
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(FG_DIM, np.float64), 0))
            self.C[k_key] = (s + x.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        self._steps_since_detect += 1
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        # Redetect targets periodically
        if self._steps_since_detect >= REDETECT_EVERY and self.n_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
            self._steps_since_detect = 0

        # Target-directed action
        if self.targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            # Find nearest unvisited target
            best = None
            best_dist = 1e9
            for t in self.targets:
                # Skip if visited
                visited = any(((t['cy'] - vy)**2 + (t['cx'] - vx)**2) < VISIT_DIST**2
                              for vy, vx in self.visited)
                if visited:
                    continue
                dist = ((t['cy'] - ay)**2 + (t['cx'] - ax)**2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = t

            if best is not None:
                # Check if at target
                if best_dist < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    action = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn
                    self._pa = action
                    self.target_actions += 1
                    return action

        # Fallback: argmin
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        self.fb_actions += 1
        return action

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None
        self.visited = []     # reset visited per episode
        self._steps_since_detect = REDETECT_EVERY  # redetect after reset

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4:
            return 0.0
        v = np.array(list(d.values()), np.float64)
        p = v / v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref:
                continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS:
                continue
            if self._h(n, a) < H_SPLIT:
                continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0 = self.C.get((n, a, top[0]))
            r1 = self.C.get((n, a, top[1]))
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2:
                continue
            diff = (r0[0] / r0[1]) - (r1[0] / r1[1])
            nm = np.linalg.norm(diff)
            if nm < 1e-8:
                continue
            self.ref[n] = (diff / nm).astype(np.float32)
            self.live.discard(n)

    def stats(self):
        return len(self.live), len(self.ref), len(self.G)


def t0():
    rng = np.random.RandomState(0)
    # Test find_rare_clusters
    mode = np.zeros((64, 64), dtype=np.int32)
    # Color 7: 10 pixels in one cluster (rare: <205px)
    mode[10:12, 10:15] = 7  # 10 pixels
    clusters = find_rare_clusters(mode)
    colors = {c['color'] for c in clusters}
    assert 7 in colors, f"Should find rare color 7: {colors}"
    # Color 0: 64*64-10 pixels (not rare) -> filtered
    assert 0 not in colors, f"Background 0 should be filtered"
    # dir_action tests
    assert dir_action(2, 5, 8, 5) == 0   # UP
    assert dir_action(10, 5, 4, 5) == 1  # DOWN
    assert dir_action(5, 2, 5, 8) == 2   # LEFT
    assert dir_action(5, 8, 5, 2) == 3   # RIGHT
    # RecodeTargeted smoke
    sub = RecodeTargeted(seed=0)
    for _ in range(5):
        f = [rng.randint(0, 16, (64, 64))]
        sub.observe(f)
        sub.act()
    sub.on_reset()
    assert sub.visited == []
    print("T0 PASS")


def main():
    t0()
    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 5
    global_cap = 280
    R = []
    t_start = time.time()

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nGlobal cap hit at seed {seed}", flush=True); break
        budget = (global_cap - elapsed) / (n_seeds - seed)
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)
        env = mk()
        sub = RecodeTargeted(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1 = l2 = None
        go = 0
        deadline = time.time() + budget

        for step in range(1, 500_001):
            if obs is None:
                obs = env.reset(seed=seed); sub.on_reset(); continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)
            if done:
                go += 1; obs = env.reset(seed=seed); sub.on_reset()
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl; sub.on_reset()
                if cl == 1 and l1 is None:
                    l1 = step
                    print(f"  s{seed} L1@{step} c={sub.stats()[0]} go={go} "
                          f"tgt={sub.n_targets_found} ta={sub.target_actions} "
                          f"fb={sub.fb_actions}", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    print(f"  s{seed} L2@{step}! tgt={sub.n_targets_found} "
                          f"ta={sub.target_actions} mode_unique={len(np.unique(sub.mode))}",
                          flush=True)
            if step % 25_000 == 0:
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={sub.stats()[0]} go={go} "
                      f"tgt={sub.n_targets_found} ta={sub.target_actions} {el:.0f}s",
                      flush=True)
            if time.time() > deadline:
                break

        nc, ns, ne = sub.stats()
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, go=go,
                      steps=step, level=level,
                      n_targets=sub.n_targets_found,
                      target_actions=sub.target_actions,
                      fb=sub.fb_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    print(f"\nResults (mode map rare-color targeting):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  tgt={r['n_targets']:>3}  "
              f"ta={r['target_actions']:>6}  fb={r['fb']:>6}")
    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    if not R:
        print("No results."); return
    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None
    avg_tgt = np.mean([r['n_targets'] for r in R])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    print(f"Avg targets found: {avg_tgt:.1f}")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps (baseline 15164, fg-enc 10407)")
    print(f"Baseline (Step 554): L1=3/3 at ~15K steps")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! FIRST L2 EVER. Mode map rare-color targeting works!")
    elif avg_tgt == 0:
        print(f"\nKILL: rare_clusters=0. No rare colors in mode map.")
    elif l1n >= 2:
        print(f"\nL1={l1n}/{len(R)}, L2=0. Targets={avg_tgt:.0f} found but L2 not reached.")
        print("Either palettes not rare-colored OR agent can't navigate to them in 129 steps.")
    else:
        print(f"\nKILL: L1={l1n}/{len(R)} < 2. Mode-map targeting hurts navigation.")


if __name__ == "__main__":
    main()
