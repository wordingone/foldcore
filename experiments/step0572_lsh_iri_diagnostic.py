"""
Step 572 — iri (palette) diagnostic + L2 attempt.

PART A FINDINGS (from game source):
- iri sprites: sprite "zba", color 11, 3x3 ring (8px), tag "iri"
  → touch = energy refill. REMOVED on touch, RESTORED on episode reset.
  → irrelevant to win condition.
- win condition: sprite "lhs", color 5, 5x5 solid (25px), tag "mae"
  → agent must reach lhs sprite when qhg() = True (state puzzle solved)
  → qhg(i) = (snw == gfy[i]) AND (tmx == vxy[i]) AND (tuv == cjl[i])
  → state changes via: gsu (snw++), gic (tmx++), bgt (tuv++)
- L1 in Step 567: agent randomly hit toggle sprites while navigating,
  accidentally solved state puzzle, then reached lhs → nje() True → next_level()
- L2 failure in Step 571: after L1, NEW LEVEL loaded. Mode map from OLD level
  → wrong candidate positions → navigating to stale targets → 0 L2

PART B:
- color 5 (lhs, win objective) IS in rare-color clusters if <5% of pixels
- zba (color 11) should also appear if present in level
- After L1 → new level → mode map stale

PART C (this experiment):
- After L1: REBUILD mode map for new level (reset freq/mode)
- Re-detect targets for new level
- Continue targeting to reach L2
- Per-seed cap: 5 min. 5 seeds. 200K steps.
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
WARMUP = 100
RARE_THRESH = 0.05
MIN_CLUSTER = 2
MAX_CLUSTER = 30  # increased to catch lhs 5x5 = 25px
VISIT_DIST = 4
REDETECT_EVERY = 500


def dir_action(ty, tx, ay, ax):
    dy = ty - ay
    dx = tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    else:
        return 2 if dx < 0 else 3


def find_rare_clusters(mode_arr):
    total = mode_arr.size
    rare_thresh_px = total * RARE_THRESH
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


class SubAdaptive:
    """Mode map targeting with level-aware mode reset after level change."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, FG_DIM).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self._last_visit = {}
        # Background model (reset per level)
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        self.level_frames = 0
        # Targeting
        self.targets = []
        self.visited = []
        self.agent_yx = None
        self.prev_arr = None
        self._curr_arr = None
        self._steps_since_detect = REDETECT_EVERY
        # Stats
        self.target_actions = 0
        self.fb_actions = 0
        self.n_targets_found = 0
        self.n_levels = 0

    def reset_mode_map(self):
        """Call after level change to rebuild mode map for new level."""
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.level_frames = 0
        self.targets = []
        self.visited = []
        self._steps_since_detect = REDETECT_EVERY
        self.prev_arr = None
        self.agent_yx = None
        self.n_levels += 1

    def _update_bg(self, arr):
        r = np.arange(64)[:, None]
        c = np.arange(64)[None, :]
        self.freq[r, c, arr] += 1
        self.n_frames += 1
        self.level_frames += 1
        if self.level_frames % MODE_EVERY == 0:
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
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
        self.prev_arr = arr.copy()
        self._curr_arr = arr
        if self.level_frames < WARMUP:
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
        if self._steps_since_detect >= REDETECT_EVERY and self.level_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
            self._steps_since_detect = 0

        if self.targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None
            best_dist = 1e9
            for t in self.targets:
                visited = any(
                    ((t['cy'] - vy) ** 2 + (t['cx'] - vx) ** 2) < VISIT_DIST ** 2
                    for vy, vx in self.visited
                )
                if visited:
                    continue
                dist = ((t['cy'] - ay) ** 2 + (t['cx'] - ax) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = t

            if best is not None:
                if best_dist < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    action = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn
                    self._pa = action
                    self.target_actions += 1
                    return action

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
        self.visited = []
        self._steps_since_detect = REDETECT_EVERY

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
    mode = np.zeros((64, 64), dtype=np.int32)
    mode[10:15, 10:15] = 5   # 25px color-5 block (lhs-sized)
    mode[20:23, 20:23] = 11  # 9px color-11 (zba-sized, minus transparent center = 8px)
    clusters = find_rare_clusters(mode)
    colors = {c['color'] for c in clusters}
    assert 5 in colors, f"Should find lhs color 5: {colors}"
    assert 11 in colors, f"Should find zba color 11: {colors}"
    assert 0 not in colors
    # Adaptive reset test
    sub = SubAdaptive(seed=0)
    for _ in range(5):
        f = [rng.randint(0, 16, (64, 64))]
        sub.observe(f)
        sub.act()
    sub.on_reset()
    old_levels = sub.n_levels
    sub.reset_mode_map()
    assert sub.n_levels == old_levels + 1
    assert sub.level_frames == 0
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
    per_seed_cap = 300
    R = []
    t_start = time.time()

    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        env = mk()
        sub = SubAdaptive(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1_step = l2_step = None
        go = 0
        seed_start = time.time()

        for step in range(1, 200_001):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_reset()
                continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)
            if done:
                go += 1
                obs = env.reset(seed=seed)
                sub.on_reset()
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                sub.on_reset()
                sub.reset_mode_map()  # REBUILD mode map for new level
                if cl == 1 and l1_step is None:
                    l1_step = step
                    print(f"  s{seed} L1@{step} tgt={sub.n_targets_found} go={go} "
                          f"rebuilding mode map for L2...", flush=True)
                if cl == 2 and l2_step is None:
                    l2_step = step
                    print(f"  s{seed} L2@{step}! tgt={sub.n_targets_found} go={go}", flush=True)
            if step % 25_000 == 0:
                el = time.time() - seed_start
                nc, ns, ne = sub.stats()
                print(f"  s{seed} @{step} lvl={level} tgt={sub.n_targets_found} "
                      f"lf={sub.level_frames} go={go} c={nc} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} seed cap @{step}", flush=True)
                break

        nc, ns, ne = sub.stats()
        R.append(dict(
            seed=seed, l1=l1_step, l2=l2_step,
            cells=nc, go=go, steps=step, level=level,
            n_targets=sub.n_targets_found,
            target_actions=sub.target_actions,
            fb=sub.fb_actions,
        ))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    print(f"\nResults (Step 572 — adaptive mode map, rebuild on level change):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  tgt={r['n_targets']:>3}  "
              f"ta={r['target_actions']:>6}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps (baseline 468)")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! Mode map rebuild on level change works!")
    elif l1n == 0:
        print(f"\nKILL: L1=0. Regression.")
    else:
        print(f"\nKILL: L1={l1n}, L2=0. Mode map rebuild insufficient — need state puzzle solved.")


if __name__ == "__main__":
    main()
