"""
Step 561 — Rudakov-style connected-component object detection for LS20 L2.

Simplified replication of arXiv:2512.24156 (3rd place, LS20 solved in 4000 actions).
Key: connected-component segmentation finds "iri" energy palette objects.
Agent navigates TOWARD novel objects; falls back to argmin when none visible.

NOT codebook-banned: pixel-level operation, no cosine matching.

Actions: 0=LEFT, 1=UP, 2=RIGHT, 3=DOWN (from Step 557 comment: action 1 = move up).

Predictions:
  L1: 3/5 (may be slower — chases objects instead of uniform exploration)
  L2: 1/5 (if iri palette is a visible CC, agent navigates to it)

Kill: L2=0/5 -> palettes not detectable as CCs, or 129-step budget too tight.
5-min cap. LS20. 5 seeds.
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel

N_A = 4
K = 16
DIM = 256
REFINE_EVERY = 2000
MIN_OBS = 4
H_SPLIT = 0.05
MIN_OBJ_SIZE = 6    # minimum CC pixels to count as object
MAX_OBJ_SIZE = 700  # filter background (large CCs)
NOVEL_VISITS = 3    # color seen < this many times = novel


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def segment(arr):
    """Find connected components (4-connectivity) per color.
    Returns list of {color, cy, cx, size} for each component.
    Filters by MIN_OBJ_SIZE <= size <= MAX_OBJ_SIZE.
    """
    objects = []
    colors = np.unique(arr)
    for color in colors:
        mask = (arr == color)
        labeled, n_comps = ndlabel(mask)
        for cid in range(1, n_comps + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if sz < MIN_OBJ_SIZE or sz > MAX_OBJ_SIZE:
                continue
            ys, xs = np.where(region)
            cy, cx = float(ys.mean()), float(xs.mean())
            objects.append({'color': int(color), 'cy': cy, 'cx': cx, 'size': sz})
    return objects


def dir_action(ty, tx, ay, ax):
    """Pick action to move from (ay, ax) toward (ty, tx).
    Assumes: 0=LEFT, 1=UP, 2=RIGHT, 3=DOWN.
    Rows increase downward, cols increase rightward.
    """
    dy = ty - ay
    dx = tx - ax
    if abs(dy) >= abs(dx):
        return 1 if dy < 0 else 3   # UP or DOWN
    else:
        return 0 if dx < 0 else 2   # LEFT or RIGHT


class RecodeRudakov:
    """Recode substrate + Rudakov-style CC object-directed navigation."""

    def __init__(self, dim=DIM, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, dim).astype(np.float32)
        self.ref = {}
        self.G = {}
        self.C = {}
        self.live = set()
        self._pn = self._pa = self._px = None
        self._cn = None
        self.t = 0
        self.dim = dim
        self._last_visit = {}
        # Object-directed navigation state
        self.color_visits = {}   # color -> times we aimed toward it (cross-episode)
        self.prev_arr = None
        self.agent_yx = None     # estimated agent position
        self._curr_arr = None
        self.obj_actions = 0     # times object-directed action was chosen
        self.fb_actions = 0      # fallback argmin count

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
        x = enc(frame)
        arr = np.array(frame[0], dtype=np.int32)

        # Update agent position from frame diff
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr)
            changed_mask = diff > 0
            n_changed = int(changed_mask.sum())
            if n_changed >= 1 and n_changed < 200:
                ys, xs = np.where(changed_mask)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))

        self.prev_arr = arr.copy()
        self._curr_arr = arr

        # Standard Recode observe
        n = self._node(x)
        self.live.add(n)
        self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(self.dim, np.float64), 0))
            self.C[k_key] = (s + self._px.astype(np.float64), c + 1)
        self._px = x
        self._cn = n
        if self.t > 0 and self.t % REFINE_EVERY == 0:
            self._refine()
        return n

    def act(self):
        """Object-directed if novel CC visible, else argmin."""
        if self._curr_arr is not None and self.agent_yx is not None:
            objects = segment(self._curr_arr)
            if objects:
                ay, ax = self.agent_yx
                best_action = None
                best_score = -1e9

                for obj in objects:
                    visits = self.color_visits.get(obj['color'], 0)
                    novelty = max(0.0, float(NOVEL_VISITS - visits))
                    dist = ((obj['cy'] - ay) ** 2 + (obj['cx'] - ax) ** 2) ** 0.5
                    if dist < 3.0:
                        continue  # already at/adjacent to object
                    # Novel objects get big bonus; all objects considered by proximity
                    score = novelty * 20.0 - dist
                    if score > best_score:
                        best_score = score
                        best_action = dir_action(obj['cy'], obj['cx'], ay, ax)
                        chosen_color = obj['color']

                if best_action is not None and best_score > -1e9:
                    self.color_visits[chosen_color] = self.color_visits.get(chosen_color, 0) + 1
                    self._pn = self._cn
                    self._pa = best_action
                    self.obj_actions += 1
                    return best_action

        # Fallback: argmin (least-visited action)
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
        # Keep color_visits (cross-episode learning)

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
    rng = np.random.RandomState(99)

    # Test segment() with 64x64 array (background=0 fills most, gets filtered by MAX_OBJ_SIZE)
    arr = np.zeros((64, 64), dtype=np.int32)
    arr[1:4, 1:4] = 5   # 9-pixel object, color=5
    arr[7, 7] = 3       # 1-pixel, too small, filtered
    arr[5:8, 5:8] = 3   # 9-pixel object, color=3
    objects = segment(arr)
    colors_found = {o['color'] for o in objects}
    assert 5 in colors_found, f"Should find color 5, found {colors_found}"
    assert 3 in colors_found, f"Should find color 3, found {colors_found}"
    # Background (0) fills ~4078 pixels > MAX_OBJ_SIZE=700, should be filtered
    assert 0 not in colors_found, f"Background should be filtered"

    # Test dir_action
    # Target below-right: DOWN or RIGHT
    a = dir_action(10, 10, 5, 5)
    assert a in (2, 3), f"Expected RIGHT or DOWN, got {a}"
    # Target above: UP
    a = dir_action(2, 5, 8, 5)
    assert a == 1, f"Expected UP(1), got {a}"
    # Target left: LEFT
    a = dir_action(5, 2, 5, 8)
    assert a == 0, f"Expected LEFT(0), got {a}"
    # Target right: RIGHT
    a = dir_action(5, 8, 5, 2)
    assert a == 2, f"Expected RIGHT(2), got {a}"

    # Test RecodeRudakov with synthetic frames
    sub = RecodeRudakov(seed=0)
    frame1 = [rng.randint(0, 16, (64, 64))]
    sub.observe(frame1)
    sub.act()

    # After one step, act() falls back (no prev_arr diff yet)
    frame2 = [rng.randint(0, 16, (64, 64))]
    sub.observe(frame2)
    a = sub.act()
    assert a in range(N_A)

    # Test on_reset
    sub.on_reset()
    assert sub.prev_arr is None
    assert sub.agent_yx is None

    print("T0 PASS")


def main():
    t0()

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}")
        return

    n_seeds = 5
    global_cap = 280
    R = []
    t_start = time.time()

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nGlobal cap hit at seed {seed}", flush=True)
            break
        seeds_left = n_seeds - seed
        budget = (global_cap - elapsed) / seeds_left
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)

        env = mk()
        sub = RecodeRudakov(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1 = l2 = None
        go = 0
        deadline = time.time() + budget

        for step in range(1, 500_001):
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
                if cl == 1 and l1 is None:
                    l1 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L1@{step} c={nc} go={go} "
                          f"obj={sub.obj_actions} fb={sub.fb_actions}", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L2@{step} c={nc} go={go} "
                          f"obj={sub.obj_actions} fb={sub.fb_actions} "
                          f"color_visits={dict(list(sub.color_visits.items())[:5])}", flush=True)

            if step % 25_000 == 0:
                nc, ns, ne = sub.stats()
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={nc} go={go} "
                      f"obj={sub.obj_actions} fb={sub.fb_actions} {el:.0f}s", flush=True)

            if time.time() > deadline:
                break

        nc, ns, ne = sub.stats()
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, go=go,
                      steps=step, level=level, obj=sub.obj_actions,
                      fb=sub.fb_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")
    print(f"\nResults (CC segmentation, novel-directed):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  obj={r['obj']:>6}  fb={r['fb']:>6}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    if not R:
        print("No results.")
        return

    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None

    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    print(f"Baseline (Step 554): L1=3/3 at ~15K steps")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps (baseline 15164)")

    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}. CC segmentation navigates to energy palettes!")
        print("Rudakov replication WORKS. Object-directed navigation reaches L2.")
    elif l1n >= 3:
        print(f"\nL1={l1n}/{len(R)} preserved. L2=0: Palettes not detected or 129-step budget too tight.")
        print("CC segmentation helps with exploration but can't reach iri sprites in budget.")
    elif l1n < 3:
        print(f"\nKILL: L1={l1n}/{len(R)} < 3. Object-chasing hurts navigation.")
        print("Agent too focused on objects to reach exit consistently.")


if __name__ == "__main__":
    main()
