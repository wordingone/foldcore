"""
Step 562 — Novel-object detection via frame diff (fix for Step 561).

Only chase objects that APPEAR between consecutive frames (not static walls).
Fast approach: single ndlabel on diff mask (0.04ms/step vs 17ms/step naive).

Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT (per Spec).

Predictions:
  L1: 3/5 (argmin fallback handles L1 when no novel objects)
  L2: 0/5 (palettes may not appear as new CCs — might be static)
  Novel count: < 5 per episode

Kill: L1 < 2/5 -> overhead hurts. novel_count = 0 -> nothing appears.
5-min cap. LS20. 5 seeds. 50K steps per seed.
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
MIN_NOVEL = 3    # min pixels for novel CC
MAX_NOVEL = 20   # max pixels (filter large background changes)
VISITED_DIST = 8  # if novel CC centroid within this dist of recently visited, skip


def enc(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


def find_novel(curr_arr, prev_arr):
    """Find novel CCs via single ndlabel on diff mask.
    Returns list of {cy, cx, size} for each appearing region 3-20px.
    """
    diff = (curr_arr != prev_arr)
    if not np.any(diff):
        return []
    labeled, n = ndlabel(diff)
    if n == 0:
        return []
    results = []
    for cid in range(1, n + 1):
        region = (labeled == cid)
        sz = int(region.sum())
        if sz < MIN_NOVEL or sz > MAX_NOVEL:
            continue
        ys, xs = np.where(region)
        cy, cx = float(ys.mean()), float(xs.mean())
        color = int(curr_arr[int(round(cy)), int(round(cx))])
        results.append({'cy': cy, 'cx': cx, 'size': sz, 'color': color})
    return results


def dir_action(ty, tx, ay, ax):
    """Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT."""
    dy = ty - ay
    dx = tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1   # UP or DOWN
    else:
        return 2 if dx < 0 else 3   # LEFT or RIGHT


class RecodeNovel:
    """Recode + novel-CC-directed navigation."""

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
        # Novel CC state
        self.prev_arr = None
        self.agent_yx = None
        self._curr_arr = None
        self.visited_yx = []    # recently visited novel CC positions
        # Stats
        self.novel_seen = 0     # total novel CCs seen
        self.novel_chased = 0   # times we chased a novel CC
        self.fb_count = 0       # argmin fallbacks

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
            diff_abs = np.abs(arr - self.prev_arr)
            changed = diff_abs > 0
            n_changed = int(changed.sum())
            if 1 <= n_changed < 200:
                ys, xs = np.where(changed)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))

        self.prev_arr = arr.copy()
        self._curr_arr = arr

        # Standard Recode
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
        """Chase novel appearing CCs if found, else argmin."""
        if self._curr_arr is not None and self.prev_arr is not None:
            novel = find_novel(self._curr_arr, self.prev_arr)
            if novel:
                self.novel_seen += len(novel)
                ay = self.agent_yx[0] if self.agent_yx else 32.0
                ax = self.agent_yx[1] if self.agent_yx else 32.0

                # Filter: skip if near a recently-visited position
                candidates = []
                for obj in novel:
                    already = False
                    for (vy, vx) in self.visited_yx[-50:]:  # check last 50
                        if ((obj['cy'] - vy) ** 2 + (obj['cx'] - vx) ** 2) < VISITED_DIST ** 2:
                            already = True
                            break
                    if not already:
                        candidates.append(obj)

                if candidates:
                    # Chase closest unvisited novel CC
                    best = min(candidates,
                               key=lambda o: (o['cy'] - ay) ** 2 + (o['cx'] - ax) ** 2)
                    dist = ((best['cy'] - ay) ** 2 + (best['cx'] - ax) ** 2) ** 0.5
                    if dist > 2.0:
                        action = dir_action(best['cy'], best['cx'], ay, ax)
                        self.visited_yx.append((best['cy'], best['cx']))
                        self._pn = self._cn
                        self._pa = action
                        self.novel_chased += 1
                        return action

        # Fallback: argmin
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn
        self._pa = action
        self.fb_count += 1
        return action

    def on_reset(self):
        self._pn = None
        self.prev_arr = None
        self.agent_yx = None
        self.visited_yx = []    # reset per episode

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
    rng = np.random.RandomState(7)

    # Test find_novel: identical frames -> no novel
    arr = rng.randint(0, 16, (64, 64)).astype(np.int32)
    assert find_novel(arr, arr) == []

    # Test find_novel: small appearing region
    prev = np.zeros((64, 64), dtype=np.int32)
    curr = np.zeros((64, 64), dtype=np.int32)
    curr[10:13, 10:13] = 7  # 9-pixel appearing block
    novel = find_novel(curr, prev)
    assert len(novel) == 1, f"Expected 1 novel CC, got {len(novel)}"
    assert novel[0]['size'] == 9
    assert abs(novel[0]['cy'] - 11.0) < 0.5
    assert abs(novel[0]['cx'] - 11.0) < 0.5

    # Large region: filtered
    curr2 = np.zeros((64, 64), dtype=np.int32)
    curr2[0:64, 0:64] = 5  # 4096 pixels, filtered
    novel2 = find_novel(curr2, prev)
    assert len(novel2) == 0, f"Large region should be filtered"

    # dir_action tests (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
    assert dir_action(2, 5, 8, 5) == 0    # target above -> UP
    assert dir_action(10, 5, 4, 5) == 1   # target below -> DOWN
    assert dir_action(5, 2, 5, 8) == 2    # target left -> LEFT
    assert dir_action(5, 8, 5, 2) == 3    # target right -> RIGHT

    # RecodeNovel smoke test
    sub = RecodeNovel(seed=0)
    frame1 = [rng.randint(0, 16, (64, 64))]
    sub.observe(frame1)
    a = sub.act()
    assert a in range(N_A)
    frame2 = [rng.randint(0, 16, (64, 64))]
    sub.observe(frame2)
    a = sub.act()
    assert a in range(N_A)
    sub.on_reset()
    assert sub.prev_arr is None

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
    total_novel_per_ep = []

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10:
            print(f"\nGlobal cap hit at seed {seed}", flush=True)
            break
        seeds_left = n_seeds - seed
        budget = (global_cap - elapsed) / seeds_left
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)

        env = mk()
        sub = RecodeNovel(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0
        l1 = l2 = None
        go = 0
        deadline = time.time() + budget
        ep_novel = 0   # novel CCs this episode

        for step in range(1, 500_001):
            if obs is None:
                obs = env.reset(seed=seed)
                sub.on_reset()
                continue

            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)

            if done:
                total_novel_per_ep.append(ep_novel)
                ep_novel = 0
                go += 1
                obs = env.reset(seed=seed)
                sub.on_reset()
            else:
                ep_novel += max(0, sub.novel_seen - sum(total_novel_per_ep) - ep_novel)

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                sub.on_reset()
                if cl == 1 and l1 is None:
                    l1 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L1@{step} c={nc} go={go} "
                          f"novel={sub.novel_seen} chased={sub.novel_chased} "
                          f"fb={sub.fb_count}", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    nc, ns, ne = sub.stats()
                    print(f"  s{seed} L2@{step} c={nc} go={go} "
                          f"novel={sub.novel_seen} chased={sub.novel_chased} "
                          f"fb={sub.fb_count}", flush=True)

            if step % 25_000 == 0:
                nc, ns, ne = sub.stats()
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={nc} go={go} "
                      f"novel={sub.novel_seen} chased={sub.novel_chased} "
                      f"fb={sub.fb_count} {el:.0f}s", flush=True)

            if time.time() > deadline:
                break

        nc, ns, ne = sub.stats()
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, go=go,
                      steps=step, level=level, novel=sub.novel_seen,
                      chased=sub.novel_chased, fb=sub.fb_count))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Elapsed: {elapsed:.0f}s")

    if total_novel_per_ep:
        arr = np.array(total_novel_per_ep)
        print(f"Novel CCs per episode: n={len(arr)} mean={arr.mean():.1f} "
              f"max={arr.max()} median={np.median(arr):.1f}")

    print(f"\nResults (novel CC detection, diff-guided):")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  novel={r['novel']:>5}  chased={r['chased']:>5}  "
              f"fb={r['fb']:>6}")

    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    if not R:
        print("No results.")
        return

    l1_steps = [r['l1'] for r in R if r['l1']]
    avg_l1 = np.mean(l1_steps) if l1_steps else None
    total_novel = sum(r['novel'] for r in R)
    total_chased = sum(r['chased'] for r in R)

    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    print(f"Total novel seen: {total_novel}  chased: {total_chased}")
    if avg_l1:
        print(f"Avg L1: {avg_l1:.0f} steps (baseline 15164, diff-guided 7318)")
    print(f"Baseline (Step 554): L1=3/3 at ~15K steps")

    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}. Novel-CC navigation reaches energy palettes!")
    elif l1n >= 3:
        if total_novel == 0:
            print(f"\nL1={l1n}/{len(R)}, L2=0/5, novel=0.")
            print("KILL (soft): No novel CCs detected. Palettes are static/always visible.")
            print("Frame-diff approach can't detect static energy palettes.")
        else:
            print(f"\nL1={l1n}/{len(R)}, L2=0/5, novel={total_novel}.")
            print("Novel CCs detected but agent can't navigate to palettes within 129 steps.")
            print("Palettes may be visible but agent doesn't reach them in energy budget.")
    elif l1n < 2:
        print(f"\nKILL: L1={l1n}/{len(R)}. Novel-CC overhead disrupts L1 navigation.")


if __name__ == "__main__":
    main()
