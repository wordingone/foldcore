"""
Step 569 — Exit-last TSP: fix agent_yx=None bug from Step 568.

Step 568 ran as pure argmin (ta=0) because on_reset() set agent_yx=None,
then act() detection fired but agent_yx=None → visit_order not rebuilt.
visit_ptr stayed exhausted → no target actions.

Fix: in act(), after detection block, rebuild visit_order if agent_yx is
now available AND visit_order is empty or ptr is exhausted.

Kill: ta=0 again OR L1<3/5 -> fix broke navigation.
Find: L2>0 -> FIRST L2 EVER (exit-last TSP works).
5-min cap. LS20. 5 seeds. Actions: 0=UP,1=DOWN,2=LEFT,3=RIGHT.
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
MAX_CLUSTER = 20
VISIT_DIST = 4
REDETECT_EVERY = 500


def dir_action(ty, tx, ay, ax):
    dy, dx = ty - ay, tx - ax
    if abs(dy) >= abs(dx): return 0 if dy < 0 else 1
    return 2 if dx < 0 else 3


def find_rare_clusters(mode_arr):
    total = mode_arr.size
    colors, counts = np.unique(mode_arr, return_counts=True)
    rare_colors = colors[counts < total * RARE_THRESH]
    clusters = []
    for color in rare_colors:
        labeled, n = ndlabel(mode_arr == color)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                                 'color': int(color), 'size': sz})
    return clusters


def greedy_order(targets, ay, ax, exit_idx=None):
    """Greedy nearest-first; if exit_idx set, force that target last."""
    remaining = list(range(len(targets)))
    if exit_idx is not None and exit_idx in remaining:
        remaining.remove(exit_idx)
    order = []
    cy, cx = ay, ax
    while remaining:
        best = min(remaining, key=lambda i: (targets[i]['cy']-cy)**2 + (targets[i]['cx']-cx)**2)
        remaining.remove(best)
        order.append(best)
        cy, cx = targets[best]['cy'], targets[best]['cx']
    if exit_idx is not None:
        order.append(exit_idx)
    return order


class RecodeExitLastFix:
    def __init__(self, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, FG_DIM).astype(np.float32)
        self.ref, self.G, self.C = {}, {}, {}
        self.live = set()
        self._pn = self._pa = self._px = self._cn = None
        self.t = 0
        self._last_visit = {}
        self.freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.mode = np.zeros((64, 64), dtype=np.int32)
        self.n_frames = 0
        self.targets = []
        self.exit_idx = None
        self.visit_order = []
        self.visit_ptr = 0
        self.agent_yx = None
        self.prev_arr = None
        self._steps_since_detect = REDETECT_EVERY
        self.target_actions = 0
        self.fb_actions = 0
        self.n_targets_found = 0
        self._last_visited_idx = None
        self._visit_order_built = False  # track whether order built this episode

    def _update_bg(self, arr):
        r, c = np.arange(64)[:, None], np.arange(64)[None, :]
        self.freq[r, c, arr] += 1
        self.n_frames += 1
        if self.n_frames % MODE_EVERY == 0:
            self.mode = np.argmax(self.freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        return (arr != self.mode).astype(np.float32).flatten()

    def _base(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

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
        x = (arr.astype(np.float32).flatten()/15.0 - arr.mean()/15.0) if self.n_frames < WARMUP else self._fg_enc(arr)
        n = self._node(x)
        self.live.add(n); self.t += 1; self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {}); d[n] = d.get(n, 0) + 1
            k_key = (self._pn, self._pa, n)
            s, c = self.C.get(k_key, (np.zeros(FG_DIM, np.float64), 0))
            self.C[k_key] = (s + x.astype(np.float64), c + 1)
        self._px = x; self._cn = n
        self._steps_since_detect += 1
        if self.t % REFINE_EVERY == 0: self._refine()
        return n

    def notify_l1(self):
        """Called when L1 achieved. Last visited target = exit."""
        if self._last_visited_idx is not None and self.exit_idx is None:
            self.exit_idx = self._last_visited_idx

    def _rebuild_order(self):
        """Rebuild visit_order from current agent position."""
        if self.targets and self.agent_yx:
            ay, ax = self.agent_yx
            self.visit_order = greedy_order(self.targets, ay, ax, self.exit_idx)
            self.visit_ptr = 0
            self._visit_order_built = True

    def act(self):
        # Scheduled redetection
        if self._steps_since_detect >= REDETECT_EVERY and self.n_frames >= WARMUP:
            self.targets = find_rare_clusters(self.mode)
            self.n_targets_found = len(self.targets)
            self._steps_since_detect = 0
            if self.targets and self.agent_yx:
                self._rebuild_order()

        # FIX: if visit_order not yet built this episode but we now have agent_yx, build it
        if self.targets and self.agent_yx and not self._visit_order_built:
            self._rebuild_order()

        if self.targets and self.agent_yx and self.visit_order:
            ay, ax = self.agent_yx
            # Advance past reached targets
            while self.visit_ptr < len(self.visit_order):
                ti = self.visit_order[self.visit_ptr]
                t = self.targets[ti]
                dist = ((t['cy']-ay)**2 + (t['cx']-ax)**2)**0.5
                if dist < VISIT_DIST:
                    self._last_visited_idx = ti
                    self.visit_ptr += 1
                else:
                    break
            if self.visit_ptr < len(self.visit_order):
                ti = self.visit_order[self.visit_ptr]
                t = self.targets[ti]
                action = dir_action(t['cy'], t['cx'], ay, ax)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1
                return action

        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; self.fb_actions += 1
        return action

    def on_reset(self):
        self._pn = None; self.prev_arr = None; self.agent_yx = None
        self._last_visited_idx = None
        self._steps_since_detect = REDETECT_EVERY
        self._visit_order_built = False  # FIX: mark order as stale, will rebuild when agent_yx available
        self.visit_ptr = 0
        # Don't clear visit_order here — _visit_order_built=False ensures rebuild on next act()

    def _h(self, n, a):
        d = self.G.get((n, a))
        if not d or sum(d.values()) < 4: return 0.0
        v = np.array(list(d.values()), np.float64); p = v/v.sum()
        return float(-np.sum(p * np.log2(np.maximum(p, 1e-15))))

    def _refine(self):
        for (n, a), d in list(self.G.items()):
            if n not in self.live or n in self.ref: continue
            if len(d) < 2 or sum(d.values()) < MIN_OBS: continue
            if self._h(n, a) < H_SPLIT: continue
            top = sorted(d, key=d.get, reverse=True)[:2]
            r0, r1 = self.C.get((n,a,top[0])), self.C.get((n,a,top[1]))
            if r0 is None or r1 is None or r0[1] < 2 or r1[1] < 2: continue
            diff = (r0[0]/r0[1]) - (r1[0]/r1[1]); nm = np.linalg.norm(diff)
            if nm < 1e-8: continue
            self.ref[n] = (diff/nm).astype(np.float32); self.live.discard(n)

    def stats(self): return len(self.live), len(self.ref), len(self.G)


def t0():
    # Test: _visit_order_built flag ensures order rebuilt after on_reset
    rng = np.random.RandomState(0)
    sub = RecodeExitLastFix(seed=0)
    # Warmup
    for _ in range(WARMUP + 10):
        sub.observe([rng.randint(0, 16, (64, 64))])
        sub.act()

    # Simulate reset + set targets
    sub.on_reset()
    assert not sub._visit_order_built, "Flag should be False after reset"
    sub.targets = [
        {'cy': 10, 'cx': 10, 'color': 3, 'size': 5},
        {'cy': 30, 'cx': 30, 'color': 5, 'size': 3},
        {'cy': 50, 'cx': 20, 'color': 7, 'size': 4},
    ]
    sub.n_targets_found = 3

    # First act() with agent_yx=None: no order built, falls through to argmin
    sub.agent_yx = None
    sub._steps_since_detect = 0  # prevent redetect
    a = sub.act()
    assert not sub._visit_order_built, "Should not build without agent_yx"

    # Set agent_yx, next act() should build order
    sub.agent_yx = (32, 32)
    a = sub.act()
    assert sub._visit_order_built, "Should build when agent_yx available"
    assert len(sub.visit_order) == 3, f"Expected 3 targets, got {len(sub.visit_order)}"

    # Test exit_idx forced last
    sub.on_reset()
    sub._visit_order_built = False
    sub.exit_idx = 0
    sub.agent_yx = (32, 32)
    sub._steps_since_detect = 0
    sub.act()
    assert sub.visit_order[-1] == 0, f"exit_idx=0 should be last: {sub.visit_order}"

    # Test greedy_order
    tgts = [{'cy': 0, 'cx': 0}, {'cy': 10, 'cx': 0}, {'cy': 20, 'cx': 0}]
    order = greedy_order(tgts, 5, 0, exit_idx=0)
    assert order[-1] == 0, f"exit should be last: {order}"

    print("T0 PASS")


def main():
    t0()
    try:
        sys.path.insert(0, '.'); import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds, global_cap, R = 5, 280, []
    t_start = time.time()

    for seed in range(n_seeds):
        elapsed = time.time() - t_start
        if elapsed > global_cap - 10: print(f"\nCap hit at seed {seed}", flush=True); break
        budget = (global_cap - elapsed) / (n_seeds - seed)
        print(f"\nseed {seed} (budget={budget:.0f}s):", flush=True)
        env = mk(); sub = RecodeExitLastFix(seed=seed*1000)
        obs = env.reset(seed=seed); level = 0; l1 = l2 = None; go = 0
        deadline = time.time() + budget

        for step in range(1, 500_001):
            if obs is None: obs = env.reset(seed=seed); sub.on_reset(); continue
            sub.observe(obs); action = sub.act()
            obs, reward, done, info = env.step(action)
            if done: go += 1; obs = env.reset(seed=seed); sub.on_reset()
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl; sub.on_reset()
                if cl == 1 and l1 is None:
                    l1 = step; sub.notify_l1()
                    print(f"  s{seed} L1@{step} exit_idx={sub.exit_idx} "
                          f"tgt={sub.n_targets_found} ta={sub.target_actions} go={go}", flush=True)
                if cl == 2 and l2 is None:
                    l2 = step
                    print(f"  s{seed} L2@{step}!! FIRST L2! exit_idx={sub.exit_idx} "
                          f"tgt={sub.n_targets_found} ta={sub.target_actions}", flush=True)
            if step % 25_000 == 0:
                el = time.time() - t_start
                print(f"  s{seed} @{step} c={sub.stats()[0]} go={go} "
                      f"ei={sub.exit_idx} ta={sub.target_actions} {el:.0f}s", flush=True)
            if time.time() > deadline: break

        nc = sub.stats()[0]
        R.append(dict(seed=seed, l1=l1, l2=l2, cells=nc, go=go, steps=step,
                      level=level, n_targets=sub.n_targets_found,
                      exit_idx=sub.exit_idx, ta=sub.target_actions, fb=sub.fb_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    print(f"\nResults (exit-last TSP, bug fixed):")
    for r in R:
        tag = "L2!" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag:>3}  c={r['cells']:>5}  go={r['go']:>4}  "
              f"steps={r['steps']:>7}  ta={r['ta']:>6}  ei={r['exit_idx']}  tgt={r['n_targets']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    if not R: print("No results."); return
    l1s = [r['l1'] for r in R if r['l1']]
    avg_ta = np.mean([r['ta'] for r in R]) if R else 0
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}  avg_ta={avg_ta:.0f}")
    if l1s: print(f"Avg L1: {np.mean(l1s):.0f} steps (568 bug: 24374, 567 baseline: 468)")
    if l2n > 0:
        print(f"\n*** FIND: L2={l2n}/{len(R)}! FIRST L2 EVER! ***")
        print("Exit-last TSP visits palette before exit!")
    elif avg_ta < 100:
        print(f"\nKILL: avg_ta={avg_ta:.0f} (still near 0). Bug not fixed.")
    elif l1n >= 3:
        print(f"\nL1={l1n}/{len(R)}, L2=0. ta={avg_ta:.0f} (targeting active).")
        print("Palette not among rare targets OR visit order doesn't help within 129 steps.")
    else:
        print(f"\nKILL: L1={l1n}/{len(R)} < 3. Exit-last strategy hurts navigation.")


if __name__ == "__main__":
    main()
