"""
Step 572b — Multi-episode Level 2 mapping for L2.

Two mode maps: l1_mode and l2_mode. l2_mode accumulates across L1->L2 transitions.
After N_MAP=30 cycles, detect Level 2 rare-color targets and use them for targeting.

Phase 1: Level 1 targeting (same as Step 567)
Phase 2: Level 2 glimpse (129 steps) — accumulate l2_mode, don't reset on death
Phase 3: After 30 cycles, detect l2_targets, use greedy targeting on Level 2
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel

N_A = 4
K = 16
FG_DIM = 4096
MODE_EVERY = 200
WARMUP = 100
RARE_THRESH = 0.05
MIN_CLUSTER = 2
MAX_CLUSTER = 30
VISIT_DIST = 4
REDETECT_EVERY = 500
N_MAP = 30  # cycles to build l2_mode before targeting


def dir_action(ty, tx, ay, ax):
    dy = ty - ay; dx = tx - ax
    if abs(dy) >= abs(dx): return 0 if dy < 0 else 1
    else: return 2 if dx < 0 else 3


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
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                                 'color': int(color), 'size': sz})
    return clusters


class SubDual:
    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, FG_DIM).astype(np.float32)
        self.G = {}; self.C = {}; self.ref = {}; self.live = set()
        self._pn = self._pa = self._px = self._cn = None
        self.t = 0; self._last_visit = {}
        # L1 mode map
        self.l1_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l1_mode = np.zeros((64, 64), dtype=np.int32)
        self.l1_frames = 0
        # L2 mode map (accumulates across episodes)
        self.l2_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l2_mode = np.zeros((64, 64), dtype=np.int32)
        self.l2_frames = 0
        # State
        self.game_level = 0  # 0=on Level 1, 1=on Level 2
        self.l1_cycles = 0   # how many times we've entered Level 2
        self.l1_targets = []; self.l2_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = REDETECT_EVERY
        # Stats
        self.target_actions = 0; self.fb_actions = 0
        self.n_l1_tgt = 0; self.n_l2_tgt = 0

    def _update_bg(self, arr):
        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        if self.game_level == 0:
            self.l1_freq[r, c, arr] += 1; self.l1_frames += 1
            if self.l1_frames % MODE_EVERY == 0:
                self.l1_mode = np.argmax(self.l1_freq, axis=2).astype(np.int32)
        else:
            self.l2_freq[r, c, arr] += 1; self.l2_frames += 1
            if self.l2_frames % MODE_EVERY == 0:
                self.l2_mode = np.argmax(self.l2_freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        mode = self.l1_mode if self.game_level == 0 else self.l2_mode
        return (arr != mode).astype(np.float32).flatten()

    def _base(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
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
        frames = self.l1_frames if self.game_level == 0 else self.l2_frames
        if frames < WARMUP:
            x = arr.astype(np.float32).flatten() / 15.0; x = x - x.mean()
        else:
            x = self._fg_enc(arr)
        n = self._node(x); self.live.add(n); self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {}); d[n] = d.get(n, 0) + 1
        self._px = x; self._cn = n; self._steps_since_detect += 1
        return n

    def on_l1(self):
        """Called when Level 2 entered."""
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        # After N_MAP cycles, detect l2 targets
        if self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self.l2_targets = find_rare_clusters(self.l2_mode)
            self.n_l2_tgt = len(self.l2_targets)

    def on_reset(self):
        """Called on episode death — back to Level 1."""
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        self._pn = None

    def _active_targets(self):
        return self.l2_targets if self.game_level == 1 else self.l1_targets

    def _active_mode(self):
        return self.l2_mode if self.game_level == 1 else self.l1_mode

    def _active_frames(self):
        return self.l2_frames if self.game_level == 1 else self.l1_frames

    def act(self):
        # Redetect L1 targets periodically when on Level 1
        if (self.game_level == 0 and self._steps_since_detect >= REDETECT_EVERY
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_rare_clusters(self.l1_mode)
            self.n_l1_tgt = len(self.l1_targets)
            self._steps_since_detect = 0

        targets = self._active_targets()
        if targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_dist = 1e9
            for t in targets:
                if any(((t['cy']-vy)**2+(t['cx']-vx)**2)<VISIT_DIST**2 for vy,vx in self.visited):
                    continue
                dist = ((t['cy']-ay)**2+(t['cx']-ax)**2)**0.5
                if dist < best_dist: best_dist = dist; best = t
            if best is not None:
                if best_dist < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    action = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn; self._pa = action
                    self.target_actions += 1; return action

        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; self.fb_actions += 1; return action


def t0():
    rng = np.random.RandomState(0)
    sub = SubDual(seed=0)
    for _ in range(5):
        f = [rng.randint(0, 16, (64, 64))]; sub.observe(f); sub.act()
    assert sub.game_level == 0 and sub.l1_frames == 5
    sub.on_l1()
    assert sub.game_level == 1
    for _ in range(3):
        f = [rng.randint(0, 16, (64, 64))]; sub.observe(f); sub.act()
    assert sub.l2_frames == 3 and sub.l1_frames == 5
    sub.on_reset()
    assert sub.game_level == 0
    print("T0 PASS")


def main():
    t0()
    try:
        sys.path.insert(0, '.'); import arcagi3; mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 5; per_seed_cap = 300; R = []; t_start = time.time()

    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        env = mk(); sub = SubDual(seed=seed * 1000)
        obs = env.reset(seed=seed)
        level = 0; l1_step = l2_step = None; go = 0; seed_start = time.time()

        for step in range(1, 200_001):
            if obs is None:
                obs = env.reset(seed=seed); sub.on_reset(); continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)
            if done:
                go += 1; obs = env.reset(seed=seed); sub.on_reset()
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > level:
                level = cl
                if cl == 1 and l1_step is None:
                    l1_step = step
                    sub.on_l1()
                    print(f"  s{seed} L1@{step} cycle={sub.l1_cycles} "
                          f"l2f={sub.l2_frames} go={go}", flush=True)
                elif cl == 1:
                    sub.on_l1()  # subsequent L1 events
                if cl == 2 and l2_step is None:
                    l2_step = step
                    l2c = find_rare_clusters(sub.l2_mode)
                    print(f"  s{seed} L2@{step}! cycle={sub.l1_cycles} "
                          f"l2_clusters={[(c['color'],c['size']) for c in l2c]}", flush=True)
                if cl >= 1: level = max(level, cl)  # don't go backward on reset
            # Track re-entry into Level 2
            elif cl == 0 and level == 1:
                pass  # still on Level 1 tracking

            if step % 25_000 == 0:
                el = time.time() - seed_start
                print(f"  s{seed} @{step} cycles={sub.l1_cycles} l1f={sub.l1_frames} "
                      f"l2f={sub.l2_frames} l2tgt={sub.n_l2_tgt} go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        # Log Level 2 clusters found
        if sub.l2_frames >= 100:
            l2c = find_rare_clusters(sub.l2_mode)
            l2c_str = [(c['color'], c['size'], f"({c['cy']:.0f},{c['cx']:.0f})") for c in l2c]
            print(f"  s{seed} L2 clusters: {l2c_str}", flush=True)

        R.append(dict(seed=seed, l1=l1_step, l2=l2_step, go=go, steps=step,
                      cycles=sub.l1_cycles, l2_frames=sub.l2_frames,
                      n_l2_tgt=sub.n_l2_tgt, ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag}  cycles={r['cycles']}  l2f={r['l2_frames']}  "
              f"l2tgt={r['n_l2_tgt']}  go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if l2n > 0: print(f"\nFIND: L2={l2n}/{len(R)}! Multi-episode L2 mapping works!")
    else: print(f"\nKILL: L2=0. Puzzle too complex for stochastic coverage in 129 steps.")


if __name__ == "__main__":
    main()
