"""
Step 572h — Wall escape for L2 navigation.

572g: lhs_v=0, kdy_t=0 across ALL seeds. NAVIGATION FAILURE confirmed.
  - c5=(42.02, 16.02) and c0=(46.67, 51.33) detected in l2_targets
  - Agent takes 150K target-directed actions but NEVER gets within VISIT_DIST=4
  - Greedy dir_action bounces off maze walls; fallback argmin([0,0,0,0])=0 = always action 0

Root causes:
  1. dir_action is stuck against a wall → same position, same action, no progress
  2. graph G is empty for novel L2 states → fallback always returns action 0 (up)

FIX: stuck detection + random escape
  - Track last N_STUCK=5 positions in L2
  - If std dev of recent positions < STUCK_THRESH=0.5: STUCK
  - When stuck: take ESCAPE_STEPS=20 random actions to explore around the wall
  - After escape: resume normal navigation

This breaks the oscillation without needing BFS or maze map.
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
MIN_CLUSTER = 2
MAX_CLUSTER = 60
VISIT_DIST = 4
REDETECT_EVERY = 500
N_MAP = 30

N_STUCK = 5        # steps to check for stuck condition
STUCK_THRESH = 0.5 # max std dev of recent positions to declare stuck
ESCAPE_STEPS = 20  # random steps to take when stuck


def find_isolated_clusters(mode_arr):
    """Find all small isolated clusters regardless of color frequency."""
    clusters = []
    for color in range(16):
        mask = (mode_arr == color)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        for cid in range(1, n + 1):
            region = (labeled == cid)
            sz = int(region.sum())
            if MIN_CLUSTER <= sz <= MAX_CLUSTER:
                ys, xs = np.where(region)
                clusters.append({'cy': float(ys.mean()), 'cx': float(xs.mean()),
                                 'color': int(color), 'size': sz})
    return clusters


def dir_action(ty, tx, ay, ax):
    dy = ty - ay; dx = tx - ax
    if abs(dy) >= abs(dx): return 0 if dy < 0 else 1
    else: return 2 if dx < 0 else 3


class SubDual:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.H = self.rng.randn(K, FG_DIM).astype(np.float32)
        self.G = {}; self.ref = {}; self.live = set()
        self._pn = self._pa = self._cn = None
        self.t = 0; self._last_visit = {}
        self.l1_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l1_mode = np.zeros((64, 64), dtype=np.int32)
        self.l1_frames = 0
        self.l2_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l2_mode = np.zeros((64, 64), dtype=np.int32)
        self.l2_frames = 0
        self.game_level = 0
        self.l1_cycles = 0
        self.l1_targets = []; self.l2_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = REDETECT_EVERY
        self.target_actions = 0; self.fb_actions = 0
        self.n_l1_tgt = 0; self.n_l2_tgt = 0
        # Stuck detection
        self._recent_pos = []     # recent (ay, ax) positions
        self._escape_remaining = 0
        self.escape_events = 0
        # Diagnostics
        self.lhs_visits = 0
        self.kdy_touches = 0

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
        self._cn = n; self._steps_since_detect += 1

        # Track recent positions in L2 for stuck detection
        if self.game_level == 1 and self.agent_yx is not None:
            ay, ax = self.agent_yx
            self._recent_pos.append((ay, ax))
            if len(self._recent_pos) > N_STUCK:
                self._recent_pos = self._recent_pos[-N_STUCK:]

            # Diagnostic counters
            # lhs at ~(42, 16)
            for t in self.l2_targets:
                if t.get('color') == 5:
                    d = ((ay - t['cy'])**2 + (ax - t['cx'])**2) ** 0.5
                    if d < VISIT_DIST:
                        self.lhs_visits += 1
                        break
            # kdy at color-0 cluster
            for t in self.l2_targets:
                if t.get('color') == 0:
                    d = ((ay - t['cy'])**2 + (ax - t['cx'])**2) ** 0.5
                    if d < VISIT_DIST:
                        self.kdy_touches += 1
                        break

        return n

    def _is_stuck(self):
        """Return True if recent positions show no movement."""
        if len(self._recent_pos) < N_STUCK:
            return False
        ys = [p[0] for p in self._recent_pos]
        xs = [p[1] for p in self._recent_pos]
        return np.std(ys) < STUCK_THRESH and np.std(xs) < STUCK_THRESH

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        self._recent_pos = []; self._escape_remaining = 0
        if self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self.l2_targets = find_isolated_clusters(self.l2_mode)
            self.n_l2_tgt = len(self.l2_targets)

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        self._recent_pos = []; self._escape_remaining = 0
        self._pn = None

    def act(self):
        if (self.game_level == 0 and self._steps_since_detect >= REDETECT_EVERY
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self.n_l1_tgt = len(self.l1_targets)
            self._steps_since_detect = 0

        # Escape mode: random actions to break wall oscillation
        if self.game_level == 1 and self._escape_remaining > 0:
            self._escape_remaining -= 1
            action = int(self.rng.randint(N_A))
            self._pn = self._cn; self._pa = action; self.fb_actions += 1; return action

        targets = self.l2_targets if self.game_level == 1 else self.l1_targets
        if targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_dist = 1e9
            for t in targets:
                if self.game_level == 0:
                    if any(((t['cy']-vy)**2+(t['cx']-vx)**2) < VISIT_DIST**2
                           for vy, vx in self.visited):
                        continue
                dist = ((t['cy']-ay)**2+(t['cx']-ax)**2)**0.5
                if dist < best_dist: best_dist = dist; best = t
            if best is not None:
                if best_dist < VISIT_DIST:
                    if self.game_level == 0:
                        self.visited.append((best['cy'], best['cx']))
                else:
                    # Check if stuck before taking target action
                    if self.game_level == 1 and self._is_stuck():
                        self._escape_remaining = ESCAPE_STEPS
                        self._recent_pos = []
                        self.escape_events += 1
                        action = int(self.rng.randint(N_A))
                        self._pn = self._cn; self._pa = action
                        self.fb_actions += 1; return action
                    action = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn; self._pa = action
                    self.target_actions += 1; return action

        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        action = int(np.argmin(counts))
        self._pn = self._cn; self._pa = action; self.fb_actions += 1; return action


def t0():
    sub = SubDual(seed=0)
    rng = np.random.RandomState(0)
    mode = np.zeros((64, 64), dtype=np.int32)
    mode[40:45, 14:19] = 5
    mode[0:3, 0:64] = 5
    clusters = find_isolated_clusters(mode)
    c5 = [c for c in clusters if c['color'] == 5]
    assert len(c5) == 1, f"Should find exactly 1 isolated color-5 cluster: {c5}"
    assert abs(c5[0]['cy'] - 42) < 2, f"cy should be ~42: {c5[0]['cy']}"
    assert abs(c5[0]['cx'] - 16) < 2, f"cx should be ~16: {c5[0]['cx']}"
    # Test stuck detection
    sub.game_level = 1
    for _ in range(N_STUCK + 1):
        sub._recent_pos.append((40.0, 16.0))
    assert sub._is_stuck(), "Should be stuck when position constant"
    sub._recent_pos = [(40.0 + i*2, 16.0) for i in range(N_STUCK)]
    assert not sub._is_stuck(), "Should not be stuck when moving"
    for _ in range(5):
        f = [rng.randint(0, 16, (64, 64))]; sub.observe(f); sub.act()
    sub.on_l1(); sub.on_reset(); sub.on_l1()
    assert sub.l1_cycles == 2
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
        l1_step = l2_step = None; go = 0; seed_start = time.time()
        prev_cl = 0

        for step in range(1, 200_001):
            if obs is None:
                obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)
            if done:
                go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0
            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > prev_cl:
                if cl == 1:
                    sub.on_l1()
                    if l1_step is None:
                        l1_step = step
                        print(f"  s{seed} L1@{step} cycle={sub.l1_cycles} go={go}", flush=True)
                if cl == 2 and l2_step is None:
                    l2_step = step
                    print(f"  s{seed} L2@{step}! cycle={sub.l1_cycles} go={go}", flush=True)
            prev_cl = cl

            if step % 25_000 == 0:
                el = time.time() - seed_start
                print(f"  s{seed} @{step} cyc={sub.l1_cycles} l2f={sub.l2_frames} "
                      f"lhs_v={sub.lhs_visits} kdy_t={sub.kdy_touches} "
                      f"esc={sub.escape_events} go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        R.append(dict(seed=seed, l1=l1_step, l2=l2_step, go=go, steps=step,
                      cycles=sub.l1_cycles, l2_frames=sub.l2_frames,
                      n_l2_tgt=sub.n_l2_tgt, lhs_v=sub.lhs_visits,
                      kdy_t=sub.kdy_touches, esc=sub.escape_events,
                      ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag}  cyc={r['cycles']}  lhs_v={r['lhs_v']}  "
              f"kdy_t={r['kdy_t']}  esc={r['esc']}  l2tgt={r['n_l2_tgt']}  "
              f"go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! Wall escape works!")
    else:
        lhs_total = sum(r['lhs_v'] for r in R)
        kdy_total = sum(r['kdy_t'] for r in R)
        esc_total = sum(r['esc'] for r in R)
        if lhs_total == 0 and esc_total == 0:
            print(f"\nKILL: L2=0. lhs_v=0, esc=0 — escape never triggered. STUCK_THRESH too tight or not reaching L2.")
        elif lhs_total == 0:
            print(f"\nKILL: L2=0. esc={esc_total} triggered but lhs_v=0 — maze too complex for random escape.")
        else:
            print(f"\nKILL: L2=0. lhs_v={lhs_total}, kdy_t={kdy_total} — "
                  f"lhs reached but state puzzle wrong (need {kdy_total} gdy touches → 3 needed).")


if __name__ == "__main__":
    main()
