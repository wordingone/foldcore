"""
Step 572g — L2 diagnostics: nav failure vs state failure.

572f result: c5=(42.02, 16.02) detected consistently, L2=0/5.
Two hypotheses:
  (A) Navigation failure: greedy dir_action can't reach (42,16) through maze walls
  (B) State failure: agent reaches lhs but tuv≠3 (need 3 kdy touches first)

CHANGE from 572f: add per-L2-episode tracking:
  - lhs_visits: times agent within VISIT_DIST of c5 cluster (cy~42, cx~16) in L2
  - kdy_touches: times agent within VISIT_DIST of any color-0 cluster in L2
  - Log both at 25k checkpoints + summary

If lhs_visits > 0: navigation works → state puzzle failure → fix touch ordering
If lhs_visits == 0: navigation fails → need BFS or different approach

Known mgu layout (set_position = col, row):
  lhs at set_position(14,40) → cy~40-44, cx~14-18 (centroid ~42,16)
  kdy at set_position(49,45) → cy~45-49, cx~49-53 (centroid ~47,51)
  tuv needs 3 kdy touches to reach required value
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

# Expected lhs position from 572f
LHS_CY = 42.02
LHS_CX = 16.02


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
        self.H = np.random.RandomState(seed).randn(K, FG_DIM).astype(np.float32)
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

        # Diagnostics: track proximity to lhs and kdy while in L2
        if self.game_level == 1 and self.agent_yx is not None:
            ay, ax = self.agent_yx
            lhs_dist = ((ay - LHS_CY)**2 + (ax - LHS_CX)**2) ** 0.5
            if lhs_dist < VISIT_DIST:
                self.lhs_visits += 1
            # Track kdy touches: color-0 clusters in l2_targets
            for t in self.l2_targets:
                if t.get('color') == 0:
                    d = ((ay - t['cy'])**2 + (ax - t['cx'])**2) ** 0.5
                    if d < VISIT_DIST:
                        self.kdy_touches += 1
                        break  # count once per step

        return n

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        if self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self.l2_targets = find_isolated_clusters(self.l2_mode)
            self.n_l2_tgt = len(self.l2_targets)

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []; self._steps_since_detect = REDETECT_EVERY
        self._pn = None

    def act(self):
        if (self.game_level == 0 and self._steps_since_detect >= REDETECT_EVERY
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self.n_l1_tgt = len(self.l1_targets)
            self._steps_since_detect = 0

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
                    print(f"  s{seed} L2@{step}! go={go}", flush=True)
            prev_cl = cl

            if step % 25_000 == 0:
                el = time.time() - seed_start
                c5_tgts = [(c['cy'], c['cx']) for c in sub.l2_targets if c.get('color') == 5]
                c0_tgts = [(c['cy'], c['cx']) for c in sub.l2_targets if c.get('color') == 0]
                print(f"  s{seed} @{step} cyc={sub.l1_cycles} l2f={sub.l2_frames} "
                      f"lhs_v={sub.lhs_visits} kdy_t={sub.kdy_touches} "
                      f"c5={c5_tgts} c0={c0_tgts} go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        R.append(dict(seed=seed, l1=l1_step, l2=l2_step, go=go, steps=step,
                      cycles=sub.l1_cycles, l2_frames=sub.l2_frames,
                      n_l2_tgt=sub.n_l2_tgt, lhs_v=sub.lhs_visits,
                      kdy_t=sub.kdy_touches, ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag}  cyc={r['cycles']}  lhs_v={r['lhs_v']}  "
              f"kdy_t={r['kdy_t']}  l2tgt={r['n_l2_tgt']}  go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}!")
    else:
        lhs_total = sum(r['lhs_v'] for r in R)
        kdy_total = sum(r['kdy_t'] for r in R)
        if lhs_total == 0:
            print(f"\nKILL: L2=0. lhs_v=0 across all seeds → NAVIGATION FAILURE. Agent can't reach (42,16).")
        else:
            print(f"\nKILL: L2=0. lhs_v={lhs_total}, kdy_t={kdy_total} → "
                  f"STATE FAILURE. Agent reaches lhs but tuv≠3 (need more kdy touches first).")


if __name__ == "__main__":
    main()
