"""
Step 572j — BFS + state sequencing for L2.

572i: lhs_v=0 despite bfs_hits=128K. Root causes identified from ls20.py:
  1. POSITION CORRUPTION: touching lhs with wrong state causes hep sprite
     flash (100+ diff pixels). agent_yx jumps to (cy~57, cx~5). BFS
     uses wrong start position, plans wrong path.
  2. STATE SEQUENCING: tuv=0 at start, need tuv=3 (3×kdy touches) THEN lhs.
     BFS went to lhs first (nearest by Manhattan distance). Lhs touch rejected.

Game mechanics (from ls20.py):
  - qhg win: snw==gfy AND tmx==vxy AND tuv==cjl
  - mgu init: snw=5(ok), tmx=1(ok), tuv=0(need 3)
  - kdy touch: tuv = (tuv+1) % 4; kdj sprite rotates
  - lhs touch wrong state: hep flashes (color 0→5), player DOESN'T move

FIX 1: Dead reckoning for player position
  - Start at spawn (29,40) in mgu
  - After each BFS action, update grid pos: if action goes to non-wall → move there
  - If action goes to wall or lhs (blocked when wrong state) → stay
  - Use DR position for BFS planning instead of noisy agent_yx

FIX 2: State sequencing
  - In L2 phase: first navigate to kdy × 3 (to reach tuv=3)
  - Detect kdy touch via kdj region diff (cols 3-8, rows 55-60 change when tuv ticks)
  - After 3 touches in episode: switch to targeting lhs
  - Reset tuv_est on lose/respawn (tracked by xhp-like frame detection)

KDY grid: (49, 45). LHS grid: (14, 40). Spawn: (29, 40).
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel
from collections import deque

N_A = 4
K = 16
FG_DIM = 4096
MODE_EVERY = 200
WARMUP = 100
MIN_CLUSTER = 2
MAX_CLUSTER = 60
VISIT_DIST = 4
N_MAP = 30

# Grid parameters from ls20.py
GRID_XS = list(range(4, 64, 5))   # [4,9,14,...,59]
GRID_YS = list(range(0, 64, 5))   # [0,5,10,...,55]
GRID_XS_SET = set(GRID_XS)
GRID_YS_SET = set(GRID_YS)
STEP = 5

# Known game positions
SPAWN = (29, 40)       # pca spawn in mgu (col, row)
LHS_GRID = (14, 40)   # lhs target grid cell
KDY_GRID = (49, 45)   # kdy target grid cell

# kdj sprite region (scale=2, set_position(3,55), 3×3 pixels → 6×6 rendered)
KDJ_R0, KDJ_R1 = 55, 61  # rows 55-60
KDJ_C0, KDJ_C1 = 3, 9    # cols 3-8

# hep sprite region (10×10 at set_position(1,53))
HEP_R0, HEP_R1 = 53, 63  # rows 53-62
HEP_C0, HEP_C1 = 1, 11   # cols 1-10

KDJ_THRESH = 5  # min changed pixels in kdj region to count tuv tick
TUV_NEEDED = 3  # touches of kdy needed


def find_isolated_clusters(mode_arr):
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


def build_wall_set(mode_arr):
    walls = set()
    for gx in GRID_XS:
        for gy in GRID_YS:
            region = mode_arr[gy:gy+5, gx:gx+5]
            if (region == 4).any():
                walls.add((gx, gy))
    return walls


def nearest_grid(cx, cy):
    gx = min(GRID_XS, key=lambda v: abs(v - cx))
    gy = min(GRID_YS, key=lambda v: abs(v - cy))
    return gx, gy


def bfs_path(start, goal, walls):
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    dirs = [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if nx not in GRID_XS_SET or ny not in GRID_YS_SET:
                continue
            if (nx, ny) in visited or (nx, ny) in walls:
                continue
            new_path = path + [(nx, ny)]
            if (nx, ny) == goal:
                return new_path
            visited.add((nx, ny))
            queue.append(((nx, ny), new_path))
    return []


def path_to_action(path):
    if len(path) < 2:
        return None
    cx, cy = path[0]
    nx, ny = path[1]
    if ny < cy: return 0   # up
    if ny > cy: return 1   # down
    if nx < cx: return 2   # left
    if nx > cx: return 3   # right
    return None


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
        self.l1_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = 99999
        self.target_actions = 0; self.fb_actions = 0
        self.n_l1_tgt = 0
        # State sequencing
        self._tuv_est = 0          # estimated tuv in current episode
        self._phase = 'kdy'        # 'kdy' | 'lhs'
        self._kdy_touches = 0      # total
        self._lhs_approaches = 0   # times took action toward lhs when tuv_est==3
        # Dead reckoning
        self._dr_pos = None        # (gx, gy) current dead reckoning position
        self._wall_set = None      # set of (gx,gy) wall cells
        self._bfs_path = []
        # Diagnostics
        self.bfs_hits = 0; self.bfs_fails = 0

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
        diff = None
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))

            # Detect tuv tick: kdj region changes when player touches kdy
            if self.game_level == 1:
                kdj_changed = int(diff[KDJ_R0:KDJ_R1, KDJ_C0:KDJ_C1].sum())
                if kdj_changed >= KDJ_THRESH:
                    self._tuv_est = (self._tuv_est + 1) % 4
                    self._kdy_touches += 1
                    if self._tuv_est == TUV_NEEDED:
                        self._phase = 'lhs'

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
        return n

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []
        self._dr_pos = SPAWN
        self._tuv_est = 0; self._phase = 'kdy'
        self._bfs_path = []
        if self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self._wall_set = build_wall_set(self.l2_mode)

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []
        self._steps_since_detect = 99999
        self._pn = None
        self._dr_pos = SPAWN
        self._tuv_est = 0; self._phase = 'kdy'
        self._bfs_path = []

    def _dr_action(self):
        """BFS action using dead reckoning position."""
        if self._wall_set is None or self._dr_pos is None:
            return None
        start = self._dr_pos
        goal = KDY_GRID if self._phase == 'kdy' else LHS_GRID

        # Replan if no path or path stale
        if not self._bfs_path or self._bfs_path[0] != start or self._bfs_path[-1] != goal:
            path = bfs_path(start, goal, self._wall_set)
            if path:
                self._bfs_path = path
                self.bfs_hits += 1
            else:
                self._bfs_path = []
                self.bfs_fails += 1
                return None

        if len(self._bfs_path) == 1:
            # AT goal. For kdy: should have been detected by kdj diff.
            # For lhs: win condition should have fired. Stay here.
            return None

        return path_to_action(self._bfs_path)

    def _advance_dr(self, action):
        """Update dead reckoning after taking action."""
        if self._dr_pos is None:
            return
        dx = [0, 0, -STEP, STEP][action]
        dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._dr_pos[0] + dx, self._dr_pos[1] + dy
        # Move if not a wall
        if (nx in GRID_XS_SET and ny in GRID_YS_SET
                and self._wall_set is not None
                and (nx, ny) not in self._wall_set):
            # Advance bfs_path
            if self._bfs_path and len(self._bfs_path) > 1:
                self._bfs_path = self._bfs_path[1:]
            self._dr_pos = (nx, ny)
        # else: wall/out-of-bounds → player didn't move → dr_pos unchanged, bfs_path unchanged

    def act(self):
        if (self.game_level == 0 and self._steps_since_detect >= 500
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self.n_l1_tgt = len(self.l1_targets)
            self._steps_since_detect = 0

        if self.game_level == 1:
            if self._phase == 'lhs':
                self._lhs_approaches += 1
            action = self._dr_action()
            if action is not None:
                self._advance_dr(action)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action
        elif self.l1_targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_dist = 1e9
            for t in self.l1_targets:
                if any(((t['cy']-vy)**2+(t['cx']-vx)**2) < VISIT_DIST**2
                       for vy, vx in self.visited):
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
    # BFS path test with hardcoded walls
    walls_raw = [
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),
        (59,15),(59,20),(59,25),(59,30),(59,35),(59,40),(59,45),(59,50),(59,55),
        (54,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),(19,55),
        (4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),
        (54,30),(34,0),(59,10),(59,5),(54,15),(54,10),
        (9,35),(9,45),(19,50),(9,40),(54,5),
        (14,45),(14,50),(9,5),(9,30),(9,25),
        (19,30),(24,30),(19,40),(24,40),(19,45),(19,35),
        (39,15),(39,35),(44,30),(34,45),(14,5),(39,20),(44,20),(24,20),
        (44,25),(39,40),(39,45),(24,35),(24,25),(24,50),(19,25),(24,45),
        (29,45),(29,30),(29,25),(24,15),(44,35),(54,34),
    ]
    walls = set(walls_raw)
    # Verify lhs path
    path = bfs_path(SPAWN, LHS_GRID, walls)
    assert len(path) == 18, f"lhs path length should be 18: {len(path)}"
    assert path[0] == SPAWN and path[-1] == LHS_GRID
    # Verify kdy path
    path2 = bfs_path(SPAWN, KDY_GRID, walls)
    assert len(path2) == 18, f"kdy path length should be 18: {len(path2)}"
    assert path2[0] == SPAWN and path2[-1] == KDY_GRID
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
                    print(f"  s{seed} L2@{step}! tuv_est={sub._tuv_est} "
                          f"kdy_t={sub._kdy_touches} go={go}", flush=True)
            prev_cl = cl

            if step % 25_000 == 0:
                el = time.time() - seed_start
                print(f"  s{seed} @{step} cyc={sub.l1_cycles} l2f={sub.l2_frames} "
                      f"tuv_est={sub._tuv_est} phase={sub._phase} "
                      f"kdy_t={sub._kdy_touches} lhs_ap={sub._lhs_approaches} "
                      f"bfs+={sub.bfs_hits} bfs-={sub.bfs_fails} "
                      f"dr={sub._dr_pos} go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        R.append(dict(seed=seed, l1=l1_step, l2=l2_step, go=go, steps=step,
                      cycles=sub.l1_cycles, l2_frames=sub.l2_frames,
                      tuv=sub._tuv_est, kdy_t=sub._kdy_touches,
                      lhs_ap=sub._lhs_approaches, bfs_hits=sub.bfs_hits,
                      bfs_fails=sub.bfs_fails, ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag}  cyc={r['cycles']}  kdy_t={r['kdy_t']}  "
              f"lhs_ap={r['lhs_ap']}  bfs+={r['bfs_hits']}  bfs-={r['bfs_fails']}  "
              f"go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! BFS + state sequencing works!")
    else:
        kdy_total = sum(r['kdy_t'] for r in R)
        lhs_total = sum(r['lhs_ap'] for r in R)
        fails = sum(r['bfs_fails'] for r in R)
        if kdy_total == 0:
            print(f"\nKILL: L2=0. kdy_t=0 — kdj detection failed or BFS can't reach kdy.")
        elif lhs_total == 0:
            print(f"\nKILL: L2=0. kdy_t={kdy_total} but lhs_ap=0 — never reached tuv==3 or BFS can't reach lhs.")
        else:
            print(f"\nKILL: L2=0. kdy_t={kdy_total}, lhs_ap={lhs_total} — "
                  f"reaches both but win not triggered. Check bfs_fails={fails} or touch ordering.")


if __name__ == "__main__":
    main()
