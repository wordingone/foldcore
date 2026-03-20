"""
Step 572m — Fix prev_cl poisoning bug from terminal frame.

572l failure: L3=0/5, l2c=1 frozen, wall=False.
Root cause: after done=True, terminal frame cl=2 poisons prev_cl=2.
  In next game, cl=1 (mgu) fails cl>prev_cl check (1>2=False).
  on_l1() never fires, game_level stays 0, BFS inactive, l2c stuck at 1.

Fix: add `continue` after resetting prev_cl=0 in done handler.
  Terminal frame cl is never processed.

All puq logic unchanged from 572l:
  PUQ_SPAWN=(9,45), GIC=(29,45), KDY=(49,10), LHS=(54,50)
  Phase: gic (1x) -> kdy (3x) -> lhs -> L3
  N_MAP_PUQ=30 L2s before building puq wall set
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
N_MAP = 30      # mgu cycles before building mgu wall set
N_MAP_PUQ = 30  # L2 completions before building puq wall set

GRID_XS = list(range(4, 64, 5))
GRID_YS = list(range(0, 64, 5))
GRID_XS_SET = set(GRID_XS)
GRID_YS_SET = set(GRID_YS)
STEP = 5

# mgu positions
MGU_SPAWN = (29, 40)
MGU_LHS_GRID = (14, 40)
MGU_KDY_GRID = (49, 45)

# puq positions
PUQ_SPAWN = (9, 45)
PUQ_GIC_GRID = (29, 45)   # "gic" tag - touch once (tmx: 0->1), then avoid
PUQ_KDY_GRID = (49, 10)   # "bgt" tag - touch 3x (tuv: 0->3)
PUQ_LHS_GRID = (54, 50)   # "mae" tag - win condition -> L3

# kdj sprite region (same in all levels)
KDJ_R0, KDJ_R1 = 55, 61
KDJ_C0, KDJ_C1 = 3, 9
KDJ_THRESH = 5
MGU_TUV_NEEDED = 3
PUQ_TUV_NEEDED = 3


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
    if ny < cy: return 0
    if ny > cy: return 1
    if nx < cx: return 2
    if nx > cx: return 3
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
        self.l3_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l3_mode = np.zeros((64, 64), dtype=np.int32)
        self.l3_frames = 0
        self.game_level = 0
        self.l1_cycles = 0
        self.l2_count = 0
        self.l3_count = 0
        self.l1_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = 99999
        self.target_actions = 0
        # mgu state
        self._mgu_tuv_est = 0
        self._mgu_phase = 'kdy'
        self._mgu_dr_pos = None
        self._mgu_wall_set = None
        self._mgu_wall_frozen = False
        self._mgu_bfs_path = []
        self.bfs_hits = 0; self.bfs_fails = 0
        # puq state
        self._puq_phase = 'gic'
        self._puq_tuv_est = 0
        self._puq_gic_done = False
        self._puq_dr_pos = None
        self._puq_wall_set = None
        self._puq_wall_frozen = False
        self._puq_bfs_path = []
        self._puq_avoid_set = set()
        self.puq_bfs_hits = 0; self.puq_bfs_fails = 0
        self._puq_lhs_ap = 0

    def _update_bg(self, arr):
        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        if self.game_level == 0:
            self.l1_freq[r, c, arr] += 1; self.l1_frames += 1
            if self.l1_frames % MODE_EVERY == 0:
                self.l1_mode = np.argmax(self.l1_freq, axis=2).astype(np.int32)
        elif self.game_level == 1:
            self.l2_freq[r, c, arr] += 1; self.l2_frames += 1
            if self.l2_frames % MODE_EVERY == 0:
                self.l2_mode = np.argmax(self.l2_freq, axis=2).astype(np.int32)
        else:
            self.l3_freq[r, c, arr] += 1; self.l3_frames += 1
            if self.l3_frames % MODE_EVERY == 0:
                self.l3_mode = np.argmax(self.l3_freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        if self.game_level == 0: mode = self.l1_mode
        elif self.game_level == 1: mode = self.l2_mode
        else: mode = self.l3_mode
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

            kdj_changed = int(diff[KDJ_R0:KDJ_R1, KDJ_C0:KDJ_C1].sum())
            if kdj_changed >= KDJ_THRESH:
                if self.game_level == 1:   # mgu: track tuv for win
                    self._mgu_tuv_est = (self._mgu_tuv_est + 1) % 4
                    if self._mgu_tuv_est == MGU_TUV_NEEDED:
                        self._mgu_phase = 'lhs'
                elif self.game_level == 2:  # puq: track tuv for win
                    self._puq_tuv_est = (self._puq_tuv_est + 1) % 4
                    if self._puq_tuv_est == PUQ_TUV_NEEDED:
                        self._puq_avoid_set.add(PUQ_KDY_GRID)
                        self._puq_phase = 'lhs'

        self.prev_arr = arr.copy()
        if self.game_level == 0: frames = self.l1_frames
        elif self.game_level == 1: frames = self.l2_frames
        else: frames = self.l3_frames
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
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'
        self._mgu_bfs_path = []
        if self._mgu_wall_frozen:
            pass  # use frozen wall set
        elif self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self._mgu_wall_set = build_wall_set(self.l2_mode)

    def on_l2(self):
        """mgu->puq transition."""
        self.game_level = 2; self.l2_count += 1
        # Freeze mgu wall set
        if not self._mgu_wall_frozen and self._mgu_wall_set is not None:
            self._mgu_wall_frozen = True
        # Init puq state for this episode
        self.visited = []
        self.agent_yx = None
        self._steps_since_detect = 99999
        self._puq_dr_pos = PUQ_SPAWN
        self._puq_phase = 'gic'
        self._puq_tuv_est = 0
        self._puq_gic_done = False
        self._puq_bfs_path = []
        self._puq_avoid_set = set()
        # Build puq wall set after N_MAP_PUQ L2 completions
        if self._puq_wall_frozen:
            pass
        elif self.l2_count >= N_MAP_PUQ and self.l3_frames >= WARMUP:
            self._puq_wall_set = build_wall_set(self.l3_mode)

    def on_l3(self):
        """puq->tmx transition = L3!"""
        self.l3_count += 1
        # Freeze puq wall set after first L3
        if not self._puq_wall_frozen and self._puq_wall_set is not None:
            self._puq_wall_frozen = True

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []
        self._steps_since_detect = 99999
        self._pn = None
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'
        self._mgu_bfs_path = []
        # mgu/puq wall sets preserved

    def _mgu_dr_action(self):
        if self._mgu_wall_set is None or self._mgu_dr_pos is None:
            return None
        start = self._mgu_dr_pos
        goal = MGU_KDY_GRID if self._mgu_phase == 'kdy' else MGU_LHS_GRID
        if not self._mgu_bfs_path or self._mgu_bfs_path[0] != start or self._mgu_bfs_path[-1] != goal:
            path = bfs_path(start, goal, self._mgu_wall_set)
            if path:
                self._mgu_bfs_path = path; self.bfs_hits += 1
            else:
                self._mgu_bfs_path = []; self.bfs_fails += 1; return None
        if len(self._mgu_bfs_path) == 1:
            return None
        return path_to_action(self._mgu_bfs_path)

    def _mgu_advance_dr(self, action):
        if self._mgu_dr_pos is None: return
        dx = [0, 0, -STEP, STEP][action]
        dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._mgu_dr_pos[0] + dx, self._mgu_dr_pos[1] + dy
        if (nx in GRID_XS_SET and ny in GRID_YS_SET
                and self._mgu_wall_set is not None
                and (nx, ny) not in self._mgu_wall_set):
            if self._mgu_bfs_path and len(self._mgu_bfs_path) > 1:
                self._mgu_bfs_path = self._mgu_bfs_path[1:]
            self._mgu_dr_pos = (nx, ny)

    def _puq_dr_action(self):
        if self._puq_wall_set is None or self._puq_dr_pos is None:
            return None
        start = self._puq_dr_pos
        if self._puq_phase == 'gic':
            goal = PUQ_GIC_GRID
        elif self._puq_phase == 'kdy':
            goal = PUQ_KDY_GRID
        else:
            goal = PUQ_LHS_GRID
        combined_walls = self._puq_wall_set | self._puq_avoid_set

        if not self._puq_bfs_path or self._puq_bfs_path[0] != start or self._puq_bfs_path[-1] != goal:
            path = bfs_path(start, goal, combined_walls)
            if path:
                self._puq_bfs_path = path; self.puq_bfs_hits += 1
            else:
                self._puq_bfs_path = []; self.puq_bfs_fails += 1; return None

        # Check if arrived at GIC_GRID for gic phase
        if self._puq_phase == 'gic' and len(self._puq_bfs_path) == 1:
            # Arrived at GIC - touch happened
            self._puq_gic_done = True
            self._puq_avoid_set.add(PUQ_GIC_GRID)
            self._puq_phase = 'kdy'
            self._puq_bfs_path = []
            return None  # Take random step this turn

        if self._puq_phase == 'lhs':
            self._puq_lhs_ap += 1

        if len(self._puq_bfs_path) == 1:
            return None
        return path_to_action(self._puq_bfs_path)

    def _puq_advance_dr(self, action):
        if self._puq_dr_pos is None: return
        dx = [0, 0, -STEP, STEP][action]
        dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._puq_dr_pos[0] + dx, self._puq_dr_pos[1] + dy
        combined_walls = (self._puq_wall_set or set()) | self._puq_avoid_set
        if (nx in GRID_XS_SET and ny in GRID_YS_SET
                and (nx, ny) not in combined_walls):
            if self._puq_bfs_path and len(self._puq_bfs_path) > 1:
                self._puq_bfs_path = self._puq_bfs_path[1:]
            self._puq_dr_pos = (nx, ny)

    def act(self):
        if self.game_level == 0 and (self._steps_since_detect >= 500
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self._steps_since_detect = 0

        if self.game_level == 1:
            action = self._mgu_dr_action()
            if action is not None:
                self._mgu_advance_dr(action)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action

        elif self.game_level == 2:
            action = self._puq_dr_action()
            if action is not None:
                self._puq_advance_dr(action)
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
        self._pn = self._cn; self._pa = action; return action


def t0():
    """Verify BFS paths exist for both mgu and puq positions."""
    # mgu walls (from T0 in 572j)
    mgu_walls = set([
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
    ])
    path = bfs_path(MGU_SPAWN, MGU_LHS_GRID, mgu_walls)
    assert len(path) == 18, f"mgu lhs path: {len(path)}"
    path2 = bfs_path(MGU_SPAWN, MGU_KDY_GRID, mgu_walls)
    assert len(path2) == 18, f"mgu kdy path: {len(path2)}"

    # puq walls (from ls20.py sprite positions)
    puq_walls = set([
        (4,0),(9,0),(4,5),(14,0),(19,0),(24,0),(29,0),(34,0),(39,0),(44,0),(49,0),(54,0),(59,0),
        (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),
        (59,5),(59,10),(59,15),(59,20),(59,25),(59,30),(59,35),(59,40),(59,45),(59,50),(59,55),
        (54,55),(49,55),(44,55),(39,55),(34,55),(29,55),(24,55),(19,55),
        (4,40),(4,45),(4,50),(9,50),(4,55),(9,55),(14,55),
        (39,10),(14,25),(19,40),(19,45),(19,35),(49,50),
        (39,35),(39,40),(39,45),(14,30),(49,45),(49,40),(14,20),(14,50),
        (39,5),(39,50),(44,45),(19,50),(44,40),(44,50),(44,20),(49,20),(39,20),
        (19,10),(14,35),(39,15),(34,35),(14,10),(14,15),(44,35),(24,35),(34,10),(24,10),
    ])
    # Check key positions not in walls
    for pos, name in [(PUQ_SPAWN,'SPAWN'),(PUQ_GIC_GRID,'GIC'),(PUQ_KDY_GRID,'KDY'),(PUQ_LHS_GRID,'LHS')]:
        assert pos not in puq_walls, f"puq {name}={pos} is a wall!"
    # Check BFS paths exist
    p1 = bfs_path(PUQ_SPAWN, PUQ_GIC_GRID, puq_walls)
    assert len(p1) > 0, f"puq SPAWN->GIC: no path"
    p2 = bfs_path(PUQ_GIC_GRID, PUQ_KDY_GRID, puq_walls | {PUQ_GIC_GRID})
    assert len(p2) > 0, f"puq GIC->KDY: no path (GIC in avoid)"
    p3 = bfs_path(PUQ_KDY_GRID, PUQ_LHS_GRID, puq_walls | {PUQ_GIC_GRID, PUQ_KDY_GRID})
    assert len(p3) > 0, f"puq KDY->LHS: no path (GIC,KDY in avoid)"
    print(f"T0 PASS: mgu lhs={len(path)} kdy={len(path2)} | puq spawn-gic={len(p1)} gic-kdy={len(p2)} kdy-lhs={len(p3)}")


def main():
    t0()
    try:
        sys.path.insert(0, '.'); import arcagi3; mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 5; per_seed_cap = 60; R = []; t_start = time.time()

    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        env = mk(); sub = SubDual(seed=seed * 1000)
        obs = env.reset(seed=seed)
        l1_step = None; l2_first = None; l3_steps = []
        go = 0; seed_start = time.time()
        prev_cl = 0

        for step in range(1, 200_001):
            if obs is None:
                obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; continue
            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)

            # FIX: skip cl processing for terminal frame — prevents prev_cl poisoning
            if done:
                go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0
                continue  # terminal frame cl must not update prev_cl

            cl = info.get('level', 0) if isinstance(info, dict) else 0
            if cl > prev_cl:
                if cl == 1 and prev_cl == 0:   # krg->mgu
                    sub.on_l1()
                    if l1_step is None:
                        l1_step = step
                        print(f"  s{seed} L1@{step} cyc={sub.l1_cycles} go={go}", flush=True)
                elif cl == 2 and prev_cl == 1:  # mgu->puq (L2)
                    sub.on_l2()
                    if l2_first is None:
                        l2_first = step
                        print(f"  s{seed} L2@{step} l2c={sub.l2_count} go={go}", flush=True)
                elif cl == 3 and prev_cl == 2:  # puq->tmx (L3!)
                    sub.on_l3()
                    l3_steps.append(step)
                    print(f"  s{seed} L3@{step}! l3c={sub.l3_count} "
                          f"puq_phase={sub._puq_phase} go={go}", flush=True)
            prev_cl = cl

            if step % 10_000 == 0:
                el = time.time() - seed_start
                print(f"  s{seed} @{step} cyc={sub.l1_cycles} l2c={sub.l2_count} l3c={sub.l3_count} "
                      f"l3f={sub.l3_frames} puq_ph={sub._puq_phase} "
                      f"puq_bfs+={sub.puq_bfs_hits} puq_bfs-={sub.puq_bfs_fails} "
                      f"puq_lhs={sub._puq_lhs_ap} wall={sub._puq_wall_set is not None} "
                      f"go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        R.append(dict(seed=seed, l1=l1_step, l2=l2_first, l3_count=sub.l3_count,
                      l2_count=sub.l2_count, go=go, steps=step,
                      cycles=sub.l1_cycles, l3_frames=sub.l3_frames,
                      puq_hits=sub.puq_bfs_hits, puq_fails=sub.puq_bfs_fails,
                      puq_lhs=sub._puq_lhs_ap, ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = f"L3x{r['l3_count']}" if r['l3_count'] > 0 else (f"L2x{r['l2_count']}" if r['l2'] else ("L1" if r['l1'] else "---"))
        print(f"  s{r['seed']}: {tag}  cyc={r['cycles']}  l2c={r['l2_count']}  l3c={r['l3_count']}  "
              f"l3f={r['l3_frames']}  puq_bfs+={r['puq_hits']}  puq_lhs={r['puq_lhs']}  "
              f"go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1'])
    l2n = sum(1 for r in R if r['l2'])
    l3n = sum(1 for r in R if r['l3_count'] > 0)
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}  L3={l3n}/{len(R)}")
    if l3n > 0:
        total_l3 = sum(r['l3_count'] for r in R)
        print(f"\nFIND: L3={l3n}/{len(R)}! total_l3={total_l3}. puq navigation confirmed!")
    elif l2n > 0:
        total_puq_hits = sum(r['puq_hits'] for r in R)
        total_puq_lhs = sum(r['puq_lhs'] for r in R)
        total_l2c = sum(r['l2_count'] for r in R)
        if total_puq_hits == 0:
            print(f"\nKILL: L3=0. puq_bfs+=0 (total_l2c={total_l2c}, need {N_MAP_PUQ}). "
                  f"Wall set not built or BFS fails after building.")
        elif total_puq_lhs == 0:
            print(f"\nKILL: L3=0. puq_bfs+={total_puq_hits} but puq_lhs=0 - "
                  f"BFS navigates but never reaches lhs phase. "
                  f"Check gic detection or tuv counting.")
        else:
            print(f"\nKILL: L3=0. Reaches lhs approach ({total_puq_lhs}) but win not triggered. "
                  f"Check state ordering or avoid_set.")
    else:
        print(f"\nKILL: L3=0 and L2=0 - regression. Check on_l2 handler.")


if __name__ == "__main__":
    main()
