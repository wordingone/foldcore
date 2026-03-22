"""
Step 612: LS20 L3 bootstrap fix (Option C from the 611 analysis).

Problem (confirmed by 611): At L2->L3 transition, l3_frames=0.
  _puq_wall_set requires l3_frames >= WARMUP (100), so it stays None.
  _puq_dr_action() returns None immediately — PUQ agent is blind.

Fix: Same bootstrap pattern as L1->L2.
  - At L2->L3: don't activate PUQ until N_MAP_L3 cycles of L3 have been seen
  - During bootstrap: use argmin (exploration) to accumulate l3_mode
  - After N_MAP_L3 cycles: build _puq_wall_set from l3_mode, activate BFS nav
  - This mirrors the mgu bootstrap: l1_cycles >= N_MAP AND l2_frames >= WARMUP

Changes from 606:
  - N_MAP_L3 = 30 (same as N_MAP and N_MAP_PUQ)
  - on_l2() condition: now ALSO checks l3_count >= N_MAP_L3 (not just frames)
  - New field: _puq_bootstrap_done (False until wall_set built for L3)
  - After each l2_count increment: check if bootstrap condition now met, build wall_set

5 seeds x 60s cap. Signal: L3 >= 3/5.
Kill: L3 = 0/5 after fix.
"""
import numpy as np
import time
import sys
from scipy.ndimage import label as ndlabel
from collections import deque

sys.path.insert(0, '.')

N_A = 4
K = 16
FG_DIM = 4096
MODE_EVERY = 200
WARMUP = 100
MIN_CLUSTER = 2
MAX_CLUSTER = 60
VISIT_DIST = 4
N_MAP = 30
N_MAP_PUQ = 30
N_MAP_L3 = 30   # NEW: cycles of L3 before building l3 wall set
DEATH_THRESHOLD = 1000

GRID_XS = list(range(4, 64, 5))
GRID_YS = list(range(0, 64, 5))
GRID_XS_SET = set(GRID_XS)
GRID_YS_SET = set(GRID_YS)
STEP = 5

MGU_SPAWN = (29, 40)
MGU_LHS_GRID = (14, 40)
MGU_KDY_GRID = (49, 45)

PUQ_SPAWN = (9, 45)
PUQ_BOOST1 = (34, 5)
PUQ_GIC_GRID = (29, 45)
PUQ_KDY_GRID = (49, 10)
PUQ_KDY_BOUNCE = (49, 5)
PUQ_LHS_GRID = (54, 50)

PUQ_WAYPOINTS = [
    PUQ_BOOST1, PUQ_GIC_GRID, PUQ_KDY_GRID, PUQ_KDY_BOUNCE,
    PUQ_KDY_GRID, PUQ_KDY_BOUNCE, PUQ_KDY_GRID, PUQ_LHS_GRID,
]
PUQ_WP_AVOID = {1: PUQ_GIC_GRID, 6: PUQ_KDY_GRID}

KDJ_R0, KDJ_R1 = 55, 61
KDJ_C0, KDJ_C1 = 3, 9
KDJ_THRESH = 5
MGU_TUV_NEEDED = 3


def find_isolated_clusters(mode_arr):
    clusters = []
    for color in range(16):
        mask = (mode_arr == color)
        if not mask.any(): continue
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
    if start == goal: return [start]
    queue = deque([(start, [start])]); visited = {start}
    dirs = [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if nx not in GRID_XS_SET or ny not in GRID_YS_SET: continue
            if (nx, ny) in visited or (nx, ny) in walls: continue
            new_path = path + [(nx, ny)]
            if (nx, ny) == goal: return new_path
            visited.add((nx, ny)); queue.append(((nx, ny), new_path))
    return []


def path_to_action(path):
    if len(path) < 2: return None
    cx, cy = path[0]; nx, ny = path[1]
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
        self.l4_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l4_mode = np.zeros((64, 64), dtype=np.int32)
        self.l4_frames = 0
        self.game_level = 0
        self.l1_cycles = 0; self.l2_count = 0; self.l3_count = 0; self.l4_count = 0
        self.l1_targets = []; self.l4_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = 99999
        self._steps_since_l4_detect = 99999
        self.target_actions = 0
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'
        self._mgu_dr_pos = None; self._mgu_wall_set = None
        self._mgu_wall_frozen = False; self._mgu_bfs_path = []
        self.bfs_hits = 0; self.bfs_fails = 0
        self._puq_wp_idx = 0; self._puq_dr_pos = None
        self._puq_wall_set = None; self._puq_wall_frozen = False
        self._puq_bfs_path = []; self._puq_avoid_set = set()
        self._puq_death_seen = False; self._puq_xhp_step = False
        self._puq_death_count = 0
        self.puq_bfs_hits = 0; self.puq_bfs_fails = 0
        self._puq_lhs_ap = 0
        # NEW: L3 bootstrap tracking
        self._puq_bootstrap_done = False  # True once l3 wall_set is built

    def _on_puq_death(self):
        self._puq_dr_pos = PUQ_SPAWN; self._puq_wp_idx = 0
        self._puq_bfs_path = []; self._puq_avoid_set = set()
        self._puq_xhp_step = True; self._puq_death_count += 1

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
        elif self.game_level == 2:
            self.l3_freq[r, c, arr] += 1; self.l3_frames += 1
            if self.l3_frames % MODE_EVERY == 0:
                self.l3_mode = np.argmax(self.l3_freq, axis=2).astype(np.int32)
                # NEW: check if bootstrap condition is now met mid-L3 run
                if (not self._puq_wall_frozen and not self._puq_bootstrap_done
                        and self.l2_count >= N_MAP_PUQ and self.l3_frames >= WARMUP):
                    self._puq_wall_set = build_wall_set(self.l3_mode)
                    self._puq_bootstrap_done = True
                    print(f"    [bootstrap] L3 wall_set built at l3_frames={self.l3_frames} "
                          f"l2_count={self.l2_count}", flush=True)
        else:
            self.l4_freq[r, c, arr] += 1; self.l4_frames += 1
            if self.l4_frames % MODE_EVERY == 0:
                self.l4_mode = np.argmax(self.l4_freq, axis=2).astype(np.int32)

    def _fg_enc(self, arr):
        if self.game_level == 0: mode = self.l1_mode
        elif self.game_level == 1: mode = self.l2_mode
        elif self.game_level == 2: mode = self.l3_mode
        else: mode = self.l4_mode
        return (arr != mode).astype(np.float32).flatten()

    def _base(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _node(self, x):
        n = self._base(x)
        while n in self.ref: n = (n, int(self.ref[n] @ x > 0))
        return n

    def observe(self, frame):
        arr = np.array(frame[0], dtype=np.int32) if hasattr(frame, '__len__') and hasattr(frame[0], '__len__') else np.array(frame, dtype=np.int32)
        if arr.ndim == 1: arr = arr.reshape(64, 64)
        elif arr.ndim == 3: arr = arr[:, :, 0] if arr.shape[2] < arr.shape[0] else arr[0]
        self._update_bg(arr)
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
            if self.game_level == 2:
                if nc > DEATH_THRESHOLD and not self._puq_death_seen:
                    self._on_puq_death(); self._puq_death_seen = True
                elif nc <= DEATH_THRESHOLD:
                    self._puq_death_seen = False
            kdj_changed = int(diff[KDJ_R0:KDJ_R1, KDJ_C0:KDJ_C1].sum())
            if kdj_changed >= KDJ_THRESH and self.game_level == 1:
                self._mgu_tuv_est = (self._mgu_tuv_est + 1) % 4
                if self._mgu_tuv_est == MGU_TUV_NEEDED:
                    self._mgu_phase = 'lhs'
        self.prev_arr = arr.copy()
        if self.game_level == 0: frames = self.l1_frames
        elif self.game_level == 1: frames = self.l2_frames
        elif self.game_level == 2: frames = self.l3_frames
        else: frames = self.l4_frames
        if frames < WARMUP:
            x = arr.astype(np.float32).flatten() / 15.0; x = x - x.mean()
        else:
            x = self._fg_enc(arr)
        n = self._node(x); self.live.add(n); self.t += 1
        self._last_visit[n] = self.t
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {}); d[n] = d.get(n, 0) + 1
        self._cn = n; self._steps_since_detect += 1; self._steps_since_l4_detect += 1
        return n

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []; self.G = {}; self._pn = None
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'; self._mgu_bfs_path = []
        if not self._mgu_wall_frozen and self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self._mgu_wall_set = build_wall_set(self.l2_mode)

    def on_l2(self):
        self.game_level = 2; self.l2_count += 1
        self.visited = []; self.G = {}; self._pn = None
        if not self._mgu_wall_frozen and self._mgu_wall_set is not None:
            self._mgu_wall_frozen = True
        self.agent_yx = None; self._steps_since_detect = 99999
        self._puq_dr_pos = PUQ_SPAWN; self._puq_wp_idx = 0
        self._puq_bfs_path = []; self._puq_avoid_set = set()
        self._puq_death_seen = False; self._puq_xhp_step = False
        # CHANGED: also check l2_count >= N_MAP_PUQ (was: l3_frames >= WARMUP only)
        # Now matches mgu bootstrap: needs enough L3 CYCLES, not just frames
        # Wall set built in _update_bg once both l2_count and l3_frames thresholds met
        if (not self._puq_wall_frozen and not self._puq_bootstrap_done
                and self.l2_count >= N_MAP_PUQ and self.l3_frames >= WARMUP):
            self._puq_wall_set = build_wall_set(self.l3_mode)
            self._puq_bootstrap_done = True
            print(f"    [on_l2] L3 wall_set built at l2_count={self.l2_count} "
                  f"l3_frames={self.l3_frames}", flush=True)
        if not self._puq_wall_frozen and self._puq_wall_set is not None:
            self._puq_wall_frozen = True

    def on_l3(self):
        self.game_level = 3; self.l3_count += 1
        self.visited = []; self.G = {}; self._pn = None
        if not self._puq_wall_frozen and self._puq_wall_set is not None:
            self._puq_wall_frozen = True
        self.agent_yx = None; self._steps_since_l4_detect = 99999
        self.l4_targets = []

    def on_l4(self):
        self.l4_count += 1; self.visited = []; self.G = {}; self._pn = None
        self.agent_yx = None

    def on_reset(self):
        self.game_level = 0; self.prev_arr = None; self.agent_yx = None
        self.visited = []; self._steps_since_detect = 99999
        self._steps_since_l4_detect = 99999; self._pn = None
        self._mgu_dr_pos = MGU_SPAWN; self._mgu_tuv_est = 0
        self._mgu_phase = 'kdy'; self._mgu_bfs_path = []

    def _mgu_dr_action(self):
        if self._mgu_wall_set is None or self._mgu_dr_pos is None: return None
        start = self._mgu_dr_pos
        goal = MGU_KDY_GRID if self._mgu_phase == 'kdy' else MGU_LHS_GRID
        if not self._mgu_bfs_path or self._mgu_bfs_path[0] != start or self._mgu_bfs_path[-1] != goal:
            path = bfs_path(start, goal, self._mgu_wall_set)
            if path: self._mgu_bfs_path = path; self.bfs_hits += 1
            else: self._mgu_bfs_path = []; self.bfs_fails += 1; return None
        if len(self._mgu_bfs_path) == 1: return None
        return path_to_action(self._mgu_bfs_path)

    def _mgu_advance_dr(self, action):
        if self._mgu_dr_pos is None: return
        dx = [0, 0, -STEP, STEP][action]; dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._mgu_dr_pos[0] + dx, self._mgu_dr_pos[1] + dy
        if (nx in GRID_XS_SET and ny in GRID_YS_SET
                and self._mgu_wall_set is not None
                and (nx, ny) not in self._mgu_wall_set):
            if self._mgu_bfs_path and len(self._mgu_bfs_path) > 1:
                self._mgu_bfs_path = self._mgu_bfs_path[1:]
            self._mgu_dr_pos = (nx, ny)

    def _puq_dr_action(self):
        if self._puq_wall_set is None or self._puq_dr_pos is None: return None
        if self._puq_wp_idx >= len(PUQ_WAYPOINTS): return None
        combined_walls = self._puq_wall_set | self._puq_avoid_set
        for _ in range(len(PUQ_WAYPOINTS) + 2):
            if self._puq_wp_idx >= len(PUQ_WAYPOINTS): return None
            start = self._puq_dr_pos; goal = PUQ_WAYPOINTS[self._puq_wp_idx]
            if not self._puq_bfs_path or self._puq_bfs_path[0] != start or self._puq_bfs_path[-1] != goal:
                path = bfs_path(start, goal, combined_walls)
                if path: self._puq_bfs_path = path; self.puq_bfs_hits += 1
                else: self._puq_bfs_path = []; self.puq_bfs_fails += 1; return None
            if len(self._puq_bfs_path) > 1: return path_to_action(self._puq_bfs_path)
            wp = self._puq_wp_idx
            if wp in PUQ_WP_AVOID:
                self._puq_avoid_set.add(PUQ_WP_AVOID[wp])
                combined_walls = self._puq_wall_set | self._puq_avoid_set
            if wp == len(PUQ_WAYPOINTS) - 1:
                self._puq_lhs_ap += 1; return None
            self._puq_wp_idx += 1; self._puq_bfs_path = []
        return None

    def _puq_advance_dr(self, action):
        if self._puq_dr_pos is None: return
        dx = [0, 0, -STEP, STEP][action]; dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._puq_dr_pos[0] + dx, self._puq_dr_pos[1] + dy
        combined_walls = (self._puq_wall_set or set()) | self._puq_avoid_set
        if (nx in GRID_XS_SET and ny in GRID_YS_SET and (nx, ny) not in combined_walls):
            if self._puq_bfs_path and len(self._puq_bfs_path) > 1:
                self._puq_bfs_path = self._puq_bfs_path[1:]
            self._puq_dr_pos = (nx, ny)

    def act(self):
        if self.game_level == 0 and self._steps_since_detect >= 500 and self.l1_frames >= WARMUP:
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self._steps_since_detect = 0
        if self.game_level == 3:
            if self._steps_since_l4_detect >= 500 and self.l4_frames >= WARMUP:
                self.l4_targets = find_isolated_clusters(self.l4_mode)
                self._steps_since_l4_detect = 0
            if self.l4_targets and self.agent_yx is not None:
                ay, ax = self.agent_yx
                best = None; best_dist = 1e9
                for t in self.l4_targets:
                    if any(((t['cy']-vy)**2+(t['cx']-vx)**2) < VISIT_DIST**2 for vy, vx in self.visited):
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
        if self.game_level == 1:
            action = self._mgu_dr_action()
            if action is not None:
                self._mgu_advance_dr(action)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action
        elif self.game_level == 2:
            if self._puq_xhp_step:
                self._puq_xhp_step = False; return 0
            action = self._puq_dr_action()
            if action is not None:
                self._puq_advance_dr(action)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action
        elif self.game_level == 0 and self.l1_targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_dist = 1e9
            for t in self.l1_targets:
                if any(((t['cy']-vy)**2+(t['cx']-vx)**2) < VISIT_DIST**2 for vy, vx in self.visited):
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


def main():
    try:
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    n_seeds = 5; per_seed_cap = 60
    t_start = time.time()
    print("Step 612: LS20 L3 bootstrap fix (Option C)", flush=True)
    print(f"  Fix: accumulate l3_mode for N_MAP_L3={N_MAP_L3} cycles before activating PUQ nav", flush=True)

    R = []
    for seed in range(n_seeds):
        print(f"\nseed {seed}:", flush=True)
        env = mk(); sub = SubDual(seed=seed * 1000)
        obs = env.reset(seed=seed)
        l1_step = [None]; l2_first = [None]; l3_steps = []; l4_steps = []
        go = 0; prev_cl = 0; t_seed = time.time()

        for step in range(1, 500_001):
            if obs is None:
                obs = env.reset(seed=seed); sub.on_reset(); prev_cl = -1; go += 1; continue

            sub.observe(obs)
            action = sub.act()
            obs, reward, done, info = env.step(action)

            if done:
                go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = -1; continue

            cl = info.get('level', 0) if isinstance(info, dict) else 0

            if prev_cl == -1:
                if cl == 1 and (l1_step[0] is None):
                    sub.on_l1(); l1_step[0] = step
                    print(f"  s{seed} L1@{step} cyc={sub.l1_cycles}", flush=True)
                elif cl == 2 and (l2_first[0] is None):
                    sub.on_l2(); l2_first[0] = step
                    print(f"  s{seed} L2@{step} l2c={sub.l2_count} bootstrap={sub._puq_bootstrap_done}", flush=True)
                elif cl == 3:
                    sub.on_l3(); l3_steps.append(step)
                    print(f"  s{seed} L3@{step}! l3c={sub.l3_count} deaths={sub._puq_death_count}", flush=True)
                elif cl == 4:
                    sub.on_l4(); l4_steps.append(step)
                    print(f"  s{seed} L4@{step}!! l4c={sub.l4_count}", flush=True)
                prev_cl = cl
            elif cl > prev_cl:
                if cl == 1:
                    sub.on_l1()
                    if l1_step[0] is None:
                        l1_step[0] = step
                        print(f"  s{seed} L1@{step} cyc={sub.l1_cycles}", flush=True)
                elif cl == 2:
                    sub.on_l2()
                    if l2_first[0] is None:
                        l2_first[0] = step
                        print(f"  s{seed} L2@{step} l2c={sub.l2_count} bootstrap={sub._puq_bootstrap_done}", flush=True)
                elif cl == 3:
                    sub.on_l3(); l3_steps.append(step)
                    print(f"  s{seed} L3@{step}! l3c={sub.l3_count} deaths={sub._puq_death_count}", flush=True)
                elif cl == 4:
                    sub.on_l4(); l4_steps.append(step)
                    print(f"  s{seed} L4@{step}!! l4c={sub.l4_count}", flush=True)
                prev_cl = cl
            else:
                prev_cl = cl

            if time.time() - t_seed > per_seed_cap:
                print(f"  cap @{step} go={go} l2c={sub.l2_count} l3c={sub.l3_count} "
                      f"l4c={sub.l4_count} bootstrap={sub._puq_bootstrap_done} "
                      f"wall={sub._puq_wall_frozen} deaths={sub._puq_death_count} "
                      f"wp={sub._puq_wp_idx} bfs_f={sub.puq_bfs_fails} lhs={sub._puq_lhs_ap}",
                      flush=True)
                break

        R.append({'seed': seed, 'l2c': sub.l2_count, 'l3c': sub.l3_count,
                  'l4c': sub.l4_count, 'l3_steps': l3_steps, 'l4_steps': l4_steps,
                  'puq_wall': sub._puq_wall_frozen, 'bootstrap': sub._puq_bootstrap_done,
                  'bfs_fails': sub.puq_bfs_fails, 'lhs_ap': sub._puq_lhs_ap,
                  'deaths': sub._puq_death_count, 'wp_idx': sub._puq_wp_idx})

    print(f"\n{'='*60}", flush=True)
    print(f"Step 612: LS20 L3 bootstrap fix", flush=True)
    l3_wins = sum(1 for r in R if r['l3c'] > 0)
    l4_wins = sum(1 for r in R if r['l4c'] > 0)
    print(f"  L3: {l3_wins}/{len(R)}  L4: {l4_wins}/{len(R)}", flush=True)
    for r in R:
        print(f"  s{r['seed']}: L2={r['l2c']} L3={r['l3c']} L4={r['l4c']} "
              f"boot={r['bootstrap']} wall={r['puq_wall']} "
              f"deaths={r['deaths']} bfs_f={r['bfs_fails']} "
              f"lhs={r['lhs_ap']} wp={r['wp_idx']} "
              f"L3@{r['l3_steps'][:3]} L4@{r['l4_steps'][:3]}", flush=True)
    print(f"  total_t={time.time()-t_start:.1f}s", flush=True)
    if l3_wins >= 3:
        print("  SIGNAL: L3 bootstrap fix works. PUQ nav active.", flush=True)
    elif l3_wins >= 1:
        print("  PARTIAL: L3 sometimes reached. Check bootstrap timing.", flush=True)
    else:
        print("  FAIL: L3 not reached. Bootstrap insufficient — check wall_set quality.", flush=True)


if __name__ == "__main__":
    main()
