"""
572o: Minimal probe. Reach first L2, then print raw info['level'] for 200 steps.
Uses full SubDual from 572m to ensure L2 is reached.
"""
import numpy as np
import sys
import time
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
GRID_XS = list(range(4, 64, 5))
GRID_YS = list(range(0, 64, 5))
GRID_XS_SET = set(GRID_XS)
GRID_YS_SET = set(GRID_YS)
STEP = 5
MGU_SPAWN = (29, 40)
MGU_LHS_GRID = (14, 40)
MGU_KDY_GRID = (49, 45)
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
        self.game_level = 0
        self.l1_cycles = 0
        self.l2_count = 0
        self.l1_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = 99999
        self.target_actions = 0
        self._mgu_tuv_est = 0
        self._mgu_phase = 'kdy'
        self._mgu_dr_pos = None
        self._mgu_wall_set = None
        self._mgu_wall_frozen = False
        self._mgu_bfs_path = []

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
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
            kdj_changed = int(diff[KDJ_R0:KDJ_R1, KDJ_C0:KDJ_C1].sum())
            if kdj_changed >= KDJ_THRESH:
                if self.game_level == 1:
                    self._mgu_tuv_est = (self._mgu_tuv_est + 1) % 4
                    if self._mgu_tuv_est == MGU_TUV_NEEDED:
                        self._mgu_phase = 'lhs'
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
            pass
        elif self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self._mgu_wall_set = build_wall_set(self.l2_mode)

    def on_l2(self):
        self.game_level = 2; self.l2_count += 1
        if not self._mgu_wall_frozen and self._mgu_wall_set is not None:
            self._mgu_wall_frozen = True

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []
        self._steps_since_detect = 99999
        self._pn = None
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'
        self._mgu_bfs_path = []

    def _mgu_dr_action(self):
        if self._mgu_wall_set is None or self._mgu_dr_pos is None:
            return None
        start = self._mgu_dr_pos
        goal = MGU_KDY_GRID if self._mgu_phase == 'kdy' else MGU_LHS_GRID
        if not self._mgu_bfs_path or self._mgu_bfs_path[0] != start or self._mgu_bfs_path[-1] != goal:
            path = bfs_path(start, goal, self._mgu_wall_set)
            if path: self._mgu_bfs_path = path
            else: self._mgu_bfs_path = []; return None
        if len(self._mgu_bfs_path) == 1: return None
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

    def act(self):
        if self.game_level == 0 and self._steps_since_detect >= 500 and self.l1_frames >= WARMUP:
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self._steps_since_detect = 0
        if self.game_level == 1:
            action = self._mgu_dr_action()
            if action is not None:
                self._mgu_advance_dr(action)
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action
        elif self.l1_targets and self.agent_yx is not None:
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
        sys.path.insert(0, '.'); import arcagi3; mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    seed = 0
    env = mk(); sub = SubDual(seed=0)
    obs = env.reset(seed=seed)
    go = 0; prev_cl = 0
    first_l2_step = None
    phase = "pre_l2"
    probe_step = 0

    for step in range(1, 200_001):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0; continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        raw_cl = info.get('level', -1) if isinstance(info, dict) else -1

        if phase == "probe":
            probe_step += 1
            print(f"  probe[{probe_step:3d}] cl={raw_cl} done={done}", flush=True)
            if done:
                print(f"DONE in probe at step {probe_step}. Game ended. Starting next game.", flush=True)
                go += 1
                obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0
                phase = "probe2"
                probe_step = 0
                continue
            if probe_step >= 200:
                print(f"200 probe steps done. raw_cl range seen.", flush=True)
                break
            continue

        if phase == "probe2":
            probe_step += 1
            print(f"  probe2[{probe_step:3d}] cl={raw_cl} done={done}", flush=True)
            if done or probe_step >= 200:
                print(f"probe2 done at step {probe_step}.", flush=True)
                break
            continue

        if done:
            go += 1
            obs = env.reset(seed=seed); sub.on_reset(); prev_cl = 0
            if phase == "post_l2":
                # Start probe of new game
                print(f"\nRESET after puq episode (go={go}). Starting probe.", flush=True)
                phase = "probe"
                probe_step = 0
            continue

        cl = raw_cl if raw_cl >= 0 else 0
        if cl > prev_cl:
            if cl == 1 and prev_cl == 0:
                sub.on_l1()
                print(f"  L1@{step} cyc={sub.l1_cycles}", flush=True)
            elif cl == 2 and prev_cl == 1:
                sub.on_l2()
                print(f"  L2@{step} l2c={sub.l2_count}. Entering puq episode.", flush=True)
                first_l2_step = step
                phase = "post_l2"
        prev_cl = cl

        if step % 1000 == 0 and phase == "pre_l2":
            print(f"  @{step} go={go} cyc={sub.l1_cycles}", flush=True)


if __name__ == "__main__":
    main()
