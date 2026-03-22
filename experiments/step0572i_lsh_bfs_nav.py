"""
Step 572i — BFS navigation for L2.

572h: esc=0 — stuck detection never triggered. Wall-bounce std > STUCK_THRESH.
572g: confirmed navigation failure — lhs at (14,40) unreachable with greedy.

Root cause (from ls20.py source):
  - Player moves 5 pixels/step. Grid: x∈{4,9,14,19,...,59}, y∈{0,5,10,...,55}
  - Spawn at (29,40). lhs at (14,40). dy=0, dx=-15 → greedy tries left
  - Wall at (24,40) blocks first step left. Agent stuck immediately.
  - Real path: (29,40)→(34,40)→(34,35)→...→(14,10)→...→(14,40). 18+ moves.
  - Wall check: tag "jdd" = nlo color 4 = large regions in mode map

FIX: BFS pathfinding in L2
  - Extract wall cells: mode_arr 5×5 grid cells containing color 4
  - BFS from current agent grid cell to nearest target grid cell
  - Return first action on BFS path
  - Recompute BFS every REPLAN_EVERY steps or when target changes
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
VISIT_DIST = 6      # increased from 4 to handle 5-pixel grid alignment
N_MAP = 30
REPLAN_EVERY = 50   # replan BFS path every N steps in L2

# Grid parameters from ls20.py
GRID_XS = list(range(4, 64, 5))   # [4,9,14,...,59]
GRID_YS = list(range(0, 64, 5))   # [0,5,10,...,55]
STEP = 5


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


def is_wall_cell(mode_arr, gx, gy):
    """Return True if the 5×5 grid cell at (gx,gy) contains any color-4 pixels."""
    region = mode_arr[gy:gy+5, gx:gx+5]
    return bool((region == 4).any())


def build_wall_set(mode_arr):
    """Build set of (gx, gy) grid positions that are walls (color 4)."""
    walls = set()
    for gx in GRID_XS:
        for gy in GRID_YS:
            if is_wall_cell(mode_arr, gx, gy):
                walls.add((gx, gy))
    return walls


def nearest_grid(cx, cy):
    """Map continuous pixel position to nearest grid cell."""
    gx = min(GRID_XS, key=lambda v: abs(v - cx))
    gy = min(GRID_YS, key=lambda v: abs(v - cy))
    return gx, gy


def bfs_path(start, goal, walls):
    """BFS from start to goal. Returns list of (gx,gy) steps or []."""
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    # Directions: (dgx, dgy) and corresponding action (0=up,1=down,2=left,3=right)
    # up = gy decreases, down = gy increases, left = gx decreases, right = gx increases
    dirs = [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in dirs:
            nx, ny = cx + dx, cy + dy
            if nx not in GRID_XS or ny not in GRID_YS:
                continue
            if (nx, ny) in visited or (nx, ny) in walls:
                continue
            new_path = path + [(nx, ny)]
            if (nx, ny) == goal:
                return new_path
            visited.add((nx, ny))
            queue.append(((nx, ny), new_path))
    return []  # unreachable


def path_to_action(path):
    """Return action for first step in path."""
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
        self.l1_targets = []; self.l2_targets = []
        self.visited = []; self.agent_yx = None; self.prev_arr = None
        self._steps_since_detect = 99999
        self.target_actions = 0; self.fb_actions = 0
        self.n_l1_tgt = 0; self.n_l2_tgt = 0
        # BFS state
        self._bfs_path = []
        self._bfs_steps = 0
        self._wall_set = None
        # Diagnostics
        self.lhs_visits = 0
        self.kdy_touches = 0
        self.bfs_hits = 0
        self.bfs_fails = 0

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

        # Diagnostics in L2
        if self.game_level == 1 and self.agent_yx is not None:
            ay, ax = self.agent_yx
            for t in self.l2_targets:
                if t.get('color') == 5:
                    if ((ay-t['cy'])**2+(ax-t['cx'])**2)**0.5 < VISIT_DIST:
                        self.lhs_visits += 1
                        break
            for t in self.l2_targets:
                if t.get('color') == 0:
                    if ((ay-t['cy'])**2+(ax-t['cx'])**2)**0.5 < VISIT_DIST:
                        self.kdy_touches += 1
                        break
        return n

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []; self._steps_since_detect = 99999
        self._bfs_path = []; self._bfs_steps = 0
        if self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self.l2_targets = find_isolated_clusters(self.l2_mode)
            self.n_l2_tgt = len(self.l2_targets)
            # Build wall set from l2 mode map
            self._wall_set = build_wall_set(self.l2_mode)

    def on_reset(self):
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None
        self.visited = []; self._steps_since_detect = 99999
        self._bfs_path = []; self._bfs_steps = 0
        self._pn = None

    def _bfs_action(self):
        """Compute BFS path to nearest target and return next action."""
        if self._wall_set is None or self.agent_yx is None or not self.l2_targets:
            return None
        ay, ax = self.agent_yx
        start = nearest_grid(ax, ay)
        # Find nearest target by grid distance
        best = None; best_dist = 1e9
        for t in self.l2_targets:
            tg = nearest_grid(t['cx'], t['cy'])
            d = abs(tg[0] - start[0]) + abs(tg[1] - start[1])
            if d < best_dist:
                best_dist = d; best = tg
        if best is None:
            return None
        # Replan if needed
        if (self._bfs_steps >= REPLAN_EVERY or not self._bfs_path
                or nearest_grid(ax, ay) != self._bfs_path[0]):
            path = bfs_path(start, best, self._wall_set)
            self._bfs_steps = 0
            if path:
                self._bfs_path = path
                self.bfs_hits += 1
            else:
                self._bfs_path = []
                self.bfs_fails += 1
        self._bfs_steps += 1
        if self._bfs_path and len(self._bfs_path) > 1:
            # Advance path if we moved
            if nearest_grid(ax, ay) == self._bfs_path[0] and len(self._bfs_path) > 1:
                action = path_to_action(self._bfs_path)
                if action is not None:
                    return action
            elif nearest_grid(ax, ay) != self._bfs_path[0]:
                # Off path, replan next step
                self._bfs_steps = REPLAN_EVERY
        return None

    def act(self):
        if (self.game_level == 0 and self._steps_since_detect >= 500
                and self.l1_frames >= WARMUP):
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self.n_l1_tgt = len(self.l1_targets)
            self._steps_since_detect = 0

        if self.game_level == 1:
            # Use BFS navigation in L2
            action = self._bfs_action()
            if action is not None:
                self._pn = self._cn; self._pa = action
                self.target_actions += 1; return action
        elif self.l1_targets and self.agent_yx is not None:
            # L1: greedy dir_action (works fine in krg)
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
    """Verify BFS finds path around a simple wall."""
    walls = {(19, 40)}  # single wall blocking direct path
    path = bfs_path((29, 40), (14, 40), walls)
    assert len(path) > 0, "Should find path around wall"
    assert path[0] == (29, 40), f"Path should start at (29,40): {path[0]}"
    assert path[-1] == (14, 40), f"Path should end at (14,40): {path[-1]}"
    # Direct path blocked, must go around
    steps = [(path[i], path[i+1]) for i in range(len(path)-1)]
    assert not any(a==(24,40) and b==(19,40) for a,b in steps), "Should not step through wall"
    # Verify nearest_grid
    gx, gy = nearest_grid(16.02, 42.02)
    assert gx == 14, f"lhs cx=16.02 → gx should be 14: {gx}"
    assert gy == 40, f"lhs cy=42.02 → gy should be 40: {gy}"
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
                      f"bfs_hits={sub.bfs_hits} bfs_fails={sub.bfs_fails} "
                      f"go={go} {el:.0f}s", flush=True)
            if time.time() - seed_start > per_seed_cap:
                print(f"  s{seed} cap @{step}", flush=True); break

        R.append(dict(seed=seed, l1=l1_step, l2=l2_step, go=go, steps=step,
                      cycles=sub.l1_cycles, l2_frames=sub.l2_frames,
                      n_l2_tgt=sub.n_l2_tgt, lhs_v=sub.lhs_visits,
                      kdy_t=sub.kdy_touches, bfs_hits=sub.bfs_hits,
                      bfs_fails=sub.bfs_fails, ta=sub.target_actions))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}\nElapsed: {elapsed:.0f}s")
    for r in R:
        tag = "L2" if r['l2'] else ("L1" if r['l1'] else "---")
        print(f"  s{r['seed']}: {tag}  cyc={r['cycles']}  lhs_v={r['lhs_v']}  "
              f"kdy_t={r['kdy_t']}  bfs+={r['bfs_hits']}  bfs-={r['bfs_fails']}  "
              f"go={r['go']}  ta={r['ta']}")
    l1n = sum(1 for r in R if r['l1']); l2n = sum(1 for r in R if r['l2'])
    print(f"\nL1={l1n}/{len(R)}  L2={l2n}/{len(R)}")
    if l2n > 0:
        print(f"\nFIND: L2={l2n}/{len(R)}! BFS navigation works!")
    else:
        lhs_total = sum(r['lhs_v'] for r in R)
        kdy_total = sum(r['kdy_t'] for r in R)
        hits = sum(r['bfs_hits'] for r in R)
        fails = sum(r['bfs_fails'] for r in R)
        if fails > hits:
            print(f"\nKILL: L2=0. bfs_fails={fails} > bfs_hits={hits} — mode map walls wrong.")
        elif lhs_total == 0:
            print(f"\nKILL: L2=0. lhs_v=0, bfs_hits={hits} — BFS finds paths but wrong targets.")
        else:
            print(f"\nKILL: L2=0. lhs_v={lhs_total}, kdy_t={kdy_total} — "
                  f"lhs reached but state wrong (need kdy_t≥3 first).")


if __name__ == "__main__":
    main()
