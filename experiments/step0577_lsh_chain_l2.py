"""
Step 577 -- Chain benchmark with L2: CIFAR-100 -> LS20 (L2) -> CIFAR-100.

L2-capable substrate: LSH (256D avgpool16, per-domain centering) + BFS waypoint
navigation for L2 (MGU waypoints from step 572u).

Chain protocol (step 508 with L2-capable substrate):
  Phase 1: CIFAR-100 1-pass (10K images) -- measure acc
  Phase 2: LS20 -- navigate to L2 (mgu completion) or 10-min cap
  Phase 3: CIFAR-100 1-pass again -- measure acc delta

Per-domain centering: mu resets on domain switch and on each LS20 death.
CIFAR and LS20 use SEPARATE edge dicts (100 vs 4 actions), SHARED hash space.

Signals:
  CIFAR acc delta <2pp = stable (no contamination)
  L2 reached yes/no, step count
  L1 in chain vs clean: compare

15-min total cap. 1 seed.

Step 546 baseline (per-domain centering, L1 only): 2/3 L1 on chain.
"""
import time
import numpy as np
import sys
from scipy.ndimage import label as ndlabel
from collections import deque

# ── constants ─────────────────────────────────────────────────────────────────

K = 16
DIM = 256
N_A = 4
N_CIFAR = 100

# Mode map
MODE_EVERY = 200
WARMUP = 100
MIN_CLUSTER = 2
MAX_CLUSTER = 60
VISIT_DIST = 4
N_MAP = 30       # L1 cycles before L2 wall map builds

# BFS grid (step 5 grid over 64x64)
STEP = 5
GRID_XS = list(range(4, 64, STEP))
GRID_YS = list(range(0, 64, STEP))
GRID_XS_SET = set(GRID_XS)
GRID_YS_SET = set(GRID_YS)

# MGU waypoints (level 2, from 572u)
MGU_SPAWN = (29, 40)
MGU_LHS_GRID = (14, 40)
MGU_KDY_GRID = (49, 45)
KDJ_R0, KDJ_R1 = 55, 61
KDJ_C0, KDJ_C1 = 3, 9
KDJ_THRESH = 5
MGU_TUV_NEEDED = 3


# ── encoding ──────────────────────────────────────────────────────────────────

def enc_ls20(frame):
    """LS20 frame[0] 64x64 [0-15] -> avgpool4 -> 16x16 = 256D (no centering)."""
    a = np.array(frame[0], dtype=np.float32) / 15.0
    return a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()


def enc_cifar(img):
    """CIFAR (32,32,3) uint8 -> grayscale -> avgpool2 -> 16x16 = 256D (no centering)."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = img.mean(axis=2).astype(np.float32) / 255.0
    return gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()


# ── helpers ───────────────────────────────────────────────────────────────────

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
            if (mode_arr[gy:gy + STEP, gx:gx + STEP] == 4).any():
                walls.add((gx, gy))
    return walls


def bfs_path(start, goal, walls):
    if start == goal:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    for dx, dy in [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]:
        pass  # silence unused warning
    while queue:
        (cx, cy), path = queue.popleft()
        for dx, dy in [(0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0)]:
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
    return 3


def dir_action(ty, tx, ay, ax):
    dy, dx = ty - ay, tx - ax
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    return 2 if dx < 0 else 3


# ── substrate ─────────────────────────────────────────────────────────────────

class SubChain577:
    """LSH 256D (per-domain centering) + BFS L2 navigation (MGU waypoints)."""

    def __init__(self, seed=0):
        self.H = np.random.RandomState(seed).randn(K, DIM).astype(np.float32)
        self.G = {}        # LS20 edge dict: (node, action 0-3) -> {next: count}
        self.G_cif = {}    # CIFAR edge dict: (node, action 0-99) -> {next: count}
        self.live = set()
        self._pn = self._pa = self._cn = None
        self.t = 0
        # Per-domain running mean (reset on domain switch and death)
        self._mu = np.zeros(DIM, dtype=np.float32)
        self._mu_n = 0
        # Mode maps for L1 (find sprites) and L2 (wall detection)
        self.l1_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l1_mode = np.zeros((64, 64), dtype=np.int32)
        self.l1_frames = 0
        self.l2_freq = np.zeros((64, 64, 16), dtype=np.int32)
        self.l2_mode = np.zeros((64, 64), dtype=np.int32)
        self.l2_frames = 0
        # Agent tracking
        self.agent_yx = None
        self.prev_arr = None
        self.game_level = 0
        self.l1_targets = []
        self.visited = []
        self._steps_since_detect = 99999
        self.target_actions = 0
        # L2 / MGU state
        self.l1_cycles = 0
        self.l2_count = 0
        self._mgu_tuv_est = 0
        self._mgu_phase = 'kdy'
        self._mgu_dr_pos = None
        self._mgu_wall_set = None
        self._mgu_wall_frozen = False
        self._mgu_bfs_path = []
        self.bfs_hits = self.bfs_fails = 0

    def _node(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8), bitorder='big').tobytes().hex(), 16)

    def _center_update(self, x_raw):
        """Per-domain centering: subtract running mean, then update it."""
        x = x_raw - self._mu
        self._mu_n += 1
        self._mu += (x_raw - self._mu) / self._mu_n
        return x

    def observe(self, frame):
        """LS20 observe."""
        arr = np.array(frame[0], dtype=np.int32)
        r = np.arange(64)[:, None]; c = np.arange(64)[None, :]
        if self.game_level == 0:
            self.l1_freq[r, c, arr] += 1; self.l1_frames += 1
            if self.l1_frames % MODE_EVERY == 0:
                self.l1_mode = np.argmax(self.l1_freq, axis=2).astype(np.int32)
        else:
            self.l2_freq[r, c, arr] += 1; self.l2_frames += 1
            if self.l2_frames % MODE_EVERY == 0:
                self.l2_mode = np.argmax(self.l2_freq, axis=2).astype(np.int32)
        if self.prev_arr is not None:
            diff = np.abs(arr - self.prev_arr) > 0
            nc = int(diff.sum())
            if 1 <= nc < 200:
                ys, xs = np.where(diff)
                self.agent_yx = (float(ys.mean()), float(xs.mean()))
            if self.game_level == 1:
                kdj_changed = int(diff[KDJ_R0:KDJ_R1, KDJ_C0:KDJ_C1].sum())
                if kdj_changed >= KDJ_THRESH:
                    self._mgu_tuv_est = (self._mgu_tuv_est + 1) % 4
                    if self._mgu_tuv_est == MGU_TUV_NEEDED:
                        self._mgu_phase = 'lhs'
        self.prev_arr = arr.copy()
        x = self._center_update(enc_ls20(frame))
        n = self._node(x); self.live.add(n); self.t += 1
        if self._pn is not None:
            d = self.G.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n; self._steps_since_detect += 1

    def observe_cifar(self, img):
        """CIFAR observe: per-domain centering, separate edge dict."""
        x = self._center_update(enc_cifar(img))
        n = self._node(x)
        if self._pn is not None:
            d = self.G_cif.setdefault((self._pn, self._pa), {})
            d[n] = d.get(n, 0) + 1
        self._cn = n

    def act(self):
        """LS20 action."""
        if self.game_level == 0 and self._steps_since_detect >= 500 and self.l1_frames >= WARMUP:
            self.l1_targets = find_isolated_clusters(self.l1_mode)
            self._steps_since_detect = 0
        if self.game_level == 1:
            a = self._mgu_dr_action()
            if a is not None:
                self._mgu_advance_dr(a)
                self._pn = self._cn; self._pa = a
                self.target_actions += 1; return a
        if self.l1_targets and self.agent_yx is not None:
            ay, ax = self.agent_yx
            best = None; best_d = 1e9
            for t in self.l1_targets:
                if any(((t['cy']-vy)**2 + (t['cx']-vx)**2) < VISIT_DIST**2
                       for vy, vx in self.visited):
                    continue
                d = ((t['cy']-ay)**2 + (t['cx']-ax)**2)**0.5
                if d < best_d: best_d = d; best = t
            if best is not None:
                if best_d < VISIT_DIST:
                    self.visited.append((best['cy'], best['cx']))
                else:
                    a = dir_action(best['cy'], best['cx'], ay, ax)
                    self._pn = self._cn; self._pa = a
                    self.target_actions += 1; return a
        counts = [sum(self.G.get((self._cn, a), {}).values()) for a in range(N_A)]
        a = int(np.argmin(counts))
        self._pn = self._cn; self._pa = a; return a

    def act_cifar(self):
        """CIFAR: argmin over 100 actions from G_cif."""
        counts = [sum(self.G_cif.get((self._cn, a), {}).values()) for a in range(N_CIFAR)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        a = cands[int(np.random.randint(len(cands)))]
        self._pn = self._cn; self._pa = a; return a

    def _mgu_dr_action(self):
        if self._mgu_wall_set is None or self._mgu_dr_pos is None:
            return None
        goal = MGU_KDY_GRID if self._mgu_phase == 'kdy' else MGU_LHS_GRID
        start = self._mgu_dr_pos
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
        if self._mgu_dr_pos is None:
            return
        dx = [0, 0, -STEP, STEP][action]
        dy = [-STEP, STEP, 0, 0][action]
        nx, ny = self._mgu_dr_pos[0] + dx, self._mgu_dr_pos[1] + dy
        if nx in GRID_XS_SET and ny in GRID_YS_SET:
            if self._mgu_bfs_path and len(self._mgu_bfs_path) > 1:
                self._mgu_bfs_path = self._mgu_bfs_path[1:]
            self._mgu_dr_pos = (nx, ny)

    def on_l1(self):
        self.game_level = 1; self.l1_cycles += 1
        self.visited = []
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'; self._mgu_bfs_path = []
        if not self._mgu_wall_frozen and self.l1_cycles >= N_MAP and self.l2_frames >= WARMUP:
            self._mgu_wall_set = build_wall_set(self.l2_mode)

    def on_l2(self):
        self.game_level = 2; self.l2_count += 1
        if not self._mgu_wall_frozen and self._mgu_wall_set is not None:
            self._mgu_wall_frozen = True

    def on_reset(self):
        """Called on LS20 death: reset per-domain mean (per-episode centering)."""
        self.game_level = 0
        self.prev_arr = None; self.agent_yx = None; self.visited = []
        self._steps_since_detect = 99999; self._pn = None
        self._mgu_dr_pos = MGU_SPAWN
        self._mgu_tuv_est = 0; self._mgu_phase = 'kdy'; self._mgu_bfs_path = []
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0

    def reset_domain(self):
        """Reset per-domain mean on domain switch (CIFAR<->LS20)."""
        self._mu = np.zeros(DIM, dtype=np.float32); self._mu_n = 0
        self._pn = self._pa = None


# ── tests ─────────────────────────────────────────────────────────────────────

def t0():
    sub = SubChain577(seed=0)
    # CIFAR encode test
    img = np.random.RandomState(1).randint(0, 256, (32, 32, 3), dtype=np.uint8)
    sub.observe_cifar(img)
    a = sub.act_cifar()
    assert 0 <= a < 100
    # LS20 encode test
    frame = [np.random.RandomState(2).randint(0, 16, (64, 64), dtype=np.uint8)]
    sub.observe(frame)
    a = sub.act()
    assert 0 <= a < 4
    # Domain separation: reset_domain changes mean
    mu1 = sub._mu.copy()
    sub.reset_domain()
    assert sub._mu_n == 0
    print("T0 PASS")


# ── phases ────────────────────────────────────────────────────────────────────

def run_cifar(sub, X, y, label):
    sub.reset_domain()
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        sub.observe_cifar(X[i])
        a = sub.act_cifar()
        if a == int(y[i]):
            correct += 1
    acc = correct / len(X) * 100
    elapsed = time.time() - t0
    print(f"  {label}: acc={acc:.2f}%  cif_edges={len(sub.G_cif)}  {elapsed:.0f}s", flush=True)
    return acc


def run_ls20_l2(sub, mk, seed=0, time_cap=600):
    sub.reset_domain()
    env = mk()
    obs = env.reset(seed=seed)
    l1_step = l2_step = None
    go = 0; prev_cl = -1; t0 = time.time()

    for step in range(1, 1_000_001):
        if obs is None:
            obs = env.reset(seed=seed); sub.on_reset(); prev_cl = -1; go += 1; continue

        sub.observe(obs)
        action = sub.act()
        obs, reward, done, info = env.step(action)

        if done:
            go += 1; obs = env.reset(seed=seed); sub.on_reset(); prev_cl = -1; continue

        cl = info.get('level', 0) if isinstance(info, dict) else 0

        if prev_cl == -1:
            if cl >= 1:
                sub.on_l1()
                if l1_step is None:
                    l1_step = step
                    print(f"  L1@{step} cyc={sub.l1_cycles} go={go}", flush=True)
            prev_cl = cl
        elif cl > prev_cl:
            if cl == 1:
                sub.on_l1()
                if l1_step is None:
                    l1_step = step
                    print(f"  L1@{step} cyc={sub.l1_cycles} go={go}", flush=True)
            elif cl == 2:
                sub.on_l2()
                if l2_step is None:
                    l2_step = step
                    print(f"  L2@{step} l2c={sub.l2_count} wall={sub._mgu_wall_frozen} go={go}", flush=True)
                    break
            prev_cl = cl
        else:
            prev_cl = cl

        if time.time() - t0 > time_cap:
            print(f"  cap @{step} l1c={sub.l1_cycles} l2c={sub.l2_count} wall={sub._mgu_wall_frozen} "
                  f"bfs_f={sub.bfs_fails} go={go}", flush=True)
            break

    elapsed = time.time() - t0
    print(f"  LS20 done: L1={l1_step} L2={l2_step} live={len(sub.live)} go={go} {elapsed:.0f}s",
          flush=True)
    return l1_step, l2_step


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0()
    print("Step 577: Chain CIFAR-100 -> LS20 L2 -> CIFAR-100", flush=True)

    try:
        sys.path.insert(0, '.')
        import arcagi3
        mk = lambda: arcagi3.make("LS20")
    except Exception as e:
        print(f"arcagi3: {e}"); return

    try:
        import torchvision
        ds = torchvision.datasets.CIFAR100(
            './data/cifar100', train=False, download=True
        )
        X = np.array(ds.data); y = np.array(ds.targets)
        print(f"CIFAR-100: {len(X)} test images", flush=True)
    except Exception as e:
        print(f"CIFAR-100 load failed: {e}"); return

    np.random.seed(0)
    sub = SubChain577(seed=0)
    t_total = time.time()

    print("\n--- Phase 1: CIFAR-100 (1-pass) ---", flush=True)
    acc1 = run_cifar(sub, X, y, "P1")

    cifar_done = time.time() - t_total
    ls20_cap = max(60, 780 - cifar_done)  # leave 2 min for phase 3

    print(f"\n--- Phase 2: LS20 (L2 target, cap={ls20_cap:.0f}s) ---", flush=True)
    l1_step, l2_step = run_ls20_l2(sub, mk, seed=0, time_cap=ls20_cap)

    print("\n--- Phase 3: CIFAR-100 (1-pass) ---", flush=True)
    acc3 = run_cifar(sub, X, y, "P3")

    print(f"\n{'='*60}", flush=True)
    print("STEP 577 SUMMARY", flush=True)
    print(f"  CIFAR P1:  {acc1:.2f}%", flush=True)
    print(f"  LS20 L1:   {'@'+str(l1_step) if l1_step else 'FAIL'}", flush=True)
    print(f"  LS20 L2:   {'@'+str(l2_step) if l2_step else 'FAIL'}", flush=True)
    print(f"  CIFAR P3:  {acc3:.2f}%  (delta={acc3-acc1:+.2f}pp)", flush=True)
    print(f"  live cells: {len(sub.live)}  cif_edges: {len(sub.G_cif)}", flush=True)
    print(f"  Total elapsed: {time.time()-t_total:.0f}s", flush=True)

    if l2_step is not None and abs(acc3 - acc1) < 2:
        print("SUCCESS: L2 reached in chain, CIFAR acc stable.")
    elif l2_step is not None:
        print(f"PARTIAL: L2 reached but CIFAR delta={acc3-acc1:+.2f}pp (contamination?)")
    elif l1_step is not None:
        print(f"PARTIAL: L1 reached (L1@{l1_step}) but L2 not reached in time cap.")
    else:
        print("FAIL: No levels reached. Chain contamination kills L1 navigation.")


if __name__ == "__main__":
    main()
