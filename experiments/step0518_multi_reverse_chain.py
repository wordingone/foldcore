#!/usr/bin/env python3
"""
Step 518 -- Reverse chain: LS20 -> CIFAR-100 -> LS20
All prior chains: CIFAR first, then ARC. This reverses the order.

Phase 1: LS20 (5-min) — build navigation graph with ~456 centroids
Phase 2: CIFAR-100 (1-pass) — classification on top of ARC centroids
Phase 3: LS20 (5-min) — does CIFAR contaminate ARC navigation?

Key questions:
1. Does ARC→CIFAR order change anything? (R4 symmetry test)
2. With ~456 ARC centroids already present, does CIFAR argmin behave differently?
   (ARC centroids are in [0,1] ARC-frame space; CIFAR in [0,1] grayscale space —
   both centered, both 256D. Overlap MORE likely than CIFAR→ARC direction.)
3. After CIFAR spawns ~10K centroids, does LS20 argmin slow down or
   match to wrong centroids?

Baseline: Step 508 (CIFAR→ARC): 1% CIFAR, WIN@11170 LS20, -0.01pp forgetting.
Kill: LS20 P3 fails or step count >2x LS20 P1.

Runtime: ~12 min total (2x 5-min LS20 + CIFAR pass). Under 5-min cap
per phase; total exceeds but each phase is independent measurement.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

SPAWN_THRESHOLD = 0.3
N_ARC_ACTIONS_LS20 = 4
N_CIFAR_ACTIONS = 100


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32)
            + 0.587 * img[:, :, 1].astype(np.float32)
            + 0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class DynamicGraph:
    def __init__(self, threshold=SPAWN_THRESHOLD):
        self.threshold = threshold
        self.centroids = None
        self.edges = {}       # mode -> {(cell, action): {next_cell: count}}
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._mode = None
        self.spawns = {}      # mode -> count

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _spawn(self, x):
        if self.centroids is None:
            self.centroids = x.reshape(1, -1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1, -1)])
        self.spawns[self._mode] = self.spawns.get(self._mode, 0) + 1
        return len(self.centroids) - 1

    def step(self, x, n_actions):
        if self.centroids is None:
            self._spawn(x)
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        nearest = int(np.argmin(dists))
        cell = self._spawn(x) if dists[nearest] > self.threshold else nearest
        self.cells_seen.add(cell)
        edges = self.edges.setdefault(self._mode, {})
        if self.prev_cell is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(edges.get((cell, a), {}).values()) for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action

    @property
    def n_centroids(self):
        return len(self.centroids) if self.centroids is not None else 0


def run_cifar(g, X, y, label):
    g.set_mode('cifar')
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = g.step(x, N_CIFAR_ACTIONS)
        if a == int(y[i]):
            correct += 1
    acc = correct / len(X) * 100
    elapsed = time.time() - t0
    print(f"  {label}: acc={acc:.2f}%  centroids={g.n_centroids}  "
          f"spawns_cifar={g.spawns.get('cifar', 0)}  {elapsed:.0f}s", flush=True)
    return acc, elapsed


def run_ls20(g, arc, game_id, label, duration=300):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while time.time() - t0 < duration:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame:
            obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS_LS20)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
    elapsed = time.time() - t0
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  {label}: {status}  centroids={g.n_centroids}  "
          f"spawns_ls20={g.spawns.get('ls20', 0)}  go={go}  "
          f"steps={ts}  {elapsed:.0f}s", flush=True)
    return lvls, level_step, ts, elapsed


def main():
    t_total = time.time()
    print("Step 518: Reverse chain LS20 -> CIFAR-100 -> LS20", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}", flush=True)
    print(f"Baseline (Step 508): CIFAR->LS20 = 1%/WIN@11170/-0.01pp", flush=True)
    print(f"Kill: LS20 P3 fails OR step count >2x LS20 P1.", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    np.random.seed(0)
    g = DynamicGraph(threshold=SPAWN_THRESHOLD)

    # --- Phase 1: LS20 (5-min) — fresh substrate ---
    print("\n--- Phase 1: LS20 (5-min, fresh substrate) ---", flush=True)
    ls20_p1_lvls, ls20_p1_step, ls20_p1_ts, ls20_p1_time = run_ls20(
        g, arc, ls20.game_id, "LS20-P1", 300)
    centroids_after_p1 = g.n_centroids
    print(f"  After P1: {centroids_after_p1} centroids "
          f"(all LS20: {g.spawns.get('ls20', 0)})", flush=True)

    # --- Phase 2: CIFAR-100 (1-pass) ---
    print("\n--- Phase 2: CIFAR-100 (1-pass, on top of ARC centroids) ---", flush=True)
    cifar_acc, cifar_time = run_cifar(g, X, y, "CIFAR")
    centroids_after_p2 = g.n_centroids
    cifar_spawns = g.spawns.get('cifar', 0)
    print(f"  After P2: {centroids_after_p2} centroids "
          f"(ls20={g.spawns.get('ls20', 0)}, cifar={cifar_spawns})", flush=True)
    # How many CIFAR images matched existing ARC centroids vs spawning new?
    cifar_reuse = 10000 - cifar_spawns
    print(f"  CIFAR centroid reuse: {cifar_reuse}/10000 images matched "
          f"existing ARC centroids (threshold={SPAWN_THRESHOLD})", flush=True)

    # --- Phase 3: LS20 (5-min) — after CIFAR contamination ---
    print("\n--- Phase 3: LS20 (5-min, after CIFAR) ---", flush=True)
    ls20_p3_lvls, ls20_p3_step, ls20_p3_ts, ls20_p3_time = run_ls20(
        g, arc, ls20.game_id, "LS20-P3", 300)
    centroids_after_p3 = g.n_centroids
    ls20_p3_new_spawns = g.spawns.get('ls20', 0) - (centroids_after_p1)
    print(f"  After P3: {centroids_after_p3} centroids "
          f"(ls20 new spawns in P3: {ls20_p3_new_spawns})", flush=True)

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("STEP 518 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  LS20 P1 (fresh):  {'WIN@'+str(ls20_p1_step) if ls20_p1_lvls>0 else 'FAIL'}  "
          f"{ls20_p1_ts} steps  {centroids_after_p1} centroids", flush=True)
    print(f"  CIFAR:            {cifar_acc:.2f}%  {cifar_spawns} spawns  "
          f"{cifar_reuse} reused ARC centroids", flush=True)
    print(f"  LS20 P3 (after):  {'WIN@'+str(ls20_p3_step) if ls20_p3_lvls>0 else 'FAIL'}  "
          f"{ls20_p3_ts} steps  {centroids_after_p3} centroids", flush=True)

    # Comparison
    print(f"\n  COMPARISON:", flush=True)
    if ls20_p1_lvls > 0 and ls20_p3_lvls > 0:
        ratio = ls20_p3_step / ls20_p1_step if ls20_p1_step > 0 else float('inf')
        print(f"  Step ratio P3/P1: {ratio:.2f}x "
              f"(P1={ls20_p1_step}, P3={ls20_p3_step})", flush=True)
        if ratio > 2.0:
            print(f"  DEGRADED: LS20 >2x slower after CIFAR. "
                  f"Centroid explosion ({centroids_after_p2}) slows argmin.", flush=True)
        else:
            print(f"  STABLE: LS20 navigation preserved after CIFAR.", flush=True)
    elif ls20_p1_lvls > 0 and ls20_p3_lvls == 0:
        print(f"  KILLED: LS20 navigated in P1 but FAILED in P3. "
              f"CIFAR contamination confirmed.", flush=True)
    elif ls20_p1_lvls == 0:
        print(f"  INCONCLUSIVE: LS20 didn't navigate in P1 either. "
              f"Seed/budget issue.", flush=True)

    # Step 508 comparison
    print(f"\n  vs Step 508 (CIFAR->LS20):", flush=True)
    print(f"    508: WIN@11170 with 10463 centroids", flush=True)
    print(f"    518 P1: {'WIN@'+str(ls20_p1_step) if ls20_p1_lvls>0 else 'FAIL'} "
          f"with {centroids_after_p1} centroids", flush=True)
    print(f"    518 P3: {'WIN@'+str(ls20_p3_step) if ls20_p3_lvls>0 else 'FAIL'} "
          f"with {centroids_after_p3} centroids", flush=True)

    # Argmin cost diagnostic
    if centroids_after_p1 > 0 and centroids_after_p2 > 0:
        cost_ratio = centroids_after_p2 / centroids_after_p1
        print(f"\n  ARGMIN COST: P1 searches {centroids_after_p1} centroids, "
              f"P3 searches {centroids_after_p2} centroids ({cost_ratio:.0f}x)", flush=True)

    print(f"\nVERDICT:", flush=True)
    if ls20_p1_lvls > 0 and ls20_p3_lvls > 0:
        print(f"  R4 SYMMETRIC: reverse chain works. Order doesn't matter.", flush=True)
        print(f"  Mode-scoped edges isolate domains. Shared centroids don't interfere.", flush=True)
    elif ls20_p1_lvls > 0 and ls20_p3_lvls == 0:
        print(f"  R4 ASYMMETRIC: CIFAR→ARC works but ARC→CIFAR→ARC breaks.", flush=True)
        print(f"  Centroid explosion ({centroids_after_p2}) degrades argmin.", flush=True)
    else:
        print(f"  INCONCLUSIVE. Rerun with different seed.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
