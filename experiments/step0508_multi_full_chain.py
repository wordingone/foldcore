#!/usr/bin/env python3
"""
Step 508 -- Full chain: CIFAR-100 -> LS20 -> FT09 -> VC33 -> CIFAR-100
Dynamic centroid growth (spawn-on-novelty, threshold=0.3).
Each domain spawns its own centroids. One persistent substrate state.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

SPAWN_THRESHOLD = 0.3
N_ARC_ACTIONS_LS20 = 4
N_FT09_ACTIONS = 69   # 64 click grid + 5 simple
N_CIFAR_ACTIONS = 100
VC33_GRID = [(gx*4+2, gy*4+2) for gy in range(16) for gx in range(16)]  # 256 positions


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3: img = img.transpose(1,2,0)
    gray = (0.299*img[:,:,0].astype(np.float32)+0.587*img[:,:,1].astype(np.float32)+0.114*img[:,:,2].astype(np.float32))/255.0
    arr = gray.reshape(16,2,16,2).mean(axis=(1,3)).flatten()
    return arr - arr.mean()

def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32)/15.0
    x = arr.reshape(16,4,16,4).mean(axis=(1,3)).flatten()
    return x - x.mean()


class DynamicGraph:
    def __init__(self, threshold=SPAWN_THRESHOLD):
        self.threshold = threshold
        self.centroids = None
        self.edges = {}   # mode -> {(cell,action): {next_cell: count}}
        self.prev_cell = None; self.prev_action = None
        self.cells_seen = set()
        self._mode = None
        self.spawns = {}  # mode -> count

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None; self.prev_action = None

    def _spawn(self, x):
        if self.centroids is None:
            self.centroids = x.reshape(1,-1).copy()
        else:
            self.centroids = np.vstack([self.centroids, x.reshape(1,-1)])
        self.spawns[self._mode] = self.spawns.get(self._mode,0)+1
        return len(self.centroids)-1

    def step(self, x, n_actions):
        if self.centroids is None:
            self._spawn(x)
            return int(np.random.randint(n_actions))
        diffs = self.centroids - x
        dists = np.sqrt(np.sum(diffs*diffs,axis=1))
        nearest = int(np.argmin(dists))
        cell = self._spawn(x) if dists[nearest] > self.threshold else nearest
        self.cells_seen.add(cell)
        edges = self.edges.setdefault(self._mode, {})
        if self.prev_cell is not None:
            d = edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell,0)+1
        counts = [sum(edges.get((cell,a),{}).values()) for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a,c in enumerate(counts) if c==min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell; self.prev_action = action
        return action

    @property
    def n_centroids(self): return len(self.centroids) if self.centroids is not None else 0


def ft09_action(action_id, action_space):
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        return action_space[5], {"x": gx*8+4, "y": gy*8+4}
    return action_space[action_id-64], {}


def discover_vc33_zones(arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action6 = env.action_space[0]
    hash_to_positions = {}
    for i, (cx, cy) in enumerate(VC33_GRID):
        obs = env.reset()
        if obs is None or obs.state == GameState.GAME_OVER: continue
        obs = env.step(action6, data={"x": cx, "y": cy})
        if obs is None or not obs.frame: continue
        h = np.array(obs.frame[0], dtype=np.uint8).tobytes().__hash__()
        hash_to_positions.setdefault(h, []).append(i)
    zones = list(hash_to_positions.values())
    zone_reps = [VC33_GRID[g[0]] for g in zones]
    print(f"  VC33 zones: {len(zones)} ({[len(z) for z in zones]} positions)", flush=True)
    return zone_reps


def run_cifar(g, X, y, label):
    g.set_mode('cifar')
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = g.step(x, N_CIFAR_ACTIONS)
        if a == int(y[i]): correct += 1
    acc = correct/len(X)*100
    print(f"  {label}: acc={acc:.2f}%  centroids={g.n_centroids}  spawns_cifar={g.spawns.get('cifar',0)}  {time.time()-t0:.0f}s", flush=True)
    return acc


def run_ls20(g, arc, game_id, duration=300):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id); action_space = env.action_space
    obs = env.reset(); ts = go = lvls = 0; level_step = None; t0 = time.time()
    while time.time()-t0 < duration:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go+=1; obs=env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs=env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_ARC_ACTIONS_LS20)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a]); ts+=1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls>0 else "FAIL"
    print(f"  LS20: {status}  centroids={g.n_centroids}  spawns_ls20={g.spawns.get('ls20',0)}  go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


def run_ft09(g, arc, game_id, duration=300):
    from arcengine import GameState
    g.set_mode('ft09')
    env = arc.make(game_id); action_space = env.action_space
    obs = env.reset(); ts = go = lvls = 0; level_step = None; t0 = time.time()
    while time.time()-t0 < duration:
        if obs is None: obs=env.reset(); continue
        if obs.state == GameState.GAME_OVER: go+=1; obs=env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs=env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, N_FT09_ACTIONS)
        action, data = ft09_action(a, action_space)
        obs_before = obs.levels_completed
        obs = env.step(action, data=data); ts+=1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls>0 else "FAIL"
    print(f"  FT09: {status}  centroids={g.n_centroids}  spawns_ft09={g.spawns.get('ft09',0)}  go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


def run_vc33(g, arc, game_id, zone_reps, duration=120):
    from arcengine import GameState
    g.set_mode('vc33')
    n_zones = len(zone_reps)
    env = arc.make(game_id); action6 = env.action_space[0]
    obs = env.reset(); ts = go = lvls = 0; level_step = None; t0 = time.time()
    while time.time()-t0 < duration:
        if obs is None: obs=env.reset(); continue
        if obs.state == GameState.GAME_OVER: go+=1; obs=env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs=env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, n_zones)
        cx, cy = zone_reps[a]
        obs_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy}); ts+=1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    status = f"WIN@{level_step}" if lvls>0 else "FAIL"
    print(f"  VC33: {status}  centroids={g.n_centroids}  spawns_vc33={g.spawns.get('vc33',0)}  go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls, level_step


def main():
    t_total = time.time()
    print("Step 508: Full chain CIFAR->LS20->FT09->VC33->CIFAR", flush=True)
    print(f"spawn_threshold={SPAWN_THRESHOLD}", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data); y = np.array(ds.targets)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    print("\nVC33 zone discovery...", flush=True)
    zone_reps = discover_vc33_zones(arc, vc33.game_id)

    np.random.seed(0)
    g = DynamicGraph(threshold=SPAWN_THRESHOLD)

    print("\n--- Phase 1: CIFAR-100 (1-pass) ---", flush=True)
    acc1 = run_cifar(g, X, y, "P1")

    print("\n--- Phase 2: LS20 (5-min) ---", flush=True)
    ls20_lvls, ls20_step = run_ls20(g, arc, ls20.game_id, 300)

    print("\n--- Phase 3: FT09 (5-min) ---", flush=True)
    ft09_lvls, ft09_step = run_ft09(g, arc, ft09.game_id, 300)

    print("\n--- Phase 4: VC33 (2-min) ---", flush=True)
    vc33_lvls, vc33_step = run_vc33(g, arc, vc33.game_id, zone_reps, 120)

    print("\n--- Phase 5: CIFAR-100 (1-pass) ---", flush=True)
    acc5 = run_cifar(g, X, y, "P5")

    print(f"\n{'='*60}", flush=True)
    print("STEP 508 SUMMARY", flush=True)
    print(f"  CIFAR P1:  {acc1:.2f}%", flush=True)
    print(f"  LS20:      {'WIN@'+str(ls20_step) if ls20_lvls>0 else 'FAIL'}", flush=True)
    print(f"  FT09:      {'WIN@'+str(ft09_step) if ft09_lvls>0 else 'FAIL'}", flush=True)
    print(f"  VC33:      {'WIN@'+str(vc33_step) if vc33_lvls>0 else 'FAIL'}", flush=True)
    print(f"  CIFAR P5:  {acc5:.2f}%  (delta={acc5-acc1:+.2f}pp)", flush=True)
    print(f"  Total centroids: {g.n_centroids}  spawns: {g.spawns}", flush=True)
    wins = sum([ls20_lvls>0, ft09_lvls>0, vc33_lvls>0])
    print(f"\nVERDICT: {wins}/3 games navigated in chain.", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
