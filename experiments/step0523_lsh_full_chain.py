#!/usr/bin/env python3
"""
Step 523 -- LSH graph full chain: CIFAR->LS20->FT09->VC33->CIFAR.
Non-codebook family balance. LSH k=12, centered_enc, avgpool16.

Step 516: LSH CIFAR->LS20->CIFAR. WIN@1116 (action-scope isolation mechanism).
This extends to FT09 (n_actions=69) and VC33 (n_actions=3).

Predictions:
- LS20: WIN (baseline from Step 516)
- FT09: 3/3 (round-robin over 69 actions, degenerate like Step 522 k-means)
- VC33: 0/3 (magic pixel timing issue)
- CIFAR return: 0pp forgetting

Key question: does action-scope isolation extend to expanded action spaces?
"""
import time, copy, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

K = 12
N_CIFAR = 10_000
CIFAR_ACTIONS = 100
LS20_ACTIONS = 4
FT09_ACTIONS = 69
MAX_LS20 = 50_000
MAX_FT09 = 50_000
MAX_VC33 = 30_000
N_SEEDS = 3
VC33_GRID = [(gx * 4 + 2, gy * 4 + 2) for gy in range(16) for gx in range(16)]


def encode_cifar(img):
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32) +
            0.587 * img[:, :, 1].astype(np.float32) +
            0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class LSHGraph:
    def __init__(self, k=K, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.edges = {}
        self.prev_cell = None
        self.prev_action = None
        self.mode_cells = {}
        self._mode = None

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, x, n_actions):
        cell = self._hash(x)
        self.mode_cells.setdefault(self._mode, set()).add(cell)
        if self.prev_cell is not None:
            d = self.edges.setdefault((self.prev_cell, self.prev_action), {})
            d[cell] = d.get(cell, 0) + 1
        counts = [sum(self.edges.get((cell, a), {}).values())
                  for a in range(n_actions)]
        min_c = min(counts)
        cands = [a for a, c in enumerate(counts) if c == min_c]
        action = cands[int(np.random.randint(len(cands)))]
        self.prev_cell = cell
        self.prev_action = action
        return action


def run_cifar(g, X, y, label):
    g.set_mode('cifar')
    correct = 0
    t0 = time.time()
    for i in range(len(X)):
        x = encode_cifar(X[i])
        a = g.step(x, CIFAR_ACTIONS)
        if a == int(y[i]):
            correct += 1
    acc = correct / len(X) * 100
    cifar_cells = len(g.mode_cells.get('cifar', set()))
    print(f"  {label}: acc={acc:.2f}%  cifar_cells={cifar_cells}/{2**K}  "
          f"edges={len(g.edges)}  {time.time()-t0:.0f}s", flush=True)
    return acc


def run_ls20(g, arc, game_id, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_LS20:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, LS20_ACTIONS)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    ls20_cells = len(g.mode_cells.get('ls20', set()))
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}: {status}  ls20_cells={ls20_cells}/{2**K}  "
          f"go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls > 0


def ft09_action(action_id, action_space):
    if action_id < 64:
        gy, gx = divmod(action_id, 8)
        return action_space[5], {"x": gx * 8 + 4, "y": gy * 8 + 4}
    return action_space[action_id - 64], {}


def run_ft09(g, arc, game_id, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('ft09')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_FT09:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, FT09_ACTIONS)
        action, data = ft09_action(a, action_space)
        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    ft09_cells = len(g.mode_cells.get('ft09', set()))
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}: {status}  ft09_cells={ft09_cells}/{2**K}  "
          f"go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls > 0


def discover_vc33_zones(arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action6 = env.action_space[0]
    hash_to_positions = {}
    for i, (cx, cy) in enumerate(VC33_GRID):
        obs = env.reset()
        if obs is None or obs.state == GameState.GAME_OVER:
            continue
        obs = env.step(action6, data={"x": cx, "y": cy})
        if obs is None or not obs.frame:
            continue
        h = np.array(obs.frame[0], dtype=np.uint8).tobytes().__hash__()
        hash_to_positions.setdefault(h, []).append(i)
    zones = list(hash_to_positions.values())
    zone_reps = [VC33_GRID[z[0]] for z in zones]
    print(f"  VC33 zones: {len(zones)} ({[len(z) for z in zones]})", flush=True)
    return zone_reps


def run_vc33(g, arc, game_id, zone_reps, seed=0):
    from arcengine import GameState
    np.random.seed(seed)
    g.set_mode('vc33')
    n_zones = len(zone_reps)
    env = arc.make(game_id)
    action6 = env.action_space[0]
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()
    while ts < MAX_VC33:
        if obs is None: obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER: go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN: break
        if not obs.frame: obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, n_zones)
        cx, cy = zone_reps[a]
        obs_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1
        if obs is None: break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None: level_step = ts
    vc33_cells = len(g.mode_cells.get('vc33', set()))
    status = f"WIN@{level_step}" if lvls > 0 else "FAIL"
    print(f"  seed={seed}: {status}  vc33_cells={vc33_cells}/{2**K}  "
          f"go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls > 0


def run_phase_multiseed(name, run_fn, g, edges_snapshot, seeds, **kwargs):
    wins = 0
    for s in seeds:
        g.edges = copy.deepcopy(edges_snapshot)
        result = run_fn(g, seed=s, **kwargs)
        if result:
            wins += 1
    print(f"  {name}: {wins}/{len(seeds)} WIN", flush=True)
    return wins


def main():
    t_total = time.time()
    print("Step 523: LSH graph full chain CIFAR->LS20->FT09->VC33->CIFAR", flush=True)
    print(f"k={K}  cifar={N_CIFAR}  seeds={N_SEEDS}  actions: "
          f"cifar={CIFAR_ACTIONS} ls20={LS20_ACTIONS} ft09={FT09_ACTIONS}", flush=True)
    print(f"Baseline: Step 516 LSH CIFAR->LS20->CIFAR WIN@1116", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100',
                                        train=False, download=True)
    X = np.array(ds.data[:N_CIFAR])
    y = np.array(ds.targets[:N_CIFAR])
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())
    ft09 = next(g for g in games if 'ft09' in g.game_id.lower())
    vc33 = next(g for g in games if 'vc33' in g.game_id.lower())

    print("\nVC33 zone discovery...", flush=True)
    zone_reps = discover_vc33_zones(arc, vc33.game_id)

    np.random.seed(0)
    g = LSHGraph(k=K, seed=0)
    seeds = list(range(N_SEEDS))

    # Phase 1: CIFAR (build contamination)
    print("\n--- Phase 1: CIFAR (1-pass, 10K images) ---", flush=True)
    acc1 = run_cifar(g, X, y, "P1")
    cifar_snapshot = copy.deepcopy(g.edges)
    cifar_cells = len(g.mode_cells.get('cifar', set()))
    print(f"  CIFAR snapshot: {len(cifar_snapshot)} edges  {cifar_cells}/{2**K} cells", flush=True)

    # Phase 2: LS20
    print(f"\n--- Phase 2: LS20 ({MAX_LS20//1000}K steps, {N_SEEDS} seeds) ---", flush=True)
    ls20_wins = run_phase_multiseed('ls20', run_ls20, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=ls20.game_id)

    # Phase 3: FT09
    print(f"\n--- Phase 3: FT09 ({MAX_FT09//1000}K steps, {N_SEEDS} seeds, {FT09_ACTIONS} actions) ---",
          flush=True)
    ft09_wins = run_phase_multiseed('ft09', run_ft09, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=ft09.game_id)

    # Phase 4: VC33
    n_zones = len(zone_reps)
    print(f"\n--- Phase 4: VC33 ({MAX_VC33//1000}K steps, {N_SEEDS} seeds, {n_zones} zones) ---",
          flush=True)
    vc33_wins = run_phase_multiseed('vc33', run_vc33, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=vc33.game_id, zone_reps=zone_reps)

    # Phase 5: CIFAR return (forgetting check)
    print("\n--- Phase 5: CIFAR return (1-pass) ---", flush=True)
    g.edges = copy.deepcopy(cifar_snapshot)
    g.mode_cells.pop('cifar', None)
    acc2 = run_cifar(g, X, y, "P5")

    # Diagnostics
    cifar_cells_set = g.mode_cells.get('cifar', set())
    ls20_cells_set = g.mode_cells.get('ls20', set())
    ft09_cells_set = g.mode_cells.get('ft09', set())
    vc33_cells_set = g.mode_cells.get('vc33', set())

    print(f"\n{'='*60}", flush=True)
    print("STEP 523 SUMMARY", flush=True)
    print(f"  CIFAR P1:   {acc1:.2f}%  ({cifar_cells}/{2**K} cells)", flush=True)
    print(f"  LS20:       {ls20_wins}/{N_SEEDS} WIN  "
          f"cells={len(ls20_cells_set)}/{2**K}", flush=True)
    print(f"  FT09:       {ft09_wins}/{N_SEEDS} WIN  "
          f"cells={len(ft09_cells_set)}/{2**K}", flush=True)
    print(f"  VC33:       {vc33_wins}/{N_SEEDS} WIN  "
          f"cells={len(vc33_cells_set)}/{2**K}", flush=True)
    print(f"  CIFAR P5:   {acc2:.2f}%  (delta={acc2-acc1:+.2f}pp)", flush=True)
    print(f"  Cell overlap: "
          f"C+LS20={len(cifar_cells_set & ls20_cells_set)}  "
          f"C+FT09={len(cifar_cells_set & ft09_cells_set)}  "
          f"C+VC33={len(cifar_cells_set & vc33_cells_set)}", flush=True)

    print(f"\nVERDICT:", flush=True)
    print(f"  Action-scope isolation extends to:", flush=True)
    print(f"    LS20 (n=4):  {'YES' if ls20_wins > 0 else 'NO'}  ({ls20_wins}/{N_SEEDS})",
          flush=True)
    print(f"    FT09 (n=69): {'YES' if ft09_wins > 0 else 'NO'}  ({ft09_wins}/{N_SEEDS})",
          flush=True)
    print(f"    VC33 (n=3):  {'YES' if vc33_wins > 0 else 'NO'}  ({vc33_wins}/{N_SEEDS})",
          flush=True)
    if ls20_wins > 0 and ft09_wins > 0:
        print(f"  FULL CHAIN PASSES. LSH chain works across all 3 games.", flush=True)
        print(f"  Action-scope isolation is robust to expanded action spaces.", flush=True)
    elif ls20_wins > 0 and ft09_wins == 0:
        print(f"  PARTIAL: LS20 passes, FT09 fails. Action space expansion breaks chain.",
              flush=True)
    else:
        print(f"  FAIL: LS20 fails. CIFAR contamination breaks LSH navigation.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
