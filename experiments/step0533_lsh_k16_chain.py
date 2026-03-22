"""
Step 533 — LSH k=16 full chain: CIFAR->LS20->FT09->VC33->CIFAR.

Step 531 killed k=12: k=16 is 3/3 WIN at 200K steps (vs k=12's 2/3).
Step 523 was k=12 chain: LS20 0/3 FAIL (CIFAR contamination on actions 0-3),
FT09 3/3 WIN, VC33 3/3 WIN.

k=16 has 65K possible cells vs k=12's 4K — far less collision between
CIFAR images and game states. Should reduce action-0-3 bias on LS20.

FT09: 69 virtual actions (64 click grid + 5 simple) mapped to actual 6 API actions.
VC33: zone discovery (discover unique frame responses), mapped to actual click API.

Predictions:
- LS20: 3/3 (k=16 reduces CIFAR contamination overlap)
- FT09: 3/3 (round-robin over 69 actions, unchanged from k=12)
- VC33: 3/3 (zone discovery unaffected by k)
Kill: LS20 3/3 (CIFAR contamination resolved by finer partition).
"""
import time
import copy
import numpy as np

K = 16
N_CIFAR = 10_000
CIFAR_ACTIONS = 100
LS20_ACTIONS = 4
FT09_ACTIONS = 69
MAX_LS20 = 50_000
MAX_FT09 = 50_000
MAX_VC33 = 30_000
N_SEEDS = 3
TIME_CAP = 270  # per seed per game phase (safety valve)
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
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame:
            obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, LS20_ACTIONS)
        obs_before = obs.levels_completed
        obs = env.step(action_space[a])
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
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
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame:
            obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, FT09_ACTIONS)
        action, data = ft09_action(a, action_space)
        obs_before = obs.levels_completed
        obs = env.step(action, data=data)
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
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
        if time.time() - t0 > TIME_CAP:
            break
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame:
            obs = env.reset(); continue
        x = encode_arc(obs.frame)
        a = g.step(x, n_zones)
        cx, cy = zone_reps[a]
        obs_before = obs.levels_completed
        obs = env.step(action6, data={"x": cx, "y": cy})
        ts += 1
        if obs is None:
            break
        if obs.levels_completed > obs_before:
            lvls = obs.levels_completed
            if level_step is None:
                level_step = ts
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


def t1():
    g = LSHGraph(k=K, seed=0)
    assert g.H.shape == (K, 256), f"H shape: {g.H.shape}"
    x = np.ones(256, dtype=np.float32)
    x -= x.mean()
    cell = g._hash(x)
    assert isinstance(cell, int)
    # step returns valid action
    a = g.step(x, 4)
    assert 0 <= a < 4
    # CIFAR encode: grayscale, 256-dim, zero-mean
    img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    enc = encode_cifar(img)
    assert enc.shape == (256,)
    assert abs(enc.mean()) < 1e-5
    # FT09 action mapping
    dummy_space = list(range(6))
    env_act, data = ft09_action(0, dummy_space)
    assert env_act == 5  # ACTION6
    env_act2, data2 = ft09_action(64, dummy_space)
    assert env_act2 == 0  # ACTION1
    print(f"T1 PASS (K={K}, 65536 possible cells)")


def main():
    t1()

    t_total = time.time()
    print(f"Step 533: LSH k={K} full chain CIFAR->LS20->FT09->VC33->CIFAR", flush=True)
    print(f"actions: cifar={CIFAR_ACTIONS} ls20={LS20_ACTIONS} "
          f"ft09={FT09_ACTIONS} vc33=zones", flush=True)
    print(f"Baseline (k=12, Step 523): LS20 0/3, FT09 3/3, VC33 3/3", flush=True)

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

    # Phase 1: CIFAR pre-training
    print(f"\n--- Phase 1: CIFAR (1-pass, {N_CIFAR} images) ---", flush=True)
    acc1 = run_cifar(g, X, y, "P1")
    cifar_snapshot = copy.deepcopy(g.edges)
    cifar_cells = len(g.mode_cells.get('cifar', set()))
    print(f"  CIFAR snapshot: {len(cifar_snapshot)} edges  "
          f"{cifar_cells}/{2**K} cells", flush=True)

    # Phase 2: LS20
    print(f"\n--- Phase 2: LS20 ({MAX_LS20//1000}K steps, {N_SEEDS} seeds) ---",
          flush=True)
    ls20_wins = run_phase_multiseed('ls20', run_ls20, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=ls20.game_id)

    # Phase 3: FT09
    print(f"\n--- Phase 3: FT09 ({MAX_FT09//1000}K steps, {N_SEEDS} seeds, "
          f"{FT09_ACTIONS} virtual actions) ---", flush=True)
    ft09_wins = run_phase_multiseed('ft09', run_ft09, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=ft09.game_id)

    # Phase 4: VC33
    n_zones = len(zone_reps)
    print(f"\n--- Phase 4: VC33 ({MAX_VC33//1000}K steps, {N_SEEDS} seeds, "
          f"{n_zones} zones) ---", flush=True)
    vc33_wins = run_phase_multiseed('vc33', run_vc33, g, cifar_snapshot, seeds,
                                    arc=arc, game_id=vc33.game_id,
                                    zone_reps=zone_reps)

    # Phase 5: CIFAR return
    print("\n--- Phase 5: CIFAR return (1-pass) ---", flush=True)
    g.edges = copy.deepcopy(cifar_snapshot)
    g.mode_cells.pop('cifar', None)
    acc2 = run_cifar(g, X, y, "P5")

    cifar_cells_set = g.mode_cells.get('cifar', set())
    ls20_cells_set = g.mode_cells.get('ls20', set())
    ft09_cells_set = g.mode_cells.get('ft09', set())
    vc33_cells_set = g.mode_cells.get('vc33', set())

    print(f"\n{'='*60}", flush=True)
    print(f"STEP 533 SUMMARY (k={K})", flush=True)
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
    if ls20_wins > 0:
        print(f"  KILL: LS20 {ls20_wins}/{N_SEEDS} WIN. k=16 isolates CIFAR contamination.",
              flush=True)
    else:
        print(f"  LS20 still FAIL. CIFAR contamination persists at k={K}.", flush=True)

    if ls20_wins > 0 and ft09_wins > 0 and vc33_wins > 0:
        overlap_ls20 = len(cifar_cells_set & ls20_cells_set)
        print(f"  FULL CHAIN WIN. C+LS20 overlap={overlap_ls20} "
              f"(vs k=12 where CIFAR biased actions 0-3).", flush=True)
    elif ft09_wins > 0 or vc33_wins > 0:
        print(f"  PARTIAL: FT09={ft09_wins}/3, VC33={vc33_wins}/3 "
              f"(round-robin/zone exploitability unchanged).", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
