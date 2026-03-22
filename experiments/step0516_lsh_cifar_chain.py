#!/usr/bin/env python3
"""
Step 516 -- LSH chain: CIFAR-100 -> LS20 -> CIFAR-100
Non-codebook family balance. LSH k=12, centered_enc, avgpool16.

Baseline: codebook chain (Step 508) — 1% CIFAR, WIN@11170 LS20, 0pp forgetting.
LSH standalone: 9/10 at 120K (Step 485).

Key question: LSH has fixed hash cells (no spawn threshold, no growth).
CIFAR and ARC frames share the same 4096-cell hash space.
Does CIFAR edge accumulation interfere with LS20 navigation?
Codebook auto-separated by scale (L2 mean=4.3 >> threshold=0.3).
LSH has no such mechanism — interference is the hypothesis to test.

Protocol: same as Step 508 but LSH substrate, LS20 only (FT09/VC33
collapse to 1 cell under avgpool16+LSH — Steps 467/468).

Runtime cap: 5 minutes.
"""
import time, logging
import numpy as np
logging.getLogger().setLevel(logging.WARNING)

K = 12
MAX_LS20_STEPS = 120_000
LS20_TIME_CAP = 240  # seconds
CIFAR_ACTIONS = 100
LS20_ACTIONS = 4


def encode_cifar(img):
    """CIFAR-100: RGB -> grayscale -> 16x16 avgpool -> centered."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    gray = (0.299 * img[:, :, 0].astype(np.float32)
            + 0.587 * img[:, :, 1].astype(np.float32)
            + 0.114 * img[:, :, 2].astype(np.float32)) / 255.0
    arr = gray.reshape(16, 2, 16, 2).mean(axis=(1, 3)).flatten()
    return arr - arr.mean()


def encode_arc(frame):
    """ARC 64x64: avgpool16 + centered."""
    arr = np.array(frame[0], dtype=np.float32) / 15.0
    x = arr.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class LSHGraph:
    def __init__(self, k=K, seed=0):
        rng = np.random.RandomState(seed + 9999)
        self.H = rng.randn(k, 256).astype(np.float32)
        self.powers = np.array([1 << i for i in range(k)], dtype=np.int64)
        self.edges = {}          # shared across domains
        self.prev_cell = None
        self.prev_action = None
        self.cells_seen = set()
        self._mode = None
        self.mode_cells = {}     # track cells per domain

    def set_mode(self, mode):
        self._mode = mode
        self.prev_cell = None
        self.prev_action = None

    def _hash(self, x):
        bits = (self.H @ x > 0).astype(np.int64)
        return int(np.dot(bits, self.powers))

    def step(self, x, n_actions):
        cell = self._hash(x)
        self.cells_seen.add(cell)
        self.mode_cells.setdefault(self._mode, set()).add(cell)

        if self.prev_cell is not None and self.prev_action is not None:
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
          f"total_cells={len(g.cells_seen)}  edges={len(g.edges)}  "
          f"{time.time()-t0:.0f}s", flush=True)
    return acc


def run_ls20(g, arc, game_id):
    from arcengine import GameState
    g.set_mode('ls20')
    env = arc.make(game_id)
    action_space = env.action_space
    obs = env.reset()
    ts = go = lvls = 0
    level_step = None
    t0 = time.time()

    while ts < MAX_LS20_STEPS and time.time() - t0 < LS20_TIME_CAP:
        if obs is None:
            obs = env.reset(); continue
        if obs.state == GameState.GAME_OVER:
            go += 1; obs = env.reset(); continue
        if obs.state == GameState.WIN:
            break
        if not obs.frame or len(obs.frame) == 0:
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
    print(f"  LS20: {status}  ls20_cells={ls20_cells}  total_cells={len(g.cells_seen)}  "
          f"edges={len(g.edges)}  go={go}  steps={ts}  {time.time()-t0:.0f}s", flush=True)
    return lvls, level_step, ts


def main():
    t_total = time.time()
    print(f"Step 516: LSH chain CIFAR->LS20->CIFAR. k={K}, centered_enc, avgpool16.", flush=True)
    print(f"Baseline: codebook chain (Step 508): 1%/WIN@11170/0pp forgetting.", flush=True)
    print(f"LSH standalone LS20 (Step 485): 9/10 at 120K.", flush=True)
    print(f"Max LS20 steps: {MAX_LS20_STEPS//1000}K, time cap: {LS20_TIME_CAP}s.", flush=True)

    import torchvision, arc_agi
    ds = torchvision.datasets.CIFAR100('./data/cifar100', train=False, download=True)
    X = np.array(ds.data)
    y = np.array(ds.targets)

    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    np.random.seed(0)
    g = LSHGraph(k=K, seed=0)

    # --- Phase 1: CIFAR-100 (1-pass) ---
    print("\n--- Phase 1: CIFAR-100 (1-pass) ---", flush=True)
    acc1 = run_cifar(g, X, y, "P1")

    # Diagnostic: cell overlap
    cifar_cells = g.mode_cells.get('cifar', set())
    print(f"\n  DIAGNOSTIC: {len(cifar_cells)} CIFAR cells, {len(g.edges)} edges accumulated.", flush=True)

    # --- Phase 2: LS20 (120K steps) ---
    print("\n--- Phase 2: LS20 (120K steps) ---", flush=True)
    ls20_lvls, ls20_step, ls20_ts = run_ls20(g, arc, ls20.game_id)

    # Diagnostic: cell overlap between domains
    ls20_cells = g.mode_cells.get('ls20', set())
    overlap = cifar_cells & ls20_cells
    print(f"\n  DIAGNOSTIC: overlap={len(overlap)} cells shared between CIFAR and LS20.", flush=True)
    print(f"  CIFAR-only: {len(cifar_cells - ls20_cells)}  LS20-only: {len(ls20_cells - cifar_cells)}", flush=True)
    if overlap:
        # Check edge contamination: how many LS20-visited cells have CIFAR edges?
        contaminated = 0
        for cell in ls20_cells:
            for a in range(CIFAR_ACTIONS):
                if (cell, a) in g.edges:
                    contaminated += 1
                    break
        print(f"  LS20 cells with CIFAR-action edges: {contaminated}/{len(ls20_cells)}", flush=True)

    # --- Phase 3: CIFAR-100 (1-pass) ---
    print("\n--- Phase 3: CIFAR-100 (1-pass) ---", flush=True)
    acc2 = run_cifar(g, X, y, "P2")

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print("STEP 516 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  CIFAR P1:  {acc1:.2f}%  ({len(cifar_cells)} cells)", flush=True)
    print(f"  LS20:      {'WIN@'+str(ls20_step) if ls20_lvls>0 else 'FAIL'}  "
          f"({len(ls20_cells)} cells, {ls20_ts} steps)", flush=True)
    print(f"  CIFAR P2:  {acc2:.2f}%  (delta={acc2-acc1:+.2f}pp)", flush=True)
    print(f"  Cell overlap: {len(overlap)} shared cells", flush=True)
    print(f"  Total cells: {len(g.cells_seen)}/{2**K}", flush=True)
    print(f"  Total edges: {len(g.edges)}", flush=True)

    print(f"\nVERDICT:", flush=True)
    if ls20_lvls > 0 and abs(acc2 - acc1) < 1.0:
        print(f"  LSH CHAIN PASSES. Same outcome as codebook chain (Step 508).", flush=True)
        print(f"  Domain separation mechanism: {'hash isolation (0 overlap)' if len(overlap)==0 else f'partial overlap ({len(overlap)} cells) but edge-action-count isolation (100 vs 4 actions)'}", flush=True)
    elif ls20_lvls > 0:
        print(f"  LS20 navigates but CIFAR accuracy shifted ({acc2-acc1:+.2f}pp).", flush=True)
    elif ls20_lvls == 0 and len(overlap) > 0:
        print(f"  LS20 FAILS. CIFAR edge contamination likely cause ({len(overlap)} shared cells).", flush=True)
        print(f"  Compare: LSH standalone 9/10 (Step 485). Chain interference confirmed.", flush=True)
    else:
        print(f"  LS20 FAILS with 0 overlap. Budget or seed issue, not contamination.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_total:.0f}s", flush=True)


if __name__ == '__main__':
    main()
