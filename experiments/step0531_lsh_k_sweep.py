"""
Step 531 — LSH k sweep on LS20, 200K steps, 3 seeds each.

k=8 (256 cells), k=12 (4096 cells), k=16 (65536 cells), k=20 (1M cells).
Uses arcagi3 wrapper. Records cells, L1 step, deaths.

Prediction: k=8 too coarse (0/3). k=12 baseline (2/3). k=16 too fine (0/3). k=20 degenerate.
Kill: if k=16 or k=20 outperforms k=12.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

N_ACTIONS = 4
MAX_STEPS = 200_000
N_SEEDS = 3
K_VALUES = [8, 12, 16, 20]
TIME_CAP = 290  # 290s to leave margin for all k values within 5 min total


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class LSHArgmin:
    def __init__(self, k, seed=0):
        self.k = k
        self.H = np.random.RandomState(seed).randn(k, 256).astype(np.float32)
        self.edges = {}
        self.prev_node = None
        self.prev_action = None
        self.cells = set()

    def _hash(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)

    def observe(self, frame):
        node = self._hash(encode(frame))
        self.cells.add(node)
        if self.prev_node is not None:
            d = self.edges.setdefault((self.prev_node, self.prev_action), {})
            d[node] = d.get(node, 0) + 1
        self.prev_node = node
        return node

    def act(self, node):
        counts = [sum(self.edges.get((node, a), {}).values()) for a in range(N_ACTIONS)]
        return int(np.argmin(counts))

    def on_reset(self):
        self.prev_node = None


def run_one(k, seed, arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    agent = LSHArgmin(k=k, seed=seed * 1000)
    obs = env.reset()
    l1_step = None
    deaths = ts = 0
    t0 = time.time()
    while ts < MAX_STEPS:
        if obs is None or not obs.frame:
            obs = env.reset(); agent.on_reset(); deaths += 1; continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); agent.on_reset(); deaths += 1; continue
        node = agent.observe(obs.frame)
        action = agent.act(node)
        agent.prev_action = action
        obs = env.step(action_space[action])
        ts += 1
        if obs and obs.levels_completed > 0 and l1_step is None:
            l1_step = ts
        if obs and obs.state == GameState.WIN:
            if l1_step is None: l1_step = ts
            break
        if time.time() - t0 > TIME_CAP:
            break
    tag = f"L1@{l1_step}" if l1_step else "FAIL"
    elapsed = time.time() - t0
    print(f"    seed={seed}: {tag}  cells={len(agent.cells):>5}  deaths={deaths}  {elapsed:.0f}s",
          flush=True)
    return l1_step is not None, len(agent.cells)


def t1():
    # k=8: 256 possible cells
    lsh8 = LSHArgmin(k=8, seed=0)
    x = np.random.RandomState(0).randn(256).astype(np.float32)
    x -= x.mean()
    dummy_frame = [np.zeros((64, 64))]
    n = lsh8._hash(encode(dummy_frame))
    assert isinstance(n, int)
    # k=20: different cell from k=8
    lsh20 = LSHArgmin(k=20, seed=0)
    n20 = lsh20._hash(encode(dummy_frame))
    assert isinstance(n20, int)
    # argmin on empty returns 0
    assert lsh8.act(0) == 0
    print("T1 PASS")


def main():
    t1()

    import arcagi3
    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    all_results = {}
    t_global = time.time()

    for k in K_VALUES:
        wins = 0
        max_cells = 0
        print(f"\n--- k={k} (2^{k}={2**k} possible cells) ---", flush=True)
        for seed in range(N_SEEDS):
            win, cells = run_one(k, seed, arc, ls20.game_id)
            if win: wins += 1
            max_cells = max(max_cells, cells)
        print(f"  k={k}: {wins}/{N_SEEDS} WIN  max_cells={max_cells}", flush=True)
        all_results[k] = (wins, max_cells)

    print(f"\n{'='*55}", flush=True)
    print("STEP 531 SUMMARY", flush=True)
    print(f"  {'k':>3}  {'wins':>6}  {'max_cells':>10}", flush=True)
    for k, (wins, mc) in all_results.items():
        print(f"  {k:>3}  {wins:>3}/{N_SEEDS}  {mc:>10}", flush=True)

    print(f"\nBaseline (k=12): {all_results[12][0]}/{N_SEEDS} wins", flush=True)
    best_k = max(all_results, key=lambda k: all_results[k][0])
    best_wins = all_results[best_k][0]
    print(f"Best: k={best_k} with {best_wins}/{N_SEEDS} wins", flush=True)

    if all_results[16][0] > all_results[12][0] or all_results[20][0] > all_results[12][0]:
        print("KILL: finer k outperforms k=12. Granularity helps.", flush=True)
    elif all_results[8][0] >= all_results[12][0]:
        print("SURPRISING: coarse k=8 matches k=12.", flush=True)
    else:
        print("CONFIRMED: k=12 dominates. k=8 too coarse, k=16/20 too fine.", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_global:.0f}s", flush=True)


if __name__ == "__main__":
    main()
