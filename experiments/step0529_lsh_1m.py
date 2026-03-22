"""
Step 529 — Pure argmin LSH on LS20, 1M steps, 3 seeds.

Extends Step 528 budget. No stochastic tricks. Pure argmin navigation.
Question: does cell growth plateau or keep climbing? When does L2 appear?

arcagi3 wrapper provides gym-style interface over arc_agi 0.9.4.
TIME_CAP=300s per seed (5-min cap; 1M steps ~400s so cap limits to ~750K).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import time
import numpy as np

K = 12
N_ACTIONS = 4
MAX_STEPS = 1_000_000
TIME_CAP = 300
N_SEEDS = 3
CHECKPOINT = 100_000


def encode(frame):
    a = np.array(frame[0], dtype=np.float32) / 15.0
    x = a.reshape(16, 4, 16, 4).mean(axis=(1, 3)).flatten()
    return x - x.mean()


class LSH:
    def __init__(self, k=K, seed=0):
        self.H = np.random.RandomState(seed).randn(k, 256).astype(np.float32)

    def __call__(self, x):
        return int(np.packbits((self.H @ x > 0).astype(np.uint8),
                               bitorder='big').tobytes().hex(), 16)


class ArgminGraph:
    def __init__(self, lsh):
        self.lsh = lsh
        self.edges = {}
        self.prev_node = None
        self.prev_action = None
        self.cells = set()

    def observe(self, frame):
        node = self.lsh(encode(frame))
        self.cells.add(node)
        if self.prev_node is not None:
            d = self.edges.setdefault((self.prev_node, self.prev_action), {})
            d[node] = d.get(node, 0) + 1
        self.prev_node = node
        return node

    def act(self, a):
        self.prev_action = a

    def on_reset(self):
        self.prev_node = None

    def argmin(self, node):
        counts = [sum(self.edges.get((node, a), {}).values())
                  for a in range(N_ACTIONS)]
        return int(np.argmin(counts))


def t1():
    lsh = LSH(seed=0)
    g = ArgminGraph(lsh)
    x = np.random.RandomState(0).randn(256).astype(np.float32)
    x -= x.mean()
    n1 = lsh(encode([np.zeros((64, 64))]))
    n2 = lsh(encode([np.zeros((64, 64))]))
    assert n1 == n2, "LSH must be deterministic"
    # argmin on empty graph returns 0
    assert g.argmin(0) == 0
    print("T1 PASS")


def run_seed(seed, arc, game_id):
    from arcengine import GameState
    env = arc.make(game_id)
    action_space = env.action_space
    g = ArgminGraph(LSH(seed=seed * 1000))
    obs = env.reset()
    level = 0
    l1_step = l2_step = None
    deaths = 0
    t0 = time.time()
    step = 0
    cells_history = []

    while step < MAX_STEPS:
        if obs is None or not obs.frame:
            obs = env.reset(); g.on_reset(); deaths += 1; continue
        if obs.state == GameState.GAME_OVER:
            obs = env.reset(); g.on_reset(); deaths += 1; continue

        node = g.observe(obs.frame)
        action = g.argmin(node)
        g.act(action)
        obs = env.step(action_space[action])
        step += 1

        if obs is None:
            obs = env.reset(); g.on_reset(); continue

        cl = obs.levels_completed
        if cl > level:
            if cl == 1 and l1_step is None:
                l1_step = step
            if cl == 2 and l2_step is None:
                l2_step = step
                print(f"  seed {seed}: L2@{step}! cells={len(g.cells)}")
            level = cl

        if obs.state == GameState.WIN:
            break

        if step % CHECKPOINT == 0:
            elapsed = time.time() - t0
            cells_history.append((step, len(g.cells)))
            print(f"  seed {seed}: step={step:>7}  cells={len(g.cells):>4}  "
                  f"deaths={deaths}  {elapsed:.0f}s", flush=True)

        if time.time() - t0 > TIME_CAP:
            break

    elapsed = time.time() - t0
    tag = f"L2@{l2_step}" if l2_step else (f"L1@{l1_step}" if l1_step else "—")
    print(f"  seed {seed}: DONE {tag}  max_cells={len(g.cells)}  "
          f"steps={step}  deaths={deaths}  {elapsed:.0f}s", flush=True)
    return dict(seed=seed, l1=l1_step, l2=l2_step,
                max_cells=len(g.cells), steps=step, deaths=deaths,
                cells_history=cells_history)


def main():
    t1()

    import arcagi3
    import arc_agi
    arc = arc_agi.Arcade()
    games = arc.get_environments()
    ls20 = next(g for g in games if 'ls20' in g.game_id.lower())

    results = []
    for seed in range(N_SEEDS):
        print(f"\n--- seed {seed} (LSH seed={seed*1000}) ---", flush=True)
        results.append(run_seed(seed, arc, ls20.game_id))

    print(f"\n{'='*55}", flush=True)
    print("STEP 529 SUMMARY", flush=True)
    for r in results:
        tag = f"L2@{r['l2']}" if r['l2'] else (f"L1@{r['l1']}" if r['l1'] else "—")
        print(f"  seed {r['seed']}: {tag:>12}  max_cells={r['max_cells']:>4}  "
              f"steps={r['steps']:>7}  deaths={r['deaths']}", flush=True)
        for (s, c) in r['cells_history']:
            print(f"    @{s:>7}: {c} cells", flush=True)

    mc = max(r['max_cells'] for r in results)
    l2_count = sum(1 for r in results if r['l2'])

    print(f"\nL2: {l2_count}/{N_SEEDS}   max_cells: {mc}", flush=True)
    if l2_count > 0:
        print("BREAKTHROUGH: L2 achieved!", flush=True)
    elif mc > 434:
        print(f"GROWTH: {mc} > 434. Cell count still climbing at 1M scale.", flush=True)
    else:
        print(f"PLATEAU: {mc} <= 434. Cell growth has saturated.", flush=True)


if __name__ == "__main__":
    main()
